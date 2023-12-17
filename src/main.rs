use std::io::Write;
use std::net::SocketAddr;

use axum::{routing::post, Router};
use candle_core::utils::{get_num_threads, has_accelerate, has_mkl};
use clap::Parser;
use clap_verbosity_flag::Verbosity;
use inquire::{Select, Text};
use oxpilot::cli::{CLICommands, CLI};
use oxpilot::cmd::Command::Prompt;
use oxpilot::llm::LLMBuilder;
use oxpilot::process::process;
use oxpilot::utils::commit::commit_then_exit;
use oxpilot::utils::diff::get_diff;
use oxpilot::utils::mistral;
use oxpilot::utils::spinner::SilentableSpinner;
use regex::Regex;
use routes::completion::completion;
use tokio::sync::mpsc;
use tracing::{debug, error, info, warn};
use tracing_log::{log, AsTrace};
use tracing_subscriber::fmt::format::FmtSpan;

pub mod llm;
pub mod process;
pub mod routes;
pub mod state;
pub mod token;
pub mod utils;

// The `#[tokio::main]` function is a macro. It transforms the async fn main()
// into a synchronous fn main() that initializes a runtime instance and executes the async main function.
// ```no_run
// #[tokio::main]
// async fn main() {
//     println!("hello");
// }
// ```
// is transformed to:
// ```no_run
// fn main() {
//     let mut rt = tokio::runtime::Runtime::new().unwrap();
//     rt.block_on(async {
//         println!("hello");
//     })
// }
// ```
#[tokio::main]
async fn main() {
    // Parse command line arguments
    let cli = CLI::parse();

    let verbosity: &Verbosity = &cli.verbose;
    let is_silent: bool = verbosity.is_silent();
    let log_level = verbosity.log_level();
    let is_debug = match log_level {
        Some(log_level) => log_level == log::Level::Debug,
        None => false,
    };

    // The tracing crate is a framework for instrumenting Rust programs to
    // collect structured, event-based diagnostic information.
    // https://github.com/tokio-rs/tracing
    // https://tokio.rs/tokio/topics/tracing
    // Start configuring a `fmt` subscriber
    let subscriber = tracing_subscriber::fmt()
        // Set log level by `-v` or `-vv` or `-vvv` or `-vvvv` or `-vvvvv
        .with_max_level(cli.verbose.log_level_filter().as_trace());
    let format = tracing_subscriber::fmt::format()
        // Use a more compact, abbreviated log format
        .compact()
        // Display source code file paths
        .with_file(is_debug)
        // Display source code line numbers
        .with_line_number(is_debug)
        // Display the thread ID an event was recorded on
        .with_thread_ids(is_debug)
        // Display the event's target (module path)
        .with_target(is_debug);

    // Set the subscriber as the default
    match tracing::subscriber::set_global_default(
        subscriber
            .event_format(format)
            .with_span_events(FmtSpan::ENTER | FmtSpan::CLOSE)
            .finish(),
    ) {
        Ok(_) => (),
        Err(error) => panic!(
            "error setting default tracer as `fmt` subscriber {:?}",
            error
        ),
    };
    if has_accelerate() {
        info!("oxpilot's candle was compiled with 'accelerate' support")
    }
    if has_mkl() {
        info!("oxpilot's candle was compiled with 'mkl' support")
    }
    info!("number of thread: {:?} used by candle", get_num_threads());

    debug!("tokenizer_repo_id: {:?}", &cli.tokenizer_repo_id);
    debug!("model_repo_id: {:?}", &cli.model_repo_id);
    debug!("model_file_name: {:?}", &cli.model_file_name);
    let llm_builder = LLMBuilder::new()
        .tokenizer_repo_id(cli.tokenizer_repo_id)
        .model_repo_id(cli.model_repo_id)
        .model_file_name(cli.model_file_name);
    let mut llm = llm_builder
        .build(is_silent)
        .await
        .expect("Failed to build LLM");

    let (tx, mut rx) = mpsc::channel(32);
    let _ = tokio::spawn(async move {
        let seed = cli.seed;
        let top_p = cli.top_p;
        let to_sample = cli.to_sample;
        let repeat_last_n = cli.repeat_last_n;
        let repeat_penalty = cli.repeat_penalty;
        let eos_token = "</s>";
        while let Some(cmd) = rx.recv().await {
            match cmd {
                // handle Command::Prompt from `tx.send().await`;
                Prompt {
                    prompt,
                    responder,
                    temperature,
                    max_sampled,
                } => {
                    debug!("prompt:{}", prompt);
                    process(
                        prompt,
                        &mut llm,
                        responder,
                        to_sample,
                        seed,
                        temperature,
                        top_p,
                        repeat_last_n,
                        repeat_penalty,
                        eos_token.to_string(),
                        max_sampled,
                    )
                    .await;
                }
            }
        }
    });

    match &cli.command {
        Some(CLICommands::Serve { port }) => {
            info!("starting copilot server on port: {}", &port);
            let state = state::AppState { tx };
            let address = SocketAddr::from(([0, 0, 0, 0], port.to_owned()));
            let listener = tokio::net::TcpListener::bind(&address).await.unwrap();
            let app = app(state);

            match axum::serve(listener, app).await {
                Ok(_) => info!("copilot server exited."),
                Err(error) => {
                    info!("server exited with error: {}", error);
                    info!("terminating LLM manager");
                }
            }
        }
        Some(CLICommands::Commit {
            dry_run,
            function_context,
            all_yes,
            signoff,
        }) => {
            let mut spinner = SilentableSpinner::new(
                is_silent,
                Some("generating commit message for staged files..."),
            );
            spinner.update("getting git diff of staged files...");
            let diff = get_diff(*function_context).await;
            if diff.len() == 0 {
                spinner.fail("no diff found, have you staged any?");
                std::process::exit(1);
            }
            let mut tip = "--function-context adds context to LLM";
            if diff.len() > 800 {
                tip = "large diff takes longer, commit often 😊"
            }
            spinner.update(format!("generating commit message... (tip: {})", tip));
            let prompt = mistral::instruct(format!("Summarize the git diff in one sentence no more then 15 words. The summary starts with 'fix: ' if the git diff fixes bugs. Starts with 'feat: ' if introducing a new feature. 'chore: ' for reformatting code or adding stuff around the build tools. 'docs: ' for documentations. The summary should be concise but comprehensive covering what has changed and explaining why.\n{}\nDo NOT start with 'This git diff' or 'committed:'.", diff));

            let (responder, mut receiver) = mpsc::channel(8);
            tx.send(Prompt {
                prompt: prompt.clone(),
                responder,
                temperature: 0.8,
                max_sampled: 256,
            })
            .await
            .expect("failed to send prompt to LLM manager");

            let mut commit_message = String::new();
            while let Some(text) = receiver.recv().await {
                commit_message.push_str(&text);
                if commit_message.len() < 90 {
                    spinner.update(commit_message.trim());
                }
            }
            commit_message = commit_message.trim().to_string();
            let regex = Regex::new(r"^(build|chore|ci|docs|feat|fix|perf|refactor|revert|style|test){1}(\([\w\-\.]+\))?(!)?: ([\w ])+([\s\S]*)").unwrap();

            if !regex.is_match(&commit_message) {
                spinner.update(
                    "retry because the message not match the conventional commits specification...",
                );
                let (responder, mut receiver) = mpsc::channel(8);
                tx.send(Prompt {
                    prompt: prompt.clone(),
                    responder,
                    temperature: 1.2,
                    max_sampled: 256,
                })
                .await
                .expect("failed to send prompt to LLM manager");
                commit_message = String::new();
                while let Some(text) = receiver.recv().await {
                    commit_message.push_str(&text);
                    if commit_message.len() < 90 {
                        spinner.update(commit_message.trim());
                    }
                }
            }
            spinner.success(&format!("generated:'{}'", commit_message));
            if !*dry_run {
                if *all_yes {
                    commit_then_exit(&commit_message, *signoff)
                }
                let options: Vec<&str> = vec!["Commit", "Edit"];

                match Select::new(
                    "Commit or edit the message? (tip: use --yes to skip this prompt)",
                    options,
                )
                .prompt()
                {
                    Ok(choice) => {
                        if choice == "Commit" {
                            commit_then_exit(&commit_message, *signoff);
                        }
                        if choice == "Edit" {
                            let edited_message =
                                Text::new("✏️").with_initial_value(&commit_message).prompt();
                            match edited_message {
                                Ok(edited_message) => {
                                    commit_then_exit(&edited_message, *signoff);
                                }
                                // user pressed ctrl-c, just exit the program
                                Err(_) => std::process::exit(0),
                            }
                        }
                    }
                    Err(_) => std::process::exit(1),
                }
            }
        }
        Some(CLICommands::Any(args)) => {
            let arg0 = args[0].clone().into_string();
            match arg0 {
                Ok(arg0) => {
                    let mut input = arg0.clone();
                    for arg in args.iter().skip(1) {
                        let next_arg_string = arg.clone().into_string();
                        match next_arg_string {
                            Ok(next_arg_string) => {
                                input.push_str(&format!(" {}", next_arg_string));
                            }
                            Err(cause) => {
                                warn! {
                                    cause = format!("{:#?}", cause),
                                    "failed to parse input arguments"
                                };
                            }
                        }
                    }
                    let prompt = mistral::instruct(input);
                    let (responder, mut receiver) = mpsc::channel(8);
                    tx.send(Prompt {
                        prompt,
                        responder,
                        temperature: 1.0,
                        max_sampled: 4096,
                    })
                    .await
                    .expect("failed to send prompt to LLM manager");
                    let mut last = String::new();
                    while let Some(text) = receiver.recv().await {
                        print!("{text}");
                        last = text;
                        std::io::stdout().flush().expect("failed to flush stdout");
                    }
                    // print a newline if the last text does not end with a newline
                    // prevent https://unix.stackexchange.com/questions/167582/why-zsh-ends-a-line-with-a-highlighted-percent-symbol
                    if last != "\n" {
                        print!("\n");
                        std::io::stdout().flush().expect("failed to flush stdout");
                    }
                }
                Err(cause) => {
                    warn! {
                        cause = format!("{:#?}", cause),
                        "failed to parse input arguments"
                    };
                }
            }
        }
        None => {
            error!("no operation specified, try `ox serve` or `ox --help` for more options");
        }
    }
}

fn app(state: state::AppState) -> Router {
    Router::new()
        .route("/v1/engines/:engine/completions", post(completion))
        .route("/v1/completions", post(completion))
        .with_state(state)
}

/// The #[cfg(test)] annotation on the tests module tells Rust to compile and run the test
/// code only when you run cargo test, not when you run cargo build. This saves compile time when you only
/// want to build the library and saves space in the resulting compiled artifact because the tests are not included.
#[cfg(test)]
mod tests {
    // imports are only for the tests
    use eventsource_stream::Eventsource; // needed for `.eventsource()`
    use futures::prelude::*; // needed for `.next().await`
    use oxpilot::types::Completion;
    use serde_json::Value::Null;
    use std::time::{SystemTime, UNIX_EPOCH};
    use tokio::net::TcpListener;

    /// `super::*` means "everything in the parent module"
    /// It will bring all of the test module’s parent’s items into scope.
    use super::*;
    /// A helper function that spawns our application in the background
    /// and returns its address (e.g. http://127.0.0.1:[random_port])
    async fn spawn_app(host: impl Into<String>) -> String {
        let _host = host.into();
        // Bind to localhost at the port 0, which will let the OS assign an available port to us
        let listener = TcpListener::bind(format!("{}:0", _host)).await.unwrap();
        // We retrieve the port assigned to us by the OS
        let port = listener.local_addr().unwrap().port();

        // The `move` keyword is used to **move** the ownership of `listener` into the task.
        let _ = tokio::spawn(async move {
            let (tx, _) = mpsc::channel(32);
            let state = state::AppState { tx };
            let app = app(state);
            axum::serve(listener, app).await.unwrap();
        });

        // We return the application address to the caller!
        format!("http://{}:{}", _host, port)
    }

    /// The #[tokio::test] annotation on the test_sse_engine_completion function is a macro.
    /// Similar to #[tokio::main] It transforms the async fn test_sse_engine_completion()
    /// into a synchronous fn test_sse_engine_completion() that initializes a runtime instance
    /// and executes the async main function.
    #[tokio::test]
    async fn test_sse_engine_completion() {
        let listening_url = spawn_app("127.0.0.1").await;
        let mut completions: Vec<Completion> = vec![];
        let model_name = "code-llama-7b";
        let body = serde_json::json!({
            "model": model_name,
            "prompt": "Hello, world!",
            "best_of": 1,
            "echo": false,
            "frequency_penalty": 0.0,
            "logit_bias": Null,
            "logit_bias": Null,
            "max_tokens": 16,
            "n": 1,
            "presence_penalty": 0,
            "seed": Null,
            "stop": Null,
            "stream": true,
            "suffix": Null,
            "temperature": 1,
            "top_p": 1,
            "user": Null,
        });

        let time_before_request = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let mut stream = reqwest::Client::new()
            .post(&format!(
                "{}/v1/engines/{engine}/completions",
                listening_url,
                engine = model_name
            ))
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await
            .unwrap()
            .bytes_stream()
            .eventsource();

        // iterate over the stream of events
        // and collect them into a vector of Completion objects
        while let Some(event) = stream.next().await {
            match event {
                Ok(event) => {
                    // break the loop at the end of SSE stream
                    if event.data == "[DONE]" {
                        break;
                    }

                    // parse the event data into a Completion object
                    let completion = serde_json::from_str::<Completion>(&event.data).unwrap();
                    completions.push(completion);
                }
                Err(_) => {
                    panic!("Error in event stream");
                }
            }
        }
        // The endpoint should return at least one completion object
        assert!(completions.len() > 0);

        // Check that each completion object has the correct fields
        // note that we didn't check all the values of the fields because
        // `serde_json::from_str::<Completion>` should panic if the field is missing or in unexpected format
        for completion in completions {
            // id should be a non-empty string
            assert!(completion.id.len() > 0);
            assert!(completion.object == "text_completion");
            assert!(completion.created >= time_before_request);
            assert!(completion.model == model_name);

            // each completion object should have at least one choice
            assert!(completion.choices.len() > 0);

            // check that each choice has a non-empty text
            for choice in completion.choices {
                assert!(choice.text.len() > 0);
                // finish_reason should can be None or Some(String)
                match choice.finish_reason {
                    Some(finish_reason) => {
                        assert!(finish_reason.len() > 0);
                    }
                    None => {}
                }
            }

            assert!(completion.system_fingerprint == "");
        }
    }
}
