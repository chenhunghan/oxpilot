use std::net::SocketAddr;

use axum::{routing::post, Router};
use candle_core::utils::{get_num_threads, has_accelerate, has_mkl};
use clap::Parser;
use oxpilot::cli::CLIArgs;
use oxpilot::cmd::Command::Prompt;
use oxpilot::llm::LLMBuilder;
use oxpilot::process::process;
use routes::completion::completion;
use tokio::sync::mpsc;
use tracing::info;
use tracing_subscriber::fmt::format::FmtSpan;
pub mod llm;
pub mod process;
pub mod routes;
pub mod state;
pub mod token;

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
    // The tracing crate is a framework for instrumenting Rust programs to
    // collect structured, event-based diagnostic information.
    // https://github.com/tokio-rs/tracing
    // https://tokio.rs/tokio/topics/tracing
    // Start configuring a `fmt` subscriber
    let subscriber = tracing_subscriber::fmt()
        // Use a more compact, abbreviated log format
        .compact()
        // Display source code file paths
        .with_file(true)
        // Display source code line numbers
        .with_line_number(true)
        // Display the thread ID an event was recorded on
        .with_thread_ids(true)
        // Display the event's target (module path)
        .with_target(true)
        // Add span events
        .with_span_events(FmtSpan::ENTER | FmtSpan::CLOSE)
        // Build the subscriber
        .finish();

    // Set the subscriber as the default
    match tracing::subscriber::set_global_default(subscriber) {
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

    let cli_args = CLIArgs::parse();

    info!("tokenizer_repo_id: {:?}", &cli_args.tokenizer_repo_id);
    info!("model_repo_id: {:?}", &cli_args.model_repo_id);
    info!("model_file_name: {:?}", &cli_args.model_file_name);
    info!("initializing LLM, downloading tokenizer and model files...");
    let llm_builder = LLMBuilder::new()
        .tokenizer_repo_id(cli_args.tokenizer_repo_id)
        .model_repo_id(cli_args.model_repo_id)
        .model_file_name(cli_args.model_file_name);
    let mut llm = llm_builder.build().await.expect("Failed to build LLM");
    info!("LLM initialized");

    let (tx, mut rx) = mpsc::channel(32);
    let manager = tokio::spawn(async move {
        let seed = cli_args.seed;
        let temperature: f64 = cli_args.temperature;
        let top_p = cli_args.top_p;
        let to_sample = cli_args.to_sample;
        let repeat_last_n = cli_args.repeat_last_n;
        let repeat_penalty = cli_args.repeat_penalty;
        info!("initializing LLM manager...");
        while let Some(cmd) = rx.recv().await {
            match cmd {
                // handle Command::Prompt from `tx.send().await`;
                Prompt { prompt, responder } => {
                    info!("prompt:{}", prompt);
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
                    )
                    .await;
                }
            }
        }
    });

    if cli_args.copilot_serve {
        info!("starting copilot server on port: {}", &cli_args.port);
        let state = state::AppState { tx };
        let address = SocketAddr::from(([0, 0, 0, 0], cli_args.port));
        let listener = tokio::net::TcpListener::bind(&address).await.unwrap();
        let app = app(state);

        match axum::serve(listener, app).await {
            Ok(_) => info!("copilot server exited."),
            Err(error) => {
                info!("server exited with error: {}", error);
                info!("terminating LLM manager");
                manager.abort();
            }
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
