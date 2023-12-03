use axum::{routing::post, Router};
use oxpilot::llm::LLMBuilder;
use routes::completion::completion;
pub mod routes;

#[tokio::main]
async fn main() {
    let llm_builder = LLMBuilder::new()
        .tokenizer_repo_id("hf-internal-testing/llama-tokenizer")
        .model_repo_id("TheBloke/CodeLlama-7B-GGUF")
        .model_file_name("codellama-7b.Q2_K.gguf");
    let _ = llm_builder.build().await.expect("Failed to build LLM");

    let listener = tokio::net::TcpListener::bind("0.0.0.0:6666").await.unwrap();
    let app = app();
    axum::serve(listener, app).await.unwrap();
}

fn app() -> Router {
    Router::new()
        .route("/v1/engines/:engine/completions", post(completion))
        .route("/v1/completions", post(completion))
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

        let _ = tokio::spawn(async move {
            let app = app();
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
                    if event.data == "[DONE]" {
                        break;
                    }

                    let completion = serde_json::from_str::<Completion>(&event.data).unwrap();
                    completions.push(completion);
                }
                Err(_) => {
                    panic!("Error in event stream");
                }
            }
        }
        // return at least one completion object
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
