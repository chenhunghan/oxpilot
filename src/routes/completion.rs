use async_stream::stream;
use axum::extract::State;
use axum::response::sse::{Event as SseEvent, KeepAlive, Sse};
use axum::Json;
use futures::stream::Stream;
use oxpilot::cmd::Command::Prompt;
use oxpilot::types::{Choice, Completion, CompletionRequest, Usage};
use serde_json::{json, to_string};
use std::convert::Infallible;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::mpsc;
use tracing::info;

use crate::state::AppState;

// Reference: https://github.com/tokio-rs/axum/blob/main/examples/sse/src/main.rs
pub async fn completion(
    State(state): State<AppState>,
    // `Json<T>` will automatically deserialize the request body to a type `T` as JSON.
    Json(body): Json<CompletionRequest>,
) -> Sse<impl Stream<Item = Result<SseEvent, Infallible>>> {
    // `stream!` is a macro from [`async_stream`](https://docs.rs/async-stream/0.3.5/async_stream/index.html)
    // that makes it easy to create a `futures::stream::Stream` from a generator.
    Sse::new(stream! {
        let prompt = body.prompt.unwrap_or("".to_string());
        // the `tx` is a `tokio::sync::mpsc::Sender` that was created in `main.rs`.
        // we can use the `tx` to send a `Command::Prompt` to the manager task.
        let tx = state.tx.clone();
        let (responder, mut receiver) = mpsc::channel(8);

        // send the `Command::Prompt` to the manager task with responder
        tx.send(Prompt {
            prompt,
            responder,
            temperature: body.temperature.unwrap_or(1.0),
        }).await.unwrap();

        // the manager task will send the completion back to us via the `responder`.
        // the receiver will receive the generated `text` from the `responder`.
        while let Some(text) = receiver.recv().await {
          info!("Received completion: {}", text);
          // Let's create one instance of `SseEvent` with the generated `text`, and respond to the SSE client.
          yield Ok(
            // Create a new `SseEvent` with the default settings.
            // `SseEvent::default().data("Hello, World!")` will return `data: Hello, World!` as the event text chuck.
            SseEvent::default().data(
              // Serialize the `Completion` struct to JSON and return it as the event text chunk.
              to_string(
                // json! is a macro from serde_json that makes it easy to create JSON values from a struct.
                &json!(
                  Completion {
                    id: "cmpl-".to_string(),
                    object: "text_completion".to_string(),
                    created: SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .unwrap()
                        .as_secs(),
                    model: body.model.clone().unwrap_or("unknown".to_string()),
                    choices: vec![Choice {
                        text: text.to_string(),
                        index: 0,
                        logprobs: None,
                        finish_reason: Some("stop".to_string()),
                    }],
                    usage: Usage {
                        prompt_tokens: 0,
                        completion_tokens: 0,
                        total_tokens: 0
                    },
                    system_fingerprint: "".to_string(),
                  }
                )).unwrap()
              )
          );
        }
    })
    .keep_alive(KeepAlive::default())
}
