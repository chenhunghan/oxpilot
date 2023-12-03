use async_stream::stream;
use axum::response::sse::{Event as SseEvent, KeepAlive, Sse};
use axum::Json;
use futures::stream::Stream;
use oxpilot::types::{Choice, Completion, CompletionRequest, Usage};
use serde_json::{json, to_string};
use std::convert::Infallible;
use std::time::{SystemTime, UNIX_EPOCH};

// Reference: https://github.com/tokio-rs/axum/blob/main/examples/sse/src/main.rs
pub async fn completion(
    // `Json<T>` will automatically deserialize the request body to a type `T` as JSON.
    Json(body): Json<CompletionRequest>,
) -> Sse<impl Stream<Item = Result<SseEvent, Infallible>>> {
    // `stream!` is a macro from [`async_stream`](https://docs.rs/async-stream/0.3.5/async_stream/index.html)
    // that makes it easy to create a `futures::stream::Stream` from a generator.
    Sse::new(stream! {
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
                  model: body.model.unwrap_or("unknown".to_string()),
                  choices: vec![Choice {
                      text: " world!".to_string(),
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
    })
    .keep_alive(KeepAlive::default())
}
