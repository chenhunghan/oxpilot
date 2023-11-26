type Responder<T> = tokio::sync::mpsc::Sender<T>;

pub enum Command {
    Prompt {
        prompt: String,
        responder: Responder<String>,
    },
}