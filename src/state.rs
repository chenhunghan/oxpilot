use oxpilot::cmd::Command;

#[derive(Clone)]
pub struct AppState {
    pub tx: tokio::sync::mpsc::Sender<Command>,
}
