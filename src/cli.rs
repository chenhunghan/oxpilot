use clap::Parser;

#[derive(Parser)]
#[command(name = "ox")]
#[command(
    about = "Pragmatic AI pipeline.",
    long_about = "Pragmatic AI pipeline for inter-process data flow."
)]
pub struct CLIArgs {
    /// Wether to start the copilot server, default is false.
    #[arg(short = 'c', long = "serve")]
    pub copilot_serve: bool,
    /// The port to bind the copilot server on, default to 9090, only used if --serve is set.
    #[arg(short = 'p', long = "port", default_value = "9090")]
    pub port: u16,
    /// The seed to use when generating random samples, default to 1.0
    #[arg(short = 't', long, default_value_t = 1.0)]
    pub temperature: f64,
    /// The seed to use when generating random samples, default to 299792458
    #[arg(short = 's', long, default_value_t = 299792458)]
    pub seed: u64,
    /// The length of the sample to generate (in tokens).
    #[arg(short = 'n', long, default_value_t = 1000)]
    pub to_sample: usize,
    /// Nucleus sampling probability cutoff.
    #[arg(long)]
    pub top_p: Option<f64>,
    /// Penalty to be applied for repeating tokens, 1. means no penalty
    #[arg(short = 'r', long, default_value_t = 1.1)]
    pub repeat_penalty: f32,
    /// The context size to consider for the repeat penalty, default to 64
    #[arg(long, default_value_t = 64)]
    pub repeat_last_n: usize,
    /// HG tokenizer repo id, default to "mistralai/Mistral-7B-v0.1"
    #[arg(long, default_value = "mistralai/Mistral-7B-v0.1")]
    pub tokenizer_repo_id: String,
    /// HG tokenizer repo revision, default to "main"
    #[arg(long, default_value = "main")]
    pub tokenizer_repo_revision: String,
    /// HG tokenizer repo tokenizer file, default to "tokenizer.json"
    #[arg(long, default_value = "tokenizer.json")]
    pub tokenizer_file: String,
    // HG model repo id, default to "TheBloke/OpenHermes-2.5-Mistral-7B-GGUF"
    #[arg(long, default_value = "TheBloke/OpenHermes-2.5-Mistral-7B-GGUF")]
    pub model_repo_id: String,
    /// HG model repo revision, default to "main"
    #[arg(long, default_value = "main")]
    pub model_repo_revision: String,
    /// HG model repo GGMl/GGUF file, default to "openhermes-2.5-mistral-7b.Q4_K_M.gguf"
    #[arg(long, default_value = "openhermes-2.5-mistral-7b.Q4_K_M.gguf")]
    pub model_file_name: String,
}
