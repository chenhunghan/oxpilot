use clap::{Parser, Subcommand};
use std::ffi::OsString;

#[derive(Parser)]
#[command(name = "ox")]
#[command(
    about = "Pragmatic AI pipeline.",
    long_about = "Pragmatic AI pipeline for inter-process data flow."
)]
pub struct CLI {
    #[command(subcommand)]
    pub command: Option<CLICommands>,
    /// Change tracing verbose level, `-q` print nothing, not even errors, no `-v` flag (default) = print errors only, `-vv` = print errors and warnings,
    /// `-vvv` = print errors, warnings, and info `-vvvv` = print errors, warnings, info, and debug, `-vvvvv` = print everything including trace
    #[command(flatten)]
    pub verbose: clap_verbosity_flag::Verbosity,
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
    #[arg(long, default_value = "mistralai/Mistral-7B-Instruct-v0.2")]
    pub tokenizer_repo_id: String,
    /// HG tokenizer repo revision, default to "main"
    #[arg(long, default_value = "main")]
    pub tokenizer_repo_revision: String,
    /// HG tokenizer repo tokenizer file, default to "tokenizer.json"
    #[arg(long, default_value = "tokenizer.json")]
    pub tokenizer_file: String,
    // HG model repo id, default to "TheBloke/OpenHermes-2.5-Mistral-7B-GGUF"
    #[arg(long, default_value = "TheBloke/Mistral-7B-Instruct-v0.2-GGUF")]
    pub model_repo_id: String,
    /// HG model repo revision, default to "main"
    #[arg(long, default_value = "main")]
    pub model_repo_revision: String,
    /// HG model repo GGMl/GGUF file, default to "openhermes-2.5-mistral-7b.Q4_K_M.gguf"
    #[arg(long, default_value = "mistral-7b-instruct-v0.2.Q4_K_M.gguf")]
    pub model_file_name: String,
}

#[derive(Debug, Subcommand)]
pub enum CLICommands {
    /// Generate a commit message from staged files.
    Commit {
        /// Do not excute the action, only print the output, use it with --commit to generate a commit message without committing.
        #[arg(long = "dry-run")]
        dry_run: bool,
        /// Generate the diff with function context, default to true. Set to false to reduce the diff size to speed up the generation.
        /// https://git-scm.com/docs/git-diff#Documentation/git-diff.txt---function-context
        #[arg(long = "function-context", default_value = "true")]
        function_context: bool,
    },
    /// Start the copilot server at `--port`, default to 9090.
    Serve {
        /// The port to bind the copilot server on, default to 9090, only used if `ox serve`.
        #[arg(short = 'p', long = "port", default_value = "9090")]
        port: u16,
    },
    /// Arbitrary inputs will be parsed as prompt. e.g. `ox How are you today?` will generate the response by prompting "How are you today?".
    #[command(external_subcommand)]
    Any(Vec<OsString>),
}
