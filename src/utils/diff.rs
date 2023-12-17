use tokio::process::Command;

/// Get the diff of the staged files. None if there is no diff/failed to get the diff.
pub async fn get_diff(function_context: bool) -> String {
    let mut git = Command::new("git");
    git.arg("diff")
        .arg("--staged")
        .arg("--ignore-all-space")
        .arg("--ignore-blank-lines")
        .arg("--diff-algorithm=histogram")
        .arg("--no-ext-diff")
        .arg("--no-color");

    if function_context {
        git.arg("--function-context");
    }
    let output = git.output().await.expect("failed to execute diff");
    return String::from_utf8(output.stdout).expect("failed to parse diff stdout");
}
