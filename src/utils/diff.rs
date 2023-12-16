use tracing::warn;

/// Get the diff of the staged files. None if there is no diff/failed to get the diff.
pub fn get_diff(function_context: bool) -> Option<String> {
    let mut git = std::process::Command::new("git");
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
    let output = git.output().expect("failed to execute diff");
    if output.status.success() {
        match String::from_utf8(output.stdout) {
            Ok(stdout) => {
                if stdout.len() > 0 {
                    return Some(stdout);
                }
                None
            }
            Err(e) => {
                warn!("failed to parse diff output: {:?}", e);
                None
            }
        }
    } else {
        None
    }
}
