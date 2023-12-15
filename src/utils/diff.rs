use tracing::warn;

/// Get the diff of the staged files. None if there is no diff/failed to get the diff.
pub fn get_diff() -> Option<String> {
    let stagged_diff = std::process::Command::new("git")
        .arg("diff")
        .arg("--staged")
        .arg("--ignore-all-space")
        .arg("--diff-algorithm=minimal")
        .arg("--function-context")
        .arg("--no-ext-diff")
        .arg("--no-color")
        .output()
        .expect("failed to execute diff");
    if stagged_diff.status.success() {
        match String::from_utf8(stagged_diff.stdout) {
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
