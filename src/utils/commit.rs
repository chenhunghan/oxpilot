/// Get the diff of the staged files. None if there is no diff/failed to get the diff.
pub fn commit_then_exit(commit_message: &str, signoff: bool) -> () {
    let mut git = std::process::Command::new("git");
    git.arg("commit").arg("-m").arg(commit_message);

    if signoff {
        git.arg("--signoff");
    }
    let output = git.output().expect("failed to execute commit");
    if output.status.success() {
        println!("{}", String::from_utf8_lossy(&output.stdout));
        std::process::exit(0);
    } else {
        panic!("failed to execute commit: {:?}", output);
    }
}
