use std::{
    io::Write,
    process::{Command, Stdio},
};
fn run(args: &[&str], input: &str) -> std::process::Output {
    let mut child = Command::new(env!("CARGO_BIN_EXE_bpe-cli"))
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .take()
        .unwrap()
        .write_all(input.as_bytes())
        .unwrap();
    child.wait_with_output().unwrap()
}
#[test]
fn cli_training_multiline_empty_decode_and_error_paths() {
    let path = std::env::temp_dir().join(format!("bpe-cli-test-{}.json", std::process::id()));
    let model = path.to_str().unwrap();
    let trained = run(
        &[
            "train",
            "--input",
            "-",
            "--vocab-size",
            "280",
            "--out",
            model,
        ],
        "{\"text\":\"hello hello\"}\n{\"text\":\"বাংলা বাংলা\"}\n",
    );
    assert!(trained.status.success(), "{:?}", trained);
    for text in ["", "  hello\nবাংলা\t\r\n", "</w> \0"] {
        let encoded = run(&["encode", "--model", model], text);
        assert!(encoded.status.success());
        let decoded = run(
            &["decode", "--model", model],
            std::str::from_utf8(&encoded.stdout).unwrap(),
        );
        assert!(decoded.status.success());
        assert_eq!(decoded.stdout, text.as_bytes());
    }
    assert!(!run(&["decode", "--model", model], "[999999]")
        .status
        .success());
    assert!(
        !run(&["train", "--input", "-", "--out", model], "{\"bad\":42}")
            .status
            .success()
    );
    let batch = run(
        &["encode", "--model", model, "--jsonl"],
        "{\"text\":\"\"}\n{\"text\":\"hi\"}\n",
    );
    assert!(batch.status.success());
    assert_eq!(
        std::str::from_utf8(&batch.stdout).unwrap().lines().count(),
        2
    );
    std::fs::remove_file(path).unwrap();
}
