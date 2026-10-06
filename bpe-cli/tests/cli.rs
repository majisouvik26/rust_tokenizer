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

#[test]
fn selectable_trainers_trace_and_parallel_batch_agree() {
    let directory = std::env::temp_dir().join(format!("bpe-cli-backends-{}", std::process::id()));
    std::fs::create_dir_all(&directory).unwrap();
    let input = "{\"text\":\"abcabcabc\"}\n{\"text\":\"aaabbb\"}\n";
    let mut models = Vec::new();
    for trainer in ["reference", "incremental"] {
        let path = directory.join(format!("{trainer}.json"));
        let output = run(
            &[
                "train",
                "--input",
                "-",
                "--pretokenizer",
                "raw",
                "--vocab-size",
                "280",
                "--trainer",
                trainer,
                "--out",
                path.to_str().unwrap(),
            ],
            input,
        );
        assert!(output.status.success(), "{:?}", output);
        models.push(std::fs::read(&path).unwrap());
    }
    assert_eq!(models[0], models[1]);
    let path = directory.join("reference.json");
    let model = path.to_str().unwrap();
    let reference = run(&["encode", "--model", model, "--jsonl"], input);
    let heap = run(
        &[
            "encode",
            "--model",
            model,
            "--jsonl",
            "--backend",
            "heap",
            "--threads",
            "2",
            "--cache-capacity",
            "8",
        ],
        input,
    );
    assert!(reference.status.success() && heap.status.success());
    assert_eq!(reference.stdout, heap.stdout);
    let a = run(
        &[
            "trace",
            "--model",
            model,
            "--text",
            "abcabc",
            "--backend",
            "reference",
        ],
        "",
    );
    let b = run(
        &[
            "trace",
            "--model",
            model,
            "--text",
            "abcabc",
            "--backend",
            "heap",
        ],
        "",
    );
    assert!(a.status.success() && b.status.success());
    assert_eq!(a.stdout, b.stdout);
    std::fs::remove_dir_all(directory).unwrap();
}
