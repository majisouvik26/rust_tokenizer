use bpe::{EncodeOptions, SpecialMode, Tokenizer};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{fs, path::PathBuf};

#[derive(Deserialize)]
struct Fixture {
    case_id: usize,
    text: String,
    ids: Vec<u32>,
    spans: Vec<[usize; 2]>,
}
fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .to_owned()
}
#[test]
fn ten_thousand_frozen_production_ids_and_preprocessing_spans() {
    let bytes =
        fs::read(root().join("fixtures/gpt2.jsonl")).expect("run scripts/generate_fixtures.py");
    let manifest: serde_json::Value =
        serde_json::from_slice(&fs::read(root().join("fixtures/gpt2.manifest.json")).unwrap())
            .unwrap();
    assert_eq!(
        format!("{:x}", Sha256::digest(&bytes)),
        manifest["fixture_sha256"].as_str().unwrap()
    );
    let tokenizer = Tokenizer::load(root().join("models/gpt2.json")).unwrap();
    assert_eq!(
        tokenizer.model().sha256().unwrap(),
        manifest["model_file_sha256"].as_str().unwrap()
    );
    let mut count = 0;
    for line in std::str::from_utf8(&bytes).unwrap().lines() {
        let f: Fixture = serde_json::from_str(line).unwrap();
        assert_eq!(
            tokenizer.encode(&f.text).unwrap(),
            f.ids,
            "IDs: case {}: {:?}",
            f.case_id,
            f.text
        );
        assert_eq!(
            tokenizer.decode_bytes(&f.ids).unwrap(),
            f.text.as_bytes(),
            "bytes: case {}",
            f.case_id
        );
        assert_eq!(
            tokenizer
                .pretoken_spans(&f.text)
                .unwrap()
                .iter()
                .map(|r| [r.start, r.end])
                .collect::<Vec<_>>(),
            f.spans,
            "spans: case {}",
            f.case_id
        );
        count += 1;
    }
    assert!(count >= 10000);
    assert_eq!(count, manifest["cases"].as_u64().unwrap());
}
#[derive(Deserialize)]
struct SpecialFixture {
    text: String,
    ids: Vec<u32>,
}
#[test]
fn allowed_specials_match_both_production_libraries() {
    let tokenizer = Tokenizer::load(root().join("models/gpt2.json")).unwrap();
    let fixtures: Vec<SpecialFixture> =
        serde_json::from_slice(&fs::read(root().join("fixtures/gpt2-special.json")).unwrap())
            .unwrap();
    let options = EncodeOptions {
        special: SpecialMode::Allow(["<|endoftext|>".into()].into()),
        ..Default::default()
    };
    for f in fixtures {
        assert_eq!(tokenizer.encode_with(&f.text, &options).unwrap(), f.ids);
        assert_eq!(tokenizer.decode_utf8(&f.ids).unwrap(), f.text);
    }
}
