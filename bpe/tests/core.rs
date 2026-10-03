use base64::{engine::general_purpose::STANDARD, Engine};
use bpe::{
    Backend, BpeModel, BpeTrainer, EncodeOptions, Merge, Pretokenizer, SpecialMode, Token,
    Tokenizer, TokenizerError, TrainConfig,
};

fn raw() -> BpeModel {
    BpeModel::byte_only(Pretokenizer::Raw).unwrap()
}
fn model(rules: &[(u32, u32)]) -> BpeModel {
    let mut data = raw().data().clone();
    for (rank, &(left, right)) in rules.iter().enumerate() {
        let mut bytes = STANDARD
            .decode(&data.tokens[left as usize].bytes_b64)
            .unwrap();
        bytes.extend(
            STANDARD
                .decode(&data.tokens[right as usize].bytes_b64)
                .unwrap(),
        );
        let out = data.tokens.len() as u32;
        data.tokens.push(Token {
            id: out,
            bytes_b64: STANDARD.encode(bytes),
        });
        data.merges.push(Merge {
            left,
            right,
            out,
            rank: rank as u32,
        });
    }
    BpeModel::from_data(data).unwrap()
}
fn train(records: &[&str], cap: usize, pretokenizer: Pretokenizer) -> BpeModel {
    BpeTrainer::new(TrainConfig {
        vocab_size: cap,
        min_frequency: 2,
        pretokenizer,
    })
    .train(records)
    .unwrap()
}

#[test]
fn spaces_and_literal_word_end_marker_are_lossless() {
    let tokenizer = Tokenizer::new(train(
        &["low low low water", "</w>  \n"],
        290,
        Pretokenizer::gpt2(),
    ))
    .unwrap();
    for text in [
        " low  water ",
        "  \t\r\n",
        "literal </w> remains </w> ",
        "",
        "\0",
        "lower",
    ] {
        let ids = tokenizer.encode(text).unwrap();
        assert_eq!(tokenizer.decode_utf8(&ids).unwrap(), text);
    }
}
#[test]
fn priority_beats_leftmost_pair() {
    let tokenizer = Tokenizer::new(model(&[(98, 99), (97, 98)])).unwrap();
    assert_eq!(tokenizer.encode("abc").unwrap(), [97, 256]);
}
#[test]
fn overlaps_are_nonoverlapping_left_to_right() {
    let tokenizer = Tokenizer::new(model(&[(97, 97)])).unwrap();
    assert_eq!(tokenizer.encode("aaaa").unwrap(), [256, 256]);
    assert_eq!(tokenizer.encode("aaaaa").unwrap(), [256, 256, 97]);
}
#[test]
fn training_ties_and_repeated_process_state_are_stable() {
    let expected = train(&["aaabbb"], 258, Pretokenizer::Raw);
    assert_eq!(
        expected.merges()[0],
        Merge {
            left: 97,
            right: 97,
            out: 256,
            rank: 0
        }
    );
    assert_eq!(
        expected.merges()[1],
        Merge {
            left: 98,
            right: 98,
            out: 257,
            rank: 1
        }
    );
    for _ in 0..20 {
        assert_eq!(
            train(&["aaabbb"], 258, Pretokenizer::Raw).sha256().unwrap(),
            expected.sha256().unwrap()
        );
    }
    let a = train(&["ab", "bc", "ab", "bc"], 260, Pretokenizer::Raw);
    let b = train(&["bc", "ab", "bc", "ab"], 260, Pretokenizer::Raw);
    assert_eq!(a.canonical_json().unwrap(), b.canonical_json().unwrap());
}
#[test]
fn corpus_and_pretoken_boundaries_block_merges() {
    assert!(train(&["a", "b", "a", "b"], 260, Pretokenizer::Raw)
        .merges()
        .is_empty());
    assert!(train(&["a!", "a!"], 260, Pretokenizer::gpt2())
        .merges()
        .is_empty());
    assert_eq!(
        train(&["a!", "a!"], 260, Pretokenizer::Raw).merges().len(),
        1
    );
}
#[test]
fn weighted_chunks_count_overlapping_adjacent_occurrences() {
    assert_eq!(train(&["aaa"], 257, Pretokenizer::Raw).merges().len(), 1); // aa count=2
    assert_eq!(
        train(&["ab", "ab", "ab"], 257, Pretokenizer::Raw).merges()[0].left,
        97
    );
    assert!(train(&["ab"], 300, Pretokenizer::Raw).merges().is_empty());
    assert!(train(&[], 300, Pretokenizer::Raw).merges().is_empty());
}
#[test]
fn all_bytes_and_partial_unicode_decode() {
    let m = raw();
    let ids: Vec<_> = (0..=255).collect();
    assert_eq!(
        m.decode_bytes(&ids).unwrap(),
        (0..=255u8).collect::<Vec<_>>()
    );
    assert!(matches!(
        m.decode_utf8(&[0xc3]),
        Err(TokenizerError::Utf8(_))
    ));
    assert_eq!(m.decode_utf8(&[0xc3, 0xa9]).unwrap(), "é");
    assert!(matches!(
        m.decode_bytes(&[999]),
        Err(TokenizerError::UnknownToken(999))
    ));
    let tokenizer = Tokenizer::new(m).unwrap();
    assert_eq!(tokenizer.decode_lossy(&[0xff]).unwrap(), "�");
}
#[test]
fn multilingual_and_seeded_unicode_roundtrips() {
    let tokenizer = Tokenizer::new(train(
        &["বাংলা বাংলা", "हिन्दी हिन्दी", "hello hello", "👩🏽‍💻 👩🏽‍💻"],
        300,
        Pretokenizer::gpt2(),
    ))
    .unwrap();
    let mut seed = 42u64;
    for _ in 0..1000 {
        let mut text = String::new();
        for _ in 0..50 {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            if let Some(c) = char::from_u32((seed >> 32) as u32 % 0x110000) {
                text.push(c);
            }
        }
        assert_eq!(
            tokenizer
                .decode_bytes(&tokenizer.encode(&text).unwrap())
                .unwrap(),
            text.as_bytes()
        );
    }
}
#[test]
fn save_load_and_canonical_hash_are_stable() {
    let m = train(&["hello hello", "বাংলা বাংলা"], 300, Pretokenizer::gpt2());
    let path = std::env::temp_dir().join(format!("bpe-day1-{}.json", std::process::id()));
    m.save(&path).unwrap();
    let loaded = BpeModel::load(&path).unwrap();
    std::fs::remove_file(path).unwrap();
    assert_eq!(m.sha256().unwrap(), loaded.sha256().unwrap());
    let pretty = serde_json::to_vec_pretty(m.data()).unwrap();
    assert_eq!(
        m.canonical_json().unwrap(),
        BpeModel::from_json(&pretty)
            .unwrap()
            .canonical_json()
            .unwrap()
    );
}
#[test]
fn duplicate_merges_and_aliases_are_rejected() {
    let mut data = model(&[(97, 98)]).data().clone();
    data.merges.push(Merge {
        left: 97,
        right: 98,
        out: 256,
        rank: 1,
    });
    assert!(BpeModel::from_data(data).is_err());
    let mut data = raw().data().clone();
    data.tokens[255].bytes_b64 = data.tokens[254].bytes_b64.clone();
    assert!(BpeModel::from_data(data).is_err());
}
#[test]
fn existing_output_id_may_be_reused_without_aliasing_bytes() {
    let mut data = model(&[(97, 98), (98, 99), (97, 257)]).data().clone();
    data.merges.push(Merge {
        left: 256,
        right: 99,
        out: 258,
        rank: 3,
    });
    let tokenizer = Tokenizer::new(BpeModel::from_data(data).unwrap()).unwrap();
    assert_eq!(tokenizer.encode("abc").unwrap(), [258]);
}
#[test]
fn malformed_models_return_typed_errors() {
    assert!(matches!(
        BpeModel::from_json(b"{"),
        Err(TokenizerError::Json(_))
    ));
    let mut d = raw().data().clone();
    d.format_version = 2;
    assert!(matches!(
        BpeModel::from_data(d),
        Err(TokenizerError::UnsupportedVersion(2))
    ));
    let mut d = raw().data().clone();
    d.tokens[0].id = 1;
    assert!(BpeModel::from_data(d).is_err());
    let mut d = raw().data().clone();
    d.tokens[0].bytes_b64 = "***".into();
    assert!(BpeModel::from_data(d).is_err());
    let mut d = raw().data().clone();
    d.tokens.pop();
    assert!(BpeModel::from_data(d).is_err());
    let mut d = model(&[(97, 98)]).data().clone();
    d.merges[0].rank = 5;
    assert!(BpeModel::from_data(d).is_err());
    let mut d = model(&[(97, 98)]).data().clone();
    d.merges[0].left = 256;
    assert!(BpeModel::from_data(d).is_err());
    let mut d = model(&[(97, 98)]).data().clone();
    d.merges[0].out = 0;
    assert!(BpeModel::from_data(d).is_err());
    let mut d = raw().data().clone();
    d.special_tokens.insert("".into(), 256);
    assert!(BpeModel::from_data(d).is_err());
    let mut d = raw().data().clone();
    d.special_tokens.insert("<x>".into(), 0);
    assert!(BpeModel::from_data(d).is_err());
    let mut d = raw().data().clone();
    d.special_tokens
        .extend([("<x>".into(), 256), ("<y>".into(), 256)]);
    assert!(BpeModel::from_data(d).is_err());
    let mut d = raw().data().clone();
    d.special_tokens.insert("a".into(), 256);
    assert!(BpeModel::from_data(d).is_err());
}
#[test]
fn specials_are_explicit_and_longest_match_wins() {
    let mut data = raw().data().clone();
    data.special_tokens
        .extend([("<x>".into(), 256), ("<x>long".into(), 257)]);
    let tokenizer = Tokenizer::new(BpeModel::from_data(data).unwrap()).unwrap();
    assert_eq!(tokenizer.encode("<x>").unwrap(), [60, 120, 62]);
    let allow = EncodeOptions {
        special: SpecialMode::Allow(["<x>".into(), "<x>long".into()].into()),
        ..Default::default()
    };
    assert_eq!(
        tokenizer.encode_with("a<x>long<x>", &allow).unwrap(),
        [97, 257, 256]
    );
    let reject = EncodeOptions {
        special: SpecialMode::Reject,
        ..Default::default()
    };
    assert!(matches!(
        tokenizer.encode_with("<x>long", &reject),
        Err(TokenizerError::DisallowedSpecial(_))
    ));
    let unknown = EncodeOptions {
        special: SpecialMode::Allow(["unknown".into()].into()),
        ..Default::default()
    };
    assert!(matches!(
        tokenizer.encode_with("", &unknown),
        Err(TokenizerError::UnknownSpecial(_))
    ));
    assert_eq!(tokenizer.decode_utf8(&[257, 256]).unwrap(), "<x>long<x>");
}
#[test]
fn serial_batch_and_auto_preserve_order() {
    let tokenizer = Tokenizer::new(raw()).unwrap();
    let texts = ["abc", "", " বাংলা "];
    let options = EncodeOptions {
        backend: Backend::Auto,
        ..Default::default()
    };
    assert_eq!(
        tokenizer.encode_batch(&texts, &options).unwrap(),
        texts
            .iter()
            .map(|t| tokenizer.encode(t).unwrap())
            .collect::<Vec<_>>()
    );
    assert!(matches!(
        tokenizer.encode_with(
            "",
            &EncodeOptions {
                backend: Backend::Heap,
                ..Default::default()
            }
        ),
        Err(TokenizerError::UnsupportedBackend(_))
    ));
}
#[test]
fn invalid_trainer_configuration_is_rejected() {
    for (vocab_size, min_frequency) in [(255, 2), (256, 0)] {
        assert!(BpeTrainer::new(TrainConfig {
            vocab_size,
            min_frequency,
            pretokenizer: Pretokenizer::Raw
        })
        .train(&["x"])
        .is_err());
    }
}

#[test]
fn raschka_toy_corpus_has_explicit_deterministic_ties() {
    let model = train(&["the cat in the hat"], 260, Pretokenizer::Raw);
    assert_eq!(
        model
            .merges()
            .iter()
            .map(|r| (r.left, r.right))
            .collect::<Vec<_>>(),
        [(97, 116), (101, 32), (104, 257), (116, 258)]
    );
    let tokenizer = Tokenizer::new(model).unwrap();
    assert_eq!(
        tokenizer.encode("the cat in the hat").unwrap(),
        [259, 99, 256, 32, 105, 110, 32, 259, 104, 256]
    );
}
