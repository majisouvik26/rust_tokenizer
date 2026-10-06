use base64::{engine::general_purpose::STANDARD, Engine};
use bpe::{
    Backend, BpeModel, BpeTrainer, EncodeOptions, Merge, Pretokenizer, RuntimeConfig, SpecialMode,
    Token, Tokenizer, TrainConfig, TrainerBackend,
};

fn trained(
    records: &[String],
    pretokenizer: Pretokenizer,
    backend: TrainerBackend,
    min_frequency: u64,
) -> BpeModel {
    BpeTrainer::new(TrainConfig {
        vocab_size: 300,
        min_frequency,
        pretokenizer,
    })
    .train_with(records, backend)
    .unwrap()
}
fn options(backend: Backend) -> EncodeOptions {
    EncodeOptions {
        backend,
        ..Default::default()
    }
}
fn next(seed: &mut u64) -> u64 {
    *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    *seed >> 32
}

#[test]
fn both_trainers_match_weighted_ties_overlaps_and_seeded_corpora() {
    let mut seed = 42;
    let alphabet = ['a', 'b', 'c', ' ', '\t', '\n', 'é', 'ক', 'ह', '🙂', '\0'];
    for case in 0..80 {
        let mut records: Vec<String> = ["aaaaaaa", "aaabbb", "ab", "bc", "ab", "bc", "", "</w>"]
            .iter()
            .map(|s| (*s).into())
            .collect();
        for _ in 0..12 {
            let length = (next(&mut seed) % 64) as usize;
            records.push(
                (0..length)
                    .map(|_| alphabet[next(&mut seed) as usize % alphabet.len()])
                    .collect(),
            );
        }
        let pre = if case % 2 == 0 {
            Pretokenizer::Raw
        } else {
            Pretokenizer::gpt2()
        };
        let min = case % 4 + 1;
        let a = trained(&records, pre.clone(), TrainerBackend::Reference, min);
        let b = trained(&records, pre.clone(), TrainerBackend::Incremental, min);
        assert_eq!(
            a.canonical_json().unwrap(),
            b.canonical_json().unwrap(),
            "corpus {case}"
        );
        records.reverse();
        assert_eq!(
            a.sha256().unwrap(),
            trained(&records, pre, TrainerBackend::Incremental, min)
                .sha256()
                .unwrap()
        );
    }
    for pre in [Pretokenizer::Raw, Pretokenizer::gpt2()] {
        assert_eq!(
            trained(&[], pre.clone(), TrainerBackend::Reference, 2)
                .canonical_json()
                .unwrap(),
            trained(&[], pre, TrainerBackend::Incremental, 2)
                .canonical_json()
                .unwrap()
        );
    }
}

#[test]
fn aliases_finish_selected_rank_before_new_lower_rank() {
    let mut data = BpeModel::byte_only(Pretokenizer::Raw)
        .unwrap()
        .data()
        .clone();
    for bytes in [b"bc".as_slice(), b"ab", b"abc", b"abca"] {
        data.tokens.push(Token {
            id: data.tokens.len() as u32,
            bytes_b64: STANDARD.encode(bytes),
        });
    }
    for (left, right, out) in [
        (98, 99, 256),
        (97, 98, 257),
        (257, 99, 258),
        (258, 97, 259),
        (97, 256, 258),
    ] {
        data.merges.push(Merge {
            left,
            right,
            out,
            rank: data.merges.len() as u32,
        });
    }
    let tokenizer = Tokenizer::new(BpeModel::from_data(data).unwrap()).unwrap();
    let reference = tokenizer
        .trace(
            "abcabc",
            &options(Backend::Reference),
            &RuntimeConfig::default(),
        )
        .unwrap();
    assert_eq!(reference.ids, [258, 258]);
    assert_eq!(
        reference,
        tokenizer
            .trace("abcabc", &options(Backend::Heap), &RuntimeConfig::default())
            .unwrap()
    );
}

#[test]
fn trace_and_ids_match_on_random_text_and_long_overlaps() {
    let records: Vec<String> = [
        "a".repeat(200),
        "abcabcabcabc".into(),
        "হिन्दী🙂".repeat(8),
        "def f(x): return x + 1\n".repeat(4),
    ]
    .into();
    let mut seed = 7;
    for pre in [Pretokenizer::Raw, Pretokenizer::gpt2()] {
        let tokenizer =
            Tokenizer::new(trained(&records, pre, TrainerBackend::Reference, 2)).unwrap();
        let mut texts = records.clone();
        texts.extend(["a".repeat(8193), "".into(), "\0\r\n".into()]);
        for _ in 0..150 {
            texts.push(
                (0..100)
                    .map(|_| ['a', 'b', 'c', ' ', 'é', '🙂'][next(&mut seed) as usize % 6])
                    .collect(),
            );
        }
        for text in texts {
            let reference = tokenizer
                .trace(
                    &text,
                    &options(Backend::Reference),
                    &RuntimeConfig::default(),
                )
                .unwrap();
            assert_eq!(reference.ids, tokenizer.encode(&text).unwrap());
            for backend in [Backend::Heap, Backend::Auto] {
                let config = RuntimeConfig {
                    heap_threshold: Some(2),
                    ..Default::default()
                };
                assert_eq!(
                    reference,
                    tokenizer.trace(&text, &options(backend), &config).unwrap()
                );
                assert_eq!(
                    reference.ids,
                    tokenizer
                        .session(config)
                        .unwrap()
                        .encode(&text, &options(backend))
                        .unwrap()
                );
            }
        }
    }
}

#[test]
fn cache_is_bounded_model_local_and_special_safe() {
    let mut data = BpeModel::byte_only(Pretokenizer::Raw)
        .unwrap()
        .data()
        .clone();
    data.special_tokens.insert("<x>".into(), 256);
    let tokenizer = Tokenizer::new(BpeModel::from_data(data).unwrap()).unwrap();
    let mut session = tokenizer
        .session(RuntimeConfig {
            cache_capacity: 2,
            cache_bytes: 48,
            scratch_capacity_limit: 8,
            ..Default::default()
        })
        .unwrap();
    for _ in 0..3 {
        assert_eq!(
            session.encode("abc", &options(Backend::Heap)).unwrap(),
            [97, 98, 99]
        );
    }
    assert_eq!(session.cache_stats().hits, 2);
    for text in ["def", "ghi", "longer than the cache budget", "<x>"] {
        session.encode(text, &options(Backend::Heap)).unwrap();
        assert!(session.cache_stats().entries <= 2);
        assert!(session.cache_stats().payload_bytes <= 48);
    }
    let allow = EncodeOptions {
        backend: Backend::Heap,
        special: SpecialMode::Allow(["<x>".into()].into()),
    };
    assert_eq!(session.encode("<x>", &allow).unwrap(), [256]);
    let reject = EncodeOptions {
        special: SpecialMode::Reject,
        ..options(Backend::Heap)
    };
    assert!(session.encode("<x>", &reject).is_err());
    assert_eq!(
        session.encode("abc", &options(Backend::Reference)).unwrap(),
        [97, 98, 99]
    );
    session.clear_cache();
    assert_eq!(session.cache_stats().entries, 0);
    assert_eq!(session.cache_stats().hits, 0);
    assert_eq!(
        session
            .encode(&"a".repeat(100000), &options(Backend::Heap))
            .unwrap()
            .len(),
        100000
    );
    assert_eq!(session.encode("a", &options(Backend::Heap)).unwrap(), [97]);
}

#[test]
fn fresh_and_reused_scratch_agree() {
    let tokenizer = Tokenizer::new(trained(
        &["abcabcabc".into()],
        Pretokenizer::Raw,
        TrainerBackend::Reference,
        1,
    ))
    .unwrap();
    for backend in [Backend::Reference, Backend::Heap, Backend::Auto] {
        let mut session = tokenizer
            .session(RuntimeConfig {
                reuse_buffers: false,
                heap_threshold: Some(2),
                ..Default::default()
            })
            .unwrap();
        for text in ["abcabc", "", "aaaa", "🙂"] {
            assert_eq!(
                session.encode(text, &options(backend)).unwrap(),
                tokenizer.encode(text).unwrap()
            );
        }
    }
}

#[test]
fn serial_build_rejects_unavailable_workers_and_invalid_options() {
    let tokenizer = Tokenizer::new(BpeModel::byte_only(Pretokenizer::Raw).unwrap()).unwrap();
    assert!(tokenizer
        .batch_encoder(0, RuntimeConfig::default())
        .is_err());
    assert!(tokenizer
        .session(RuntimeConfig {
            heap_threshold: Some(1),
            ..Default::default()
        })
        .is_err());
    assert!(tokenizer
        .session(RuntimeConfig {
            cache_capacity: 4,
            cache_bytes: 0,
            ..Default::default()
        })
        .is_err());
    #[cfg(not(all(feature = "parallel", not(target_arch = "wasm32"))))]
    assert!(tokenizer
        .batch_encoder(2, RuntimeConfig::default())
        .is_err());
}

#[cfg(all(feature = "parallel", not(target_arch = "wasm32")))]
#[test]
fn parallel_batches_keep_order_with_and_without_cache() {
    let tokenizer = Tokenizer::new(trained(
        &["abcabcabc".into()],
        Pretokenizer::Raw,
        TrainerBackend::Reference,
        1,
    ))
    .unwrap();
    let texts: Vec<_> = (0..137)
        .map(|i| format!("{i}:abcabc🙂{}", "a".repeat(i % 17)))
        .collect();
    let expected = tokenizer
        .encode_batch(&texts, &options(Backend::Reference))
        .unwrap();
    for threads in [1, 2, 4] {
        for cache_capacity in [0, 8] {
            let mut batch = tokenizer
                .batch_encoder(
                    threads,
                    RuntimeConfig {
                        cache_capacity,
                        ..Default::default()
                    },
                )
                .unwrap();
            for _ in 0..3 {
                assert_eq!(
                    expected,
                    batch.encode(&texts, &options(Backend::Heap)).unwrap()
                );
            }
            assert_eq!(
                batch.encode(&[""], &options(Backend::Heap)).unwrap(),
                [Vec::<u32>::new()]
            );
            assert!(batch
                .encode::<String>(&[], &options(Backend::Heap))
                .unwrap()
                .is_empty());
            assert_eq!(batch.cache_stats().unwrap().len(), threads);
            batch.clear_cache().unwrap();
        }
    }
}

#[test]
fn trace_offsets_follow_special_strings_and_empty_batches_validate_options() {
    let mut data = trained(
        &["abababab".into()],
        Pretokenizer::Raw,
        TrainerBackend::Reference,
        2,
    )
    .data()
    .clone();
    let special_id = data.tokens.len() as u32;
    data.special_tokens.insert("<x>".into(), special_id);
    let tokenizer = Tokenizer::new(BpeModel::from_data(data).unwrap()).unwrap();
    let reference = EncodeOptions {
        backend: Backend::Reference,
        special: SpecialMode::Allow(["<x>".into()].into()),
    };
    let heap = EncodeOptions {
        backend: Backend::Heap,
        ..reference.clone()
    };
    let a = tokenizer
        .trace("ab<x>ab", &reference, &RuntimeConfig::default())
        .unwrap();
    let b = tokenizer
        .trace("ab<x>ab", &heap, &RuntimeConfig::default())
        .unwrap();
    assert_eq!(a, b);
    assert_eq!(
        a.events
            .iter()
            .map(|event| (event.start, event.end))
            .collect::<Vec<_>>(),
        [(0, 2), (5, 7)]
    );
    assert_eq!(tokenizer.decode_utf8(&a.ids).unwrap(), "ab<x>ab");
    let invalid = EncodeOptions {
        special: SpecialMode::Allow(["unknown".into()].into()),
        ..options(Backend::Heap)
    };
    assert!(tokenizer.encode_batch::<String>(&[], &invalid).is_err());
    assert!(tokenizer
        .batch_encoder(1, RuntimeConfig::default())
        .unwrap()
        .encode::<String>(&[], &invalid)
        .is_err());
}
