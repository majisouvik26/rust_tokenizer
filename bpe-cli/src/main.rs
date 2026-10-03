use bpe::{Backend, BpeTrainer, EncodeOptions, Pretokenizer, SpecialMode, Tokenizer, TrainConfig};
use clap::{Parser, Subcommand, ValueEnum};
use serde::Deserialize;
use std::{
    collections::BTreeSet,
    fs,
    io::{self, Read, Write},
    path::PathBuf,
};

#[derive(Parser)]
#[command(version, about = "Lossless byte BPE reference tokenizer")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}
#[derive(Clone, Copy, ValueEnum)]
enum Pre {
    Raw,
    Gpt2,
}
#[derive(Clone, Copy, ValueEnum)]
enum Engine {
    Reference,
    Auto,
    Heap,
}
#[derive(Clone, Copy, ValueEnum)]
enum Policy {
    Ordinary,
    Allow,
    Reject,
}
#[derive(Subcommand)]
enum Command {
    Train {
        /// JSONL records, each containing exactly {"text": "..."}; '-' reads stdin.
        #[arg(long)]
        input: PathBuf,
        #[arg(long, default_value_t = 8192)]
        vocab_size: usize,
        #[arg(long, default_value_t = 2)]
        min_frequency: u64,
        #[arg(long, value_enum, default_value_t = Pre::Gpt2)]
        pretokenizer: Pre,
        #[arg(long)]
        out: PathBuf,
    },
    Encode {
        #[arg(long)]
        model: PathBuf,
        #[arg(long, conflicts_with = "input")]
        text: Option<String>,
        /// UTF-8 text file or JSONL with --jsonl; omitted/'-' reads stdin.
        #[arg(long)]
        input: Option<PathBuf>,
        #[arg(long, conflicts_with = "text")]
        jsonl: bool,
        #[arg(long, value_enum, default_value_t = Engine::Reference)]
        backend: Engine,
        #[arg(long, value_enum, default_value_t = Policy::Ordinary)]
        special: Policy,
        /// Repeat for each allowed special. Requires --special allow.
        #[arg(long = "allow-special", requires = "special")]
        allow_special: Vec<String>,
    },
    Decode {
        #[arg(long)]
        model: PathBuf,
        /// JSON array of integer IDs; omitted/'-' reads stdin.
        #[arg(long)]
        ids_file: Option<PathBuf>,
        #[arg(long, conflicts_with = "lossy")]
        bytes: bool,
        #[arg(long)]
        lossy: bool,
    },
    Inspect {
        #[arg(long)]
        model: PathBuf,
    },
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Record {
    text: String,
}
fn read(path: Option<&PathBuf>) -> Result<String, Box<dyn std::error::Error>> {
    if let Some(path) = path.filter(|path| path.as_os_str() != "-") {
        return Ok(fs::read_to_string(path)?);
    }
    let mut input = String::new();
    io::stdin().read_to_string(&mut input)?;
    Ok(input)
}
fn records(input: &str) -> Result<Vec<String>, Box<dyn std::error::Error>> {
    input
        .lines()
        .enumerate()
        .map(|(index, line)| {
            serde_json::from_str::<Record>(line)
                .map(|r| r.text)
                .map_err(|error| format!("JSONL line {}: {error}", index + 1).into())
        })
        .collect()
}
fn run() -> Result<(), Box<dyn std::error::Error>> {
    match Cli::parse().command {
        Command::Train {
            input,
            vocab_size,
            min_frequency,
            pretokenizer,
            out,
        } => {
            let texts = records(&read(Some(&input))?)?;
            let pretokenizer = match pretokenizer {
                Pre::Raw => Pretokenizer::Raw,
                Pre::Gpt2 => Pretokenizer::gpt2(),
            };
            let model = BpeTrainer::new(TrainConfig {
                vocab_size,
                min_frequency,
                pretokenizer,
            })
            .train(&texts)?;
            model.save(&out)?;
            println!(
                "{}",
                serde_json::json!({"vocab_size":model.vocab_size(), "merges":model.merges().len(), "model_sha256":model.sha256()?, "out":out})
            );
        }
        Command::Encode {
            model,
            text,
            input,
            jsonl,
            backend,
            special,
            allow_special,
        } => {
            if !matches!(special, Policy::Allow) && !allow_special.is_empty() {
                return Err("--allow-special requires --special allow".into());
            }
            let tokenizer = Tokenizer::load(model)?;
            let options = EncodeOptions {
                backend: match backend {
                    Engine::Reference => Backend::Reference,
                    Engine::Auto => Backend::Auto,
                    Engine::Heap => Backend::Heap,
                },
                special: match special {
                    Policy::Ordinary => SpecialMode::Ordinary,
                    Policy::Reject => SpecialMode::Reject,
                    Policy::Allow => {
                        SpecialMode::Allow(allow_special.into_iter().collect::<BTreeSet<_>>())
                    }
                },
            };
            let input = match text {
                Some(text) => text,
                None => read(input.as_ref())?,
            };
            if jsonl {
                for ids in tokenizer.encode_batch(&records(&input)?, &options)? {
                    println!("{}", serde_json::to_string(&ids)?);
                }
            } else {
                println!(
                    "{}",
                    serde_json::to_string(&tokenizer.encode_with(&input, &options)?)?
                );
            }
        }
        Command::Decode {
            model,
            ids_file,
            bytes,
            lossy,
        } => {
            let tokenizer = Tokenizer::load(model)?;
            let ids: Vec<u32> = serde_json::from_str(&read(ids_file.as_ref())?)?;
            if bytes {
                println!("{}", serde_json::to_string(&tokenizer.decode_bytes(&ids)?)?);
            } else {
                let decoded = if lossy {
                    tokenizer.decode_lossy(&ids)?
                } else {
                    tokenizer.decode_utf8(&ids)?
                };
                io::stdout().write_all(decoded.as_bytes())?; // No invented trailing newline.
            }
        }
        Command::Inspect { model } => {
            let tokenizer = Tokenizer::load(model)?;
            let m = tokenizer.model();
            println!(
                "{}",
                serde_json::to_string_pretty(
                    &serde_json::json!({"format_version":1, "profile":m.data().profile, "pretokenizer":m.pretokenizer(), "vocab_size":m.vocab_size(), "merges":m.merges().len(), "special_tokens":m.special_tokens(), "model_sha256":m.sha256()?})
                )?
            );
        }
    }
    Ok(())
}
fn main() {
    if let Err(error) = run() {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}
