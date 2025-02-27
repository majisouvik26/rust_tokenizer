use bpe_tokenizer::{model::BpeModel, trainer::BpeTrainer, errors::TokenizerError};

#[test]
fn test_integration() -> Result<(), TokenizerError> {
    let corpus = "low low low low low low water";
    let trainer = BpeTrainer::new(3);
    let model = trainer.train(corpus)?;

    let encoded = model.encode("lower")?;
    let decoded = model.decode(&encoded)?;
    assert_eq!(decoded, "lower");
    Ok(())
}
