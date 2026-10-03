"""Pinned compatible GPT-2 adapters. No automatic BOS/EOS or prefix space."""
from import_gpt2 import source_data
from tokenizers import Tokenizer, AddedToken, models, pre_tokenizers, decoders
import tiktoken

def engines():
    vocab, merges = source_data()
    hf = Tokenizer(models.BPE(vocab=vocab, merges=merges, cache_capacity=0))
    hf.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    hf.decoder = decoders.ByteLevel()
    hf_allowed = Tokenizer(models.BPE(vocab=vocab, merges=merges, cache_capacity=0))
    hf_allowed.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    hf_allowed.decoder = decoders.ByteLevel()
    hf_allowed.add_special_tokens([AddedToken("<|endoftext|>", normalized=False, special=True)])
    tk = tiktoken.get_encoding("gpt2")
    return tk, hf, hf_allowed
