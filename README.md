# Rust Tokenizer

A basic [Byte Pair Encoding (BPE)](https://en.wikipedia.org/wiki/Byte_pair_encoding) tokenizer implemented in Rust. This project provides a foundational implementation for tokenizing text using custom merge operations and vocabulary management. It is designed to be modular, efficient, and easily extendable for industry-grade NLP applications.

## Features

- **Basic BPE Implementation:** Encode text into token IDs and decode them back to strings.
- **Custom Vocabulary & Merges:** Dynamically build vocabularies and define merge rules.
- **Error Handling:** Basic error management for out-of-vocabulary tokens and token ID mismatches.
- **Modular Structure:** Separated modules for models, training, and error definitions.
- **Testing:** Includes unit and integration tests to ensure basic functionality.

## Getting Started

### Prerequisites

- [Rust](https://www.rust-lang.org/) (Edition 2021 or later)
- Cargo (Rust’s package manager)

### Installation

Clone the repository and build the project:

```bash
git clone https://github.com/yourusername/rust_tokenizer.git
cd rust_tokenizer/bpe
cargo build
```

### Usage

Run the basic implementation:

```bash
cargo test
```
This command executes all unit and integration tests, validating the functionality of the tokenizer.

## Roadmap & Future Enhancements

We have outlined several phases for further development:

- [ ] **Phase 1: Basic Functionality**
  - Finalize and optimize the current implementation.
  - Expand unit and integration tests.
- [ ] **Phase 2: Enhanced Error Handling**
  - Implement robust error handling for edge cases.
  - Improve logging and detailed error messages.
- [ ] **Phase 3: Advanced Tokenization Techniques**
  - Extend the trainer with more sophisticated BPE algorithms.
  - Introduce support for subword tokenization and dynamic vocabulary management.
- [ ] **Phase 4: Industry-Ready Features**
  - Optimize performance for large-scale corpora.
  - Benchmark performance and integrate with popular NLP pipelines.

## Contributing

Contributions are welcome! Please see the [CONTRIBUTING.md](https://github.com/majisouvik26/rust_tokenizer/blob/main/CONTRIBUTING.md) file for guidelines on how to get involved.

## Contact

For any inquiries or feedback, please contact me at [b22cs089@iitj.ac.in](mailto:b22cs089@iitj.ac.in).