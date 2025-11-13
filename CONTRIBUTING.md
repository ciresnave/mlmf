# Contributing to MLMF

Thank you for your interest in contributing to MLMF! We welcome contributions of all types.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/your-username/mlmf.git`
3. Create a feature branch: `git checkout -b feature/your-feature-name`
4. Make your changes
5. Run the tests: `cargo test --all-features`
6. Submit a pull request

## Development Setup

```bash
# Install Rust (if needed)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Clone and build
git clone https://github.com/CireSnave/mlmf.git
cd mlmf
cargo build --all-features
cargo test --all-features
```

## Guidelines

- Follow Rust naming conventions
- Add documentation for public APIs
- Include tests for new functionality
- Update the CHANGELOG.md for notable changes
- Ensure all examples still compile and run

## Code Style

We use the standard Rust formatter:

```bash
cargo fmt
```

## Reporting Issues

Please use GitHub Issues to report bugs or request features. Include:

- Rust version
- Operating system
- Minimal reproduction case
- Error messages (if any)

## License

By contributing, you agree that your contributions will be licensed under the same terms as the project (MIT OR Apache-2.0).