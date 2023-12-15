# `oxpilot` - rusty AI copilot

`oxpilot` is a copilot server powered by LLM (large language models) written in Rust 🦀. It's built on top of [candle](https://github.com/huggingface/candle), with aims to be minimalist, fast, and accessable (in terms of computing resource) copilot for everyone.

<p align="center">
  <img src="./doc/img/rusty-copilot.jpeg" width="320" height="320" alt="A rusty programming copilot" />
</p>

## Commands

### Start the copilot server

```sh
ox serve
```

### Commmit with LLM

```sh
git add . # staging files.
ox commit # like `git commit -m` but with the LLM generated messages.
```

## Goal of this project

The primary goal of this project is to teach (myself, and everyone else) idiomatic Rust, similar to [mini-redis](https://github.com/tokio-rs/mini-redis), thefore the code is overly heavily documented, there is an article introducing the core concepts [I made a Copilot in Rust 🦀 , here is what I have learned... (as a TypeScript dev)](https://dev.to/chenhunghan/i-made-a-copilot-in-rust-here-is-what-i-have-learned-as-a-typescript-dev-2n2p-temp-slug-6347339?preview=542b15b40bd1c6551c37ba5132030656b8fe8db5467a160112e8389e1ad7c6d901c13fd836c53124a72ab38bb0ae39d7f6de01969655b70ba69851d7), I recommand to read first, and [PRs description](https://github.com/chenhunghan/oxpilot/pulls?q=is%3Apr) are packed with design patterns used in the code base.

- [Introduction](https://dev.to/chenhunghan/i-made-a-copilot-in-rust-here-is-what-i-have-learned-as-a-typescript-dev-2n2p-temp-slug-6347339?preview=542b15b40bd1c6551c37ba5132030656b8fe8db5467a160112e8389e1ad7c6d901c13fd836c53124a72ab38bb0ae39d7f6de01969655b70ba69851d7)
- Design Patterns
  - [Builder and `impl Into<String>`](https://github.com/chenhunghan/oxpilot/pull/1)
  - [Type State: Friendly API for Better DX](https://github.com/chenhunghan/oxpilot/pull/5)
  - [BDD `POST /v1/engines/:engine/completions`](https://github.com/chenhunghan/oxpilot/pull/6)
