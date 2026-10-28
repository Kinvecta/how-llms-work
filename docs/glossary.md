# Glossary and notation

Every term and symbol the released episodes use, in plain language. The series fixes each symbol once and keeps it, so a letter introduced in Episode 1 means the same thing in Episode 30. New entries are added as episodes are released.

[Back to the README](../README.md) · [syllabus](../course/)

## Notation

| Symbol | Read it as | Meaning | First used |
|---|---|---|---|
| θ | "theta" | All the learned parameters of a model, together: the dials | Ep 1 |
| V | "vocabulary" | The set of every token the model can output. \|V\| = 50,257 for GPT-2 | Ep 1 |
| x₁, x₂, …, xₜ | "x one, x two, up to x t" | The tokens of a sequence, numbered in order | Ep 1 |
| x<ₜ | "x less than t" | The prefix: every token before position t | Ep 1 |
| p<sub>θ</sub>(xₜ \| x<ₜ) | "p theta of x t given x less than t" | The model's probability for the next token, given the prefix. The one equation of Episode 1 | Ep 1 |
| ~ | "is drawn from" | xₜ ~ p<sub>θ</sub> means the next token is sampled from the distribution | Ep 1 |
| ∏ₜ | "product over t" | Multiply the terms for every position t (the chain rule for a whole sentence) | Ep 1 |

## Terms

| Term | Meaning | Episode |
|---|---|---|
| Large language model (LLM) | A learned function from the text so far to a probability for every possible next token, plus the small program that runs it | 1 |
| Parameters (weights) | The learned numbers inside a model. Llama 2 70B has 70 billion of them; GPT-2 small has 124 million | 1 |
| Token | A chunk of text the model reads and writes: usually a word or part of a word, often with its leading space | 1 |
| Vocabulary | The fixed list of every token a model knows | 1 |
| Prefix | The tokens so far, before the position being predicted | 1 |
| Next-token prediction | Computing a probability for every token in the vocabulary as the next one | 1 |
| Generation (autoregressive) | Predict, pick, append, repeat: one token at a time, each added to the prefix | 1 |
| Greedy decoding | Always picking the single most likely next token | 1 |
| Sampling | Drawing the next token at random, in proportion to its probability | 1 |
| Chain rule of probability | The probability of a whole sequence is the product of each token's probability given the tokens before it | 1 (in depth: 4) |
| Pretraining | Training on a very large corpus to predict the next token at every position. The text supplies its own answers (self-supervised) | 1 (in depth: 6–14, 23) |
| Post-training | Adapting a pretrained model to follow instructions, for example by fine-tuning on human-written answers and optimizing against a model of human rankings | 1 (in depth: 26) |
| Reward model | A model trained to predict which of several answers people would rank higher | 1 |
| Reasoning reinforcement learning | Further training on problems with checkable answers, such as math and code | 1 |
| Inference | Running a trained model to produce output. The parameters are read, not changed | 1 |
| Probe | A small model trained to read a property (such as an Othello board) out of another model's internal activations | 1 (in depth: 14) |
| Memorization | Reproducing training text word for word. Real, but not the same as looking text up | 1 (in depth: 29) |
| World model | An internal representation of the state of the world that generated the data (a board, a street map) | 1 |
| Stochastic parrot | The 2021 argument that language models stitch together seen forms by probability, without reference to meaning. One position in an open debate | 1 |
