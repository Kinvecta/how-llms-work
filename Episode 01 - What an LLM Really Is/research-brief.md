# Episode 1 research brief: What an LLM Really Is

The facts behind Episode 1, with sources. Every probability in the video was measured on **GPT-2 small** (124M parameters); the [companion notebook](notebook.ipynb) reproduces each one.

**Running example, used across the whole series:**

> "The cat sat on the mat because it was tired."

## Learning objectives

1. Explain that an LLM is a probability distribution over the next token, plus the code that samples from it.
2. Write next-token prediction as an equation and explain how generation repeats it.
3. Describe the stages of building a modern LLM: pretraining, post-training, a newer reasoning stage, and inference.
4. Weigh the evidence on three common claims: "it's a database," "it understands nothing," and "it thinks like a person."

## Notation introduced in this episode

| Symbol | Meaning |
|---|---|
| θ | All learned parameters of the model (the "dials") |
| V | The vocabulary: every token the model can output (50,257 for GPT-2) |
| x₁, x₂, … xₜ | The tokens of a sequence, numbered |
| x<ₜ | The prefix: every token before position t |
| p<sub>θ</sub>(xₜ \| x<ₜ) | The model's probability for the next token, given the prefix |
| ~ | "is drawn from": xₜ ~ p<sub>θ</sub> means the next token is sampled from the distribution |

Temperature is introduced in Episode 21.

## 1. What lands on your disk

An open model is a file of parameters plus a small program that runs them. Karpathy frames it exactly this way in his 2023 talk: for Llama 2 70B, a parameters file of about 140 GB (70 billion numbers at 2 bytes each) and roughly 500 lines of C to run it, with no other dependencies. [S1]

## 2. Next-token prediction, measured

GPT-2 small, prompt "The cat sat on the" (reproduced in the notebook, Step 3):

| Rank | Next token | Probability |
|---:|---|---:|
| 1 | floor | 7.6% |
| 2 | bed | 6.5% |
| 3 | couch | 5.4% |
| 4 | ground | 5.2% |
| 5 | edge | 4.8% |
| 31 | **mat** | **0.40%** |

The model returns one probability for each of the 50,257 tokens in V, and they sum to 1. The expected word, "mat," is far down the list: the model is not retrieving a stored sentence.

**The prefix grows, the distribution is recomputed** (notebook, Step 4):

| Prefix | Next token in our sentence | Probability (rank) |
|---|---|---:|
| The cat sat | on | 22% (1) |
| The cat sat on | the | 45% (1) |
| The cat sat on the mat because | it | 22% (1) |
| The cat sat on the mat because it | was | 32% (1) |

**The one equation:**

> p<sub>θ</sub>(xₜ | x<ₜ)

**Generation is the equation in a loop:** predict the distribution, pick a token (xₜ ~ p<sub>θ</sub>), append it, repeat. Greedy decoding (always take the top token) from "The cat sat on the" gives *"The cat sat on the floor, and the cat was still asleep."* Random sampling gives a different continuation on every draw.

**Probability of a whole sentence (chain rule):**

> p(x₁, …, x<sub>T</sub>) = ∏ₜ p<sub>θ</sub>(xₜ | x<ₜ)

For the running sentence, GPT-2 small assigns about 1.5 × 10⁻¹⁷ to the ten tokens after "The." Tiny, because every sentence is one of an astronomical number of possible sentences. Episode 4 turns numbers like this into a comparable score.

## 3. Why prediction is a deep problem

Shannon (1951) measured how predictable English is by having people guess the next letter of a text. With long context he estimated 0.6 to 1.3 bits per letter, meaning roughly three quarters of the letters are predictable in principle. [S2] Predicting well takes grammar, facts, tracking who "it" is, and knowing that questions tend to be followed by answers. Prediction is the objective; capability is what the objective rewards.

## 4. How the dials get set

| Stage | What happens | Where in the series |
|---|---|---|
| **Pretraining** | Predict the next token at every position of a very large text corpus. The text supplies its own answers (self-supervised). | Episodes 6–14, 23 |
| **Post-training** | Adapt the model to follow instructions. InstructGPT (2022) set the template: about 40 labelers wrote answers to about 13,000 prompts for fine-tuning; labelers then ranked 4 to 9 answers per prompt to train a reward model; the model was then optimized against that reward on about 31,000 more prompts. [S3] | Episode 26 |
| **Reasoning RL (newer)** | Reinforcement learning on problems with checkable answers, such as math and code. DeepSeek-R1-Zero raised AIME 2024 pass@1 from 15.6% to 71.0% with this stage alone. A 2025 development, not a universal recipe. [S8] | Later in Part V |
| **Inference** | Running the finished dials, one token at a time. The part you use. | Episode 21 (sampling) |

## 5. Three claims, checked

**"It's a database that copies text."** Partly true: models can reproduce some training text verbatim, so memorization is real (Episode 29). But there is no corpus inside to look up: training compresses patterns into parameters, and a 140 GB file of numbers cannot hold the internet. If our sentence were stored, "mat" would not come back at rank 31.

**"It understands nothing."** Evidence of internal structure: Li et al. trained a GPT-style model on Othello move sequences, with no rules and no board. A nonlinear probe recovered the board state from its activations with 1.7% error, against about 26% for a randomly initialized network (linear probes did not work in that paper). [S4] Nanda, Lee and Wattenberg then found the board is linearly encoded when squares are labelled *mine* or *yours* relative to the player to move, and that editing those directions changes the model's predicted moves. [S5]
Evidence of limits: Vafa et al. trained transformers on turn-by-turn taxi directions in Manhattan. Next-turn prediction was near perfect, yet the street map implied by the models bore little resemblance to Manhattan, and small random detours made performance collapse. [S6] Strong prediction does not guarantee a coherent world model.

**"It thinks like a person."** The 2021 "stochastic parrots" paper argues that language models stitch together linguistic forms from their training data according to probabilistic information about how they combine, without reference to meaning. [S7] It is one serious position; the Othello results are a serious reply. Fluency is not thinking, and whether these systems understand anything is still argued.

## Check question

If a model is trained only to predict the next token, why might it still learn to answer questions?

<details><summary>Answer</summary>

Text is full of questions followed by answers. The cheapest way to predict what comes after a question, across millions of them, is to learn the structure that produces good answers.

</details>

## Sources

| ID | Source | Used for |
|---|---|---|
| S1 | Karpathy, "Intro to Large Language Models" (talk, November 2023) | An LLM as a parameter file plus code |
| S2 | Shannon, "Prediction and Entropy of Printed English," *Bell System Technical Journal* 30 (1951) | Predictability of English |
| S3 | Ouyang et al., "Training language models to follow instructions with human feedback," arXiv:2203.02155 (2022) | Post-training steps and dataset sizes |
| S4 | Li et al., "Emergent World Representations: Exploring a Sequence Model Trained on a Synthetic Task," ICLR 2023, arXiv:2210.13382 | Othello-GPT probes |
| S5 | Nanda, Lee and Wattenberg, "Emergent Linear Representations in World Models of Self-Supervised Sequence Models," arXiv:2309.00941 (2023) | Linear mine/yours representation |
| S6 | Vafa et al., "Evaluating the World Model Implicit in a Generative Model," NeurIPS 2024, arXiv:2406.03689 | Taxi-map study |
| S7 | Bender, Gebru, McMillan-Major and Mitchell, "On the Dangers of Stochastic Parrots: Can Language Models Be Too Big?" FAccT 2021, pp. 610–623, doi:10.1145/3442188.3445922 | The "stochastic parrot" position |
| S8 | DeepSeek-AI, "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning," arXiv:2501.12948 (2025) | Reasoning RL stage |
| S9 | Radford et al., "Language Models are Unsupervised Multitask Learners" (OpenAI, 2019) | GPT-2, the model measured in this episode |
| S10 | Jurafsky and Martin, *Speech and Language Processing*, 3rd edition draft | Background reading (n-grams, transformers, post-training) |

**Further exploration:** [Transformer Explainer](https://poloclub.github.io/transformer-explainer/) (Polo Club) runs GPT-2 small in your browser and shows the same next-token probabilities.
