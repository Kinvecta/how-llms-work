# Syllabus

**How Large Language Models Work: From Math to Transformers** is a free video course in 30 episodes and six parts. It starts with "what is an LLM?" and builds every piece from the math up, to a working transformer and how models are trained, scaled and evaluated.

[YouTube playlist](https://www.youtube.com/playlist?list=PLAw1yNagEJRM) · [series trailer](https://youtu.be/rXTCgq-qoJA) · [back to the README](../README.md)

## How every episode works

- **Plain words first, with a picture you already know.** Then a **Math Lab** works through the derivation for anyone who wants it.
- **One running sentence**, "The cat sat on the mat because it was tired.", is tokenized, embedded, attended over and generated from across the whole series.
- **A companion notebook** reproduces every number on screen, with an `assert` on each one.
- **A research brief** lists the sources behind every claim.
- **Notation is fixed once and kept.** A symbol introduced in one episode means the same thing in every later one.

## By the end of the series you can

1. Explain what an LLM computes, in words and as an equation, and how text is generated from it.
2. Do the math the transformer is built on: vectors, matrices, probability, entropy and the chain rule of calculus.
3. Train a small network end to end and explain each step: loss, gradients, backpropagation and the optimizer.
4. Explain tokenization, embeddings, attention and the transformer block, and build a tiny GPT.
5. Describe how real models are pretrained, scaled, post-trained and evaluated, and where they fail.

## Who it's for

Curious learners, students and professionals who want to understand LLMs past the buzzwords. High-school algebra is enough to start; every other piece of math is built in the series. The notebooks need no install in Colab.

## Episodes

| # | Part | Episode | Notebook | Video |
|:--:|:--:|---|---|---|
| 1 | I | **[What an LLM Really Is](../Episode%2001%20-%20What%20an%20LLM%20Really%20Is/)**<br><sub>The model as a function from the text so far to a probability for every next token; generation as a loop; how the dials get set; three claims checked</sub> | [notebook](../Episode%2001%20-%20What%20an%20LLM%20Really%20Is/notebook.ipynb) | [![Watch](https://img.shields.io/badge/watch-YouTube-red?logo=youtube&logoColor=white)](https://youtu.be/ZJ4BbrnXBRo) |
| 2 | I | **[Numbers as Geometry: Vectors](../Episode%2002%20-%20Numbers%20as%20Geometry%20Vectors/)**<br><sub>Words as lists of numbers; dot product, length, cosine similarity</sub> | [notebook](../Episode%2002%20-%20Numbers%20as%20Geometry%20Vectors/notebook.ipynb) | [video](https://youtu.be/onxRHQlY-TI) |
| 3 | I | **Matrices as Machines**<br><sub>A matrix as a machine that maps vectors; weights and biases</sub> | [notebook](../notebooks/Episode%2003%20-%20Matrices.ipynb) | video in production |
| 4 | I | **Probability, Entropy and Perplexity**<br><sub>Distributions, softmax, surprise, cross-entropy and perplexity</sub> | [notebook](../notebooks/Episode%2004%20-%20Probability%20Entropy%20Perplexity.ipynb) | video in production |
| 5 | I | **Calculus for Learning**<br><sub>Derivatives, the gradient and the chain rule, as the tools learning needs</sub> | [notebook](../notebooks/Episode%2005%20-%20Calculus%20for%20Learning.ipynb) | video in production |
| 6 | II | **What Learning Means: Data, Models, Loss**<br><sub>Training data, a model with parameters, and a loss that scores it</sub> | [notebook](../notebooks/Episode%2006%20-%20Learning%20Loss.ipynb) | video in production |
| 7 | II | **Gradient Descent: Walking Downhill**<br><sub>Following the gradient; learning rate; steps toward lower loss</sub> | [notebook](../notebooks/Episode%2007%20-%20Gradient%20Descent.ipynb) | video in production |
| 8 | II | **Generalization: Overfitting, Validation, Regularization**<br><sub>Practice score vs exam score; held-out data; keeping models honest</sub> | [notebook](../notebooks/Episode%2008%20-%20Generalization.ipynb) | video in production |
| 9 | II | **Turning Text into Training Examples**<br><sub>How one page of text becomes many next-token questions</sub> | [notebook](../notebooks/Episode%2009%20-%20Training%20Examples.ipynb) | video in production |
| 10 | II | **Tokens and Embeddings**<br><sub>Byte-pair encoding and the embedding table</sub> | [notebook](../notebooks/Episode%2010%20-%20Tokens%20and%20Embeddings.ipynb) | video in production |
| 11 | III | **The Artificial Neuron and Activation Functions**<br><sub>Weighted sums, biases and the bend (ReLU, GELU) that makes depth useful</sub> | [notebook](../notebooks/Episode%2011%20-%20Neuron%20and%20Activations.ipynb) | video in production |
| 12 | III | **Deep Networks and Backpropagation**<br><sub>Layers, and how blame flows backward through them</sub> | [notebook](../notebooks/Episode%2012%20-%20Backpropagation.ipynb) | video in production |
| 13 | III | **Training a Network End to End**<br><sub>Putting the pieces together: data, forward pass, loss, gradients, optimizer</sub> | [notebook](../notebooks/Episode%2013%20-%20Training%20End%20to%20End.ipynb) | video in production |
| 14 | III | **What Networks Learn: Representations**<br><sub>Probing what the hidden layers encode</sub> | [notebook](../notebooks/Episode%2014%20-%20Representations.ipynb) | video in production |
| 15 | IV | **Before Transformers: Counting Words**<br><sub>N-gram models, smoothing and what counting can and cannot do</sub> | [notebook](../notebooks/Episode%2015%20-%20Counting%20Words.ipynb) | video in production |
| 16–22 | IV | **The Transformer**<br><sub>Recurrent networks, attention, the transformer block, sampling and temperature, and building a tiny GPT</sub> | with each episode | planned |
| 23–26 | V | **Building and Scaling a Model**<br><sub>Pretraining, scaling laws, training at scale, and post-training (instruction tuning, human feedback, reasoning RL)</sub> | with each episode | planned |
| 27–30 | VI | **Beyond the Basics**<br><sub>New architectures, LLM systems, how LLMs fail (memorization included), and evaluation and governance</sub> | with each episode | planned |

Episode topics and order after Episode 15 may change during production. The notebooks for Episodes 3–15 are published ahead of the videos; their numbers match the episode research, and the videos will follow.

## The six parts

| Part | Title | Episodes |
|:--:|---|---|
| I | Foundations | 1–5: what an LLM is, vectors, matrices, probability, calculus |
| II | How Machines Learn | 6–10: loss, gradient descent, generalization, training examples, tokens and embeddings |
| III | Neural Networks | 11–14: neurons, backpropagation, training, representations |
| IV | The Transformer | 15–22: n-grams, RNNs, attention, the transformer block, building a tiny GPT |
| V | Building and Scaling a Model | 23–26: pretraining, scaling laws, training at scale, post-training |
| VI | Beyond the Basics | 27–30: new architectures, LLM systems, how LLMs fail, evaluation and governance |
