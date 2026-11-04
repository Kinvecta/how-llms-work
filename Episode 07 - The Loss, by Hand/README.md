# Episode 7 · The Loss, by Hand

[![Watch on YouTube](https://img.shields.io/badge/watch-YouTube-red?logo=youtube&logoColor=white)](https://youtu.be/ia7aFhn_20k) [![Open in Colab](https://img.shields.io/badge/open-Colab%20notebook-F9AB00?logo=googlecolab&logoColor=white)](https://colab.research.google.com/github/Kinvecta/how-llms-work/blob/main/Episode%2007%20-%20The%20Loss%2C%20by%20Hand/notebook.ipynb) [![Slides](https://img.shields.io/badge/view-slides%20%28PDF%29-8A2BE2)](slides.pdf) [![Research brief](https://img.shields.io/badge/read-brief-blue)](research-brief.md) [![Transcript](https://img.shields.io/badge/read-transcript-lightgrey)](transcript/Episode%2007%20-%20The%20Loss%2C%20by%20Hand%20-%20transcript.txt) [![Short](https://img.shields.io/badge/watch-the%20Short-red?logo=youtube&logoColor=white)](https://youtu.be/qzDae9gvggc)

<a href="https://youtu.be/ia7aFhn_20k"><img src="thumbnail.png" width="560" alt="Watch Episode 7: The Loss, by Hand"></a>

**Length:** 11:21 · **Part II:** How Machines Learn · **Math:** ŷ = Wu + β, L(θ) = (1/N) Σ (yᵢ − ŷᵢ)², H(q, p) = −Σ q ln p and perplexity = e^L · **Code:** NumPy and matplotlib, runs anywhere

Three dots on a grid. Two straight lines through them. One line misses a single dot by one; the other misses all three, but only a little. Which line is better? You probably have a gut feeling. A computer needs a number, and not just "better": how much better. Episode 6 learned to step downhill; this episode, part 1 of What Learning Means, names the height we step down on: the loss. Every step is worked by hand: the mean squared error of two lines (1/3 and 1/18, six times smaller), a third line for good measure, cross-entropy for a word that collapses to one term, ln 0.30 from ln 3 and ln 10, and a two-word loss of 0.857 nats. Then the myth: a model that is right can still pay.

## What you'll learn

- What learning changes: the knobs θ (here W and β; millions in a language model), and the difference between training and inference
- How to compute mean squared error by hand: predict, take the error, square, add, divide by N
- Why the line that misses every dot (1/18) beats the line that is perfect twice and wrong once (1/3), and why it is the least-squares fit
- Why cross-entropy against a one-hot target keeps only one term: the surprisal of the observed word
- How to work −ln 0.30 = 1.204 nats by hand, and why one term is not the loss: the loss is the mean, 0.857 nats
- Why the loss and perplexity are the same measurement in two units: e^0.857 ≈ 2.36
- Why "the loss counts wrong answers" is a myth

## Key ideas

- A model is a function with knobs. Learning means turning the knobs θ to make one number small: the loss. Training changes θ; inference runs the model with θ held fixed.
- The loss is a mean over N terms. For a number, each term is a squared error: L(θ) = (1/N) Σ (yᵢ − ŷᵢ)².
- Squaring stops a miss above and a miss below from cancelling, and makes a big miss expensive.
- For a word, the target q is one-hot (1 for the observed word, 0 for the rest), so the cross-entropy H(q, p) = −Σ q ln p is just −ln p of the observed word: its surprisal.
- The loss for words is the mean surprisal, L(θ) = (1/N) Σ −ln p_θ(observed token). Perplexity is e^L: push one down and the other follows.
- The loss is a penalty that training pushes down. It is not a count of mistakes: a right answer at 0.45 still pays 0.799 nats.

## Worked by hand

**The data and the model.** Three examples (u, y): (1, 2), (2, 4), (3, 5), so N = 3. The model: ŷ = Wu + β, two knobs.

**Line A: W = 2, β = 0.**

| u | Label y | Prediction ŷ | Error y − ŷ | Square |
|:--:|:--:|:--:|:--:|:--:|
| 1 | 2 | 2 × 1 = 2 | 0 | 0 |
| 2 | 4 | 2 × 2 = 4 | 0 | 0 |
| 3 | 5 | 2 × 3 = 6 | 5 − 6 = −1 | (−1)² = 1 |
| | | | Sum | 1 |
| | | | ÷ N = 3 | **MSE = 1/3 ≈ 0.333** |

**Line B: W = 1.5, β = 2/3** (the least-squares fit).

| u | Label y | Prediction ŷ | Error y − ŷ | Square |
|:--:|:--:|:--:|:--:|:--:|
| 1 | 2 | 1.5 + 2/3 = 13/6 | −1/6 | 1/36 |
| 2 | 4 | 3 + 2/3 = 11/3 | +1/3 | 1/9 = 4/36 |
| 3 | 5 | 4.5 + 2/3 = 31/6 | −1/6 | 1/36 |
| | | | Sum | 6/36 = 1/6 |
| | | | ÷ N = 3 | **MSE = 1/18 ≈ 0.056** |

Line B misses every dot, and its loss is six times smaller than line A's. No other W and β does better. One more: **line C**, W = 1.5, β = 1/2, predicts 2, 3.5 and 5; the errors are 0, 1/2 and 0, and the MSE is **1/12 ≈ 0.083**, still worse than line B.

**The loss for words** (after "The cat sat on the"; Episode 1's probabilities are illustrative; the observed word is mat).

| Step | Work | Result |
|---|---|---|
| Four terms | −(1 · ln 0.30 + 0 · ln 0.20 + 0 · ln 0.15 + 0 · ln 0.35) | −ln 0.30 |
| By hand | ln 0.3 = ln 3 − ln 10 ≈ 1.099 − 2.303 = −1.204 | **1.204 nats** |
| Second word, p = 0.60 | −(ln 6 − ln 10) = −(1.792 − 2.303) | **0.511 nats** |
| The loss, N = 2 | (1.204 + 0.511) / 2 = 1.715 / 2 | **0.857 nats** |
| Perplexity | e^0.857 | **≈ 2.36** |

**The bet, the myth and the check question.**

| Step | Work | Result |
|---|---|---|
| Bet: share on the true word | 0.9 · 0.5 · 0.1 · 0.01, you pay −ln(share) | **0.105 · 0.693 · 2.303 · 4.605** |
| Myth: right at 0.45 | −ln 0.45 | **0.799 nats** |
| Check: two terms | −ln 0.25 · −ln 0.50 | **1.386 · 0.693** |
| Check: the loss | (1.386 + 0.693) / 2 | **1.040 nats** |
| Check: first rises to 0.40 | −ln 0.40 = 0.916; (0.916 + 0.693) / 2 | **0.805 nats**, down |

## Key terms

| Term | Meaning |
|---|---|
| Parameters, θ | The model's knobs: W and β here, millions in a language model |
| Training | Changing θ to make the loss small |
| Inference | Running the model with θ held fixed |
| Label, y | The right answer for one example |
| Prediction, ŷ | What the model outputs: ŷ = Wu + β |
| Loss, L(θ) | One number, the mean of N per-term penalties, that training pushes down |
| Mean squared error (MSE) | The loss for numbers: (1/N) Σ (yᵢ − ŷᵢ)² |
| Least-squares fit | The W and β with the smallest MSE: line B here |
| One-hot target, q | 1 for the observed word, 0 for every other word |
| Cross-entropy, H(q, p) | −Σ q ln p; with a one-hot q, the surprisal of the observed word |
| Surprisal | −ln p of the word that actually came next, in nats (Episode 4) |
| Perplexity | e raised to the loss: the same measurement in another unit |

## Run the notebook

The [companion notebook](notebook.ipynb) works every calculation the way the video does, prints each line of the worksheet with exact fractions, plots the lines, the gaps and their squares, and checks every value with an `assert`. It runs in seconds.

| Part | What it shows | Matches the video |
|:--:|---|---|
| 1 | The knobs ŷ = Wu + β and the three data points | (1, 2), (2, 4), (3, 5) · N = 3 |
| 2 | `mse_by_hand` for lines A, B and C; line B checked with `np.polyfit` and a grid of 150,801 knob settings; the three lines with their gaps and squares (plot) | 1/3 ≈ 0.333 · 1/18 ≈ 0.056 · 1/12 ≈ 0.083 |
| 3 | Cross-entropy written as four terms; ln 0.30 = ln 3 − ln 10; the second word; the mean; perplexity, checked as 1/√0.18; the probabilities and the two penalty bars (plot) | 1.204 · 0.511 · 0.857 · 2.36 |
| 4 | The bet payouts on the curve −ln p (plot); the myth; the check question | 0.105 · 0.693 · 2.303 · 4.605 · 0.799 · 1.040 → 0.805 |
| 5 | Try it: your own W and β, your own probability on mat, and three words of your choice, each with a self-check | your numbers |

Or run it locally:

```bash
git clone https://github.com/Kinvecta/how-llms-work.git
cd how-llms-work
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python tools/check_notebooks.py "Episode 07 - The Loss, by Hand/notebook.ipynb"
# PASS  Episode 07 - The Loss, by Hand/notebook.ipynb
```

## Try it

About 15 minutes.

1. Pick your own W and β for the three dots and compute the mean squared error by hand. Can you beat 1/18? Put your knobs and your answer in the notebook's `my_W`, `my_beta` and `my_mse` to check.
2. In the notebook, change the probability on mat (`my_p_mat`) and watch its surprisal change.
3. Compute the loss for three words you choose, then e to that loss, and check it against perplexity (`my_probs`, `my_loss`, `my_perplexity`).

## Check your understanding

1. What does training change, and what does inference do with it?
2. Line A (W = 2, β = 0) predicts 2, 4 and 6 for the labels 2, 4 and 5. What is its mean squared error, step by step?
3. Line B misses every dot and line A hits two exactly. Why does line B have the smaller loss, and can any other W and β beat it?
4. The model gives mat 0.30, floor 0.20, sofa 0.15 and other 0.35, and the observed word is mat. Why does only one term of the cross-entropy survive, and what is it?
5. A batch has two terms. The model gives the observed word probability 0.25 the first time and 0.50 the second. What is each term's surprisal, what is the loss, and does raising the first probability to 0.40 lower it?
6. True or false: the loss counts wrong answers.

<details><summary>Answers</summary>

1. Training changes the parameters θ (the knobs, here W and β) to make the loss small. Inference runs the model with θ held fixed.
2. Errors 0, 0, −1; squares 0, 0, 1; sum 1; divide by N = 3: MSE = 1/3 ≈ 0.333.
3. Squaring makes the big miss expensive. Line A's one miss of 1 costs 1; line B's misses of 1/6, 1/3 and 1/6 cost 1/36 + 4/36 + 1/36 = 1/6 in total, so its MSE is 1/18 ≈ 0.056, six times smaller. No: line B is the least-squares fit, so no other W and β gives a smaller MSE for these three points.
4. The target q is one-hot: 1 for mat, 0 for the rest, so every other term is multiplied by zero. What is left is −ln 0.30 = −(ln 3 − ln 10) ≈ −(1.099 − 2.303) = 1.204 nats, mat's surprisal.
5. 1.386 and 0.693 nats; the loss is their mean, 1.040 nats. Yes: −ln 0.40 = 0.916, so the loss becomes (0.916 + 0.693) / 2 = 0.805 nats.
6. False. A perfect fit has zero squared error, and all probability on the observed word gives zero cross-entropy, but a model whose top choice at 0.45 is the observed word still pays −ln 0.45 = 0.799 nats. The loss is a penalty that training pushes down, not a count of mistakes.

</details>

## Further reading

- The [research brief](research-brief.md) for this episode, with every source
- Jurafsky and Martin, *Speech and Language Processing*, 3rd ed. draft, chapter 7, §§7.6–7.8 (training with the average cross-entropy loss, decoding, and perplexity)
- Goodfellow, Bengio and Courville, *Deep Learning* (MIT Press, 2016), §3.13 (cross-entropy) and §5.5.1 (mean squared error, eq. 5.66)
- 3Blue1Brown, "Gradient descent, how neural networks learn", Chapter 2 (the cost of one example, then the average over the training examples)

---

**Next:** Episode 8, What Learning Means, Part 2: Maximum Likelihood. Why these two losses, and not some other? Multiply 0.30 by 0.60, take a log, and something surprising falls out. Its notebook is already in [`notebooks/`](../notebooks/).

[← Back to the course](../README.md)
