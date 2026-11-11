# Episode 8 · Maximum Likelihood, by Hand

[![Watch on YouTube](https://img.shields.io/badge/watch-YouTube-red?logo=youtube&logoColor=white)](https://youtu.be/u1bSbpKfF6g) [![Open in Colab](https://img.shields.io/badge/open-Colab%20notebook-F9AB00?logo=googlecolab&logoColor=white)](https://colab.research.google.com/github/Kinvecta/how-llms-work/blob/main/Episode%2008%20-%20Maximum%20Likelihood/notebook.ipynb) [![Slides](https://img.shields.io/badge/view-slides%20%28PDF%29-8A2BE2)](slides.pdf) [![Research brief](https://img.shields.io/badge/read-brief-blue)](research-brief.md) [![Transcript](https://img.shields.io/badge/read-transcript-lightgrey)](transcript/Episode%2008%20-%20Maximum%20Likelihood%20-%20transcript.txt) [![Short](https://img.shields.io/badge/watch-the%20Short-red?logo=youtube&logoColor=white)](https://youtu.be/nXAHA0oHIo4)

<a href="https://youtu.be/u1bSbpKfF6g"><img src="thumbnail.png" width="560" alt="Watch Episode 8: Maximum Likelihood, by Hand"></a>

**Length:** 10:36 · **Part II:** How Machines Learn · **Math:** Lik(θ) = Π p_θ(xₜ | x<ₜ), −(1/N) ln Lik(θ) = L(θ), −ln p(y) = ½ ln(2πτ²) + (y − ŷ)²/(2τ²) · **Code:** NumPy and matplotlib, runs anywhere

Take two probabilities from Episode 7, 0.30 and 0.60. Multiply them: 0.18. Take minus the log: 1.715. Divide by two: 0.857. That is exactly the loss Episode 7 found by averaging surprisals, but by a completely different road. It's not a coincidence: it's the reason the loss exists. Episode 7 met two losses, squared error for numbers and cross-entropy for words, and they looked like two separate choices. This episode, part 2 of What Learning Means, shows they are one idea wearing two outfits: maximum likelihood. Every step is worked by hand: the likelihood from the chain rule, why a computer can't just multiply (0.5¹⁰⁷⁵ rounds to zero), the four moves from likelihood to loss, the bell curve that hides the square in squared error (0.919, 1.419, and 3.257 versus 2.840 for Episode 7's two lines), and the fine print behind it all.

## What you'll learn

- What the likelihood is: the probability the model gives to the data that actually happened, with the data fixed and the knobs θ turned
- Why the likelihood of a text is a product of the probabilities we've been using, exactly, by Episode 4's chain rule
- Why a computer can't just multiply the probabilities, and how the log fixes it by turning the product into a sum
- The four moves from likelihood to loss (log, the arg max, divide by N, flip the sign) and why maximizing the likelihood is minimizing the loss
- Where the square in squared error comes from: Gaussian noise with a fixed spread τ
- Why one principle gives both losses: mean squared error for numbers, cross-entropy for words
- The four assumptions behind it, why dividing by N matters for the gradient, and why "a low training loss means a good model" is a myth

## Key ideas

- The likelihood is the probability the model gives to the data that actually happened. The data stay fixed; training turns the knobs θ.
- For text, the chain rule makes the likelihood exactly a product of each word's probability given the words before it. No independence between the words is needed.
- Products of probabilities shrink fast: 0.5¹⁰⁰⁰ ≈ 10⁻³⁰¹, and 0.5¹⁰⁷⁵ rounds to zero on an ordinary computer. The log turns the product into a sum.
- The log only goes up, so it doesn't change the arg max. Dividing by N is rescaling. Flipping the sign turns the biggest into the smallest. So −(1/N) ln Lik(θ) is the mean surprisal, and maximizing the likelihood is minimizing the loss.
- Goodfellow et al. name the cost "the negative log-likelihood, equivalently described as the cross-entropy". Cross-entropy wasn't an arbitrary pick.
- If y = ŷ + Gaussian noise with a fixed spread τ, then −ln p(y) = ½ ln(2πτ²) + (y − ŷ)²/(2τ²): a constant plus the squared error. With τ fixed, the highest likelihood is the least-squares line.
- One principle, two losses. Only the assumed distribution changes: Gaussian for a real number, categorical (the softmax output) for a word.
- Dividing by N doesn't change the winner, but it sets the size of the loss and its gradient, which is why the step size η is set against the mean.

## Worked by hand

**Two roads, one number** (Episode 7's two observed words, probabilities illustrative).

| Road | Work | Result |
|---|---|---|
| 1: multiply | 0.30 × 0.60 = 0.18; −ln 0.18 = 1.715; ÷ 2 | **0.857** |
| 2: average surprisals | (1.204 + 0.511) / 2 | **0.857** |

**The likelihood, two models.**

| Model | Probabilities of the two words | Likelihood | Loss (Episode 7) |
|:--:|:--:|:--:|:--:|
| 1 | 0.30, 0.60 | 0.30 × 0.60 = **0.18** | 0.857 |
| 2 | 0.40, 0.50 | 0.40 × 0.50 = **0.20** | **0.805** |

Higher likelihood, lower loss.

**From likelihood to loss.**

| Step | Work | Result |
|---|---|---|
| The problem | 0.5¹⁰⁰⁰ · 0.5¹⁰⁷⁵ in float64 | **≈ 9.33 × 10⁻³⁰²** (about 10⁻³⁰¹) · **0** |
| Move 1: log, product → sum | ln(0.3 × 0.6) = ln 0.3 + ln 0.6 = −1.204 + (−0.511) | **−1.715** |
| Move 2: the arg max doesn't change | 0.20 > 0.18, and ln 0.20 = −1.609 > ln 0.18 = −1.715 | same winner |
| Move 3: divide by N = 2 | −1.715 / 2 | **−0.857** |
| Move 4: flip the sign | −(−0.857) | **0.857 = the loss** |

**The Gaussian case, τ = 1** (Episode 7's dots (1, 2), (2, 4), (3, 5)).

| Step | Work | Result |
|---|---|---|
| The constant | ½ ln 2π: 2π ≈ 6.283, ln 6.283 ≈ 1.838, ½ × 1.838 | **0.919** |
| Line A's miss of 1 | 0.919 + 1²/2 = 0.919 + 0.5 | **1.419** |
| A perfect hit | the constant only | **0.919** |
| Three constants | 3 × 0.919 | 2.757 |
| Line A (W = 2, β = 0), squared errors sum to 1 | 2.757 + ½ × 1 | **3.257** |
| Line B (W = 1.5, β = 2/3), squared errors sum to 1/6 | 2.757 + ½ × 1/6 = 2.757 + 0.083 | **2.840** |

The constant is the same for both lines; only half the sum of squared errors differs. So with a fixed τ, the line with the highest likelihood is exactly the least-squares line, line B.

**Two losses, one principle.**

| Output | Assumed distribution | Maximum likelihood gives |
|---|---|---|
| A real number | Gaussian noise, fixed τ | Mean squared error |
| A word from the vocabulary | Categorical (the softmax output) | Cross-entropy |

**Sum or mean.** The two-word sum, 1.715, has a gradient twice as big as the mean, 0.857 (N = 2). With millions of words, the sum's gradient would be millions of times bigger.

## Key terms

| Term | Meaning |
|---|---|
| Likelihood, Lik(θ) | The probability the model gives to the data that actually happened, as a function of the knobs θ |
| Maximum likelihood | Choosing the θ that makes the real data most probable |
| Chain rule | p(sequence) = the product of each word's probability given the words before it (Episode 4); exact |
| Log likelihood | ln Lik(θ): the sum of the log probabilities |
| Negative log likelihood (NLL) | −ln Lik(θ); divided by N, it is the loss |
| Underflow | A product so small the computer rounds it to zero (0.5¹⁰⁷⁵ in float64) |
| Arg max | The θ that gives the largest value; the log doesn't change it |
| Gaussian (bell curve) | The noise distribution behind squared error |
| Spread, τ (tau) | How wide the bell curve is. Goodfellow et al. use another letter; this series saves σ for the sigmoid |
| Categorical distribution | A probability for each word in the vocabulary: the softmax output |
| Generalization | Doing well on data the model has never seen (Episode 12) |

## Run the notebook

The [companion notebook](notebook.ipynb) works every calculation the way the video does, prints each line of the worksheet, plots the likelihoods, the underflow, the bell curve and the grid search, and checks every value with an `assert`. It runs in seconds.

| Part | What it shows | Matches the video |
|:--:|---|---|
| 1 | Two roads to one number; the chain rule checked on a made-up joint distribution; two models side by side (plot) | 0.18 → 1.715 → 0.857 · 0.20 · 0.805 |
| 2 | Underflow in float64 (plot); log turns product into sum; the arg max unchanged; divide by N and flip the sign | 9.33 × 10⁻³⁰² · 0 · −1.715 · −1.609 · −0.857 → 0.857 |
| 3 | The Gaussian density and its negative log likelihood with spread τ, checked numerically; τ = 1 by hand; lines A and B (plot); a grid search over W and β where the lowest MSE and the highest likelihood land on the same line (plot); the two-losses table | 0.919 · 1.419 · 3.257 · 2.840 · W 1.5, β 2/3 |
| 4 | The four assumptions; the myth; the gradient of the sum versus the mean, factor N | 1.715 vs 0.857: × 2 |
| 5 | The check question | 0.1 · 0.768 · 2.919 |
| 6 | Try it: three probabilities two ways, the grid search with your own τ, and line C's Gaussian negative log likelihood, each with a self-check | your numbers |

Or run it locally:

```bash
git clone https://github.com/Kinvecta/how-llms-work.git
cd how-llms-work
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python tools/check_notebooks.py "Episode 08 - Maximum Likelihood/notebook.ipynb"
# PASS  Episode 08 - Maximum Likelihood/notebook.ipynb
```

## Try it

About 15 minutes.

1. Multiply three probabilities, then add their logs, and check you get the same answer both ways (`my_probs`, `my_product`, `my_log_sum`).
2. In the notebook, run the grid search and watch the lowest loss and the highest likelihood land on the same line. Change `my_tau` and check the winner doesn't move.
3. Work out the Gaussian negative log likelihood (τ = 1) for line C from Episode 7: W = 1.5, β = 1/2. Put your answer in `my_nll_C`; the notebook keeps the answer hidden until you do.

## Check your understanding

1. What does the likelihood measure, and what stays fixed while training turns the knobs?
2. Why is the likelihood of a text a product of the word probabilities, and does that need the words to be independent?
3. Model 1 gives two observed words 0.30 and 0.60; model 2 gives them 0.40 and 0.50. Which has the higher likelihood, and which has the lower loss?
4. Why does a computer work with the log likelihood, and why doesn't taking the log change which θ wins?
5. A batch has three words, with probabilities 0.5, 0.25 and 0.8 on the observed words. What is the likelihood, and what is the mean loss L?
6. With τ = 1, what does a prediction that misses by two pay in negative log likelihood?
7. Where does the square in squared error come from, and what must stay fixed for least squares to be maximum likelihood?
8. True or false: a low training loss means a good model.

<details><summary>Answers</summary>

1. The probability the model gives to the data that actually happened. The data stay fixed; training turns θ and watches that probability change.
2. Episode 4's chain rule: the probability of a sequence is exactly the product of each word's probability given the words before it. No: each factor already conditions on the words before it, so no independence between the words is needed (only between training sequences).
3. Model 2: 0.40 × 0.50 = 0.20 beats 0.30 × 0.60 = 0.18, and its loss, 0.805, is below model 1's 0.857. Higher likelihood, lower loss.
4. Products of many probabilities underflow: 0.5¹⁰⁷⁵ rounds to zero in float64. The log turns the product into a sum. The log only goes up, so the biggest likelihood still has the biggest log likelihood: ln 0.20 = −1.609 beats ln 0.18 = −1.715.
5. 0.5 × 0.25 = 0.125, × 0.8 = 0.1. −ln 0.1 = 2.303, divided by three terms: L = 0.768 nats.
6. The constant ½ ln 2π = 0.919, plus 2²/2 = 2: 2.919.
7. From assuming y = ŷ + Gaussian noise: minus the log of the bell curve's density is a constant, ½ ln(2πτ²), plus the squared error over 2τ². The spread τ must stay fixed, so the constant and the factor 1/(2τ²) don't depend on the knobs.
8. False. The loss is averaged over the training data. A model can memorize those and still stumble on text it has never seen; whether it generalizes is a separate question (Episode 12).

</details>

## Further reading

- The [research brief](research-brief.md) for this episode, with every source
- Goodfellow, Bengio and Courville, *Deep Learning* (MIT Press, 2016), §5.5 (maximum likelihood, the log and 1/N), §5.5.1 (conditional log-likelihood and mean squared error, eqs. 5.63–5.66), §6.2.1.1 (cross-entropy as negative log-likelihood, eqs. 6.12–6.13) and §6.2.2.1 (linear units for Gaussian output, eq. 6.17)
- Bishop, *Pattern Recognition and Machine Learning* (Springer, 2006), §1.2.3 (the negative log likelihood as "an error function") and §4.3.4 (multiclass logistic regression)
- Jurafsky and Martin, *Speech and Language Processing*, 3rd ed. draft, chapter 7, §7.7 (training with the average cross-entropy loss)

---

**Previous:** [Episode 7, The Loss, by Hand](https://youtu.be/ia7aFhn_20k) · [episode page](../Episode%2007%20-%20The%20Loss%2C%20by%20Hand/)

**Next:** [Episode 9, Gradient Descent, Part 1: The Learning Rate, by Hand](../Episode%2009%20-%20Gradient%20Descent%2C%20Part%201/): we have a loss and a gradient; now the model takes thousands of steps, and we meet the learning rate, the one knob that decides whether it settles or blows up ([watch](https://youtu.be/9H0EDfR4wFo)).

**Note:** generalization, called "episode ten" in this video, is now [Episode 12](../course/README.md#episodes): gradient descent became three episodes, 9, 10 and 11.

[← Back to the course](../README.md)
