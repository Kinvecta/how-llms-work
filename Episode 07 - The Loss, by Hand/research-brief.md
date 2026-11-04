# Episode 7: What Learning Means, Part 1: The Loss, by Hand

**Series:** How Large Language Models Work: From Math to Transformers
**Runtime:** 11:21
**Companion:** Colab notebook "Episode 07 - The Loss, by Hand" (MSE for lines A, B and C with a least-squares and grid check; cross-entropy, the mean loss and perplexity; the bet payouts, the myth and the check question; every number asserted).

> **Note.** This is part 1 of a two-part topic. The maximum-likelihood Math Lab (likelihood, log, mean, the Gaussian case and its assumptions), the table of the two losses ("why two losses") and the second myth ("a low training loss means a good model") are in Episode 8, What Learning Means, Part 2: Maximum Likelihood.

---

## Learning objectives

By the end of this episode, viewers should be able to:

1. Explain a loss as the number training minimizes, and what the mean over N terms measures.
2. Compute mean squared error for a real-valued output.
3. Compute cross-entropy for a chosen word as −ln p of the observed token, averaged over N tokens.

## Running example

> "The cat sat on the mat because it was tired."

Episode 5 gave the surprisal of "mat" after "The cat sat on the" as −ln 0.30 = 1.204 nats [Ep 4; S31], using the illustrative Episode 1 probabilities. This episode asks what that number is for. It is one term. Training averages many terms into one loss.

**Term:** −ln p is **surprisal**, the name from Episode 4. "Score" stays reserved for the vector h.

## Notation introduced

| Symbol | Meaning |
|---|---|
| N | Number of loss terms. The mean divides by N. One term per predicted token, or per example |
| L(θ) | The loss: the mean of the per-term losses, as a function of the parameters θ |
| t, i | Index of a loss term (t) and of an example (i, MSE example only) |
| y | Label: the target value for one regression example. MSE example only |
| ŷ | Prediction for one example: ŷ = Wu + β. MSE example only |
| u, W, β | Input, weight and bias of the Ep 3 linear layer. In the MSE example u is one number and W is 1 × 1 |

Reused: θ and p_θ (Ep 1); p, q, H(q, p), ln (Ep 4). Goodfellow's m is N here. (The noise symbol τ arrives with the Gaussian case in Episode 8.)

## Section 1: Cold open

Show two fits to the same scattered points, a good line and a poor one, with the gaps drawn in. Ask: how would a machine know which is better, and by how much?

**Talking point.** Training needs one number to push down. That number is the loss.

## Section 2: Learning means adjusting parameters

A model is a function with adjustable numbers θ. In Episode 3's linear layer, ŷ = Wu + β, and W and β are the adjustable numbers. The draft Jurafsky & Martin chapter says the weights are "adjusted to minimize this average cross-entropy loss over the batch via gradient descent" [S53, §7.7, p. 26].

Inference is running the trained model: "at inference time, we iteratively generate the next tokens one at a time" [S53, §7.8, p. 28]. The draft does not say outright that parameters are frozen then. We state it as our definition: training changes θ, and inference runs the model with θ fixed [S53; derived].

## Section 3: Numbers: mean squared error

Take pairs (u, y). The model predicts ŷ, and the loss averages the squared gaps over N examples:

L(θ) = (1/N) Σᵢ (yᵢ − ŷᵢ)², with ŷᵢ = Wuᵢ + β and θ = (W, β)

Goodfellow defines the same mean squared error on a conditional Gaussian model, with m for N [S47, eq. 5.66]. The 3Blue1Brown lesson on cost functions frames it the same way: the cost of one example, then the average over "the tens of thousands of training examples" [S51]. Chapter 1 has no cost function [S52], so Chapter 2 is the lesson to cite.

**Toy example (illustrative).** u = (1, 2, 3), y = (2, 4, 5).
- Line A: W = 2, β = 0. Predictions (2, 4, 6). Errors (0, 0, −1). MSE = 1/3.
- Line B: W = 1.5, β = 2/3, the least-squares fit. Predictions (13/6, 11/3, 31/6). MSE = 1/18 ≈ 0.056.
- Line C: W = 1.5, β = 1/2. Predictions (2, 3.5, 5). Errors (0, 1/2, 0). MSE = 1/12 ≈ 0.083, still worse than line B [derived; asserted in the notebook].

Training means finding W and β that make L small [derived; checked numerically].

**Analogy.** Darts: the miss distance from the bullseye, squared and averaged over the throws, is the mean squared error. Line A hits the bullseye twice and then misses by a lot; line B lands close every time. Squaring makes the big miss expensive.

## Section 4: Words: cross-entropy and surprisal

For one word, the model's distribution is p = p_θ(· | x<ₜ). The target q puts all its weight on the observed token. The cross-entropy is H(q, p) = −Σₖ qₖ ln pₖ [S31, eq. 3.51; Ep 4]. With a one-hot q, only one term survives:

H(q, p) = −ln p(observed token)

So the per-token cross-entropy is the surprisal of the observed token. The draft writes the same term as the negative log probability the model assigns to the next word, and averages it over the batch [S53, eqs. 7.54–7.55, pp. 25–26]. Averaged over N terms:

L(θ) = (1/N) Σₜ −ln p_θ(xₜ | x<ₜ)

Episode 4's perplexity is the exponential of this mean [S53, §7.7.1; the mean is eq. 7.58; Ep 4].

**Running-sentence beat (illustrative, N = 2).** After "The cat sat on the," the model gives "mat" probability 0.30: surprisal 1.204 nats. A second observed token gets probability 0.60: surprisal 0.511 nats. The loss is their mean, (1.204 + 0.511) / 2 = 0.857 nats. The 1.204 is one term. The loss is the average. Its perplexity is e^0.857 ≈ 2.36 [derived; asserted in the notebook].

**Analogy.** A bet: spread one stack of chips (probability, not money) over four words; the true word is revealed; you pay −ln of your share on it. Shares 0.9, 0.5, 0.1 and 0.01 pay 0.105, 0.693, 2.303 and 4.605 [derived; asserted in the notebook]. The less you backed the truth, the faster the bill grows.

## Section 5: Myth check: "the loss counts wrong answers"

**For the idea.** A perfect fit has zero squared error, and a model that puts all its probability on the observed token has zero cross-entropy [derived].

**Against it.** Cross-entropy penalizes any probability below 1 on the observed token, even when that token is the model's top choice. If the observed token is the top choice with probability 0.45, its surprisal is −ln 0.45 = 0.799 nats [derived]. The model is right and still pays a penalty.

*Line for narration:* "The loss is a penalty that training pushes down. It is not a count of mistakes."

## Section 6: Recap and check question

**Recap.** Training minimizes one number, the mean loss over N terms. A real output gives mean squared error: on the three dots, 1/3 for line A and 1/18 for line B. A word gives cross-entropy, which for one observed token is its surprisal: 1.204 and 0.511, mean 0.857. A model can be right and still pay.

**Check question.** A batch has N = 2 terms. The model gives the observed token probability 0.25 at the first term and 0.50 at the second. (a) What is each term's surprisal? (b) What is L? (c) Does raising the first probability to 0.40 lower L?

*Answer:* (a) 1.386 and 0.693 nats. (b) L = 1.040 nats. (c) Yes. −ln 0.40 = 0.916, so L = (0.916 + 0.693) / 2 = 0.805 nats.

**After this, you can explain…**
- What a loss is, and why training minimizes an average over N terms.
- How to compute mean squared error by hand, and why squaring makes a big miss expensive.
- Why cross-entropy for a word is the surprisal of the observed token, and why the loss is the mean of those terms.
- What training and inference each do to θ.

## Sources

Status key: **Read** = checked. **Summary** = secondary only; check the primary before quoting.

| ID | Citation and link | Used for | Status |
|---|---|---|---|
| S31 | Goodfellow, Bengio & Courville, *Deep Learning* (MIT Press, 2016), Ch 3 §§3.6, 3.13. https://www.deeplearningbook.org/contents/prob.html | Chain rule; cross-entropy (from Ep 4) | Read |
| S47 | Goodfellow et al., Ch 5 §5.5.1 (pp. 130–131), eq. 5.66. https://www.deeplearningbook.org/contents/ml.html | Mean squared error | Read (web text) |
| S51 | 3Blue1Brown, "Gradient descent, how neural networks learn," Chapter 2 (Oct 16, 2017). https://www.3blue1brown.com/lessons/gradient-descent | Cost of one example; average over examples | Read |
| S52 | 3Blue1Brown, "But what is a Neural Network?" Chapter 1 (Oct 5, 2017). https://3blue1brown.com/lessons/neural-networks | Negative check: no cost function | Read |
| S53 | Jurafsky & Martin, *Speech and Language Processing*, 3rd ed. draft (Aug 19, 2026), Ch 7 §§7.6–7.8, eqs. 7.53–7.58. https://web.stanford.edu/~jurafsky/slp3/7.pdf | Training, decoding, inference; average cross-entropy; perplexity | Read (draft; pages inferred from headers) |
| S56 | Manning, "3Blue1Brown: Neural Networks" (course listing). https://www.manning.com/books/3blue1brown-neural-networks | Series title for S51 and S52: "Neural networks" | Summary (retailer listing) |
| S57 | Barnes & Noble, table of contents for *Deep Learning*. https://valsec.barnesandnoble.com/w/deep-learning-ian-goodfellow/1126352242 | Section start pages (§5.5 p. 128) | Summary (retailer TOC) |

The sources for maximum likelihood, the Gaussian case and the two-losses table (Goodfellow §5.5 and §6.2.1, Bishop, Srihari) are listed in the Episode 8 brief.
