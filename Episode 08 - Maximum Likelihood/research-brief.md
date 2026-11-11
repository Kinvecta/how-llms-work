# Episode 8: What Learning Means, Part 2: Maximum Likelihood, by Hand

**Series:** How Large Language Models Work: From Math to Transformers
**Runtime:** 10:36
**Companion:** Colab notebook "Episode 08 - Maximum Likelihood" (two roads to 0.857; the likelihood from the chain rule and a second model; underflow; log, arg max, divide by N and flip; the Gaussian negative log likelihood with spread τ, lines A and B and a grid search; the gradient of the sum versus the mean; the check question; every number asserted).

> **Note.** This is part 2 of a two-part topic. The loss itself (learning as adjusting θ, training and inference, mean squared error for lines A, B and C, cross-entropy for a word as the surprisal of the observed token, the mean over N terms, perplexity, the bet analogy and the myth "the loss counts wrong answers") is in Episode 7, What Learning Means, Part 1: The Loss, by Hand. This brief keeps the maximum-likelihood half of the original research file: the Math Lab (likelihood, log, mean, the Gaussian case and its assumptions), the table of the two losses ("why two losses") and the second myth ("a low training loss means a good model"). Episode numbers are updated to the final order: generalization is Episode 12; gradient descent is three episodes, the learning rate in Episode 9, the mean versus the sum and mini-batches in Episode 10, and where it gets stuck in Episode 11.

---

## Learning objectives

By the end of this episode, viewers should be able to:

1. State the likelihood of a text as a product of conditional probabilities (the chain rule), with the data fixed and θ varied.
2. Turn the likelihood into the mean loss (log, divide by N, flip the sign) and explain why maximizing one is minimizing the other.
3. Derive the Gaussian negative log likelihood and name the noise assumption behind mean squared error.
4. State the assumptions behind the equivalence, and why dividing by N matters for the gradient.

## Running example

> "The cat sat on the mat because it was tired."

Episode 7 gave "mat" after "The cat sat on the" probability 0.30 (surprisal 1.204 nats) and a second observed token probability 0.60 (0.511 nats); their mean, 0.857 nats, is the loss [Ep 7]. The probabilities are illustrative. This episode multiplies the same two probabilities instead of averaging their surprisals, and arrives at the same number.

**Term:** −ln p is **surprisal** (Episode 4). "Score" stays reserved for the vector h.

## Notation introduced

| Symbol | Meaning |
|---|---|
| Lik(θ) | The likelihood: the probability the model gives to the training data, as a function of θ |
| τ | Standard deviation (spread) of fixed Gaussian noise. Goodfellow writes σ; this series reserves σ for the sigmoid |
| η | Step size (learning rate). Named here; used in Episode 9 |

Reused: θ and p_θ (Ep 1); p, q, H(q, p), ln, chain rule (Ep 4); N, L(θ), y, ŷ, u, W, β (Ep 7). Goodfellow's m is N here.

## Section 1: Cold open

Multiply the two probabilities: 0.30 × 0.60 = 0.18. Take minus the log: −ln 0.18 = 1.715. Divide by two: 0.857. That is Episode 7's loss, found there by averaging surprisals, (1.204 + 0.511) / 2 = 0.857 [derived; asserted in the notebook].

**Talking point.** Two roads, one number. Not a coincidence: it is the reason the loss exists.

## Section 2 (Math Lab): the likelihood

**Step 1, the likelihood.** Suppose the training data are M sequences, drawn independently. Within a sequence, the chain rule of probability [S31 §3.6; Ep 4] gives the sequence's probability as an exact product of conditionals, with no independence assumed among its tokens. So the likelihood of the N terms is

Lik(θ) = Π_t p_θ(xₜ | x<ₜ)

The data stay fixed; θ varies. A good model gives the real data a lot of probability.

**Two models (illustrative).** Model 1: 0.30 × 0.60 = 0.18, loss 0.857. Model 2 (Episode 7's check question): 0.40 × 0.50 = 0.20 > 0.18, loss 0.805 < 0.857. Higher likelihood, lower loss [derived].

**Step 2, the log.** Products of probabilities underflow: 0.5¹⁰⁰⁰ ≈ 9.33 × 10⁻³⁰² (about 10⁻³⁰¹), and 0.5¹⁰⁷⁵ rounds to 0 in IEEE-754 double precision (float64), whose smallest positive value is 2⁻¹⁰⁷⁴ [derived; asserted in the notebook]. The logarithm turns the product into a sum,

ln Lik(θ) = Σₜ ln p_θ(xₜ | x<ₜ), e.g. ln(0.3 × 0.6) = −1.204 + (−0.511) = −1.715,

and does not change the maximizer [S47, §5.5]: taking the logarithm "does not change its arg max". The log only goes up: ln 0.20 = −1.609 > ln 0.18 = −1.715.

**Step 3, the mean.** Dividing by N does not change the maximizer [S47, §5.5, p. 129]: −1.715 / 2 = −0.857. Flip the sign: since (1/N) ln Lik(θ) = −L(θ),

argmax_θ Lik(θ) = argmin_θ L(θ)

Maximum likelihood is minimizing the mean surprisal, which is the mean cross-entropy against one-hot targets. Goodfellow states that the cost is "the negative log-likelihood, equivalently described as the cross-entropy" [S48, §6.2.1.1, eq. 6.12].

## Section 3 (Math Lab): the Gaussian case

**Step 4, the Gaussian case.** If y = ŷ + noise, with fixed-τ Gaussian noise, the density of y is p(y | u; θ) = (2πτ²)^(−1/2) exp(−(y − ŷ)² / (2τ²)). Then

−ln p = ½ ln(2πτ²) + (y − ŷ)² / (2τ²)

For fixed τ, minimizing the summed negative log-likelihood over θ is minimizing the squared errors [S47, eqs. 5.63–5.66; S48, eq. 6.13].

**By hand, τ = 1 (Episode 7's dots and lines).** The constant ½ ln 2π: 2π ≈ 6.283, ln 6.283 ≈ 1.838, half is 0.919. Line A's miss of 1 pays 0.919 + 1²/2 = 1.419; a perfect hit pays 0.919. Over three points, the constant part is 3 × 0.919 = 2.757. Line A: 2.757 + ½ × 1 = 3.257. Line B: 2.757 + ½ × 1/6 = 2.757 + 0.083 = 2.840. The constant is the same for both lines; only half the sum of squared errors differs, so with fixed τ the highest likelihood is the least-squares line [derived; a grid of 150,801 settings of W and β in the notebook puts the lowest MSE and the highest likelihood at the same point, W = 1.5, β ≈ 2/3].

## Section 4: Why two losses

| Output | Assumed distribution | Loss | Sources |
|---|---|---|---|
| A real number | Gaussian, fixed τ | Mean squared error | S47 §5.5.1; S48 §6.2.1.1 |
| One word from the vocabulary | Categorical (softmax over the vocabulary) | Cross-entropy | S48 §6.2.1.1; S32 §6.2.2.3 |

Goodfellow notes that a linear output layer "is often used to produce the mean of a conditional Gaussian distribution" [S48, §6.2.2.1, eq. 6.17]. Bishop calls the negative log of the likelihood "an error function" [S49, §1.2.3, p. 23] and treats multiclass logistic regression in Chapter 4 [S49; S50]. One principle gives both losses. Only the assumed output distribution changes.

## Section 5 (Math Lab): the assumptions and the scale

**Step 5, the assumptions.**
1. **Independence.** Sequences (or examples) are drawn independently: "drawn independently" [S47, §5.5, p. 128], later i.i.d. [S47, §5.5.1, p. 130]. Tokens within a sequence need no such assumption (Step 1).
2. **Fixed model family.** θ ranges over p_θ only. If the data lie outside it, the best θ is the closest member in cross-entropy, not the truth [derived from H(q, p) = H(q) + D_KL(q‖p), S31].
3. **Training sample.** The fit uses the training data. Generalization is Episode 12.
4. **Fixed noise for MSE.** If τ is also learned, the term N ln τ no longer drops out of the negative log-likelihood, so the loss is no longer plain squared error [derived].

**Step 6, the scale.** Dividing by N rescales the loss. It does not change the minimizer, but it changes the size of the gradient [S47, §5.5, p. 129]. The two-word sum, 1.715, has a gradient twice as big as the mean, 0.857 (N = 2); with millions of words, millions of times bigger. So the step size η is set against the mean: Episode 9's learning rate, and Episode 10's mean versus sum [derived; asserted in the notebook].

## Section 6: Myth check: "a low training loss means a good model"

**For the idea.** Low loss means high likelihood [Step 3].

**Against it.** The loss is averaged over the training data [S47, §5.5]. A model can memorize those and still stumble on text it has never seen. Whether the model generalizes is a separate question, for Episode 12.

## Section 7: Recap and check question

**Recap.** The likelihood is the probability the model gives to the data that actually happened. Take its log, divide by N, flip the sign, and you get the mean loss, so maximizing the likelihood is minimizing the loss. For words, that loss is cross-entropy. For numbers with fixed Gaussian noise, it is mean squared error. It rests on independent samples, a fixed model family, the training data and a fixed τ.

**Check question.** A batch has three words, and the model gives the observed words probabilities 0.5, 0.25 and 0.8. (1) What is the likelihood? (2) What is the mean loss L? (3) With τ = 1, what does a prediction that misses by two pay in negative log likelihood?

*Answer:* (1) 0.5 × 0.25 = 0.125, × 0.8 = 0.1. (2) −ln 0.1 = 2.303, ÷ 3 = 0.768 nats. (3) 0.919 + 2²/2 = 0.919 + 2 = 2.919.

**Try it.** Multiply three probabilities and add their logs; run the grid search in the notebook; work out the Gaussian negative log likelihood for line C (W = 1.5, β = 1/2). The line C answer is a self-check in the notebook and is not shown on the try slide.

**After this, you can explain…**
- What the likelihood is, and why the chain rule makes it a product of the probabilities a language model already gives.
- Why maximizing the likelihood is minimizing the mean loss: log, divide by N, flip the sign.
- Why mean squared error is maximum likelihood for a real output with fixed-τ Gaussian noise, and cross-entropy for a word.
- The assumptions behind that equivalence, and why the mean, not the sum, sets the gradient's scale.

## Derived numbers (all asserted in the notebook)

| Quantity | Work | Value |
|---|---|---|
| Two roads | 0.30 × 0.60; −ln 0.18; ÷ 2 · (1.204 + 0.511) / 2 | 0.18; 1.715; 0.857 · 0.857 |
| Model 2 | 0.40 × 0.50; loss (Ep 7) | 0.20; 0.805 |
| Underflow | 0.5¹⁰⁰⁰ · 0.5¹⁰⁷⁵ (float64) | 9.33 × 10⁻³⁰² · 0 |
| Log, product → sum | ln 0.3 + ln 0.6 | −1.204 + (−0.511) = −1.715 |
| Log is monotonic | ln 0.20 vs ln 0.18 | −1.609 > −1.715 |
| Divide by N, flip | −1.715 / 2; × −1 | −0.857; 0.857 |
| Gaussian constant, τ = 1 | 2π; ln 2π; ½ ln 2π | 6.283; 1.838; 0.919 |
| Line A's miss of 1 | 0.919 + 1²/2 | 1.419 |
| Three constants | 3 × 0.919 | 2.757 |
| Line A total | 2.757 + ½ × 1 | 3.257 |
| Line B total | 2.757 + ½ × 1/6 | 2.840 |
| Line C total (try-it self-check, not on screen) | 2.757 + ½ × 1/4 | 2.882 |
| Gradient of sum vs mean | N = 2 | × 2 |
| Check question | 0.5 × 0.25 × 0.8; −ln 0.1 / 3; 0.919 + 2²/2 | 0.1; 0.768; 2.919 |

## Sources

Status key: **Read** = checked. **Summary** = secondary only; check the primary before quoting.

| ID | Citation and link | Used for | Status |
|---|---|---|---|
| S31 | Goodfellow, Bengio & Courville, *Deep Learning* (MIT Press, 2016), Ch 3 §§3.6, 3.13. https://www.deeplearningbook.org/contents/prob.html | Chain rule; cross-entropy and KL (from Ep 4) | Read |
| S32 | Goodfellow et al., Ch 6 §6.2.2.3. https://www.deeplearningbook.org/contents/mlp.html | Softmax (from Ep 4) | Read |
| S47 | Goodfellow et al., Ch 5 §5.5 (pp. 128–129) and §5.5.1 (pp. 130–131), eqs. 5.56–5.66. https://www.deeplearningbook.org/contents/ml.html | Maximum likelihood; log and 1/N; independence; Gaussian and MSE | Read (web text; §5.5 start agrees with S57) |
| S48 | Goodfellow et al., Ch 6 §6.2.1 "Cost Functions," §6.2.1.1 (pp. 173–174, eqs. 6.12–6.13), §6.2.2.1 (p. 177, eq. 6.17); same page as S32 | Cross-entropy as negative log-likelihood; MSE from a Gaussian; linear Gaussian output | Read (web text; §6.2 starts p. 171 per S57) |
| S49 | Bishop, *Pattern Recognition and Machine Learning* (Springer, 2006): Ch 4 (p. 179), §4.3.4 (p. 209), §1.2.3 (p. 23), §1.1 (p. 9). https://noiselab.ucsd.edu/ECE228_2019/Bishop.pdf | Titles; "error function"; least squares as maximum likelihood | Read (contents and excerpt only) |
| S50 | Srihari, slides "Multiclass Logistic Regression." https://cedar.buffalo.edu/~srihari/CSE574/Chap4/4.3.4-MultiLogistic.pdf | Multiclass cross-entropy, 1-of-K targets | Summary (slides; formula text garbled) |
| S53 | Jurafsky & Martin, *Speech and Language Processing*, 3rd ed. draft (Aug 19, 2026), Ch 7 §7.7, eqs. 7.54–7.58. https://web.stanford.edu/~jurafsky/slp3/7.pdf | Average cross-entropy over the batch (the loss being maximized as a likelihood) | Read (draft; pages inferred from headers) |
| S57 | Barnes & Noble, table of contents for *Deep Learning*. https://valsec.barnesandnoble.com/w/deep-learning-ian-goodfellow/1126352242 | Section start pages (§5.5 p. 128; §6.2 p. 171) | Summary (retailer TOC) |

The sources for mean squared error and cross-entropy as losses, training and inference, and the 3Blue1Brown cost-function lessons (S51, S52, S56) are listed in the Episode 7 brief.

## Open items

- Print check of Goodfellow pp. 128–131 and 173–177 (section starts already agree with S57).
- Bishop eqs. 4.90 and 4.108, and §3.1.1, on a full copy.
- Assumption 4 wording (resolved). The research draft said that if τ is learned too, "the least-squares answer is no longer the winner". With one shared τ learned jointly, the negative log likelihood is N ln τ + Σ(y − ŷ)²/(2τ²) + constant, so for every τ the best W and β are still the least-squares ones (and τ² comes out as the MSE); what changes is that the loss is no longer plain squared error. The narration was re-voiced to say exactly that ("Learn tau too, and the loss is no longer plain squared error").
- Counting convention for N: whether the first token of a sequence is a loss term. Set in Episode 9.
