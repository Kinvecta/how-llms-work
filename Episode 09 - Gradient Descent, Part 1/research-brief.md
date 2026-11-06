# Episode 9: Gradient Descent, Part 1: The Learning Rate, by Hand

**Series:** How Large Language Models Work: From Math to Transformers
**Runtime:** 16:44
**Companion:** Colab notebook "Episode 09 - Gradient Descent, Part 1" (the update rule and one step on Episode 6's bowl; steps that shrink by themselves; the valley κz² and the factor 1 − 2ηκ; six learning rates, the zones and the step counts; the bowl z₁² + 2z₂², its limit, the best repeated step and the race against 0.278; stretched bowls, including Episode 7's three dots; mat's surprisal; the myth; the check question; every number asserted).

> **Note.** This is part 1 of a two-part topic. The original research file, "Gradient Descent: Walking Downhill", was planned as one video of about 16 minutes and split in two. This brief keeps the first half: the update rule and the learning rate (Section 2), the learning-rate slider on mat's loss (Section 3), Math Lab Steps 1–3 (the one-knob valley, the cases, and the bowl with its stretched relatives) and the first myth ("a smaller learning rate is always safer"). **Episode 10, Gradient Descent, Part 2,** takes Math Lab Steps 4–6 (the 1/N scale of mean versus sum, mini-batch gradients, and pointers to momentum and Adam), Section 4 (mini-batches), Section 5 (what gradient descent does not promise: saddles, flat regions, local minima) and the other two myths, plus the research file's check part (c), the sum of four copies of 3z².
>
> **Numbering.** The research file used the old episode order. Every reference here uses the current one: its "Episode 5" bowl and hiker are **Episode 6** (the gradient); its "Episode 5 nudged z for mat" is **Episode 5** (derivatives); its "Episode 6 toy" is **Episode 7** (the loss, three dots); its "weights behind z" episode is **Episode 15** (backpropagation). Later topics: Episode 10 gradient descent part 2, 11 generalization, 12 training examples, 13 tokens and embeddings, 14 the neuron, 15 backpropagation, 16 training end to end (Adam), 17 representations, 18 counting words.

---

## Learning objectives

By the end of this episode, viewers should be able to:

1. Write the update θ ← θ − η∇_θ L(θ) and say what each part does.
2. Derive the factor 1 − 2ηκ for the valley κz² and use it to predict, before running anything, whether a learning rate slides in, lands in one step, hops, bounces or blows up.
3. Explain why, in a bowl with several directions, the steepest direction sets the limit on η, and why a stretched bowl is slow for every η.
4. Explain why the best single step is not the best step to repeat, and why "a smaller learning rate is always safer" is a myth.

## Running example

> "The cat sat on the mat because it was tired."

Episode 5 nudged the score z for "mat" (the other candidates at total weight R = 0.70), with slope −0.70 for mat's surprisal with respect to its score. This episode turns z with a learning rate, on a loss with **N = 1** term: the surprisal of "mat", −ln 0.30 = 1.204 nats [Ep 4; S31]. Here z is the only knob; in a real model, the weights behind z are what gets adjusted (Episode 15). The probabilities are illustrative.

The main worked examples are exact quadratics: the valley f(z) = κz², and Episode 6's bowl f = z₁² + 2z₂² from (1, 1), height 3.

## Notation introduced

| Symbol | Meaning |
|---|---|
| ∇_θ L(θ) | Gradient of the loss with respect to all the knobs (Episode 6) |
| η | The learning rate (named in Episode 8). Goodfellow writes ε; this series uses η, like Ruder |
| κ | Steepness of f(z) = κz², κ > 0 (not a: Episodes 2 and 5 use a) |
| z⁽ᵏ⁾, z_k | z after k steps; k is a step counter, a dummy index like Episode 3's summed k |
| 1 − 2ηκ | "The factor": what one step multiplies z by |

Reused: θ (Ep 1); ln, surprisal (Ep 4); slopes and the power rule (Ep 5); the gradient and the bowl (Ep 6); L(θ), N, the three dots, u, y, W, β (Ep 7).

## Section 1: Cold open

Three hikers start on one foggy hill and feel the same slope: tiny steps (still near the top at sunset), giant leaps (flung up the far side), steps that are just right (down in five).

**Talking point.** All three walk the same way; only the stride differs, and the stride decides everything. When a language model learns, it walks downhill on its loss, and the stride is the learning rate. Episode 6 took one step on the bowl (3 → 1.36 at η = 0.1; 3 → 19 at η = 1). Training takes thousands.

## Section 2: The update rule and the learning rate

Episode 6 found the direction: against the gradient. Goodfellow writes the step x′ = x − ε∇ₓf(x), where ε is "the learning rate, a positive scalar determining the size of the step" [S37, §4.3, eq. 4.5]. Ruder writes θ = θ − η·∇θJ(θ) [S58, §2.1]. Here:

> θ ← θ − η ∇_θ L(θ)

Read piece by piece: θ is all the knobs; the gradient is the list of slopes and points uphill the fastest; the minus sign turns us downhill; η is how far to go. Repeat until the gradient is zero or very close [S37, §4.3]. The draft Jurafsky & Martin chapter says language-model training runs via gradient descent [S53, §7.7].

**Steps shrink by themselves.** The step is η × slope, so as the ground flattens near the bottom, the steps shrink even with a fixed η [derived; on z² with η = 0.1 the steps are 0.2, 0.16, 0.128, …].

**Why η needs care.** Ruder names both failures: "painfully slow convergence" when too small, and fluctuation "or even to diverge" when too large [S58, §3]. 3Blue1Brown describes overshooting and oscillating [S51].

## Math Lab, Step 1: one knob

f(z) = κz², κ > 0, so f′(z) = 2κz [S42; Ep 5]. Then

z⁽ᵏ⁺¹⁾ = z⁽ᵏ⁾ − η·2κz⁽ᵏ⁾ = (1 − 2ηκ) z⁽ᵏ⁾, so z⁽ᵏ⁾ = (1 − 2ηκ)ᵏ z⁽⁰⁾ [derived].

Since the bottom is at 0, z is the distance from the bottom: a factor of 0.8 keeps 80% of it every step.

**Six learning rates** (κ = 1, z⁽⁰⁾ = 1) [derived; asserted in the notebook]:

| η | Factor | z after 1, 2, 3 steps | |
|---|---|---|---|
| 0.1 | 0.8 | 0.8, 0.64, 0.512 (then 0.4096) | slides in, slowly |
| 0.25 | 0.5 | 0.5, 0.25, 0.125 | halves |
| 0.5 | 0 | 0 | one step: slope at z = 1 is 2, 0.5 × 2 = 1 = the distance |
| 0.75 | −0.5 | −0.5, 0.25, −0.125 | hops sides, shrinks |
| 1 | −1 | −1, 1, −1 | bounces forever |
| 1.25 | −1.5 | −1.5, 2.25, −3.375 | diverges |

## Math Lab, Step 2: the cases

|1 − 2ηκ| < 1 exactly when 0 < η < 1/κ: converges. Between 0 and 1/(2κ), z slides in from one side. η = 1/(2κ) lands on 0 in one step. Between 1/(2κ) and 1/κ, z hops across 0 but shrinks. At η = 1/κ the factor is −1: z bounces between ±z⁽⁰⁾ forever. Above 1/κ it grows: divergence [derived].

**Steps to reach |z| ≤ 0.01** (κ = 1, z⁽⁰⁾ = 1): 228 at η = 0.01, 7 at 0.25, 1 at 0.5. For small η, the count is about ln 100 / (2η) ≈ 2.3/η [derived; within 2% for η ≤ 0.01 in the notebook].

## Math Lab, Step 3: Episode 6's bowl

f = z₁² + 2z₂², ∇f = (2z₁, 4z₂). The rule separates by coordinate: from (1, 1), z₁⁽ᵏ⁾ = (1 − 2η)ᵏ and z₂⁽ᵏ⁾ = (1 − 4η)ᵏ, so f⁽ᵏ⁾ = (1 − 2η)²ᵏ + 2(1 − 4η)²ᵏ. Both factors need size below 1: η < 1 and η < 0.5. The steepest direction sets the limit.

| η | f after 0, 1, 2, 3 steps |
|---|---|
| 0.1 | 3, 1.36, 0.669, 0.355 (factors 0.8 and 0.6: 0.64 + 0.72 = 1.36) |
| 1/3 | 3, 0.333, 0.037, 0.004 (exactly 3/9ᵏ) |
| 0.278 | 3, 0.222, 0.039, 0.008 |
| 0.5 | 3, 2, 2, 2 (z₂ = −1, 1, −1) |
| 0.52 | 3, 2.334, 2.721 |
| 1 | 3, 19, 163, 1459 |

The first steps reproduce Episode 6's 1.36 and 19. Episode 6's 0.556 limit (5/9) and best single step (η ≈ 0.278, that is 5/18, height 0.222) are for one step from (1, 1); repeating the rule is stable only below 0.5, and at η = 0.52 the second step gives back what the first one won [derived].

**The best repeated step.** It makes the two factors equal in size: 1 − 2η = −(1 − 4η), so η = 1/3 and both factors are 1/3. Five steps bring both coordinates within 0.01 ((1/3)⁴ ≈ 0.0123, (1/3)⁵ ≈ 0.0041). The race: 0.278 is ahead after one step (0.222 against 0.333), behind by step 2 (0.039 against 0.037), and about twice as high at step 3 (0.008 against 0.004) [derived].

**Why unequal steepness is slow.** For f = z₁² + κz₂² the best repeated η is 1/(1 + κ) and the best factor is (κ − 1)/(κ + 1). With one direction 50 times steeper (f = z₁² + 50z₂²), the best factor is 49/51 ≈ 0.961 and it takes 116 steps to shrink the distance 100-fold. Episode 7's loss for the three dots (u = (1, 2, 3), y = (2, 4, 5), ŷ = Wu + β, MSE) is such a bowl: its curvature matrix is (2/3)[[14, 6], [6, 3]], so κ = 5.546 and 0.1202 along its two main directions (about 5.55 and 0.120; ratio 46.1), best factor 0.958, and 106.2, so about 107 steps for a 100-fold shrink [derived]. Goodfellow: ill-conditioning "can manifest by causing SGD to get 'stuck'" [S59, §8.2.1].

**Analogy (original).** A skateboarder in a long half-pipe that slopes gently downhill: steep walls, gentle floor. A stride safe for the walls is tiny along the floor.

## Section 3: The slider on the sentence's loss

N = 1. L(z) = −ln p, with p = e^z / (e^z + R), R = 0.70. Start at z = ln 0.30 = −1.204: p = 0.300, L = 1.204. The gradient is dL/dz = −(1 − p) = −0.70 [Ep 5; derived]. One step is z ← z + 0.70η, so after one step p = 1 / (1 + (7/3)·e^(−0.70η)). Later steps repeat the rule (derived, rounded to three decimals, asserted in the notebook):

| η | After 1 step: p, L | After 10 steps: p, L |
|---|---|---|
| 0.1 | 0.315, 1.155 | 0.447, 0.806 |
| 1 | 0.463, 0.770 | 0.894, 0.112 |
| 5 | 0.934, 0.068 | 0.984, 0.016 |

Bigger η is faster in every run checked (η from 0.01 to 1000, after 1 and 10 steps; checked numerically, not proved), and nothing blows up: dL/dz is negative everywhere, so L falls whenever z rises, for every η [derived]. There is no bottom to overshoot, and p only approaches 1, so the loss keeps falling toward zero and never reaches it. The bowl is the hard case. The video shows the one-step p values only.

## Section 4: Myth check: "a smaller learning rate is always safer"

**For.** Any η in (0, 1/κ) is stable; too large can diverge [S58, §3].

**Against.** Steps needed grow like 1/η (228 at 0.01 against 7 at 0.25, Step 2). In flat regions the gradient is already small, so a small η crawls [S59, §8.2.3; derived]. The best bowl η (1/3) is not the smallest (Step 3). Real training runs have a budget: "stall" here means the budget runs out before you arrive, not a theorem.

## Section 5: Recap and check question

**Recap.** Gradient descent repeats θ ← θ − η∇_θ L(θ). In the valley κz², every step multiplies the distance from the bottom by the same factor, 1 − 2ηκ, so η must stay below 1/κ, and exactly half of that lands in one step. In a bowl with several directions, the steepest one sets the limit. When a bowl is stretched, every η is slow.

**Check question.** f(z) = 3z². (1) For which η does gradient descent settle? (2) From z = 1 with η = 0.1, what are z and f after one step? (3) Which η lands exactly at the bottom in a single step?

*Answer:* (1) κ = 3, so 0 < η < 1/3. (2) The factor is 1 − 2(0.1)(3) = 1 − 0.6 = 0.4, so z = 0.4 and f = 3 × 0.16 = 0.48, down from 3. (3) Half of 1/3: η = 1/6 [derived]. (The research file's part (c), the sum of four copies of f, belongs to Episode 10.)

**Try it.** Pick your own κ and find the one-step η; in the notebook, slide η across 1/κ; on the bowl, race five steps at 0.278 against five at 1/3. The one mistake to avoid: judging a learning rate by a single step.

**After this, you can explain…**
- What each piece of the update rule does, and why the steps shrink near the bottom.
- Why the factor 1 − 2ηκ decides everything in a one-knob valley, and why η must stay below 1/κ.
- Why "too big" depends on the steepest direction, and why a stretched bowl is slow.
- Why the best single step is not the best step to repeat.

## Visual plan

| Beat | Visual | Idea source |
|---|---|---|
| Cold open | Three hikers in fog, three strides | Kinvecta adaptation of `x_gradient`, `x_learnrate` |
| Update rule | Labelled rule; footprints, not a rolling ball | Original; Ruder uses the ball for momentum [S58, §4.1] |
| Valley | Walker on κz², numbered gold arrows, factor meter, distance bars | Original |
| Zones | η number line: slides in / one step / hops / bounces / blows up | Original |
| Bowl | Episode 6's bowl, paths for η = 0.1, 1/3, 0.5; the race against 0.278 | Original |
| Stretched | 50 to 1 bowl; Episode 7's loss; the half-pipe skateboarder | Original |
| Mat | Learning-rate slider on the "mat" loss, p readouts | Original |
| Myth check | Steps to converge against η | Original |

## Derived numbers (all asserted in the notebook)

| Quantity | Work | Value |
|---|---|---|
| Episode 6's one step | (1, 1) − η(2, 4), η = 0.1 and 1 | 1.36; 19 |
| Factor, κ = 1 | 1 − 2η for η = 0.1, 0.25, 0.5, 0.75, 1, 1.25 | 0.8, 0.5, 0, −0.5, −1, −1.5 |
| Sequences | factorᵏ | 0.8, 0.64, 0.512 (0.4096) · 0.5, 0.25, 0.125 · 0 · −0.5, 0.25, −0.125 · −1, 1, −1 · −1.5, 2.25, −3.375 |
| One-step landing, κ = 1 | 0.5 × slope 2 | 1 = the distance |
| Steps to 0.01 | κ = 1, z⁽⁰⁾ = 1 | 228 (η 0.01), 7 (0.25), 1 (0.5); ≈ 2.3/η |
| Bowl factors, η = 0.1 | 1 − 2η, 1 − 4η; 0.8² + 2 × 0.6² | 0.8, 0.6; 0.64 + 0.72 = 1.36 |
| Bowl heights | η = 0.1 · 0.5 · 1 · 0.52 | 3, 1.36, 0.669, 0.355 · 3, 2, 2, 2 · 3, 19, 163, 1459 · 3, 2.334, 2.721 |
| Bowl limit | min(1, 0.5) | 0.5 |
| One-step results (Ep 6) | 5/9; 5/18 → height | 0.556; 0.278 → 0.222 |
| Best repeated step | 1 − 2η = −(1 − 4η) | η = 1/3, heights 3/9ᵏ: 0.333, 0.037, 0.004; 5 steps to 0.01 |
| The race, η = 0.278 | heights | 0.222, 0.039, 0.008 |
| Stretched 50 to 1 | 49/51; ln 0.01 / ln(49/51) | 0.961; 116 steps |
| Episode 7's three dots | eigenvalues of (2/3)[[14, 6], [6, 3]], halved | κ 5.546, 0.1202; ratio 46.1; factor 0.958; 106.2 → 107 steps |
| Mat, one step | p from 0.30, η = 0.1, 1, 5 | 0.315, 0.463, 0.934 (L 1.155, 0.770, 0.068) |
| Mat, ten steps | same | 0.447, 0.894, 0.984 (L 0.806, 0.112, 0.016) |
| Check question | κ = 3 | η < 1/3; factor 0.4 → z 0.4, f 0.48; η = 1/6 |

## Sources

Status key: **Read** = checked. **Summary** = secondary only; check the primary before quoting.

| ID | Citation and link | Used for | Status |
|---|---|---|---|
| S31 | Goodfellow, Bengio & Courville, *Deep Learning* (MIT Press, 2016), Ch 3 §3.13. https://www.deeplearningbook.org/contents/prob.html | Surprisal (from Ep 4) | Read |
| S37 | Goodfellow et al., Ch 4 §4.3, eq. 4.5. https://www.deeplearningbook.org/contents/numerical.html | The update rule; "a positive scalar determining the size of the step"; stopping when the gradient is near zero | Read (inner pages vary between fetches; cite by section) |
| S42 | OpenStax, *Calculus Volume 1*, §3.3. https://openstax.org/books/calculus-volume-1/pages/3-3-differentiation-rules | Power rule: the slope of z² is 2z (from Ep 5) | Read |
| S51 | 3Blue1Brown, "Gradient descent, how neural networks learn," *Neural networks* chapter 2 (Oct 16, 2017). https://www.3blue1brown.com/lessons/gradient-descent | Overshooting and oscillating | Read |
| S53 | Jurafsky & Martin, *Speech and Language Processing*, 3rd ed. draft, Ch 7 §7.7. https://web.stanford.edu/~jurafsky/slp3/7.pdf | Language-model training by gradient descent | Read (from Ep 8) |
| S57 | Barnes & Noble, table of contents for *Deep Learning*. https://valsec.barnesandnoble.com/w/deep-learning-ian-goodfellow/1126352242 | Start pages: Ch 8 p. 267; §8.2 p. 275 (for S59) | Summary (retailer TOC) |
| S58 | Ruder, "An overview of gradient descent optimization algorithms," arXiv:1609.04747 (v1 2016; v2 Jun 15, 2017). https://arxiv.org/abs/1609.04747 | §2.1 the update θ = θ − η·∇θJ(θ); §3 "painfully slow convergence", "or even to diverge"; §4.1 (the ball is momentum) | Read (PDF via fetch) |
| S59 | Goodfellow et al., Ch 8 §8.2.1 (ill-conditioning) and §8.2.3 (flat regions). https://www.deeplearningbook.org/contents/optimization.html | "can manifest by causing SGD to get 'stuck'" | Read (web text) |

Episode references: Episode 5 (derivatives; the power rule; mat's slope −0.70 with R = 0.70), Episode 6 (the gradient; the bowl f = z₁² + 2z₂² from (1, 1); one step 1.36 and 19; one-step limit 0.556 and best single step 0.278), Episode 7 (the loss; the three dots and the mean squared error), Episode 8 (maximum likelihood; η named; the mean, not the sum, sets the gradient's scale). The sources for mini-batches, the 1/N scale, saddle points and local minima (S58 §§2.2–2.3, 4.6; S59 §§8.1.3, 8.2.2–8.2.3, 8.3.1; S60–S63) are in the Episode 10 brief.

## Open items

- Optional print check of Goodfellow §4.3 (eq. 4.5) and §8.2.1, and of Ruder's equation numbers.
- Wording: Goodfellow's "stuck" sentence is about SGD; the narration says "gradient descent". Close enough for the video (SGD is gradient descent on mini-batches, Episode 10), but quote it as SGD in print.
- "Bigger was faster in every run checked" for mat's loss is a numerical check (η from 0.01 to 1000), not a proof; the video says "in every run we checked".
- The steps-to-0.01 rule of thumb 2.3/η is approximate: 230 against the exact 228 at η = 0.01.
- Learning-rate schedules (decreasing η) and "epoch": define in Episode 10 (mini-batch noise) or Episode 16 (training end to end).
- Episode 8's published page and video say generalization is "episode ten"; it is now Episode 11. Add the note to Episode 8's description and README at this episode's publish.
