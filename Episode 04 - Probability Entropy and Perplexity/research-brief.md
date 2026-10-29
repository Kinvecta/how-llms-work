# Episode 4: Probability, Entropy, and Perplexity

**Series:** How Large Language Models Work: From Math to Transformers
**Runtime target:** 12–18 minutes (plan for about 16)
**Companion:** Transformer Explainer [S10] for the live probability display. the companion notebook in this folder.

---

## Learning objectives

By the end of this episode, viewers should be able to:

1. Write the chain rule for a sentence as a product of next-token probabilities, and say what each factor conditions on.
2. Compute a softmax from a list of scores, and explain why its outputs are positive and sum to 1.
3. Compute entropy in nats and bits, and explain it as the average surprisal of a distribution.
4. Compute perplexity as the exponential of the average per-token cross-entropy, and say when "number of choices" is only an intuition.

## Running example

> "The cat sat on the mat because it was tired."

Episode 1 gave the next-word probabilities after "The cat sat on the": mat 0.30, floor 0.20, sofa 0.15, other 0.35. These are illustrative, and this episode reuses them exactly. The Episode 3 layer outputs are not probabilities and do not appear here.

## Notation introduced in this episode

| Symbol | Meaning |
|---|---|
| p | A probability distribution over outcomes. For the model, p is p_θ(· \| x<ₜ) from Episode 1 |
| q | A second distribution: the target (data) distribution in cross-entropy |
| ln | Natural logarithm, base e. Bits use log₂ |
| h | The score vector, the input to softmax. This is the Episode 3 symbol h = Wu + β |
| softmax(h)ᵢ | exp(hᵢ) / Σⱼ exp(hⱼ) |
| s | A constant added to every score (Math Lab only) |
| c | Number of equally likely outcomes (uniform case only) |
| H(p) | Entropy: −Σᵢ pᵢ ln pᵢ, in nats |
| H(q, p) | Cross-entropy of q under the model p |
| PP | Perplexity |
| exp(x) | eˣ |

Reused: V, x<ₜ, Σ, i and j, n (sequence length). Avoided: N (Episode 6 mean loss), y and ŷ (labels), and σ (the outline uses it for the sigmoid later).

## Section 1: Cold open (about 1 minute)

Show the four Episode 1 candidates as bars. Then show a second chart where one bar is 0.97. Ask: which model is more sure of itself? Both charts have a tall bar. The question is how to measure the difference.

**Talking point.** Shannon's 1951 paper, "Prediction and Entropy of Printed English," estimated how predictable English is [S2]. His 1948 paper defined the measure itself: a single number for any probability distribution, built to capture uncertainty [S30].

## Section 2: Probability and the chain rule (about 2 minutes)

A probability distribution gives each outcome a number between 0 and 1, and the numbers sum to 1. A conditional probability is the probability of an outcome given some context [S31 §3.6]. In a language model, p(xₜ | x<ₜ) is the distribution over the next token given the prefix.

The chain rule of probability multiplies these factors [S31 Eq. 3.6; S33 Eq. 3.4]:

> P(x₁, …, xₙ) = Π (t = 1 to n) p(xₜ | x<ₜ)

The first factor has an empty prefix (Episode 9: with no start token, the model scores only the later factors). Each later factor conditions on everything before it. In the running example, the factor for "mat" after "The cat sat on the" is 0.30 (illustrative).

**Speaker note.** The calculus chain rule (Episode 5) is a different rule with the same name.

## Section 3: Softmax, from scores to probabilities (about 3 minutes)

The last layer of a language model produces a score vector h, one number per token in the vocabulary V. Scores can be negative and need not sum to anything. Softmax converts them into a distribution [S32 Eq. 6.29]:

> softmax(h)ᵢ = exp(hᵢ) / Σⱼ exp(hⱼ)

Goodfellow calls the same vector z [S32 Eq. 6.28]; this series calls it h. Three properties follow from the formula:

- Every output is positive, since exp is positive.
- The outputs sum to 1, since the denominator is the sum of the numerators.
- The largest score gets the largest probability. Goodfellow notes that softmax is "more closely related to the arg max function than the max function" [S32].

Adding the same number s to every score changes nothing, because exp(s) cancels in the ratio (Math Lab, Step 3). This is why the numerically stable form subtracts the largest score [S32 Eq. 6.33].

## Section 4: Entropy as average surprise (about 2 minutes)

The surprisal of an outcome with probability p is −ln p. Rare outcomes have high surprisal. Goodfellow defines self-information as I(x) = −ln P(x), in nats, and notes that "learning that an unlikely event has occurred is more informative than learning that a likely event has occurred" [S31 §3.13, Eq. 3.48].

Entropy is the average surprisal under a distribution [S31 Eq. 3.49]:

> H(p) = −Σᵢ pᵢ ln pᵢ

Shannon introduced this measure in 1948 as a measure of choice and uncertainty: H = −K Σ pᵢ log pᵢ, where the constant K "merely amounts to a choice of a unit of measure" [S30]. Base 2 gives bits [S30 §1]. Shannon also writes that "only when we are certain of the outcome does H vanish" [S30]. So H is zero for a certain outcome and grows as the outcomes spread out. The minus sign makes H non-negative, since ln p ≤ 0 for p ≤ 1; the sign is printed in the paper itself: on the rendered page of the MIT CSAIL copy, Theorem 2 reads H = −K Σ pᵢ log pᵢ, and the sentence defining entropy reads H = −Σ pᵢ log pᵢ [S30]. A secondary source agrees [S34]. Text extraction of the web copies drops the sign, so check the page image, not copied text.

## Section 5: Cross-entropy and perplexity (about 3 minutes)

Suppose outcomes come from a target distribution q, but the model says p. The cross-entropy is the average surprisal under the model [S31 Eq. 3.51]:

> H(q, p) = −Σᵢ qᵢ ln pᵢ

It splits into the target's own entropy plus a mismatch term [S31 Eqs. 3.50–3.51; S36]:

> H(q, p) = H(q) + D_KL(q ‖ p)

KL divergence is never negative, and it is zero only when p = q [S35]. So cross-entropy is at least the target's entropy, with equality only when the model matches the target.

For a test text of n tokens, the observed tokens give a per-token cross-entropy, CE = −(1/n) Σₜ ln p(xₜ | x<ₜ). Perplexity is its exponential [S33 Eqs. 3.14, 3.42]:

> PP = exp(CE)

Jurafsky & Martin write the same quantity in bits, PP = 2^H(W), with H(W) = −(1/N) log₂ P(W), where N is our n [S33 §3.7, Eq. 3.42]. The two forms agree because 2^(x / ln 2) = eˣ. We avoid N here because Episode 6 uses it for the mean loss.

Two more points from the same chapter: perplexity can be thought of as the "weighted average branching factor of a language" [S33 §3.3.1], and two models can be compared only if they use identical vocabularies [S33 Ch 3].

## Math Lab (about 3 minutes)

All probabilities are the illustrative Episode 1 values. Logs are natural.

**Step 1, check the distribution.** p = (mat 0.30, floor 0.20, sofa 0.15, other 0.35). The sum is 1.00.

**Step 2, softmax returns it.** Set h = (ln 0.30, ln 0.20, ln 0.15, ln 0.35) ≈ (−1.20, −1.61, −1.90, −1.05). Then exp(hᵢ) = pᵢ, and the denominator is the sum of the pᵢ, which is 1. So softmax(h) = p exactly. A second example: h = (0, 1, 2) gives (0.090, 0.245, 0.665).

**Step 3, the shift check.** Adding s to every score multiplies every numerator and the denominator by exp(s). The ratio is unchanged. Adding 5 to (0, 1, 2) gives (5, 6, 7), and softmax still gives (0.090, 0.245, 0.665).

**Step 4, entropy.** Surprisals: mat 1.20 nats (1.74 bits); floor 1.61 (2.32); sofa 1.90 (2.74); other 1.05 (1.51).
H(p) = 0.30(1.204) + 0.20(1.609) + 0.15(1.897) + 0.35(1.050) = 1.335 nats = 1.93 bits.

**Step 5, perplexity of this distribution.** exp(1.335) = 3.80. That is close to 4 but not 4, because the outcomes are unequal. For c equally likely outcomes, H = ln c and exp(H) = c. For c = 4, H = 1.386 nats (2 bits) and exp(H) = 4.

**Step 6, the lumped bucket.** "Other" stands for many tokens. Splitting it into its tokens gives H_full = 1.335 + 0.35 × H(tokens within "other"), and the added term is zero or more. So 1.335 nats is a lower bound on the entropy over the full vocabulary.

**Step 7, perplexity of one observed token.** If the true next token is "mat," the cross-entropy is −ln 0.30 = 1.204 nats, and PP = exp(1.204) = 3.33 = 1/0.30. A single observed token always gives PP = 1/p.

## Misconception check: "perplexity is the number of choices" (about 1 minute)

The intuition is partly right. For one distribution, exp(H) is the number of equally likely choices that would give the same entropy. It is exact only for a uniform distribution (Step 5). Three limits:

- **Scope.** For a model evaluated on text, perplexity is exp of the average surprisal of the observed tokens, not exp(H) of the model's own predictions. Since H(q, p) ≥ H(q) [S31; S35], perplexity is at least exp of the target's entropy, and equal to it only when the model matches the target.
- **Confidence is not correctness.** Entropy depends only on the model's probabilities. If the model gives 0.97 to "sofa" and 0.01 to each other word, its entropy is 0.17 nats, which is low. If the observed token is "mat," its perplexity is 1/0.01 = 100 [derived from the definitions].
- **Comparability.** Perplexities compare only on identical vocabularies [S33].

Episode 1's taxi-map result is the same lesson at scale: near-perfect next-token accuracy alongside an implied map that does not match the streets [S6].

**Line for narration.** "Perplexity says how surprised the model is by the text it sees. It does not say whether the model is right."

## Recap and check question (about 1 minute)

**Recap.** The chain rule turns a sentence into a product of next-token probabilities. Softmax turns scores into those probabilities. Entropy is the average surprisal of one distribution. Perplexity is the exponential of the average surprisal of the observed tokens.

**Check question.** A model spreads its probability evenly over four candidate words. (a) What is its entropy, in nats and bits? (b) What is its perplexity? (c) A second model gives "mat" 0.97 and 0.01 to each of the other three. Is its entropy higher or lower than the first model's?

*Answer for the script:* (a) ln 4 = 1.39 nats = 2 bits. (b) 4. (c) Lower: 0.17 nats against 1.39.

**After this, you can explain…**
- Why a sentence's probability is a product of next-token probabilities, and what each factor conditions on.
- How softmax turns scores into probabilities, and why adding a constant to every score changes nothing.
- Entropy as the average surprisal of a distribution, in nats and bits.
- What perplexity measures, why it is exp of cross-entropy, and when "number of choices" is only an intuition.

## Visual plan for Episode 4

| Beat | Visual | Idea source |
|---|---|---|
| Cold open | Four candidate bars after "The cat sat on the," then a 0.97 bar for contrast | Original; display idea from [S10] |
| Chain rule | Sentence as boxes, each labelled with its factor p(xₜ \| x<ₜ) | Original design |
| Softmax | Scores → exponentials → normalized bars that sum to 1 | Original; "probabilities that add to 1" from [S11] |
| Shift check | All score bars rise by the same amount; output bars stay still | Original |
| Surprisal | Bar height = −ln p per candidate; average drawn as a line at 1.34 nats | Original |
| Perplexity | Four equal slots against unequal bars, marked "3.8 of 4" | Original; "branching factor" framing from [S33] |
| Misconception | Low-entropy bar with the observed token at 0.01, PP = 100 | Original |
| Analogy (proposal) | Weather forecast: "97% rain" (low entropy) against four equal outcomes (high), then the day the 1% outcome arrives | Kinvecta library: checked against the catalog; `x_forecast` is a hurricane cone, not spread, so build the card new; spec in the analogy specs |


## Worked in the video (added for the published episode)

- Chain rule with GPT-2 small's measured factors (Episode 1 notebook): p(on | The cat sat) ≈ 0.22, p(the | The cat sat on) ≈ 0.45, product 0.099; all ten scored tokens ≈ 1.5 × 10⁻¹⁷; average surprisal 3.88 nats, perplexity ≈ 48; two tokens: (1.51 + 0.80) ÷ 2 = 1.16 nats, perplexity ≈ 3.2 = 1 ÷ √0.099.
- Softmax worked in full: e⁰ = 1, e¹ = 2.718, e² = 7.389, sum 11.107; shift +5: 148.4, 403.4, 1096.6, sum 1648.5; stable form (−2, −1, 0): 0.135, 0.368, 1, sum 1.503; all give 0.090, 0.245, 0.665.
- Entropy term by term: uniform 4 × 0.347 = 1.386 nats (2 bits); Model A 0.361 + 0.322 + 0.285 + 0.367 = 1.335 nats (1.93 bits); Model B 0.030 + 3 × 0.046 ≈ 0.17 nats.
- GPT-2 small's entropy over all 50,257 tokens after "The cat sat on the": about 5.4 nats (Episode 1 notebook), far wider than the four-bar picture.

## Sources

Status key: **Read** = checked in this research pass. **Summary** = secondary source only; check the primary before quoting. **Memory** = recalled, not checked.

| ID | Source | Used for | Status |
|---|---|---|---|
| S2 | Shannon, "Prediction and Entropy of Printed English," *Bell System Technical Journal*, 1951. https://sites.socsci.uci.edu/~rfutrell/teaching/itl-davis/readings/shannon1951prediction.pdf | Cold open | Read (Ep 1 pass) |
| S6 | Vafa et al., NeurIPS 2024, arXiv:2406.03689. https://arxiv.org/html/2406.03689 | Taxi-map limit | Read (Ep 1 pass) |
| S9 | Jurafsky & Martin, *Speech and Language Processing*, 3rd ed. draft (Aug 19, 2026), chapter list. https://web.stanford.edu/~jurafsky/slp3/ | Chapter numbers | Read (draft) |
| S10 | Transformer Explainer (Polo Club). https://poloclub.github.io/transformer-explainer/ | Display idea | Read (earlier pass) |
| S11 | Medium, "Transformer architecture explained" (Amanatullah). https://medium.com/@amanatulla1606/transformer-architecture-explained-2c49e2257b4c | "Softmax ... probabilities (that add to 1)"; visual idea only | Read (text only; re-read this pass) |
| S12 | DataCamp, "How transformers work." https://www.datacamp.com/tutorial/how-transformers-work | Negative check on the spinner idea | Read (text only) |
| S30 | Shannon, "A Mathematical Theory of Communication," *Bell System Technical Journal* 27 (1948), pp. 379–423, 623–656. https://www.dcs.gla.ac.uk/~wpc/grcs/shannon1948.pdf (also https://people.csail.mit.edu/dmoshkov/courses/codes/shannon1948.pdf and https://www.cs.yale.edu/homes/yry/readings/general/shannon1948.pdf) | Entropy as choice and uncertainty; K as a unit; bits; H vanishes when certain | Read (Sections 1 and 6; three web copies, which drop the minus in text extraction; the minus is printed on the rendered MIT CSAIL page, printed p. 11, in Theorem 2 and the entropy definition; reprint page numbers) |
| S31 | Goodfellow, Bengio & Courville, *Deep Learning* (MIT Press, 2016), Ch 3 §3.6 (Eq 3.6) and §3.13 (Eqs 3.48–3.51). https://www.deeplearningbook.org/contents/prob.html | Chain rule; self-information; entropy; cross-entropy; KL; nats | Read (§3.6 starts p. 57 and §3.13 p. 70 per S57) |
| S32 | Goodfellow et al., Ch 6 §6.2.2.3 (Eqs 6.28, 6.29, 6.33). https://www.deeplearningbook.org/contents/mlp.html | Softmax; argmax; stable form | Read (page truncated after §6.5.5; §6.2.2.3 read in full) |
| S33 | Jurafsky & Martin (draft), Ch 3 §§3.3, 3.3.1, 3.7 (Eqs 3.4, 3.14, 3.42). https://web.stanford.edu/~jurafsky/slp3/3.pdf | Perplexity; base-2 form; branching factor; vocabulary caveat | Read (draft; section numbers from extraction) |
| S34 | Ben Eggers, "A Mathematical Theory of Communication" (expository paper, not Shannon's text), §3.1. https://sites.math.washington.edu/~morrow/336_16/papers/ben.pdf | Minus sign in H = −K Σ p log p | Summary (secondary) |
| S35 | "Nonnegativity of Information Measures," course wiki. https://classes.up-microlab.org/index.php/Nonnegativity_of_Information_Measures | KL ≥ 0; zero only when equal | Read (course wiki; no author) |
| S36 | Jacar, "Entropy, Cross-Entropy and KL Divergence." https://jacar.es/en/entropy-cross-entropy-kl-divergence/ | Identity H(P,Q) = H(P) + D_KL(P‖Q) | Summary (secondary; checked against S31 algebra) |
| S57 | Barnes & Noble, table of contents for Goodfellow et al., *Deep Learning*. https://valsec.barnesandnoble.com/w/deep-learning-ian-goodfellow/1126352242 | Section start pages for S31 | Summary (retailer TOC) |


