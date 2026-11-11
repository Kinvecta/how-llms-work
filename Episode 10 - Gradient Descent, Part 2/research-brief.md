# Episode 10: Gradient Descent, Part 2: Mean vs Sum, and Mini-batches

**Series:** How Large Language Models Work: From Math to Transformers
**Runtime:** 17:47
**Companion:** Colab notebook "Episode 10 - Gradient Descent, Part 2" (the experiment on line A, mean against sum; the per-dot gradients and both steps with exact fractions; the curvature of the mean loss, the safe limits 2/λmax, a sweep of η and the rule "η on the sum = N × η on the mean"; check 1; mini-batch draws, with every mini-batch of one, two and eight listed; sampling with and without replacement; 100× the examples; line B and mini-batch steps at the bottom with a fixed and a shrinking η; check 2; every number asserted).

> **Note.** Gradient descent, part 2 was split again. The Episode 9 brief gave "Episode 10, Gradient Descent, Part 2" all of the research file's second half: Math Lab Steps 4–6, Section 4 (mini-batches), Section 5 (what gradient descent does not promise) and the other two myths. **Episode 10 is now only this:** Math Lab Step 4 (the 1/N scale of mean versus sum), Step 5 and Section 4 (mini-batch gradients: unbiased, noisy, the 1/√ rule, noise at a minimum), and the research file's check part (c), the sum of four copies of 3z². **Episode 11, where gradient descent gets stuck,** takes Section 5 (saddles, flat regions, valleys that are not the deepest), the remaining myths, and the pointer to momentum from Step 6. **Generalization is now Episode 12.**
>
> **Numbering.** Because of the split, every topic after gradient descent moves down by one from the list in the Episode 9 brief: 11 where gradient descent gets stuck, 12 generalization, 13 training examples, 14 tokens and embeddings, 15 the neuron, 16 backpropagation, 17 training end to end (Adam), 18 representations, 19 counting words. (Only 11 and 12 are confirmed in this episode's script; the rest is the same list shifted, to be confirmed in the series ledger.) Earlier references are unchanged: Episode 5 (derivatives), Episode 6 (the gradient), Episode 7 (the loss, three dots), Episode 8 (maximum likelihood, cross-entropy), Episode 9 (the learning rate).
>
> **Format.** New in this episode: predict-then-reveal. Five short questions each end with a few seconds of silence before the answer (the cold-open guess, dot 1's slopes, the step on the sum, the long-run average of the draws, the nine pairs), and two longer checks ask the viewer to pause (the sum of four copies of 3z²; slopes 0, 3, 9).

---

## Learning objectives

By the end of this episode, viewers should be able to:

1. Compute one dot's gradient by hand (−2 × input × error for W, −2 × error for β) and combine the per-dot gradients as a mean and as a sum.
2. Explain why the gradient of the sum is N times the gradient of the mean, why that makes the safe step N times smaller, and predict whether a given η settles or explodes on each.
3. Say what a mini-batch is, and why its gradient is unbiased (right on average) when the picks are random.
4. Compute the variance of a mini-batch gradient for a small example, and explain why the typical miss shrinks only like 1/√(batch size), so many quick steps beat a few perfect ones.
5. Explain why a mini-batch step still moves at a minimum, and why the learning rate is lowered over time.

## Running example

Episode 7's three dots: (u, y) = (1, 2), (2, 4), (3, 5), N = 3, model ŷ = W·u + β, loss = the mean of the squared errors. **Line A** is W = 2, β = 0 (errors 0, 0, −1; loss 1/3). **Line B** is the best fit, W = 1.5, β = 2/3 (loss 1/18).

The series sentence, "The cat sat on the mat because it was tired," is not used for a calculation in this episode. The language model enters only as scale: N is billions of words.

## Notation introduced

| Symbol or word | Meaning |
|---|---|
| error | The dot's height minus the line's height, y − ŷ. Negative means the line is too high |
| mean, sum | The two ways to combine the N penalties (and so the N per-dot gradients) |
| full batch, mini-batch | All N terms; a small random handful of them |
| batch size | How many terms are in a mini-batch (the series ledger's N′; the notebook's `m`). No symbol on screen |
| stochastic | Random: each step uses a random mini-batch |
| unbiased | Usually off on any one draw, right on average |
| variance | The average squared distance from the mean |
| typical miss | √variance: the standard deviation of the mini-batch gradient (Goodfellow's "standard error" of the mean) |
| λmax | Largest eigenvalue of the loss's curvature matrix. Notebook only; the video says "the notebook works out the safe limit" |

Reused: θ, η, κ and the factor 1 − 2ηκ (Ep 9); the gradient (Ep 6); L, N, the three dots, u, y, W, β, lines A and B (Ep 7); cross-entropy (Ep 8).

## Section 1: Cold open

**The experiment.** Line A, η = 0.1. One step on the mean loss: 1/3 → 2/15 ≈ 0.133. Change one word, mean → sum, with the same dots, line and η: scored as the mean squared error, the loss goes to 1.2, then 5.947, then 31.71 (the video says 5.95 and 31.7) [derived].

**Second puzzle.** A language model learns from billions of words but never looks at all of them before a step. **Analogy (original):** a chef tasting a chunky stew by the spoonful. Stir = shuffle; spoonful = mini-batch; one spoonful may be mostly broth, the next mostly potato.

## Section 2: Reminder from Episode 9

Valley κz²: z is the distance from the bottom, κ the steepness, η the step size (learning rate). Each step multiplies z by 1 − 2ηκ, so η must stay below 1/κ. "The steeper the valley, the smaller the safe step" [Ep 9; derived].

## Math Lab, Step 1: one dot at a time

The loss is the average of N penalties, error². Differentiation is linear, so the slope of an average is the average of the slopes [derived; checked numerically on 100 random lines in the notebook].

For one dot, with error e = y − (W·u + β): ∂(e²)/∂W = −2·u·e and ∂(e²)/∂β = −2·e [derived: chain rule; Ep 5's power rule gives the 2].

On line A [derived; exact fractions in the notebook]:

| Dot | Error | Gradient (W, β) |
|---|---|---|
| 1 | 0 | (0, 0) |
| 2 | 0 | (0, 0) |
| 3 | 5 − 6 = −1 | (−2 × 3 × (−1), −2 × (−1)) = (6, 2) |

Mean: (2, 2/3). Sum: (6, 2). Same direction, exactly N = 3 times longer.

## Math Lab, Step 2: both steps, η = 0.1

| | New (W, β) | Errors | Score (MSE) |
|---|---|---|---|
| Mean | (1.8, −1/15) | 4/15, 7/15, −1/3 | 2/15 ≈ 0.133 |
| Sum | (1.4, −0.2) | 0.8, 1.4, 1 | (0.64 + 1.96 + 1)/3 = 3.6/3 = 1.2 |

The score is the mean squared error both times, so the two runs are compared on the same scale [derived].

## Math Lab, Step 3: why

The sum loss is N times the mean loss, so every slope, and every curvature, is N times bigger. The mean loss in (W, β) is a quadratic bowl with curvature matrix (Hessian) (2/N)·XᵀX = (2/3)·[[14, 6], [6, 3]]. Along an eigen-direction with eigenvalue λ, one step multiplies the distance from the bottom by 1 − ηλ, so gradient descent settles exactly when η < 2/λmax. This is Episode 9's η < 1/κ with κ = λ/2 [derived].

- Mean: λmax = (17 + √265)/3 ≈ 11.093 (κ = 5.546, as in Episode 9), **safe limit 0.1803**.
- Sum: λmax ≈ 33.279, **safe limit 0.0601** = 0.1803 ÷ 3.
- η = 0.1 sits between them. Notebook sweep: the mean settles at 0.18 and diverges at 0.19; the sum settles at 0.06 and diverges at 0.065 [derived; checked numerically].

**The rule.** η on the sum is the same walk as N·η on the mean (identical trajectories, asserted). So 0.1 on the sum of three dots = 0.3 on the mean, past 0.18 [derived].

**Analogy (original).** A teacher adjusting tomorrow's lesson from today's quiz: the class average, or the total of all thirty students' missed points (thirty times bigger). The total is not wrong; she would have to react thirty times more gently, and change that whenever the class size changes.

**In real code.** PyTorch's `torch.nn.CrossEntropyLoss` has `reduction='mean'` as its default [S62]. **Common mistake:** copying a learning rate without checking whether the loss is a mean or a sum. The claim "the mean is the usual default because the same step size keeps working when the amount of data changes" is our explanation, shown in the notebook (three dots against thirty: the mean's limit does not move, the sum's falls tenfold) [derived], not a quote from a source.

## Check 1

f(z) = 3z² (Episode 9: η = 0.1 gives the factor 0.4 and settles). The sum of four copies is 12z², so κ = 12 and the safe limit is 1/12 ≈ 0.083. At η = 0.1: 2 × 0.1 × 12 = 2.4; 1 − 2.4 = −1.4. Each step lands on the other side, 1.4 times farther out [derived]. This is the research file's check part (c).

## Section 3: Mini-batches

**The problem.** With N in the billions, one full-batch step needs a slope for every term. Ruder: batch gradient descent "can be very slow and is intractable for datasets that do not fit in memory" [S58, §2.1].

**The idea.** Shuffle, take a small random handful (a mini-batch), average those slopes, step, repeat. Ruder: "Mini-batch gradient descent is typically the algorithm of choice when training a neural network…" and "Common mini-batch sizes range between 50 and 256, but can vary for different applications." [S58, §2.3]. The video quotes only "the algorithm of choice" and gives no batch sizes.

**Is the direction wrong?** Each step, a little. 3Blue1Brown's picture, from the stochastic gradient descent section of "What is backpropagation really doing?": someone stumbling quickly downhill, against someone who carefully works out the exact downhill direction before each slow step [S60]. The narration paraphrases it and separates it from Episode 9's three hikers (those differed in stride; this one differs in aim).

## Math Lab, Step 4: unbiased

W slopes on line A: 0, 0, 6. Full gradient: 2. A mini-batch of one gives 0, 0 or 6, each a third of the time, and never 2. Long-run average: (0 + 0 + 6)/3 = 2 [derived]. Notebook: a seeded run of 60 draws averages 2.6, and 60,000 draws average 1.997.

Unbiased = usually off on any one draw, right on average. It needs random picks: Goodfellow et al. stress that minibatches must be selected randomly [S59, §8.1.3].

## Math Lab, Step 5: the noise

Variance = the average squared distance from the mean [derived; all by exact enumeration in the notebook]:

| Batch size | All equally likely batches | Variance | Typical miss |
|---|---|---|---|
| 1 | 3: values 0, 0, 6 | (4 + 4 + 16)/3 = 24/3 = 8 | √8 ≈ 2.83 |
| 2, put back | 9 pairs: average 0 (4 pairs), 3 (4 pairs), 6 (1 pair) | (4×4 + 4×1 + 1×16)/9 = 36/9 = 4 | 2 |
| 8, put back | 3⁸ = 6,561 | 1 | 1 |

The variance is (variance of one draw)/(batch size); the typical miss falls like 1/√(batch size). Eight times the data, and the miss did not even drop to a third (2.83 → 1).

Goodfellow et al. compare a gradient estimated from 100 examples with one from 10,000: 100 times more computation, the standard error reduced only by a factor of 10 [S59, §8.1.3]. On our slopes: √(8/100) = 0.2828 against √(8/10,000) = 0.0283.

## Check 2

Slopes 0, 3, 9. Average (0 + 3 + 9)/3 = 4. Distances 4, 1, 5; squares 16, 1, 25; total 42; variance 14. Two per batch: 7. Typical miss √14 ≈ 3.74 → √7 ≈ 2.65 (the video says about 3.7 and about 2.6): a factor of √2, not 2 [derived].

## Section 4: Still moving at the bottom

Line B (W = 1.5, β = 2/3): errors −1/6, 1/3, −1/6. Per-dot W slopes 1/3, −4/3, 1 (sum 0); per-dot β slopes 1/3, −2/3, 1/3 (sum 0). The full gradient is zero, but no single dot's is, so a mini-batch step still moves [derived].

Goodfellow et al.: the SGD gradient estimator introduces noise that "does not vanish even when we arrive at a minimum", so it is necessary to gradually decrease the learning rate [S59, §8.3.1].

Notebook (seeded, 3,000 mini-batch-of-one steps starting exactly on line B): with a fixed η = 0.05 the typical distance from line B stays near 0.08; with η = 0.05/(1 + step/100) it falls from 0.027 to 0.014; the full batch does not move at all. Illustrative: the values depend on the seed and the schedule, and are not in the video.

## Section 5: Recap and try it

**Recap.** The gradient of an average is the average of the gradients. Add instead of average, and every slope gets N times bigger: the valley is N times as steep, so the safe step is N times smaller. A mini-batch averages a random handful: usually off on any one draw, right on average, with a typical miss that shrinks like 1/√(batch size), so many quick steps beat a few perfect ones.

**Try it.** In the notebook, run line A with the mean and with the sum; draw mini-batches of one, two and eight; make up three slopes and work out the variance by hand. The one mistake to avoid: reusing a learning rate without checking whether the loss is a mean or a sum.

**After this, you can explain…**
- Why the same η can be calm on a mean and explosive on a sum.
- Why the mean is the usual default.
- What a mini-batch is, and why its gradient is unbiased but noisy.
- Why a hundred times the data buys only ten times the precision, and why the learning rate is lowered over time.

**Next.** Episode 11, where gradient descent gets stuck: saddles, valleys that are not the deepest, and momentum.

## Visual plan

| Beat | Visual | Idea source |
|---|---|---|
| Cold open | The three dots and line A; one gold step (mean), then coral steps (sum); a "your guess" card with a countdown | Original |
| Chef | A chef tasting a chunky stew by the spoonful | Original |
| Per-dot slopes | Worksheet lines, one per spoken phrase: −2 × 3 × (−1) = 6 | Original |
| Mean against sum | Two arrows, same direction, one three times longer; both stepped lines | Original |
| Why | The η number line with both limits and 0.1 between them | Original; Episode 9's number line |
| Teacher | Class average against the total of thirty students | Original |
| Mini-batch | Stir, spoonful, step; the stumbling hiker | Original; idea only from 3Blue1Brown [S60] |
| Draws | Draws of 0, 0, 6 piling up; the running average settling on 2 | Original |
| Variance | The three draws, the nine pairs, the typical-miss bars for one, two and eight | Original |
| Line B | Three per-dot slopes that cancel; a jiggle at the bottom of the bowl | Original |

Colours: gold = the mean, the focus, correct; coral = the sum, the explosion, noise; steel = neutral data. The notebook's plots use the same three.

## Derived numbers (all asserted in the notebook)

| Quantity | Work | Value |
|---|---|---|
| Line A | errors y − (2u + 0); mean of squares | 0, 0, −1; 1/3 |
| Per-dot gradients on line A | (−2·u·e, −2·e) | (0, 0), (0, 0), (6, 2) |
| Mean and sum gradient | average; total | (2, 2/3); (6, 2) = 3 × the mean |
| Mean step, η = 0.1 | (2, 0) − 0.1 × (2, 2/3) | W 1.8, β −1/15 ≈ −0.07; errors 4/15, 7/15, −1/3 (≈ 0.27, 0.47, −0.33); MSE 2/15 ≈ 0.133 |
| Sum step, η = 0.1 | (2, 0) − 0.1 × (6, 2) | W 1.4, β −0.2; errors 0.8, 1.4, 1; squares 0.64, 1.96, 1; total 3.6; MSE 1.2 |
| Further sum steps | repeat; score as MSE | 5.947 (5.95), 31.71 (31.7) |
| Curvature of the mean loss | (2/3)·[[14, 6], [6, 3]]; λmax = (17 + √265)/3 | 11.093 (κ = 5.546) |
| Safe limit, mean | 2/λmax | 0.1803 (≈ 0.18) |
| Safe limit, sum | 2/(3 λmax) | 0.0601 (≈ 0.06) |
| Sweep of η | settles / diverges | mean 0.18 / 0.19; sum 0.06 / 0.065 |
| The rule | η on the sum = N·η on the mean | 0.1 → 0.3 (past 0.18); identical trajectories |
| More data | thirty dots (ten copies) | mean limit unchanged, 0.1803; sum limit 0.0060 |
| Check 1 | 4 × 3z²; 1/12; 1 − 2 × 0.1 × 12 | 12z²; 0.083; 2.4; −1.4 (one copy: 0.4) |
| Draws from 0, 0, 6 | long-run average | 2; seeded: 2.6 after 60 draws, 1.997 after 60,000 |
| Variance, batch of 1 | (4 + 4 + 16)/3 | 8; typical miss 2.83 |
| Variance, batch of 2 (put back) | 9 pairs: 0 ×4, 3 ×4, 6 ×1; 36/9 | 4; typical miss 2 |
| Variance, batch of 8 (put back) | all 6,561 batches | 1; typical miss 1 |
| Pairs without replacement | 3 pairs: averages 0, 3, 3 | mean 2; variance 2 |
| Finite-population factor | (N − m)/(N − 1) | 1/2 for N = 3, m = 2; 0.9831 for N = 60; 0.99999992 for N = 3 billion, m = 256 |
| 100 against 10,000 examples | √(8/100), √(8/10,000) | 0.2828, 0.0283: 10 times smaller for 100 times the work |
| Line B | errors; per-dot W slopes; β slopes | −1/6, 1/3, −1/6; 1/3, −4/3, 1 (sum 0); 1/3, −2/3, 1/3 (sum 0); MSE 1/18 |
| Jiggle at line B (seeded) | 3,000 steps, batch of one | fixed η: 0.085 then 0.077; shrinking η: 0.027 then 0.014 |
| Check 2 | slopes 0, 3, 9 | average 4; squares 16, 1, 25; total 42; variance 14; pairs 7; √14 ≈ 3.74, √7 ≈ 2.65 |

## Sources

Status key: **Read** = checked. **Summary** = secondary only; check the primary before quoting.

| ID | Citation and link | Used for | Status |
|---|---|---|---|
| S58 | Ruder, "An overview of gradient descent optimization algorithms," arXiv:1609.04747 (v1 2016; v2 Jun 15, 2017). https://arxiv.org/abs/1609.04747 | §2.1: "batch gradient descent can be very slow and is intractable for datasets that do not fit in memory". §2.3: "Mini-batch gradient descent is typically the algorithm of choice when training a neural network…"; "Common mini-batch sizes range between 50 and 256, but can vary for different applications." | Read |
| S59 | Goodfellow, Bengio & Courville, *Deep Learning* (MIT Press, 2016), Ch 8 §8.1.3 and §8.3.1. https://www.deeplearningbook.org/contents/optimization.html | §8.1.3: 100 against 10,000 examples needs 100 times more computation and reduces the standard error only by a factor of 10; minibatches must be selected randomly. §8.3.1: noise that "does not vanish even when we arrive at a minimum"; it is necessary to gradually decrease the learning rate | Read (web text; cite by section) |
| S60 | 3Blue1Brown, "What is backpropagation really doing?," *Neural networks* (Nov 3, 2017). https://www.3blue1brown.com/lessons/backpropagation | The stochastic gradient descent section: quick, stumbling steps downhill against carefully calculated slow ones. Idea only; paraphrased | Read |
| S62 | PyTorch documentation, `torch.nn.CrossEntropyLoss`. https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html | The default is `reduction='mean'` | Read |

Episode references: Episode 5 (derivatives; the power rule), Episode 6 (the gradient), Episode 7 (the loss; the three dots, lines A and B, the mean squared error), Episode 8 (cross-entropy; the mean, not the sum, sets the gradient's scale), Episode 9 (the valley κz², the factor 1 − 2ηκ, η < 1/κ; f = 3z² with factor 0.4; the three dots' κ = 5.546). The sources for saddle points, flat regions and local minima (S58 §§3, 4.1; S59 §§8.2.2–8.2.3) move to the Episode 11 brief.

## Open items

- **With or without replacement.** The video and the main notebook cells draw with replacement ("put the dot back"), which gives the clean rule variance = (variance of one draw)/(batch size) and lets a batch of eight come from three dots. Real training usually shuffles and samples without replacement within an epoch. For three dots and pairs that changes the variance from 4 to 2 (factor (N − m)/(N − 1) = 1/2); for large N the factor is 1 to many decimals. The notebook has a note and a cell on this; the video does not mention it. Add a line to the description if a viewer asks.
- **"Error" = dot − line.** Error is y − ŷ throughout (Episode 7's convention), so line A's error at dot 3 is −1 and the slopes carry a minus sign: −2 × input × error. Many texts use ŷ − y, which flips the sign of the error and drops the minus from the slope. The gradients are the same either way.
- **Safe limit = 2/λmax.** The video says "the notebook works out the safe limit". It is 2/λmax of the curvature matrix, which is Episode 9's 1/κ with κ = λmax/2. The values 0.1803 and 0.0601 are for this bowl (exact quadratic); "safe" means the repeated step settles, not that it is fast. At η = 0.18 the walk settles very slowly (factor −0.997 along the steep direction).
- Unbiasedness is shown here for the training mean (uniform random picks, exact enumeration). Goodfellow §8.1.3 states it for the gradient of the generalization error with fresh samples; the episode cites Goodfellow only for random selection, the 100 against 10,000 comparison and the noise at a minimum.
- Goodfellow's comparison is about the standard error of the mean; the video says "typical error" and "typical miss". The wording in §8.1.3 and §8.3.1 beyond the quoted phrase is paraphrased here; check the print edition before quoting more.
- PyTorch: with class-index targets, no class weights and no ignored targets, `reduction='mean'` divides by the batch size. The page was read at the stable URL; pin a version if this is quoted in print.
- 3Blue1Brown's image is paraphrased in the narration ("stumbles a bit left and right, but always leaning downhill"); do not present it as a quote.
- The fixed-η against shrinking-η simulation at line B and the 60 / 60,000 draws are seeded illustrations, not video numbers.
- Episode 9's published brief lists "Episode 10" as covering saddles, flat regions and local minima too, and generalization as Episode 11 (its episode page only promises mean versus sum and mini-batches, which is still right). Add a note at this episode's publish: those topics are Episode 11, and generalization is Episode 12. Learning-rate schedules and "epoch" are still to be defined (Episode 11 or the training end-to-end episode).
