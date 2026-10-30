# Episode 6: Calculus for Learning, Part 2 · The Gradient and the Step

> The research for *Calculus for Learning* covers both parts. During production the topic was split into two episodes so every step could be worked more slowly. This episode (11:31) uses the sections on partial derivatives and the gradient, the steepest direction, the gradient step and the myth check, plus directional slopes by hand and a derivation of 3 − 20η + 36η². [Episode 5](../Episode%2005%20-%20Derivatives%20and%20the%20Chain%20Rule/research-brief.md) uses the sections on the derivative, the calculus chain rule and the Math Lab.

**Series:** How Large Language Models Work: From Math to Transformers
**Companion:** the companion notebook in this folder.

---

## Learning objectives

By the end of this episode, viewers should be able to:

1. Read f′(z) as the slope of f at z, and use f(z + Δ) ≈ f(z) + Δ·f′(z) for a small change Δ.
2. Apply the calculus chain rule, d/dz f(g(z)) = f′(g(z))·g′(z), and keep it separate from the chain rule of probability.
3. Compute a gradient of a two-variable function, and say which way it points.
4. Explain why a small step against the gradient lowers a function, and why "small" matters.

## Running example

> "The cat sat on the mat because it was tired."

Episode 4 gave the surprisal of "mat" after "The cat sat on the" as −ln 0.30 = 1.204 nats [S31], using the illustrative Episode 1 probabilities. This episode pushes that surprisal down and tracks how the probability and the surprisal move when the model's score z for "mat" is nudged.

In this episode the quantity −ln p is called **surprisal**, the term Episode 4 defined. It is not yet called a loss; the loss L is defined in Episode 7 as the mean of these terms. The raw numbers that softmax turns into probabilities are still called scores (the score vector h from Episode 4). The score z for "mat" is one of those.

## Notation introduced

| Symbol | Meaning |
|---|---|
| z | Calculus variable: one number, or z = (z₁, z₂) for two inputs. In the running example, z is the score for "mat" (an entry of h). Not Goodfellow's z |
| f, g | Generic functions. f(g(z)) is a composition |
| ∂f/∂z₁ | Partial derivative: how f changes when only z₁ changes |
| ∇f | Gradient: the vector of all partial derivatives |
| D_a f | Directional derivative: slope of f in the unit direction a |
| Δ | A small change in an input |
| η | Step size: how far we move against the gradient. Episode 9 calls it the learning rate |
| R | Total weight of the other candidates (0.70 here), held fixed while z changes |

Not used: H (still entropy from Episode 4), ε (replaced by Δ and η), and θ for angles (φ from Episode 2).

## Section 1: Cold open (about 1 minute)

Show a hiker in fog who can feel only the slope under their boots. They want to get lower and have no map. Ask: what can you learn from the ground under your feet?

*Analogy (proposal; checked against the Kinvecta catalog, where the existing `x_gradient` scene is a close base to adapt): a hiker in fog.*

Talking point: a derivative is the local answer to that question.

## Section 2: Derivative as slope (about 1.5 minutes)

The derivative f′(z) is the slope of f at z [S37, p. 81]. For a small change Δ, f(z + Δ) ≈ f(z) + Δ·f′(z) [S37, p. 81; no equation number in the text]. Two rules we need: the power rule d/dz z² = 2z [S42, Thm 3.3] and d/dz ln z = 1/z [S44, §3.3.1].

## Section 3: The calculus chain rule (about 2.5 minutes)

Compose two functions: g first, then f. Then

> d/dz f(g(z)) = f′(g(z))·g′(z) [S41]

Call this the **calculus chain rule**. The chain rule of probability from Episode 4 (a sentence's probability is a product of conditional probabilities [S31]) is a different idea with a similar name. On screen, always use the full name.

Narrate it as cause and effect. A nudge in z moves g by g′(z) times the nudge. That move in g moves f by f′(g(z)) times itself. Multiply the two effects. The 3Blue1Brown lesson "Visualizing the chain rule and product rule" (Chapter 5) states the rule in this form [S38].

## Section 4: Partial derivatives and the gradient (about 2 minutes)

For f(z₁, z₂), the partial derivative ∂f/∂z_i measures how f changes when only z_i increases [S37, p. 82]. The gradient ∇f collects all of them [S37, p. 82].

Example: f(z₁, z₂) = z₁² + 2z₂². Then ∂f/∂z₁ = 2z₁ and ∂f/∂z₂ = 4z₂ [S42, power and constant-multiple rules], so ∇f = (2z₁, 4z₂). Its contour lines are ellipses around (0, 0): a bowl. At (1, 1), f = 3 and ∇f = (2, 4).

## Section 5: Steepest direction, and when it holds (about 2 minutes)

For a unit vector a, the directional derivative is D_a f = ∇f · a = ‖∇f‖ cos φ, where φ is the angle between a and ∇f [S43, §14.5; S37, Eq. 4.4, where Goodfellow writes θ].

- The largest value comes when φ = 0, so a points along ∇f. **The gradient points in the direction of steepest ascent**, and ‖∇f‖ is the slope in that direction [S43].
- The smallest value comes when a points against ∇f. So **steepest descent is −∇f** [S43; S37, Eq. 4.3].

The claim holds under three conditions:

1. f is differentiable at the point, so the formula applies [S43, page metadata].
2. The gradient is nonzero. At a zero gradient the direction is undefined: "the gradient vector ... has no direction" [S43, Example 3]. Saddle points are zero-gradient points [S37, p. 82].
3. "Unit" and "steepest" are Euclidean. Goodfellow's formulas use the norm ‖a‖₂ [S37, Eq. 4.4].

## Section 6: Why a small step works, and why "small" matters (about 2.5 minutes)

A gradient step moves z to z − η∇f [S37, Eq. 4.5, p. 83]. Goodfellow states that f(x − ε·sign(f′(x))) is less than f(x) "for small enough ε" [S37, p. 81]. The limit comes from curvature. A second-order expansion gives

> f(z − η∇f) ≈ f(z) − η(∇f)·(∇f) + ½η²·(curvature term) [S37, Eq. 4.9]

Goodfellow calls the η² term "the correction we must apply to account for the curvature of the function," and warns that when it is too large, "the gradient descent step can actually move uphill" [S37, p. 85].

**Worked example, the bowl at (1, 1).** Here (∇f)·(∇f) = 20 and the curvature term is 72. So, exactly for this quadratic, f(η) = 3 − 20η + 36η² (derived; the 36η² term matches Eq. 4.9).

- η = 0.1: f = 1.36. Down.
- η = 1.0: f = 19. Up.
- f falls for every η between 0 and 0.556 (= 40/72). The best step is η = 20/72 ≈ 0.278, giving f ≈ 0.222. The optimal-step formula in Goodfellow's Eq. 4.10 gives the same 0.278 [S37].

So "small" means small compared with the curvature. Episode 9 turns this into a slider.

## Section 7: Math Lab (about 3 minutes)

All probabilities are the illustrative Episode 1 values, and Episode 4's identity softmax(ln p) = p holds.

**Step 1, two links.** Let S(z) = f(g(z)), where f(p) = −ln p and g(z) = e^z / (e^z + R). Here z is the score for "mat". R = 0.20 + 0.15 + 0.35 = 0.70 is the weight of the other candidates, held fixed. At z = ln 0.30, g(z) = 0.30.

**Step 2, outer link.** f′(p) = −1/p [S44; S41; constant-multiple rule, S42].

**Step 3, inner link.** By the quotient rule [S45] and the derivative of e^z [S41], g′(z) = R·e^z / (e^z + R)². Since 1 − g = R / (e^z + R), this equals g(1 − g) = p(1 − p).

**Step 4, calculus chain rule.** d/dz S = f′(g(z))·g′(z) = (−1/0.30)(0.30)(0.70) = **−0.70** (derived; checked numerically).

**Step 5, the nudge.** Raising z by 0.10 moves p from 0.300 to 0.321 and the surprisal from 1.204 to 1.135 nats. The first-order estimate was −0.070, and the true change is −0.069. A step of 1.0 changes the surprisal by −0.584, against an estimate of −0.70, so the straight-line estimate fails at large steps.

**Note.** You cannot nudge one probability by itself: under softmax the others must move so that the total stays 1. The knob is the score z. The same calculation for all four candidates gives −(1 − p) for "mat" and +p for each other candidate. This is the first look at the backward pass in Episode 14.

## Section 8: Myth check: "the gradient points to the bottom" (about 1.5 minutes)

**For the idea:** near a point, a small step against the gradient lowers the function [S37, p. 81]. In the bowl, that is true.

**Against it:**

1. The gradient is local. It tells you which way is uphill from here [S37, p. 83].
2. Even in a bowl, the gradient usually misses the bottom. At (1, 1), −∇f = (−2, −4) points 18.4° away from the bottom at (0, 0) (derived). The contours are ellipses, not circles.
3. A zero gradient gives no direction. Saddle points are critical points that are neither maxima nor minima [S37, p. 82].

*Line for narration:* "The gradient tells you which way is down from here. It does not tell you where the bottom is."

## Section 9: Recap and check question (about 1 minute)

**Recap.** A derivative is a slope. The calculus chain rule multiplies the effects along a composition. The gradient collects the partial derivatives and points uphill; its negative points steepest downhill. A small step lowers a function, and the curvature sets how small "small" must be.

**Check question.** For f(z₁, z₂) = z₁² + 2z₂² at (2, 1): (a) What is ∇f? (b) Which unit direction decreases f fastest? (c) Roughly how much does f fall per unit of distance moved that way?

*Answer for the script:* (a) (4, 4). (b) (−1, −1)/√2. (c) About ‖∇f‖ = 4√2 ≈ 5.66 per unit distance (derived).

**After this, you can explain…**
- What a derivative measures, and how a small change Δ moves a function.
- The calculus chain rule, and how it differs from the chain rule of probability.
- What a gradient is, which way it points, and the conditions under which "steepest" holds.
- Why a gradient step lowers a function only when the step is small enough.

## Visual plan

| Beat | Visual | Idea source |
|---|---|---|
| Cold open | Hiker in fog, feeling the slope under their boots | Kinvecta analogy: adapt `x_gradient`; spec in the analogy specs |
| Derivative | A tangent line on a curve, with Δ marked | Original |
| Chain rule | Two linked boxes, z → g(z) → f(g(z)), each with its own slope | Original. Idea only from [S38]; no image reused |
| Gradient bowl (outline visual) | Ellipse contours with a gradient arrow uphill and its negative downhill, labelled "surprisal" | Original. No reviewed source has this image (visual-inspiration-v1 lists Episode 5 as a gap) |
| Running example | Curve of −ln p against p, with the tangent at 0.30 and the chord to 0.32 | Original. Idea from the Math Lab |
| Small step | Parabola f(η) = 3 − 20η + 36η², with the decrease zone (0 to 0.556) shaded | Original. Idea from [S37, pp. 84–85] (text, not a figure) |
| Myth check | Gradient arrow and direction-to-bottom arrow at (1, 1), 18.4° apart | Original |

No image from [S37] or [S38] is reused.

## Worked in the video (added for the published episode)

- Slope the slow way for f(z) = z² at 3: steps 1, 0.1, 0.01 give 7, 6.1, 6.01; the power rule gives 6.
- Surprisal's slope at p = 0.30 is −1/p = −3.33; nudging p to 0.32 predicts 1.137 against the true 1.139.
- Tiny chain: g(z) = 3z, f(u) = u² at z = 1 gives 6 × 3 = 18, and 9z² → 18z → 18 the long way.
- The four-word gradient is −0.70, +0.20, +0.15, +0.35 (sum 0), included as a first look at backpropagation (Episode 14).
- Check question answer worked in the video: ∇f = (4, 4), (−1, −1)/√2, √32 ≈ 5.66.

## Sources

Status key: **Read** = checked in this research pass. **Summary** = secondary source only. **Memory** = recalled, not checked (none used here).

| ID | Source | Used for | Status |
|---|---|---|---|
| S31 | Goodfellow et al., *Deep Learning* (MIT Press, 2016), Ch 3 §3.13 (reused from Episode 4) | Surprisal −ln p | Read (Ep 4 pass) |
| S37 | Goodfellow, Bengio & Courville, *Deep Learning*, Ch 4 §4.3 (begins p. 79; cited statements on pp. 81–85) and §4.3.1 (pp. 85–86). https://www.deeplearningbook.org/contents/numerical.html | Slope; partial derivatives; gradient; uphill and downhill; directional derivative; small-step condition; curvature; saddle points. Eqs 4.3, 4.4, 4.5, 4.9, 4.10 | Read (web text; §4.3 start agrees with S57; inner pages and Eq 4.5 from the fetch; check print) |
| S38 | 3Blue1Brown, "Visualizing the chain rule and product rule," Chapter 5. https://3blue1brown.com/lessons/chain-rule-and-product-rule | Chapter title and number; chain-rule statement | Read (lesson page; dated May 1, 2017) |
| S39 | Manning, "3Blue1Brown: Essence of Calculus" (book listing). https://www.manning.com/books/3blue1brown-essence-of-calculus | Series title (backed by S55) | Summary (secondary listing only) |
| S40 | 3Blue1Brown, calculus topic page. https://3blue1brown.com/topics/calculus | Negative check on the series title | Read (shows no series title or chapter list) |
| S41 | OpenStax, *Calculus Volume 1*, Ch 3 Key Equations. https://openstax.org/books/calculus-volume-1/pages/3-key-equations | Chain rule; d/dx ln g = g′/g; d/dx e^g = e^g·g′ | Read (no section numbers on the page) |
| S42 | OpenStax, *Calculus Volume 1*, §3.3 Differentiation Rules. https://openstax.org/books/calculus-volume-1/pages/3-3-differentiation-rules | Power rule (Thm 3.3, n a positive integer); constant-multiple and sum rules | Read (excerpt ends before the quotient rule) |
| S43 | LibreTexts, *Calculus (Guichard)* §14.5, Directional Derivatives. https://math.libretexts.org/Bookshelves/Calculus/Book%3A_Calculus_(Guichard)/14%3A_Partial_Differentiation/14.05%3A_Directional_Derivatives | D_a f = ∇f·a; steepest ascent and descent; zero-gradient case | Read |
| S44 | *Applied Calculus* (based on Boelkins, *Active Calculus*) §3.3.1, Derivatives of Log Functions. https://mathbooks.unl.edu/AppliedCalculus/sec-2-7-der-log.html | d/dx ln x = 1/x; d/dx ln g = g′/g | Read (page heading says §3.3; URL slug says sec-2-7) |
| S45 | LibreTexts, Chabot College MTH 15, §3.3 Product and Quotient Rules. https://math.libretexts.org/Courses/Chabot_College/MTH_15%3A_Applied_Calculus_I/03%3A_The_Derivative/3.03%3A_Product_and_Quotient_Rules | Quotient rule | Read |
| S46 | Srihari, CSE 676 lecture slides, "Numerical Computation." https://cedar.buffalo.edu/~srihari/CSE676/NumericalComputation.pdf | Corroboration only | Summary (slides that follow Goodfellow; not the book) |
| S55 | 3Blue1Brown, "The Essence of Calculus," Chapter 1, and "The paradox of the derivative," Chapter 2. https://3blue1brown.com/lessons/essence-of-calculus ; https://3blue1brown.com/lessons/derivatives | Series name (Chapter 1 page title and slug; the text says "this series"); Chapter 2 introduces the derivative | Read |
| S57 | Barnes & Noble, table of contents for Goodfellow et al., *Deep Learning*. https://valsec.barnesandnoble.com/w/deep-learning-ian-goodfellow/1126352242 | §4.3 start page (p. 79) | Summary (retailer TOC) |


