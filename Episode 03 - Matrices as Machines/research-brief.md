# Episode 3 research brief: Matrices as Machines

The facts behind Episode 3, with sources. All weights and word vectors are hand-chosen teaching values, not learned ones; the [companion notebook](notebook.ipynb) reproduces every number, with an `assert` on each one. For the geometry, 3Blue1Brown's "Linear transformations and matrices" [S25] and "Matrix multiplication as composition" [S26] are the best companions.

**Running example, used across the whole series:**

> "The cat sat on the mat because it was tired."

Episode 2 gave cat, mat and tired toy vectors: cat = (1, 1, 0), mat = (1, 0, 0), tired = (0, 0, 1). This episode stacks them into one matrix, one word per row.

## Learning objectives

1. Read a matrix as a rule that moves vectors, and say what its columns mean.
2. Multiply two matrices by hand and check the result's shape with the rule (m×k)(k×r) = m×r.
3. Write a linear layer as h = Wu + β, and explain why the bias makes it affine rather than strictly linear.
4. Explain why two linear layers with no nonlinearity between them compute no more than one.

## Notation introduced in this episode

| Symbol | Meaning |
|---|---|
| u | Input vector to a linear layer, with d components |
| W | Weight matrix of a linear layer, shape d_out × d |
| β | Bias vector, with d_out components (beta, because b is the Episode 2 vector) |
| h | Output vector of a linear layer, with d_out components |
| U | Stacked inputs: one input vector per row, shape n × d |
| n | Number of rows in U (sequence length in the running example) |
| A, B | Generic matrices in the product definition |
| m, k, r | Generic shapes: A is m×k, B is k×r, AB is m×r. k is also the summed index |
| (AB)ᵢⱼ | Entry in row i, column j of AB |
| Aᵀ | Transpose: (Aᵀ)ᵢⱼ = Aⱼᵢ |

Reused: d (dimension), Σ (sum), i and j (indices). Avoided: x and y (tokens, labels), p (probability, Episode 4), H (entropy, Episode 4).

## 1. Cold open: four numbers

A grid with one arrow in it. Multiply by a 2×2 matrix and the grid bends and stretches while it stays a grid: lines stay lines, evenly spaced, and the origin never moves. A grid of numbers can be read as a rule that moves every point in space [S25].

## 2. Matrices as transformations

The informal test for a linear transformation: lines stay lines and the origin stays fixed [S25]. A matrix describes one such transformation, and its columns say where the basis vectors land: the first column is where the first basis vector goes, the second column where the second goes [S25].

Matrix-vector multiplication follows from that picture: multiply each coordinate of the vector by the matching column, and add the results [S25]. Each output component is also a dot product: component i of Wu is row i of W dotted with u [S27, Eq. 2.5], using the dot product from Episode 2 [S19].

Two things the picture can hide. A non-square matrix changes dimension: a 2×3 matrix takes three numbers in and returns two. And a linear map keeps the origin fixed [S25], so it cannot shift a vector. Shifting is the job of the bias.

## 3. The matrix product and its shape rule

For A with m rows and k columns, and B with k rows and r columns:

> (AB)ᵢⱼ = Σₖ AᵢₖBₖⱼ

The product is defined only when the inner sizes match, and the result takes the outer sizes: (m×k)(k×r) gives m×r [S27, §2.2]. Three checks: (2×3)(3×2) gives 2×2; (3×2)(2×3) gives 3×3; (2×3)(2×3) is undefined. Matrix multiplication is distributive and associative, and not commutative: AB and BA can differ [S27, §2.2, Eqs. 2.6–2.7].

**Worked by hand (the episode does all four entries).** A = [[1, 0, 1], [0, 2, 0]] (2×3) and B = [[1, 0], [0, 1], [1, 1]] (3×2), so AB is 2×2.

| Entry | Row of A · column of B | Products | Sum |
|---|---|---|---|
| (1, 1) | (1, 0, 1) · (1, 0, 1) | 1·1, 0·0, 1·1 | 1 + 0 + 1 = **2** |
| (1, 2) | (1, 0, 1) · (0, 1, 1) | 1·0, 0·1, 1·1 | 0 + 0 + 1 = **1** |
| (2, 1) | (0, 2, 0) · (1, 0, 1) | 0·1, 2·0, 0·1 | 0 + 0 + 0 = **0** |
| (2, 2) | (0, 2, 0) · (0, 1, 1) | 0·0, 2·1, 0·1 | 0 + 2 + 0 = **2** |

So AB = [[2, 1], [0, 2]]. Four dot products, four entries.

**Composition.** Multiplying two matrices applies one transformation after another, with the right-hand matrix acting first: ABu = A(Bu) [S26]. A rotation then a stretch and the stretch then the rotation give different pictures: order changes the result [S26].

## 4. The linear layer and its bias

> h = Wu + β

u has d components, W has shape d_out × d, and β and h have d_out components. Goodfellow, Bengio and Courville write the same form as z = Wx + b and call it an affine transformation [S28, §6.3]; the symbols here differ because x and b are reserved in this series.

"Linear layer" is the common name, and the book calls an affine map with no nonlinearity a "linear unit" [S28, §6.2.2.1]. Strictly, a linear map keeps the origin fixed [S25]; when β is not zero the origin moves, so the map is affine. W bends the grid; β moves it. A hidden layer then applies a fixed nonlinearity ψ, entry by entry, after the affine step [S28, §6.3]; Episode 11 covers ψ.

**The mixing board (Kinvecta analogy, original to this series).** Three input strips (cat, mat, tired) feed two output buses through a 2×3 grid of dials: each output is a weighted sum of the inputs, and the grid of dials is W. A trim knob on each bus adds a fixed amount: the trim is the bias. Disclosed gaps: a real board has no such knob, and here the dials are set by hand, while a model learns them (Episode 7).

## 5. Math Lab

All weights are hand-chosen and illustrative, not learned.

**Setup.** W = [[1, 0, 0], [0, 2, 1]] (2×3), and β = (0, 1).

**Step 1, stack the words.** U has rows cat, mat, tired: U = [[1, 1, 0], [1, 0, 0], [0, 0, 1]] (3×3).

**Step 2, apply W to each row.** Each output component is a row of W dotted with the input. cat: (1·1 + 0·1 + 0·0, 0·1 + 2·1 + 1·0) = (1, 2); mat: (1, 0); tired: (0, 1).

**Step 3, add β = (0, 1).** The outputs are (1, 3), (1, 1) and (0, 2).

**Step 4, batched form.** Stack the inputs as the rows of U (n × d); the MIT introml notes use the same row convention, row i = token i [S29, §9.4.1]. Row i of U Wᵀ equals (W uᵢ)ᵀ, by the transpose definition [S27, Eq. 2.3], so the batched layer is U Wᵀ with β added to every row, shape (n × d)(d × d_out) = n × d_out. Here Wᵀ = [[1, 0], [0, 2], [0, 1]] (3×2) and U Wᵀ is (3×3)(3×2) = 3×2. Row 1 by hand: cat · column 1 = 1·1 + 1·0 + 0·0 = 1; cat · column 2 = 1·0 + 1·2 + 0·1 = 2, so row 1 is (1, 2). Rows 2 and 3 are (1, 0) and (0, 1): the Step 2 results in one product. Some notes write the input as a column and multiply by Wᵀ, others as a row and multiply by W; both describe the same map [S29, §9.3, §9.4.2].

**Step 5, preview: two layers collapse.** Add a second layer W′ = [1, 1] (1×2) with no bias. W′(W cat) = W′(1, 2) = 3. The single matrix W′W = [1, 2, 1] gives [1, 2, 1] · (1, 1, 0) = 3, the same answer. In general W′(Wu + β) = (W′W)u + W′β, by distributivity and associativity [S27, §2.2].

## 6. Myth check: "more linear layers means more power"

**The intuition.** Each layer adds parameters, so a stack should be more capable than one layer.

**Evidence against.** Step 5 collapses two layers into one matrix. Goodfellow's argument is that if a layer's function were linear, "the feedforward network as a whole would remain a linear function of its input," and "we must use a nonlinear function to describe the features" [S28, §6.1].

**Evidence for a narrower claim.** Linear layers still do real work: the 3-to-2 layer changes dimension and mixes input components. Added power comes from the nonlinearity between layers (Episode 11).

**A second overstatement: "every layer is a matrix multiply."** A hidden layer is an affine map followed by a nonlinearity [S28, §6.3], and attention applies a softmax to each row of the score matrix [S29, §9.4.1], which is not a matrix product.

A stack of linear maps is still one linear map. The bend has to come from somewhere else.

## Check question

A layer takes 3-number inputs and returns 2-number outputs. (a) What shape is W? (b) If 10 token vectors are stacked as rows, what shape is the batched output? (c) Two linear layers have no nonlinearity between them. Why does one matrix do the same job?

<details><summary>Answer</summary>

(a) 2×3, which is d_out × d. (b) 10×2. (c) The two matrices multiply to one matrix, so the stack computes the same map as one layer.

</details>

## Sources

| ID | Source | Used for |
|---|---|---|
| S19 | Wyss-Gallifent, "Section 11.3: Dot Product," Math 241 notes, University of Maryland. https://www.math.umd.edu/~immortal/MATH241/lecturenotes/ch11-3.pdf | Dot product (from Episode 2) |
| S25 | 3Blue1Brown, "Linear transformations and matrices," Essence of linear algebra, chapter 3 (2016). https://www.3blue1brown.com/lessons/linear-transformations | Linearity test (informal); columns as landing points; matrix-vector as column combination |
| S26 | 3Blue1Brown, "Matrix multiplication as composition," chapter 4 (2016). https://3blue1brown.com/lessons/matrix-multiplication | Product as composition; right-to-left order |
| S27 | Goodfellow, Bengio and Courville, *Deep Learning* (MIT Press, 2016), ch. 2, §§2.1–2.2, Eqs. 2.3–2.7. https://www.deeplearningbook.org/contents/linear_algebra.html | Product formula and shape rule; transpose; algebraic properties |
| S28 | Goodfellow, Bengio and Courville, *Deep Learning*, ch. 6, §§6.1, 6.2.2.1, 6.3. https://www.deeplearningbook.org/contents/mlp.html | Affine layer; "linear units"; collapse argument; need for nonlinearity |
| S29 | MIT introml, "Transformers" notes, §§9.3–9.4. https://introml.mit.edu/notes/transformers.html | Row convention and n × d shapes; row-wise softmax |
| S54 | 3Blue1Brown, "Essence of linear algebra preview," Chapter 0. https://www.3blue1brown.com/lessons/eola-preview | Series name |

**Further exploration:** the 3Blue1Brown *Essence of linear algebra* series [S54] for the geometry; Goodfellow et al., chapter 2, for the algebra in full.
