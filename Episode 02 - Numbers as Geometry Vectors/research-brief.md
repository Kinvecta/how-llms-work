# Episode 2 research brief: Numbers as Geometry: Vectors

The facts behind Episode 2, with sources. The toy vectors in the video are hand-built teaching values, not learned ones; the [companion notebook](notebook.ipynb) reproduces every number from them, with an `assert` on each one. For the geometry, 3Blue1Brown's "Dot products and duality" [S14] is the best companion.

**Running example, used across the whole series:**

> "The cat sat on the mat because it was tired."

Episode 1 used this sentence for next-token probabilities. Here each word gets a toy vector, and we compare "cat" with "mat" and with "tired." Episode 3 stacks these vectors into a table.

## Learning objectives

1. Describe a word's vector as a list of numbers that can be read as a point or as an arrow.
2. Compute a dot product, a norm, and a cosine similarity by hand, and say what each measures.
3. Explain why cosine similarity ignores length, and why "close in vector space" is a tendency, not a guarantee of similar meaning.
4. Describe the king − man + woman ≈ queen result, and name two reasons it is imperfect.

## Notation introduced in this episode

| Symbol | Meaning |
|---|---|
| a, b | Vectors: ordered lists of numbers |
| d | Dimension: how many numbers a vector has |
| aᵢ, bᵢ | The i-th component of a or b |
| Σ | Sum. Σᵢ aᵢbᵢ means "multiply matching components, then add" |
| a · b | Dot product |
| ‖a‖ | Norm: the length of a |
| φ | Angle between two vectors. φ is used because θ is reserved for model parameters (Episode 1) |
| cos φ | Cosine similarity |

x and y are already in use (x for tokens in Episode 1; y for labels in Episode 6), so this episode avoids them.

## 1. Cold open: three words

Cat, mat, tired. Which pair is more alike? A model has no dictionary entry for "alike." It has a list of numbers for every word, and one calculation that turns two lists into a score. The word2vec paper describes its goal as computing "continuous vector representations of words from very large data sets" [S16]. Those vectors are the subject of this episode.

## 2. Vectors as lists and arrows

A vector is an ordered list of numbers. Read it two ways: as a point in space, or as an arrow from the origin to that point. The dimension d counts the numbers.

The toy word vectors have three dimensions, with values chosen by hand. Real vectors are much longer: the word2vec paper tested dimensions from 20 to 1,000 [S16], and the four contextual models in one study (BERT, RoBERTa, GPT-2, XLNet) each use 768-dimensional representations [S23]. Real vectors are learned from text, not typed in [S16]. Every 2-D or 3-D picture in the episode is a teaching aid; there is no picture of 768 dimensions.

## 3. The dot product and the norm

The dot product multiplies matching components and adds the results:

> a · b = Σᵢ aᵢbᵢ

The norm is the length of a vector:

> ‖a‖ = √(a · a)

The dot product also has a geometric form. If φ is the angle between a and b:

> a · b = ‖a‖ ‖b‖ cos φ [S19]

Two readings. First, the dot product measures how far one vector reaches along another: project one onto the other, then multiply by the length [S14]. Second, the sign gives the relationship: positive means roughly the same direction, zero means perpendicular, negative means opposing [S14, S19]. IBM's transformer overview makes the same point in an attention setting: "If they're not aligned, their dot product will be small or negative" [S24]. Episode 17 returns to this.

**The sun picture (Kinvecta analogy, original to this series).** Take the two arrows outside at noon. Sunlight arrives parallel, so every point drops straight down. Lay the first arrow along the ground; the second arrow's shadow on it is the projection, and shadow length (with its sign) × the first arrow's length is the dot product. A flashlight is a point source and would stretch the shadow with distance, so the sun, not a flashlight, is the honest picture.

## 4. Cosine similarity

Divide out the lengths and you get cosine similarity:

> cos φ = a · b / (‖a‖ ‖b‖)

The value runs from −1 (opposite directions) to 1 (same direction), and it does not depend on how long the vectors are. Turney and Pantel note that the most popular way to measure the similarity of two frequency vectors is to take their cosine [S18]. The word2vec paper finds nearest words by cosine distance [S16].

## 5. Math Lab

Hand-built features, illustrative and not learned: axis 1 = concrete thing; axis 2 = living; axis 3 = feeling or state.

- cat = (1, 1, 0)
- mat = (1, 0, 0)
- tired = (0, 0, 1)

**Step 1, dot products.** cat · mat = 1·1 + 1·0 + 0·0 = 1. cat · tired = 1·0 + 1·0 + 0·1 = 0.

**Step 2, norms.** ‖cat‖ = √2 ≈ 1.41; ‖mat‖ = 1; ‖tired‖ = 1.

**Step 3, cosines.** cat and mat: cos φ = 1 / (√2 × 1) ≈ 0.71, so φ = 45°. cat and tired: cos φ = 0, so φ = 90°.

**Step 4, the scaling check.** Let cat′ = (3, 3, 0): the same direction as cat, three times longer. cat′ · mat = 3, so the dot product changed. cos φ between cat′ and mat = 3 / (3√2 × 1) ≈ 0.71: the cosine did not change.

**Takeaway.** cat and mat share the first axis, so they point roughly the same way. cat and tired share no axis. Real models learn their dimensions from text [S16]; nobody names them in advance.

## 6. From similarity to analogy

Mikolov, Yih and Zweig report that "King − Man + Woman results in a vector very close to Queen" [S17]. For "man is to woman as king is to ___," the method computes woman − man + king and returns the word whose vector has the greatest cosine similarity to that result [S17]. The method assumes that "all pairs of words sharing a particular relation are related by the same constant offset" [S17].

Two limits to present alongside it:

1. **The metric.** Linzen (2016) argues that the cosine-based offset test "conflates offset consistency with largely irrelevant neighborhood structure" [S20].
2. **The strength of the claim.** Rogers, Drozd and Li (2017) argue against "linguistic regularities" as a model of relations in vector spaces, and report that the offset method "suffers from dependence on vector similarity" [S21]. Even the 2013 RNN-based model answered "almost 40%" of 8,000 syntactic questions correctly [S17].

## 7. Myth check: "close vectors mean similar meaning"

**The case for.** The distributional hypothesis says that "words that occur in similar contexts tend to have similar meanings" (Harris, 1954) [S18]. Word2vec's nearest neighbors, found by cosine, follow the same pattern [S16].

**The case against.** In transformer representations, cosine can be dominated by a few coordinates. Ethayarajh (2019) reports that in GPT-2's last layer, "two random words will on average have almost perfect cosine similarity" [S22]. Timkey and van Schijndel (2021) find that "a small number of rogue dimensions, often just 1-3, dominate these measures," meaning cosine similarity and Euclidean distance [S23].

Cosine similarity is a useful measuring stick, but in some models a few large coordinates can bend it. "Close" is a tendency, not a guarantee.

## Check question

Using the toy vectors: which pair has the larger cosine, cat–mat or cat–tired? If we replace cat with cat′ = (3, 3, 0), which quantity changes: the dot product with mat, or the cosine?

<details><summary>Answer</summary>

cat–mat (0.71 against 0). The dot product with mat changes from 1 to 3. The cosine stays at 0.71.

</details>

## Sources

| ID | Source | Used for |
|---|---|---|
| S14 | 3Blue1Brown, "Dot products and duality," Essence of linear algebra, Chapter 9 (2016). https://3blue1brown.com/lessons/dot-products | Projection and sign of the dot product |
| S16 | Mikolov, Chen, Corrado and Dean, "Efficient Estimation of Word Representations in Vector Space," arXiv:1301.3781 (2013). https://arxiv.org/abs/1301.3781 | Learned word vectors; dimensions tested (20 to 1,000); cosine distance for nearest words |
| S17 | Mikolov, Yih and Zweig, "Linguistic Regularities in Continuous Space Word Representations," NAACL-HLT 2013, pp. 746–751. https://aclanthology.org/N13-1090.pdf | Vector-offset method; the King − Man + Woman result; constant-offset assumption; "almost 40%" syntactic accuracy |
| S18 | Turney and Pantel, "From Frequency to Meaning: Vector Space Models of Semantics," JAIR 37 (2010), doi:10.1613/jair.2934. https://arxiv.org/pdf/1003.1141 | Distributional hypothesis (Harris, 1954); cosine as the most popular similarity measure |
| S19 | Wyss-Gallifent, "Section 11.3: Dot Product," Math 241 lecture notes, University of Maryland. https://www.math.umd.edu/~immortal/MATH241/lecturenotes/ch11-3.pdf | Dot-product formulas; norm; zero when perpendicular |
| S20 | Linzen, "Issues in evaluating semantic spaces using word analogies," RepEval 2016; arXiv:1606.07736. https://arxiv.org/abs/1606.07736 | Limits of cosine-based analogy evaluation |
| S21 | Rogers, Drozd and Li, "The (too Many) Problems of Analogical Reasoning with Word Vectors," *SEM 2017, pp. 135–148. https://aclanthology.org/S17-1017/ | Critique of "linguistic regularities" |
| S22 | Ethayarajh, "How Contextual are Contextualized Word Representations?" EMNLP-IJCNLP 2019; arXiv:1909.00512. https://arxiv.org/abs/1909.00512 | Near-perfect cosine between random words in GPT-2's last layer |
| S23 | Timkey and van Schijndel, "All Bark and No Bite: Rogue Dimensions in Transformer Language Models Obscure Representational Quality," arXiv:2109.04404 (2021). https://arxiv.org/abs/2109.04404 | Rogue dimensions dominate cosine; 768-dimensional models |
| S24 | IBM, "What is a transformer model?" https://www.ibm.com/think/topics/transformer-model | Dot product as alignment |
| S54 | 3Blue1Brown, "Essence of linear algebra preview," Chapter 0. https://www.3blue1brown.com/lessons/eola-preview | Series name |

**Further exploration:** the 3Blue1Brown *Essence of linear algebra* series [S54] for the geometry; the word2vec paper [S16] for how real word vectors are learned.
