# 📘 Information Retrieval System — Extended Boolean vs Fuzzy Model

## 1. Overview
This project demonstrates a comparative implementation of two classical **Information Retrieval (IR)** models — the **Extended Boolean Model** and the **Fuzzy Model** — using a small experimental corpus consisting of **15 documents (D1–D15)** and **4 key terms** (`bird`, `cat`, `dog`, `tiger`).

The system allows users to explore and visualize how both models process document–query similarity through an **interactive Streamlit web interface**, enabling direct experimentation with queries, parameters, and ranking outputs.

---

## 2. Objective
The main objectives of this project are to:

- Illustrate the **mathematical foundation** and **procedural difference** between the Extended Boolean and Fuzzy models.  
- Allow users to **interactively adjust** parameters such as *p-norm value* or *query composition*.  
- Provide an **educational tool** that visualizes ranking results and intermediate calculations step-by-step.

---

## 3. Features
✅ **Full Computation Pipeline (no mock data)**  
All values — TF, IDF, weights, correlations, and similarities — are computed directly from the document corpus.

✅ **Interactive Streamlit Web Interface**  
Users can select models, modify queries, and visualize top-ranked results dynamically.

✅ **Multi-Page Design**
1. **📊 Page 1 — Ranking Visualization**  
   Displays *Top-k similarity scores* of both models using horizontal bar charts.
2. **📘 Page 2 — Extended Boolean Model**  
   Shows TF, IDF, weight calculation, similarity, and ranking.
3. **📗 Page 3 — Fuzzy Model**  
   Demonstrates membership, correlation, and completion steps with mathematical formulas.

✅ **Custom Query Parsing**  
Supports Boolean operators and parentheses:  


✅ **Export Functionality**  
All computed results can be downloaded as `.csv` for further analysis.

---

## 4. Methodology

### 4.1 Extended Boolean Model
The Extended Boolean Model generalizes standard Boolean retrieval by introducing **graded similarity** through *p-norm functions*.

#### **Computation Steps**
1. **TF–IDF Weighting**
   - \( tf_{t,d} = \frac{\text{count}(t,d)}{\max_x \text{count}(x,d)} \)
   - \( idf_t = \log_{10}(N / n_t) \)
   - Normalized by the maximum IDF.
   - \( w_{t,d} = tf_{t,d} \times idf_t \)

2. **Similarity (p-norm)**
   - **DeMorgan version:**  
     \( AND(x,y) = 1 - \left( \frac{(1-x)^p + (1-y)^p}{2} \right)^{1/p} \)
   - **Direct p-norm version:**  
     \( AND(x,y) = \left( \frac{x^p + y^p}{2} \right)^{1/p} \)
   - \( OR(x,y) = 1 - AND(1-x,1-y) \),  \( NOT(x) = 1 - x \)

3. **Ranking**  
   Documents are ranked by descending similarity scores.

---

### 4.2 Fuzzy Model
The Fuzzy Model introduces **soft membership** and **correlation-based completion**, capturing partial relevance between terms and documents.

#### **Computation Steps**
1. **Membership Calculation**
   - \( \mu_t(D) = \frac{\text{count}(t,D)}{\max_x \text{count}(x,D)} \)

2. **Weighted Jaccard Correlation**
   - \( c(i,j) = \frac{\sum_D \min(\mu_i(D), \mu_j(D))}{\sum_D \max(\mu_i(D), \mu_j(D))} \)

3. **Membership Completion**
   - If term \(t\) is missing in document \(D\):  
     \( \mu'_t(D) = \max_{x \in T(D)} [\mu_x(D) \cdot c(t,x)] \)

4. **Similarity Evaluation**
   - \( AND(a,b) = \min(a,b) \)
   - \( OR(a,b) = \max(a,b) \)
   - \( NOT(x) = 1-x \)

5. **Ranking**
   - The final similarity \( S(D) \) is calculated and sorted to produce ranked results.

---

## 5. System Architecture

| Component | Description |
|------------|-------------|
| **Frontend** | Streamlit Web UI for visualization and parameter control |
| **Backend** | Python-based computation (NumPy, Pandas, Math) |
| **Visualization** | Plotly Express for interactive bar charts |
| **Storage** | In-memory document corpus (no external database) |

---

## 6. Dataset Description
| Document | Example Tokens |
|-----------|----------------|
| D1 | {bird, cat, bird, cat, dog, dog, bird} |
| D2 | {cat, dog, tiger} |
| D3 | {bird, dog, bird} |
| D4 | {cat, tiger} |
| D5 | {tiger, cat, dog, tiger, dog} |
| ... | ... |
| D15 | {tiger, cat, bird, tiger} |

---

## 7. Example Query
**User Query:**  
> “I love cats and dogs but not tigers.”  
Mathematically expressed as:  
\[
Q = (cat \, AND \, dog) \, AND \, NOT \, tiger
\]

This query is consistently used across both models for evaluation.

---

## 8. Experimental Results Summary
| Model | Best Documents | Observations |
|--------|----------------|---------------|
| **Extended Boolean** | D8, D1, D7 | Produces sharper score differences, emphasizing exact matches. |
| **Fuzzy** | D1, D11, D14 | Handles missing terms gracefully through correlation-based inference. |

**Interpretation:**  
- Extended Boolean focuses on **precision** and strict term overlap.  
- Fuzzy Model emphasizes **semantic flexibility**, inferring similarity through correlated terms (e.g., `cat ↔ tiger`).

---

## 9. Installation & Execution

### Prerequisites
- Python ≥ 3.10  
- Required Libraries:  
  ```bash
  pip install streamlit pandas numpy plotly

