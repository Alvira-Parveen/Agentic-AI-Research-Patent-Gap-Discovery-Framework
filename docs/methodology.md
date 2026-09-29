# Research Methodology: Novelty Scoring & Gap Discovery

## 1. Mathematical Novelty Scoring Model

A critical requirement of this project is avoiding simplistic, scientifically ungrounded formulas like $\text{Novelty} = 100 - \text{Similarity}$. As demonstrated in prior-art literature (*Ikoma & Mitamura 2025*, *Han & Qu 2026*), semantic similarity alone does not determine patent novelty: two documents can share vocabulary while disclosing fundamentally distinct patentable claim mechanisms.

### 1.1 Factor Breakdown
Our framework decomposes prior-art disclosure into three constituent factors:
1. **Semantic Similarity ($S_{sem}$)**:
   $$S_{sem} = \max_{c \in \mathcal{C}_{ret}} \cos(\mathbf{e}_{query}, \mathbf{e}_c)$$
   The peak cosine similarity across retrieved patent and paper chunks.
2. **Technical Feature Overlap ($O_{tech}$)**:
   $$O_{tech} = \frac{|\mathcal{F}_{idea} \cap \mathcal{F}_{prior}|}{|\mathcal{F}_{idea}|}$$
   The ratio of proposed technical features matching disclosures in retrieved documents.
3. **Claim Feature Coverage ($C_{claim}$)**:
   The degree to which independent and dependent claims in retrieved patents specifically cover the core technical elements of the idea.

### 1.2 Prior-Art Overlap Index (POI)
The combined prior-art density is computed as a weighted linear combination:
$$POI = w_{sem} \cdot S_{sem} + w_{tech} \cdot O_{tech} + w_{claim} \cdot C_{claim}$$
where default research parameters are:
- $w_{sem} = 0.40$
- $w_{tech} = 0.35$
- $w_{claim} = 0.25$
- Constraints: $w_{sem} + w_{tech} + w_{claim} = 1.0$.

### 1.3 Evidence Strength Adjustment ($E_{str}$)
To prevent skewing when few documents are available:
$$E_{str} = \min\left(1.0, \frac{|\mathcal{C}_{ret}|}{K_{target}}\right) \times \left(0.5 + 0.5 \cdot \bar{S}\right)$$
where $|\mathcal{C}_{ret}|$ is the number of retrieved items and $\bar{S}$ is the mean similarity.

### 1.4 Preliminary Novelty Score
$$\text{Novelty Score} = 1.0 - \left(POI \times (0.60 + 0.40 \cdot E_{str})\right)$$
Bounded in $[0.0, 1.0]$.

Categorization:
- **High**: $\text{Score} \ge 0.70$
- **Moderate**: $0.40 \le \text{Score} < 0.70$
- **Low**: $\text{Score} < 0.40$

---

## 2. Evidence-Grounded Gap Discovery

White-space discovery contrasts two distinct literature traditions:
1. **Patent Literature**: Focuses on patentable claims, industrial utility, manufacturing implementations, and commercial system architectures.
2. **Scientific Literature**: Focuses on theoretical novelty, empirical benchmarks, training methodologies, and ablation studies.

### Gap Detection Algorithm
For each extracted feature $f_i \in \mathcal{F}_{idea}$:
- Determine Patent Coverage: $\text{Cov}_{pat}(f_i) \in \{\text{Fully Covered}, \text{Partially Covered}, \text{Uncovered}\}$
- Determine Paper Coverage: $\text{Cov}_{pap}(f_i) \in \{\text{Fully Covered}, \text{Partially Covered}, \text{Uncovered}\}$

**Classification Rules**:
- If $f_i$ is uncovered in both $\implies$ **Underrepresented Feature**.
- If $f_a$ is covered in patents, $f_b$ is covered in papers, but their joint intersection $(f_a \wedge f_b)$ is absent $\implies$ **Unexplored System Combination Gap**.
- The system explicitly qualifies: *"Potential gap based on retrieved evidence — indicates technical underrepresentation in the retrieved local corpus and does not establish global absence"*.

---

## 3. Global Research Disclaimer
> *Research prototype only. Results represent AI-assisted technical overlap analysis against the retrieved local corpus. They are not a legal patentability opinion, exhaustive prior-art search, or guarantee of novelty.*

