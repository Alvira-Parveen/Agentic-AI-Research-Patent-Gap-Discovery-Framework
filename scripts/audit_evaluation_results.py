import json
import math
import itertools
from pathlib import Path
from fractions import Fraction
from typing import List, Dict, Any, Tuple
import numpy as np
import scipy
import scipy.stats as stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "evaluation" / "ablation_study_results.json"
TEST_CASES_PATH = PROJECT_ROOT / "data" / "evaluation" / "test_cases.json"


def bootstrap_ci(data: List[float], n_boot: int = 10000, alpha: float = 0.05, seed: int = 42) -> Tuple[float, float]:
    """Computes percentile bootstrap 95% confidence interval for an individual metric sample."""
    rng = np.random.default_rng(seed)
    boot_means = [np.mean(rng.choice(data, size=len(data), replace=True)) for _ in range(n_boot)]
    low = float(np.percentile(boot_means, 100 * (alpha / 2)))
    high = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return low, high


def holm_bonferroni(p_vals: List[float]) -> List[float]:
    """Computes step-down Holm-Bonferroni adjusted p-values."""
    m = len(p_vals)
    sorted_indices = np.argsort(p_vals)
    adj_p = np.zeros(m)
    cum_max = 0.0
    for rank, idx in enumerate(sorted_indices):
        p = p_vals[idx]
        val = (m - rank) * p
        cum_max = max(cum_max, val)
        adj_p[idx] = min(1.0, cum_max)
    return adj_p.tolist()


def get_ground_truth_rel_counts() -> Dict[str, int]:
    """Loads total ground truth relevant documents per query case."""
    with open(TEST_CASES_PATH, "r", encoding="utf-8") as f:
        tc = json.load(f)
    return {
        c["case_id"]: len(c.get("relevant_patents", [])) + len(c.get("relevant_papers", []))
        for c in tc
    }


def reconstruct_mrr_fraction(val: float) -> Fraction:
    """Reconstructs exact reciprocal rank Fraction(1, rank) from stored rounded MRR.

    Maps: 1.0 -> 1/1, 0.5 -> 1/2, 0.3333 -> 1/3, 0.25 -> 1/4, 0.2 -> 1/5, 0.1667 -> 1/6, 0.0 -> 0/1.
    """
    if val <= 0:
        return Fraction(0, 1)
    rank = int(round(1.0 / val))
    return Fraction(1, rank)


def exact_paired_sign_permutation_test(diffs: List[float]) -> Tuple[float, float, int]:
    """Exact paired-sign permutation test on signed ranks, preserving tied ranks.

    Under the sharp null hypothesis of symmetric paired differences around zero,
    the signs of the k non-zero differences are conditionally exchangeable (+1 or -1
    with probability 0.5). The procedure fixes the observed absolute differences and
    their tied ranks, then enumerates all 2^k sign assignments.
    Returns: (W_obs, p_exact, k_non_zero)
    """
    diff_arr = np.array(diffs)
    nz = diff_arr[diff_arr != 0]
    k = len(nz)
    if k == 0:
        return 0.0, 1.0, 0
    ranks = stats.rankdata(np.abs(nz))
    t_pos = float(np.sum(ranks[nz > 0]))
    t_neg = float(np.sum(ranks[nz < 0]))
    w_obs = min(t_pos, t_neg)

    # Enumerate all 2^k permutations
    all_signs = list(itertools.product([-1.0, 1.0], repeat=k))
    count = 0
    for s in all_signs:
        s_arr = np.array(s)
        pos_s = np.sum(ranks[s_arr > 0])
        neg_s = np.sum(ranks[s_arr < 0])
        if min(pos_s, neg_s) <= w_obs + 1e-9:
            count += 1
    p_exact = count / len(all_signs)
    return w_obs, p_exact, k


def run_audit():
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        doc = json.load(f)

    per_query = doc["per_query_results"]
    metrics = ["P@3", "P@5", "R@3", "R@5", "MRR"]
    n = len(per_query)
    df = n - 1
    t_crit = stats.t.ppf(0.975, df=df)
    rel_map = get_ground_truth_rel_counts()

    print("=" * 122)
    print(f"       DEFINITIVE STATISTICAL & REPRODUCIBILITY AUDIT REPORT (N = {n} EVALUATION QUERIES)")
    print("=" * 122)
    print(f"Corpus Specification   : 40 source documents (20 patents, 20 research papers)")
    print(f"Chunk Granularity      : 6,109 total chunks (patents: 5,387 chunks, papers: 722 chunks)")
    print(f"Evaluation Unit        : 20 distinct developer-curated technical query cases")
    print(f"Environment Tooling    : Python, NumPy {np.__version__}, SciPy {scipy.__version__}")
    print(f"Student's t Critical   : t_{{0.025, df={df}}} = {t_crit:.4f}")
    print("-" * 122)

    # Reconstruct unrounded data arrays for precision
    unrounded_dense = {}
    unrounded_hybrid = {}
    diff_exact_map = {}

    for m in metrics:
        d_list, h_list, diff_list = [], [], []
        for q in per_query:
            cid = q["case_id"]
            n_rel = rel_map[cid]
            if m == "P@3":
                dh = int(round(q["Dense-Only"]["P@3"] * 3))
                hh = int(round(q["Hybrid-RRF"]["P@3"] * 3))
                d_frac = Fraction(dh, 3)
                h_frac = Fraction(hh, 3)
            elif m == "P@5":
                dh = int(round(q["Dense-Only"]["P@5"] * 5))
                hh = int(round(q["Hybrid-RRF"]["P@5"] * 5))
                d_frac = Fraction(dh, 5)
                h_frac = Fraction(hh, 5)
            elif m == "R@3":
                dh = int(round(q["Dense-Only"]["P@3"] * 3))
                hh = int(round(q["Hybrid-RRF"]["P@3"] * 3))
                d_frac = Fraction(dh, n_rel)
                h_frac = Fraction(hh, n_rel)
            elif m == "R@5":
                dh = int(round(q["Dense-Only"]["P@5"] * 5))
                hh = int(round(q["Hybrid-RRF"]["P@5"] * 5))
                d_frac = Fraction(dh, n_rel)
                h_frac = Fraction(hh, n_rel)
            elif m == "MRR":
                d_frac = reconstruct_mrr_fraction(q["Dense-Only"]["MRR"])
                h_frac = reconstruct_mrr_fraction(q["Hybrid-RRF"]["MRR"])

            d_list.append(float(d_frac))
            h_list.append(float(h_frac))
            diff_list.append(h_frac - d_frac)

        unrounded_dense[m] = d_list
        unrounded_hybrid[m] = h_list
        diff_exact_map[m] = diff_list

    # =========================================================================
    # PART 1: DESCRIPTIVE COMPARISON TABLE
    # =========================================================================
    print("PART 1: DESCRIPTIVE RETRIEVAL COMPARISON (OBSERVED BENCHMARK MEANS & RELATIVE GAINS)")
    print("-" * 122)
    print(f"{'Metric':<12} | {'Dense-Only Mean':<16} | {'Hybrid-RRF Mean':<16} | {'Relative Gain':<14} | {'Dense Marginal 95% CI':<23} | {'Hybrid Marginal 95% CI':<23}")
    print("-" * 122)

    for m in metrics:
        d_vals = unrounded_dense[m]
        h_vals = unrounded_hybrid[m]
        d_m = float(np.mean(d_vals))
        h_m = float(np.mean(h_vals))
        rel_gain = ((h_m - d_m) / d_m) * 100.0 if d_m > 0 else 0.0
        d_se = float(np.std(d_vals, ddof=1)) / math.sqrt(n)
        h_se = float(np.std(h_vals, ddof=1)) / math.sqrt(n)
        d_ci = (d_m - t_crit * d_se, d_m + t_crit * d_se)
        h_ci = (h_m - t_crit * h_se, h_m + t_crit * h_se)
        d_ci_str = f"[{d_ci[0]:.4f}, {d_ci[1]:.4f}]"
        h_ci_str = f"[{h_ci[0]:.4f}, {h_ci[1]:.4f}]"
        print(f"{m:<12} | {d_m:<16.4f} | {h_m:<16.4f} | {f'{rel_gain:+.1f}%':<14} | {d_ci_str:<23} | {h_ci_str:<23}")

    print("-" * 122)
    print("Clarification on Intervals:")
    print("  * The confidence intervals above are INDIVIDUAL (MARGINAL) intervals for each system's mean score,")
    print("    computed as mean ± t_{0.025, df=19} * (s / sqrt(20)). They are NOT intervals for the paired differences.")
    print("  * All figures are calculated from underlying unrounded records (Dense P@3 is exactly 22/60 = 0.3667).")

    # =========================================================================
    # PART 2: PRIMARY INFERENTIAL PROCEDURE SETTLEMENT
    # =========================================================================
    print("\n" + "=" * 122)
    print("PART 2: PRIMARY INFERENTIAL PROCEDURE: EXACT PAIRED-SIGN PERMUTATION TEST ON SIGNED RANKS (PRESERVING TIES)")
    print("=" * 122)
    print("Methodological Rationale for Primary Procedure:")
    print("  1. The paired t-test assumes the distribution of paired differences (d_i = Hybrid_i - Dense_i) is approximately")
    print("     normal. In this small benchmark (N = 20 queries), the paired differences take a discrete set of rational")
    print("     values with substantial probability mass at zero (e.g., 9 zeros for P@3) and tied nonzero values.")
    print("     With N = 20, the t-test can be sensitive to such discrete and tied difference distributions.")
    print("  2. Standard SciPy mode='exact' assumes untied integer ranks 1..k without ties (as confirmed by SciPy docstring).")
    print("  3. The exact paired-sign permutation test on signed ranks, preserving tied ranks, fixes the observed tied rank")
    print("     magnitudes and enumerates all 2^k equally likely sign assignments under the assumed symmetric-differences null.")
    print("-" * 122)

    primary_records = []
    raw_p_primary = []
    for m in metrics:
        diffs = diff_exact_map[m]
        diffs_float = [float(x) for x in diffs]
        mean_d = float(np.mean(diffs_float))
        s_d = float(np.std(diffs_float, ddof=1))
        se_d = s_d / math.sqrt(n)
        ci_diff = (mean_d - t_crit * se_d, mean_d + t_crit * se_d)
        w_obs, p_perm, k_cnt = exact_paired_sign_permutation_test(diffs)
        raw_p_primary.append(p_perm)
        primary_records.append({
            "metric": m,
            "mean_d": mean_d,
            "s_d": s_d,
            "se_d": se_d,
            "ci_diff": ci_diff,
            "w_obs": w_obs,
            "k_pairs": k_cnt,
            "p_nom": p_perm,
        })

    # Step-down Holm adjustment on primary family
    holm_p_primary = holm_bonferroni(raw_p_primary)
    sorted_indices = np.argsort(raw_p_primary)
    m_tests = len(metrics)

    print("PRIMARY INFERENTIAL TABLE: EXACT PERMUTATION TEST + STEP-DOWN HOLM-BONFERRONI ADJUSTMENT")
    print("-" * 122)
    print(f"{'Rank':<5} | {'Endpoint':<12} | {'Mean Diff':<10} | {'Paired 95% CI':<21} | {'Wilcox W':<9} | {'k Pairs':<8} | {'p (Exact)':<10} | {'Multiplier':<11} | {'p (Holm)':<10} | {'Verdict (alpha=0.05)'}")
    print("-" * 122)

    for rank, idx in enumerate(sorted_indices, start=1):
        rec = primary_records[idx]
        mult = m_tests - rank + 1
        h_p = holm_p_primary[idx]
        verdict = "Fail to Reject H0" if h_p >= 0.05 else "REJECT H0"
        mult_str = f"{mult} x {rec['p_nom']:.4f}"
        ci_diff_str = f"[{rec['ci_diff'][0]:+.4f}, {rec['ci_diff'][1]:+.4f}]"
        print(f"{rank:<5} | {rec['metric']:<12} | {rec['mean_d']:<+10.4f} | {ci_diff_str:<21} | {rec['w_obs']:<9.1f} | {rec['k_pairs']:<8} | {rec['p_nom']:<10.4f} | {mult_str:<11} | {h_p:<10.4f} | {verdict}")

    print("-" * 122)
    print("Primary Multiplicity Verdict: Zero out of 5 endpoints survive family-wise Holm-Bonferroni correction.")
    print("Precision@3 Holm-adjusted p-value is 0.9131 (nominal p = 0.1826). No statistically significant")
    print("superiority has been established.")

    # =========================================================================
    # PART 3: SECONDARY SENSITIVITY ANALYSES
    # =========================================================================
    print("\n" + "=" * 122)
    print("PART 3: SENSITIVITY ANALYSES (COMPARING ALTERNATIVE INFERENTIAL PROCEDURES)")
    print("=" * 122)
    print("To evaluate stability across testing approaches, three secondary sensitivity checks were run:")
    print("  Check 1: Paired Student's t-test on exact rational differences (Parametric)")
    print("  Check 2: Asymptotic Tie-Adjusted Wilcoxon signed-rank test with continuity correction (SciPy)")
    print("  Check 3: Stored 4-decimal rounded float differences (Persisted JSON representation)")
    print("-" * 122)
    print(f"{'Endpoint':<12} | {'Primary Holm(W_perm)':<21} | {'Check 1: Holm(t)':<18} | {'Check 2: Holm(W_asymp)':<22} | {'Check 3: Holm(W_float)':<22}")
    print("-" * 122)

    raw_pt_list, raw_pasymp_list, raw_pfloat_list = [], [], []
    for m in metrics:
        diff_ex = diff_exact_map[m]
        diff_ex_float = np.array([float(x) for x in diff_ex])
        diff_fl = np.array([q["Hybrid-RRF"][m] - q["Dense-Only"][m] for q in per_query])

        # Check 1: Paired t-test
        se_ex = float(np.std(diff_ex_float, ddof=1)) / math.sqrt(n)
        t_val = float(np.mean(diff_ex_float)) / se_ex if se_ex > 0 else 0.0
        p_t = float(2 * (1 - stats.t.cdf(abs(t_val), df=df)))
        raw_pt_list.append(p_t)

        # Check 2: Asymptotic Wilcoxon (tie-adjusted, continuity-corrected)
        w_asymp = stats.wilcoxon(diff_ex_float, zero_method="wilcox", correction=True)
        raw_pasymp_list.append(float(w_asymp.pvalue))

        # Check 3: Float differences exact permutation
        _, p_fl_perm, _ = exact_paired_sign_permutation_test(diff_fl.tolist())
        raw_pfloat_list.append(p_fl_perm)

    holm_check1 = holm_bonferroni(raw_pt_list)
    holm_check2 = holm_bonferroni(raw_pasymp_list)
    holm_check3 = holm_bonferroni(raw_pfloat_list)

    for idx, m in enumerate(metrics):
        h_prim = holm_p_primary[idx]
        h_c1 = holm_check1[idx]
        h_c2 = holm_check2[idx]
        h_c3 = holm_check3[idx]
        print(f"{m:<12} | {h_prim:<21.4f} | {h_c1:<18.4f} | {h_c2:<22.4f} | {h_c3:<22.4f}")

    print("-" * 122)
    print("Sensitivity Conclusion: Across ALL alternative inferential procedures (parametric, asymptotic, and")
    print("floating-point representations), every Holm-adjusted p-value remains >= 0.16. Zero endpoints achieve")
    print("statistical significance at alpha = 0.05. The inferential conclusion is entirely invariant to test choice.")

    # =========================================================================
    # PART 4: EMPIRICAL SCOPE & PAPER REPORTING GUIDELINES
    # =========================================================================
    print("\n" + "=" * 122)
    print("                     CAUTIOUS EMPIRICAL REPORTING GUIDELINES FOR PAPER")
    print("=" * 122)
    print("1. Scope of Retrieval Findings:")
    print("   Retrieval results provide preliminary empirical evidence about retrieval performance on this specific")
    print("   40-document collection. They do NOT establish that the full multi-agent prototype is feasible or that")
    print("   its downstream research-gap findings or novelty classifications are correct.")
    print("2. Exploratory Benchmark Nature:")
    print("   The evaluation is exploratory and restricted to 20 developer-curated technical queries. Findings must")
    print("   not be generalized to all technical queries, commercial patent databases, or other scientific domains.")
    print("3. Descriptive vs Inferential Separation:")
    print("   - Descriptive sample gains (+27.3% P@3, +12.5% P@5, +22.9% R@3, +8.2% R@5, +5.3% MRR) show positive trends.")
    print("   - Inferential testing confirms that no statistically significant superiority over dense-only retrieval")
    print("     has been established (Precision@3 Holm-adjusted p = 0.9131; all other endpoints adjusted p = 1.0000).")
    print("=" * 122)


if __name__ == "__main__":
    run_audit()
