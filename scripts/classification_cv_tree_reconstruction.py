#!/usr/bin/env python3
"""
Classification CV to predicted distance matrix to UPGMA tree vs reference phylogeny

Standalone pipeline (does not import pairwise_distance_classification_oversampling).
Logic is aligned with:
  - scripts/pairwise_distance_classification_oversampling.py (data, features, extended CV)
  - scripts/species_blocked_rf_quantile_tree.py (UPGMA Newick)

Default model: SCHEME_4B + RandomOverSampler + RandomForest, species-level **StratifiedKFold**
(deep vs non-deep: Marsupialia + Afrotheria) so deep-divergence taxa are spread across folds,
train on train-train pairs only, predict test-test ∪ test-train, aggregate mean
predict_proba per unordered pair. Class->MY mapping uses per-fold means of patristic
distance on train-train pairs only, then averaged across folds.

Outputs: results/classification_cv_tree_reconstruction/ (or --output-dir / --run-id).

Run from project root:
  python3 scripts/classification_cv_tree_reconstruction.py

Requires: numpy, pandas, scipy, scikit-learn, matplotlib, ete3, imbalanced-learn.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import traceback
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

matplotlib = __import__("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

try:
    from ete3 import Tree
except ImportError as exc:
    raise SystemExit("This script requires ete3. Install with: pip install ete3") from exc

try:
    from imblearn.over_sampling import RandomOverSampler
except ImportError as exc:
    raise SystemExit(
        "This script requires imbalanced-learn (RandomOverSampler). "
        "Install with: pip install imbalanced-learn"
    ) from exc

# -----------------------------------------------------------------------------
# Paths (aligned with pairwise_distance_classification_oversampling.py)
# -----------------------------------------------------------------------------

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPTS_DIR)

DATA_RAW = os.path.join(PROJECT_ROOT, "data_raw")
TREE_PATH = os.path.join(PROJECT_ROOT, "MammalsPhylogeny.nwk")
DOMAIN_COUNTS_PATH = os.path.join(DATA_RAW, "MammalDomainCount.tsv")
SPECIES_LIST_PATH = os.path.join(DATA_RAW, "MammalsList.txt")

DEFAULT_OUTPUT_PARENT = os.path.join(PROJECT_ROOT, "results", "classification_cv_tree_reconstruction")

NORMALIZATION_EPS = 1e-12
EXCLUDE_PLATYPUS = True

RANDOM_STATE = 42
N_FOLDS_DEFAULT = 5
N_ESTIMATORS_DEFAULT = 200

TREE_NAME_MAP = {
    "Neogale vison": "Neovison vison",
    "Neogale_vison": "Neovison vison",
    "Bos grunniens": "Bos mutus grunniens",
    "Bos_grunniens": "Bos mutus grunniens",
    "Physeter catodon": "Physeter macrocephalus",
    "Physeter_catodon": "Physeter macrocephalus",
}

# Marsupialia + Afrotheria (aligned with elastic_net_regression_stratified_species_cv.py
# CLADE_TO_SPECIES_UNDERSCORE); converted to space-separated binomials, intersected with data.
DEEP_SPECIES_UNDERSCORE: Tuple[str, ...] = (
    "Monodelphis_domestica",
    "Sarcophilus_harrisii",
    "Vombatus_ursinus",
    "Phascolarctos_cinereus",
    "Loxodonta_africana",
    "Trichechus_manatus_latirostris",
    "Chrysochloris_asiatica",
    "Orycteropus_afer_afer",
)


def deep_species_present(species_list: List[str]) -> Tuple[frozenset, List[str]]:
    """
    Return (frozenset of deep species names present in species_list, sorted).
    Tries exact space form from underscore list; if missing, tries unique prefix match
    (e.g. Trichechus binomial if subspecies string absent).
    """
    species_set = set(species_list)
    found: List[str] = []
    missing_log: List[str] = []
    for us in DEEP_SPECIES_UNDERSCORE:
        spaced = us.replace("_", " ")
        if spaced in species_set:
            found.append(spaced)
            continue
        tokens = us.split("_")
        if len(tokens) >= 2:
            binomial = f"{tokens[0]} {tokens[1]}"
            candidates = [s for s in species_list if s == binomial or s.startswith(binomial + " ")]
            if len(candidates) == 1:
                found.append(candidates[0])
            elif len(candidates) > 1:
                pick = sorted(candidates)[0]
                log(f"  Deep taxon ambiguous for {us}: picked '{pick}' among {candidates}")
                found.append(pick)
            else:
                missing_log.append(us)
        else:
            missing_log.append(us)
    if missing_log:
        log(f"  Warning: deep clade species not in current species list: {missing_log}")
    found_unique = sorted(set(found))
    return frozenset(found_unique), found_unique


def log(msg: str) -> None:
    print(f"[cv_tree] {msg}")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def format_species_label_for_newick(species_name: str) -> str:
    s = species_name.replace(" ", "_").replace("(", "").replace(")", "")
    return s.replace(",", "").replace(";", "")


def build_scheme_definitions() -> Dict[str, Dict[str, Any]]:
    return {
        "SCHEME_4B": {"cutoffs": [0, 150, 175, 190, np.inf], "labels": ["D0", "D1", "D2", "D3"]},
        "SCHEME_4C": {"cutoffs": [0, 150, 180, 195, np.inf], "labels": ["D0", "D1", "D2", "D3"]},
    }


def load_species_list(path: str) -> List[str]:
    species: List[str] = []
    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                species.append(stripped)
    return species


def load_domain_counts(path: str, species_order: List[str]) -> pd.DataFrame:
    log(f"Loading domain counts from {path}")
    df = pd.read_csv(path, sep="\t", skiprows=1, index_col=0, low_memory=False)
    X_raw = df.transpose()
    X_raw.index.name = "species"
    X_raw.index = [name.split("(")[0].strip() for name in X_raw.index]

    target_set = set(species_order)
    mapping: Dict[str, str] = {}
    for label in X_raw.index:
        if label in target_set:
            mapping[label] = label
        else:
            for sp in species_order:
                if label.lower() == sp.lower():
                    mapping[label] = sp
                    break

    mapped_labels = [lab for lab in X_raw.index if lab in mapping]
    X_raw = X_raw.loc[mapped_labels]
    X_raw.index = [mapping[lab] for lab in X_raw.index]

    if X_raw.index.duplicated().any():
        log("  Warning: duplicate species after mapping; averaging.")
        X_raw = X_raw.groupby(X_raw.index).mean()

    missing = set(species_order) - set(X_raw.index)
    if missing:
        log(f"  Warning: {len(missing)} species missing from domain counts (ignored).")
        species_order = [sp for sp in species_order if sp in X_raw.index]

    X_raw = X_raw.loc[species_order]
    log(f"  Loaded {X_raw.shape[0]} species x {X_raw.shape[1]} domains")
    return X_raw


def compute_patristic_matrix(tree_path: str, species_list: List[str]) -> pd.DataFrame:
    log("Computing patristic distance matrix from tree...")
    tree = Tree(tree_path, format=1)
    species_set = set(species_list)
    tree_species_map: Dict[str, str] = {}

    for leaf in tree.iter_leaves():
        leaf_name_orig = leaf.name
        leaf_name = leaf_name_orig.replace("_", " ")

        if leaf_name_orig in TREE_NAME_MAP:
            mapped = TREE_NAME_MAP[leaf_name_orig]
        elif leaf_name in TREE_NAME_MAP:
            mapped = TREE_NAME_MAP[leaf_name]
        else:
            mapped = leaf_name

        if mapped in species_set:
            tree_species_map[leaf_name_orig] = mapped
        else:
            for sp in species_list:
                if mapped.lower() == sp.lower():
                    tree_species_map[leaf_name_orig] = sp
                    break

    for leaf in list(tree.iter_leaves()):
        if leaf.name in tree_species_map:
            leaf.name = tree_species_map[leaf.name]
        else:
            leaf.detach()

    available_species = [sp for sp in species_list if sp in {x.name for x in tree.iter_leaves()}]
    tree.prune(available_species, preserve_branch_length=True)

    n = len(available_species)
    T_matrix = np.zeros((n, n), dtype=float)
    for i, sp_i in enumerate(available_species):
        for j in range(i + 1, n):
            sp_j = available_species[j]
            dist = tree.get_distance(sp_i, sp_j)
            T_matrix[i, j] = T_matrix[j, i] = dist

    T_df = pd.DataFrame(T_matrix, index=available_species, columns=available_species)
    log(f"  Patristic matrix: {n} species")
    return T_df


def load_and_preprocess_data() -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    log("=" * 72)
    log("Load and preprocess (domain counts + tree)")
    log("=" * 72)

    species_from_list = load_species_list(SPECIES_LIST_PATH)
    X_counts = load_domain_counts(DOMAIN_COUNTS_PATH, species_from_list)
    domain_species = list(X_counts.index)

    dist_matrix_full = compute_patristic_matrix(TREE_PATH, domain_species)
    tree_species = list(dist_matrix_full.index)

    species_intersection = sorted(set(domain_species) & set(tree_species))
    X_counts = X_counts.loc[species_intersection]
    dist_matrix = dist_matrix_full.loc[species_intersection, species_intersection]

    if EXCLUDE_PLATYPUS:
        outgroup_name = "Ornithorhynchus anatinus"
        if outgroup_name in X_counts.index:
            log(f"  Excluding outgroup: {outgroup_name}")
            X_counts = X_counts.drop(index=outgroup_name)
            dist_matrix = dist_matrix.drop(index=outgroup_name, columns=outgroup_name)

    species_final = list(X_counts.index)

    domain_variance = X_counts.var(axis=0)
    zero_var_mask = domain_variance == 0
    if zero_var_mask.any():
        log(f"  Dropping {int(zero_var_mask.sum())} zero-variance domains")
        X_counts = X_counts.loc[:, ~zero_var_mask]

    row_sums = X_counts.sum(axis=1).values.reshape(-1, 1)
    X_norm_arr = X_counts.values.astype(float) / (row_sums + NORMALIZATION_EPS)
    X_norm = pd.DataFrame(X_norm_arr, index=X_counts.index, columns=X_counts.columns)

    log(f"  Final species: {len(species_final)}; domains: {X_norm.shape[1]}")
    return X_norm, dist_matrix, species_final


def construct_pairwise_features(
    X_norm: pd.DataFrame,
    dist_matrix: pd.DataFrame,
    species_list: List[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    species_list = sorted(species_list)
    n_species = len(species_list)
    n_domains = X_norm.shape[1]

    if set(species_list) - set(dist_matrix.index):
        raise ValueError("Species missing from distance matrix")

    X_subset = X_norm.loc[species_list].values
    dist_subset = dist_matrix.loc[species_list, species_list].values

    n_pairs = n_species * (n_species - 1) // 2
    X_pairs = np.zeros((n_pairs, n_domains), dtype=float)

    for d in range(n_domains):
        col = X_subset[:, d].reshape(-1, 1)
        X_pairs[:, d] = pdist(col, metric="cityblock")

    i_idx, j_idx = np.triu_indices(n_species, k=1)
    y_dist = dist_subset[i_idx, j_idx]
    pair_sp1 = np.array([species_list[i] for i in i_idx])
    pair_sp2 = np.array([species_list[j] for j in j_idx])

    if np.isnan(X_pairs).any() or np.isinf(X_pairs).any():
        raise ValueError("NaN/inf in feature matrix")
    if np.isnan(y_dist).any() or np.isinf(y_dist).any():
        raise ValueError("NaN/inf in distance vector")

    return X_pairs, y_dist, pair_sp1, pair_sp2


def assign_classes(
    y_dist: np.ndarray,
    cutoffs: List[float],
    labels: List[str],
) -> Tuple[np.ndarray, int, List[int]]:
    n_classes = len(labels)
    if len(cutoffs) != n_classes + 1:
        raise ValueError("cutoffs must have length n_classes + 1")

    y_class = pd.cut(
        y_dist,
        bins=cutoffs,
        right=False,
        include_lowest=True,
        labels=False,
    )
    y_arr = np.asarray(y_class)
    if pd.isna(y_arr).any():
        raise ValueError("Some distances did not fall into manual bin edges.")

    y_arr = y_arr.astype(np.intp)
    class_counts = [int((y_arr == k).sum()) for k in range(n_classes)]
    return y_arr, n_classes, class_counts


def build_stratified_deep_species_fold_indices(
    species_list: List[str],
    pair_sp1: np.ndarray,
    pair_sp2: np.ndarray,
    n_folds: int,
    random_state: int,
) -> Tuple[List[Tuple[np.ndarray, np.ndarray, frozenset, frozenset]], frozenset]:
    """
    StratifiedKFold on species with label deep=1 / non-deep=0 (Marsupialia + Afrotheria).
    Extended test: train=train-train; test=any pair with >=1 test species.
    Returns (folds, deep_species_used).
    """
    species_arr = np.array(sorted(species_list))
    deep_set, _deep_list = deep_species_present(species_list)
    y_strat = np.array([1 if s in deep_set else 0 for s in species_arr], dtype=int)
    n_deep = int(y_strat.sum())
    n_nondeep = int(len(y_strat) - n_deep)
    if n_deep < n_folds or n_nondeep < n_folds:
        raise SystemExit(
            f"StratifiedKFold requires at least n_folds={n_folds} samples per class; "
            f"got deep={n_deep}, non_deep={n_nondeep}. Check species list and deep taxa mapping."
        )

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    X_dummy = np.zeros((len(species_arr), 1), dtype=float)
    folds: List[Tuple[np.ndarray, np.ndarray, frozenset, frozenset]] = []

    try:
        splits = list(skf.split(X_dummy, y_strat))
    except ValueError as e:
        raise SystemExit(f"StratifiedKFold failed: {e}") from e

    for fold_i, (tr_sp_idx, te_sp_idx) in enumerate(splits, start=1):
        train_species = frozenset(species_arr[tr_sp_idx])
        test_species = frozenset(species_arr[te_sp_idx])
        if train_species & test_species:
            raise RuntimeError("Train/test species overlap")

        n_deep_train = sum(1 for s in train_species if s in deep_set)
        n_deep_test = sum(1 for s in test_species if s in deep_set)
        log(
            f"  Fold {fold_i}: test species n={len(test_species)} "
            f"(deep in test={n_deep_test}, deep in train={n_deep_train})"
        )

        train_mask = np.array(
            [(a in train_species and b in train_species) for a, b in zip(pair_sp1, pair_sp2)],
            dtype=bool,
        )
        test_mask = np.array(
            [(a in test_species or b in test_species) for a, b in zip(pair_sp1, pair_sp2)],
            dtype=bool,
        )
        folds.append((np.where(train_mask)[0], np.where(test_mask)[0], train_species, test_species))

    return folds, deep_set


def canonical_pair_key(sp1: str, sp2: str) -> Tuple[str, str]:
    return (sp1, sp2) if sp1 <= sp2 else (sp2, sp1)


def align_proba_to_full_matrix(
    clf: RandomForestClassifier,
    proba: np.ndarray,
    n_classes: int,
) -> np.ndarray:
    n_samples = proba.shape[0]
    out = np.zeros((n_samples, n_classes), dtype=float)
    for j, c in enumerate(clf.classes_):
        out[:, int(c)] = proba[:, j]
    return out


def compute_fold_train_class_means(
    tr_idx: np.ndarray,
    y_class: np.ndarray,
    y_dist: np.ndarray,
    n_classes: int,
) -> np.ndarray:
    """Mean patristic MY per class on train-train pairs only; NaN if class absent."""
    means = np.full(n_classes, np.nan, dtype=float)
    yt = y_class[tr_idx].astype(int)
    yd = y_dist[tr_idx]
    for k in range(n_classes):
        m = yt == k
        if np.any(m):
            means[k] = float(np.mean(yd[m]))
    return means


def fit_fold_random_oversample_rf(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_classes: int,
    random_state: int,
    n_estimators: int,
) -> Optional[Tuple[np.ndarray, RandomForestClassifier]]:
    """
    StandardScaler on train; RandomOverSampler on scaled train; RF fit; predict_proba on test.
    Returns (proba_full, clf) or None if fold invalid.
    """
    if len(np.unique(y_train)) < 2:
        log("    Skipping fold: training has <2 classes.")
        return None

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    try:
        ros = RandomOverSampler(random_state=random_state)
        X_tr_s, y_tr = ros.fit_resample(X_train_s, y_train)
    except Exception as e:
        log(f"    RandomOverSampler failed: {e}\n{traceback.format_exc()}")
        return None

    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=None,
        random_state=random_state,
        class_weight=None,
        n_jobs=-1,
    )
    clf.fit(X_tr_s, y_tr)
    proba_raw = clf.predict_proba(X_test_s)
    proba_full = align_proba_to_full_matrix(clf, proba_raw, n_classes)
    y_pred = clf.predict(X_test_s)
    label_idx = list(range(n_classes))
    bal = float(recall_score(y_test, y_pred, labels=label_idx, average="macro", zero_division=0))
    macro = float(f1_score(y_test, y_pred, labels=label_idx, average="macro", zero_division=0))
    log(f"    Fold diagnostic: balanced_acc(macro-recall)={bal:.4f}, macro_f1={macro:.4f}")
    return proba_full, clf


def average_linkage_matrix(distance_values: np.ndarray) -> np.ndarray:
    """Average linkage (UPGMA) Z matrix from symmetric distance array."""
    condensed = squareform(distance_values, checks=False)
    return hierarchy.average(condensed)


def save_upgma_dendrogram_pdf(
    D: np.ndarray,
    species_ordered: List[str],
    pdf_path: str,
) -> None:
    """Dendrogram from same average linkage as UPGMA Newick; wide figure for many leaves."""
    Z = average_linkage_matrix(D)
    fig, ax = plt.subplots(figsize=(28, 10))
    hierarchy.dendrogram(
        Z,
        labels=species_ordered,
        ax=ax,
        leaf_rotation=90.0,
        leaf_font_size=3,
        color_threshold=0,
    )
    ax.set_title("UPGMA (average linkage) - predicted pairwise distances (MY)")
    plt.tight_layout()
    plt.savefig(pdf_path, format="pdf", bbox_inches="tight")
    plt.close()


def build_upgma_newick(distance_df: pd.DataFrame) -> str:
    """UPGMA via average linkage (same pattern as species_blocked_rf_quantile_tree.py)."""
    species = list(distance_df.index)
    labels = [format_species_label_for_newick(s) for s in species]
    linkage_matrix = average_linkage_matrix(distance_df.values)
    tree = hierarchy.to_tree(linkage_matrix, rd=False)

    def build_newick(node) -> Tuple[str, float]:
        if node.is_leaf():
            return labels[node.id], 0.0
        left_str, left_height = build_newick(node.left)
        right_str, right_height = build_newick(node.right)
        branch_left = max(node.dist - left_height, 0.0)
        branch_right = max(node.dist - right_height, 0.0)
        newick = f"({left_str}:{branch_left:.10f},{right_str}:{branch_right:.10f})"
        return newick, node.dist

    newick_str, _ = build_newick(tree)
    return f"{newick_str};"


def load_and_prune_reference_tree(species_list: List[str]) -> Tree:
    """Reference tree with leaves renamed to match species_list (space-separated names)."""
    ref = Tree(TREE_PATH, format=1)
    species_set = set(species_list)
    for leaf in list(ref.iter_leaves()):
        orig = leaf.name
        name = orig.replace("_", " ")
        if orig in TREE_NAME_MAP:
            name = TREE_NAME_MAP[orig]
        elif name in TREE_NAME_MAP:
            name = TREE_NAME_MAP[name]
        if name in species_set:
            leaf.name = name
        else:
            leaf.detach()
    ref.prune(species_list, preserve_branch_length=True)
    leaves = {leaf.name for leaf in ref.iter_leaves()}
    if leaves != species_set:
        raise RuntimeError(f"Reference prune mismatch: extra={leaves - species_set}, missing={species_set - leaves}")
    return ref


def reference_tree_to_formatted_newick(ref_tree: Tree) -> str:
    """Copy tree with Newick-safe leaf names for side-by-side comparison files."""
    t = copy.deepcopy(ref_tree)
    for leaf in t.iter_leaves():
        leaf.name = format_species_label_for_newick(leaf.name)
    return t.write(format=1)


def validate_distance_matrix(D: np.ndarray, species_ordered: List[str]) -> None:
    n = D.shape[0]
    if D.shape != (n, n):
        raise ValueError("Distance matrix must be square")
    if not np.allclose(D, D.T, rtol=0, atol=1e-9):
        raise ValueError("Distance matrix is not symmetric")
    if not np.allclose(np.diag(D), 0.0, rtol=0, atol=1e-9):
        raise ValueError("Diagonal is not zero")
    if len(species_ordered) != n:
        raise ValueError("Species order length != matrix dimension")
    log("  Validation OK: symmetric matrix, zero diagonal, dimensions match species count.")


def write_methodology_md(
    path: str,
    scheme_name: str,
    cutoffs: List[float],
    labels: List[str],
    n_folds: int,
    n_estimators: int,
    deep_species_sorted: List[str],
) -> None:
    deep_bullet = "\n".join(f"  - `{s}`" for s in deep_species_sorted) if deep_species_sorted else "  - (none)"
    text = f"""# Methodology: classification CV -> tree reconstruction

## Goal
Species-blocked cross-validation with **no leakage**: each fold trains only on **train-train**
pairs (both species in the training species set). Test pairs are **test-test ∪ test-train**
(any pair involving at least one held-out species). **StandardScaler** is fit on training
features only; the same scaler transforms all test rows. **RandomOverSampler** is applied
only to scaled training data.

## Model (best-performing setup from oversampling study)
- Binning: **{scheme_name}** with cutoffs `{cutoffs}` and class labels `{labels}`.
- Classifier: **RandomForestClassifier** (`n_estimators={n_estimators}`, `class_weight=None`).
- Resampling: **RandomOverSampler** (imbalanced-learn) after scaling, training only.
- CV: **StratifiedKFold** on species (`n_splits={n_folds}`, shuffled, fixed random seed), stratifying
  **deep** (Marsupialia + Afrotheria) vs **non-deep** so each fold's held-out set contains a
  balanced mix; this keeps deep-divergence taxa represented in the **training** portion of every fold
  and stabilizes train-only class centroids (especially the D3 bin).
- Deep species labels used in this run (intersected with data):
{deep_bullet}

## Aggregating predictions
Each unordered species pair may receive predictions from multiple folds (when the two species
fall in different test folds). For each pair we average **predicted class probabilities**
across folds that included that pair as a test example, then take **argmax** to obtain one
predicted class per pair.

## Class -> numeric distance (training-based centroids)
For each fold `f`, on **train-train** pairs only, we compute

  μ[k,f] = mean(patristic distance (MY) among training pairs with true class k.

For each class `k`, the representative distance is

  μ_k = mean over folds f of μ[k,f],

using only folds where at least one train-train pair had class k (NaN folds omitted from the mean).

**Rationale:** μ_k is estimated from **training** pairs within each fold, so true distances of
held-out pairs do not define the mapping from predicted class to MY. This avoids using
test-set patristic distances to set the scale of predicted distances.

**Contrast:** A simpler alternative is the global mean patristic distance per class over **all**
unordered pairs (used in some exploratory distance-matrix exports). That uses the full dataset
and is slightly less conservative for a CV narrative; this script uses **train-only fold centroids**
averaged across folds.

## Tree reconstruction
**UPGMA** (average linkage / `scipy.cluster.hierarchy.average`) on the predicted symmetric
distance matrix, converted to Newick with branch lengths.

## Comparison metrics
- **Robinson-Foulds** distance between unrooted reference and predicted trees (`ete3`), on the
  same set of taxa (leaf labels normalized for Newick).
- **Pearson / Spearman** correlation between upper-triangle true patristic MY and predicted MY
  (same pair ordering as feature construction).
"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("Run from")[0].strip())
    p.add_argument(
        "--output-dir",
        default=None,
        help=f"Output directory (default: {DEFAULT_OUTPUT_PARENT} or subfolder with --run-id).",
    )
    p.add_argument("--run-id", default=None, help="Optional subfolder name under default parent.")
    p.add_argument("--n-estimators", type=int, default=N_ESTIMATORS_DEFAULT)
    p.add_argument("--n-folds", type=int, default=N_FOLDS_DEFAULT)
    p.add_argument("--random-state", type=int, default=RANDOM_STATE)
    p.add_argument(
        "--scheme",
        default="SCHEME_4B",
        choices=["SCHEME_4B", "SCHEME_4C"],
        help="Manual binning scheme (default SCHEME_4B).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.n_folds < 2:
        raise SystemExit("--n-folds must be at least 2")

    schemes = build_scheme_definitions()
    if args.scheme not in schemes:
        raise SystemExit(f"Unknown scheme {args.scheme}")
    spec = schemes[args.scheme]
    cutoffs = [float(x) for x in spec["cutoffs"]]
    class_labels = list(spec["labels"])

    if args.output_dir:
        out_dir = os.path.abspath(args.output_dir)
    elif args.run_id:
        out_dir = os.path.join(DEFAULT_OUTPUT_PARENT, args.run_id)
    else:
        out_dir = DEFAULT_OUTPUT_PARENT

    ensure_dir(out_dir)
    log(f"Output directory: {out_dir}")

    X_norm, dist_matrix, species_final = load_and_preprocess_data()
    species_ordered = sorted(species_final)
    if species_final != species_ordered:
        log("  Note: reordering to sorted species order for matrix consistency with features.")
    X, y_dist, pair_sp1, pair_sp2 = construct_pairwise_features(X_norm, dist_matrix, species_ordered)

    y_class, n_classes, class_counts = assign_classes(y_dist, cutoffs, class_labels)
    log(f"Scheme {args.scheme}: class counts on all pairs: {dict(zip(class_labels, class_counts))}")

    n_pairs_expected = len(y_dist)
    n_species = len(species_ordered)
    if n_pairs_expected != n_species * (n_species - 1) // 2:
        raise RuntimeError("Pair count mismatch")

    folds, deep_used = build_stratified_deep_species_fold_indices(
        species_ordered, pair_sp1, pair_sp2, args.n_folds, args.random_state
    )
    deep_sorted = sorted(deep_used)
    log(f"Stratified CV: {len(deep_sorted)} deep species in analysis: {deep_sorted}")

    proba_sum: Dict[Tuple[str, str], np.ndarray] = defaultdict(
        lambda: np.zeros(n_classes, dtype=float)
    )
    proba_count: Dict[Tuple[str, str], int] = defaultdict(int)
    y_true_by_key: Dict[Tuple[str, str], int] = {}

    fold_means_list: List[np.ndarray] = []
    fold_rows_diag: List[Dict[str, Any]] = []

    for fold_i, (tr_idx, te_idx, train_sp, test_sp) in enumerate(folds, start=1):
        log(f"--- Fold {fold_i}/{args.n_folds} ---")
        means_f = compute_fold_train_class_means(tr_idx, y_class, y_dist, n_classes)
        fold_means_list.append(means_f)
        fold_rows_diag.append(
            {
                "fold": fold_i,
                **{f"mu_{class_labels[k]}_MY": means_f[k] for k in range(n_classes)},
            }
        )

        X_train, y_train = X[tr_idx], y_class[tr_idx]
        X_test, y_test = X[te_idx], y_class[te_idx]

        result = fit_fold_random_oversample_rf(
            X_train,
            y_train,
            X_test,
            y_test,
            n_classes,
            random_state=args.random_state + fold_i,
            n_estimators=args.n_estimators,
        )
        if result is None:
            raise RuntimeError(f"Fold {fold_i} failed; cannot build full prediction table.")

        proba_full, _clf = result
        for row_i, gi in enumerate(te_idx):
            sp1, sp2 = str(pair_sp1[gi]), str(pair_sp2[gi])
            key = canonical_pair_key(sp1, sp2)
            proba_sum[key] += proba_full[row_i]
            proba_count[key] += 1
            yt = int(y_class[gi])
            if key in y_true_by_key and y_true_by_key[key] != yt:
                raise RuntimeError(f"Inconsistent y_true for {key}")
            y_true_by_key[key] = yt

    if len(y_true_by_key) != n_pairs_expected:
        raise RuntimeError(
            f"Coverage: {len(y_true_by_key)} unique pairs, expected {n_pairs_expected}"
        )

    # Per-fold train centroids -> CSV
    fold_mu_df = pd.DataFrame(fold_rows_diag)
    fold_mu_path = os.path.join(out_dir, "class_representative_distances_per_fold.csv")
    fold_mu_df.to_csv(fold_mu_path, index=False)
    log(f"Saved {fold_mu_path}")

    fold_stack = np.vstack(fold_means_list)
    mu_k = np.zeros(n_classes, dtype=float)
    for k in range(n_classes):
        col = fold_stack[:, k]
        valid = ~np.isnan(col)
        if not np.any(valid):
            raise RuntimeError(f"No train-train examples for class {k} ({class_labels[k]}) in any fold.")
        mu_k[k] = float(np.mean(col[valid]))

    rep_rows = [
        {"class_index": k, "label": class_labels[k], "mu_k_MY": mu_k[k]} for k in range(n_classes)
    ]
    rep_df = pd.DataFrame(rep_rows)
    rep_path = os.path.join(out_dir, "class_representative_distances.csv")
    rep_df.to_csv(rep_path, index=False)
    log(f"Saved {rep_path} (μ_k = mean over folds of train-only class means)")

    # Aggregated predictions + matrix
    agg_rows: List[Dict[str, Any]] = []
    D = np.zeros((n_species, n_species), dtype=float)
    sp_to_i = {s: i for i, s in enumerate(species_ordered)}

    for i in range(n_species):
        for j in range(i + 1, n_species):
            sp_i, sp_j = species_ordered[i], species_ordered[j]
            key = canonical_pair_key(sp_i, sp_j)
            mean_p = proba_sum[key] / float(proba_count[key])
            pred_c = int(np.argmax(mean_p))
            d_ij = mu_k[pred_c]
            D[i, j] = D[j, i] = d_ij
            agg_rows.append(
                {
                    "sp1": key[0],
                    "sp2": key[1],
                    "n_folds_averaged": proba_count[key],
                    "y_true": y_true_by_key[key],
                    "y_pred": pred_c,
                    "y_true_label": class_labels[y_true_by_key[key]],
                    "y_pred_label": class_labels[pred_c],
                    "predicted_distance_MY": d_ij,
                }
            )

    pred_csv = os.path.join(out_dir, "predictions_aggregated.csv")
    pd.DataFrame(agg_rows).to_csv(pred_csv, index=False)
    log(f"Saved {pred_csv} ({len(agg_rows)} pairs)")

    validate_distance_matrix(D, species_ordered)

    D_df = pd.DataFrame(D, index=species_ordered, columns=species_ordered)
    D_df.to_csv(os.path.join(out_dir, "distance_matrix_predicted.csv"))
    D_df.to_csv(os.path.join(out_dir, "distance_matrix_predicted.tsv"), sep="\t")
    np.save(os.path.join(out_dir, "distance_matrix_predicted.npy"), D)
    log("Saved distance_matrix_predicted.csv, .tsv, .npy")

    # True upper triangle (same order as y_dist)
    true_vec = y_dist.astype(float)
    pred_vec = np.array(
        [
            D[sp_to_i[pair_sp1[g]] , sp_to_i[pair_sp2[g]]]
            for g in range(n_pairs_expected)
        ],
        dtype=float,
    )
    r_p, p_p = pearsonr(true_vec, pred_vec)
    r_s, p_s = spearmanr(true_vec, pred_vec)
    log(f"Pairwise vector correlation (true vs predicted MY): Pearson r={r_p:.4f}, Spearman r={r_s:.4f}")

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(true_vec, pred_vec, alpha=0.25, s=8, c="steelblue")
    ax.set_xlabel("True patristic distance (MY)")
    ax.set_ylabel("Predicted distance (MY, class centroid)")
    ax.set_title(f"Upper-triangle pairs ({args.scheme}, CV aggregated)")
    lim = max(float(true_vec.max()), float(pred_vec.max())) * 1.05
    ax.plot([0, lim], [0, lim], "k--", alpha=0.4, label="y=x")
    ax.legend()
    ax.set_aspect("equal", adjustable="box")
    plt.tight_layout()
    scatter_path = os.path.join(out_dir, "scatter_true_vs_predicted_distances.png")
    plt.savefig(scatter_path, dpi=150)
    plt.close()
    log(f"Saved {scatter_path}")

    fig2, ax2 = plt.subplots(figsize=(8, 7))
    im = ax2.imshow(D, cmap="viridis", aspect="equal")
    ax2.set_xticks(range(n_species))
    ax2.set_yticks(range(n_species))
    ax2.set_xticklabels(species_ordered, rotation=90, fontsize=4)
    ax2.set_yticklabels(species_ordered, fontsize=4)
    ax2.set_title("Predicted distance matrix (MY)")
    plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    plt.tight_layout()
    heat_path = os.path.join(out_dir, "heatmap_predicted_distance_matrix.png")
    plt.savefig(heat_path, dpi=150)
    plt.close()
    log(f"Saved {heat_path}")

    newick_pred = build_upgma_newick(D_df)
    pred_tree_path = os.path.join(out_dir, "predicted_tree_UPGMA.nwk")
    with open(pred_tree_path, "w", encoding="utf-8") as f:
        f.write(newick_pred)
    log(f"Saved {pred_tree_path}")

    pdf_tree_path = os.path.join(out_dir, "predicted_tree_UPGMA.pdf")
    save_upgma_dendrogram_pdf(D, species_ordered, pdf_tree_path)
    log(f"Saved {pdf_tree_path}")

    ref_tree = load_and_prune_reference_tree(species_ordered)
    ref_nwk = reference_tree_to_formatted_newick(ref_tree)
    ref_path = os.path.join(out_dir, "reference_tree_pruned.nwk")
    with open(ref_path, "w", encoding="utf-8") as f:
        f.write(ref_nwk)
    log(f"Saved {ref_path}")

    tree_pred = Tree(newick_pred, format=1)
    tree_ref_fmt = Tree(ref_nwk, format=1)
    leaves_pred = {format_species_label_for_newick(x.name) for x in tree_pred.iter_leaves()}
    leaves_ref = {format_species_label_for_newick(x.name) for x in tree_ref_fmt.iter_leaves()}
    expected = {format_species_label_for_newick(s) for s in species_ordered}
    if leaves_pred != expected or leaves_ref != expected:
        raise RuntimeError(
            f"Leaf set mismatch: pred={leaves_pred ^ expected}, ref={leaves_ref ^ expected}"
        )

    rf_result = tree_ref_fmt.robinson_foulds(tree_pred, unrooted_trees=True)
    rf_dist = int(rf_result[0])
    rf_max = int(rf_result[1])
    rf_norm = float(rf_dist / rf_max) if rf_max > 0 else float("nan")

    def _json_float(x: float) -> Any:
        xf = float(x)
        if np.isnan(xf) or np.isinf(xf):
            return None
        return xf

    summary = {
        "scheme": args.scheme,
        "cutoffs": [float(c) if not np.isinf(c) else "inf" for c in cutoffs],
        "class_labels": class_labels,
        "n_species": n_species,
        "n_pairs": n_pairs_expected,
        "n_folds": args.n_folds,
        "n_estimators": args.n_estimators,
        "random_state": args.random_state,
        "class_representative_MY_mu_k": {class_labels[k]: float(mu_k[k]) for k in range(n_classes)},
        "pearson_r": float(r_p),
        "pearson_p": _json_float(p_p),
        "spearman_r": float(r_s),
        "spearman_p": _json_float(p_s),
        "pearson_true_vs_pred_upper_triangle": float(r_p),
        "pearson_p_value": _json_float(p_p),
        "spearman_true_vs_pred_upper_triangle": float(r_s),
        "spearman_p_value": _json_float(p_s),
        "robinson_foulds_distance": rf_dist,
        "robinson_foulds_max": rf_max,
        "robinson_foulds_normalized": _json_float(rf_norm),
    }

    summary_path = os.path.join(out_dir, "comparison_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, allow_nan=False)
    log(f"Saved {summary_path}")

    report_path = os.path.join(out_dir, "comparison_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Predicted UPGMA tree vs reference (pruned, same taxa)\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Scheme: {args.scheme}\n")
        f.write(f"Species: {n_species}; unordered pairs: {n_pairs_expected}\n\n")
        f.write("Upper-triangle patristic MY vs predicted MY (class centroids from train-only CV):\n")
        f.write(f"  pearson_r = {r_p:.6f}, pearson_p = {p_p:.6e}\n")
        f.write(f"  spearman_r = {r_s:.6f}, spearman_p = {p_s:.6e}\n")
        f.write(f"  (same vectors as scatter; legacy keys in JSON: pearson_true_vs_pred_upper_triangle, etc.)\n")
        f.write(f"  Pearson r  = {r_p:.6f} (p = {p_p:.4e})\n")
        f.write(f"  Spearman r = {r_s:.6f} (p = {p_s:.4e})\n\n")
        f.write("Robinson-Foulds (unrooted, ete3):\n")
        f.write(f"  RF distance = {rf_dist} / {rf_max}\n")
        f.write(f"  Normalized  = {rf_norm:.6f}\n\n")
        f.write("Interpretation: RF counts differing splits (lower = more similar topology).\n")
        f.write("Correlation summarizes matrix fidelity, not topology identity.\n")

    log(f"Saved {report_path}")
    log(f"Robinson-Foulds: {rf_dist} / {rf_max} (normalized {rf_norm:.4f})")

    meth_path = os.path.join(out_dir, "methodology.md")
    write_methodology_md(
        meth_path,
        args.scheme,
        cutoffs,
        class_labels,
        args.n_folds,
        args.n_estimators,
        deep_sorted,
    )
    log(f"Saved {meth_path}")

    cfg = {
        "scheme": args.scheme,
        "cutoffs": cutoffs,
        "labels": class_labels,
        "n_folds": args.n_folds,
        "n_estimators": args.n_estimators,
        "random_state": args.random_state,
        "species_fold": "StratifiedKFold_deep_vs_nondeep",
        "deep_species_in_analysis": deep_sorted,
        "extended_test_cv": True,
        "aggregation": "mean_predict_proba_then_argmax",
        "class_to_distance": "mean_over_folds_of_train_train_class_mean_patristic_MY",
        "output_dir": out_dir,
    }
    with open(os.path.join(out_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    log("Done.")


if __name__ == "__main__":
    main()
