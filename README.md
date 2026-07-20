# Mammalian phylogenetic signal in protein-domain copy number

Code and data for our study of how much of the mammalian phylogeny can be recovered from protein-domain content alone, and which domains carry that signal. For each of ~100 mammals we take the number of copies of every Pfam protein domain in its proteome and ask three questions: (1) how well does a distance built from those counts track divergence time; (2) can a classifier trained on pairwise domain differences recover the divergence structure and a clade-level tree; and (3) which domains distinguish the major clades, and what do they do biologically. The short answer is that domain content carries a real but coarse signal - it separates the deep lineages and recovers the broad clades, but not fine branching order - and the informative domains fall into a few interpretable functional groups (diet/metabolism, immunity, reproduction).

The per-species domain counts come from [UMAP of Life](https://umap.mcdb.ucla.edu); the reference timetree is from [TimeTree](http://www.timetree.org) (Kumar et al. 2017, 2022). The full methods and results are described in the accompanying manuscript (Yoon & Pellegrini, in preparation).

There is nothing to install as a package and no build step. Each script is a standalone analysis run from the command line with the standard scientific-Python stack (`requirements.txt`); it reads the input files under `data_raw/` and writes figures and tables under `results/`. Set `MPLCONFIGDIR` if you want to avoid matplotlib cache warnings.

## Data

- **`data_raw/MammalDomainCount.tsv`** - per-species Pfam domain copy numbers, 23,656 domains × 115 species, downloaded from UMAP of Life. Each entry is the number of times a Pfam domain occurs across that species' proteome. (The first line is a header banner, so loaders use `skiprows=1` and then transpose.)
- **`data_raw/MammalsPhylogeny.nwk`** - the TimeTree reference (Newick, 108 tips). Branch lengths are calibrated so that a pairwise patristic distance is roughly twice the two species' divergence time in millions of years; `verify_timetree_distances.py` checks this.
- **`data_raw/MammalsList.txt`** - the species list submitted to TimeTree.
- **`data_raw/MammalsUMAP.nwk`** - an alternative domain-based reference tree used by some of the comparison scripts.

After intersecting the domain matrix with the tree tips and dropping the platypus outgroup, **107 species** are analysed throughout, with **8,020 domains** left after removing zero-variance columns, giving **5,671 unordered species pairs**. A copy of `MammalsPhylogeny.nwk` also sits at the repository root because the classification pipeline reads it from there.

## Part 1 - Domain-content distance vs divergence time

Motivated by a random-walk (Brownian-motion) model in which the expected squared difference in a domain's count grows linearly with divergence time.

- **`scripts/domain_time_scatter.py`** - the core of Part 1. For each species pair it builds a distance summed over the top-N domains (ranked by how strongly their squared count difference tracks divergence time), under two weightings - by cross-species variance and by the regression slope (`alpha`). It sweeps N, and writes the correlation tables and the density scatter plots. Spearman is the primary metric because the small-N distances are zero-inflated.
- **`scripts/select_best_n.py`** - reads the correlation table and recommends an N.
- **`scripts/test_metric_variants.py`** - compares six ways of scaling the per-domain difference (squared, linear, sqrt, log, and two fractional powers) and writes the ranked domain list (`TopDomains.txt`) that some Part-1 scripts consume. Run this once before the scripts that need `TopDomains.txt`.
- **`scripts/mantel_significance.py`** - the honest significance test for the distance-time correlation. Because the 5,671 pairs are not independent, p-values come from a Mantel test that permutes species labels (9,999 permutations), so the 107 species - not the pairs - are the unit.

## Part 2 - Classification and tree reconstruction

- **`scripts/classification_cv_tree_reconstruction.py`** - the main pipeline, and the source of the headline result. Features are the absolute per-domain count differences for each pair; the target is the patristic distance binned into four ordinal classes (`SCHEME_4B`, cutoffs at 150/175/190 MY). A random forest is trained under species-blocked, clade-stratified 5-fold cross-validation (no species in a test fold appears in any training pair), with random oversampling on the training folds only. Predicted classes are mapped back to millions of years using train-fold means, a UPGMA tree is built from the predicted distances, and it is compared to the reference by Robinson-Foulds. Outputs: confusion matrix, distance-recovery scatter, predicted tree, and `run_config.json`.
- **`scripts/classification_cv_tree_reconstruction_include_platypus.py`** - the same pipeline with the platypus outgroup kept in, for the sensitivity check.
- **`scripts/verify_cv_tree_reconstruction.py`** - an independent verifier. It re-derives the folds from scratch and asserts there is no train/test species leakage, that the per-pair probability aggregation is correct, and that the saved CSVs, distance matrix, and tree are mutually consistent. Run it after the reconstruction; it exits non-zero on any failure.
- **`scripts/manual_distance_binning_search.py`** - the search over bin schemes (3-6 bins) that settled on the four-class cutoffs. Equal-frequency (quantile) binning fails here because a large block of pairs is tied at ~188 MY, which no quantile split can separate; this is why the cutoffs are placed by hand.
- **`scripts/pairwise_distance_classification_oversampling.py`** - compares the binning schemes against the class-imbalance strategies (no correction, class weights, random oversampling, SMOTE), which is how random oversampling was chosen.

## Part 3 - Clade enrichment and interpretation

- **`scripts/domain_clade_enrichment.py`** - per-clade enrichment. For each clade and domain it compares the per-species counts in that clade against all other species with a Mann-Whitney U test, correcting across domains with the Benjamini-Hochberg FDR. Enrichment uses absolute copy number; direction (enriched vs depleted) is the sign of the log2 fold-change.
- **`scripts/filter_clade_enrichment_secondary.py`** - keeps the hits that pass FDR < 0.05 and |log2 fold-change| ≥ 0.5.
- **`scripts/plot_filtered_enrichment_heatmap.py`** - the clade × domain enrichment heatmap.
- **`scripts/domain_clade_interpretation.py`** - the predictive-importance side: random-forest Gini importance, held-out permutation importance, one-way ANOVA across clades, and per-clade marker domains, all on the same preprocessing as the classifier. ANOVA and RF agree on the leading domains (PF02885, a glycosyl transferase, is first by both).
- **`scripts/elastic_net_regression_stratified_species_cv.py`** - this file defines the clade -> species mapping that the two scripts above import, so it is kept even though the elastic-net regressor it also contains is *not* the final predictor (we moved from regression to classification because the divergence-time distribution is too discontinuous for regression to fit well).

## Quality checks

- **`scripts/sanity_checks_loocv_domain_phylogeny.py`** - validates the distance and feature construction without training a model.
- **`scripts/verify_timetree_distances.py`** - confirms the ~2× relationship between the TimeTree branch lengths and divergence time.

## Robustness

- **`supplementary/baseline_trees_presence_absence.py`** - recomputes the domain distance from binarized presence/absence (Jaccard, Hamming) and under the centered-log-ratio (Aitchison) transform, and builds naive UPGMA and Neighbor-Joining trees directly, with no supervised step. This shows the conclusions do not depend on copy number vs presence/absence, on the compositional closure, or on the classifier - every character and method plateaus at a similar Robinson-Foulds distance, which locates the limit in the character (domain content is coarse and homoplasy-prone) rather than the model.
- **`supplementary/species_blocked_rf_quantile_tree.py`** - a quantile-binning variant of the Part-2 tree.

## Figures

- **`figures/render_part1.py`, `render_part2.py`, `render_part3_enrichment.py`, `render_part3_heatmap.py`, `render_part3_interpretation.py`** - render the individual panels from the saved results.
- **`figures/render_composites.py`** - assembles the multi-panel manuscript figures from those panels.
- **`figures/render_schematic.py`** - the workflow schematic.
- **`figures/pubstyle.py`** - shared plotting style (fonts, sizes, the clade colour palette).
- **`figures/make_itol_annotation.py`** - writes the iTOL annotation used to colour the reference and predicted trees by clade (the radial trees themselves are exported from iTOL; see `figures/paper_assets/`).
- **`figures/pfam_name_fixes.py`** - a small lookup that tidies a few Pfam domain names for display.

## Reproducing the analysis

```bash
pip install -r requirements.txt
export MPLCONFIGDIR="$PWD/.cache/matplotlib"   # optional; silences matplotlib cache warnings

# Part 1
python3 scripts/test_metric_variants.py                 # writes data_raw/TopDomains.txt first
python3 scripts/domain_time_scatter.py
python3 scripts/mantel_significance.py

# Part 2 (headline result)
python3 scripts/classification_cv_tree_reconstruction.py
python3 scripts/verify_cv_tree_reconstruction.py --output-dir results/classification_cv_tree_reconstruction

# Part 3
python3 scripts/domain_clade_enrichment.py
python3 scripts/filter_clade_enrichment_secondary.py
python3 scripts/plot_filtered_enrichment_heatmap.py
python3 scripts/domain_clade_interpretation.py

# Figures (after the analyses above have populated results/)
python3 figures/render_part1.py
python3 figures/render_part2.py
python3 figures/render_composites.py
```

The random seed is fixed (`RANDOM_STATE = 42`) so the cross-validation folds, oversampling, and classifier are reproducible; the classification pipeline writes a `run_config.json` that the verifier reads back to re-derive the folds.

## Data and code availability

- **Domain counts:** UMAP of Life . `MammalDomainCount.tsv` is included here for convenience; the authoritative source is UMAP of Life.
- **Reference phylogeny:** TimeTree - Kumar S, Stecher G, Suleski M, Hedges SB (2017), *Mol Biol Evol* 34(7):1812-1819; and Kumar S et al. (2022), *Mol Biol Evol* 39(8):msac174.
- **This repository** is archived at `[Zenodo DOI - to be added on release]`.

## Citation

If you use this code or data, please cite:

> Yoon A, Pellegrini M. Phylogenetic signal in mammalian protein-domain copy number: recovering divergence depth and broad clades. In preparation.

Corresponding author: Matteo Pellegrini (matteop@mcdb.ucla.edu), Department of Molecular, Cell, and Developmental Biology, University of California, Los Angeles.
