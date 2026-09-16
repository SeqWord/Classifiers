# network_parser/config.py
"""
network_parser.config

Central configuration object for NetworkParser.

Architecture
--------------------
Input -> preprocessing
      -> central feature filtering
      -> ML protocol / model selector
      -> conditional decision tree branch
      -> optional downstream interaction validation
"""

from dataclasses import dataclass
from typing import Literal, Optional, cast


@dataclass
class NetworkParserConfig:
    # -------------------------------------------------
    # 1) Input / Output behavior
    # -------------------------------------------------
    include_intermediate_files: bool = True
    generic_name: str = "matrix"

    # -------------------------------------------------
    # 2) VCF-level QC / callability (DataLoader + query)
    # Shared semantics live in network_parser.vcf_call_semantics.
    # -------------------------------------------------
    qual_threshold: float = 30.0
    min_dp_per_sample: int = 10
    min_gq_per_sample: int = 20
    mq_threshold: float = 40.0
    mq0f_threshold: float = 0.1
    biallelic_only: bool = True
    # Ploidies accepted by the shared binary VCF interpreter. Calls with any
    # other ploidy are unresolved; they are never reduced implicitly.
    vcf_supported_ploidies: tuple = (1, 2)
    # Respect VCF FILTER column (PASS / . allowed by default).
    vcf_respect_filter: bool = True
    vcf_allowed_filters: str = "PASS,."
    # Safe default: sites absent from a variant-only VCF are NOT treated as REF.
    # Set True only for legacy single-sample variant-only cohorts; emits a warning.
    assume_absent_variant_is_reference: bool = False
    # Expand / resolve gVCF reference blocks (END) as callable reference.
    expand_gvcf_ref_blocks: bool = True
    # Optional REF-allele check against the loaded reference genome.
    validate_ref_against_genome: bool = False
    # Query prediction gates (fractions of selected features).
    min_feature_recovery_fraction: float = 0.5
    min_callable_fraction: float = 0.5
    enforce_query_callability_gates: bool = True
    # FASTQ calling: emit gVCF-style callable reference where supported.
    fastq_emit_gvcf: bool = True
    fastq_gvcf_min_dp: int = 10

    # Matrix contract: missingness limits (NaN = non-callable; never silent 0).
    # Legacy ``max_missing_fraction`` is applied to both sample and feature axes
    # unless the explicit per-axis knobs are set.
    max_missing_fraction: float = 0.5
    max_missing_fraction_per_sample: Optional[float] = None
    max_missing_fraction_per_feature: Optional[float] = None
    drop_high_missing_samples: bool = True
    drop_high_missing_features: bool = True
    # Imputation for algorithms that cannot accept NaN. Fit on train only.
    # none | baseline | feature_mode | constant
    genotype_impute_strategy: Literal[
        "none", "baseline", "feature_mode", "constant"
    ] = "baseline"
    genotype_impute_constant: float = 0.0
    add_missing_indicator_features: bool = False
    allow_missing_as_category: bool = False
    # Explicit contig aliases for VCF query (query_name → training/manifest name).
    # JSON object or comma pairs "query=train,q2=t2". Never use position-only match.
    contig_alias_map: Optional[str] = None
    # Declared biological reference identity and explicitly circular contigs
    # used in manifests/annotation (comma-separated names or a sequence).
    reference_id: Optional[str] = None
    reference_circular_contigs: Optional[str] = None
    allow_position_only_vcf_match: bool = False

    min_spacing_bp: int = 10

    # -------------------------------------------------
    # 3) Cohort-level SNP filtering
    # -------------------------------------------------
    min_sample_presence: int = 10

    # -------------------------------------------------
    # 4) Binary encoding strategy
    # -------------------------------------------------
    ancestral_allele: Literal["Y", "N"] = "Y"

    # -------------------------------------------------
    # 5) Lightweight preprocessing
    # -------------------------------------------------
    remove_invariant: bool = True
    min_minor_count: int = 10

    # -------------------------------------------------
    # 6) Artifact filtering controls
    # -------------------------------------------------
    matrices_min_count: int = 10
    matrices_repeat_number: int = 5
    matrices_type: Literal["all", "coding", "sense-mutations"] = "all"
    matrices_fix: str = ""
    matrices_redundancy_sample_threshold: int = 2000
    matrices_redundancy_sample_size: int = 256

    # -------------------------------------------------
    # 7) Central statistical feature filtering
    # -------------------------------------------------
    run_central_feature_filtering: bool = True

    # Explicit central feature-filter choice.
    # "auto" preserves legacy behavior:
    #   run_rf_fdr_feature_selection=True  -> rf_fdr
    #   run_rf_fdr_feature_selection=False -> chi2_fdr or fisher_fdr via statistical_test
    central_feature_filter_method: Literal[
        "auto", "rf_fdr", "chi2_fdr", "fisher_fdr", "chi2_perm_fdr"
    ] = "auto"
    resolved_central_feature_filter_method: str = "rf_fdr"

    statistical_test: Literal["chi2", "fisher"] = "chi2"
    significance_level: float = 0.05
    fdr_alpha: float = 0.05
    fdr_threshold: float = 0.05
    chi2_min_expected: int = 5
    # Used by chi2_perm_fdr and optional downstream interaction permutation tests.
    n_permutation_tests: int = 1000
    feature_filter_fallback_strategy: Literal["stop", "unfiltered"] = "stop"

    # Legacy internal prefilter compatibility
    prefilter_alpha: float = 0.05
    min_nonmissing_prefilter: float = 0.20
    min_maf_prefilter: float = 0.0
    max_prefiltered_features: Optional[int] = 10000

    multiple_testing_method: Literal["fdr_bh", "bonferroni"] = "fdr_bh"
    # -------------------------------------------------
    # RF-FDR central feature selection
    # -------------------------------------------------
    run_rf_fdr_feature_selection: bool = True

    rf_selector_n_estimators: int = 300
    rf_selector_n_observed_repeats: int = 10

    # Higher default gives better empirical p-value resolution for FDR-BH.
    # This improves robust inference when RF-FDR is used as the main filter.
    rf_selector_n_permutations: int = 1000

    rf_selector_fdr_alpha: float = 0.05
    rf_selector_max_features: str = "sqrt"
    rf_selector_min_samples_leaf: int = 1
    rf_selector_class_weight: Optional[str] = "balanced"
    rf_selector_min_importance: float = 0.0
    rf_selector_top_n: Optional[int] = None
    rf_selector_random_state: int = 42

    # stop = do not silently pass exploratory top-N features as if they were FDR-significant.
    # Use "top_n" only for exploratory/smoke testing.
    rf_selector_fallback_strategy: Literal["top_n", "unfiltered", "stop"] = "stop"
    rf_selector_fallback_top_n: int = 500

    # Hierarchy-protocol fallback control (classic two-level and multi-level).
    # False = fail loudly if ML training fails.
    # True = allow RF fallback model and record it explicitly.
    allow_hierarchy_rf_fallback: bool = False
    # Backward-compatible alias for allow_hierarchy_rf_fallback
    allow_two_level_rf_fallback: bool = False
    # -------------------------------------------------
    # 8) Decision Tree parameters (capacity constrained)
    # -------------------------------------------------
    max_depth: Optional[int] = 12
    max_branch_depth: int = 3
    min_samples_split: int = 4
    min_samples_leaf: int = 2
    min_information_gain: float = 0.001

    min_group_size: int = 2

    # -------------------------------------------------
    # 9) Tree-path interaction candidates (POST-TREE)
    # -------------------------------------------------
    epistasis_strength_threshold: float = 0.05  # MI synergy gate for candidates
    max_tree_path_interaction_candidates: int = 50
    max_epistatic_interactions: int = 50  # deprecated alias
    interaction_min_joint_support: int = 10
    interaction_min_bootstrap_stability: float = 0.5
    interaction_fdr_alpha: float = 0.05

    # -------------------------------------------------
    # 10) Post-tree bootstrap / stability
    # -------------------------------------------------
    # Unambiguous: number of independent cohort bootstrap resamples (B).
    n_bootstrap_resamples: int = 100
    # Unambiguous: fraction of cohort drawn with replacement each resample.
    bootstrap_cohort_sample_fraction: float = 1.0
    # Legacy aliases (synced in __post_init__; do not use for new code)
    n_bootstrap: int = 100
    bootstrap_sample_fraction: float = 1.0
    n_bootstrap_samples: int = 100  # was misnamed; not sample size
    bootstrap_samples_per_iter: int = 100  # was misused as sample size
    bootstrap_outer_iters: int = 1  # deprecated outer loop
    min_bootstrap_support: float = 0.7

    # -------------------------------------------------
    # 11) Optional matrix compression
    # -------------------------------------------------
    use_integer_variant_ids: bool = False

    # -------------------------------------------------
    # 12) Performance & reproducibility
    # -------------------------------------------------
    n_jobs: int = -1
    random_state: int = 42
    memory_efficient: bool = False

    # Query-time parallelism. Hierarchy query can process independent samples
    # concurrently; child-level inference is also batched by parent group.
    query_parallel_samples: bool = True
    query_parallel_n_jobs: Optional[int] = None

    # Training-time parallelism for independent workloads.
    # All of these auto-scale: on a 16 GB laptop they collapse toward serial;
    # on a 128 GB / 24-core node they use concurrent model fits. Never required.
    hierarchy_parallel_fallback_training: bool = True
    level2_parallel_group_training: bool = True
    # Parallel training of sibling hierarchy children (path-local nodes).
    hierarchy_parallel_child_nodes: bool = True
    feature_panel_parallel_scoring: bool = True
    association_test_batch_size: int = 250
    # Soft RAM estimate per concurrent outer model fit (filter + CV + train).
    # Used only to cap outer workers: ~16 GB → 1–2 workers; ~128 GB → many.
    parallel_memory_per_worker_gb: float = 4.0
    # Optional hard cap on workers regardless of n_jobs / CPU count.
    parallel_max_workers: Optional[int] = None
    # Skip re-training nodes that already have node_summary.json + model.
    hierarchy_resume_completed_nodes: bool = False
    # Optional named biological hierarchy preset (see hierarchy_artifacts.HIERARCHY_PRESETS).
    hierarchy_preset: Optional[str] = None

    # -------------------------------------------------
    # 13) FASTQ query preprocessing
    # -------------------------------------------------
    # These controls are used only when query_input_type="fastq".
    # FASTQ reads are converted to per-sample VCF.GZ files and then routed
    # through the existing DataLoader VCF-directory pathway. No statistical
    # filtering or model training happens in this stage.
    fastq_max_parallel_samples: int = 1
    fastq_threads: Optional[int] = None
    fastq_memory_per_sample_mb: Optional[int] = None
    fastq_clean_intermediates: bool = False
    fastq_auto_index_reference: bool = True
    fastq_min_mapping_quality: int = 20
    fastq_sample_platform: str = "ILLUMINA"
    fastq_sort_memory: str = "1G"
    # When multiple R1/R2 lanes map to the same sample id:
    #   fail  — raise with a clear multi-lane listing (default, safe)
    #   merge — pair by lane key then concatenate in lane order
    fastq_multi_lane_policy: Literal["fail", "merge"] = "fail"
    # Haploid (1) is the microbial default; diploid (2) allowed for optional use.
    fastq_ploidy: int = 1
    # samtools sort -m is memory *per sort thread*; total ≈ threads × this value.
    # Keep modest when sort threads > 1.
    fastq_sort_memory_per_thread: str = "512M"
    # Write flagstat/stats (lighter than full depth matrices).
    fastq_write_alignment_stats: bool = True
    # Normalize VCF/gVCF with bcftools norm -f against the documented reference.
    fastq_normalize_vcf: bool = True
    # Variant calling strategy after alignment:
    #   full            — whole-genome bcftools mpileup|call (default legacy)
    #   panel_bcftools  — bcftools restricted to trained panel BED (-R)
    #   panel_majority  — samtools mpileup on panel + median/majority base call
    fastq_call_mode: Literal["full", "panel_bcftools", "panel_majority"] = "full"
    # Optional BED / manifest of panel sites (Contig:Pos:Ref:Alt in BED name or Feature_ID col).
    # When empty and call mode is panel_*, query engine should inject sites from the
    # selected-feature manifest (trained markers only).
    fastq_panel_sites_bed: Optional[str] = None
    fastq_panel_manifest: Optional[str] = None
    # Majority pileup thresholds (panel_majority mode)
    fastq_panel_min_depth: int = 10
    fastq_panel_min_majority_fraction: float = 0.7
    fastq_panel_min_base_quality: int = 20
    # Emit REF panel sites as GT=0 (usually False: use assume_absent_variant_is_reference)
    fastq_panel_emit_reference_sites: bool = False

    # -------------------------------------------------
    # 14) Pipeline mode
    # -------------------------------------------------
    pipeline_mode: Literal[
        "matrix_only",
        "decision_tree_only",
        "ml_only",
        "both",
        "hierarchy",
        "two_level",  # alias of hierarchy
    ] = "both"

    # Backward compatibility
    run_ml_protocol: bool = False
    # -------------------------------------------------
    # Hierarchy protocol (2+ levels; classic two-label route uses level1/level2)
    # -------------------------------------------------
    level1_label_column: Optional[str] = None
    level2_label_column: Optional[str] = None
    # Optional target for the global Level-2 fallback. When unset, the global
    # fallback uses level2_label_column / --level2_label. When set, group-specific
    # Level-2 models still use the detailed --level2_label, while the global
    # fallback can use a broader endpoint such as AMR_binary.
    global_level2_label_column: Optional[str] = None
    train_global_level2: bool = True
    # Optional absolute override for group-specific Level-2 training. When None,
    # group eligibility is adaptive: the minimum training sample count scales
    # with the number of Level-2 labels represented in that group.
    min_level2_samples_per_group: Optional[int] = None

    # Label-support gate applied before hierarchy-node and Level-2 statistical
    # filtering. Classes with fewer than level2_min_class_count samples are
    # excluded so stratified CV and model screening remain feasible.
    level2_drop_low_support_classes: bool = True
    level2_min_class_count: int = 2

    # Query-time reporting for labels that were rare in training. Instead of
    # emitting a hard class call, NetworkParser can surface a review-required
    # endpoint and ask the user to inspect the sample or merge rare classes in
    # metadata when that is biologically appropriate.
    low_support_review_enabled: bool = True
    low_support_review_min_class_count: int = 10
    low_support_review_label: str = "low_support_review_required"
    low_support_review_action_message: str = (
        "Manually review this sample or merge rare classes in metadata if that "
        "grouping is biologically appropriate."
    )

    # Query-time AMR evidence guard. Branch AMR models can call susceptible with
    # high probability when too few lineage-specific resistance markers resolve.
    # When that happens, optionally escalate to a saved terminal AMR fallback
    # model; otherwise warn (default) or block the class label (legacy).
    #
    # amr_weak_evidence_mode:
    #   warn  — keep the model class (e.g. susceptible) for TP/FP/TN/FN tables,
    #           and attach reason / review flags in side columns.
    #   block — replace the reported prediction with amr_weak_evidence_review_label
    #           (counts as abstention/review in evaluation, not a phenotype class).
    amr_weak_evidence_review_enabled: bool = True
    amr_weak_evidence_mode: Literal["warn", "block"] = "warn"
    amr_weak_evidence_min_resolved_fraction: float = 0.05
    amr_weak_evidence_review_label: str = "amr_evidence_review_required"
    amr_weak_evidence_review_action_message: str = (
        "Branch AMR prediction had insufficient resolved resistance-marker evidence. "
        "Manually review the resistance phenotype or inspect marker coverage before "
        "accepting a susceptible call."
    )
    amr_evidence_guard_label_columns: Optional[str] = None
    hierarchy_global_amr_fallback_on_weak_evidence: bool = True
    hierarchy_global_amr_fallback_min_resistant_probability: float = 0.50

    # Global (cohort-wide) fallback models — opt-in by hierarchy label.
    #
    # hierarchy_global_fallback_labels (comma-separated):
    #   "none" or ""     → train no global models (default; path-local only)
    #   "terminal"       → global model for the last hierarchy label only
    #   "legacy" or "*"  → previous defaults: terminal global + lineage global
    #   "Lineage_clean,AMR_binary" → globals only for those columns
    #
    # Parent-conditioned fallbacks (e.g. terminal phenotype within each lineage)
    # are separate and remain useful when thin branches skip path-local models.
    hierarchy_global_fallback_labels: str = "none"
    hierarchy_train_parent_conditioned_fallbacks: bool = True

    # Recursive-hierarchy global lineage fallback (used when lineage is listed in
    # hierarchy_global_fallback_labels, or when that setting is "legacy").
    hierarchy_train_global_lineage_fallback: bool = True
    hierarchy_global_lineage_fallback_label: Optional[str] = None
    hierarchy_global_lineage_fallback_on_low_confidence: bool = True
    hierarchy_global_lineage_fallback_on_disagreement: bool = True
    hierarchy_global_lineage_fallback_min_support_delta: float = 0.0

    # Optional additional Level-2 global binary fallback. This trains a
    # resistant/susceptible endpoint across all lineages and is used only when
    # a group-specific Level-2 model is unavailable. The binary target can come
    # from a dedicated metadata column or from an explicit mapping file that
    # collapses detailed Level-2 labels into resistant/susceptible states.
    level2_train_binary_global_fallback: bool = False
    level2_binary_label_column: Optional[str] = None
    level2_binary_label_mapping_file: Optional[str] = None
    level2_binary_resistant_values: str = (
        "R,resistant,RESISTANT,Resistant,1,true,TRUE,True"
    )
    level2_binary_susceptible_values: str = (
        "S,susceptible,SUSCEPTIBLE,Susceptible,0,false,FALSE,False"
    )

    # -------------------------------------------------
    # 15) Updated orchestration flags
    # -------------------------------------------------
    run_model_selector: bool = True
    run_conditional_dt: bool = True
    disable_conditional_dt: bool = False
    trigger_decision_tree_on_selected: bool = True
    trigger_decision_tree_if_candidate: bool = True
    decision_tree_requires_selector_match: bool = False
    # Explicit DT candidate rule (used when trigger_decision_tree_if_candidate=True).
    # DT runs as a candidate only when its probe score is among the top-k
    # finite scores AND within max_margin of the best finite probe score.
    # Set decision_tree_always_run_interpretability=True to force DT whenever
    # pipeline mode includes the tree branch (document as always-run stage).
    decision_tree_candidate_top_k: int = 2
    decision_tree_candidate_max_margin: float = 0.05
    decision_tree_always_run_interpretability: bool = False

    # -------------------------------------------------
    # 16) ML protocol branch
    # -------------------------------------------------
    ml_algorithm: str = "auto"
    ml_lr_max_iter: int = 2000
    # Decision / minimum-support threshold grid (formerly misnamed "sensitivity").
    # Legacy aliases ml_min/max/step_sensitivity still accepted via property-like names.
    ml_min_decision_threshold: float = 0.5
    ml_max_decision_threshold: float = 1.0
    ml_step_decision_threshold: float = 0.1
    # Backward-compatible aliases (deprecated)
    ml_min_sensitivity: float = 0.5
    ml_max_sensitivity: float = 1.0
    ml_step_sensitivity: float = 0.1
    # Selective-classification threshold objective (OOF only).
    # Default avoids the broken end-to-end accuracy objective (higher threshold
    # cannot improve e2e accuracy when abstentions count as errors).
    # - min_called_error_subject_to_coverage: minimise error among *called*
    #   samples subject to min_coverage (call rate) constraint
    # - utility: maximise reward_correct*n_correct - cost_wrong*n_wrong - cost_abstain*n_abstain
    # - balanced_called_and_coverage: weighted mix of called accuracy and call rate
    # - accuracy_called_only / call_rate: single-component diagnostics
    # - accuracy_end_to_end: legacy (not recommended for threshold selection)
    ml_threshold_objective: Literal[
        "min_called_error_subject_to_coverage",
        "utility",
        "balanced_called_and_coverage",
        "accuracy_called_only",
        "call_rate",
        "accuracy_end_to_end",
    ] = "min_called_error_subject_to_coverage"
    ml_threshold_min_coverage: float = 0.5
    ml_threshold_utility_reward_correct: float = 1.0
    ml_threshold_utility_cost_wrong: float = 2.0
    ml_threshold_utility_cost_abstain: float = 0.5
    ml_threshold_objective_called_weight: float = 0.5
    ml_threshold_objective_coverage_weight: float = 0.5
    # Optional isotonic calibration of support scores on OOF folds (binary/multiclass top score).
    # When False, scores are reported as support_score, not probability/confidence.
    ml_calibrate_support_scores: bool = False
    ml_calibration_method: Literal["none", "isotonic"] = "none"
    # Select decision threshold on out-of-fold predictions (not final training fit).
    ml_select_threshold_out_of_fold: bool = True
    ml_threshold_cv_splits: int = 5
    # Model-selector rule thresholds (documented; not magic constants in code only).
    # linear_score_high: recommend LR when mean linear probe ≥ this and nonlinear gain small
    selector_linear_score_high: float = 0.85
    selector_nonlinear_delta_ignore: float = 0.03
    selector_nonlinear_delta_prefer: float = 0.05
    selector_run_clustering_diagnostics: bool = False
    # Minimum successful bootstrap resamples required for stability reporting
    bootstrap_min_successful_resamples: int = 20

    ml_empty_symbol: str = ""
    ml_remove_empty_field_threshold: float = 1.0

    # Leakage-aware CV grouping: metadata column for blocked/grouped folds.
    cv_group_column: Optional[str] = None
    # Nested CV policy (authoritative evaluation route).
    # strict: fold-local preprocessing only (default, publication-grade).
    # exploratory_transductive: may reuse global preprocessing — never for claims.
    cv_preprocessing_mode: Literal["strict", "exploratory_transductive"] = "strict"
    # Publication summaries require every requested fold unless user opts in.
    cv_allow_partial_results: bool = False
    # Require every class to appear in enough distinct groups when grouped CV.
    cv_min_groups_per_class: int = 2
    # When True, main single-label orchestrator runs nested CV as evaluation.
    run_nested_cv_evaluation: bool = True
    cv_n_repeats: int = 3
    cv_n_splits: int = 5

    # -------------------------------------------------
    # 17) Ranked feature-panel separability check
    # -------------------------------------------------
    # Runs after central statistical filtering and before ML / tree fitting.
    # It ranks retained features by corrected/empirical/raw p-value, evaluates
    # top-N panels, and forwards the smallest panel with acceptable separability.
    run_feature_panel_separability_check: bool = True
    feature_panel_sizes: tuple = (100, 200, 500, 1000)
    feature_panel_metric: Literal[
        "balanced_accuracy",
        "adjusted_rand",
        "normalized_mutual_info",
        "silhouette",
    ] = "balanced_accuracy"

    # Supervised probe used to score top-N panels by balanced accuracy.
    # "lr" is fast and stable after scaling; "rf" is slower but captures
    # nonlinear feature combinations while keeping the stage pre-model.
    feature_panel_classifier: Literal["lr", "rf"] = "lr"
    feature_panel_lr_max_iter: int = 2000
    feature_panel_lr_tol: float = 1e-4
    feature_panel_rf_n_estimators: int = 300
    feature_panel_rf_max_features: Optional[str] = "sqrt"
    feature_panel_rf_min_samples_leaf: int = 1
    feature_panel_rf_class_weight: Optional[str] = "balanced"
    feature_panel_rf_n_jobs: Optional[int] = None

    feature_panel_min_score: float = 0.75
    feature_panel_selection_rule: Literal[
        "smallest_passing", "best_passing", "best_available"
    ] = "smallest_passing"
    feature_panel_cv_splits: int = 5
    # Keep the model-ready search restricted to the configured compact panels.
    # The complete centrally filtered matrix is available only as an explicit
    # exploratory opt-in and is never added silently.
    feature_panel_always_include_full_filtered: bool = False
    # stop = do not train a model when no candidate panel reaches min_score.
    # best_available = explicit exploratory opt-in to the old fallback policy.
    feature_panel_threshold_failure_strategy: Literal["stop", "best_available"] = "stop"
    feature_panel_max_silhouette_samples: int = 5000
    feature_panel_run_clustering_diagnostics: bool = False
    feature_panel_large_feature_threshold: int = 5000
    feature_panel_large_max_scoring_features: int = 5000
    feature_panel_large_pool_multiplier: int = 4
    feature_panel_score_full_large_matrix: bool = False
    # Variance prefilter is OFF by default — must be explicit to replace statistical pool.
    feature_panel_allow_variance_prefilter: bool = False
    feature_panel_strict_failure: bool = True

    # -------------------------------------------------
    # Optional known-marker seed for phenotype endpoints
    # -------------------------------------------------
    # When True, catalogue/known mutations present in the filtered matrix are
    # force-included at the start of feature panels for matching stages
    # (AMR / resistance profile by default). Lineage stages are skipped unless
    # their stage name matches seed_known_markers_stage_substrings.
    # Default False — pure statistical ranking until the user opts in.
    seed_known_markers: bool = False
    # Path to WHO-style catalogue TSV (Position/Ref/Alt/Contig) or Feature_ID list.
    known_markers_path: Optional[str] = None
    # force_include / rank_boost: known markers occupy the first panel slots.
    seed_known_markers_mode: Literal["force_include", "rank_boost"] = "force_include"
    # Stage-name substrings (case-insensitive) that receive seeding.
    # Comma-separated string or sequence. Empty = all stages.
    seed_known_markers_stage_substrings: tuple = (
        "amr",
        "resistance",
        "pheno",
        "profile",
        "resistant",
        "susceptible",
    )
    # Optional cap on how many known markers to seed (None = all matches).
    seed_known_markers_max: Optional[int] = None
    # RF-FDR permutation resolution: warn | fail
    rf_selector_permutation_resolution_policy: Literal["warn", "fail"] = "warn"
    # -------------------------------------------------
    # Cross-validation input loading
    # -------------------------------------------------
    # False = preserve training-style unsupervised matrix filters during CV loading.
    # True = relax filters to min_sample_presence=1 before fold-specific filtering.
    cv_relax_input_filters: bool = True

    # -------------------------------------------------
    # 18) Two-level binary model bundle output
    # -------------------------------------------------
    # Build a portable .npb bundle automatically at the end of hierarchy
    # training. This keeps query-ready deployment artifacts next to the
    # registry without requiring a second manual CLI command.
    build_model_bundle: bool = True
    model_bundle_filename: str = "networkparser_model_bundle.npb"
    model_bundle_include_model_payloads: bool = True
    model_bundle_include_feature_manifests: bool = True
    model_bundle_include_ranked_feature_tables: bool = True
    model_bundle_fail_on_error: bool = False

    def __post_init__(self) -> None:
        self.min_group_size = self.min_samples_split

        if self.ancestral_allele not in {"Y", "N"}:
            raise ValueError("ancestral_allele must be 'Y' or 'N'")

        if self.statistical_test not in {"chi2", "fisher"}:
            raise ValueError("statistical_test must be 'chi2' or 'fisher'")

        supported_central_filters = {
            "auto",
            "rf_fdr",
            "chi2_fdr",
            "fisher_fdr",
            "chi2_perm_fdr",
        }
        if self.central_feature_filter_method not in supported_central_filters:
            raise ValueError(
                "central_feature_filter_method must be one of: "
                "'auto', 'rf_fdr', 'chi2_fdr', 'fisher_fdr', or 'chi2_perm_fdr'"
            )

        # Resolve the explicit selector into the legacy booleans/tests used by
        # older parts of the codebase. This preserves backward compatibility
        # while giving users a clean method switch.
        if self.central_feature_filter_method == "auto":
            if bool(self.run_rf_fdr_feature_selection):
                self.resolved_central_feature_filter_method = "rf_fdr"
            elif self.statistical_test == "fisher":
                self.resolved_central_feature_filter_method = "fisher_fdr"
            else:
                self.resolved_central_feature_filter_method = "chi2_fdr"
        else:
            self.resolved_central_feature_filter_method = (
                self.central_feature_filter_method
            )

        if self.resolved_central_feature_filter_method == "rf_fdr":
            self.run_rf_fdr_feature_selection = True
        elif self.resolved_central_feature_filter_method == "chi2_fdr":
            self.run_rf_fdr_feature_selection = False
            self.statistical_test = "chi2"
        elif self.resolved_central_feature_filter_method == "fisher_fdr":
            self.run_rf_fdr_feature_selection = False
            self.statistical_test = "fisher"
        elif self.resolved_central_feature_filter_method == "chi2_perm_fdr":
            self.run_rf_fdr_feature_selection = False
            self.statistical_test = "chi2"

        if self.feature_filter_fallback_strategy not in {"stop", "unfiltered"}:
            raise ValueError(
                "feature_filter_fallback_strategy must be one of: 'stop' or 'unfiltered'"
            )

        if self.matrices_type not in {"all", "coding", "sense-mutations"}:
            raise ValueError(
                "matrices_type must be one of: all, coding, sense-mutations"
            )

        if self.multiple_testing_method not in {"fdr_bh", "bonferroni"}:
            raise ValueError("multiple_testing_method must be 'fdr_bh' or 'bonferroni'")

        supported_modes = {
            "matrix_only",
            "decision_tree_only",
            "ml_only",
            "both",
            "hierarchy",
            "two_level",  # alias of hierarchy
        }
        if self.pipeline_mode not in supported_modes:
            raise ValueError(f"pipeline_mode must be one of: {sorted(supported_modes)}")
        if self.pipeline_mode == "two_level":
            self.pipeline_mode = "hierarchy"
        if self.global_level2_label_column is not None:
            self.global_level2_label_column = (
                str(self.global_level2_label_column).strip() or None
            )
        if (
            self.min_level2_samples_per_group is not None
            and self.min_level2_samples_per_group < 2
        ):
            raise ValueError("min_level2_samples_per_group must be >= 2 or None")
        if not isinstance(self.level2_drop_low_support_classes, bool):
            raise ValueError("level2_drop_low_support_classes must be boolean")
        if int(self.level2_min_class_count) < 2:
            raise ValueError("level2_min_class_count must be >= 2")
        self.level2_min_class_count = int(self.level2_min_class_count)
        if not isinstance(self.low_support_review_enabled, bool):
            raise ValueError("low_support_review_enabled must be boolean")
        review_min = int(self.low_support_review_min_class_count)
        if review_min < 2:
            raise ValueError("low_support_review_min_class_count must be >= 2")
        self.low_support_review_min_class_count = review_min
        review_label = str(self.low_support_review_label or "").strip()
        if not review_label:
            raise ValueError("low_support_review_label must be a non-empty string")
        self.low_support_review_label = review_label
        action_message = str(self.low_support_review_action_message or "").strip()
        if not action_message:
            raise ValueError(
                "low_support_review_action_message must be a non-empty string"
            )
        self.low_support_review_action_message = action_message
        if not isinstance(self.amr_weak_evidence_review_enabled, bool):
            raise ValueError("amr_weak_evidence_review_enabled must be boolean")
        amr_mode = str(getattr(self, "amr_weak_evidence_mode", "warn") or "warn").strip().lower()
        if amr_mode not in {"warn", "block"}:
            raise ValueError("amr_weak_evidence_mode must be 'warn' or 'block'")
        self.amr_weak_evidence_mode = amr_mode  # type: ignore[assignment]
        amr_min_frac = float(self.amr_weak_evidence_min_resolved_fraction)
        if not (0.0 <= amr_min_frac <= 1.0):
            raise ValueError(
                "amr_weak_evidence_min_resolved_fraction must be between 0 and 1"
            )
        self.amr_weak_evidence_min_resolved_fraction = amr_min_frac
        amr_review_label = str(self.amr_weak_evidence_review_label or "").strip()
        if not amr_review_label:
            raise ValueError(
                "amr_weak_evidence_review_label must be a non-empty string"
            )
        self.amr_weak_evidence_review_label = amr_review_label
        amr_action = str(self.amr_weak_evidence_review_action_message or "").strip()
        if not amr_action:
            raise ValueError(
                "amr_weak_evidence_review_action_message must be a non-empty string"
            )
        self.amr_weak_evidence_review_action_message = amr_action
        if self.amr_evidence_guard_label_columns is not None:
            cols = [
                str(x).strip()
                for x in str(self.amr_evidence_guard_label_columns).split(",")
                if str(x).strip()
            ]
            self.amr_evidence_guard_label_columns = ",".join(cols) if cols else None
        if not isinstance(self.hierarchy_global_amr_fallback_on_weak_evidence, bool):
            raise ValueError(
                "hierarchy_global_amr_fallback_on_weak_evidence must be boolean"
            )
        amr_res_prob = float(
            self.hierarchy_global_amr_fallback_min_resistant_probability
        )
        if not (0.0 <= amr_res_prob <= 1.0):
            raise ValueError(
                "hierarchy_global_amr_fallback_min_resistant_probability must be between 0 and 1"
            )
        self.hierarchy_global_amr_fallback_min_resistant_probability = amr_res_prob
        if self.hierarchy_global_lineage_fallback_label is not None:
            self.hierarchy_global_lineage_fallback_label = (
                str(self.hierarchy_global_lineage_fallback_label).strip() or None
            )
        # Normalise global-fallback label list
        gfl = str(getattr(self, "hierarchy_global_fallback_labels", "none") or "none").strip()
        self.hierarchy_global_fallback_labels = gfl
        if not isinstance(self.hierarchy_train_parent_conditioned_fallbacks, bool):
            raise ValueError(
                "hierarchy_train_parent_conditioned_fallbacks must be boolean"
            )
        if not isinstance(self.hierarchy_train_global_lineage_fallback, bool):
            raise ValueError("hierarchy_train_global_lineage_fallback must be boolean")
        if not isinstance(
            self.hierarchy_global_lineage_fallback_on_low_confidence, bool
        ):
            raise ValueError(
                "hierarchy_global_lineage_fallback_on_low_confidence must be boolean"
            )
        if not isinstance(self.hierarchy_global_lineage_fallback_on_disagreement, bool):
            raise ValueError(
                "hierarchy_global_lineage_fallback_on_disagreement must be boolean"
            )
        delta = float(self.hierarchy_global_lineage_fallback_min_support_delta)
        if delta < 0.0:
            raise ValueError(
                "hierarchy_global_lineage_fallback_min_support_delta must be >= 0"
            )
        self.hierarchy_global_lineage_fallback_min_support_delta = delta

        if not isinstance(self.query_parallel_samples, bool):
            raise ValueError("query_parallel_samples must be boolean")
        if (
            self.query_parallel_n_jobs is not None
            and int(self.query_parallel_n_jobs) == 0
        ):
            raise ValueError("query_parallel_n_jobs must be != 0 or None")
        if not isinstance(self.hierarchy_parallel_fallback_training, bool):
            raise ValueError("hierarchy_parallel_fallback_training must be boolean")
        if not isinstance(self.level2_parallel_group_training, bool):
            raise ValueError("level2_parallel_group_training must be boolean")
        if not isinstance(self.hierarchy_parallel_child_nodes, bool):
            raise ValueError("hierarchy_parallel_child_nodes must be boolean")
        if not isinstance(self.hierarchy_resume_completed_nodes, bool):
            raise ValueError("hierarchy_resume_completed_nodes must be boolean")
        if not isinstance(self.feature_panel_parallel_scoring, bool):
            raise ValueError("feature_panel_parallel_scoring must be boolean")
        mem_pw = float(getattr(self, "parallel_memory_per_worker_gb", 4.0) or 4.0)
        if mem_pw <= 0:
            raise ValueError("parallel_memory_per_worker_gb must be > 0")
        self.parallel_memory_per_worker_gb = mem_pw
        if self.parallel_max_workers is not None:
            self.parallel_max_workers = int(self.parallel_max_workers)
            if self.parallel_max_workers < 1:
                raise ValueError("parallel_max_workers must be >= 1 or None")
        batch_size = int(self.association_test_batch_size)
        if batch_size < 1:
            raise ValueError("association_test_batch_size must be >= 1")
        self.association_test_batch_size = batch_size
        if not isinstance(self.level2_train_binary_global_fallback, bool):
            raise ValueError("level2_train_binary_global_fallback must be boolean")
        if self.level2_binary_label_column is not None:
            self.level2_binary_label_column = (
                str(self.level2_binary_label_column).strip() or None
            )
        if self.level2_binary_label_mapping_file is not None:
            self.level2_binary_label_mapping_file = (
                str(self.level2_binary_label_mapping_file).strip() or None
            )
        if not str(self.level2_binary_resistant_values).strip():
            raise ValueError("level2_binary_resistant_values cannot be empty")
        if not str(self.level2_binary_susceptible_values).strip():
            raise ValueError("level2_binary_susceptible_values cannot be empty")

        # Correct common typo while preserving backward compatibility.
        if self.ml_algorithm == "SCV":
            self.ml_algorithm = "SVC"

        supported_ml = {"auto", "RF", "MLP", "LR", "MBCS", "DT", "SVC", "DNL"}
        if self.ml_algorithm not in supported_ml:
            raise ValueError(f"ml_algorithm must be one of: {sorted(supported_ml)}")

        if self.run_ml_protocol and self.pipeline_mode == "decision_tree_only":
            self.pipeline_mode = "both"

        if int(self.fastq_max_parallel_samples) < 1:
            raise ValueError("fastq_max_parallel_samples must be >= 1")
        self.fastq_max_parallel_samples = int(self.fastq_max_parallel_samples)
        if self.fastq_threads is not None:
            self.fastq_threads = int(self.fastq_threads)
            if self.fastq_threads < 1:
                raise ValueError("fastq_threads must be >= 1 or None")
        if self.fastq_memory_per_sample_mb is not None:
            self.fastq_memory_per_sample_mb = int(self.fastq_memory_per_sample_mb)
            if self.fastq_memory_per_sample_mb < 256:
                raise ValueError("fastq_memory_per_sample_mb must be >= 256 or None")
        if not isinstance(self.fastq_clean_intermediates, bool):
            raise ValueError("fastq_clean_intermediates must be boolean")
        if not isinstance(self.fastq_auto_index_reference, bool):
            raise ValueError("fastq_auto_index_reference must be boolean")
        if int(self.fastq_min_mapping_quality) < 0:
            raise ValueError("fastq_min_mapping_quality must be >= 0")
        self.fastq_min_mapping_quality = int(self.fastq_min_mapping_quality)
        self.fastq_sample_platform = (
            str(self.fastq_sample_platform).strip() or "ILLUMINA"
        )
        self.fastq_sort_memory = str(self.fastq_sort_memory).strip() or "1G"
        multi_lane = (
            str(getattr(self, "fastq_multi_lane_policy", "fail") or "fail")
            .strip()
            .lower()
        )
        if multi_lane not in {"fail", "merge"}:
            raise ValueError("fastq_multi_lane_policy must be 'fail' or 'merge'")
        self.fastq_multi_lane_policy = multi_lane  # type: ignore[assignment]
        ploidy = int(getattr(self, "fastq_ploidy", 1) or 1)
        if ploidy not in {1, 2}:
            raise ValueError(
                "fastq_ploidy must be 1 (haploid, microbial default) or 2 (diploid); "
                f"got {ploidy}"
            )
        self.fastq_ploidy = ploidy
        call_mode = str(getattr(self, "fastq_call_mode", "full") or "full").strip().lower()
        if call_mode not in {"full", "panel_bcftools", "panel_majority"}:
            raise ValueError(
                "fastq_call_mode must be one of: full, panel_bcftools, panel_majority"
            )
        self.fastq_call_mode = call_mode  # type: ignore[assignment]
        if self.fastq_panel_sites_bed is not None:
            self.fastq_panel_sites_bed = str(self.fastq_panel_sites_bed).strip() or None
        if self.fastq_panel_manifest is not None:
            self.fastq_panel_manifest = str(self.fastq_panel_manifest).strip() or None
        self.fastq_panel_min_depth = int(getattr(self, "fastq_panel_min_depth", 10) or 10)
        if self.fastq_panel_min_depth < 1:
            raise ValueError("fastq_panel_min_depth must be >= 1")
        frac = float(getattr(self, "fastq_panel_min_majority_fraction", 0.7) or 0.7)
        if not 0.0 < frac <= 1.0:
            raise ValueError("fastq_panel_min_majority_fraction must be in (0, 1]")
        self.fastq_panel_min_majority_fraction = frac
        self.fastq_panel_min_base_quality = int(
            getattr(self, "fastq_panel_min_base_quality", 20) or 20
        )
        if self.fastq_panel_min_base_quality < 0:
            raise ValueError("fastq_panel_min_base_quality must be >= 0")
        if not isinstance(
            getattr(self, "fastq_panel_emit_reference_sites", False), bool
        ):
            raise ValueError("fastq_panel_emit_reference_sites must be boolean")
        # Prefer explicit per-thread sort memory; keep legacy fastq_sort_memory as fallback alias.
        sort_m = str(
            getattr(self, "fastq_sort_memory_per_thread", None)
            or getattr(self, "fastq_sort_memory", "512M")
            or "512M"
        ).strip()
        self.fastq_sort_memory_per_thread = sort_m
        self.fastq_sort_memory = sort_m  # legacy alias points at per-thread value
        if not isinstance(self.fastq_write_alignment_stats, bool):
            raise ValueError("fastq_write_alignment_stats must be boolean")
        if not isinstance(self.fastq_normalize_vcf, bool):
            raise ValueError("fastq_normalize_vcf must be boolean")

        if int(getattr(self, "decision_tree_candidate_top_k", 2)) < 1:
            raise ValueError("decision_tree_candidate_top_k must be >= 1")
        self.decision_tree_candidate_top_k = int(self.decision_tree_candidate_top_k)
        if float(getattr(self, "decision_tree_candidate_max_margin", 0.05)) < 0:
            raise ValueError("decision_tree_candidate_max_margin must be >= 0")
        self.decision_tree_candidate_max_margin = float(
            self.decision_tree_candidate_max_margin
        )

        if self.qual_threshold < 0:
            raise ValueError("qual_threshold must be >= 0")
        if self.min_dp_per_sample < 0:
            raise ValueError("min_dp_per_sample must be >= 0")
        if self.min_gq_per_sample < 0:
            raise ValueError("min_gq_per_sample must be >= 0")
        if self.mq_threshold < 0:
            raise ValueError("mq_threshold must be >= 0")
        if not 0 <= self.mq0f_threshold <= 1:
            raise ValueError("mq0f_threshold must be in [0, 1]")
        if not isinstance(self.vcf_respect_filter, bool):
            raise ValueError("vcf_respect_filter must be boolean")
        if not str(self.vcf_allowed_filters).strip():
            raise ValueError("vcf_allowed_filters cannot be empty")
        if not isinstance(self.assume_absent_variant_is_reference, bool):
            raise ValueError("assume_absent_variant_is_reference must be boolean")
        if not isinstance(self.expand_gvcf_ref_blocks, bool):
            raise ValueError("expand_gvcf_ref_blocks must be boolean")
        if not isinstance(self.validate_ref_against_genome, bool):
            raise ValueError("validate_ref_against_genome must be boolean")
        if not 0.0 <= float(self.min_feature_recovery_fraction) <= 1.0:
            raise ValueError("min_feature_recovery_fraction must be in [0, 1]")
        if not 0.0 <= float(self.min_callable_fraction) <= 1.0:
            raise ValueError("min_callable_fraction must be in [0, 1]")
        if not isinstance(self.enforce_query_callability_gates, bool):
            raise ValueError("enforce_query_callability_gates must be boolean")
        raw_supported_ploidies = self.vcf_supported_ploidies
        if isinstance(raw_supported_ploidies, str):
            raw_supported_ploidies = [
                value.strip()
                for value in raw_supported_ploidies.split(",")
                if value.strip()
            ]
        try:
            supported_ploidies = tuple(int(value) for value in raw_supported_ploidies)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "vcf_supported_ploidies must contain positive integers"
            ) from exc
        if not supported_ploidies or any(value < 1 for value in supported_ploidies):
            raise ValueError("vcf_supported_ploidies must contain positive integers")
        self.vcf_supported_ploidies = supported_ploidies
        if not isinstance(self.fastq_emit_gvcf, bool):
            raise ValueError("fastq_emit_gvcf must be boolean")
        if int(self.fastq_gvcf_min_dp) < 0:
            raise ValueError("fastq_gvcf_min_dp must be >= 0")
        self.fastq_gvcf_min_dp = int(self.fastq_gvcf_min_dp)
        if self.min_sample_presence < 1:
            raise ValueError("min_sample_presence must be >= 1")
        if not 0 <= self.max_missing_fraction <= 1:
            raise ValueError("max_missing_fraction must be in [0, 1]")
        if self.max_missing_fraction_per_sample is not None:
            self.max_missing_fraction_per_sample = float(
                self.max_missing_fraction_per_sample
            )
            if not 0 <= self.max_missing_fraction_per_sample <= 1:
                raise ValueError("max_missing_fraction_per_sample must be in [0, 1]")
        if self.max_missing_fraction_per_feature is not None:
            self.max_missing_fraction_per_feature = float(
                self.max_missing_fraction_per_feature
            )
            if not 0 <= self.max_missing_fraction_per_feature <= 1:
                raise ValueError("max_missing_fraction_per_feature must be in [0, 1]")
        if str(self.genotype_impute_strategy) not in {
            "none",
            "baseline",
            "feature_mode",
            "constant",
        }:
            raise ValueError(
                "genotype_impute_strategy must be one of: none, baseline, feature_mode, constant"
            )
        if str(self.cv_preprocessing_mode) not in {
            "strict",
            "exploratory_transductive",
        }:
            raise ValueError(
                "cv_preprocessing_mode must be 'strict' or 'exploratory_transductive'"
            )
        if int(self.cv_min_groups_per_class) < 1:
            raise ValueError("cv_min_groups_per_class must be >= 1")
        if int(self.cv_n_repeats) < 1:
            raise ValueError("cv_n_repeats must be >= 1")
        if int(self.cv_n_splits) < 2:
            raise ValueError("cv_n_splits must be >= 2")
        if self.min_spacing_bp < 0:
            raise ValueError("min_spacing_bp must be >= 0")
        if self.min_minor_count < 0:
            raise ValueError("min_minor_count must be >= 0")
        if self.matrices_min_count < 0:
            raise ValueError("matrices_min_count must be >= 0")
        if self.matrices_repeat_number < 1:
            raise ValueError("matrices_repeat_number must be >= 1")
        if self.matrices_redundancy_sample_threshold < 1:
            raise ValueError("matrices_redundancy_sample_threshold must be >= 1")
        if self.matrices_redundancy_sample_size < 1:
            raise ValueError("matrices_redundancy_sample_size must be >= 1")
        if not 0 < self.significance_level <= 1:
            raise ValueError("significance_level must be in (0, 1]")
        if not 0 < self.fdr_alpha <= 1:
            raise ValueError("fdr_alpha must be in (0, 1]")
        if not 0 < self.fdr_threshold <= 1:
            raise ValueError("fdr_threshold must be in (0, 1]")
        if self.chi2_min_expected < 1:
            raise ValueError("chi2_min_expected must be >= 1")
        if self.n_permutation_tests < 0:
            raise ValueError("n_permutation_tests must be >= 0")
        if (
            self.resolved_central_feature_filter_method == "chi2_perm_fdr"
            and self.n_permutation_tests < 1
        ):
            raise ValueError(
                "n_permutation_tests must be >= 1 when using chi2_perm_fdr"
            )
        if not 0 < self.prefilter_alpha <= 1:
            raise ValueError("prefilter_alpha must be in (0, 1]")
        if not 0 <= self.min_nonmissing_prefilter <= 1:
            raise ValueError("min_nonmissing_prefilter must be in [0, 1]")
        if not 0 <= self.min_maf_prefilter <= 0.5:
            raise ValueError("min_maf_prefilter must be in [0, 0.5]")
        if (
            self.max_prefiltered_features is not None
            and self.max_prefiltered_features < 1
        ):
            raise ValueError("max_prefiltered_features must be >= 1 or None")
        if self.max_depth is not None and self.max_depth <= 0:
            raise ValueError("max_depth must be positive or None")
        if self.max_branch_depth < 1:
            raise ValueError("max_branch_depth must be >= 1")
        if self.min_samples_split < 2:
            raise ValueError("min_samples_split must be >= 2")
        if self.min_samples_leaf < 1:
            raise ValueError("min_samples_leaf must be >= 1")
        if self.epistasis_strength_threshold < 0:
            raise ValueError("epistasis_strength_threshold must be >= 0")
        if (
            int(self.max_tree_path_interaction_candidates) == 50
            and int(self.max_epistatic_interactions) != 50
        ):
            self.max_tree_path_interaction_candidates = int(
                self.max_epistatic_interactions
            )
        else:
            self.max_epistatic_interactions = int(
                self.max_tree_path_interaction_candidates
            )
        if self.max_tree_path_interaction_candidates < 1:
            raise ValueError("max_tree_path_interaction_candidates must be >= 1")
        # Prefer unambiguous bootstrap fields; accept legacy aliases.
        if int(self.n_bootstrap_resamples) == 100 and int(self.n_bootstrap) not in (
            0,
            100,
        ):
            self.n_bootstrap_resamples = int(self.n_bootstrap)
        elif int(self.n_bootstrap_resamples) == 100 and int(
            self.n_bootstrap_samples
        ) not in (0, 100, 1000):
            self.n_bootstrap_resamples = int(self.n_bootstrap_samples)
        self.n_bootstrap = int(self.n_bootstrap_resamples)
        self.n_bootstrap_samples = int(self.n_bootstrap_resamples)
        if (
            float(self.bootstrap_cohort_sample_fraction) == 1.0
            and 0 < float(self.bootstrap_sample_fraction) <= 1
        ):
            # keep cohort fraction if explicitly set away from default via legacy
            if float(self.bootstrap_sample_fraction) != 0.8:
                self.bootstrap_cohort_sample_fraction = float(
                    self.bootstrap_sample_fraction
                )
        self.bootstrap_sample_fraction = float(self.bootstrap_cohort_sample_fraction)
        if self.n_bootstrap_resamples < 0:
            raise ValueError("n_bootstrap_resamples must be >= 0")
        if not 0 < float(self.bootstrap_cohort_sample_fraction) <= 1:
            raise ValueError("bootstrap_cohort_sample_fraction must be in (0, 1]")
        if not 0 <= self.min_bootstrap_support <= 1:
            raise ValueError("min_bootstrap_support must be in [0, 1]")
        if self.interaction_min_joint_support < 1:
            raise ValueError("interaction_min_joint_support must be >= 1")
        if not 0 <= float(self.interaction_min_bootstrap_stability) <= 1:
            raise ValueError("interaction_min_bootstrap_stability must be in [0, 1]")
        if not 0 < float(self.interaction_fdr_alpha) <= 1:
            raise ValueError("interaction_fdr_alpha must be in (0, 1]")
        # Sync legacy sensitivity aliases ↔ decision_threshold names
        if (
            float(self.ml_min_decision_threshold) == 0.5
            and float(self.ml_min_sensitivity) != 0.5
        ):
            self.ml_min_decision_threshold = float(self.ml_min_sensitivity)
        else:
            self.ml_min_sensitivity = float(self.ml_min_decision_threshold)
        if (
            float(self.ml_max_decision_threshold) == 1.0
            and float(self.ml_max_sensitivity) != 1.0
        ):
            self.ml_max_decision_threshold = float(self.ml_max_sensitivity)
        else:
            self.ml_max_sensitivity = float(self.ml_max_decision_threshold)
        if (
            float(self.ml_step_decision_threshold) == 0.1
            and float(self.ml_step_sensitivity) != 0.1
        ):
            self.ml_step_decision_threshold = float(self.ml_step_sensitivity)
        else:
            self.ml_step_sensitivity = float(self.ml_step_decision_threshold)

        if not 0 <= self.ml_min_decision_threshold <= 1:
            raise ValueError("ml_min_decision_threshold must be in [0, 1]")
        if not 0 <= self.ml_max_decision_threshold <= 1:
            raise ValueError("ml_max_decision_threshold must be in [0, 1]")
        if self.ml_min_decision_threshold > self.ml_max_decision_threshold:
            raise ValueError(
                "ml_min_decision_threshold cannot exceed ml_max_decision_threshold"
            )
        if self.ml_step_decision_threshold <= 0:
            raise ValueError("ml_step_decision_threshold must be > 0")
        if self.ml_threshold_objective not in {
            "min_called_error_subject_to_coverage",
            "utility",
            "balanced_called_and_coverage",
            "accuracy_called_only",
            "call_rate",
            "accuracy_end_to_end",
        }:
            raise ValueError("ml_threshold_objective has an unsupported value")
        if not 0 <= float(self.ml_threshold_min_coverage) <= 1:
            raise ValueError("ml_threshold_min_coverage must be in [0, 1]")
        if not 0 <= float(self.ml_threshold_objective_called_weight) <= 1:
            raise ValueError("ml_threshold_objective_called_weight must be in [0, 1]")
        if not 0 <= float(self.ml_threshold_objective_coverage_weight) <= 1:
            raise ValueError("ml_threshold_objective_coverage_weight must be in [0, 1]")
        if str(self.ml_calibration_method) not in {"none", "isotonic"}:
            raise ValueError("ml_calibration_method must be 'none' or 'isotonic'")
        if int(self.ml_threshold_cv_splits) < 2:
            raise ValueError("ml_threshold_cv_splits must be >= 2")
        if int(self.bootstrap_min_successful_resamples) < 1:
            raise ValueError("bootstrap_min_successful_resamples must be >= 1")
        if self.cv_group_column is not None:
            self.cv_group_column = str(self.cv_group_column).strip() or None
        if not 0 <= self.ml_remove_empty_field_threshold <= 1:
            raise ValueError("ml_remove_empty_field_threshold must be in [0, 1]")
        if self.ml_lr_max_iter < 100:
            raise ValueError("ml_lr_max_iter must be >= 100")

        if not isinstance(self.run_feature_panel_separability_check, bool):
            raise ValueError("run_feature_panel_separability_check must be boolean")
        if not self.feature_panel_sizes:
            raise ValueError(
                "feature_panel_sizes must contain at least one positive integer"
            )
        try:
            self.feature_panel_sizes = tuple(
                int(x.strip()) if isinstance(x, str) else int(x)
                for x in (
                    self.feature_panel_sizes.split(",")
                    if isinstance(self.feature_panel_sizes, str)
                    else self.feature_panel_sizes
                )
            )
        except Exception as exc:
            raise ValueError(
                "feature_panel_sizes must be a tuple/list or comma-separated string of positive integers"
            ) from exc
        if any(x < 1 for x in self.feature_panel_sizes):
            raise ValueError("feature_panel_sizes must contain positive integers")
        if self.feature_panel_metric not in {
            "balanced_accuracy",
            "adjusted_rand",
            "normalized_mutual_info",
            "silhouette",
        }:
            raise ValueError(
                "feature_panel_metric must be one of: balanced_accuracy, adjusted_rand, "
                "normalized_mutual_info, silhouette"
            )

        if not isinstance(self.seed_known_markers, bool):
            raise ValueError("seed_known_markers must be boolean")
        if self.known_markers_path is not None:
            self.known_markers_path = str(self.known_markers_path).strip() or None
        seed_mode = str(
            getattr(self, "seed_known_markers_mode", "force_include") or "force_include"
        ).strip().lower()
        if seed_mode not in {"force_include", "rank_boost"}:
            raise ValueError(
                "seed_known_markers_mode must be force_include or rank_boost"
            )
        self.seed_known_markers_mode = seed_mode  # type: ignore[assignment]
        raw_subs = getattr(self, "seed_known_markers_stage_substrings", ())
        if isinstance(raw_subs, str):
            self.seed_known_markers_stage_substrings = tuple(
                s.strip() for s in raw_subs.split(",") if s.strip()
            )
        else:
            self.seed_known_markers_stage_substrings = tuple(
                str(s).strip() for s in raw_subs if str(s).strip()
            )
        if self.seed_known_markers_max is not None:
            self.seed_known_markers_max = int(self.seed_known_markers_max)
            if self.seed_known_markers_max < 1:
                raise ValueError("seed_known_markers_max must be >= 1 or None")
        if self.seed_known_markers and not self.known_markers_path:
            # Allow True with empty path only as soft warn at runtime; keep valid.
            pass

        panel_classifier_aliases = {
            "logistic": "lr",
            "logistic_regression": "lr",
            "randomforest": "rf",
            "random_forest": "rf",
            "random_forest_classifier": "rf",
        }
        normalized_panel_classifier = panel_classifier_aliases.get(
            str(self.feature_panel_classifier).strip().lower().replace("-", "_"),
            str(self.feature_panel_classifier).strip().lower(),
        )
        if normalized_panel_classifier not in {"lr", "rf"}:
            raise ValueError("feature_panel_classifier must be one of: lr, rf")
        self.feature_panel_classifier = cast(
            Literal["lr", "rf"], normalized_panel_classifier
        )
        if self.feature_panel_lr_max_iter < 100:
            raise ValueError("feature_panel_lr_max_iter must be >= 100")
        if not 0 < self.feature_panel_lr_tol <= 1:
            raise ValueError("feature_panel_lr_tol must be in (0, 1]")
        if self.feature_panel_rf_n_estimators < 1:
            raise ValueError("feature_panel_rf_n_estimators must be >= 1")
        if self.feature_panel_rf_min_samples_leaf < 1:
            raise ValueError("feature_panel_rf_min_samples_leaf must be >= 1")
        if (
            self.feature_panel_rf_n_jobs is not None
            and int(self.feature_panel_rf_n_jobs) == 0
        ):
            raise ValueError(
                "feature_panel_rf_n_jobs cannot be 0; use 1, -1, or another non-zero integer"
            )
        if isinstance(self.feature_panel_rf_class_weight, str):
            cw = self.feature_panel_rf_class_weight.strip().lower()
            self.feature_panel_rf_class_weight = (
                None if cw in {"", "none", "null"} else cw
            )
        if self.feature_panel_rf_class_weight not in {
            None,
            "balanced",
            "balanced_subsample",
        }:
            raise ValueError(
                "feature_panel_rf_class_weight must be one of: balanced, balanced_subsample, none"
            )
        if isinstance(self.feature_panel_rf_max_features, str):
            mf = self.feature_panel_rf_max_features.strip().lower()
            self.feature_panel_rf_max_features = (
                None if mf in {"", "none", "null"} else mf
            )
            if self.feature_panel_rf_max_features not in {None, "sqrt", "log2"}:
                raise ValueError(
                    "feature_panel_rf_max_features must be one of: sqrt, log2, none"
                )

        if not 0 <= self.feature_panel_min_score <= 1:
            raise ValueError("feature_panel_min_score must be in [0, 1]")
        if self.feature_panel_selection_rule not in {
            "smallest_passing",
            "best_passing",
            "best_available",
        }:
            raise ValueError(
                "feature_panel_selection_rule must be one of: smallest_passing, best_passing, best_available"
            )
        if self.feature_panel_cv_splits < 2:
            raise ValueError("feature_panel_cv_splits must be >= 2")
        if not isinstance(self.feature_panel_always_include_full_filtered, bool):
            raise ValueError(
                "feature_panel_always_include_full_filtered must be boolean"
            )
        if self.feature_panel_threshold_failure_strategy not in {
            "stop",
            "best_available",
        }:
            raise ValueError(
                "feature_panel_threshold_failure_strategy must be one of: "
                "stop, best_available"
            )
        if self.feature_panel_max_silhouette_samples < 2:
            raise ValueError("feature_panel_max_silhouette_samples must be >= 2")
        if self.feature_panel_large_feature_threshold < 1:
            raise ValueError("feature_panel_large_feature_threshold must be >= 1")
        if self.feature_panel_large_max_scoring_features < 1:
            raise ValueError("feature_panel_large_max_scoring_features must be >= 1")
        if self.feature_panel_large_pool_multiplier < 1:
            raise ValueError("feature_panel_large_pool_multiplier must be >= 1")
        if not isinstance(self.feature_panel_score_full_large_matrix, bool):
            raise ValueError("feature_panel_score_full_large_matrix must be boolean")

        if not isinstance(self.build_model_bundle, bool):
            raise ValueError("build_model_bundle must be boolean")
        self.model_bundle_filename = str(self.model_bundle_filename).strip()
        if not self.model_bundle_filename:
            raise ValueError("model_bundle_filename cannot be empty")
        if not self.model_bundle_filename.endswith(".npb"):
            self.model_bundle_filename = f"{self.model_bundle_filename}.npb"
        if not isinstance(self.model_bundle_include_model_payloads, bool):
            raise ValueError("model_bundle_include_model_payloads must be boolean")
        if not isinstance(self.model_bundle_include_feature_manifests, bool):
            raise ValueError("model_bundle_include_feature_manifests must be boolean")
        if not isinstance(self.model_bundle_include_ranked_feature_tables, bool):
            raise ValueError(
                "model_bundle_include_ranked_feature_tables must be boolean"
            )
        if not isinstance(self.model_bundle_fail_on_error, bool):
            raise ValueError("model_bundle_fail_on_error must be boolean")

        if self.rf_selector_n_estimators < 1:
            raise ValueError("rf_selector_n_estimators must be >= 1")

        if self.rf_selector_n_observed_repeats < 1:
            raise ValueError("rf_selector_n_observed_repeats must be >= 1")

        if self.rf_selector_n_permutations < 1:
            raise ValueError("rf_selector_n_permutations must be >= 1")

        if not 0 < self.rf_selector_fdr_alpha <= 1:
            raise ValueError("rf_selector_fdr_alpha must be in (0, 1]")

        if self.rf_selector_min_samples_leaf < 1:
            raise ValueError("rf_selector_min_samples_leaf must be >= 1")

        if self.rf_selector_min_importance < 0:
            raise ValueError("rf_selector_min_importance must be >= 0")

        if self.rf_selector_top_n is not None and self.rf_selector_top_n < 1:
            raise ValueError("rf_selector_top_n must be >= 1 or None")

        if self.rf_selector_fallback_strategy not in {"top_n", "unfiltered", "stop"}:
            raise ValueError(
                "rf_selector_fallback_strategy must be one of: "
                "'top_n', 'unfiltered', or 'stop'"
            )

        if self.rf_selector_fallback_top_n < 1:
            raise ValueError("rf_selector_fallback_top_n must be >= 1")

        if not isinstance(self.allow_hierarchy_rf_fallback, bool):
            raise ValueError("allow_hierarchy_rf_fallback must be boolean")
        if not isinstance(self.allow_two_level_rf_fallback, bool):
            raise ValueError("allow_two_level_rf_fallback must be boolean")
        # Keep legacy alias and preferred name synchronized
        if self.allow_two_level_rf_fallback and not self.allow_hierarchy_rf_fallback:
            self.allow_hierarchy_rf_fallback = True
        if self.allow_hierarchy_rf_fallback and not self.allow_two_level_rf_fallback:
            self.allow_two_level_rf_fallback = True
