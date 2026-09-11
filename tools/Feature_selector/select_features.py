import sys
import os
import csv
import argparse
from typing import Dict, List, Tuple, Set


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Select diagnostic SNP/probe features from an annotation TSV/CSV file "
            "according to Chi-square/FDR predictions, genomic proximity, and "
            "redundancy of allelic-state patterns."
        )
    )

    parser.add_argument(
        "project_folder",
        type=str,
        help="Project subfolder inside input_folder. Mandatory positional argument."
    )

    parser.add_argument(
        "-i", "--input_folder",
        type=str,
        default="input",
        help="Input root folder (default: input)"
    )

    parser.add_argument(
        "-o", "--output_folder",
        type=str,
        default="output",
        help="Output root folder (default: output)"
    )

    parser.add_argument(
        "-m", "--matrix_file",
        type=str,
        required=True,
        help=(
            "Mandatory TSV/CSV matrix of allelic states represented as 0/1 or A,T,G,C "
            "characters (like '*/cluster_algorithm/_matrix_filtered_matrix.csv'; "
            "this file was created at step 6.Feature_scoring)."
        )
    )

    parser.add_argument(
        "-c", "--prediction_file",
        type=str,
        required=True,
        help=(
            "Mandatory CSV/TSV file with Chi-square/FDR feature predictions "
            "(like '*/cluster_algorithm/_matrix_chi2_fdr_results.csv'; "
            "this file was created at step 6.Feature_scoring)"
        )
    )

    parser.add_argument(
        "-f", "--feature_file",
        type=str,
        required=True,
        help=(
            "Mandatory TSV/CSV file with the initial list of annotated features "
            "(like '*_positions_filtered.tsv'; this file was created at step 4.Filter_features)."
        )
    )

    parser.add_argument(
        "-a", "--allele_file",
        type=str,
        required=True,
        help=(
            "Mandatory CSV/TSV file with common allelic states at each polymorphism "
            "(like '*_digit_alleles_filtered.csv'; this file was created at step 4.Filter_features)."
        )
    )

    parser.add_argument(
        "-p", "--p_value",
        type=float,
        default=0.05,
        help="Maximum accepted raw p-value (default: 0.05)."
    )

    parser.add_argument(
        "-d", "--fdr_value",
        type=float,
        default=0.05,
        help="Maximum accepted level FDR p-value (default: 0.05)."
    )

    parser.add_argument(
        "-g", "--global_fdr_value",
        type=float,
        default=0.05,
        help="Maximum accepted global FDR p-value (default: 0.05)."
    )

    # Preserve the old misspelled long option for compatibility while also
    # providing a correctly spelled name.
    parser.add_argument(
        "-s", "--similarity_threshold", "--similarity_threshols",
        dest="similarity_threshold",
        type=float,
        default=10.0,
        help=(
            "Maximum percentage of mismatching allelic states allowed within a "
            "redundancy cluster (0..50; default: 10). A value of 0 groups only "
            "identical SNP patterns."
        )
    )

    parser.add_argument(
        "-r", "--redundancy",
        type=int,
        default=5,
        help="Maximum number of SNPs to retain in one redundancy cluster (default: 5)"
    )

    parser.add_argument(
        "-x", "--minimum_distance_between_SNP",
        type=int,
        default=160,
        help=(
            "Minimum allowed reference-sequence distance in bp between retained "
            "neighbouring SNPs (default: 160)."
        )
    )

    parser.add_argument(
        "-n", "--feature_number_to_keep",
        type=int,
        default=500,
        help="Total maximum number of features to keep (default: 500)"
    )

    parser.add_argument(
        "--evaluate",
        type=str,
        default=None,
        help=(
            "Optional comma-separated feature counts for Random-Forest accuracy "
            "evaluation before the final feature-number limit, e.g. --evaluate 50,100,1000. "
            "Values below 10 or above the number of features remaining after filtering "
            "are ignored with a warning."
        )
    )

    return parser.parse_args()


def parse_matrix(in_file: str) -> List[List[str]]:
    """Read TSV/TXT/CSV file and return non-empty rows as stripped strings."""

    ext = os.path.splitext(in_file)[1].lower()

    if ext in {".tsv", ".txt"}:
        delimiter = "\t"
    elif ext == ".csv":
        delimiter = ","
    else:
        raise ValueError(f"Unrecognized format of input file: {in_file}")

    with open(in_file, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter=delimiter)
        return [
            [cell.strip() for cell in row]
            for row in reader
            if row and any(cell.strip() for cell in row)
        ]


def transpose_snp_matrix(matrix_rows: List[List[str]]) -> List[List[str]]:
    """
    Convert a genome x SNP matrix to SNP x genome format.

    Input format:
        Genome, SNP_1, SNP_2, ...
        genome_A, 0, 1, ...
        genome_B, 1, 1, ...

    Output format:
        [SNP_1, state_A, state_B, ...]
        [SNP_2, state_A, state_B, ...]
    """

    if len(matrix_rows) < 2:
        raise ValueError("SNP matrix must contain a header and at least one genome row.")

    expected_columns = len(matrix_rows[0])
    if expected_columns < 2:
        raise ValueError("SNP matrix must contain a genome-ID column and at least one SNP column.")

    for row_number, row in enumerate(matrix_rows[1:], start=2):
        if len(row) != expected_columns:
            raise ValueError(
                f"SNP matrix row {row_number} contains {len(row)} columns; "
                f"expected {expected_columns}."
            )

    snp_titles = matrix_rows[0][1:]
    data = [row[1:] for row in matrix_rows[1:]]

    return [
        [snp_title, *states]
        for snp_title, states in zip(snp_titles, zip(*data))
    ]


def build_neighbourhoods(
    feature_body: List[List[str]],
    header: List[str],
    minimum_distance: int
) -> Tuple[List[List[str]], Dict[str, int]]:
    """
    Build connected groups of SNPs whose adjacent reference coordinates are no
    farther apart than `minimum_distance`.

    Returns
    -------
    neighbourhoods : list[list[str]]
        SNP IDs in each multi-SNP neighbourhood.
    membership : dict[str, int]
        SNP ID -> neighbourhood index.

    Notes
    -----
    SNPs without a valid reference coordinate are not placed in a neighbourhood
    and therefore are not removed by the proximity filter.
    """

    try:
        snp_col = header.index("SNP")
    except ValueError:
        snp_col = 0

    try:
        location_col = header.index("Location in reference sequence")
    except ValueError as exc:
        raise ValueError(
            "Feature file does not contain the required column "
            "'Location in reference sequence'."
        ) from exc

    located: List[Tuple[str, int]] = []

    for row in feature_body:
        if len(row) <= max(snp_col, location_col):
            continue

        snp_id = row[snp_col].strip()
        location_text = row[location_col].strip()

        if not snp_id or not location_text:
            continue

        try:
            location = int(location_text)
        except ValueError:
            continue

        located.append((snp_id, location))

    located.sort(key=lambda item: item[1])

    neighbourhoods: List[List[str]] = []

    if not located:
        return neighbourhoods, {}

    current_group: List[Tuple[str, int]] = [located[0]]

    for current in located[1:]:
        previous = current_group[-1]

        if current[1] - previous[1] <= minimum_distance:
            current_group.append(current)
        else:
            if len(current_group) > 1:
                neighbourhoods.append([item[0] for item in current_group])
            current_group = [current]

    if len(current_group) > 1:
        neighbourhoods.append([item[0] for item in current_group])

    membership: Dict[str, int] = {}
    for group_id, members in enumerate(neighbourhoods):
        for snp_id in members:
            membership[snp_id] = group_id

    return neighbourhoods, membership


def keep_best_from_neighbourhoods(
    ranked_predictions: List[List[str]],
    membership: Dict[str, int],
    snp_id_col: int
) -> Tuple[List[List[str]], List[str]]:
    """
    Keep the first (therefore best-ranked) SNP encountered in each genomic
    neighbourhood and remove subsequent SNPs belonging to the same group.

    ``ranked_predictions`` contains rows from the new prediction table, where
    the SNP identifier is stored in the ``Location`` column rather than column 0.
    """

    used_groups: Set[int] = set()
    kept: List[List[str]] = []
    removed: List[str] = []

    for row in ranked_predictions:
        if len(row) <= snp_id_col:
            continue

        snp_id = row[snp_id_col].strip()
        if not snp_id:
            continue

        group_id = membership.get(snp_id)

        if group_id is None:
            kept.append(row)
            continue

        if group_id in used_groups:
            removed.append(snp_id)
            continue

        used_groups.add(group_id)
        kept.append(row)

    return kept, removed


def check_redundancy(
    matrix: List[List[str]],
    ranked_snp_ids: List[str],
    similarity_threshold: float,
    redundancy: int
) -> List[str]:
    """
    Identify redundant SNPs while preserving the statistically best-ranked
    representatives of each redundancy cluster.

    `matrix` must be transposed SNP x genome data:
        [SNP_ID, state_genome_1, state_genome_2, ...]

    `similarity_threshold` is interpreted as a maximum Hamming mismatch
    percentage. Complete-linkage clustering is used so that every pair of SNP
    patterns inside a cluster differs by no more than the requested threshold.

    SNPs are NOT removed randomly. `ranked_snp_ids` must be ordered from best to
    worst statistical efficacy; up to `redundancy` best-ranked SNPs are retained
    from each redundancy cluster.
    """

    import numpy as np
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import pdist

    if not 0 <= similarity_threshold <= 50:
        raise ValueError("similarity_threshold must be between 0 and 50.")

    if redundancy < 1:
        raise ValueError("redundancy must be >= 1.")

    if not matrix or not ranked_snp_ids:
        return []

    row_length = len(matrix[0])
    if row_length < 2:
        raise ValueError(
            "Each transposed matrix row must contain an SNP ID followed by allelic states."
        )

    pattern_by_id: Dict[str, List[str]] = {}

    for row_number, row in enumerate(matrix, start=1):
        if len(row) != row_length:
            raise ValueError(
                f"Transposed SNP matrix row {row_number} has {len(row)} columns; "
                f"expected {row_length}."
            )

        snp_id = str(row[0]).strip()
        if not snp_id:
            raise ValueError(f"Empty SNP ID in transposed matrix row {row_number}.")

        if snp_id in pattern_by_id:
            raise ValueError(f"Duplicate SNP ID in SNP matrix: {snp_id}")

        pattern_by_id[snp_id] = [
            str(value).strip().upper()
            for value in row[1:]
        ]

    available_ids = [snp_id for snp_id in ranked_snp_ids if snp_id in pattern_by_id]

    if len(available_ids) <= 1:
        return []

    # Encode arbitrary symbolic states as integers. Hamming distance only cares
    # whether two states are equal, so a global symbol mapping is sufficient.
    symbols = sorted({
        state
        for snp_id in available_ids
        for state in pattern_by_id[snp_id]
    })
    symbol_to_int = {symbol: i for i, symbol in enumerate(symbols)}

    encoded = np.asarray(
        [
            [symbol_to_int[state] for state in pattern_by_id[snp_id]]
            for snp_id in available_ids
        ],
        dtype=np.int16
    )

    condensed = pdist(encoded, metric="hamming")

    tree = linkage(condensed, method="complete")
    threshold = similarity_threshold / 100.0
    cluster_ids = fcluster(tree, t=threshold, criterion="distance")

    clusters: Dict[int, List[str]] = {}
    for snp_id, cluster_id in zip(available_ids, cluster_ids):
        clusters.setdefault(int(cluster_id), []).append(snp_id)

    rank = {snp_id: i for i, snp_id in enumerate(ranked_snp_ids)}
    snp_to_remove: List[str] = []

    for members in clusters.values():
        if len(members) <= redundancy:
            continue

        members.sort(key=lambda snp_id: rank[snp_id])
        snp_to_remove.extend(members[redundancy:])

    return snp_to_remove


def main():
    args = parse_args()

    # ------------------------------------------------------------
    # Validate scalar arguments
    # ------------------------------------------------------------
    if not 0.0 <= args.p_value <= 1.0:
        raise ValueError("--p_value must be between 0 and 1.")

    if not 0.0 <= args.fdr_value <= 1.0:
        raise ValueError("--fdr_value must be between 0 and 1.")

    if not 0.0 <= args.global_fdr_value <= 1.0:
        raise ValueError("--global_fdr_value must be between 0 and 1.")

    if args.minimum_distance_between_SNP < 0:
        raise ValueError("--minimum_distance_between_SNP must be >= 0.")

    if args.redundancy < 1:
        raise ValueError("--redundancy must be >= 1.")

    if not 0.0 <= args.similarity_threshold <= 50.0:
        raise ValueError("--similarity_threshold must be between 0 and 50.")

    if args.feature_number_to_keep < 1:
        raise ValueError("--feature_number_to_keep must be >= 1.")

    # ------------------------------------------------------------
    # Input/output project folders
    # ------------------------------------------------------------
    in_path = os.path.join(args.input_folder, args.project_folder)
    if not os.path.isdir(in_path):
        raise FileNotFoundError(f"Project folder does not exist: {in_path}")

    out_path = os.path.join(args.output_folder, args.project_folder)
    os.makedirs(out_path, exist_ok=True)

    feature_file = os.path.join(in_path, args.feature_file)
    allele_file = os.path.join(in_path, args.allele_file)
    prediction_file = os.path.join(in_path, args.prediction_file)
    matrix_file = os.path.join(in_path, args.matrix_file)

    for label, path in (
        ("feature", feature_file),
        ("allele", allele_file),
        ("prediction", prediction_file),
        ("matrix", matrix_file),
    ):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Input {label} file does not exist: {path}")

    # ------------------------------------------------------------
    # Read feature annotation file
    # ------------------------------------------------------------
    features = parse_matrix(feature_file)

    if len(features) < 3:
        raise ValueError(f"Feature file contains too few rows: {feature_file}")

    # Expected structure:
    # row 0 = #Reference genome ...
    # row 1 = SNP / alignment_position / Location in reference sequence / ...
    # row 2+ = feature records
    headings = features[:2]
    feature_header = features[1]
    feature_body = features[2:]

    # ------------------------------------------------------------
    # Read allele file
    # ------------------------------------------------------------
    alleles = parse_matrix(allele_file)
    if len(alleles) < 2:
        raise ValueError(
            "Allele file must contain at least two rows: SNP IDs and allelic states."
        )

    if len(alleles[0]) != len(alleles[1]):
        raise ValueError(
            "Allele file SNP-ID row and allelic-state row have different lengths."
        )

    all_alleles = dict(zip(alleles[0], alleles[1]))

    # ------------------------------------------------------------
    # Read and transpose SNP matrix
    # ------------------------------------------------------------
    matrix_rows = parse_matrix(matrix_file)
    snp_matrix = transpose_snp_matrix(matrix_rows)
    matrix_snp_ids = {row[0] for row in snp_matrix}

    # ------------------------------------------------------------
    # Read prediction file
    # ------------------------------------------------------------
    prediction_table = parse_matrix(prediction_file)

    if len(prediction_table) < 2:
        raise ValueError(
            f"Prediction file contains no prediction records: {prediction_file}"
        )

    prediction_header = prediction_table[0]
    predictions = prediction_table[1:]

    # New prediction-file format: one record per feature per hierarchy level.
    required_prediction_columns = [
        "Level",
        "Location",
        "p-value",
        "Level FDR p-value",
        "Global FDR p-value",
    ]

    missing_columns = [
        name for name in required_prediction_columns
        if name not in prediction_header
    ]
    if missing_columns:
        raise ValueError(
            "Prediction table is missing required column(s): "
            + ", ".join(missing_columns)
        )

    level_col = prediction_header.index("Level")
    snp_id_col = prediction_header.index("Location")
    raw_p_col = prediction_header.index("p-value")
    level_fdr_col = prediction_header.index("Level FDR p-value")
    global_fdr_col = prediction_header.index("Global FDR p-value")

    # Report the hierarchy levels represented in the prediction table.
    detected_levels = sorted({
        row[level_col].strip()
        for row in predictions
        if len(row) > level_col and row[level_col].strip()
    })
    print(
        "Hierarchical levels represented in predictions: "
        + (", ".join(detected_levels) if detected_levels else "none")
    )

    # ------------------------------------------------------------
    # 1. Statistical filtering
    # ------------------------------------------------------------
    # A SNP is eligible if it is significant at AT LEAST ONE hierarchy level.
    # Because the new prediction file contains one row per level, several rows
    # can refer to the same SNP.  Keep only the statistically best qualifying
    # row for each SNP; the chosen row is then used to rank the feature before
    # neighbourhood and redundancy filtering.
    statistically_valid_by_snp: Dict[str, List[str]] = {}
    malformed_predictions = 0

    for row in predictions:
        max_required_col = max(
            snp_id_col, raw_p_col, level_fdr_col, global_fdr_col
        )
        if len(row) <= max_required_col:
            malformed_predictions += 1
            continue

        snp_id = row[snp_id_col].strip()
        if not snp_id:
            malformed_predictions += 1
            continue

        try:
            raw_p = float(row[raw_p_col])
            level_fdr = float(row[level_fdr_col])
            global_fdr = float(row[global_fdr_col])
        except ValueError:
            malformed_predictions += 1
            continue

        if (
            raw_p > args.p_value
            or level_fdr > args.fdr_value
            or global_fdr > args.global_fdr_value
        ):
            continue

        previous = statistically_valid_by_snp.get(snp_id)
        if previous is None:
            statistically_valid_by_snp[snp_id] = row
            continue

        current_rank = (global_fdr, level_fdr, raw_p)
        previous_rank = (
            float(previous[global_fdr_col]),
            float(previous[level_fdr_col]),
            float(previous[raw_p_col]),
        )

        if current_rank < previous_rank:
            statistically_valid_by_snp[snp_id] = row

    statistically_valid = list(statistically_valid_by_snp.values())

    # Rank features before applying neighbourhood/redundancy filters.
    # Lower Global FDR is best; Level FDR and raw p-value are tie-breakers.
    statistically_valid.sort(
        key=lambda row: (
            float(row[global_fdr_col]),
            float(row[level_fdr_col]),
            float(row[raw_p_col]),
            row[snp_id_col],
        )
    )

    print(
        f"Statistically valid unique SNPs (significant at >=1 level): "
        f"{len(statistically_valid)}"
    )
    if malformed_predictions:
        print(
            f"WARNING: skipped malformed prediction rows: "
            f"{malformed_predictions}"
        )

    # ------------------------------------------------------------
    # 2. Genomic-neighborhood filtering
    # ------------------------------------------------------------
    neighbourhoods, neighbourhood_membership = build_neighbourhoods(
        feature_body=feature_body,
        header=feature_header,
        minimum_distance=args.minimum_distance_between_SNP
    )

    after_neighbourhood, removed_neighbours = keep_best_from_neighbourhoods(
        statistically_valid,
        neighbourhood_membership,
        snp_id_col
    )

    print(f"Reference SNP neighbourhoods detected: {len(neighbourhoods)}")
    print(f"SNPs removed because a better nearby SNP was retained: {len(removed_neighbours)}")
    print(f"SNPs after neighbourhood filtering: {len(after_neighbourhood)}")

    # ------------------------------------------------------------
    # 3. Redundancy filtering
    # ------------------------------------------------------------
    ranked_ids = [row[snp_id_col] for row in after_neighbourhood]
    missing_in_matrix = [snp_id for snp_id in ranked_ids if snp_id not in matrix_snp_ids]

    if missing_in_matrix:
        print(
            f"WARNING: {len(missing_in_matrix)} statistically selected SNPs are absent "
            "from the SNP matrix and cannot be assessed for redundancy."
        )

    snp_to_remove = check_redundancy(
        matrix=snp_matrix,
        ranked_snp_ids=ranked_ids,
        similarity_threshold=args.similarity_threshold,
        redundancy=args.redundancy
    )

    remove_set = set(snp_to_remove)
    after_redundancy = [
        row
        for row in after_neighbourhood
        if row[snp_id_col] not in remove_set
    ]

    print(f"SNPs removed as redundant: {len(snp_to_remove)}")
    print(f"SNPs after redundancy filtering: {len(after_redundancy)}")

    # ------------------------------------------------------------
    # Optional evaluation of feature-matrix size versus RF accuracy
    # ------------------------------------------------------------
    if args.evaluate is not None:
        raw_values = [value.strip() for value in args.evaluate.split(",") if value.strip()]
        if not raw_values:
            print("WARNING: --evaluate was set but contains no values; evaluation skipped.")
        else:
            try:
                requested_sizes = [int(value) for value in raw_values]
            except ValueError as exc:
                raise ValueError(
                    "--evaluate must be a comma-separated list of integers, "
                    "for example: --evaluate 50,100,1000"
                ) from exc

            evaluation_ranked_feature_ids = [
                row[snp_id_col]
                for row in after_redundancy
                if row[snp_id_col] in matrix_snp_ids
            ]
            current_feature_count = len(evaluation_ranked_feature_ids)
            valid_sizes: List[int] = []
            removed_sizes: List[int] = []

            for size in requested_sizes:
                if size < 10 or size > current_feature_count:
                    removed_sizes.append(size)
                elif size not in valid_sizes:
                    valid_sizes.append(size)

            valid_sizes.sort()

            if removed_sizes:
                print(
                    "WARNING: removed --evaluate value(s) outside the allowed range "
                    f"10..{current_feature_count}: "
                    + ", ".join(str(value) for value in removed_sizes)
                )

            if len(valid_sizes) < len([v for v in requested_sizes if 10 <= v <= current_feature_count]):
                print("INFO: duplicate --evaluate values were removed.")

            if valid_sizes:
                script_dir = os.path.dirname(os.path.abspath(__file__))
                lib_dir = os.path.abspath(os.path.join(script_dir, "lib"))
                if lib_dir not in sys.path:
                    sys.path.insert(0, lib_dir)

                try:
                    from evavuate_matrix_size_to_accuracy import main as evaluate_matrix_accuracy
                except ImportError as exc:
                    raise ImportError(
                        "Could not import ../lib/evavuate_matrix_size_to_accuracy.py. "
                        f"Expected library directory: {lib_dir}"
                    ) from exc

                matrix_stem = os.path.splitext(os.path.basename(args.matrix_file))[0]
                evaluate_matrix_accuracy(
                    matrix_rows=matrix_rows,
                    ranked_feature_ids=evaluation_ranked_feature_ids,
                    feature_counts=valid_sizes,
                    output_dir=out_path,
                    output_prefix=matrix_stem,
                )
            else:
                print(
                    "WARNING: no valid --evaluate values remain after filtering; "
                    "RF accuracy evaluation skipped."
                )

    # ------------------------------------------------------------
    # 4. Final feature-number limit
    # ------------------------------------------------------------
    selected_predictions = after_redundancy[:args.feature_number_to_keep]

    print(
        f"Final selected SNPs: {len(selected_predictions)} "
        f"(maximum requested: {args.feature_number_to_keep})"
    )

    # ------------------------------------------------------------
    # Create feature lookup
    # ------------------------------------------------------------
    feature_ids = [row[0] for row in feature_body if row]

    if len(set(feature_ids)) != len(feature_ids):
        raise ValueError("Feature annotation file contains duplicate SNP IDs.")

    all_features = dict(zip(feature_ids, feature_body))

    # ------------------------------------------------------------
    # Retrieve annotation and allele records
    # ------------------------------------------------------------
    selected_feature_records: List[List[str]] = []
    selected_alleles: List[List[str]] = [[], []]
    selected_feature_ids: List[str] = []
    missing_features: List[str] = []
    missing_alleles: List[str] = []
    missing_matrix: List[str] = []

    for prediction in selected_predictions:
        feature_id = prediction[snp_id_col]

        record = all_features.get(feature_id)
        if record is None:
            missing_features.append(feature_id)
            continue

        allele = all_alleles.get(feature_id)
        if allele is None:
            missing_alleles.append(feature_id)
            continue

        # Keep all three output files synchronized.  A selected SNP must also
        # exist as a column in the input matrix before it is written to any
        # condensed output.
        if feature_id not in matrix_snp_ids:
            missing_matrix.append(feature_id)
            continue

        selected_feature_ids.append(feature_id)
        selected_feature_records.append(record)
        selected_alleles[0].append(feature_id)
        selected_alleles[1].append(allele)

    if missing_features:
        print(
            f"WARNING: {len(missing_features)} selected SNPs were not found in the "
            "feature annotation file."
        )

    if missing_alleles:
        print(
            f"WARNING: {len(missing_alleles)} selected SNPs were not found in the allele file."
        )

    if missing_matrix:
        print(
            f"WARNING: {len(missing_matrix)} selected SNPs were not found in the input matrix "
            "and were omitted from all condensed outputs."
        )

    selected_features = headings + selected_feature_records

    # ------------------------------------------------------------
    # Build filtered SNP matrix
    # ------------------------------------------------------------
    # Preserve genome rows and the original first-column genome/strain labels,
    # but retain only the final selected SNP columns.  SNP columns are written
    # in the same order as selected_feature_ids so that the matrix, annotation
    # file, and allele-state file describe the same ordered feature set.
    matrix_header = matrix_rows[0]

    if not matrix_header:
        raise ValueError("Input SNP matrix has an empty header row.")

    matrix_column_index: Dict[str, int] = {}
    for column_index, column_name in enumerate(matrix_header[1:], start=1):
        snp_id = str(column_name).strip()
        if snp_id in matrix_column_index:
            raise ValueError(f"Input SNP matrix contains duplicate SNP column: {snp_id}")
        matrix_column_index[snp_id] = column_index

    selected_matrix: List[List[str]] = [
        [matrix_header[0]] + selected_feature_ids
    ]

    for row_number, row in enumerate(matrix_rows[1:], start=2):
        if not row:
            continue

        values = [row[0]]
        for feature_id in selected_feature_ids:
            column_index = matrix_column_index[feature_id]
            if column_index >= len(row):
                raise ValueError(
                    f"Input SNP matrix row {row_number} is too short for selected "
                    f"feature {feature_id}."
                )
            values.append(row[column_index])

        selected_matrix.append(values)

    # ------------------------------------------------------------
    # Output filenames
    # ------------------------------------------------------------
    allele_stem, allele_ext = os.path.splitext(args.allele_file)
    out_allele_file = os.path.join(
        out_path,
        allele_stem + "_condensed" + allele_ext
    )
    allele_delimiter = "\t" if allele_ext.lower() in {".tsv", ".txt"} else ","

    feature_stem, feature_ext = os.path.splitext(args.feature_file)
    out_feature_file = os.path.join(
        out_path,
        feature_stem + "_condensed" + feature_ext
    )
    feature_delimiter = "\t" if feature_ext.lower() in {".tsv", ".txt"} else ","

    matrix_stem, matrix_ext = os.path.splitext(args.matrix_file)
    out_matrix_file = os.path.join(
        out_path,
        matrix_stem + "_condensed" + matrix_ext
    )
    matrix_delimiter = "\t" if matrix_ext.lower() in {".tsv", ".txt"} else ","

    # ------------------------------------------------------------
    # Write results
    # ------------------------------------------------------------
    with open(out_feature_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(
            f,
            delimiter=feature_delimiter,
            lineterminator="\n"
        )
        writer.writerows(selected_features)

    print(f"Selected feature annotations saved to: {out_feature_file}")

    with open(out_allele_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(
            f,
            delimiter=allele_delimiter,
            lineterminator="\n"
        )
        writer.writerows(selected_alleles)

    print(f"Selected alleles saved to: {out_allele_file}")

    with open(out_matrix_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(
            f,
            delimiter=matrix_delimiter,
            lineterminator="\n"
        )
        writer.writerows(selected_matrix)

    print(f"Selected SNP matrix saved to: {out_matrix_file}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)
