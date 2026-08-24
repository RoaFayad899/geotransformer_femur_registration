# ============================================================
# inference_real_us_originalscale.py
#
# PURE INFERENCE ONLY -- NO TRAINING
#
# New dataset structure:
#   1 original aligned case
#   3 translation-only cases
#   3 rotation-only cases
#   3 combined cases
#   = 10 samples total
#
# IMPORTANT:
# - source = perturbed real US
# - target = fixed CT
# - T_gt maps source -> reference alignment
#
# The original_aligned case has:
#   T_extra = Identity
#   T_gt    = Identity
#
# Main aggregate statistics are calculated ONLY on the
# 9 artificially perturbed samples so they remain comparable
# with the previous 9-case inference experiments.
#
# The original aligned case is reported separately.
# ============================================================


import os
import re
import glob
import json
import csv

import numpy as np
import torch
import open3d as o3d

from geotransformer.utils.data import registration_collate_fn_stack_mode
from geotransformer.utils.torch import to_cuda, release_cuda

from config import make_cfg
from model import create_model


# ============================================================
# CHANGE ONLY THESE TWO FOR EACH RUN
# ============================================================

# ------------------------------------------------------------
# PROXIMAL example
# ------------------------------------------------------------

INFERENCE_DATASET_NAME = (                                                   #######################################
    "geotransformer_dataset_inference_"
    "USProximal_to_CTIntact_originalscale_new"
)

EXPERIMENT_NAME = (                                                          ##########################################
    "exp_fulldataset_best4stages_2000_large_00"
)


# ------------------------------------------------------------
# DISTAL:
#
# Replace INFERENCE_DATASET_NAME above with:
#
# INFERENCE_DATASET_NAME = (
#     "geotransformer_dataset_inference_"
#     "USDistal_to_CTIntact_originalscale_new"
# )
# ------------------------------------------------------------


# ============================================================
# VISUALIZATION OPTIONS
# ============================================================

VISUALIZE_BEST = True

# True:
#   Gray  = CT target
#   Green = predicted US
#   Blue  = ground-truth US
#
# False:
#   Gray  = CT target
#   Green = predicted US

SHOW_GROUND_TRUTH = True


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = (
    "/home/roa.fayad/"
    "geotransformer_femur_registration"
)

INFERENCE_DATASET_ROOT = os.path.join(
    PROJECT_ROOT,
    "data",
    "inference",
    INFERENCE_DATASET_NAME,
)

EXPERIMENT_ROOT = os.path.join(
    PROJECT_ROOT,
    "output",
    EXPERIMENT_NAME,
)

SNAPSHOT_DIR = os.path.join(
    EXPERIMENT_ROOT,
    "snapshots",
)

RESULT_DIR = os.path.join(
    EXPERIMENT_ROOT,
    "real_us_inference",
    INFERENCE_DATASET_NAME,
)

os.makedirs(
    RESULT_DIR,
    exist_ok=True,
)


# ============================================================
# KPConv NEIGHBOR LIMITS
# ============================================================

# Keep exactly the same values used by your previous
# inference script / training configuration.

NEIGHBOR_LIMITS = [
    67,
    71,
    93,
    61,
]


# ============================================================
# CHECKPOINT
# ============================================================

def find_checkpoint(snapshot_dir):

    patterns = [
        os.path.join(
            snapshot_dir,
            "*.pth.tar",
        ),
        os.path.join(
            snapshot_dir,
            "*.pth",
        ),
        os.path.join(
            snapshot_dir,
            "*.pt",
        ),
    ]

    files = []

    for pattern in patterns:
        files.extend(
            glob.glob(pattern)
        )

    if len(files) == 0:
        raise FileNotFoundError(
            f"\nNo checkpoint found in:\n"
            f"{snapshot_dir}"
        )

    # --------------------------------------------------------
    # Prefer checkpoint explicitly containing "best"
    # --------------------------------------------------------

    best_files = [
        f
        for f in files
        if "best" in os.path.basename(f).lower()
    ]

    if len(best_files) > 0:
        return max(
            best_files,
            key=os.path.getmtime,
        )

    # --------------------------------------------------------
    # Otherwise highest epoch
    # --------------------------------------------------------

    def extract_epoch(path):

        name = os.path.basename(path).lower()

        matches = re.findall(
            r"(?:epoch|snapshot)[-_]?(\d+)",
            name,
        )

        if matches:
            return int(matches[-1])

        numbers = re.findall(
            r"\d+",
            name,
        )

        if numbers:
            return int(numbers[-1])

        return -1

    epoch_files = [
        (
            extract_epoch(f),
            f,
        )
        for f in files
    ]

    epoch_files.sort(
        key=lambda x: x[0]
    )

    if epoch_files[-1][0] >= 0:
        return epoch_files[-1][1]

    return max(
        files,
        key=os.path.getmtime,
    )


def load_checkpoint(
    model,
    checkpoint_path,
):

    print("\nLoading checkpoint:")
    print(checkpoint_path)

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
    )

    if (
        isinstance(checkpoint, dict)
        and "model" in checkpoint
    ):
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    model.load_state_dict(
        state_dict,
        strict=True,
    )

    return model


# ============================================================
# LOAD INFERENCE SAMPLE
# ============================================================

def load_npz_sample(file_path):

    data = np.load(
        file_path,
        allow_pickle=True,
    )

    required_keys = [
        "source",
        "target",
        "T_gt",
    ]

    for key in required_keys:

        if key not in data:

            raise KeyError(
                f"\n{file_path}\n"
                f"does not contain '{key}'.\n"
                f"Available keys:\n"
                f"{list(data.keys())}"
            )

    # --------------------------------------------------------
    # Convention
    #
    # source = perturbed real US
    # target = fixed CT
    #
    # T_gt maps:
    # source --> reference alignment
    # --------------------------------------------------------

    src_points = np.asarray(
        data["source"],
        dtype=np.float32,
    )

    ref_points = np.asarray(
        data["target"],
        dtype=np.float32,
    )

    transform = np.asarray(
        data["T_gt"],
        dtype=np.float32,
    )

    # --------------------------------------------------------
    # Basic checks
    # --------------------------------------------------------

    if (
        src_points.ndim != 2
        or src_points.shape[1] != 3
    ):
        raise ValueError(
            f"source must have shape (N,3), "
            f"got {src_points.shape}"
        )

    if (
        ref_points.ndim != 2
        or ref_points.shape[1] != 3
    ):
        raise ValueError(
            f"target must have shape (M,3), "
            f"got {ref_points.shape}"
        )

    if transform.shape != (4, 4):
        raise ValueError(
            f"T_gt must have shape (4,4), "
            f"got {transform.shape}"
        )

    # --------------------------------------------------------
    # Sample ID
    # --------------------------------------------------------

    if "sample_id" in data:

        sample_id = str(
            np.asarray(
                data["sample_id"]
            ).item()
        )

    else:

        sample_id = (
            os.path.splitext(
                os.path.basename(file_path)
            )[0]
        )

    # --------------------------------------------------------
    # NEW:
    # case_name
    #
    # Allows us to identify:
    # original_aligned
    # translation_...
    # rotation_...
    # combined_...
    # --------------------------------------------------------

    if "case_name" in data:

        case_name = str(
            np.asarray(
                data["case_name"]
            ).item()
        )

    else:

        case_name = (
            os.path.splitext(
                os.path.basename(file_path)
            )[0]
        )

    # --------------------------------------------------------
    # Normalization scale
    # --------------------------------------------------------

    if "normalization_scale" in data:

        normalization_scale = float(
            np.asarray(
                data[
                    "normalization_scale"
                ]
            ).reshape(-1)[0]
        )

    else:

        normalization_scale = np.nan

    # --------------------------------------------------------
    # Translation magnitude
    # --------------------------------------------------------

    if "translation_magnitude" in data:

        translation_magnitude = float(
            np.asarray(
                data[
                    "translation_magnitude"
                ]
            ).reshape(-1)[0]
        )

    else:

        translation_magnitude = np.nan

    # --------------------------------------------------------
    # Rotation angle
    # --------------------------------------------------------

    if "rotation_angle_deg" in data:

        rotation_angle_deg = float(
            np.asarray(
                data[
                    "rotation_angle_deg"
                ]
            ).reshape(-1)[0]
        )

    else:

        rotation_angle_deg = np.nan

    return {

        "source":
            src_points,

        "target":
            ref_points,

        "T_gt":
            transform,

        "sample_id":
            sample_id,

        "case_name":
            case_name,

        "normalization_scale":
            normalization_scale,

        "translation_magnitude":
            translation_magnitude,

        "rotation_angle_deg":
            rotation_angle_deg,
    }


# ============================================================
# GEOTRANSFORMER INPUT PREPROCESSING
# ============================================================

def prepare_geotransformer_input(
    sample,
    cfg,
):

    src_points = sample["source"]
    ref_points = sample["target"]

    transform = sample["T_gt"]

    # --------------------------------------------------------
    # Constant point features
    # --------------------------------------------------------

    src_feats = np.ones(
        (
            src_points.shape[0],
            1,
        ),
        dtype=np.float32,
    )

    ref_feats = np.ones(
        (
            ref_points.shape[0],
            1,
        ),
        dtype=np.float32,
    )

    data_dict = {

        "ref_points":
            ref_points.astype(
                np.float32
            ),

        "src_points":
            src_points.astype(
                np.float32
            ),

        "ref_feats":
            ref_feats,

        "src_feats":
            src_feats,

        "transform":
            transform.astype(
                np.float32
            ),
    }

    # --------------------------------------------------------
    # KPConv preprocessing
    # --------------------------------------------------------

    data_dict = (
        registration_collate_fn_stack_mode(
            [data_dict],
            cfg.backbone.num_stages,
            cfg.backbone.init_voxel_size,
            cfg.backbone.init_radius,
            NEIGHBOR_LIMITS,
        )
    )

    return data_dict


# ============================================================
# TRANSFORMATION UTILITIES
# ============================================================

def apply_transform_np(
    points,
    transform,
):

    R = transform[
        :3,
        :3,
    ]

    t = transform[
        :3,
        3,
    ]

    transformed_points = (
        points @ R.T
        + t
    )

    return transformed_points


# ============================================================
# METRICS
# ============================================================

def compute_rre_rte(
    T_gt,
    T_pred,
):

    R_gt = T_gt[
        :3,
        :3,
    ]

    t_gt = T_gt[
        :3,
        3,
    ]

    R_pred = T_pred[
        :3,
        :3,
    ]

    t_pred = T_pred[
        :3,
        3,
    ]

    # --------------------------------------------------------
    # Relative Rotation Error
    # --------------------------------------------------------

    R_error = (
        R_pred.T
        @ R_gt
    )

    trace = np.trace(
        R_error
    )

    cos_theta = (
        trace - 1.0
    ) / 2.0

    cos_theta = np.clip(
        cos_theta,
        -1.0,
        1.0,
    )

    rre = np.degrees(
        np.arccos(
            cos_theta
        )
    )

    # --------------------------------------------------------
    # Relative Translation Error
    # --------------------------------------------------------

    rte = np.linalg.norm(
        t_pred
        - t_gt
    )

    return (
        float(rre),
        float(rte),
    )


def compute_rmse(
    source_points,
    T_gt,
    T_pred,
):

    gt_points = (
        apply_transform_np(
            source_points,
            T_gt,
        )
    )

    predicted_points = (
        apply_transform_np(
            source_points,
            T_pred,
        )
    )

    squared_distances = np.sum(
        (
            predicted_points
            - gt_points
        ) ** 2,
        axis=1,
    )

    rmse = np.sqrt(
        np.mean(
            squared_distances
        )
    )

    return float(rmse)


def safe_mm(
    normalized_value,
    normalization_scale,
):

    if np.isnan(
        normalization_scale
    ):
        return np.nan

    return float(
        normalized_value
        * normalization_scale
    )


# ============================================================
# VISUALIZATION
# ============================================================

def visualize_registration(
    source,
    target,
    T_pred,
    T_gt=None,
    title="GeoTransformer Registration",
    show_ground_truth=True,
):

    source = np.asarray(
        source,
        dtype=np.float64,
    )

    target = np.asarray(
        target,
        dtype=np.float64,
    )

    # --------------------------------------------------------
    # Predicted US
    # --------------------------------------------------------

    predicted_source = (
        apply_transform_np(
            source,
            T_pred,
        )
    )

    # --------------------------------------------------------
    # CT target
    # --------------------------------------------------------

    target_pcd = (
        o3d.geometry.PointCloud()
    )

    target_pcd.points = (
        o3d.utility.Vector3dVector(
            target
        )
    )

    target_pcd.paint_uniform_color(
        [
            0.65,
            0.65,
            0.65,
        ]
    )

    # --------------------------------------------------------
    # Predicted US
    # --------------------------------------------------------

    predicted_pcd = (
        o3d.geometry.PointCloud()
    )

    predicted_pcd.points = (
        o3d.utility.Vector3dVector(
            predicted_source
        )
    )

    predicted_pcd.paint_uniform_color(
        [
            0.10,
            0.90,
            0.10,
        ]
    )

    geometries = [
        target_pcd,
        predicted_pcd,
    ]

    # --------------------------------------------------------
    # Ground-truth US
    # --------------------------------------------------------

    if (
        show_ground_truth
        and T_gt is not None
    ):

        gt_source = (
            apply_transform_np(
                source,
                T_gt,
            )
        )

        gt_pcd = (
            o3d.geometry.PointCloud()
        )

        gt_pcd.points = (
            o3d.utility.Vector3dVector(
                gt_source
            )
        )

        gt_pcd.paint_uniform_color(
            [
                0.10,
                0.30,
                1.00,
            ]
        )

        geometries.append(
            gt_pcd
        )

    print("\nVisualization colors:")
    print("Gray  = CT target")
    print(
        "Green = US transformed "
        "with T_pred"
    )

    if show_ground_truth:
        print(
            "Blue  = US transformed "
            "with T_gt"
        )

    try:

        o3d.visualization.draw_geometries(
            geometries,
            window_name=title,
            width=1200,
            height=900,
        )

    except Exception as error:

        print(
            "\nWARNING:"
        )

        print(
            "Open3D visualization "
            "could not be opened."
        )

        print(
            "This can happen on a "
            "remote server without "
            "graphical display."
        )

        print(error)


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "=" * 80
    )

    print(
        "REAL US -> CT "
        "GEOTRANSFORMER INFERENCE"
    )

    print(
        "=" * 80
    )

    print(
        "\nDataset:"
    )

    print(
        INFERENCE_DATASET_ROOT
    )

    print(
        "\nExperiment:"
    )

    print(
        EXPERIMENT_NAME
    )


    # ========================================================
    # CONFIG
    # ========================================================

    cfg = make_cfg()


    # ========================================================
    # FIND ALL NPZ FILES
    #
    # Expected for the new datasets:
    # 10 samples
    # ========================================================

    npz_files = sorted(
        glob.glob(
            os.path.join(
                INFERENCE_DATASET_ROOT,
                "**",
                "*.npz",
            ),
            recursive=True,
        )
    )

    if len(npz_files) == 0:

        raise FileNotFoundError(
            f"\nNo .npz files "
            f"found in:\n"
            f"{INFERENCE_DATASET_ROOT}"
        )

    print(
        f"\nFound "
        f"{len(npz_files)} "
        f"samples."
    )

    for file_path in npz_files:

        print(
            "  ",
            os.path.basename(
                file_path
            ),
        )


    # ========================================================
    # LOAD TRAINED MODEL
    # ========================================================

    checkpoint_path = (
        find_checkpoint(
            SNAPSHOT_DIR
        )
    )

    model = (
        create_model(
            cfg
        ).cuda()
    )

    model = (
        load_checkpoint(
            model,
            checkpoint_path,
        )
    )

    # PURE INFERENCE
    model.eval()


    # ========================================================
    # STORAGE
    # ========================================================

    rows = []

    best_rmse = np.inf

    best_visualization = None


    # ========================================================
    # LOOP THROUGH ALL 10 TEST CASES
    # ========================================================

    for index, file_path in enumerate(
        npz_files,
        start=1,
    ):

        print(
            "\n"
            + "-" * 80
        )

        print(
            f"[{index}/"
            f"{len(npz_files)}] "
            f"{os.path.basename(file_path)}"
        )

        # ----------------------------------------------------
        # Load sample
        # ----------------------------------------------------

        sample = (
            load_npz_sample(
                file_path
            )
        )

        print(
            "Case          :",
            sample["case_name"]
        )

        # ----------------------------------------------------
        # Prepare GeoTransformer input
        # ----------------------------------------------------

        data_dict = (
            prepare_geotransformer_input(
                sample,
                cfg,
            )
        )

        data_dict = (
            to_cuda(
                data_dict
            )
        )

        # ----------------------------------------------------
        # PURE INFERENCE
        # ----------------------------------------------------

        with torch.no_grad():

            output_dict = (
                model(
                    data_dict
                )
            )

        # ----------------------------------------------------
        # CPU
        # ----------------------------------------------------

        output_dict = (
            release_cuda(
                output_dict
            )
        )

        data_dict = (
            release_cuda(
                data_dict
            )
        )


        # ====================================================
        # COARSE / FINE CORRESPONDENCES
        # ====================================================

        num_coarse_corr = len(
            np.asarray(
                output_dict[
                    "ref_node_corr_indices"
                ]
            ).reshape(-1)
        )

        corr_scores_array = np.asarray(
            output_dict[
                "corr_scores"
            ]
        )

        if corr_scores_array.ndim == 0:
            num_fine_corr = 1
        else:
            num_fine_corr = (
                corr_scores_array.shape[0]
            )

        print(
            f"Coarse correspondences : "
            f"{num_coarse_corr}"
        )

        print(
            f"Fine correspondences   : "
            f"{num_fine_corr}"
        )


        # ====================================================
        # PREDICTED TRANSFORM
        # ====================================================

        T_pred = np.asarray(
            output_dict[
                "estimated_transform"
            ],
            dtype=np.float64,
        )

        T_gt = (
            sample[
                "T_gt"
            ].astype(
                np.float64
            )
        )


        # ====================================================
        # METRICS
        # ====================================================

        rre, rte = (
            compute_rre_rte(
                T_gt,
                T_pred,
            )
        )

        rmse = (
            compute_rmse(
                sample[
                    "source"
                ].astype(
                    np.float64
                ),
                T_gt,
                T_pred,
            )
        )

        normalization_scale = (
            sample[
                "normalization_scale"
            ]
        )

        rte_mm = safe_mm(
            rte,
            normalization_scale,
        )

        rmse_mm = safe_mm(
            rmse,
            normalization_scale,
        )


        # ====================================================
        # NUMBER OF FINAL CORRESPONDENCES
        # ====================================================

        if "corr_scores" in output_dict:

            corr_scores_array = np.asarray(
                output_dict[
                    "corr_scores"
                ]
            )

            num_corr = (
                1
                if corr_scores_array.ndim == 0
                else corr_scores_array.shape[0]
            )

        else:

            num_corr = 0


        # ====================================================
        # SAVE BEST SAMPLE FOR VISUALIZATION
        # ====================================================

        if rmse < best_rmse:

            best_rmse = rmse

            best_visualization = {

                "sample_id":
                    sample[
                        "sample_id"
                    ],

                "case_name":
                    sample[
                        "case_name"
                    ],

                "source":
                    sample[
                        "source"
                    ].copy(),

                "target":
                    sample[
                        "target"
                    ].copy(),

                "T_pred":
                    T_pred.copy(),

                "T_gt":
                    T_gt.copy(),

                "rre":
                    rre,

                "rte":
                    rte,

                "rte_mm":
                    rte_mm,

                "rmse":
                    rmse,

                "rmse_mm":
                    rmse_mm,
            }


        # ====================================================
        # RESULT ROW
        # ====================================================

        row = {

            "sample_id":
                sample[
                    "sample_id"
                ],

            # NEW
            "case_name":
                sample[
                    "case_name"
                ],

            "file":
                os.path.basename(
                    file_path
                ),

            "translation_magnitude":
                sample[
                    "translation_magnitude"
                ],

            "rotation_angle_deg":
                sample[
                    "rotation_angle_deg"
                ],

            "normalization_scale":
                normalization_scale,

            "num_correspondences":
                int(
                    num_corr
                ),

            "RRE_deg":
                rre,

            "RTE_normalized":
                rte,

            "RTE_mm":
                rte_mm,

            "RMSE_normalized":
                rmse,

            "RMSE_mm":
                rmse_mm,
        }

        rows.append(
            row
        )


        # ====================================================
        # SAVE PER-SAMPLE RESULT
        # ====================================================

        sample_result_path = (
            os.path.join(
                RESULT_DIR,
                (
                    f"{sample['sample_id']}"
                    f"_prediction.npz"
                ),
            )
        )

        np.savez_compressed(

            sample_result_path,

            sample_id=
                sample[
                    "sample_id"
                ],

            case_name=
                sample[
                    "case_name"
                ],

            T_pred=
                T_pred,

            T_gt=
                T_gt,

            source=
                sample[
                    "source"
                ],

            target=
                sample[
                    "target"
                ],

            ref_corr_points=
                output_dict.get(
                    "ref_corr_points",
                    np.empty(
                        (0, 3)
                    ),
                ),

            src_corr_points=
                output_dict.get(
                    "src_corr_points",
                    np.empty(
                        (0, 3)
                    ),
                ),

            corr_scores=
                output_dict.get(
                    "corr_scores",
                    np.empty(
                        (0,)
                    ),
                ),

            RRE_deg=
                rre,

            RTE_normalized=
                rte,

            RTE_mm=
                rte_mm,

            RMSE_normalized=
                rmse,

            RMSE_mm=
                rmse_mm,
        )


        # ====================================================
        # PRINT SAMPLE RESULT
        # ====================================================

        print(
            f"\nSample        : "
            f"{sample['sample_id']}"
        )

        print(
            f"Case          : "
            f"{sample['case_name']}"
        )

        print(
            f"RRE           : "
            f"{rre:.6f} deg"
        )

        print(
            f"RTE norm      : "
            f"{rte:.6f}"
        )

        if np.isnan(
            rte_mm
        ):

            print(
                "RTE mm        : N/A"
            )

        else:

            print(
                f"RTE mm        : "
                f"{rte_mm:.6f} mm"
            )

        print(
            f"RMSE norm     : "
            f"{rmse:.6f}"
        )

        if np.isnan(
            rmse_mm
        ):

            print(
                "RMSE mm       : N/A"
            )

        else:

            print(
                f"RMSE mm       : "
                f"{rmse_mm:.6f} mm"
            )

        print(
            f"# correspond. : "
            f"{num_corr}"
        )


    # ========================================================
    # NEW:
    # SEPARATE ORIGINAL CASE FROM 9 PERTURBED CASES
    # ========================================================

    original_rows = [
        row
        for row in rows
        if row[
            "case_name"
        ] == "original_aligned"
    ]

    perturbed_rows = [
        row
        for row in rows
        if row[
            "case_name"
        ] != "original_aligned"
    ]


    print(
        "\n========================================"
    )

    print(
        "SAMPLE GROUPS"
    )

    print(
        "========================================"
    )

    print(
        "Total samples      :",
        len(rows)
    )

    print(
        "Original cases     :",
        len(original_rows)
    )

    print(
        "Perturbed cases    :",
        len(perturbed_rows)
    )


    # ========================================================
    # GLOBAL SUMMARY
    #
    # IMPORTANT:
    # calculate mean/std/median ONLY over the 9 perturbed
    # samples for comparison with your previous experiments.
    # ========================================================

    numeric_metrics = [

        "RRE_deg",

        "RTE_normalized",

        "RTE_mm",

        "RMSE_normalized",

        "RMSE_mm",
    ]


    summary = {

        "dataset":
            INFERENCE_DATASET_NAME,

        "experiment":
            EXPERIMENT_NAME,

        "checkpoint":
            checkpoint_path,

        "num_samples":
            len(rows),

        "num_original_samples":
            len(original_rows),

        "num_perturbed_samples":
            len(perturbed_rows),
    }


    # --------------------------------------------------------
    # Original aligned case stored separately
    # --------------------------------------------------------

    if len(original_rows) == 1:

        original = original_rows[0]

        summary[
            "original_aligned"
        ] = {

            "RRE_deg":
                original[
                    "RRE_deg"
                ],

            "RTE_normalized":
                original[
                    "RTE_normalized"
                ],

            "RTE_mm":
                original[
                    "RTE_mm"
                ],

            "RMSE_normalized":
                original[
                    "RMSE_normalized"
                ],

            "RMSE_mm":
                original[
                    "RMSE_mm"
                ],

            "num_correspondences":
                original[
                    "num_correspondences"
                ],
        }

    else:

        summary[
            "original_aligned"
        ] = None


    # --------------------------------------------------------
    # Statistics ONLY for perturbed cases
    # --------------------------------------------------------

    for metric in numeric_metrics:

        values = np.asarray(
            [
                row[
                    metric
                ]
                for row in perturbed_rows
            ],
            dtype=np.float64,
        )

        valid_values = (
            values[
                ~np.isnan(
                    values
                )
            ]
        )

        if len(valid_values) > 0:

            summary[
                f"mean_{metric}"
            ] = float(
                np.mean(
                    valid_values
                )
            )

            summary[
                f"std_{metric}"
            ] = float(
                np.std(
                    valid_values
                )
            )

            summary[
                f"median_{metric}"
            ] = float(
                np.median(
                    valid_values
                )
            )

        else:

            summary[
                f"mean_{metric}"
            ] = None

            summary[
                f"std_{metric}"
            ] = None

            summary[
                f"median_{metric}"
            ] = None


    # ========================================================
    # SAVE CSV
    # ========================================================

    csv_path = os.path.join(
        RESULT_DIR,
        "inference_results.csv",
    )

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:

        writer = (
            csv.DictWriter(
                csv_file,
                fieldnames=list(
                    rows[0].keys()
                ),
            )
        )

        writer.writeheader()

        writer.writerows(
            rows
        )


    # ========================================================
    # SAVE JSON
    # ========================================================

    json_path = os.path.join(
        RESULT_DIR,
        "inference_summary.json",
    )

    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as json_file:

        json.dump(
            {
                "summary":
                    summary,

                "samples":
                    rows,
            },
            json_file,
            indent=4,
        )


    # ========================================================
    # PRINT ORIGINAL ALIGNED CASE SEPARATELY
    # ========================================================

    if len(original_rows) == 1:

        original = original_rows[0]

        print(
            "\n"
            + "=" * 80
        )

        print(
            "ORIGINAL ALIGNED CASE "
            "(NO ARTIFICIAL PERTURBATION)"
        )

        print(
            "=" * 80
        )

        print(
            f"RRE           : "
            f"{original['RRE_deg']:.6f} deg"
        )

        print(
            f"RTE norm      : "
            f"{original['RTE_normalized']:.6f}"
        )

        print(
            f"RTE mm        : "
            f"{original['RTE_mm']:.6f} mm"
        )

        print(
            f"RMSE norm     : "
            f"{original['RMSE_normalized']:.6f}"
        )

        print(
            f"RMSE mm       : "
            f"{original['RMSE_mm']:.6f} mm"
        )

        print(
            f"# correspond. : "
            f"{original['num_correspondences']}"
        )


    # ========================================================
    # PRINT FINAL SUMMARY FOR 9 PERTURBED CASES
    # ========================================================

    print(
        "\n\n"
        + "=" * 80
    )

    print(
        "FINAL SUMMARY - "
        "9 PERTURBED CASES ONLY"
    )

    print(
        "=" * 80
    )

    print(
        f"Dataset    : "
        f"{INFERENCE_DATASET_NAME}"
    )

    print(
        f"Experiment : "
        f"{EXPERIMENT_NAME}"
    )

    print(
        f"Checkpoint : "
        f"{checkpoint_path}"
    )

    print(
        f"Total      : "
        f"{len(rows)}"
    )

    print(
        f"Perturbed  : "
        f"{len(perturbed_rows)}"
    )

    print(
        f"\nMean RRE      : "
        f"{summary['mean_RRE_deg']:.6f} deg"
    )

    print(
        f"Mean RTE norm : "
        f"{summary['mean_RTE_normalized']:.6f}"
    )

    if (
        summary[
            "mean_RTE_mm"
        ]
        is not None
    ):

        print(
            f"Mean RTE mm   : "
            f"{summary['mean_RTE_mm']:.6f} mm"
        )

    print(
        f"Mean RMSE norm: "
        f"{summary['mean_RMSE_normalized']:.6f}"
    )

    if (
        summary[
            "mean_RMSE_mm"
        ]
        is not None
    ):

        print(
            f"Mean RMSE mm  : "
            f"{summary['mean_RMSE_mm']:.6f} mm"
        )


    print(
        "\nSaved:"
    )

    print(
        csv_path
    )

    print(
        json_path
    )


    # ========================================================
    # BEST REGISTRATION
    # ========================================================

    if (
        best_visualization
        is not None
    ):

        best = (
            best_visualization
        )

        print(
            "\n"
            + "=" * 80
        )

        print(
            "BEST REGISTRATION "
            "(LOWEST RMSE)"
        )

        print(
            "=" * 80
        )

        print(
            f"Sample : "
            f"{best['sample_id']}"
        )

        print(
            f"Case   : "
            f"{best['case_name']}"
        )

        print(
            f"RRE    : "
            f"{best['rre']:.6f} deg"
        )

        print(
            f"RTE    : "
            f"{best['rte']:.6f}"
        )

        if not np.isnan(
            best[
                "rte_mm"
            ]
        ):

            print(
                f"RTE mm : "
                f"{best['rte_mm']:.6f} mm"
            )

        print(
            f"RMSE   : "
            f"{best['rmse']:.6f}"
        )

        if not np.isnan(
            best[
                "rmse_mm"
            ]
        ):

            print(
                f"RMSE mm: "
                f"{best['rmse_mm']:.6f} mm"
            )

        # ----------------------------------------------------
        # Save best separately
        # ----------------------------------------------------

        best_path = os.path.join(
            RESULT_DIR,
            "best_registration.npz",
        )

        np.savez_compressed(

            best_path,

            sample_id=
                best[
                    "sample_id"
                ],

            case_name=
                best[
                    "case_name"
                ],

            source=
                best[
                    "source"
                ],

            target=
                best[
                    "target"
                ],

            T_pred=
                best[
                    "T_pred"
                ],

            T_gt=
                best[
                    "T_gt"
                ],

            RRE_deg=
                best[
                    "rre"
                ],

            RTE_normalized=
                best[
                    "rte"
                ],

            RTE_mm=
                best[
                    "rte_mm"
                ],

            RMSE_normalized=
                best[
                    "rmse"
                ],

            RMSE_mm=
                best[
                    "rmse_mm"
                ],
        )

        print(
            "\nBest result saved:"
        )

        print(
            best_path
        )


        # ====================================================
        # VISUALIZE BEST
        # ====================================================

        if VISUALIZE_BEST:

            print(
                "\nOpening best "
                "registration..."
            )

            visualize_registration(

                source=
                    best[
                        "source"
                    ],

                target=
                    best[
                        "target"
                    ],

                T_pred=
                    best[
                        "T_pred"
                    ],

                T_gt=
                    best[
                        "T_gt"
                    ],

                title=(
                    "Best GeoTransformer "
                    "Registration - "
                    f"{best['case_name']}"
                ),

                show_ground_truth=
                    SHOW_GROUND_TRUTH,
            )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()