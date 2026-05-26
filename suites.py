"""Experiment suite definitions for the paper.

Each suite is a list of YAML config files that will be executed across
multiple seeds by run_suite.py.
"""

EXPERIMENT_SUITES = {
    # Part 1 — GNN architectures
    "part1_arch": [
        "config/Part1_Exp1_1_GCN7.yaml",
        "config/Part1_Exp1_1_GCN60.yaml",
        "config/Part1_Exp1_1_GIN.yaml",
        "config/Part1_Exp1_1_GINE.yaml",
    ],

    # Part 1 — Graph constructors
    "part1_constructor": [
        "config/Part1_Exp1_2_VG.yaml",
        "config/Part1_Exp1_2_HVG.yaml",
        "config/Part1_Exp1_2_kVG10.yaml",
    ],

    # Part 1 — Node and edge feature ablations
    "part1_features": [
        "config/Part1_Exp1_3_NodeFeat_RR.yaml",
        "config/Part1_Exp1_3_NodeFeat_Stats.yaml",
        "config/Part1_Exp1_3_EdgeFeat_DeltaT.yaml",
        "config/Part1_Exp1_3_EdgeFeat_Slope.yaml",
    ],

    # Part 2 — CNN and sequential baselines
    "part2_baselines": [
        "config/Part2_Exp2_1_SimpleCNN.yaml",
        "config/Part2_Exp2_1_HannunResNet.yaml",
        "config/Part2_Exp2_1_DANet.yaml",
        "config/Part2_Exp2_1_MousaviSeq2Seq.yaml",
        "config/Part2_GINESeq2Seq.yaml",
        "config/Part2_GINESeq2Seq_Focal.yaml",
    ],

    # Part 3 — Readout and positional encoding ablations
    "part3_ablations": [
        "config/Part3_Exp3_1_Readout_Mean.yaml",
        "config/Part3_Exp3_1_Readout_Set2Set.yaml",
        "config/Part3_Exp3_1_Readout_SortPool.yaml",
        "config/Part3_Exp3_1_PE_Sin.yaml",
        "config/Part3_Exp3_1_PE_LapPE.yaml",
        "config/Part3_Exp3_1_PE_RWSE.yaml",
    ],
}

# Convenience aliases
EXPERIMENT_SUITES["part1"] = (
    EXPERIMENT_SUITES["part1_arch"]
    + EXPERIMENT_SUITES["part1_constructor"]
    + EXPERIMENT_SUITES["part1_features"]
)
EXPERIMENT_SUITES["part2"] = EXPERIMENT_SUITES["part2_baselines"]
EXPERIMENT_SUITES["part3"] = EXPERIMENT_SUITES["part3_ablations"]

# Full paper reproduction suite
EXPERIMENT_SUITES["paper"] = (
    EXPERIMENT_SUITES["part1"]
    + EXPERIMENT_SUITES["part2"]
    + EXPERIMENT_SUITES["part3"]
)


def get_suite_names():
    return list(EXPERIMENT_SUITES.keys())


def get_suite_configs(suite_name):
    return EXPERIMENT_SUITES.get(suite_name, [])


def get_suite_info(suite_name):
    configs = get_suite_configs(suite_name)
    return {
        "name": suite_name,
        "config_count": len(configs),
        "configs": configs,
        "total_experiments": len(configs) * 5,
    }
