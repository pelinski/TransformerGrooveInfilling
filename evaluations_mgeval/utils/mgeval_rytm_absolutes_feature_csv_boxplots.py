from evaluations_mgeval.utils.mgeval_rytm_utils import *
from copy import deepcopy
from evaluations_mgeval.utils.inter_intra_utils import get_feature_dict_from
if __name__ == '__main__':

    # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    ANALYSIS_SETS = ["InfillingClosedHH_Symbolic", "InfillingClosedHH", "InfillingKicksAndSnares",
                     "InfillingRandomLow", "InfillingRandom"]
    # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

    tmp_eval = pickle.load(open("datasets/final_evaluators/InfillingEvaluator_0.3.2/InfillingClosedHH_validation_0.1.2_evaluator.pickle", "rb"))

    size = tmp_eval._prediction_hvos_array.shape[0]
    # down_size = size
    # final_indices = sample_uniformly(tmp_eval, num_samples=down_size) if down_size < size else list(range(size))
    # final_indices = list(range(size))

    # Compile data (flatten styles)
    new_names = {
        "InfillingClosedHH_Symbolic": "ClosedHH (Symbolic)",
        "InfillingClosedHH": "ClosedHH",
        "InfillingKicksAndSnares": "Kicks & Snares",
        "InfillingRandomLow": "Random (Low)",
        "InfillingRandom": "Random (High)"
    }

    evaluator_paths = {
        "InfillingClosedHH_Symbolic":
            "datasets/final_evaluators/InfillingEvaluator_0.3.2/InfillingClosedHH_Symbolic_validation_0.1.1_evaluator.pickle",
        "InfillingClosedHH": "datasets/final_evaluators/InfillingEvaluator_0.3.2/InfillingClosedHH_validation_0.1.2_evaluator.pickle",
        "InfillingKicksAndSnares": "datasets/final_evaluators/InfillingEvaluator_0.3.2/InfillingKicksAndSnares_validation_0.1.2_evaluator.pickle",
        "InfillingRandomLow": "datasets/final_evaluators/InfillingEvaluator_0.3.2/InfillingRandomLow_validation_0.0.0_evaluator.pickle",
        "InfillingRandom": "datasets/final_evaluators/InfillingEvaluator_0.3.2/InfillingRandom_validation_0.0.0_evaluator.pickle",
    }


    evaluators = dict((ANALYSIS_SET, pickle.load(open(evaluator_paths[ANALYSIS_SET], "rb"))) for ANALYSIS_SET in ANALYSIS_SETS)
    sets_evals = dict((new_names[key], value) for (key, value) in evaluators.items())

    feature_sets = dict()
    # compile and flatten features
    for (set_name, eval) in sets_evals.items():
        gt_hvo_seqs = eval._gt_hvo_sequences
        pd_hvo_seqs = flatten(eval._prediction_subsets)
        gt_features = get_feature_dict_from(gt_hvo_seqs, set_tag="ignore", use_voices=None)
        pd_features = get_feature_dict_from(pd_hvo_seqs, set_tag="ignore", use_voices=None)
        feature_sets.update({f"{set_name} (GT)": gt_features})
        feature_sets.update({set_name: pd_features})



    # ----- grab selected indices (samples)
    for set_name, set_dict in feature_sets.items():
        for key, array in set_dict.items():
            feature_sets[set_name][key] = array

    # --- remove unnecessary features
    '''just_show = ["Statistical::NoI", "Statistical::Total Step Density", "Statistical::Avg Voice Density",
                        "Statistical::Lowness", "Statistical::Midness", "Statistical::Hiness",
                        "Statistical::Vel Similarity Score", "Statistical::Weak to Strong Ratio",
                        "Syncopation::Lowsync", "Syncopation::Midsync", "Syncopation::Hisync",
                        "Syncopation::Lowsyness", "Syncopation::Midsyness", "Syncopation::Hisyness", "Syncopation::Complexity"]'''
    #just_show = None # Show all
    # just_show = ['Statistical::Poly Velocity Mean', 'Statistical::Poly Velocity std']
    just_show = [
        'Statistical::Total Step Density', 'Statistical::Avg Voice Density',
        'Statistical::Vel Similarity Score', 'Statistical::Weak to Strong Ratio',
        'Syncopation::Lowsync', 'Syncopation::Midsync', 'Syncopation::Hisync',
        'Micro-Timing::Swingness'
    ]
    for set_name in feature_sets.keys():
        allowed_analysis = feature_sets[set_name].keys() if just_show is None else just_show
        for key in list(feature_sets[set_name].keys()):
            if key not in allowed_analysis:
                feature_sets[set_name].pop(key)


    # ================================================================
    # ---- Analysis 1: Absolute Measures According to
    # Yang, Li-Chia, and Alexander Lerch. "On the evaluation of generative models in music."
    #           Neural Computing and Applications 32.9 (2020): 4773-4784.
    # ================================================================

    fig_path = f"evaluations_mgeval/figures/"


    # Compile Absolute Measures
    generate_these = True
    if generate_these is not False:
        csv_path = os.path.join(fig_path, "csv_absolutes.csv")
        print(csv_path)
        pd_final = get_absolute_measures_for_multiple_sets(feature_sets, csv_file=csv_path)

        fig_path_box_plots = os.path.join(fig_path, "boxplots")
        os.makedirs(fig_path_box_plots, exist_ok=True)
        boxplot_absolute_measures(feature_sets, fs=20, legend_fs=14, legend_ncols=5, fig_path=fig_path_box_plots, show=True, ncols=4,
                                  figsize=(20, 5), color_map="tab20c", sharey=False, share_legend=True, show_legend=True) # , force_ylim = (-0.5, 0.5)
