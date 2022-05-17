from evaluations_mgeval.utils.mgeval_rytm_utils import *
from copy import deepcopy

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
        gt = flatten_subset_genres(get_gt_feats_from_evaluator(sets_evals[set_name]), remove_nans=True)
        pd = flatten_subset_genres(get_pd_feats_from_evaluator(eval), remove_nans=True)
        feature_sets.update({f"{set_name} (GT)": gt})
        feature_sets.update({set_name: pd})

    # ----- grab selected indices (samples)
    for set_name, set_dict in feature_sets.items():
        for key, array in set_dict.items():
            feature_sets[set_name][key] = array

    # --- remove unnecessary features
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
    # ---- Analysis 0:  Accuracy Vs. Precision
    # ----              Also, utiming and velocity analysis
    # from sklearn.metrics import precision_score, accuracy_score
    # ================================================================
    absolute_sets_evals = dict()
    for set_key in sets_evals.keys():
        gmd_eval = deepcopy(sets_evals[set_key])
        gmd_eval._prediction_hvos_array = gmd_eval._gt_hvos_array
        absolute_sets_evals.update({f"{set_key} (GT)": gmd_eval})
        absolute_sets_evals.update({set_key: sets_evals[set_key]})

    # n_hits_in_gt = sum(sets_evals[list(sets_evals.keys())[0]].get_ground_truth_hvos_array()[:,:,:9].flatten())
    # n_silence_in_gt = sets_evals[list(sets_evals.keys())[0]].get_ground_truth_hvos_array()[:,:,:9].size - n_hits_in_gt
    # hit_weight = n_silence_in_gt / n_hits_in_gt
    stats_sets = get_positive_negative_hit_stats(sets_evals, hit_weight=1)

    fig_path = f"evaluations_mgeval/figures/"
    for ANALYSIS_SET in ANALYSIS_SETS:
        fig_path += f"{ANALYSIS_SET}_"

    os.makedirs(fig_path, exist_ok=True)

    group_hit_labels = ['PPV']  # ['GT Hits', 'Predicted Hits']    #, 'True Hits (Matching GT)', 'False Hits (Different from GT)']
    # gt_hits = stats_sets[list(stats_sets.keys())[0]]['Actual Hits']
    # hit_analysis= {'GT': {'Total Hits': gt_hits, 'True Hits (Matching GT)': gt_hits, 'False Hits (Different from GT)': gt_hits}}
    hit_analysis = dict()
    only_leave_these_in_stats = {''}
    for key in list(stats_sets.keys()):
        hit_analysis.update({key: {}})
        for key_ in list(stats_sets[key].keys()):
            if key_ in group_hit_labels:
                hit_analysis[key].update({key_: stats_sets[key][key_]})
                stats_sets[key].pop(key_)

    generate_these = False
    if generate_these is not False:
        boxplot_absolute_measures(hit_analysis, fs=30, legend_fs=20, legend_ncols=8, fig_path=fig_path, show=False, ncols=3,
                                  figsize=(30, 5), color_map="tab20c", filename="Stats_hits", share_legend=True, sharey=True,
                                  show_legend=False)

    generate_these = False
    if generate_these is not False:
        boxplot_absolute_measures(stats_sets, fs=30, legend_fs=10, legend_ncols=4, fig_path=fig_path, show=True, ncols=3,
                                  figsize=(30, 25), color_map="tab20c", filename="Hits_performance",
                                  sharey=False, share_legend=False, shift_colors_by=0,
                                  show_legend=False)

    # TPR -> How many of ground truth hits were correctly predicted
    # FPR -> How many of ground truth silences were predicted as hits
    # to

    generate_these = False
    if generate_these is not False:
        vel_stats_sets = get_positive_negative_vel_stats(absolute_sets_evals)
        boxplot_absolute_measures(vel_stats_sets, fs=30, legend_fs=30, legend_ncols=8, fig_path=fig_path, show=True, ncols=3,
                                  figsize=(30, 10), color_map="tab20c", filename="Stats_vels", share_legend=True, show_legend=False,
                                  shift_colors_by=0)
    generate_these = False
    if generate_these is not False:
        ut_stats_sets = get_positive_negative_utiming_stats(absolute_sets_evals)
        boxplot_absolute_measures(ut_stats_sets, fs=30, legend_fs=10, legend_ncols=4, fig_path=fig_path, show=True, ncols=3,
                                  figsize=(30, 10), color_map="tab20c", filename="Stats_ut",
                                  sharey=False, share_legend=True, shift_colors_by=0,
                                  show_legend=False)
