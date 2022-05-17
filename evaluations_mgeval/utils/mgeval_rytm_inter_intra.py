from post_processing_scripts.mgeval_rytm_utils import *
from copy import deepcopy


if __name__ == '__main__':

    gmd_eval = pickle.load(open(
            f"post_processing_scripts/evaluators_monotonic_groove_transformer_v1/"
            f"validation_set_evaluator_run_misunderstood-bush-246_Epoch_26.Eval","rb"))

    down_size = 400
    final_indices = sample_uniformly(gmd_eval, num_samples=down_size) if down_size < 1024 else list(range(1024))



    # Compile data (flatten styles)
    '''"robust":
                pickle.load(open(f"post_processing_scripts/evaluators_monotonic_groove_transformer_v1/"
                                 f"validation_set_evaluator_run_robust_sweep_29.Eval", "rb")),
            "colorful":
                pickle.load(open(f"post_processing_scripts/evaluators_monotonic_groove_transformer_v1/"
                                 f"validation_set_evaluator_run_colorful_sweep_41.Eval", "rb"))'''

    sets_evals = {
        "groovae":
            pickle.load(open(f"post_processing_scripts/evaluators_monotonic_groove_transformer_v1/"
                             f"validation_set_evaluator_run_groovae.Eval", "rb")),
        "rosy":
            pickle.load(open(f"post_processing_scripts/evaluators_monotonic_groove_transformer_v1/"
                             f"validation_set_evaluator_run_rosy-durian-248_Epoch_26.Eval", "rb")),
        "hopeful":
            pickle.load(open(f"post_processing_scripts/evaluators_monotonic_groove_transformer_v1/"
                             f"validation_set_evaluator_run_hopeful-gorge-252_Epoch_90.Eval", "rb")),
        "solar":
            pickle.load(open(f"post_processing_scripts/evaluators_monotonic_groove_transformer_v1/"
                             f"validation_set_evaluator_run_solar-shadow-247_Epoch_41.Eval", "rb")),
        "misunderstood":
            pickle.load(open(f"post_processing_scripts/evaluators_monotonic_groove_transformer_v1/"
                             f"validation_set_evaluator_run_misunderstood-bush-246_Epoch_26.Eval", "rb")),

    }

    # Compile data (flatten styles)
    '''    new_names = {
            "rosy": "Model 1",
            "hopeful": "Model 2",
            "solar": "Model 3",
            "misunderstood": "Model 4",
            "robust": "Model 5",
            "colorful": "Model 6",
            "groovae": "GrooVAE"
    
        }'''

    new_names = {
        "rosy": "Model 1",
        "hopeful": "Model 2",
        "solar": "Model 3",
        "misunderstood": "Model 4",
        "groovae": "GrooVAE"
    }

    sets_evals = dict((new_names[key], value) for (key, value) in sets_evals.items())

    # compile and flatten features
    feature_sets = {"GMD": flatten_subset_genres(get_gt_feats_from_evaluator(list(sets_evals.values())[0]))}
    feature_sets.update({
        set_name:flatten_subset_genres(get_pd_feats_from_evaluator(eval)) for (set_name, eval) in sets_evals.items()
    })

    # ----- grab selected indices (samples)
    for set_name, set_dict in feature_sets.items():
        for key, array in set_dict.items():
            feature_sets[set_name][key] = array[final_indices]

    # --- remove unnecessary features
    '''allowed_analysis = ["Statistical::NoI", "Statistical::Total Step Density", "Statistical::Avg Voice Density",
                        "Statistical::Lowness", "Statistical::Midness", "Statistical::Hiness",
                        "Statistical::Vel Similarity Score", "Statistical::Weak to Strong Ratio",
                        "Syncopation::Lowsync", "Syncopation::Midsync", "Syncopation::Hisync",
                        "Syncopation::Lowsyness", "Syncopation::Midsyness", "Syncopation::Hisyness", "Syncopation::Complexity"]
    '''

    allowed_analysis = list(feature_sets['GMD'].keys())

    for set_name in feature_sets.keys():
        for key in list(feature_sets[set_name].keys()):
            if key not in allowed_analysis:
                feature_sets[set_name].pop(key)



    # ================================================================
    # ---- Analysis 2: Comparing the 4 models we made with each other
    # 1.a. Calculate intraset distances of gmd and 4 models
    #   b. Calculate mean and std of each
    # 2.a. Calculate interset distances of each 4 models from gmd
    #   b. for each set, calculate KLD and OLD against gmd
    # 3. Create a table similar to Table 4 in Yang et. al.
    # ================================================================
    # 1.a.
    eval_labels = []


    eval_labels.append(['GMD', new_names['hopeful'], new_names['rosy']])
    eval_labels.append(['GMD', new_names['solar'], new_names['rosy']])
    eval_labels.append(['GMD', new_names['misunderstood'], new_names['rosy']])
    #eval_labels.append(['GMD', new_names['robust'], new_names['rosy']])
    #eval_labels.append(['GMD', new_names['colorful'], new_names['rosy']])
    eval_labels.append(['GMD', new_names['groovae'], new_names['rosy']])

    figs_list= []
    axes_list = []

    for set_labels in eval_labels:

        gt = feature_sets[set_labels[0]]
        set1 = feature_sets[set_labels[1]]
        set2 = feature_sets[set_labels[2]]

        # Export Analysis to Table
        csv_path = f"post_processing_scripts/evaluators_monotonic_groove_transformer_v1/mgeval_results/" \
                   f"{set_labels[0]}_{set_labels[1]}_{set_labels[2]}/table4_compiled.csv"
        df, raw_data = compare_two_sets_against_ground_truth(gt, set1, set2, set_labels=set_labels, csv_path=csv_path)

        # Generate inter_intra_pdfs feature plots
        fig_path = f"post_processing_scripts/evaluators_monotonic_groove_transformer_v1/mgeval_results/" \
                   f"{set_labels[0]}_{set_labels[1]}_{set_labels[2]}"

        # plot inter/intra pdfs
        generate_these = False
        if generate_these is True:
            fig_, axes_ = plot_inter_intra_pdfs(
                raw_data, fig_path, set_labels, show=True, figsize=(30, 30), fs=15, legend_fs=9, ncols=6)

        # Generate inter-intra distance plots
        generate_these = True
        if generate_these is True:
            fig_1, axes_1 = plot_intersets(
                df, fig_path, set_labels, show=False, ncols=4, figsize=(8, 5), max_features_in_plot=28,
                legend_fs=20 , fs=24, add_legend=False, add_text=True, min_line_len_for_text=.2,
                set1_name=set_labels[1], set2_name=set_labels[2], legend_ncols=7, force_xlim=(0, 4), force_ylim=(0.65, 1))
            figs_list.append(fig_1)
            axes_list.append(axes_1)

            fig_1.show()

    change_attributes = True
    if change_attributes is True:
        new_x_lim = (0, 4)
        new_y_lim = (0.65, 1)
        new_text_fontsize = 24
        label_fs = 24
        new_fig_dimensions = (8, 5)
        axis_tickfontsize = 12
        for i, set_labels in enumerate(eval_labels[2:3]):
            if fig_path is not None:
                ax_ = axes_list[i]
                ax_.set_xlim(new_x_lim)
                ax_.set_ylim(new_y_lim)
                for text in ax_.texts:
                    text.set_fontsize(new_text_fontsize)
                ax_.set_xlabel("KL", fontsize=label_fs)
                ax_.set_ylabel("OA", fontsize=label_fs)
                os.makedirs(fig_path, exist_ok=True)
                now = datetime.now()
                time_txt = f"{now.day}_{now.month}_{now.year}-{now.hour}-{now.min}-{now.second}"
                filename = os.path.join(fig_path,
                                        f"inter({set_labels[1]},{set_labels[0]})_inter({set_labels[2]},{set_labels[0]})_vs_Intra({set_labels[0]})_at_{time_txt}")
                ax_.get_figure().set_size_inches(new_fig_dimensions)

                ax_.set_xticklabels(ax_.get_xticklabels(), fontsize = axis_tickfontsize)
                ax_.set_yticklabels(ax_.get_yticklabels(), fontsize = axis_tickfontsize)

                ax_.get_figure().savefig(filename)
                ax_.get_figure().show()

        fig_.show()
