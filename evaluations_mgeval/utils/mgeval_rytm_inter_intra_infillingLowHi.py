from evaluations_mgeval.utils.inter_intra_utils import *


if __name__ == '__main__':
    # SET PARAMS HERE
    num_samples_per_master_id = 3
    soundfont = "hvo_sequence/soundfonts/Standard_Drum_Kit.sf2"
    mix_pred_with_input = True
    soundfontname = soundfont.split("/")[-1].split(".sf")[0]

    with open("configs/paths.yaml", "r") as f:
        paths = yaml.safe_load(f)

    with open("datasets/preprocessed_evaluators/preprocessed_evaluators_parameters.json") as f:
        params = json.load(f)

    # Compile data (flatten styles)
    new_names = {
        "GMD": "Ground Truth",
        "InfillingRandomLow": "Random (Low)",
        "InfillingRandom": "Random (High)"
    }

    eval_labels = []

    eval_labels.append(['Ground Truth', new_names['InfillingRandomLow'], new_names['InfillingRandom']])

    # SUPER IMPORTANT!!! sampled gt samples will be added as a prediction to this set. Hence, use _pd_ rather than _gt_ for analysis
    # evaluators_pd = {key: pickle.load(open(path, "rb")) for key, path in evaluators_paths.items()}
    evaluator_gt = pickle.load(open(
        "datasets/final_evaluators/InfillingEvaluator_0.3.2/"
        "InfillingClosedHH_Symbolic_validation_0.1.1_evaluator.pickle", "rb"))

    gt_hvo_seqs_downsampled, gt_hvos_array_downsampled, sampled_loop_ids = downsample_set(evaluator_gt, num_samples_per_master_id=num_samples_per_master_id)

    gt_features = get_feature_dict_from (gt_hvo_seqs_downsampled, set_tag="Ground Truth")

    hvo_seqs_list_removed_rl = get_version_with_randomly_removed_voices(gt_hvo_seqs_downsampled, set_tag="Ground Truth", thres_range=(0.1, 0.3))
    hvo_seqs_list_removed_rh = get_version_with_randomly_removed_voices(gt_hvo_seqs_downsampled, set_tag="Ground Truth", thres_range=(0.4, 0.7))

    hvos_gt = np.array([hvo_sample.hvo for hvo_sample in gt_hvo_seqs_downsampled])
    empty_hvo_seqs_template =  [hvo_sample.copy_zero() for hvo_sample in gt_hvo_seqs_downsampled]

    corresponding_hvo_inputs = {
        "InfillingRandomLow": np.array([hvo_sample.hvo for hvo_sample in hvo_seqs_list_removed_rl]),
        "InfillingRandom": np.array([hvo_sample.hvo for hvo_sample in hvo_seqs_list_removed_rh])
    }

    msos_rl_input = convert_hvo_seqs_to_msos(hvo_seqs_list_removed_rl, sf_path=soundfont)
    msos_rh_input = convert_hvo_seqs_to_msos(hvo_seqs_list_removed_rh, sf_path=soundfont)

    model_inputs = {
        "InfillingRandomLow": msos_rl_input,
        "InfillingRandom": msos_rh_input
    }

    model_outputs = dict()

    for model_name, input in model_inputs.items():
        model_path = "trained_models"
        model = load_model(model_path, model_name)
        h, v, o = model.predict(torch.tensor(input, dtype=torch.float32))
        predicted_hvos_array = torch.cat((h, v, o), axis=2).detach().numpy()
        input_with_predicted_output = mix_input_with_prediction(corresponding_hvo_inputs[model_name], predicted_hvos_array) if mix_pred_with_input else predicted_hvos_array

        # add to hvo_sequence objects
        output_hvo_sequences = deepcopy(empty_hvo_seqs_template)
        print(output_hvo_sequences[-1])
        for ix, predicted_hvo_array in enumerate(input_with_predicted_output):
            output_hvo_sequences[ix].hvo = predicted_hvo_array
        model_outputs.update({model_name: output_hvo_sequences})



    # Compile features into a single dictionary
    feature_dict = deepcopy(get_feature_dict_from(gt_hvo_seqs_downsampled, set_tag="Ground Truth"))
    feature_dict.pop('metadata')
    feature_dicts = {"GMD": feature_dict}
    print(feature_dict['Micro-Timing::Accuracy'])
    for model_name, model_output_hvo_seqs in model_outputs.items():
        feature_dict = deepcopy(get_feature_dict_from(model_output_hvo_seqs, set_tag="Ground Truth"))
        feature_dict.pop('metadata')
        feature_dicts.update( {model_name: feature_dict})

    feature_sets = dict((new_names[key], value) for (key, value) in feature_dicts.items())

    # --- remove unnecessary features
    '''allowed_analysis = ["Statistical::NoI", "Statistical::Total Step Density", "Statistical::Avg Voice Density",
                        "Statistical::Lowness", "Statistical::Midness", "Statistical::Hiness",
                        "Statistical::Vel Similarity Score", "Statistical::Weak to Strong Ratio",
                        "Syncopation::Lowsync", "Syncopation::Midsync", "Syncopation::Hisync",
                        "Syncopation::Lowsyness", "Syncopation::Midsyness", "Syncopation::Hisyness", "Syncopation::Complexity"]
'''
    allowed_analysis = [
        'Statistical::Total Step Density', 'Statistical::Avg Voice Density',
        'Statistical::Vel Similarity Score', 'Statistical::Weak to Strong Ratio',
        'Syncopation::Lowsync', 'Syncopation::Midsync', 'Syncopation::Hisync',
        'Micro-Timing::Swingness'
    ]

    # allowed_analysis = feature_sets['Ground Truth'].keys()
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
    figs_list= []
    axes_list = []

    for set_labels in eval_labels:

        gt = feature_sets[set_labels[0]]
        set1 = feature_sets[set_labels[1]]
        set2 = feature_sets[set_labels[2]]

        fig_path = f"evaluations_mgeval/figures/inter_intra/{set_labels[0]}_{set_labels[1]}_{set_labels[2]}_sf_{soundfontname}"

        # Export Analysis to Table
        csv_path = os.path.join(fig_path, "table4_compiled.csv")
        df, raw_data = compare_two_sets_against_ground_truth(gt, set1, set2, set_labels=set_labels, csv_path=csv_path)

        # Generate inter_intra_pdfs feature plots
        os.makedirs(fig_path, exist_ok=True)

        # plot inter/intra pdfs
        generate_these = False
        if generate_these is True:
            fig_, axes_ = plot_inter_intra_pdfs(
                raw_data, fig_path, set_labels, show=True, figsize=(30, 12), fs=18, legend_fs=8, ncols=4)

        # Generate inter-intra distance plots
        # Generate inter-intra distance plots
        # Generate inter-intra distance plots
        generate_these = True
        if generate_these is True:
            fig_1, axes_1 = plot_intersets(
                df, fig_path, set_labels, show=False, ncols=4, figsize=(3.5, 4.5), max_features_in_plot=14,
                legend_fs=9, fs=11, add_legend=True, add_text=False, min_line_len_for_text=.1,
                set1_name=set_labels[1], set2_name=set_labels[2], legend_ncols=1, force_xlim=(0, 0.2),
                force_ylim=(0.55, 1))
            figs_list.append(fig_1)
            axes_list.append(axes_1)

            fig_1.show()

    change_attributes = False
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
                                        f"inter({set_labels[1]},{set_labels[0]})_inter({set_labels[2]},{set_labels[0]})_vs_Intra({set_labels[0]})_at_{time_txt}_mix_pred_with_input_{mix_pred_with_input}")
                ax_.get_figure().set_size_inches(new_fig_dimensions)

                ax_.set_xticklabels(ax_.get_xticklabels(), fontsize = axis_tickfontsize)
                ax_.set_yticklabels(ax_.get_yticklabels(), fontsize = axis_tickfontsize)

                ax_.get_figure().savefig(filename)
                ax_.get_figure().show()

