from evaluations_mgeval.utils.inter_intra_utils import *

soundfontnames = [
    'hvo_sequence/soundfonts/GoldDrums.sf2', 'hvo_sequence/soundfonts/80sAcDanceDrums.sf2',
     'hvo_sequence/soundfonts/TamaRockSTAR.sf2', 'hvo_sequence/soundfonts/HipHop.sf2',
     'hvo_sequence/soundfonts/HOUSE2.sf2', 'hvo_sequence/soundfonts/phunked.sf2',
     'hvo_sequence/soundfonts/MelottiDrums.sf2', 'hvo_sequence/soundfonts/SJDrums.sf2',
     'hvo_sequence/soundfonts/insane_kit.sf2', 'hvo_sequence/soundfonts/DrumsBySlavo v1.0.sf2',
     'hvo_sequence/soundfonts/ROCK.sf2', 'hvo_sequence/soundfonts/566-HARD ROCK DRUMS V3.sf2',
     'hvo_sequence/soundfonts/Standard_Drum_Kit.sf2', 'hvo_sequence/soundfonts/1115-Standard Rock Set.sf2',
     'hvo_sequence/soundfonts/PremierKit.sf2', 'hvo_sequence/soundfonts/1276-The KiKaZ DrUmZ.sf2',
     'hvo_sequence/soundfonts/test_soundfonts.py', 'hvo_sequence/soundfonts/HardRockDrums.sf2',
     'hvo_sequence/soundfonts/fUNKNewKit.sf2', 'hvo_sequence/soundfonts/5-TamaRockSTAR.sf2',
     'hvo_sequence/soundfonts/Tamburo_Formula.sf2', 'hvo_sequence/soundfonts/DRRP_RocketDrums.sf2',
     'hvo_sequence/soundfonts/kraze_kit.sf2', 'hvo_sequence/soundfonts/ElectricKit.sf2',
     'hvo_sequence/soundfonts/RealAcousticDrumsEXTRA.sf2', 'hvo_sequence/soundfonts/UltimateDrums.sf2']

if __name__ == '__main__':
    allowed_analysis = [
        'Statistical::Total Step Density', 'Statistical::Avg Voice Density',
        'Statistical::Vel Similarity Score', 'Statistical::Weak to Strong Ratio',
        'Syncopation::Lowsync', 'Syncopation::Midsync', 'Syncopation::Hisync',
        'Micro-Timing::Swingness'
    ]

    rh_compiled_kls = {feat.split("::")[-1]: [] for feat in allowed_analysis}
    rh_compiled_OAs = {feat.split("::")[-1]: [] for feat in allowed_analysis}
    rl_compiled_kls = {feat.split("::")[-1]: [] for feat in allowed_analysis}
    rl_compiled_OAs = {feat.split("::")[-1]: [] for feat in allowed_analysis}

    num_samples_per_master_id = 3
    pickle_path = "evaluations_mgeval/figures/inter_intra_soundfontanalysis/"
    pickle_filenam = os.path.join(pickle_path, f"{num_samples_per_master_id}.sfAnalys")
    os.makedirs(pickle_path, exist_ok=True)

    mix_pred_with_input = True

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

    if os.path.exists(pickle_filenam):
        data_dict = pickle.load(open(pickle_filenam, "rb"))

    else:
        for ix, soundfont in enumerate(soundfontnames):

            print(f"!!!!!!!!!!!!!!!!!!! Soundfont {ix} of {len(soundfontnames)} !!!!!!!!!!!!!!!!!!!")
            print(f"!!!!!!!!!!!!!!!!!!! Soundfont {ix} of {len(soundfontnames)} !!!!!!!!!!!!!!!!!!!")
            print(f"!!!!!!!!!!!!!!!!!!! Soundfont {ix} of {len(soundfontnames)} !!!!!!!!!!!!!!!!!!!")

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
                soundfontname = soundfont.split("/")[-1].split(".sf")[0]
                gt = feature_sets[set_labels[0]]
                set1 = feature_sets[set_labels[1]]
                set2 = feature_sets[set_labels[2]]

                fig_path = f"evaluations_mgeval/figures/inter_intra_soundfontanalysis/{set_labels[0]}_{set_labels[1]}_{set_labels[2]}_sf_{soundfontname}"

                # Export Analysis to Table
                csv_path = os.path.join(fig_path, "table4_compiled.csv")
                df, raw_data = compare_two_sets_against_ground_truth(gt, set1, set2, set_labels=set_labels, csv_path=csv_path)


                for feat in rh_compiled_kls.keys():
                    rh_compiled_kls[feat].append(df[('Random (High)', 'Inter-set', 'KL')][feat])
                    rl_compiled_kls[feat].append(df[('Random (Low)', 'Inter-set', 'KL')][feat])
                    rh_compiled_OAs[feat].append(df[('Random (High)', 'Inter-set', 'OA')][feat])
                    rl_compiled_OAs[feat].append(df[('Random (Low)', 'Inter-set', 'OA')][feat])

        data_dict = {
            "soundfontnames": soundfontnames,
            "RH KL": rh_compiled_kls,
            "RH OA": rh_compiled_OAs,
            "RL KL": rl_compiled_kls,
            "RL OA": rl_compiled_OAs,

        }

        pickle.dump(data_dict, open(pickle_filenam, "wb"))


    for group_tag in ["KL", "OA"]:
        data_toplot = {}
        labels = data_toplot
        for key in data_dict.keys():
            if key !="soundfontnames":
                if group_tag in key:
                    data_toplot[key] = data_dict[key]

    data_toplot_reorganized = {feat: {"RL KL": None, "RL OA": None, "RH KL": None, "RH OA": None, }
                               for feat in data_dict["RH KL"].keys()}

    labels_ = [feat for feat in data_dict["RH KL"].keys()]

    for feat in data_dict["RH KL"].keys():
        data_toplot_reorganized[feat]["RL KL"] = data_dict["RL KL"][feat]
        data_toplot_reorganized[feat]["RL OA"] = data_dict["RH OA"][feat]
        data_toplot_reorganized[feat]["RH KL"] = data_dict["RH KL"][feat]
        data_toplot_reorganized[feat]["RH OA"] = data_dict["RH OA"][feat]

    boxplot_soundfonts(data_toplot_reorganized, fs=18, legend_fs=12, legend_ncols=1, fig_path=pickle_path, show=False, ncols=5,
                              figsize=(20, 3), color_map="tab20c", filename=f"KL_{num_samples_per_master_id}_{group_tag}", share_legend=True,
                              sharey=False,
                              show_legend=True, bbox_to_anchor=(.9, 0.12))

