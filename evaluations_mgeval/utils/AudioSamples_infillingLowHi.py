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

    gt_hvo_seqs_downsampled, gt_hvos_array_downsampled, sampled_loop_ids = downsample_set(evaluator_gt,
                                                                                          num_samples_per_master_id=num_samples_per_master_id)

    gt_features = get_feature_dict_from(gt_hvo_seqs_downsampled, set_tag="Ground Truth")

    hvo_seqs_list_removed_rl = get_version_with_randomly_removed_voices(gt_hvo_seqs_downsampled, set_tag="Ground Truth",
                                                                        thres_range=(0.1, 0.3))
    hvo_seqs_list_removed_rh = get_version_with_randomly_removed_voices(gt_hvo_seqs_downsampled, set_tag="Ground Truth",
                                                                        thres_range=(0.4, 0.7))
    hvos_gt = np.array([hvo_sample.hvo for hvo_sample in gt_hvo_seqs_downsampled])
    empty_hvo_seqs_template =  [hvo_sample.copy_zero() for hvo_sample in gt_hvo_seqs_downsampled]
    target_hvo_seqs = deepcopy([hvo_sample for hvo_sample in gt_hvo_seqs_downsampled])

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

    input_hvo_sequences = deepcopy(empty_hvo_seqs_template)
    predicted_infilling_hvo_seqs = deepcopy(empty_hvo_seqs_template)
    mixed_hvo_seqs = deepcopy(empty_hvo_seqs_template)

    for model_name, input in model_inputs.items():
        audio_path = f"audios_validation_set/{model_name}"

        os.makedirs(audio_path, exist_ok=True)

        model_path = "trained_models"
        model = load_model(model_path, model_name)
        h, v, o = model.predict(torch.tensor(input, dtype=torch.float32))
        predicted_hvos_array = torch.cat((h, v, o), axis=2).detach().numpy()
        input_with_predicted_output = mix_input_with_prediction(corresponding_hvo_inputs[model_name],
                                                                predicted_hvos_array) if mix_pred_with_input else predicted_hvos_array

        # add to hvo_sequence objects
        for ix, predicted_hvo_array in enumerate(input_with_predicted_output):
            predicted_infilling_hvo_seqs[ix].hvo = predicted_hvos_array[ix]
            mixed_hvo_seqs[ix].hvo = input_with_predicted_output[ix]

            '''input_hvo_seqs[ix].save_audio(filename=os.path.join(audio_path, f"{ix}_masked_input_"), sr=44100,
                                          sf_path=soundfont)()
            predicted_infilling_hvo_seqs.save_audio(filename=os.path.join(audio_path, f"{ix}_masked_input_"), sr=44100,
                                          sf_path=soundfont)()'''
            target_hvo_seqs[ix].save_audio(filename=os.path.join(audio_path, f"{ix}_A_target.wav"), sr=44100,
                                           sf_path=soundfont)
            predicted_infilling_hvo_seqs[ix].save_audio(
                filename=os.path.join(audio_path, f"{ix}_B_generation_only.wav"), sr=44100,
                sf_path=soundfont)
            mixed_hvo_seqs[ix].save_audio(filename=os.path.join(audio_path, f"{ix}_C_generation_mixed_with_input.wav"),
                                          sr=44100,
                                          sf_path=soundfont)