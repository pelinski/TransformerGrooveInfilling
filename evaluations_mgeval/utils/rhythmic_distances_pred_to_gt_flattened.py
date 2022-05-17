from post_processing_scripts.mgeval_rytm_utils import *
from copy import deepcopy
from tqdm import tqdm

if __name__ == '__main__':

    gmd_eval = pickle.load(open(
            f"post_processing_scripts/evaluators_monotonic_groove_transformer_v1/"
            f"validation_set_evaluator_run_misunderstood-bush-246_Epoch_26.Eval","rb"))

    down_size = 1024
    final_indices = sample_uniformly(gmd_eval, num_samples=down_size) if down_size < 1024 else list(range(1024))

    # Compile data (flatten styles)
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
        "robust":
            pickle.load(open(f"post_processing_scripts/evaluators_monotonic_groove_transformer_v1/"
                             f"validation_set_evaluator_run_robust_sweep_29.Eval", "rb")),
        "colorful":
            pickle.load(open(f"post_processing_scripts/evaluators_monotonic_groove_transformer_v1/"
                             f"validation_set_evaluator_run_colorful_sweep_41.Eval", "rb"))
    }

    prediction_hvos = {set_name: flatten(set_evaluator._prediction_subsets) for (set_name, set_evaluator) in sets_evals.items()}
    gt_hvos = {set_name: flatten(set_evaluator._gt_subsets) for (set_name, set_evaluator) in sets_evals.items()}
    distance_dicts = {set_name: None for set_name in sets_evals.keys()}


    for set_name in tqdm(sets_evals.keys()):
        for hvo_pd, hvo_gt in zip(prediction_hvos[set_name], gt_hvos[set_name]):
            hvo_pd.hvo = hvo_pd.flatten_voices(voice_idx=2, velocity_aggregator_modes=3)
            hvo_gt.hvo = hvo_gt.flatten_voices(voice_idx=2, velocity_aggregator_modes=3)
            calculated_distances_dict = hvo_pd.calculate_all_distances_with(hvo_gt)
            if distance_dicts[set_name] is None:
                distance_dicts[set_name] = {distance_measure: [] for distance_measure in calculated_distances_dict.keys()}
            for distance_measure in calculated_distances_dict.keys():
                distance_dicts[set_name][distance_measure].append(calculated_distances_dict[distance_measure])

    remove_distances = ['l1_distance_hvo', 'l1_distance_h', 'l1_distance_v', 'l1_distance_o', 'l2_distance_hvo', 'l2_distance_h', 'l2_distance_v', 'l2_distance_o']

    updated_distance_dicts = {set_name: {} for set_name in sets_evals.keys()}
    for set_name in sets_evals.keys():
        for feature in distance_dicts[set_name].keys():
            if feature not in remove_distances:
                vals = np.array(distance_dicts[set_name][feature])
                vals = vals[~np.isnan(vals)]
                updated_distance_dicts[set_name].update({feature: vals})


    fig_path = "post_processing_scripts/evaluators_monotonic_groove_transformer_v1/mgeval_results/boxplots"
    boxplot_absolute_measures(updated_distance_dicts, fs=12, legend_fs=10, legend_ncols=3, fig_path=fig_path, show=True, ncols=4,
                              figsize=(20, 10), color_map="tab20c", filename="distances_flattened.png", shift_colors_by=1,
                              auto_adjust_ylim=True)
