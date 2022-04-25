import torch
import json
from evaluator import InfillingEvaluator
from process_dataset import load_processed_dataset

with open("datasets/evaluation_subsets/evaluation_subsets_parameters.json") as f:
    params = json.load(f)

if __name__ == "__main__":

    testing = False

    exps = ["InfillingClosedHH"]
    splits = ["test", "train", "validation"]
    for exp in exps:
        print("------------------------\n" + exp + "\n------------------------\n")
        for split in splits:
            print("Split: ", split)

            _exp = exp + "_testing" if testing else exp
            if testing:
                params["evaluator"]["n_samples_to_use"] = 10
                params["evaluator"]["n_samples_to_synthesize_visualize_per_subset"] = 5

            pred_horizontal = (
                False
                if exp == "InfillingRandom" or exp == "InfillingRandomLow"
                else True
            )

            dataset = load_processed_dataset(
                params["dataset_paths"][_exp][split], exp=exp
            )

            evaluator = InfillingEvaluator(
                pickle_source_path=dataset.subset_info["pickle_source_path"],
                set_subfolder=dataset.subset_info["subset"],
                hvo_pickle_filename=dataset.subset_info["hvo_pickle_filename"],
                max_hvo_shape=(32, 27),
                n_samples_to_use=params["evaluator"]["n_samples_to_use"],
                n_samples_to_synthesize_visualize_per_subset=params["evaluator"][
                    "n_samples_to_synthesize_visualize_per_subset"
                ],
                _identifier=split.capitalize() + "_Set",
                disable_tqdm=False,
                analyze_heatmap=True,
                analyze_global_features=False,  # pdf
                dataset=dataset,
                horizontal=pred_horizontal,
                device="cuda" if torch.cuda.is_available() else "cpu",
            )

            evaluator.save_as_pickle(
                save_evaluator_path=params["evaluator"]["save_evaluator_path"]
            )
