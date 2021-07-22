import copy
from evaluator import InfillingEvaluator
from preprocess_dataset import load_preprocessed_dataset

params = {
    "dataset_paths": {
        "InfillingKicksAndSnares": {
            "train": '../datasets/InfillingKicksAndSnares/0.1.2/train',
            'test': '../datasets/InfillingKicksAndSnares/0.1.2/test'
        },
        "InfillingRandom" : {
            "train":  '../datasets/InfillingRandom/0.0.0/train',
            'test':  '../datasets/InfillingRandom/0.0.0/test'
        },
        "InfillingKicksAndSnares_testing": {
            "train": '../datasets/InfillingKicksAndSnares_testing/0.1.2/train',
            'test': '../datasets/InfillingKicksAndSnares_testing/0.1.2/test'
        },
        "InfillingRandom_testing":
            {
                "train":'../datasets/InfillingRandom_testing/0.0.0/train',
                'test': '../datasets/InfillingRandom_testing/0.0.0/test'
            }
    },
    "evaluator": {
        "n_samples_to_use": 2048,  # 2048
        "n_samples_to_synthesize_visualize_per_subset": 10,
        "save_evaluator_path": '../evaluators/'}}

if __name__ == "__main__":
    exps = ['InfillingKicksAndSnares']
    splits = ['train', 'test']
    for exp in exps:
        print('------------------------\n' + exp + '\n------------------------\n')
        for split in splits:
            print('Split: ', split)
            dataset = load_preprocessed_dataset(params["dataset_paths"][exp][split], exp=exp)

            params_exp = copy.deepcopy(params)
            params_exp['evaluator']['save_evaluator_path'] = params_exp['evaluator']['save_evaluator_path'] + exp + '/'

            pred_horizontal = False if (exp == 'InfillingRandom' or exp == 'InfillingRandom_testing') else True
            evaluator = InfillingEvaluator(
                pickle_source_path=dataset.subset_info["pickle_source_path"],
                set_subfolder=dataset.subset_info["subset"],
                hvo_pickle_filename=dataset.subset_info["hvo_pickle_filename"],
                max_hvo_shape=(32, 27),
                n_samples_to_use=params['evaluator']["n_samples_to_use"],
                n_samples_to_synthesize_visualize_per_subset=params['evaluator'][
                    "n_samples_to_synthesize_visualize_per_subset"],
                _identifier=split.capitalize() + '_Set',
                disable_tqdm=False,
                analyze_heatmap=True,
                analyze_global_features=False,  # pdf
                dataset=dataset,
                horizontal=pred_horizontal)

            evaluator.save_as_pickle(save_evaluator_path=params_exp['evaluator']['save_evaluator_path'])