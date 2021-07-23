import torch
from evaluator import InfillingEvaluator
from preprocess_dataset import load_preprocessed_dataset

params = {
    "dataset_paths": {
        "InfillingKicksAndSnares": {
            "train": '../datasets/InfillingKicksAndSnares/0.1.2/train',
            'test': '../datasets/InfillingKicksAndSnares/0.1.2/test'
        },
        "InfillingRandom": {
            "train": '../datasets/InfillingRandom/0.0.0/train',
            'test': '../datasets/InfillingRandom/0.0.0/test'
        },
        "InfillingKicksAndSnares_testing": {
            "train": '../datasets/InfillingKicksAndSnares_testing/0.1.3/train',
            'test': '../datasets/InfillingKicksAndSnares_testing/0.1.3/test'
        },
        "InfillingRandom_testing":
            {
                "train": '../datasets/InfillingRandom_testing/0.0.1/train',
                'test': '../datasets/InfillingRandom_testing/0.0.1/test'
            }
    },
    "evaluator": {
        "n_samples_to_use": 10,  # 2048
        "n_samples_to_synthesize_visualize_per_subset": 5,  # 10
        "save_evaluator_path": '../evaluators/'}}

if __name__ == "__main__":

    testing = True

    exps = ['InfillingRandom']
    splits = ['train', 'test']
    for exp in exps:
        print('------------------------\n' + exp + '\n------------------------\n')
        for split in splits:
            print('Split: ', split)

            _exp = exp + '_testing' if testing else exp

            pred_horizontal = False if exp == 'InfillingRandom' else True

            dataset = load_preprocessed_dataset(params["dataset_paths"][_exp][split], exp=exp)  # FIXME

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
                horizontal=pred_horizontal,
                device='cuda' if torch.cuda.is_available() else 'cpu')

            save_evaluator_path = params['evaluator']['save_evaluator_path'] + _exp + '/' + dataset.__version__ + '/'
            evaluator.save_as_pickle(save_evaluator_path=save_evaluator_path)
