import sys

sys.path.insert(1, "../../GrooveEvaluator")
from GrooveEvaluator.evaluator import Evaluator


class InfillingEvaluator(Evaluator):
    def __init__(self,
                 pickle_source_path,
                 set_subfolder,
                 hvo_pickle_filename,
                 list_of_filter_dicts_for_subsets,
                 _identifier="Train",
                 n_samples_to_use=1024,
                 max_hvo_shape=(32, 27),
                 n_samples_to_synthesize_visualize_per_subset=20,
                 analyze_heatmap=True,
                 analyze_global_features=True,
                 disable_tqdm=True,
                 ):
        Evaluator.__init__(self,
                           pickle_source_path,
                           set_subfolder,
                           hvo_pickle_filename,
                           list_of_filter_dicts_for_subsets,
                           _identifier=_identifier,
                           n_samples_to_use=n_samples_to_use,
                           max_hvo_shape=max_hvo_shape,
                           n_samples_to_synthesize_visualize_per_subset=20,
                           analyze_heatmap=analyze_heatmap,
                           analyze_global_features=analyze_global_features,
                           disable_tqdm=disable_tqdm,
                           )

    def set_gt_hvo_arrays(self, gt_hvo_array):
        self._gt_hvos_array = gt_hvo_array

    def set_gt_hvo_sequences(self, gt_hvo_sequences):
        self._gt_hvo_sequences = gt_hvo_sequences
