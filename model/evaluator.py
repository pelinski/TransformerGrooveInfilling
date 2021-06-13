import sys
import numpy as np

sys.path.insert(1, "../../GrooveEvaluator")
from GrooveEvaluator.evaluator import Evaluator
from utils import get_hvo_idx_for_voice


class InfillingEvaluator(Evaluator):
    def __init__(self,
                 pickle_source_path,
                 set_subfolder,
                 hvo_pickle_filename,
                 _identifier="Train",
                 n_samples_to_use=1024,
                 max_hvo_shape=(32, 27),
                 n_samples_to_synthesize_visualize_per_subset=20,
                 analyze_heatmap=True,
                 analyze_global_features=True,
                 disable_tqdm=True,
                 dataset=None,
                 model=None,
                 n_epochs=None):

        # common filters
        eval_styles = ["hiphop", "funk", "reggae", "soul", "latin", "jazz", "pop", "afrobeat", "highlife", "punk",
                       "rock"]
        list_of_filter_dicts_for_subsets = []
        for style in eval_styles:
            list_of_filter_dicts_for_subsets.append(
                {"style_primary": [style], "beat_type": ["beat"], "time_signature": ["4-4"]}
            )

        super(InfillingEvaluator, self).__init__(pickle_source_path,
                                                 set_subfolder,
                                                 hvo_pickle_filename,
                                                 list_of_filter_dicts_for_subsets,
                                                 _identifier=_identifier,
                                                 n_samples_to_use=n_samples_to_use,
                                                 max_hvo_shape=max_hvo_shape,
                                                 n_samples_to_synthesize_visualize_per_subset=n_samples_to_synthesize_visualize_per_subset,
                                                 analyze_heatmap=analyze_heatmap,
                                                 analyze_global_features=analyze_global_features,
                                                 disable_tqdm=disable_tqdm,
                                                 )

        self.dataset = dataset
        self.model = model
        self.eps = n_epochs


        # log frequency
        first_epochs_step = 1
        first_epochs_lim = 10 if self.eps >= 10 else self.eps
        self.epoch_save_partial = np.arange(first_epochs_lim, step=first_epochs_step)
        self.epoch_save_all = np.arange(first_epochs_lim, step=first_epochs_step)
        if first_epochs_lim != self.eps:
            remaining_epochs_step_partial, remaining_epochs_step_all = 5, 10
            self.epoch_save_partial = np.append(self.epoch_save_partial,
                                           np.arange(start=first_epochs_lim, step=remaining_epochs_step_partial,
                                                     stop=self.eps))
            self.epoch_save_all = np.append(self.epoch_save_all,
                                       np.arange(start=first_epochs_lim, step=remaining_epochs_step_all, stop=self.eps))

    def set_gt(self):
        # get gt evaluator
        evaluator_subset = self.get_ground_truth_hvo_sequences()

        # preprocess evaluator_subset
        (self.eval_processed_inputs, self.eval_processed_gt), \
        (_, _, eval_hvo_sequences_gt), \
        (self.eval_hvo_index, self.eval_voices_reduced, self.eval_soundfonts) = self.dataset.preprocess_dataset(
            evaluator_subset)

        # get gt
        eval_hvo_array = np.stack([hvo_seq.hvo for hvo_seq in eval_hvo_sequences_gt])

        self._gt_hvos_array = eval_hvo_array
        self._gt_hvo_sequences = eval_hvo_sequences_gt

    def set_pred(self):
        eval_pred = self.model.predict(self.eval_processed_inputs, use_thres=True, thres=0.5)

        eval_pred_hvo_array = np.concatenate(eval_pred, axis=2)
        eval_pred  = np.zeros_like(eval_pred_hvo_array)
        # sets all voices different from voices_reduced to 0
        # sync between hits and vels+offs is done when converted to hvo sequence
        for idx in range(eval_pred_hvo_array.shape[0]):  # N
            # FIXME works for only one voice
            h_idx, v_idx, o_idx = get_hvo_idx_for_voice(voice_idx=self.eval_voices_reduced[idx],
                                                        n_voices=eval_pred_hvo_array.shape[2] // 3)
            eval_pred[idx, :, h_idx] = eval_pred_hvo_array[idx][:, h_idx]
            eval_pred[idx, :, v_idx] = eval_pred_hvo_array[idx][:, v_idx]
            eval_pred[idx, :, o_idx] = eval_pred_hvo_array[idx][:, o_idx]

        self.add_predictions(eval_pred)
