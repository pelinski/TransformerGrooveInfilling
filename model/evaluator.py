import sys
import numpy as np
from tqdm import tqdm
import copy

sys.path.insert(1, "../../GrooveEvaluator")
from GrooveEvaluator.evaluator import Evaluator, HVOSeq_SubSet_Evaluator

sys.path.insert(1, "../preprocessed_dataset/")
from Subset_Creators import subsetters
from utils import get_hvo_idx_for_voice, _convert_hvos_array_to_subsets


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

        self.sf_dict = {}

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
                                                 disable_tqdm=disable_tqdm)

        self.dataset = dataset
        self.model = model
        self.eps = n_epochs

        # Create subsets of data
        gt_subsetter_sampler = subsetters.GrooveMidiSubsetterAndSampler(
            pickle_source_path=pickle_source_path, subset=set_subfolder, hvo_pickle_filename=hvo_pickle_filename,
            list_of_filter_dicts_for_subsets=list_of_filter_dicts_for_subsets,
            number_of_samples=n_samples_to_use,
            max_hvo_shape=max_hvo_shape,
            at_least_one_hit_in_voices=dataset.voices_params["voice_idx"]
        )

        # _gt_tags --> ground truth tags for each subset in _gt_subsets
        self._gt_tags, self._gt_subsets = gt_subsetter_sampler.get_subsets()

        # _gt_hvos_array_tags --> ground truth tags for each
        # _gt_hvos_arrayS --> a numpy array containing all samples in hvo format
        self._gmd_gt_hvo_sequences = []
        self._gt_hvos_array_tags, self._gmd_gt_hvos_array, self._prediction_hvo_seq_templates = [], [], []
        for subset_ix, tag in enumerate(self._gt_tags):
            for sample_ix, sample_hvo in enumerate(self._gt_subsets[subset_ix]):
                self._gmd_gt_hvo_sequences.append(sample_hvo)
                self._gt_hvos_array_tags.append(tag)
                self._gmd_gt_hvos_array.append(sample_hvo.get("hvo"))
                self._prediction_hvo_seq_templates.append(sample_hvo.copy_empty())

        self._gmd_gt_hvos_array = np.stack(self._gmd_gt_hvos_array)
        self._gmd_gt_hvo_sequences = self._gmd_gt_hvo_sequences

        # preprocess evaluator_subset
        preprocessed_dict = self.dataset.preprocess_dataset(self._gmd_gt_hvo_sequences)
        self.eval_processed_inputs = preprocessed_dict["processed_inputs"]
        self.eval_processed_gt = preprocessed_dict["processed_outputs"]
        eval_hvo_sequences_gt = preprocessed_dict["hvo_sequences_outputs"]
        self.eval_hvo_index = preprocessed_dict["hvo_index"]
        self.eval_voices_reduced = preprocessed_dict["voices_reduced"]
        self.eval_soundfonts = preprocessed_dict["soundfonts"]

        # get gt
        eval_hvo_array = np.stack([hvo_seq.hvo for hvo_seq in eval_hvo_sequences_gt])

        self._gt_hvos_array = eval_hvo_array
        self._gt_hvo_sequences = eval_hvo_sequences_gt

        self.gt_SubSet_Evaluator = HVOSeq_SubSet_InfillingEvaluator(
            self._gt_subsets,  # Ground Truth typically
            self._gt_tags,
            "{}_Set_Ground_Truth".format(self._identifier),  # a name for the subset
            disable_tqdm=self.disable_tqdm,
            group_by_minor_keys=True)

        self.audio_sample_locations = self.get_sample_indices(n_samples_to_synthesize_visualize_per_subset)

    def set_pred(self):
        eval_pred = self.model.predict(self.eval_processed_inputs, use_thres=True, thres=0.5)

        eval_pred_hvo_array = np.concatenate(eval_pred, axis=2)
        eval_pred = np.zeros_like(eval_pred_hvo_array)
        # sets all voices different from voices_reduced to 0
        # sync between hits and vels+offs is done when converted to hvo sequence
        # FIXME avoid for loop
        for idx in range(eval_pred_hvo_array.shape[0]):  # N
            # FIXME works for only one voice
            h_idx, v_idx, o_idx = get_hvo_idx_for_voice(voice_idx=self.eval_voices_reduced[idx],
                                                        n_voices=eval_pred_hvo_array.shape[2] // 3)
            eval_pred[idx, :, h_idx] = eval_pred_hvo_array[idx][:, h_idx]
            eval_pred[idx, :, v_idx] = eval_pred_hvo_array[idx][:, v_idx]
            eval_pred[idx, :, o_idx] = eval_pred_hvo_array[idx][:, o_idx]


        self._prediction_hvos_array = eval_pred
        self._prediction_tags, self._prediction_subsets, self._subset_hvo_array_index = \
            _convert_hvos_array_to_subsets(
                self._gt_hvos_array_tags,
                self._prediction_hvos_array,
                self._prediction_hvo_seq_templates
            )

        self.prediction_SubSet_Evaluator = HVOSeq_SubSet_InfillingEvaluator(
            self._prediction_subsets,
            self._prediction_tags,
            "{}_Set_Predictions".format(self._identifier),  # a name for the subset
            disable_tqdm=self.disable_tqdm,
            group_by_minor_keys=True)

        sf_dict = {}
        for key in self.audio_sample_locations.keys():
            sf_dict[key] = []
            for idx in self.audio_sample_locations[key]:
                sf_dict[key].append(self.eval_soundfonts[self._subset_hvo_array_index[key][idx]])

        self.sf_dict = sf_dict

        #set soundfonts in subset classes
        self.gt_SubSet_Evaluator.sf_dict = self.sf_dict
        self.prediction_SubSet_Evaluator.sf_dict = self.sf_dict

    def get_gmd_ground_truth_hvo_sequences(self): # for testing
        return copy.deepcopy(self._gmd_gt_hvo_sequences)




class HVOSeq_SubSet_InfillingEvaluator(HVOSeq_SubSet_Evaluator):
    def __init__(
            self,
            set_subsets,
            set_tags,
            set_identifier,
            max_samples_in_subset=None,
            n_samples_to_synthesize_visualize=10,
            disable_tqdm=True,
            group_by_minor_keys=True,
            analyze_heatmap=True,
            analyze_global_features=True,
            sf_dict = {}
    ):
        super(HVOSeq_SubSet_InfillingEvaluator, self).__init__(set_subsets,
                                                               set_tags,
                                                               set_identifier,
                                                               max_samples_in_subset,
                                                               n_samples_to_synthesize_visualize,
                                                               disable_tqdm,
                                                               group_by_minor_keys,
                                                               analyze_heatmap,
                                                               analyze_global_features)

        self.sf_dict = sf_dict

    def get_audios(self, _, use_specific_samples_at=None):
        """ use_specific_samples_at: must be a list of tuples of (subset_ix, sample_ix) denoting to get
        audio from the sample_ix in subset_ix """

        self._sampled_hvos = self.get_hvo_samples_located_at(use_specific_samples_at)

        audios = []
        captions = []

        for key in tqdm(self._sampled_hvos.keys(),
                        desc='Synthesizing samples - {} '.format(self.set_identifier),
                        disable=self.disable_tqdm):
            for idx, sample_hvo in enumerate(self._sampled_hvos[key]):
                sf_path = self.sf_dict[key][idx]    # force usage of sf_dict
                audios.append(sample_hvo.synthesize(sf_path=sf_path))
                captions.append("{}_{}_{}.wav".format(
                    self.set_identifier, sample_hvo.metadata.style_primary,
                    sample_hvo.metadata.master_id.replace("/", "_")
                ))

        return list(zip(captions, audios))
