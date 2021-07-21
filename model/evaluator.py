import sys
import numpy as np
from tqdm import tqdm
import copy

sys.path.insert(1, "../../GrooveEvaluator")
from GrooveEvaluator.evaluator import Evaluator, HVOSeq_SubSet_Evaluator
from GrooveEvaluator.plotting_utils import separate_figues_by_tabs

sys.path.insert(1, "../preprocessed_dataset/")
from utils import _convert_hvos_array_to_subsets


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
                 n_epochs=None,
                 horizontal=True):

        self.__version___ = "0.3.0"

        self.sf_dict = {}
        self.hvo_comp_dict = {}
        self.horizontal = horizontal

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

        self._gmd_gt_hvo_sequences = []
        self._gt_hvos_array_tags, self._gmd_gt_hvos_array = [], []
        for subset_ix, tag in enumerate(self._gt_tags):
            for sample_ix, sample_hvo in enumerate(self._gt_subsets[subset_ix]):
                self._gmd_gt_hvo_sequences.append(sample_hvo)
                self._gt_hvos_array_tags.append(tag)
                self._gmd_gt_hvos_array.append(sample_hvo.get("hvo"))

        self._gmd_gt_hvos_array = np.stack(self._gmd_gt_hvos_array)

        # preprocess evaluator_subset
        preprocessed_dict = self.dataset.preprocess_dataset(self._gmd_gt_hvo_sequences)
        for key in preprocessed_dict.keys():
            self.__setattr__(key, preprocessed_dict[key])
        del self.processed_outputs
        self.processed_gt = preprocessed_dict["processed_outputs"]
        self._gt_hvo_sequences = preprocessed_dict["hvo_sequences_outputs"]
        self._gt_hvos_array = np.stack([hvo_seq.hvo for hvo_seq in self._gt_hvo_sequences])

        tags = list(set(self._gt_hvos_array_tags))
        hvo_index_dict = {tag: [] for tag in tags}

        for i in range(self._gmd_gt_hvos_array.shape[0]):
            hvo_index_dict[self._gt_hvos_array_tags[i]].append(i)

        # clean unused items (solves out of range index in sfs)
        for subset_idx, subset in enumerate(self._gt_tags):
            items_to_remove = np.where(np.isin(hvo_index_dict[subset], self.unused_items))[0]
            self._gt_subsets[subset_idx] = np.delete(self._gt_subsets[subset_idx], items_to_remove).tolist()
            if len(self._gt_subsets[subset_idx]) == 0:
                self._gt_tags[subset_idx] = None
        self._gt_tags = list(filter(None, self._gt_tags))

        # remove items from _gt that are unused
        self._gmd_gt_hvos_array = np.delete(self._gmd_gt_hvos_array, self.unused_items, axis=0)
        self._gmd_gt_hvo_sequences = np.delete(self._gmd_gt_hvo_sequences, self.unused_items).tolist()

        # add augmented items
        _gt_hvos_array_tags = []
        for idx in self.hvo_index:
            _gt_hvos_array_tags.append(self._gt_hvos_array_tags[idx])
        self._gt_hvos_array_tags = _gt_hvos_array_tags

        hvo_index_dict_gt = {tag: [] for tag in tags}
        for i in range(self._gt_hvos_array.shape[0]):
            hvo_index_dict_gt[self._gt_hvos_array_tags[i]].append(i)

        _gt_subsets = [[] for _ in self._gt_tags]
        for subset_idx, subset in enumerate(self._gt_tags):
            for idx in hvo_index_dict_gt[subset]:
                _gt_subsets[subset_idx].append(self._gt_hvo_sequences[idx])
        self._gt_subsets = _gt_subsets

        self._prediction_hvo_seq_templates = []
        for subset_ix, tag in enumerate(self._gt_tags):
            for sample_ix, sample_hvo in enumerate(self._gt_subsets[subset_ix]):
                self._prediction_hvo_seq_templates.append(sample_hvo.copy_empty())

        # gt subset evaluator
        self.gt_SubSet_Evaluator = HVOSeq_SubSet_InfillingEvaluator(
            self._gt_subsets,  # Ground Truth typically
            self._gt_tags,
            "{}_Ground_Truth".format(self._identifier),  # a name for the subset
            disable_tqdm=self.disable_tqdm,
            group_by_minor_keys=True,
            horizontal=self.horizontal,
            is_gt=True)

        self.audio_sample_locations = self.get_sample_indices(n_samples_to_synthesize_visualize_per_subset)

    def set_pred(self):
        eval_pred = self.model.predict(self.processed_inputs, use_thres=True, thres=0.5)
        eval_pred = [_.cpu() for _ in eval_pred]
        eval_pred = np.concatenate(eval_pred, axis=2)
        """ 
        # This part sets to 0 the hits that are present in the non removed part.
        # This has been removed since the loss calculation in the training process is done without this
        
        eval_pred_hvo_array = np.concatenate(eval_pred, axis=2)
        eval_pred = np.zeros_like(eval_pred_hvo_array)

        n_voices = eval_pred_hvo_array.shape[2] // 3

        for idx in range(eval_pred_hvo_array.shape[0]):

            if horizontal:  # horizontally removing voices
                if isinstance(self.voices_reduced[idx], int):
                    self.voices_reduced[idx] = [self.voices_reduced[idx]]
                h_idx, v_idx, o_idx = get_hvo_idxs_for_voice(voice_idx=list(self.voices_reduced[idx]),
                                                             n_voices=n_voices)
                eval_pred[idx, :, h_idx + v_idx + o_idx] = eval_pred_hvo_array[idx, :, h_idx + v_idx + o_idx]


            else:  # randomly removing events       # FIXME don't set to 0 the predictions that overlap with the input
                hits = self.hvo_sequences_inputs[idx].hvo[:n_voices]
                input_hits_idx = np.nonzero(hits)
                eval_pred[idx, :, :] = eval_pred_hvo_array[idx, :, :]
                eval_pred[idx, tuple(input_hits_idx)] = 0
        """

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
            "{}_Predictions".format(self._identifier),  # a name for the subset
            disable_tqdm=self.disable_tqdm,
            group_by_minor_keys=True,
            horizontal=self.horizontal,
            is_gt=False
            )

        sf_dict, hvo_comp_dict = {}, {}
        for key in self.audio_sample_locations.keys():
            sf_dict[key] = []
            hvo_comp_dict[key] = []
            for idx in self.audio_sample_locations[key]:
                sf_dict[key].append(self.soundfonts[self._subset_hvo_array_index[key][idx]])
                hvo_comp_dict[key].append(self.hvo_sequences_inputs[self._subset_hvo_array_index[key][idx]])
        self.sf_dict = sf_dict
        self.hvo_comp_dict = hvo_comp_dict

        # set soundfonts in subset classes and hvo comp to render non-removed voices in get_audio
        self.gt_SubSet_Evaluator.sf_dict = self.sf_dict
        self.prediction_SubSet_Evaluator.sf_dict = self.sf_dict
        self.gt_SubSet_Evaluator.hvo_comp_dict = self.hvo_comp_dict
        self.prediction_SubSet_Evaluator.hvo_comp_dict = self.hvo_comp_dict

    def get_gmd_ground_truth_hvo_sequences(self):  # for testing
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
            sf_dict={},
            hvo_comp_dict={},
            horizontal=True,
            is_gt=None
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

        self.horizontal = horizontal
        self.is_gt = is_gt
        self.sf_dict = sf_dict
        self.hvo_comp_dict = hvo_comp_dict

    def get_audios(self, _, use_specific_samples_at=None):
        """ use_specific_samples_at: must be a list of tuples of (subset_ix, sample_ix) denoting to get
        audio from the sample_ix in subset_ix """

        self._sampled_hvos = self.get_hvo_samples_located_at(use_specific_samples_at)

        audios, captions = [], []

        for key in tqdm(self._sampled_hvos.keys(),
                        desc='Synthesizing samples - {} '.format(self.set_identifier),
                        disable=self.disable_tqdm):
            for idx, _sample_hvo in enumerate(self._sampled_hvos[key]):
                sample_hvo = _sample_hvo.copy()  # make sure not to modify og hvo

                # add 'context'
                sample_hvo = self.add_removed_part_to_hvo(sample_hvo, key, idx)

                sf_path = self.sf_dict[key][idx]  # force usage of sf_dict
                audios.append(sample_hvo.synthesize(sf_path=sf_path))
                captions.append("{}_{}_{}_{}.wav".format(
                    self.set_identifier, sample_hvo.metadata.style_primary,
                    sample_hvo.metadata.master_id.replace("/", "_"), str(idx)
                ))

        # sort so that they are alphabetically ordered in wandb
        sort_index = np.argsort(captions)
        captions = np.array(captions)[sort_index].tolist()
        audios = np.array(audios)[sort_index].tolist()

        return list(zip(captions, audios))

    def get_piano_rolls(self, use_specific_samples_at=None):
        """ use_specific_samples_at: must be a dict of lists of (sample_ix) """

        self._sampled_hvos = self.get_hvo_samples_located_at(use_specific_samples_at)
        tab_titles = []
        piano_roll_tabs = []
        for subset_ix, tag in tqdm(enumerate(self._sampled_hvos.keys()),
                                   desc='Creating Piano rolls for ' + self.set_identifier,
                                   disable=self.disable_tqdm):
            piano_rolls = []
            for idx, _sample_hvo in enumerate(self._sampled_hvos[tag]):
                sample_hvo = _sample_hvo.copy()  # make sure not to modify og hvo

                # add 'context'
                if self.horizontal and self.is_gt:
                    sample_hvo = self.add_removed_part_to_hvo(sample_hvo, tag, idx)

                title = "{}_{}_{}_{}".format(
                    self.set_identifier, sample_hvo.metadata.style_primary,
                    sample_hvo.metadata.master_id.replace("/", "_"), str(idx))
                piano_rolls.append(sample_hvo.to_html_plot(filename=title))
            piano_roll_tabs.append(separate_figues_by_tabs(piano_rolls, [str(x) for x in range(len(piano_rolls))]))
            tab_titles.append(tag)

        # sort so that they are alphabetically ordered in wandb
        sort_index = np.argsort(tab_titles)
        tab_titles = np.array(tab_titles)[sort_index].tolist()
        piano_roll_tabs = np.array(piano_roll_tabs)[sort_index].tolist()

        return separate_figues_by_tabs(piano_roll_tabs, [tag for tag in tab_titles])

    def add_removed_part_to_hvo(self, sample_hvo, key, idx):

        hvo_comp = self.hvo_comp_dict[key][idx]
        non_zero_idx = np.nonzero(hvo_comp.hvo[:, :len(hvo_comp.drum_mapping)])
        sample_hvo.hvo[non_zero_idx] = 0  # make sure that predicted hits don't overwrite input hits
        sample_hvo.hvo = sample_hvo.hvo + hvo_comp.hvo

        return sample_hvo

    def get_wandb_logging_media(self, velocity_heatmap_html=True, global_features_html=True,
                                piano_roll_html=True, audio_files=True, sf_paths=None, use_specific_samples_at=None):

        _wandb_media = super(HVOSeq_SubSet_InfillingEvaluator, self).get_wandb_logging_media(
            velocity_heatmap_html=velocity_heatmap_html,
            global_features_html=global_features_html,
            piano_roll_html=piano_roll_html,
            audio_files=audio_files,
            sf_paths=sf_paths,
            use_specific_samples_at=use_specific_samples_at)

        wandb_media = copy.deepcopy(_wandb_media)

        if ('_').join(self.set_identifier.split('_')[2:]) == 'Ground_Truth' and self.horizontal:
            wandb_media['audios'][self.set_identifier + '_with_removed_parts'] = _wandb_media['audios'][
                self.set_identifier]
            wandb_media['piano_roll_html'][self.set_identifier + '_with_removed_parts'] = \
                _wandb_media['piano_roll_html'][self.set_identifier]

            del wandb_media['audios'][self.set_identifier]
            del wandb_media['piano_roll_html'][self.set_identifier]

        return wandb_media
