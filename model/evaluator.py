import sys
import os
import numpy as np
from tqdm import tqdm
import copy
import pickle
import wandb
import torch

from bokeh.embed import file_html
from bokeh.resources import CDN

sys.path.insert(1, "../../GrooveEvaluator")
from GrooveEvaluator.evaluator import Evaluator, HVOSeq_SubSet_Evaluator
from GrooveEvaluator.plotting_utils import separate_figues_by_tabs
from GrooveEvaluator.utils import get_stats_from_evaluator

sys.path.insert(1, "../../hvo_sequence")
from hvo_sequence.drum_mappings import ROLAND_REDUCED_MAPPING

sys.path.insert(1, "../preprocessed_dataset/")
from utils import _convert_hvos_array_to_subsets, save_to_pickle


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
                 horizontal=True,
                 device='cuda' if torch.cuda.is_available() else 'cpu'):

        self.__version___ = "0.3.2"

        self.sf_dict = {}
        self.hvo_comp_dict = {}
        self.horizontal = horizontal
        self.device = device

        # common filters
        eval_styles = ["hiphop", "funk", "reggae", "soul", "latin", "jazz", "pop", "afrobeat", "highlife", "punk",
                       "rock"]
        list_of_filter_dicts_for_subsets = []
        for style in eval_styles:
            list_of_filter_dicts_for_subsets.append(
                {"style_primary": [style], "beat_type": ["beat"], "time_signature": ["4-4"]}
            )
        # TODO bypass feature extractor
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
        self._identifier = _identifier
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
            "Ground_Truth_" + self._identifier,  # a name for the subset
            disable_tqdm=self.disable_tqdm,
            group_by_minor_keys=True,
            horizontal=self.horizontal,
            is_gt=True)

        self.audio_sample_locations = self.get_sample_indices(n_samples_to_synthesize_visualize_per_subset)

    def set_pred(self, model):
        self.processed_inputs = self.processed_inputs.to(self.device)
        eval_pred = model.predict(self.processed_inputs, use_thres=True, thres=0.5)
        eval_pred = [_.cpu() for _ in eval_pred]
        eval_pred = np.concatenate(eval_pred, axis=2)

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
            "Predictions_" + self._identifier,  # a name for the subset
            disable_tqdm=self.disable_tqdm,
            group_by_minor_keys=True,
            horizontal=self.horizontal,
            is_gt=False
        )
        self.gt_SubSet_Evaluator.set_identifier = 'Ground_Truth_' + self._identifier

        sf_dict, hvo_comp_dict = {}, {}
        for key in self.audio_sample_locations.keys():
            sf_dict[key] = []
            hvo_comp_dict[key] = []
            for idx in self.audio_sample_locations[key]:
                if hasattr(self, 'soundfonts'):
                    sf_dict[key].append(self.soundfonts[self._subset_hvo_array_index[key][idx]])
                else:  # symbolic dataset
                    sf_dict[key].append('../soundfonts/filtered_soundfonts/Standard_Drum_Kit.sf2')
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

    def save_as_pickle(self, save_evaluator_path):

        save_evaluator_path = os.path.join(save_evaluator_path, 'InfillingEvaluator_' + self.__version___)

        if not os.path.exists(save_evaluator_path):
            os.makedirs(save_evaluator_path)

        filename = os.path.join(save_evaluator_path, self.dataset.dataset_name + '_' + self.dataset.split +
                                '_' + self.dataset.__version__ + '_evaluator.pickle')
        save_to_pickle(self, filename)


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
            is_gt=None,
            epoch=None
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
        self.epoch = epoch

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

                title = "{}_{}_{}_{}.wav".format(
                    self.set_identifier, sample_hvo.metadata.style_primary,
                    sample_hvo.metadata.master_id.replace("/", "_"), str(idx)
                )
                if not self.is_gt:
                    title = "epoch_{}_{}".format(self.epoch, title)
                captions.append(title)

        # sort so that they are alphabetically ordered in wandb
        sort_index = np.argsort(captions)
        captions = np.array(captions)[sort_index].tolist()
        audios = np.array(audios)[sort_index].tolist()

        return list(zip(captions, audios))

    def get_piano_rolls(self, use_specific_samples_at=None, add_inputs=False):
        """ use_specific_samples_at: must be a dict of lists of (sample_ix) """

        self._sampled_hvos = self.get_hvo_samples_located_at(use_specific_samples_at)
        tab_titles, piano_roll_tabs = [], []
        for subset_ix, tag in tqdm(enumerate(self._sampled_hvos.keys()),
                                   desc='Creating Piano rolls for ' + self.set_identifier,
                                   disable=self.disable_tqdm):
            piano_rolls = []
            for idx, _sample_hvo in enumerate(self._sampled_hvos[tag]):
                sample_hvo = _sample_hvo.copy()  # make sure not to modify og hvo

                if add_inputs:
                    sample_hvo = self.add_removed_part_to_hvo(sample_hvo, tag, idx)

                title = "{}_{}_{}_{}".format(
                    self.set_identifier, sample_hvo.metadata.style_primary,
                    sample_hvo.metadata.master_id.replace("/", "_"), str(idx))
                if not self.is_gt:
                    title = "epoch_{}_{}".format(self.epoch, title)

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

    def get_logging_dict(self, velocity_heatmap_html=True, global_features_html=True,
                         piano_roll_html=True, audio_files=True, sf_paths=None, use_specific_samples_at=None):

        logging_dict = super(HVOSeq_SubSet_InfillingEvaluator, self).get_logging_dict(
            velocity_heatmap_html=velocity_heatmap_html,
            global_features_html=global_features_html,
            piano_roll_html=piano_roll_html,
            audio_files=audio_files,
            sf_paths=sf_paths,
            use_specific_samples_at=use_specific_samples_at)

        if piano_roll_html is True:
            logging_dict.update({"piano_rolls_plus_inputs": self.get_piano_rolls(use_specific_samples_at,
                                                                                 add_inputs=True)})

        return logging_dict

    def get_wandb_logging_media(self, velocity_heatmap_html=True, global_features_html=True,
                                piano_roll_html=True, audio_files=True, sf_paths=None, use_specific_samples_at=None):

        self._sampled_hvos = self.get_hvo_samples_located_at(use_specific_samples_at)

        logging_dict = self.get_logging_dict(velocity_heatmap_html, global_features_html,
                                             piano_roll_html, audio_files, sf_paths, use_specific_samples_at)

        wandb_media_dict = {}
        for key in logging_dict.keys():
            if velocity_heatmap_html is True and key in "velocity_heatmaps":
                wandb_media_dict.update(
                    {
                        "velocity_heatmaps":
                            {
                                self.set_identifier:
                                    wandb.Html(file_html(
                                        logging_dict["velocity_heatmaps"], CDN, "vel_heatmap_{}_Epoch_{}".format(
                                            self.set_identifier, self.epoch)))
                            }
                    }
                )

            if global_features_html is True and key in "global_feature_pdfs":
                wandb_media_dict.update(
                    {
                        "global_feature_pdfs":
                            {
                                self.set_identifier:
                                    wandb.Html(file_html(
                                        logging_dict["global_feature_pdfs"], CDN,
                                        "feature_pdfs_" + self.set_identifier))
                            }
                    }
                )

            if audio_files is True and key in "captions_audios":
                captions_audios_tuples = logging_dict["captions_audios"]
                wandb_media_dict.update(
                    {
                        "audios":
                            {
                                self.set_identifier + '_plus_inputs':
                                    [
                                        wandb.Audio(c_a[1], caption=c_a[0], sample_rate=44100)
                                        for c_a in captions_audios_tuples
                                    ]
                            }
                    }
                )

            if piano_roll_html is True and key in "piano_rolls":
                wandb_media_dict.update(
                    {
                        "piano_roll_html":
                            {
                                self.set_identifier:
                                    wandb.Html(file_html(
                                        logging_dict["piano_rolls"], CDN, "piano_rolls_{}_Epoch_{}".format(
                                            self.set_identifier, self.epoch))),

                                self.set_identifier + '_plus_inputs':
                                    wandb.Html(file_html(
                                        logging_dict["piano_rolls_plus_inputs"], CDN,
                                        "piano_rolls_plus_inputs_{}_{}".format(self.set_identifier, self.epoch)))
                            }
                    }

                )
        print(self.epoch)
        return wandb_media_dict


# training script evaluator-related code wrappers

def init_evaluator(evaluator_path, device):
    with open(evaluator_path, 'rb') as f:
        evaluator = pickle.load(f)

    evaluator.device = device
    evaluator.processed_inputs.to(device)
    evaluator.processed_gt.to(device)

    return evaluator


def log_eval(evaluator, model, log_media, epoch, dump):
    evaluator.set_pred(model)

    evaluator.gt_SubSet_Evaluator.epoch = epoch
    evaluator.prediction_SubSet_Evaluator.epoch = epoch

    acc_h = evaluator.get_hits_accuracies(drum_mapping=ROLAND_REDUCED_MAPPING)
    mse_v = evaluator.get_velocity_errors(drum_mapping=ROLAND_REDUCED_MAPPING)
    mse_o = evaluator.get_micro_timing_errors(drum_mapping=ROLAND_REDUCED_MAPPING)
    wandb.log({**acc_h, **mse_v, **mse_o}, commit=False)

    if log_media:
        wandb_media = evaluator.get_wandb_logging_media(global_features_html=False, recalculate_ground_truth=False)
        if len(wandb_media.keys()) > 0:
            wandb.log({evaluator._identifier: wandb_media}, commit=False)

        # log stats
        csv_filename = os.path.join(wandb.run.dir, "stats_{}_Epoch_{}.csv".format(wandb.run.id, epoch))
        # csv_filename="stats/stats_{}_Epoch_{}.csv".format(wandb.run.id, epoch)
        df = get_stats_from_evaluator(evaluator, csv_file=csv_filename)
        df = df.drop(columns=['Statistical::Lowness__Ground_Truth', # drop columns that are not relevant for infilling
                              'Statistical::Lowness__Prediction',
                              'Statistical::Midness__Ground_Truth',
                              'Statistical::Midness__Prediction',
                              'Statistical::Hiness__Ground_Truth',
                              'Statistical::Hiness__Prediction',
                              'Statistical::Poly Velocity Mean__Ground_Truth',
                              'Statistical::Poly Velocity Mean__Prediction',
                              'Statistical::Poly Velocity std__Ground_Truth',
                              'Statistical::Poly Velocity std__Prediction',
                              'Statistical::Poly Offset Mean__Ground_Truth',
                              'Statistical::Poly Offset Mean__Prediction',
                              'Statistical::Poly Offset std__Ground_Truth',
                              'Statistical::Poly Offset std__Prediction',
                              'Syncopation::Combined__Ground_Truth',
                              'Syncopation::Combined__Prediction',
                              'Syncopation::Polyphonic__Ground_Truth',
                              'Syncopation::Polyphonic__Prediction',
                              'Syncopation::Lowsync__Ground_Truth',
                              'Syncopation::Lowsync__Prediction',
                              'Syncopation::Midsync__Ground_Truth',
                              'Syncopation::Midsync__Prediction',
                              'Syncopation::Hisync__Ground_Truth',
                              'Syncopation::Hisync__Prediction',
                              'Syncopation::Lowsyness__Ground_Truth',
                              'Syncopation::Lowsyness__Prediction',
                              'Syncopation::Midsyness__Ground_Truth',
                              'Syncopation::Midsyness__Prediction',
                              'Syncopation::Hisyness__Ground_Truth',
                              'Syncopation::Hisyness__Prediction',
                              'Syncopation::Complexity__Ground_Truth',
                              'Syncopation::Complexity__Prediction',
                              'Micro-Timing::Swingness__Ground_Truth',
                              'Micro-Timing::Swingness__Prediction',
                              'Micro-Timing::Laidbackness__Ground_Truth',
                              'Micro-Timing::Laidbackness__Prediction',
                              ])
        df = df.dropna(axis=1)  # remove nans
        html = df.to_html()
        wandb.save(csv_filename, base_path=wandb.run.dir)
        wandb.log({evaluator._identifier + '_stats': wandb.Html(html)}, commit=False)

    # move torch tensors to cpu before saving so that they can be loaded in cpu machines
    if dump:
        evaluator.processed_inputs.to(device='cpu')
        evaluator.processed_gt.to(device='cpu')

        # save_filename = os.path.join(wandb.run.dir, "evaluator/evaluator_{}_run_{}_Epoch_{}.Eval".format(
        #    evaluator._identifier, wandb.run.name,epoch))
        evaluator.dump(
            "evaluator/evaluator_{}_run_{}_Epoch_{}.Eval".format(evaluator._identifier, wandb.run.name, epoch))
        # wandb.save(save_filename, base_path=os.path.join(wandb.run.dir,'evaluator'))

    # rhythmic_distances = evaluator.get_rhythmic_distances()
    # wandb.log(rhythmic_distances, commit=False)
