import sys
import numpy as np
from tqdm import tqdm
import wandb


sys.path.insert(1, "../../GrooveEvaluator")
from GrooveEvaluator.evaluator import Evaluator, HVOSeq_SubSet_Evaluator
sys.path.insert(1, "../preprocessed_dataset/")
from Subset_Creators import subsetters
from utils import get_hvo_idx_for_voice
from bokeh.embed import file_html
from bokeh.resources import CDN


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
        self._gt_hvo_sequences = []
        self._gt_hvos_array_tags, self._gt_hvos_array, self._prediction_hvo_seq_templates = [], [], []
        for subset_ix, tag in enumerate(self._gt_tags):
            for sample_ix, sample_hvo in enumerate(self._gt_subsets[subset_ix]):
                self._gt_hvo_sequences.append(sample_hvo)
                self._gt_hvos_array_tags.append(tag)
                self._gt_hvos_array.append(sample_hvo.get("hvo"))
                self._prediction_hvo_seq_templates.append(sample_hvo.copy_empty())

        self._gt_hvos_array = np.stack(self._gt_hvos_array)

        self.gt_SubSet_Evaluator = HVOSeq_SubSet_InfillingEvaluator(
            self._gt_subsets,              # Ground Truth typically
            self._gt_tags,
            "{}_Set_Ground_Truth".format(self._identifier),             # a name for the subset
            disable_tqdm=self.disable_tqdm,
            group_by_minor_keys=True)

        self.audio_sample_locations = self.get_sample_indices(n_samples_to_synthesize_visualize_per_subset)


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
                                            np.arange(start=first_epochs_lim, step=remaining_epochs_step_all,
                                                      stop=self.eps))


    def get_wandb_logging_media(self, velocity_heatmap_html=True, global_features_html=True,
                         piano_roll_html=True, audio_files=True,
                         use_sf_dict = False, sf_paths=[
                "../hvo_sequence/hvo_sequence/soundfonts/Standard_Drum_Kit.sf2"], recalculate_ground_truth=True):

        sf_dict = {}
        if use_sf_dict:
            for key in self.audio_sample_locations.keys():
                sf_dict[key] = []
                for idx in self.audio_sample_locations[key]:
                    # FIXME
                    sf_dict[key].append(self.eval_soundfonts[self._subset_hvo_array_index[key][idx]])


        # Get logging data for ground truth data
        if recalculate_ground_truth is True or self._gt_logged_once_wandb is False:


            gt_logging_media = self.gt_SubSet_Evaluator.get_wandb_logging_media(
                velocity_heatmap_html=velocity_heatmap_html,
                global_features_html=global_features_html,
                piano_roll_html=piano_roll_html,
                audio_files=audio_files,
                sf_paths=sf_paths,
                sf_dict=sf_dict,
                use_specific_samples_at=self.audio_sample_locations
            )
            self._gt_logged_once_wandb = True
        else:
            gt_logging_media = {}

        predicted_logging_media = self.prediction_SubSet_Evaluator.get_wandb_logging_media(
            velocity_heatmap_html=velocity_heatmap_html,
            global_features_html=global_features_html,
            piano_roll_html=piano_roll_html,
            audio_files=audio_files,
            sf_paths=sf_paths,
            sf_dict=sf_dict,
            use_specific_samples_at=self.audio_sample_locations
        ) if self.prediction_SubSet_Evaluator is not None else {}

        results = {x: {} for x in gt_logging_media.keys()}
        results.update({x: {} for x in predicted_logging_media.keys()})

        for key in results.keys():
            if key in gt_logging_media.keys():
                results[key].update(gt_logging_media[key])
            if key in predicted_logging_media.keys():
                results[key].update(predicted_logging_media[key])

        return results

    def add_predictions(self, prediction_hvos_array):
        self._prediction_hvos_array = prediction_hvos_array
        self._prediction_tags, self._prediction_subsets, self._subset_hvo_array_index = \
            subsetters.convert_hvos_array_to_subsets(
                self._gt_hvos_array_tags,
                prediction_hvos_array,
                self._prediction_hvo_seq_templates
            )

        self.prediction_SubSet_Evaluator = HVOSeq_SubSet_InfillingEvaluator(
            self._prediction_subsets,
            self._prediction_tags,
            "{}_Set_Predictions".format(self._identifier),             # a name for the subset
            disable_tqdm=self.disable_tqdm,
            group_by_minor_keys=True)

    def set_gt(self):
        # get gt evaluator
        evaluator_subset = self.get_ground_truth_hvo_sequences()

        # TODO outputs as dict ?
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

        self.add_predictions(eval_pred)

        # TODO update get_audios

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
            analyze_global_features=True
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



    def get_logging_dict(self, velocity_heatmap_html=True, global_features_html=True,
                         piano_roll_html=True, audio_files=True, sf_paths=None,
                         sf_dict=False, use_specific_samples_at=None):
        if audio_files is True:
            assert sf_paths is not None, "Provide sound_file path(s) for synthesizing samples"

        logging_dict = {}
        if velocity_heatmap_html is True:
            logging_dict.update({"velocity_heatmaps": self.get_vel_heatmap_bokeh_figures()})
        if global_features_html is True:
            logging_dict.update({"global_feature_pdfs": self.get_global_features_bokeh_figure()})
        if audio_files is True:
            captions_audios_tuples = self.get_audios(sf_paths=sf_paths, sf_dict=sf_dict,
                                                     use_specific_samples_at=use_specific_samples_at)
            captions_audios = [(c_a[0], c_a[1]) for c_a in captions_audios_tuples]

            logging_dict.update({"captions_audios": captions_audios})
        if piano_roll_html is True:
            logging_dict.update({"piano_rolls": self.get_piano_rolls(use_specific_samples_at)})

        return logging_dict

    def get_wandb_logging_media(self, velocity_heatmap_html=True, global_features_html=True,
                                piano_roll_html=True, audio_files=True, sf_paths=None,
                                sf_dict=False, use_specific_samples_at=None):

        logging_dict = self.get_logging_dict(velocity_heatmap_html, global_features_html,
                                             piano_roll_html, audio_files, sf_paths, sf_dict,
                                             use_specific_samples_at)

        wandb_media_dict = {}
        for key in logging_dict.keys():
            if velocity_heatmap_html is True and key in "velocity_heatmaps":
                wandb_media_dict.update(
                    {
                        "velocity_heatmaps":
                            {
                            self.set_identifier:
                                wandb.Html(file_html(
                                    logging_dict["velocity_heatmaps"], CDN, "vel_heatmap_"+self.set_identifier))
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
                                        logging_dict["global_feature_pdfs"], CDN, "feature_pdfs_"+self.set_identifier))
                            }
                    }
                )

            if audio_files is True and key in "captions_audios":
                captions_audios_tuples = logging_dict["captions_audios"]
                wandb_media_dict.update(
                    {
                        "audios":
                            {
                                self.set_identifier:
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
                                        logging_dict["piano_rolls"], CDN, "piano_rolls_"+self.set_identifier))
                            }
                    }
                )

        return wandb_media_dict


    def get_audios(self, sf_paths=None, sf_dict=None, use_specific_samples_at=None):
        """ use_specific_samples_at: must be a list of tuples of (subset_ix, sample_ix) denoting to get
        audio from the sample_ix in subset_ix """

        self._sampled_hvos = self.get_hvo_samples_located_at(use_specific_samples_at)

        if not None and not isinstance(sf_paths, list):
            sf_paths = [sf_paths]

        audios = []
        captions = []

        for key in tqdm(self._sampled_hvos.keys(),
                        desc='Synthesizing samples - {} '.format(self.set_identifier),
                        disable=self.disable_tqdm):
            for idx, sample_hvo in enumerate(self._sampled_hvos[key]):
                if sf_dict:
                    sf_path = sf_dict[key][idx]
                else:
                    # randomly select a sound font
                    sf_path = sf_paths[np.random.randint(0, len(sf_paths))]
                audios.append(sample_hvo.synthesize(sf_path=sf_path))
                captions.append("{}_{}_{}.wav".format(
                    self.set_identifier, sample_hvo.metadata.style_primary, sample_hvo.metadata.master_id.replace("/", "_")
                ))

        return list(zip(captions, audios))
