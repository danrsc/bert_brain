import os
from collections import OrderedDict
from dataclasses import dataclass
from typing import Mapping, Sequence, Optional, Union
from functools import partial
import numpy as np

from scipy.io import loadmat
from scipy.ndimage.filters import gaussian_filter1d

import nibabel
import cortex

from .corpus_base import CorpusBase, CorpusExampleUnifier
from .input_features import RawData, KindData, ResponseKind


__all__ = ['HarryPotterCorpus', 'read_harry_potter_story_features', 'harry_potter_split_by_fmri_run',
           'harry_potter_fmri_make_split_function']


@dataclass
class _HarryPotterWordFMRI:
    word: str
    index_in_sentence: int
    image: int
    run: int
    story_features: Mapping


class HarryPotterCorpus(CorpusBase):

    def __init__(
            self,
            path: str,
            include_meg: bool = True,
            use_pca_meg: bool = True,
            separate_meg_axes: Union[str, Sequence[str]] = None,
            fmri_subjects: Optional[Sequence[str]] = None,
            fmri_smooth_factor: float = 1.):
        self.path = path
        self.fmri_subjects = fmri_subjects
        self.include_meg = include_meg
        self.use_pca_meg = use_pca_meg
        self.separate_meg_axes = separate_meg_axes
        self.fmri_smooth_factor = fmri_smooth_factor

    def _load(self, example_manager: CorpusExampleUnifier):
        data = OrderedDict()
        metadata = None
        if self.include_meg:
            meg = self._read_meg(example_manager)
            for k in meg:
                data[k] = KindData(ResponseKind.hp_meg, meg[k])
        if self.fmri_subjects is None or len(self.fmri_subjects) > 0:
            fmri, image_runs = self._read_fmri(example_manager)
            metadata = dict(response_runs=image_runs)
            for k in fmri:
                data[k] = KindData(ResponseKind.hp_fmri, fmri[k])

        for k in data:
            data[k].data.setflags(write=False)

        return RawData(input_examples=[example_manager.iterate_examples()], response_data=data, metadata=metadata)

    def _read_meg(self, example_manager: CorpusExampleUnifier):

        # separate_task_axes should be a tuple of strings in 'roi', 'subject', 'time'
        separate_task_axes = self.separate_meg_axes
        if separate_task_axes is None:
            separate_task_axes = []
        elif isinstance(separate_task_axes, str):
            separate_task_axes = [separate_task_axes]
        for axis in separate_task_axes:
            if axis not in ['roi', 'subject', 'time']:
                raise ValueError('Unknown separate_task_axis: {}'.format(axis))

        # see make_harry_potter.ipynb for how these are constructed
        meg_path = os.path.join(
            self.path, 'harry_potter_meg_100ms.npz' if not self.use_pca_meg else 'harry_potter_meg_100ms_pca.npz')

        loaded = np.load(meg_path)

        stimuli = loaded['stimuli']
        # blocks should be int, but is stored as float
        blocks = loaded['blocks']
        blocks = np.round(blocks).astype(np.int64)
        # (subjects, words, rois, 100ms_slices)
        data = loaded['data']
        rois = loaded['rois']
        subjects = loaded['subjects']
        times = np.arange(data.shape[-1]) * 100

        # -> (words, subjects, rois, 100ms_slices)
        data = np.transpose(data, axes=(1, 0, 2, 3))

        assert (len(rois) == data.shape[2])

        # mark examples as between pluses
        indicator_plus = stimuli == '+'

        # I used to make longer passages that went between pluses. Leaving the code here
        # as a comment in case we want to use that again in the future
        # if examples_between_plus:
        #     indicator_block = np.concatenate([np.full(1, True), np.diff(blocks) > 0])
        #     example_id_words = np.zeros(len(stimuli), dtype=np.int64)
        #     indices_plus = np.where(np.logical_or(indicator_plus, indicator_block))[0]
        #     if indices_plus[0] == 0:
        #         indices_plus = indices_plus[1:]
        #     last = 0
        #     current_passage = 0
        #     for i in indices_plus:
        #         example_id_words[last:i] = current_passage
        #         last = i
        #         current_passage += 1
        #     example_id_words[last:] = current_passage
        # else:  # use sentences as examples

        story_features = read_harry_potter_story_features(os.path.join(self.path, 'story_features.mat'))
        indices_in_sentences = story_features[('Word_Num', 'sentence_length')]
        assert (len(indices_in_sentences) == len(data))
        example_id_words = list()
        for id_example, group in enumerate(_group_sentence_indices(indices_in_sentences)):
            example_id_words.extend([id_example] * len(group))
        example_id_words = np.array(example_id_words)
        assert (len(example_id_words) == len(data))

        not_plus = np.logical_not(indicator_plus)
        data = data[not_plus]
        stimuli = stimuli[not_plus]
        blocks = blocks[not_plus]
        example_id_words = example_id_words[not_plus]

        # data is (words, subjects, rois, 100ms_slices)
        data = OrderedDict([('MEG', data)])

        subject_axis = 1
        roi_axis = 2
        time_axis = 3
        if 'subject' in separate_task_axes:
            separated_data = OrderedDict()
            for k in data:
                assert (len(subjects) == data[k].shape[subject_axis])
                for idx_subject, subject in enumerate(subjects):
                    k_new = k + '.{}'.format(subject) if k != 'MEG' else subject
                    separated_data[k_new] = np.take(data[k], idx_subject, axis=subject_axis)
            data = separated_data
            roi_axis -= 1
            time_axis -= 1
        if 'roi' in separate_task_axes:
            separated_data = OrderedDict()
            for k in data:
                assert (len(rois) == data[k].shape[roi_axis])
                for idx_roi, roi in enumerate(rois):
                    k_new = k + '.{}'.format(roi) if k != 'MEG' else roi
                    separated_data[k_new] = np.take(data[k], idx_roi, axis=roi_axis)
            data = separated_data
            time_axis -= 1
        if 'time' in separate_task_axes:
            separated_data = OrderedDict()
            for k in data:
                assert (len(times) == data[k].shape[time_axis])
                for idx_time, window in enumerate(times):
                    k_new = k + '.{}'.format(window) if k != 'MEG' else '{}'.format(window)
                    separated_data[k_new] = np.take(data[k], idx_time, axis=time_axis)
            data = separated_data

        offset = 0
        for example_id in np.unique(example_id_words):
            indicator_example = example_id_words == example_id
            example_stimuli = stimuli[indicator_example]
            # we don't currently use blocks for anything, but we could use it as an input to the model
            # example_blocks = blocks[indicator_example]
            example_manager.add_example(example_stimuli, [k for k in data], np.arange(len(example_stimuli)) + offset)

        return data

    def _read_harry_potter_fmri_files(self):
        # noinspection PyPep8
        subject_runs = dict(
            F=[4, 5, 6, 7],
            G=[3, 4, 5, 6],
            H=[3, 4, 9, 10],
            I=[7, 8, 9, 10],
            J=[7, 8, 9, 10],
            K=[7, 8, 9, 10],
            L=[7, 8, 9, 10],
            M=[7, 8, 9, 10],
            N=[7, 8, 9, 10])

        subjects = self.fmri_subjects
        if subjects is None:
            subjects = list(subject_runs.keys())

        if isinstance(subjects, str):
            subjects = [subjects]

        path_fmt = os.path.join(self.path, 'fmri', '{subject}', 'funct', '{run}', 'ars{run:03}a001.hdr')

        all_subject_data = OrderedDict()
        masks = OrderedDict()

        for subject in subjects:
            if subject not in subject_runs:
                raise ValueError('Unknown subject: {}. Known values are: {}'.format(subject, list(subject_runs.keys())))
            subject_data = list()
            for run in subject_runs[subject]:
                functional_file = path_fmt.format(subject=subject, run=run)
                data = nibabel.load(functional_file).get_data()
                subject_data.append(np.transpose(data))

            masks[subject] = cortex.db.get_mask('fMRI_story_{}'.format(subject), '{}_ars'.format(subject), 'thick')
            all_subject_data[subject] = subject_data

        return all_subject_data, masks

    def _harry_potter_fmri_word_info(self, run_lengths):

        time_images = np.arange(1351) * 2
        words = np.load(os.path.join(self.path, 'words_fmri.npy'))
        time_words = np.load(os.path.join(self.path, 'time_words_fmri.npy'))
        assert (len(words) == len(time_words))

        # searchsorted returns first location, i, such that time_word < time_images[i]
        word_images = np.searchsorted(time_images, time_words, side='right') - 1

        story_features = read_harry_potter_story_features(os.path.join(self.path, 'story_features.mat'))
        indices_in_sentences = story_features[('Word_Num', 'sentence_length')]
        assert (len(indices_in_sentences) == len(words))

        run_ids = np.concatenate([[idx] * run_lengths[idx] for idx in range(len(run_lengths))])
        assert (len(run_ids) == len(time_images))

        words = [
            _HarryPotterWordFMRI(
                _clean_word(words[i]), indices_in_sentences[i], word_images[i], run_ids[word_images[i]],
                dict((k, story_features[k][i]) for k in story_features)) for i in range(len(words))]

        sentences = list()
        for sentence in _group_sentences(words):
            has_plus = False
            for idx_w, word in enumerate(sentence):
                if word.word == '+':
                    assert (idx_w == 0)  # assert no natural pluses
                    has_plus = True
            if has_plus:
                sentence = sentence[1:]
            if len(sentence) > 0:
                sentences.append(sentence)

        return sentences

    def _read_fmri(self, example_manager: CorpusExampleUnifier):

        data, spatial_masks = self._read_harry_potter_fmri_files()

        # we assume that the runs are the same across subjects below. assert it here
        run_lengths = None
        for subject in data:
            if run_lengths is None:
                run_lengths = [len(r) for r in data[subject]]
            else:
                assert (np.array_equal([len(r) for r in data[subject]], run_lengths))

        # get the words, image indices, and story features per sentence
        sentences = self._harry_potter_fmri_word_info(run_lengths)

        # split up the sentences by run
        run_to_sentence = OrderedDict()  # use ordered dict to keep the runs in order
        for sentence_id, sentence in enumerate(sentences):
            assert (all(w.run == sentence[0].run for w in sentence))  # assert no sentence spans more than 1 run
            if sentence[0].run not in run_to_sentence:
                run_to_sentence[sentence[0].run] = [sentence_id]
            else:
                run_to_sentence[sentence[0].run].append(sentence_id)

        # remove the first run_start sentences and the last run_end sentences since these can both have strange data
        run_start = 5
        run_end = 3
        excluded = set()
        for run in run_to_sentence:
            sentence_ids = run_to_sentence[run]
            excluded.update(sentence_ids[:run_start + 1])
            excluded.update(sentence_ids[-run_end:])

        # filter unused images
        active_sentence_images_to_new_index = dict()
        image_runs = list()
        active_images = np.full(np.sum(run_lengths), False)
        for sentence_id, sentence in enumerate(sentences):
            if sentence_id in excluded:
                continue
            for w in sentence:
                # store the new index
                new_index = len(active_sentence_images_to_new_index)
                active_sentence_images_to_new_index[w.image] = new_index
                active_images[w.image] = True
                image_runs.append(w.run)

        masked_data = OrderedDict()
        for subject in data:
            subject_data = np.concatenate(data[subject])[active_images]
            if self.fmri_smooth_factor is not None:
                subject_data = gaussian_filter1d(
                    subject_data, sigma=self.fmri_smooth_factor, axis=1, order=0, mode='reflect', truncate=4.0)
            masked_data['hp_fmri_{}'.format(subject)] = subject_data[:, spatial_masks[subject]]
        data = masked_data

        for sentence_id, sentence in enumerate(sentences):
            if sentence_id in excluded:
                continue
            images = [active_sentence_images_to_new_index[w.image] for w in sentence]
            example_manager.add_example(
                [w.word for w in sentence], [k for k in data], images, is_apply_data_id_to_entire_group=True)

        return data, image_runs


def read_harry_potter_story_features(path):
    mat = loadmat(path)
    features = np.squeeze(mat['features'], axis=0)
    result = dict()
    for index_feature_type, feature_type in enumerate(features['type']):
        feature_type = np.squeeze(feature_type, axis=0)
        if feature_type.dtype.type != np.str_:
            assert(len(feature_type) == 0)
            feature_type = None
        else:
            feature_type = str(feature_type)

        if feature_type is None:
            continue  # there are some empty ones which also have no values; not sure why

        feature_names = np.squeeze(features['names'][index_feature_type], axis=0)
        if feature_names.dtype.type == np.object_:
            feature_names = [str(np.squeeze(s, axis=0)) for s in feature_names]
        elif feature_names.dtype.type != np.str_:
            assert(len(feature_names) == 0)
            feature_names = None
        else:
            feature_names = [str(feature_names)]

        feature_values = features['values'][index_feature_type]
        if feature_names is not None:
            if len(feature_names) != feature_values.shape[1]:
                if feature_names[0] == 'word_length':
                    feature_names = feature_names[:-1]
            assert(len(feature_names) == feature_values.shape[1])
            feature_values = [np.squeeze(x, axis=1) for x in np.split(feature_values, len(feature_names), axis=1)]
            for feature_name, feature_value in zip(feature_names, feature_values):
                result[(feature_type, feature_name)] = feature_value
        else:
            result[feature_type] = feature_values
    return result


def _group_sentences(words):
    current = list()
    for word in words:
        if len(current) > 0:
            if word.index_in_sentence <= current[-1].index_in_sentence:
                yield current
                current = list()
        current.append(word)
    if len(current) > 0:
        yield current


def _group_sentence_indices(indices_in_sentences):
    current = list()
    for index_in_sentence in indices_in_sentences:
        if len(current) > 0:
            if index_in_sentence <= current[-1]:
                yield current
                current = list()
        current.append(index_in_sentence)
    if len(current) > 0:
        yield current


def _clean_word(w):
    return str(w).replace('@', '').replace('\\', '')


def harry_potter_fmri_make_split_function(index_variation_run):
    return partial(harry_potter_split_by_fmri_run, index_variation_run=index_variation_run)


def harry_potter_split_by_fmri_run(raw_data, index_variation_run, random_state=None, shuffle=True):
    runs = raw_data.metadata['response_runs']
    unique_runs = np.unique(runs)
    index_validation = index_variation_run % len(unique_runs)
    validation_run = unique_runs[index_validation]
    train_examples = list()
    validation_examples = list()
    for example in raw_data.input_examples:
        run = None
        for data_id in example.data_ids:
            if data_id >= 0:
                run = runs[data_id]
                break
        assert(run is not None)
        if run == validation_run:
            validation_examples.append(example)
        else:
            train_examples.append(example)
    if shuffle:
        if random_state is not None:
            random_state.shuffle(train_examples)
            random_state.shuffle(validation_examples)
        else:
            np.random.shuffle(train_examples)
            np.random.shuffle(validation_examples)
    test_examples = None
    return train_examples, validation_examples, test_examples
