import os
from collections import OrderedDict
from dataclasses import dataclass
from typing import Mapping
import numpy as np

from scipy.io import loadmat
from scipy.ndimage.filters import gaussian_filter1d

import nibabel
import cortex

from bert_erp_tokenization import bert_tokenize_with_spacy_meta, RawData
from .data_preparer import split_data


__all__ = ['read_harry_potter_story_features', 'read_harry_potter_meg', 'harry_potter_meg_data',
           'harry_potter_fmri_data', 'harry_potter_split_by_run']


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


def read_harry_potter_fmri(path, subjects=None):
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

    if subjects is None:
        subjects = list(subject_runs.keys())

    if isinstance(subjects, str):
        subjects = [subjects]

    path_fmt = os.path.join(path, 'fmri', '{subject}', 'funct', '{run}', 'ars{run:03}a001.hdr')

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


def read_harry_potter_meg(spacy_tokenize_model, bert_tokenizer, path, separate_task_axes=None):

    # separate_task_axes should be a tuple of strings in 'roi', 'subject', 'time'
    if separate_task_axes is None:
        separate_task_axes = []
    elif isinstance(separate_task_axes, str):
        separate_task_axes = [separate_task_axes]
    for axis in separate_task_axes:
        if axis not in ['roi', 'subject', 'time']:
            raise ValueError('Unknown separate_task_axis: {}'.format(axis))

    loaded = np.load(path)

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

    assert(len(rois) == data.shape[2])

    # mark passages as between pluses
    indicator_plus = stimuli == '+'
    indicator_block = np.concatenate([np.full(1, True), np.diff(blocks) > 0])
    passage_id_words = np.zeros(len(stimuli), dtype=np.int64)
    indices_plus = np.where(np.logical_or(indicator_plus, indicator_block))[0]
    if indices_plus[0] == 0:
        indices_plus = indices_plus[1:]
    last = 0
    current_passage = 0
    for i in indices_plus:
        passage_id_words[last:i] = current_passage
        last = i
        current_passage += 1
    passage_id_words[last:] = current_passage

    not_plus = np.logical_not(indicator_plus)
    data = data[not_plus]
    stimuli = stimuli[not_plus]
    blocks = blocks[not_plus]
    passage_id_words = passage_id_words[not_plus]

    offset = 0
    examples = list()
    for passage_id in np.unique(passage_id_words):
        indicator_passage = passage_id_words == passage_id
        passage_stimuli = stimuli[indicator_passage]
        # we don't currently use blocks for anything, but we could use it as an input to the model
        # passage_blocks = blocks[indicator_passage]
        examples.append(bert_tokenize_with_spacy_meta(
            spacy_tokenize_model, bert_tokenizer, passage_id, passage_stimuli, offset))

    # data is (words, subjects, rois, 100ms_slices)
    data = {'MEG': data}

    subject_axis = 1
    roi_axis = 2
    time_axis = 3
    if 'subject' in separate_task_axes:
        separated_data = dict()
        for k in data:
            assert(len(subjects) == data[k].shape[subject_axis])
            for idx_subject, subject in enumerate(subjects):
                k_new = k + '.{}'.format(subject) if k != 'MEG' else subject
                separated_data[k_new] = np.take(data[k], idx_subject, axis=subject_axis)
        data = separated_data
        roi_axis -= 1
        time_axis -= 1
    if 'roi' in separate_task_axes:
        separated_data = dict()
        for k in data:
            assert(len(rois) == data[k].shape[roi_axis])
            for idx_roi, roi in enumerate(rois):
                k_new = k + '.{}'.format(roi) if k != 'MEG' else roi
                separated_data[k_new] = np.take(data[k], idx_roi, axis=roi_axis)
        data = separated_data
        time_axis -= 1
    if 'time' in separate_task_axes:
        separated_data = dict()
        for k in data:
            assert (len(times) == data[k].shape[time_axis])
            for idx_time, window in enumerate(times):
                k_new = k + '.{}'.format(window) if k != 'MEG' else '{}'.format(window)
                separated_data[k_new] = np.take(data[k], idx_time, axis=time_axis)
        data = separated_data

    return examples, data


def harry_potter_meg_data(
        spacy_tokenize_model,
        bert_tokenizer,
        path,
        separate_task_axes=None,
        use_pca=False):

    meg_path = os.path.join(path, 'harry_potter_meg_100ms.npz' if not use_pca else 'harry_potter_meg_100ms_pca.npz')

    examples, data = read_harry_potter_meg(
        spacy_tokenize_model, bert_tokenizer, meg_path, separate_task_axes=separate_task_axes)
    examples_train, examples_validation, examples_test = split_data(examples, 0., 0.1, shuffle=False)

    return RawData(
        examples_train,
        data,
        validation_input_examples=examples_validation,
        test_input_examples=examples_test,
        is_pre_split=True)


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


def _clean_word(w):
    return str(w).replace('@', '').replace('\\', '')


@dataclass
class HarryPotterWordFMRI:
    word: str
    index_in_sentence: int
    image: int
    run: int
    story_features: Mapping


def _harry_potter_fmri_word_info(path, run_lengths):

    time_images = np.arange(1351) * 2
    words = np.load(os.path.join(path, 'words_fmri.npy'))
    time_words = np.load(os.path.join(path, 'time_words_fmri.npy'))
    assert (len(words) == len(time_words))

    # searchsorted returns first location, i, such that time_word < time_images[i]
    word_images = np.searchsorted(time_images, time_words, side='right') - 1

    story_features = read_harry_potter_story_features(os.path.join(path, 'story_features.mat'))
    indices_in_sentences = story_features[('Word_Num', 'sentence_length')]
    assert (len(indices_in_sentences) == len(words))

    run_ids = np.concatenate([[idx] * run_lengths[idx] for idx in range(len(run_lengths))])
    assert (len(run_ids) == len(time_images))

    words = [
        HarryPotterWordFMRI(
            _clean_word(words[i]), indices_in_sentences[i], word_images[i], run_ids[word_images[i]],
            dict((k, story_features[k][i]) for k in story_features)) for i in range(len(words))]

    sentences = list()
    for sentence in _group_sentences(words):
        has_plus = False
        for idx_w, word in enumerate(sentence):
            if word.word == '+':
                assert(idx_w == 0)  # assert no natural pluses
                has_plus = True
        if has_plus:
            sentence = sentence[1:]
        if len(sentence) > 0:
            sentences.append(sentence)

    return sentences


def harry_potter_fmri_data(spacy_tokenize_model, bert_tokenizer, path, subjects=None, smooth_factor=1):

    data, spatial_masks = read_harry_potter_fmri(path, subjects)

    # we assume that the runs are the same across subjects below. assert it here
    run_lengths = None
    for subject in data:
        if run_lengths is None:
            run_lengths = [len(r) for r in data[subject]]
        else:
            assert(np.array_equal([len(r) for r in data[subject]], run_lengths))

    # get the words, image indices, and story features per sentence
    sentences = _harry_potter_fmri_word_info(path, run_lengths)

    # split up the sentences by run
    run_to_sentence = OrderedDict()  # use ordered dict to keep the runs in order
    for sentence_id, sentence in enumerate(sentences):
        assert(all(w.run == sentence[0].run for w in sentence))  # assert no sentence spans more than 1 run
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
        excluded.update(sentence_ids[:run_start+1])
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
        if smooth_factor is not None:
            subject_data = gaussian_filter1d(
                subject_data, sigma=smooth_factor, axis=1, order=0, mode='reflect', truncate=4.0)
        masked_data['hp_fmri_{}'.format(subject)] = subject_data[:, spatial_masks[subject]]
    data = masked_data

    examples = list()
    for sentence_id, sentence in enumerate(sentences):
        if sentence_id in excluded:
            continue
        images = [active_sentence_images_to_new_index[w.image] for w in sentence]
        examples.append(bert_tokenize_with_spacy_meta(
            spacy_tokenize_model, bert_tokenizer, sentence_id, [w.word for w in sentence], images))

    examples_train, examples_validation, examples_test = split_data(examples, 0., 0.1, shuffle=False)
    return RawData(
        input_examples=examples_train,
        response_data=data,
        test_input_examples=examples_test,
        validation_input_examples=examples_validation,
        is_pre_split=True,
        metadata=dict(response_runs=np.array(image_runs)))


def harry_potter_split_by_run(raw_data, random_state=None, shuffle=False):
    runs = raw_data.metadata['response_runs']
    unique_runs = np.unique(runs)
    if random_state is not None:
        index_validation = random_state.randint(len(unique_runs))
    else:
        index_validation = np.random.randint(len(unique_runs))
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

