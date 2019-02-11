import numpy as np

from bert_erp_tokenization import bert_tokenize_with_spacy_meta, RawData
from .data_preparer import split_data


__all__ = ['read_harry_potter', 'harry_potter_data']


def read_harry_potter(spacy_tokenize_model, bert_tokenizer, path, separate_task_axes=None):

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


def harry_potter_data(
        spacy_tokenize_model,
        bert_tokenizer,
        path,
        separate_task_axes=None):

    examples, data = read_harry_potter(
        spacy_tokenize_model, bert_tokenizer, path, separate_task_axes=separate_task_axes)
    examples_train, examples_validation, examples_test = split_data(examples, 0., 0.1, shuffle=False)

    return RawData(
        examples_train,
        data,
        validation_input_examples=examples_validation,
        test_input_examples=examples_test, is_pre_split=True)
