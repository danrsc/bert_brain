import os
from dataclasses import dataclass
import numpy as np
from scipy.io import loadmat

from .input_features import RawData, KindData, ResponseKind
from .corpus_base import CorpusBase, CorpusExampleUnifier, path_attribute_field


__all__ = ['DundeeCorpus']


@dataclass(frozen=True)
class DundeeCorpus(CorpusBase):
    path: str = path_attribute_field('dundee_path')

    def _load(self, example_manager: CorpusExampleUnifier) -> RawData:
        data = loadmat(os.path.join(self.path, 'data.mat'))
        line_pos = data['line_pos'].squeeze(axis=1)
        sent_pos = data['sent_pos'].squeeze(axis=1)
        reading_time_first_pass = data['RTfpass']
        reading_time_go_past = data['RTgopast']
        reading_time_right_bounded = data['RTrb']
        words = data['objects'].squeeze(axis=1)

        words = [str(w[0]) for w in words]
        # fix up some encoding issues. These may not be exhaustive. Just the ones I've noticed
        for idx, word in enumerate(words):
            if word.startswith('egalit‚'):
                words[idx] = word.replace('egalit‚', 'égalité')
            elif word.startswith('libert‚'):
                words[idx] = word.replace('libert‚', 'liberté')
            elif word.startswith('fraternit‚'):
                words[idx] = word.replace('fraternit‚', 'fraternité')
            elif word.startswith('‚litisme'):
                words[idx] = word.replace('‚litisme', 'élitisme')
            elif word.startswith('na‹ve'):
                words[idx] = word.replace('na‹ve', 'naive')

        words = np.array(words)

        # we can use the tagged texts to figure out which part the words belong to. Unfortunately, that is not in
        # the raw data
        tagged_files = list()
        for current_directory, sub_directories, files in os.walk(self.path):
            for current_file in files:
                if current_file.endswith('.pos.txt'):
                    tagged_files.append(os.path.join(current_directory, current_file))

        num_sentences_per_part = list()
        for tagged_file in tagged_files:
            num_sentences_per_part.append(0)
            with open(tagged_file, 'rt') as tagged:
                for sentence in tagged:
                    sentence = sentence.strip()
                    if len(sentence) > 0:
                        num_sentences_per_part[-1] += 1

        cum_sentences_per_part = np.cumsum(num_sentences_per_part)

        indices_new_sentence = np.where(np.diff(np.asarray(sent_pos, dtype=np.int32)) <= 0)[0] + 1
        last = 0
        sentence_id = 0
        part_id = 0
        sentence_ids = list()
        part_ids = list()
        for index_new_sentence in indices_new_sentence:
            sentence_ids.extend([sentence_id] * (index_new_sentence - last))
            part_ids.extend([part_id] * (index_new_sentence - last))
            sentence_id += 1
            if sentence_id == cum_sentences_per_part[part_id]:
                part_id += 1
            last = index_new_sentence
        sentence_ids.extend([sentence_id] * (len(sent_pos) - last))
        part_ids.extend([part_id] * (len(sent_pos) - last))

        sentence_ids = np.array(sentence_ids)
        part_ids = np.array(part_ids)

        examples = list()

        for sentence_id in np.unique(sentence_ids):

            sentence_words = words[sentence_ids == sentence_id]
            data_indices = np.arange(len(sentence_ids))[sentence_ids == sentence_id]

            examples.append(example_manager.add_example(
                sentence_id,
                sentence_words,
                [sentence_id] * len(sentence_words),
                ['dun_fst_pst', 'dun_go_pst', 'dun_rt_bnd'],
                data_indices))

        def _readonly(arr: np.ndarray):
            arr.setflags(write=False)
            return arr

        return RawData(
            examples,
            response_data={
                'dun_fst_pst': KindData(ResponseKind.dundee_eye, _readonly(reading_time_first_pass)),
                'dun_go_pst': KindData(ResponseKind.dundee_eye, _readonly(reading_time_go_past)),
                'dun_rt_bnd': KindData(ResponseKind.dundee_eye, _readonly(reading_time_right_bounded))},
            validation_proportion_of_train=0.25,
            metadata={
                'line_pos': _readonly(line_pos),
                'sent_pos': _readonly(sent_pos),
                'part_id': _readonly(part_ids)})

    def _run_info(self, index_run):
        # use 4-fold CV
        return index_run % 4
