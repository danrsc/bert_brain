from dataclasses import dataclass
from itertools import groupby
import csv

import numpy as np

from .input_features import RawData, KindData, ResponseKind
from .corpus_base import CorpusBase, CorpusExampleUnifier, path_attribute_field


__all__ = ['GhentEyeTrackingCorpus']


@dataclass(frozen=True)
class GhentEyeTrackingCorpus(CorpusBase):
    path: str = path_attribute_field('geco_path')

    def _load(self, example_manager: CorpusExampleUnifier) -> RawData:
        participants = dict()
        word_data = dict()
        id_to_word = dict()

        ignore_fields = {'GROUP', 'LANGUAGE_RANK', 'LANGUAGE', 'WORD_ID',
                         'WORD', 'PART', 'TRIAL', 'WORD_ID_WITHIN_TRIAL', 'PP_NR'}

        with open(self.path, newline='', encoding='utf-8-sig') as f:
            eye_tracking_reader = csv.DictReader(f)
            for index_row, row in enumerate(eye_tracking_reader):

                id_part = int(row['PART'])
                id_trial_in_part = int(row['TRIAL'])
                id_word_in_trial = int(row['WORD_ID_WITHIN_TRIAL'])

                id_word = (id_part, id_trial_in_part, id_word_in_trial)

                participant = row['PP_NR']
                if participant not in participants:
                    participants[participant] = len(participants)

                word = row['WORD']
                if word == 'TRUE':
                    word = 'true'
                if word == 'FALSE':
                    word = 'false'

                id_to_word[id_word] = word

                for key in row:
                    if key in ignore_fields:
                        continue
                    if id_word not in word_data:
                        word_data[id_word] = dict()
                    if key not in word_data[id_word]:
                        word_data[id_word][key] = dict()
                    word_data[id_word][key][participant] = row[key]

        participants = [p for p in sorted(participants, key=lambda p_: participants[p_])]

        def flatten(to_flatten):
            all_keys = set()

            for id_ in to_flatten:
                for k in to_flatten[id_]:
                    all_keys.add(k)

            all_keys = list(sorted(all_keys))
            ids = list(sorted(to_flatten))

            result = dict()
            for k in all_keys:
                values = list()
                is_int = True
                for id_ in ids:
                    values.append(list())
                    for p in participants:
                        if k in to_flatten[id_] and p in to_flatten[id_][k]:
                            raw = to_flatten[id_][k][p]
                        else:
                            raw = None
                        if raw == '.':
                            raw = None
                        if is_int:
                            if raw is None:
                                parsed = -1
                            else:
                                try:
                                    parsed = int(raw)
                                except ValueError:
                                    parsed = float(raw)
                        else:
                            if raw is None:
                                parsed = -1
                            else:
                                parsed = float(raw)
                        values[-1].append(parsed)
                result[k] = np.array(values)
                if result[k].dtype == 'float32':
                    result[k] = np.where(result[k] < 0, np.nan, result[k])
                result[k].setflags(write=False)

            return ids, result

        word_ids, flat_word_data = flatten(word_data)
        if 'WORD_SKIP' in flat_word_data:
            flat_word_data['WORD_SKIP'] = flat_word_data['WORD_SKIP'].astype(bool)

        offset = 0
        examples = list()
        for (part_id, trial_id), group_word_ids in groupby(word_ids, key=lambda wid: (wid[0], wid[1])):
            examples.append(example_manager.add_example(
                (part_id, trial_id),
                words=[id_to_word[w] for w in group_word_ids],
                sentence_ids=[0] * len(group_word_ids),
                data_key=[k.lower() for k in flat_word_data],
                data_ids=np.arange(len(group_word_ids)) + offset))
            offset += len(group_word_ids)

        return RawData(
            examples,
            response_data=dict(
                (k.lower(), KindData(ResponseKind.geco, flat_word_data[k])) for k in flat_word_data),
            validation_proportion_of_train=0.25)

    def _run_info(self, index_run):
        # use 4-fold CV
        return index_run % 4