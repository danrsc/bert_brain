import os
import itertools
import csv
from dataclasses import dataclass
import dataclasses
import numpy as np


from bert_erp_tokenization import bert_tokenize_with_spacy_meta, RawData


__all__ = ['read_natural_stories', 'natural_stories_data', 'read_natural_story_codings']


@dataclass
class _BatchRecord:
    worker_id: str
    work_time_in_seconds: int
    correct: int
    item: int
    zone: int
    reaction_time: int


@dataclass
class _WordRecord:
    word: str
    item: int
    zone: int
    sentence: int


def _read_batch(path):
    with open(path, 'rt', newline='') as f:
        for record in csv.DictReader(f):
            yield _BatchRecord(
                worker_id=record['WorkerId'],
                work_time_in_seconds=int(record['WorkTimeInSeconds']),
                correct=int(record['correct']),
                item=int(record['item']),
                zone=int(record['zone']) - 2,  # subtract 2 to match with zone in stories
                reaction_time=int(record['RT']))


def _read_sentence_ids(directory_path):
    sentence_id = 0
    result = dict()
    with open(os.path.join(directory_path, 'stories-aligned.conllx'), 'rt') as conllx_file:
        for line in conllx_file:
            # ID:     Word index, integer starting at 1 for each new sentence; may be a range for multiword tokens;
            #         may be a decimal number for empty nodes
            #         (decimal numbers can be lower than 1 but must be greater than 0).
            # FORM:   Word form or punctuation symbol.
            # LEMMA:  Lemma or stem of word form.
            # UPOS:   Universal part-of-speech tag.
            # XPOS:   Language-specific part-of-speech tag; underscore if not available.
            # FEATS:  List of morphological features from the universal feature inventory or from a defined
            #         language-specific extension; underscore if not available.
            # HEAD:   Head of the current word, which is either a value of ID or zero (0).
            # DEPREL: Universal dependency relation to the HEAD (root iff HEAD = 0) or a
            #         defined language-specific subtype of one.
            # DEPS:   Enhanced dependency graph in the form of a list of head-deprel pairs.
            # MISC:   Any other annotation.
            line = line.strip()
            if line.startswith('#'):
                continue
            if len(line) == 0:
                sentence_id += 1
                continue
            id_, form, lemma, upos, xpos, feats, head, deprel, deps, misc = line.split('\t')
            token_id = [int(p) for p in misc[len('TokenId='):].split('.')]
            if len(token_id) > 2:
                item, zone, part = token_id
            else:
                item, zone = token_id
            result[(item, zone)] = sentence_id
    return result


def _sentence_iterator(word_records):
    sentence = list()
    for record in word_records:
        if len(sentence) > 0 and sentence[-1].sentence != record.sentence:
            yield sentence
            sentence = list()
        sentence.append(record)
    if len(sentence) > 0:
        yield sentence


def _read_story_sentences(directory_path):
    sentence_ids = _read_sentence_ids(directory_path)
    story_records = dict()
    with open(os.path.join(directory_path, 'all_stories.tok'), 'rt', newline='') as all_stories_file:
        for record in csv.DictReader(all_stories_file, delimiter='\t'):
            item = int(record['item'])
            zone = int(record['zone'])
            record = _WordRecord(record['word'], item, zone, sentence_ids[(item, zone)])
            if record.item not in story_records:
                story_records[record.item] = list()
            story_records[record.item].append(record)
    for item in sorted(story_records):  # sort so we get consistent unique ids
        story = sorted(story_records[item], key=lambda r: r.zone)
        for sentence in _sentence_iterator(story):
            yield sentence


@dataclass
class _CodingRecord:
    # categorized into groups as in http://www.lrec-conf.org/proceedings/lrec2018/pdf/337.pdf
    # see that link for examples of each

    story_name: str
    sentence_number: int
    sentence: str

    # conjunction
    local_verb_phrase_conjunction: int
    non_local_verb_phrase_conjunction: int
    local_noun_phrase_conjunction: int
    non_local_noun_phrase_conjunction: int
    sentential_co_ordination: int
    complementizer_phrase_conjunctions: int
    adjective_conjunction: int

    # relative clauses
    source_relative_clause_restrictive: int
    source_relative_clause_non_restrictive: int
    object_relative_clause_restrictive: int
    object_relative_clause_non_restrictive: int
    object_relative_clause_non_canonical: int
    no_relativizer_object_relative_clause: int
    adverbial_relative_clause: int
    free_relative_clause: int

    # ambiguity
    noun_phrase_sentence_ambiguity: int
    main_verb_reduced_relative_ambiguity_easier: int
    main_verb_reduced_relative_ambiguity_hard: int
    prepositional_phrase_attachment_ambiguity: int

    # displacement
    tough_movement: int
    parenthetical: int
    topicalization: int
    question_wh_subject: int
    question_wh_other: int

    # miscellaneous
    non_local_subject_verb: int
    non_local_verb_direct_object: int
    gerund_modifier: int
    sentential_subject: int
    post_nominal_adjective: int
    idiom: int
    quotation: int
    it_cleft: int
    even_than_construction: int
    if_then_construction: int
    as_as_construction: int
    so_that_construction: int
    question_yn: int
    infinitive_verb_phrase_subject: int
    inanimate_subjects: int
    animacy_hierarchy_violation: int


def _read_coding(path, story_name):

    field_name_map = {
        'sentence_number': 'SentNum',
        'sentence': 'Sentence',

        # conjunction
        'local_verb_phrase_conjunction': 'local VP conjunction',
        'non_local_verb_phrase_conjunction': 'non-local VP conjunction',
        'local_noun_phrase_conjunction': 'local NP conjunction',
        'non_local_noun_phrase_conjunction': 'non-local NP conjunction',
        'sentential_co_ordination': 'S co-ordination',
        'complementizer_phrase_conjunctions': 'CP conjunctions',
        'adjective_conjunction': 'adj conjunction',

        # relative clauses
        'source_relative_clause_restrictive': 'SRC restr',
        'source_relative_clause_non_restrictive': 'SRC non-restr',
        'object_relative_clause_restrictive': 'ORC restr',
        'object_relative_clause_non_restrictive': 'ORC non-restr',
        'object_relative_clause_non_canonical': 'ORC non-canon',
        'no_relativizer_object_relative_clause': 'no-relativizer ORC',
        'adverbial_relative_clause': 'adverbial RC',
        'free_relative_clause': 'free relative',

        # ambiguity
        'noun_phrase_sentence_ambiguity': 'NP/S ambig',
        'main_verb_reduced_relative_ambiguity_easier': 'MV/RR ambig EASIER',
        'main_verb_reduced_relative_ambiguity_hard': 'MV/RR ambig HARD',
        'prepositional_phrase_attachment_ambiguity': 'attachment ambig',

        # displacement
        'tough_movement': 'tough mvt',
        'parenthetical': 'parenthetical',
        'topicalization': 'topicalization',
        'question_wh_subject': 'question_wh_subj',
        'question_wh_other': 'question_wh_other',

        # miscellaneous
        'non_local_subject_verb': 'non-local SV',
        'non_local_verb_direct_object': 'non-local verb-DO',
        'gerund_modifier': 'gerund modifier',
        'sentential_subject': 'sent subj',
        'post_nominal_adjective': 'post-nominal adj',
        'idiom': 'idiom',
        'quotation': 'quote',
        'it_cleft': 'it-cleft',
        'even_than_construction': 'even…than',
        'if_then_construction': 'if...then constr',
        'as_as_construction': 'as…as constr',
        'so_that_construction': 'so…that constr',
        'question_yn': 'question_YN',
        'infinitive_verb_phrase_subject': 'inf VP subject',
        'inanimate_subjects': 'inanimate subjects',
        'animacy_hierarchy_violation': 'animacy hierarchy viol'
    }

    fields = dataclasses.fields(_CodingRecord)

    for field in fields:
        if field.name == 'story_name':
            continue
        if field.name not in field_name_map:
            raise ValueError('missing field in field_name_map: {}'.format(field.name))

    records = list()
    with open(path, 'rt', newline='') as coding_file:
        for record in csv.DictReader(coding_file, delimiter='\t'):
            values = dict()
            for field in fields:
                if field.name == 'story_name':
                    values[field.name] = story_name
                    continue
                record_name = field_name_map[field.name]
                str_value = record[record_name].strip()
                if len(str_value) == 0 and field.type == int:
                    values[field.name] = 0
                else:
                    values[field.name] = field.type(record[record_name])
            records.append(_CodingRecord(**values))
    return records


def _read_codings(directory_path):

    result = dict()
    for coding_file_name in [
            'aqua.txt', 'boar.txt', 'elvis.txt', 'high_school.txt', 'king_of_birds.txt', 'matchstick.txt',
            'mr_sticky.txt', 'roswell.txt', 'tourette.txt', 'tulips.txt']:
        story_name = os.path.splitext(coding_file_name)[0]
        for record in _read_coding(os.path.join(directory_path, coding_file_name), story_name):
            result[record.sentence] = record
    return result


def _read_reaction_times(directory_path):
    groups = dict()
    all_worker_ids = set()
    for record in itertools.chain(
            _read_batch(os.path.join(directory_path, 'batch1_pro.csv')),
            _read_batch(os.path.join(directory_path, 'batch2_pro.csv'))):
        if record.correct < 5:  # rater had poor comprehension
            continue
        all_worker_ids.add(record.worker_id)
        key = (record.item, record.zone)
        if key not in groups:
            groups[key] = dict()
        groups[key][record.worker_id] = record.reaction_time

    reaction_times = np.full((len(groups), len(all_worker_ids)), np.nan)
    sorted_keys = list(sorted(groups))
    all_worker_ids = list(sorted(all_worker_ids))
    for idx_key, key in enumerate(sorted_keys):
        for idx_worker, worker in enumerate(all_worker_ids):
            if worker in groups[key]:
                reaction_times[idx_key, idx_worker] = groups[key][worker]

    return reaction_times, sorted_keys, all_worker_ids


def read_natural_story_codings(directory_path):
    codings = _read_codings(directory_path)
    result = dict()
    for unique_id, sentence_word_records in enumerate(_read_story_sentences(directory_path)):
        text = ' '.join(sentence_word_records)
        if text not in codings:
            raise ValueError('Unable to find codings for sentence: {}'.format(text))
        result[unique_id] = codings[text]


def read_natural_stories(spacy_tokenize_model, bert_tokenizer, directory_path):
    reaction_times, keys, _ = _read_reaction_times(directory_path)
    key_to_row = dict((k, i) for i, k in enumerate(keys))
    examples = list()
    for unique_id, sentence_word_records in enumerate(_read_story_sentences(directory_path)):
        offsets = [key_to_row[(r.item, r.zone)] for r in sentence_word_records]
        assert(np.all(np.diff(offsets) == 1))  # assert these are contiguous
        input_features = bert_tokenize_with_spacy_meta(
            spacy_tokenize_model, bert_tokenizer, unique_id,
            [r.word for r in sentence_word_records], offsets[0])
        examples.append(input_features)

    return examples, {'ns_spr': reaction_times}


def natural_stories_data(spacy_tokenize_model, bert_tokenizer, path):
    examples, data = read_natural_stories(spacy_tokenize_model, bert_tokenizer, path)
    return RawData(examples, data, test_proportion=0., validation_proportion_of_train=0.1)


# library(plyr)
# library(dplyr)
# library(ggplot2)
#
# #read in RT data from 2 separate files
# b2 <- read.csv('batch2_pro.csv')
# b1 <- read.csv('batch1_pro.csv')
# d <- rbind(b1, b2)
#
# ##subtract 2 from zone to properly align region...should confirm with Hal that this is correct,
# ## but the RTs seem to line up correctly in plots
# d$zone <- d$zone - 2
#
# #read in story words and region
# #item is story (1-10), zone is RT region
# word.df <- read.csv('all_stories.tok', sep = '\t')
# d <- merge(d, word.df, by= c('item', 'zone'), all.x = T, all.y = T)
#
# #remove regions that do not have words
# d <- filter(d, !is.na(word))
#
# #exclude stories where subject does not get more than 4/6 correct
# unfiltered <- d
# d <- filter(d, correct > 4)
#
# #exclude data points less than 50 ms, greater than 3000 ms
# d <- d[d$RT > 100 & d$RT < 3000, ]
# d$l <- nchar(as.character(d$word))
#
#
# #calculate by-word statistics
#
# gmean <- function(x) exp(mean(log(x)))
# gsd   <- function(x) exp(sd(log(x)))
#
# word.info = d %>%
#   group_by(word, zone, item) %>%
#     summarise(nItem=length(RT),
#               meanItemRT=mean(RT),
# 	      sdItemRT=sd(RT),
# 	      gmeanItemRT=sd(RT),
# 	      gsdItemRT=gsd(RT))
#
# d <- inner_join(d, word.info, by=c("word", "zone", "item"))
#
# #write processed output, by word, overall
# #write.table(word.info, 'processed_wordinfo.tsv', quote = F, row.names=F, sep="\t")
# #write.table(d, 'processed_RTs.tsv', quote=F, row.names=F, sep="\t")
#
# ggplot(d, aes(RT)) + facet_grid( . ~ WorkerId) + geom_histogram()
#
#
# ##make plot
# make.story.plot <- function(item, group)
# {
# return (
#   ggplot(word.info[word.info$item == item & word.info$group == group, ],
#   aes(x = zone, y = meanItemRT, group = group)) +
#   geom_line(colour = 'grey') +
#   geom_text(aes( x = zone, y= meanItemRT, label = word), size = 2) + facet_grid(group~ .) +
#   theme_bw() + coord_cartesian(ylim = c(min(word.info$meanItemRT), 550)))
# }
#
# word.info$group <- cut(word.info$zone, 20, labels = F)
#
# pdf('practice_RTplot.pdf')
# for (i in seq(1:20))
# print(make.story.plot(1, i))
# dev.off()
