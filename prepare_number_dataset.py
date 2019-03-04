import os
from syntactic_dependency import preprocess_english_morphology, collect_paradigms, extract_dependency_patterns, \
    generate_morph_pattern_test, DependencyTree, universal_dependency_reader, make_token_to_paradigms, make_ltm_to_word
from bert_erp_paths import Paths


def main():
    paths = Paths()
    data_loader = paths.make_data_loader()
    bert_tokenizer = data_loader.make_bert_tokenizer()

    conll_reader = universal_dependency_reader
    path = data_loader.english_web_universal_dependencies_v_2_3_path

    paradigms = collect_paradigms(
        os.path.join(path, 'en_ewt-ud-train.conllu'), morphology_preprocess_fn=preprocess_english_morphology)

    trees = [
        DependencyTree.from_conll_rows(sentence_rows, conll_reader.root_index, conll_reader.offset, text)
        for sentence_rows, text in conll_reader.iterate_sentences(
            os.path.join(path, 'en_ewt-ud-train.conllu'), morphology_preprocess_fn=preprocess_english_morphology)]

    syntax_patterns = extract_dependency_patterns(trees, 5, feature_keys={'Number'})

    paradigms = make_token_to_paradigms(paradigms)

    ltm_paradigms = make_ltm_to_word(paradigms)

    examples = list()
    for pattern in syntax_patterns:
        examples.extend(generate_morph_pattern_test(trees, pattern, ltm_paradigms, paradigms, bert_tokenizer))

    if not os.path.exists(data_loader.number_dataset_path):
        os.makedirs(data_loader.number_dataset_path)

    with open(os.path.join(data_loader.number_dataset_path, 'generated.txt'), 'wt') as generated_output:
        for example in examples:
            generated_output.write(example.delimited())
            generated_output.write('\n')


if __name__ == '__main__':
    main()
