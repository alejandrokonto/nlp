import json
import jieba
from hanziconv import HanziConv
import jieba.posseg as pseg
import thulac
from collections import Counter
import mafan
from hanzi_convertion.langconv import *
import os
import numpy as np


"""
    This file preprocesses data coming from PycharmProjects/NLP/ and 
    provides data structures usable by our model
"""
"""
    DIRECTORY BASES
"""
data_dir = '../data'

"""
    LOAD DATA FUNCTIONS
"""


def load_model_data():

    with open('../data/plain_word_2_senses.json', 'r', encoding='utf-8') as file:
        plw2sense = json.load(file)
    with open('../data/disambiguated_word_ID_2_plain_word.json', 'r', encoding='utf-8') as file:
        dwID2plw = json.load(file)
    with open('../data/disambiguated_word_ID_2_disamb_word.json', 'r', encoding='utf-8') as file:
        dwID2dw = json.load(file)
    with open('../data/senses.json', 'r', encoding='utf-8') as file:
        senses = json.load(file)
    with open('../data/senses_pos.json', 'r', encoding='utf-8') as file:
        senses_pos = json.load(file)
    with open('../data/senses_sub_chars.json', 'r', encoding='utf-8') as file:
        senses_sub_chars = json.load(file)
    with open('../data/plain_definitions.json', 'r', encoding='utf-8') as file:
        plain_defs = json.load(file)

    return senses, senses_pos, senses_sub_chars, plw2sense, dwID2dw, dwID2plw, plain_defs


def load_xinhua_data(path_to_files, ci_file, zi_file):

    with open(os.path.join(path_to_files, ci_file), 'r', encoding='utf-8') as file:
        word_data = json.load(file)

    with open(os.path.join(path_to_files, zi_file), 'r', encoding='utf-8') as file:
        character_data = json.load(file)

    return word_data, character_data


def load_wikipedia_data(path_to_file, file):

    with open(os.path.join(path_to_file, file), 'r', encoding='utf-8') as file:
        data_lines = file.readlines()

    data = [line.split('\t') for line in data_lines]

    yield data


def load_word2vec_embeddings(path_to_file, w_file, v_file):

    words_path = os.path.join(path_to_file, w_file)
    vectors_path = os.path.join(path_to_file, v_file)

    words, vectors = np.load(words_path), np.load(vectors_path)
    word2vec = dict()

    for word, vector in zip(words, vectors):
        word2vec[word] = vector

    return word2vec


"""
    PREPROCESS DATA 
"""


def check_traditional_simplified_conflict(tokens_list):

    original_form, converted_tokens = [], []
    for token in tokens_list:
        original_form.append(token['word'])
        if mafan.is_traditional(token['word']):
            converted_tokens.append(Converter('zh-hans').convert(token['word']))
        else:
            converted_tokens.append(token['word'])

    original_set = set(original_form)
    converted_set = set(converted_tokens)

    return original_set - converted_set


def re_process_wiki_corpora(corpora, output_file):

    segpos = thulac.thulac(seg_only=True)
    cnt = 0
    with open(os.path.join(data_dir, output_file), 'w', encoding='utf-8') as write_file:

        for token_list in corpora:
            cnt += 1
            new_token_list = []
            for token in token_list:
                if mafan.is_traditional(token):
                    new_token_list.append(mafan.to_simplified(token))
                else:
                    new_token_list.append(token)
            sentence = "".join(new_token_list)
            new_token_list = [token[0] for token in segpos.cut(sentence)]

            write_file.write('\t'.join(new_token_list) + '\n')
            if not cnt % 100000:
                print(cnt)


def find_non_common_words(corpora, words):

    # accumulate wiki data
    wiki_words = []
    for line in next(corpora):
        wiki_words += line

    wiki_words = set(wiki_words)

    # accumulate dictionary data
    dict_words = []
    for word in words:
        dict_words.append(word['word'])

    dict_words = set(dict_words)

    return dict_words - wiki_words


def find_non_common_chars(corpora, characters):

    # accumulate wiki data
    wiki_chars = set()
    cnt = 0
    for line in next(corpora):
        cnt += 1
        line = ''.join(line)
        chars = [ch for ch in line]
        wiki_chars = wiki_chars | set(chars)

        if cnt % 100000 == 0:
            print('Already processed ' + str(cnt))

    # accumulate dictionary data
    dict_chars = []
    for character in characters:
        dict_chars.append(character['character'])

    dict_chars = set(dict_chars)

    return dict_chars - wiki_chars


def dictionary_filtering(dictionary, token_list, file_output, char_mode=False):

    new_dict = []
    if not char_mode:
        for elem in dictionary:
            if elem['word'] in token_list:
                continue
            else:
                new_dict.append(elem)
    else:
        for elem in dictionary:
            if elem['character'] in token_list:
                continue
            else:
                new_dict.append(elem)

    with open(os.path.join(data_dir, file_output), 'w', encoding='utf-8') as file:
        json.dump(new_dict, file, ensure_ascii=False)


def align_words_defs(word_dict, char_dict):

    segpos = thulac.thulac(seg_only=True)
    cnt = 0

    words = []
    all_definitions = []
    print('Gathering definitions')
    for entry in word_dict:
        words.append(entry['word'])
        for definition in entry['definitions']:
                all_definitions.append([token[0] for token in segpos.cut(definition)])

    for entry in char_dict:
        for definitions in entry['definitions']:
            for definition in definitions:
                all_definitions.append([token[0] for token in segpos.cut(definition)])

    print('Flattening')
    all_definitions = [item for sublist in all_definitions for item in sublist]
    all_definition_words = set(all_definitions)
    del all_definitions

    print('Start aligning')
    entries_to_erase = []
    for word in words:
        cnt += 1
        if word not in all_definition_words:
            entries_to_erase.append(word)

        if cnt % 10000 == 0:
            print(str(cnt))

    print('Start filtering')
    dictionary_filtering(word_dict, entries_to_erase, 'zh_word_defs_filtered.json', char_mode=False)


def align_chars_defs(word_dict, char_dict):

    cnt = 0
    chars = dict()
    all_definitions = []

    print('Gathering definitions')
    for entry in word_dict:
        for definition in entry['definitions']:
            all_definitions.append(definition)

    for entry in char_dict:
        chars[entry['character']] = 0
        for definitions in entry['definitions']:
            for definition in definitions:
                all_definitions.append(definition)

    print('Flattening')
    all_definitions = [item for sublist in all_definitions for item in sublist]
    all_definitions = set(all_definitions)

    print('Start aligning')
    for definition in all_definitions:
        for ch in definition:
            cnt += 1
            if cnt % 10000 == 0:
                print(str(cnt))
            try:
                chars[ch] += 1
            except KeyError:
                continue

    entries_to_erase = []
    for k,v in chars.items():
        if v == 0:
            entries_to_erase.append(k)

    print('Start filtering')
    dictionary_filtering(char_dict, entries_to_erase, 'zh_char_defs_filtered.json', char_mode=True)


def clear_empty_def_entries(token_dict):

    entries_to_erase = []
    for entry in token_dict:
        if entry['definitions'] == []:
            entries_to_erase.append(entry['word'])

    print('Start filtering')
    dictionary_filtering(token_dict, entries_to_erase, 'zh_word_defs_filtered.json', char_mode=False)


"""
    CONSTRUCTING DATA STRUCTURES
"""


def construct_data_structures(word_dict, char_dict):

    plw2sense, dwID2plw, dwID2dw = dict(), dict(), dict()
    senses, senses_pos, senses_sub_chars, plain_defs = dict(), dict(), dict(), dict()

    index = 0

    # populate plw2sense and plain_defs for word_dict
    print('Populating data with word dictionary')
    for entry in word_dict:
        sub_sense_index = 1
        indices, plw2dw = [], []
        for definition in entry['definitions']:
            indices.append(index)
            plw2dw.append(entry['word'] + '.' + str(sub_sense_index))
            plain_defs[index] = definition
            sub_sense_index += 1
            index += 1
        plw2sense[entry['word']] = {'plw2dw': plw2dw, 'plw2dwID': indices}

    # populate plw2sense and plain_defs for char_dict
    print('Populating data with character dictionary')
    for entry in char_dict:
        sub_sense_index = 1
        indices, plw2dw = [], []
        for definitions, grammar in zip(entry['definitions'], entry['grammars']):
            for definition in definitions:
                indices.append(index)
                plw2dw.append(entry['character'] + '.' + grammar + '.' + str(sub_sense_index))
                plain_defs[index] = definition
                sub_sense_index += 1
                index += 1
        plw2sense[entry['character']] = {'plw2dw': plw2dw, 'plw2dwID': indices}

    # populate dwID2plw and dwID2dw
    for k, v in plw2sense.items():

        for id in v['plw2dwID']:
            dwID2plw[id] = k

        for dw, id in zip(v['plw2dw'], v['plw2dwID']):
            dwID2dw[id] = dw


    # calculate number of senses to be learned
    senses_number = len(plain_defs)

    # load the supplementary word2vec data:
    word2vec = load_word2vec_embeddings(data_dir, 'zh_wiki_14_emb_words.npy', 'zh_wiki_14_emb_vectors.npy')
    word2vec_2index = dict()

    id = senses_number
    for word in word2vec.keys():
        word2vec_2index[word] = id
        id += 1

    # initialize segmentation and POS tagging tool
    segpos = thulac.thulac()

    cnt = 0

    print('Populating the senses structure')
    # populate the senses structure

    for k, definition in plain_defs.items():

        # counter for progress output
        cnt += 1

        tokenized_def = segpos.cut(definition)
        tokens = [item[0] for item in tokenized_def]
        pos_tags = [item[1] for item in tokenized_def]

        sub_sense, sub_sense_sub_char = [], []
        for token in tokens:

            if token in plw2sense.keys():
                senses_dict = plw2sense[token]
                sub_sense.append(senses_dict['plw2dwID'])
                sub_sense_sub_char.append(sub_chars_defs(token, plw2sense))
            elif token in word2vec_2index.keys():
                # here, sense_id is only one number, there are no multiple senses
                sense_id = word2vec_2index[token]
                sub_sense.append([sense_id])
                sub_sense_sub_char.append(sub_chars_defs(token, plw2sense))
            else:
                # again, sense_id is only one integer, so we pass it as list
                sense_id = word2vec_2index['<UNDEFINED_WORD>']
                sub_sense.append([sense_id])
                sub_sense_sub_char.append([[-1]])

        # finished plain definition to senses encoding
        senses[k] = sub_sense
        # shunbian populate the pos tags that correspond to definition tokens
        senses_pos[k] = pos_tags
        # and also save the sub-char defs that correspond to each token in each definition
        senses_sub_chars[k] = sub_sense_sub_char

        if cnt % 1000 == 0:
            print('Already processed: ' + str(cnt))

    return senses, senses_pos, senses_sub_chars, plw2sense, dwID2dw, dwID2plw, plain_defs


def remove_meaningless_definitions(senses, senses_pos, senses_sub_chars, plw2sense, dwID2dw, dwID2plw, plain_defs):

    # gather the definitions of specified length
    definitions_to_erase = []
    for k, def_ in plain_defs.items():
        if len(def_) > 210 or len(def_) == 0 or def_ == "又。":
            definitions_to_erase.append(k)

    # gather disambiguate words and plai words that correspond to these IDs and shunbian clear ALL the dictionaries
    disamb_words_to_erase, plain_words_keys = [], []
    for k in definitions_to_erase:

        disamb_words_to_erase.append(dwID2dw[str(k)])
        dwID2dw.pop(str(k))

        plain_words_keys.append(dwID2plw[str(k)])
        dwID2plw.pop(str(k))

        senses.pop(str(k))
        senses_pos.pop(str(k))
        senses_sub_chars.pop(str(k))
        plain_defs.pop(str(k))

    print('Definitions erased: ' + str(len(definitions_to_erase)))
    # update the main dictionary 'plw2sense'
    for word_key, disamb_word, disamb_word_id in zip(plain_words_keys, disamb_words_to_erase, definitions_to_erase):
        entry = plw2sense[word_key]
        entry['plw2dw'].remove(disamb_word)
        entry['plw2dwID'].remove(int(disamb_word_id))

    # check for empty entries in the main dictionary
    cnt = 0
    for k in set(plain_words_keys):
        if plw2sense[k]['plw2dw'] == [] and plw2sense[k]['plw2dwID'] == []:
            cnt += 1
            plw2sense.pop(k)
    print('Entries erased: ' + str(cnt))

    return 1


def senses_padding(sub_sense, max_padding):

    padding = max_padding - len(sub_sense)
    return sub_sense + [-1]*padding


def definition_padding(definition, max_sense_padding, max_def_padding):

    def_padding = max_def_padding - len(definition)
    return definition + [[-1]*max_sense_padding]*def_padding


def sub_chars_defs(word, plw_dict):

    characters_def = []
    if len(word) == 1:
        return [[-1]]
    else:
        for char in word:
            try:
                characters_def.append(plw_dict[char]['plw2dwID'])
            except KeyError:
                characters_def.append([-1])
    return characters_def
# def sense_pos_padding
# def sense_sub_char_padding


if __name__ == "__main__":

    # words, characters = load_xinhua_data(data_dir, 'zh_word_defs_filtered.json', 'zh_char_defs_filtered.json')

    senses, senses_pos, senses_sub_chars, plw2sense, dwID2dw, dwID2plw, plain_defs = load_model_data()
    print('Loaded data')

    dict_out = []
    for k, _ in plw2sense.items():
        dict_out.append(k)

    with open(os.path.join(data_dir, "dictionary4tokenizer.txt"), 'w', encoding='utf-8') as file:
        file.write("\n".join(dict_out))

    # TESTING THULAC SEGMENTATION EFFICIENCY PART
    # text = "春江潮水连海平，海上明月共潮生。滟滟随波千万里，何处春江无月明！"
    #
    #
    # segpos = thulac.thulac()
    #
    # tokenized_def = segpos.cut(text)
    # tokens = [item[0] for item in tokenized_def]
    # pos_tags = [item[1] for item in tokenized_def]


    # remove_meaningless_definitions(senses, senses_pos, senses_sub_chars, plw2sense, dwID2dw, dwID2plw, plain_defs)
    #
    # print('Start saving data')
    # with open('../data/plain_word_2_senses.json', 'w', encoding='utf-8') as file:
    #     json.dump(plw2sense, file, ensure_ascii=False)
    # with open('../data/disambiguated_word_ID_2_plain_word.json', 'w', encoding='utf-8') as file:
    #     json.dump(dwID2plw, file, ensure_ascii=False)
    # with open('../data/disambiguated_word_ID_2_disamb_word.json', 'w', encoding='utf-8') as file:
    #     json.dump(dwID2dw, file, ensure_ascii=False)
    # with open('../data/senses.json', 'w', encoding='utf-8') as file:
    #     json.dump(senses, file)
    # with open('../data/senses_pos.json', 'w', encoding='utf-8') as file:
    #     json.dump(senses_pos, file)
    # with open('../data/senses_sub_chars.json', 'w', encoding='utf-8') as file:
    #     json.dump(senses_sub_chars, file)
    # with open('../data/plain_definitions.json', 'w', encoding='utf-8') as file:
    #     json.dump(plain_defs, file)
