import numpy as np
import json


def load_both_dictionaries():

    eywedu_words = np.load('../../data/zd.eywedu.com/total_words.npy')
    eywedu_defs = np.load('../../data/zd.eywedu.com/total_words_definitions.npy')

    eywedu_data = []
    for word, definition in zip(eywedu_words, eywedu_defs):
        eywedu_data.append({"word": word, "definitions":definition})

    with open('../../data/aies.cn/extracted_data/words_definitions.json', 'r', encoding='utf-8') as file:
        aies_data = json.load(file)

    return eywedu_data, aies_data


def extract_missing_words(dict_1, dict_2):

    words_in_dict_2 = []
    for word_2 in dict_2:
        words_in_dict_2.append(word_2['word'])
    set_dict_2 = set(words_in_dict_2)
    print('Loaded all dict 2')

    words_in_dict_1 = []
    for word_1 in dict_1:
        words_in_dict_1.append(word_1['word'])
    set_dict_1 = set(words_in_dict_1)
    print('Loaded all dict 1')

    return set_dict_1 - set_dict_2


def align_dictionaries(dict_1, dict_2, words_list):

    for word in words_list:

        # retreive the definition of the word from dict_2
        definitions = []
        for word_ in dict_2:
            if word_['word'] == word:
                definitions = word_['definitions']
                break

        # add it up to the big one
        examples_missing = len(definitions)
        examples = [[] for _ in range(examples_missing)]

        dict_1.append({"word": word, "definitions": definitions, "examples": examples})

    return dict_1


if __name__ == "__main__":

    # load extracted word data and return them as lists of dictionaries for each word
    eywedu_data, aies_data = load_both_dictionaries()

    # find the words missing from aies.cn dictionary
    words_to_fill_up = extract_missing_words(eywedu_data, aies_data)

    # add the missing words to aies.cn dictionary and return it
    print("Start aligning....")
    aligned_dictionary = align_dictionaries(aies_data, eywedu_data, words_to_fill_up)

    # save it
    with open('../../data/aligned_data/words_definitions_aligned.json', 'w', encoding='utf-8') as file:
        json.dump(aligned_dictionary, file, ensure_ascii=False)