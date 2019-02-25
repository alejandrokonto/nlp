import re
from bs4 import BeautifulSoup
import glob
import os
import numpy as np
import json

grammar_re = re.compile("〈.〉")
parenthesis_re = re.compile("（.*）")
en_pinyin_content_re = re.compile("（.*）|\[.*\]")
english_translations_re = re.compile("\[.*\]")
chinese_re = re.compile("[\u4e00-\u9fff]+")
examples_re = re.compile("又如:")
tongbenyi_re = re.compile("同本义")
tong_re = re.compile("通")
youru_re = re.compile(" *又如：")

vowels_pinyin = 'aeiouvüāēīōūǖáéíóúǘǎěǐǒǔǚàèìòùǜAEIOUVÜĀĒĪŌŪǕÁÉÍÓÚǗǍĚǏǑǓǙÀÈÌÒÙǛ'
consonants_pinyin = 'bpmfdtnlgkhjqxzcsrwyBPMFDTNLGKHJQXZCSRWY'

vowels_en = 'aeiouAEIOU'
consonants_en = 'bcdfghjklmnpqrstvwxzBCDFGHJKLMNPQRSTVWXZ'


def extract_character(html):

    # extract the character the corresponds to the definitions in the page
    main_content = html.find_all('div', 'mcon')[1]
    character = main_content.find('img').get('alt').split(' ')[1]

    return character


def extract_subsections(html):

    # find the subsections that correspond to “基本字义”， “详细解释”， “词性变化”
    zi_content = html.find('div', id='zicontent')

    # defs are storing the definitions, ps are storing the grammar notations
    defs, ps = [], []
    for child in zi_content.children:
        if child.name == 'ol':
            defs.append(child)
        if child.name == 'p':
            ps.append(child)

    return defs, ps


def extract_grammars(ps):
    grammars = []
    # the first element of the grammars list is not holding any grammar value. Take the rest
    for el in ps:
        try:
            grammar_object = grammar_re.search(el.string)
        except TypeError as e:
            print("Current file seems to have problem locating the character grammar infor. Ignoring for now")
            return None

        if grammar_object:
            s_i, e_i = grammar_object.span()
            grammars.append(el.string[s_i + 1: e_i - 1])

    return grammars


def extract_basic_definitions(defs, zi):

    definitions = []
    examples = []

    for el in defs.children:
        raw_line = el.string
        # check if it is a literature component:
        literature_entity = False

        # if there are other html elements, the el.string will return a None type. First, ignore them
        if not raw_line:
            return None, None

        for ch in raw_line:
            if ch == "《":
                literature_entity = True
                break
        if literature_entity:
            continue

        # first check if the whole dictionary entry has the structure "definition : examples"
        if "：" in raw_line:
            # check how many ":" exist
            components = raw_line.split("：")
            if len(components) > 2:
                definition = "，".join(components[:-1])
                example = components[-1]
            else:
                definition, example = raw_line.split("：")
        else:
            definition = raw_line
            example = ""

        # check definition string has trash content and proceed accordingly
        parenthesis_object = parenthesis_re.search(definition)
        if parenthesis_object:
            s_i, e_i = parenthesis_object.span()
            parenthesis_text = definition[s_i + 1: e_i - 1]
            if check_pinyin_or_en_content(parenthesis_text):
                definition = definition[:s_i] + definition[e_i:]

        # check the examples string and proceed with processing.
        examples_list = example.split("。")
        subexamples = []
        # Loop through every example except the last element which is an empty string
        for example_ in examples_list[:-1]:

            parenthesis_object = parenthesis_re.search(example_)
            if parenthesis_object:
                s_i, e_i = parenthesis_object.span()
                parenthesis_text = example_[s_i + 1: e_i - 1]
                trash_content = check_pinyin_or_en_content(parenthesis_text)

                if trash_content:
                    example_final = example_[: s_i] + example_[e_i:]
                    subexamples.append(example_final.replace("～", zi))
                else:
                    if "；" in parenthesis_text:
                        parenthesis_elements = parenthesis_text.split("；")
                    else:
                        parenthesis_elements = [parenthesis_text]
                    example_final = example_[: s_i] + example_[e_i:]
                    for el in parenthesis_elements:
                        example_final += "， " + el
                    example_final += "。"
                    subexamples.append(example_final.replace("～", zi))
            else:
                subexamples.append(example_.replace("～", zi))

        examples.append(subexamples)
        definitions.append(definition)

    return definitions, examples


def check_pinyin_or_en_content(text):
    for ch in text:
        if ch in consonants_pinyin or ch in consonants_en or ch in vowels_pinyin or ch in vowels_en:
            return True
    return False


def extract_detailed_definitions(defs):
    definitions = []
    examples = []
    cnt = 0

    for el in defs.children:
        # cnt total definitions
        cnt += 1
        if not el.find('ol'):
            raw_text = str(el.string)
            # check first if it is the line with the main definition (benyi)
            if "本义:" in raw_text:
                s_i = raw_text.find("本义:")
                definitions.append(raw_text[s_i + 3:].strip(")"))
                examples.append([])
            elif tongbenyi_re.match(raw_text) or tong_re.match(raw_text) or "另见 " in raw_text:
                continue
            elif "又如:" in raw_text or "又如∶" in raw_text:
                example = raw_text.strip(" ")[3:]
                curr_i = len(examples) - 1
                # check if the previous definition has been accepted or rejected:
                if curr_i < 0:
                    continue
                else:
                    # else proceed normally
                    if ";" in raw_text:
                        examples_list = example.split(";")
                        examples[curr_i] = examples_list
                    else:
                        examples[curr_i] = [example]
            elif "如:" in raw_text:
                components = raw_text.split("如:")
                if len(components) > 2:
                    definition = components[0]
                    example = '、'.join(components[1:])
                else:
                    definition, example = raw_text.split("如:")

                translation_object = english_translations_re.search(definition)
                if translation_object:
                    s_i, e_i = translation_object.span()
                    definitions.append(definition[:s_i] + definition[e_i:])
                else:
                    definitions.append(definition)

                if "《" in example:
                    examples.append([])
                elif ";" in example:
                    examples.append(example.split(";"))
                else:
                    examples.append([example])

            elif cnt == 2:
                translation_object = english_translations_re.search(raw_text)
                curr_i = len(definitions) - 1
                # check if curr_i is a valid value:
                if curr_i < 0:
                    continue
                else:
                    # else, proceed normally
                    if translation_object:
                        s_i, e_i = translation_object.span()
                        definitions[curr_i - 1] += "， " + raw_text[:s_i] + raw_text[e_i:]
                    else:
                        definitions[curr_i - 1] += "， " + raw_text
            else:
                translation_object = english_translations_re.search(raw_text)
                if translation_object:
                    s_i, e_i = translation_object.span()
                    definitions.append(raw_text[:s_i] + raw_text[e_i:])
                else:
                    definitions.append(raw_text)
                examples.append([])
        else:
            definition = next(el.stripped_strings)
            if tongbenyi_re.match(definition) or tong_re.match(definition):
                continue
            elif "本义:" in definition:
                s_i = definition.find("本义:")
                e_i = definition[s_i + 3:].find(")")
                definitions.append(definition[s_i + 3: s_i + 3 + e_i + 1])
                examples.append([])
            else:
                translation_object = english_translations_re.search(definition)
                if translation_object:
                    s_i, e_i = translation_object.span()
                    definitions.append(definition[:s_i] + definition[e_i:])
                else:
                    definitions.append(definition)
                examples.append([])

    return definitions, examples


def extract_data(files_dir):

    files = glob.glob(files_dir + '/*.txt')
    cnt = 0
    files_num = len(files)

    # return lists
    characters, main_defs, main_defs_examples, extended_defs, extended_defs_examples, defs_grammars = [], [], [], [], [], []

    # collect the "rejected" characters
    no_definition_chars = []

    for file in files:

        # increase the counter
        cnt += 1

        print("Processing file " + str(cnt) + " of " + str(files_num) + " : " + os.path.basename(file))
        with open(file, encoding='utf-8') as file_:
            html_text = file_.read()

        html_doc = BeautifulSoup(html_text, 'html.parser')

        # character for which we re extracting definitions
        character = extract_character(html_doc)

        # all the definitions and p elements that hold grammar notations
        defs, ps = extract_subsections(html_doc)

        # extract grammar notations
        grammars = extract_grammars(ps)
        # if there are no grammar informaton, then don't consider this character at all
        if not grammars:
            print("Current file seems not to contain grammar info. We proceed with the next one.")
            no_definition_chars.append(character)
            continue

        # the number of definitions (extended) to be extracted
        ext_defs_num = len(grammars)

        # extract main definitions
        main_defs_, main_defs_examples_ = extract_basic_definitions(defs[0], character)
        if not main_defs_:
            print("Current file has corrupted html tag on main definitions part. Ignore. Move with next one.")
            no_definition_chars.append(character)
            continue

        # extract detailed definitions
        subdefs, subexmps = [], []
        subdefs_num = 0

        if len(grammars) == len(defs):
            subdefs.append(main_defs_)
            subexmps.append(main_defs_examples_)
        elif len(grammars) > len(defs):
            print("We have no way of aligning anything in defintions-grammars elements. Keep moving")
            no_definition_chars.append(character)
            continue
        else:
            for i in range(ext_defs_num):
                # we add one because the first one is the basic (simple) definitions paragraph
                def_p = defs[i + 1]
                definitions, examples = extract_detailed_definitions(def_p)

                subdefs_num += len(definitions)

                subdefs.append(definitions)
                subexmps.append(examples)

        # check if the detailed definitions are more than the basic ones. If not, substitute em with the basic ones.
        if len(main_defs_) > subdefs_num and subdefs_num <= 3:
            extended_defs.append([main_defs_])
            extended_defs_examples.append([main_defs_examples_])
            # grammar notation de hua, keep only the basic one (the first one). According to my understanding,
            # basic definitions do correspond to the definitions of the basic grammar form (that is, the first one)
            grammars = [grammars[0]]
        else:
            extended_defs.append(subdefs)
            extended_defs_examples.append(subexmps)

        # save the rest of the data
        defs_grammars.append(grammars)
        characters.append(character)
        main_defs.append(main_defs_)
        main_defs_examples.append(main_defs_examples_)

        print("Processing has finished successfully")

    # return all values
    return characters, defs_grammars, main_defs, main_defs_examples, extended_defs, extended_defs_examples, no_definition_chars


def convert_txt_json(txt_file):

    with open(txt_file, 'r', encoding='utf-8') as file:
        txt = file.read()

    ch_defs_border = '-------------------------------------------------------------------------------\n'
    defs = txt.split(ch_defs_border)

    # list that will hold the final data
    data = []
    defs_cnt = len(defs)
    for cnt, character_section in enumerate(defs):
        print(str(cnt + 1) + " from " + str(defs_cnt) + " files.")

        character = ''
        examples_list, definitions_list, grammars = [], [], []
        lines = character_section.split('\n')
        i = 0
        finished = False
        while not finished:

            if lines[i] == "Character:":
                i += 1
                character = lines[i].strip()

            elif lines[i] == "Corresponding grammars:":
                i += 1
                grammars = lines[i].split(' ')

            elif lines[i] == "Detailed definitions n examples:":
                i += 1
                definitions_sections = '\n'.join(lines[i:])
                definitions_sections = definitions_sections.split("Definitions Section:\n")
                # print(definitions_sections)
                for definition_section in definitions_sections:
                    subdefs_list, subexamples_list = [], []
                    if definition_section == '':
                        continue
                    subdefs = definition_section.split('\n')

                    for subdef in subdefs:

                        components = subdef.strip("\n").split("[examples]:")
                        definition, examples = components[0], components[1:]
                        # For every definition: 1. Add full stop at the end of each definition 2. Substitute "~" with the
                        # corresponding character 3. clear pinyin (with their parenthesis)
                        definition = definition.strip()
                        definition = definition.replace('～', character)

                        # check definition string has trash content and proceed accordingly
                        parenthesis_object = parenthesis_re.search(definition)
                        if parenthesis_object:
                            s_i, e_i = parenthesis_object.span()
                            parenthesis_text = definition[s_i + 1: e_i - 1]
                            if check_pinyin_or_en_content(parenthesis_text):
                                definition = definition[:s_i] + definition[e_i:]

                        if definition[-1] != "。":
                            definition += "。"

                        # create the examples list:
                        examples_list = examples.split("\t")
                        examples_list = [exmp.strip('\t') for exmp in examples_list]

                        # definitions
                        definitions_list.append(definition)

                # terminate
                finished = True
            else:
                i += 1
        data.append({'character': character, 'grammars':grammars, 'definitions':definitions_list, 'examples': examples_list})

    with open('../../../data/aies.cn/extracted_data/characters_definitions.json', 'w') as file:
        json.dump(data, file, ensure_ascii=False)


if __name__ == "__main__":

    # BELOW IS THE SCRIPTS OF EXTRACTING AND SAVING DATA TO TEXT
    # files_dir = ['../../../data/aies.cn/character_pages/id=MjA5MQ==.txt']
    # files_dir = '../../../data/aies.cn/character_pages'
    #
    # characters, defs_grammars, main_defs, main_defs_examples, extended_defs, extended_defs_examples, no_def_chars = extract_data(files_dir)
    # data = []
    # for character, grammar_list, definitions, examples in zip(characters, defs_grammars, extended_defs, extended_defs_examples):
    #
    #     data.append({'character':character, 'grammars':grammar_list, 'definitions':definitions, 'examples':examples})
    #
    # with open('../../../data/aies.cn/extracted_data/characters_definitions.json', 'w', encoding='utf-8') as file:
    #     json.dump(data, file, ensure_ascii=False)

    # # save characters with no definition here
    # np.save('../../../data/aies.cn/extracted_data/zi_no_def_found.npy', no_def_chars)


    # CLEAN THE DEFINITIONS

    with open('../../../data/aies.cn/extracted_data/characters_definitions.json', 'r', encoding='utf-8') as file:
        data = json.load(file)

    data_filtered = []
    cnt = 0
    for data_point in data:

        all_definitions = data_point['definitions']
        character = data_point['character']
        all_defs = []
        for subdefinitions in all_definitions:
            subdefs = []
            for definition in subdefinitions:
                # For every definition: 1. Add full stop at the end of each definition 2. Substitute "~" with the
                # corresponding character 3. clear pinyin (with their parenthesis)
                definition = definition.strip()
                definition = definition.replace('～', character)
                if definition == '同本义':
                    cnt += 1
                # check definition string has trash content and proceed accordingly
                parenthesis_object = parenthesis_re.search(definition)
                if parenthesis_object:
                    s_i, e_i = parenthesis_object.span()
                    parenthesis_text = definition[s_i + 1: e_i - 1]
                    if check_pinyin_or_en_content(parenthesis_text):
                        definition = definition[:s_i] + definition[e_i:]

                if definition != "":
                    if definition[-1] != "。":
                        definition += "。"

                subdefs.append(definition)
            all_defs.append(subdefs)
        data_point['definitions'] = all_defs
        data_filtered.append(data_point)

    with open('../../../data/aies.cn/extracted_data/characters_definitions.json', 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False)