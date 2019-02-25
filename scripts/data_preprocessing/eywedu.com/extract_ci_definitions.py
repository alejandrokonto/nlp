import re
import json
import glob
import os

vowels = 'aeiouvüāēīōūǖáéíóúǘǎěǐǒǔǚàèìòùǜAEIOUVÜĀĒĪŌŪǕÁÉÍÓÚǗǍĚǏǑǓǙÀÈÌÒÙǛ'
consonants = 'bpmfdtnlgkhjqxzcsrwyBPMFDTNLGKHJQXZCSRWY'


def extract_content_subsections(file):

    with open(file, encoding='utf-8') as f:
        data = json.load(f)

    # it returns a list so keep the only element, to have a string
    data = data[0]

    # here we select the "more" part of the json file which contains the detailed definitions of a characters and it's
    # corresponding words
    data = data['more']

    # each section is divided by a double newline (not everytime, however, for extracting word definitions this
    # criterion will suffice
    data_sections = data.split('\n\n')

    # retreive only the subsection which doesn't contain characters 【 or 】, which are used for character grammar
    # explanation
    for section in data_sections:
        if "【" not in section or "】" not in section:
            return section

    # if section not found, return an empty string
    return ""


def extract_pinyin_positions(words_list):
    positions = []
    index = 0
    for line in words_list:
        # check if it's the pinyin line
        pinyin = True
        for ch in line:
            if ch not in vowels and ch not in consonants and ch != '-':
                pinyin = False
                break

        # if you found pinyin line, save it
        if pinyin:
            positions.append(index)

        index += 1

    return positions, len(positions)


def separate_definitons_n_examples(lines_list):
    translation_reg = re.compile("〖.*〗(\[方言\])*∶*")
    reference_reg = re.compile("见")
    definitions, examples = [], []
    positions = []

    for i, line in enumerate(lines_list):
        res = translation_reg.match(line)
        if res:
            _, end = res.span()
            n_line = line[end:]
            ref = reference_reg.match(n_line)
            if ref:
                _, end = ref.span()
                definitions.append(n_line[end:].strip("“”"))
                positions.append(i)
            else:
                definitions.append(n_line)
                positions.append(i)

    max_len = len(positions)

    for i, pos in enumerate(positions):

        if i < max_len - 1:
            examples.append(lines_list[pos + 1:positions[i + 1]])
        else:
            examples.append(lines_list[pos + 1:])

    return definitions, examples


def extract_word_info(lines_list):

    ciyu, jieshi, lizi = [], [], []
    pinyin_inds, pinyin_num = extract_pinyin_positions(lines_list)

    for cnt, ind in enumerate(pinyin_inds):
        ciyu.append(lines_list[ind - 1])

        if cnt < pinyin_num - 1:
            start_ind = ind + 1
            end_ind = pinyin_inds[cnt + 1] - 1
            definitions, examples = separate_definitons_n_examples(lines_list[start_ind:end_ind])
        else:
            start_ind = ind + 1
            definitions, examples = separate_definitons_n_examples(lines_list[start_ind:])
        jieshi.append(definitions)
        lizi.append(examples)

    return ciyu, jieshi, lizi


def extract_data_from_files(dir):

    files = glob.glob(dir)

    # the "total" collection of words, definitions, and examples. Actually examples we'll pass for now.
    ciyu_t, jieshi_t, lizi_t = [], [], []

    # count the total files and the files with no word occurences
    cnt_files, cnt_no_data_files = len(files), 0

    # keep a general purpose counter
    cnt = 0

    for file in files:
        cnt += 1

        data = extract_content_subsections(file)
        if data == "":
            cnt_no_data_files += 1
            print("No word entries found for file :" + os.path.basename(file))
        else:
            lines = data.split("\n")
            ciyu, jieshi, _ = extract_word_info(lines)
            ciyu_t.extend(ciyu)
            jieshi_t.extend(jieshi)

        if not cnt % 100:
            print("Already processed " + str(cnt) + "/" + str(cnt_files) + " files")

    print("Finished processing. Didn't find words for " + str(cnt_no_data_files) + " files")

    return ciyu_t, jieshi_t


if __name__ == '__main__':

    words, definitions = extract_data_from_files('../../../data/zd.eywedu.com/*.json')