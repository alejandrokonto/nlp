import re
from bs4 import BeautifulSoup
import glob
import os
import numpy as np
import json

ciyu_re = re.compile(" *【词语】 *")
jieshi_re = re.compile(" *【解释】 *")
list_re = re.compile("([1-9]\.*)+")
list_circled_nums = "①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳㉑㉒㉓㉔㉕㉖㉗㉘㉙㉚㉛㉜㉝㉞㉟㊱㊲㊳㊴㊵㊶㊷㊸㊹㊺㊻㊼㊽㊾㊿"
ru_structure = re.compile("如：| 如:")


def extract_subsections(html_doc):

    # These are holding the words to be defined
    h2_s = html_doc.find_all('h2', 'f14')

    # These are holding the definitions
    p_s = html_doc.find_all('p', 'f14')

    return h2_s, p_s


def extract_word(h2_els):

    ciyu = ''
    for el in h2_els:
        raw_string = str(el.string)
        result = ciyu_re.search(raw_string)
        if result:
            s_i, e_i = result.span()
            ciyu = raw_string[:s_i] + raw_string[e_i:]
            ciyu = ciyu.strip()
        else:
            continue

    return ciyu


def extract_definition(p_els):

    defs = []
    for p in p_els:
        # remove the hidden <span> element. If this element doesn't exist, then, it's not a definition <p>
        if not p.find('span'):
            continue
        p.span.extract()
        b_el = p.find('br')

        # remove the <br> element if there
        if b_el:
            p.br.extract()
        # hold the remaining contents of the <p> element
        p_contents = p.contents

        for content in p_contents:

            # check if it is the part that contains the definitions
            result = jieshi_re.search(content)

            if result:
                _, e_i = result.span()
                # jieshi holds the definitions as string (or navigable string, check it)
                jieshi = content[e_i:]
                # receive each sub-definition which is merged with it's examples
                defs = sub_definitions(str(jieshi))

            else:
                continue

    return defs


def separate_defs_from_examps(whole_defs_list, ci):

    defs, exmps = [], []

    for itemlist in whole_defs_list:
        # first see if there is literature notation or reference in the whole definition. If that's the case, discard
        if "《" in itemlist:
            continue
        # split into defintion 'n' example (if there are examples)
        if ru_structure.search(itemlist):
            indices = []
            for match in ru_structure.finditer(itemlist):
                indices.append(match.span())
            # the first pair of indices gives us the the definition (on left) and the rest of examples (on the right)
            s_i, e_i = indices[0]
            defs.append(itemlist[:s_i])
            if len(indices) == 1:
                exmps.append([itemlist[e_i:]])
            else:
                s_p, e_p = s_i, e_i
                exmps_ = []
                for spans in indices[1:]:
                    s_n, e_n = spans
                    exmps_.append(itemlist[e_p:s_n])
                    s_p, e_p = spans
                exmps_.append(itemlist[e_p:])
                exmps.append(exmps_)

        elif "：" in itemlist:
            # for now, if there are more than one ":" on definition i won't handle them
            components = itemlist.split("：")
            if len(components) > 2:
                return None, None
            # else continue normally
            def_, exmp_ = itemlist.split("：")
            defs.append(def_)
            # split example into subexamples, if they exist
            exmps_ = []
            if "｜" in exmp_:
                exmps_ = exmp_.split("｜")
            elif "ㄧ" in exmp_:
                exmps_ = exmp_.split("ㄧ")
            elif "◇" in exmp_:
                exmps_ = exmp_.split("◇")
            else:
                exmps_ = [exmp_]
            exmps.append([ex.replace("～", ci) for ex in exmps_])
        else:
            defs.append(itemlist)
            exmps.append([])

    return defs, exmps


def sub_definitions(raw_text):
    # the extracted sub definitions list
    sub_defs = []

    # listed definitions under the form 1.  2. ...
    result = list_re.findall(raw_text)

    if result:
        matches_iter = list_re.finditer(raw_text)
        spans = []

        for match in matches_iter:
            spans.append(match.span())

        prev_span = None

        for i, span in enumerate(spans):
            if not prev_span:
                prev_span = span
                continue
            else:
                _, e_p = prev_span
                s_n, _ = span
                prev_span = span

                sub_defs.append(raw_text[e_p:s_n])

        # add the last definition (after the end of the last list element)
        _, e_p = prev_span
        sub_defs.append(raw_text[e_p:])
    elif circled_num_list(raw_text):
        sub_defs = circled_num_list_extract_elements(raw_text)
    else:
        sub_defs.append(raw_text)

    return sub_defs


def circled_num_list_extract_elements(raw_text):

    # The elements of the list
    elements = []

    indices = []
    for num in list_circled_nums:
        index = raw_text.find(num)
        if index != -1:
            indices.append(index)
        else:
            break

    prev = None
    for i in indices:
        if prev == None:
            prev = i
        else:
            elements.append(raw_text[prev + 1: i])
            prev = i

    # include the last one
    # if prev is STILL None, this means that some corrupted data occured, reject the input, return none
    if not prev:
        return None
    else:
        elements.append(raw_text[prev + 1:])
        return elements


def circled_num_list(raw_text):

    for ch in raw_text:
        if ch in list_circled_nums:
            return True

    return False


def extract_data(files):

    ciyus, definitions, examples, chengyus = [], [], [], []
    rejected_words = []
    processed_files = []
    cnt, no_def_cnt, rejected_cnt = 0, 0, 0
    total_files = len(files)

    for file_ in files:
        cnt += 1
        with open(file_, 'r', encoding='utf-8') as file:
            html_text = file.read()

        html_doc = BeautifulSoup(html_text, 'html.parser')

        h2s, ps = extract_subsections(html_doc)

        # print(os.path.basename(file_))
        # check if the necessary html subsections exist or not
        if not h2s and not ps:
            no_def_cnt += 1
            print("Not found in: " + os.path.basename(file_))

        # the word we are extracting info about
        ciyu = extract_word(h2s)
        # first check if it is a possible chengyu. If it is not, proceed with word definition extraction
        if len(ciyu) == 4:
            chengyus.append(ciyu)
            # and keep a record for the file
            processed_files.append(os.path.basename(file_))
            # and skip the rest of the process
            continue
        # else continue processing

        # here we get a list with all of the definition of the word WITH their merged examples
        defs_n_exmps = extract_definition(ps)
        if not defs_n_exmps:
            rejected_words.append(ciyu)
            rejected_cnt += 1
            print("This file cannot be processed for now")
            # and keep a record for the file
            processed_files.append(os.path.basename(file_))
            # and skip the rest of the process
            continue

        # proceed with partioning of strings to definition and example
        defs, exmps = separate_defs_from_examps(defs_n_exmps, ciyu)
        if defs is None and exmps is None:
            rejected_words.append(ciyu)
            rejected_cnt += 1
            print("This file cannot be processed for now")
            # and keep a record for the file
            processed_files.append(os.path.basename(file_))
            # and skip the rest of the process
            continue

        ciyus.append(ciyu)
        definitions.append(defs)
        examples.append(exmps)

        # keep a record for processed files
        processed_files.append(os.path.basename(file_))

        if cnt % 1000 == 0:
            print("Already chuli-ed " + str(cnt) + " from " + str(total_files) + " files. ( No definition: " + str(no_def_cnt) + " No able to process: " + str(rejected_cnt) + ")")

    return ciyus, definitions, examples, chengyus, rejected_words


if __name__ == '__main__':
    print("Loading non-processed files")
    files = glob.glob('../../../data/aies.cn/word_pages/*.txt')

    # to test one file use this
    print("Start processing")
    # files = ['../../../data/aies.cn/word_pages/id=MTQ3OTMy.txt', '../../../data/aies.cn/word_pages/id=MjA5MDk4.txt',
    #          '../../../data/aies.cn/word_pages/id=MTUyOTMy.txt']
    cis, defs, exs, chengyu, rejected_cis = extract_data(files)

    # clear cis from rejected_cis. Now, the lists "final_cis", "defs", "exs" are all aligned.
    final_cis = []
    for ci in cis:
        if ci not in rejected_cis:
            final_cis.append(ci)

    # write words data to json file
    data = []
    for ci, definitions, examples in zip(final_cis, defs, exs):
        data.append({'word': ci, 'definitions': definitions, 'examples':examples})

    with open('../../../data/aies.cn/extracted_data/words_definitions.json', 'w') as file:
        json.dump(data, file, ensure_ascii=False)

    # write chengyu's as npy file
    np.save('../../../data/aies.cn/extracted_data/chengyu_no_def.npy', chengyu)

    # if you want to load words and their definitions run the following script
    # with open(file, encoding='utf-8') as f:
    #     data = json.load(f)
