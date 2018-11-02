import glob
import re
import numpy as np
from bs4 import BeautifulSoup


class PreprocessWikipedia:

    def __init__(self, dir):
        # self.files = glob.glob(dir + "/*.xml")
        self.dir = dir


    @staticmethod
    def check_line_beginning(str_1, str_2):
        if len(str_2) < len(str_1):
            return False
        else:
            for i, ch in enumerate(str_1):
                if str_2[i] != ch:
                    return False

        return True

    @staticmethod
    def extract_attribute(att_name, string_):
        att_str = att_name + '="'
        start_i = string_.find(att_str)

        index = start_i + len(att_str)
        val = ""
        while True:
            if string_[index] == '"':
                break
            val += string_[index]
            index += 1

        return val

    def parse_xml_file(self, file = None, dump_lang="el", name_id = True, categories=False, redirects=False,
                        content=True, links=False, assoc_lang=[]):
        """
        Choose the content to be extracted
        :param dump_lang:
        :param name_id:
        :param categories:
        :param redirects:
        :param content:
        :param assoc_lang:
        :return:
        """
        article_names = []
        links_out = []
        contents = []

        if file:

            f = open(self.dir + '/' + file, "r")

            line = f.readline().strip()
            cnt = 0

            while line != "":

                # check for the first "article" tag
                if self.check_line_beginning("<article ", line):
                    article_name = [self.extract_attribute("name", line)]
                    cnt += 1

                    # keep track according to the number of articles parsed so far
                    if cnt % 1000 == 0:
                        print("Article no:" + str(cnt))

                # after finding the first "crosslanguage link" iterate over all languages to get the elements you need
                if self.check_line_beginning("<crosslanguage_link", line):
                    while self.check_line_beginning("<crosslanguage_link", line):
                        for lang in assoc_lang:
                            if lang == self.extract_attribute("language", line):
                                article_name.append(self.extract_attribute("name", line))
                        line = f.readline().strip()
                    article_names.append(article_name)

                # check if there is any number of out-links in the content
                if links:
                    if self.check_line_beginning("<links_out", line):
                        links_out.append(self.extract_attribute("name", line))

                # keep the content
                if self.check_line_beginning("<content", line):
                    content = ""
                    while True:
                        line = f.readline().strip()
                        if self.check_line_beginning("</content>", line):
                            break
                        else:
                            content += line

                    contents.append(content)

                line = f.readline().strip()

            f.close()

        return article_names, contents

    def extract_parallel_data_indices(self, main_list, languages_no):

        indices = []

        for index, element in enumerate(main_list):
            if len(element) == languages_no:
                indices.append(index)

        return indices

    def write_data_to_txt(self, dir, file, data):

        f = open(dir + "/" + file, "w")

        if type(data[0]) is not list:
            for data_ in data:
                f.write(data_ + "\n")
        else:
            for data_ in data:
                f.write("\t".join(data_) + "\n")

        f.close()

    def extract_parallel_data(self, dump_files, parallel_article_names, languages):

        lang_no = len(languages)
        parall_names_temp = [x[0] for x in parallel_article_names]
        articles_extracted = []
        for file in dump_files:
            with open(self.dir + '/' + file, 'r', encoding='UTF-8') as openfile:
                cnt = 0

                for line in openfile:

                    if self.check_line_beginning("<article ", line):
                        cnt += 1
                        if cnt % 10000 == 0:
                            print("Already processed " + str(cnt) + " articles")
                        article_name = self.extract_attribute("name", line)
                        for name in parall_names_temp:
                            if article_name == name:
                                articles_extracted.append(name)
                                break

        return articles_extracted

    def clean_article_contents(self, dump_files, languages):

        lang_index = 0
        for file in dump_files:
            writefile = open("data/" + languages[lang_index] + "__article_contents_cleaned.txt", "w", encoding='UTF-8')
            print("cleaning articles for language" + "[" + languages[lang_index] + "]")

            with open(self.dir + '/' + file, 'r', encoding='UTF-8') as openfile:


                record = False
                cleared_content = ""

                for line in openfile:
                    if self.check_line_beginning("<content", line):

                        record = True
                        continue

                    if record:
                        if self.check_line_beginning("</content>", line):
                            record = False
                            cleared_content = BeautifulSoup(cleared_content, "lxml").text
                            writefile.write(cleared_content + '\n')
                            cleared_content = ""
                            continue
                        else:
                            cleared_content += line

            writefile.close()
            lang_index += 1





if __name__ == "__main__":

    preprocess = PreprocessWikipedia("wiki_dumps")
    # names = []
    # names_file = open("data/el__article_names.txt", 'r', encoding='UTF-8')
    # line = names_file.readline().strip()
    # while line != "":
    #
    #     names_l = line.split('\t')
    #     names.append([names_l[1], names_l[2]])
    #
    #     line = names_file.readline().strip()
    # names_file.close()
    preprocess.clean_article_contents(dump_files=["elwiki-20140728-corpus.xml", "enwiki-20140707-corpus.xml",
                                                  "zhwiki-20140804-corpus.xml"], languages=["el", "en", "zh"])




