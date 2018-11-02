from nltk import sent_tokenize, word_tokenize
import jieba
from collections import Counter
import re


class TokenizeAndExtractVocabulary:

    def __init__(self, clear_data_dir, tokenized_file_dir, vocab_dir):
        self.read_dir = clear_data_dir
        self.token_fl_dir = tokenized_file_dir
        self.vocab_dir = vocab_dir
        # self.token_n_filt_fl_dir = tokenized_n_filtered_data
        jieba.initialize()

        # regex area
        # edw mporeis na allakseis ta regexes na entopizoun strings pou den einai h glwssa tou keimenou apla
        self.detect_en = re.compile("[a-zA-z]+")
        self.detect_el = re.compile("[α-ωΑ-ΩίϊΐόάέύϋΰήώΊΌΆΈΎΉΏ]+")
        self.detect_zh = re.compile("[\u4e00-\u9fff]+")

        # self.detect_undefined_words = re.compile("[0-9]*[a-zA-z]+|[^0-9]*[a-zA-z]+|[0-9]*[^0-9]*[a-zA-z]+|"
        #                                     "[^0-9]*[0-9]*[a-zA-z]+|"
        #                                     "|[0-9]*[α-ωΑ-ΩίϊΐόάέύϋΰήώΊΌΆΈΎΉΏ]+|[^0-9]*[α-ωΑ-ΩίϊΐόάέύϋΰήώΊΌΆΈΎΉΏ]+|"
        #                                     "[0-9]*[^0-9]*[α-ωΑ-ΩίϊΐόάέύϋΰήώΊΌΆΈΎΉΏ]+|"
        #                                     "[^0-9]*[0-9]*[α-ωΑ-ΩίϊΐόάέύϋΰήώΊΌΆΈΎΉΏ]+|")

    @staticmethod
    def sent_tokenize_zh(string):
        """
        Segment paragraph and long strings to sentences. For now i am just splitting on chinese full stop.
        :param string: the sentence to be segmented
        :return: a list of sentences
        """
        splitted_string = string.split("。")
        # while returning the splitted string ignore the last element of the string since it's an empty string
        return splitted_string[:-1]

    def tokenize_file(self, files):

        """

        :param files: files have a specific name structure. That is: xx__file_name.txt, where xx is the two letter
                      codename for a language
        :return:
        """

        for file in files:

            language = file[:2]

            # checking progress
            print('started tokenization of language: "' + language + '"')

            with open(self.token_fl_dir + '/' + file[:2] + '__article_sentences_tokenized.txt', 'w', encoding='UTF-8') as writefile:
                with open(self.read_dir + '/' + file, 'r', encoding='UTF-8') as readfile:
                    cnt = 0
                    for line in readfile.readlines():

                        line = line.strip()

                        # just ignore empty lines
                        if line == "":
                            continue

                        if cnt % 10000 == 0:
                            print('Language "' + language + '", article No.' + str(cnt))

                        if language in ["el", "en"]:
                            tokenization_lang = 'greek' if language == "el" else "english"
                            sentences = sent_tokenize(line.strip('\t'), language=tokenization_lang)
                            for s in sentences:
                                cnt += 1
                                writefile.write("\t".join(word_tokenize(s)) + "\n")
                        elif language in ["zh"]:
                            sentences = self.sent_tokenize_zh(line.strip('\t'))
                            for sentence in sentences:
                                cnt += 1
                                writefile.write("\t".join(jieba.cut(sentence)) + "\t。\n")
                        else:
                            print("Please change the name of the corpora file according to the noted instructions.")
                            return 0

            print("Total number of tokenized sentences before filtering is: " + str(cnt))

    # just a custom method/function for extracting vocabulary, later can delete
    def extract_vocabulary(self, tokenized_file, low_th=5, high_th=50, vocabulary_size=50000):

        lang = tokenized_file[:2]
        cnt = Counter()
        # check 进展
        i = 0

        with open(self.token_fl_dir + '/' + tokenized_file, 'r', encoding='utf-8') as readfile:
            with open(self.vocab_dir + '/' + lang + '__vocabulary_tokens.txt', 'w', encoding='utf-8') as writefile:

                for line in readfile.readlines():
                    if i % 10000 == 0:
                        print("sentence No. " + str(i))

                    line_tokens = line.strip().split('\t')
                    length = len(line_tokens)

                    if low_th < length < high_th:
                        for token in line_tokens:
                            cnt[token] += 1

                    i += 1

                print("In total found: " + str(len(list(cnt.elements()))) + " tokens")

                for token_tuple in cnt.most_common(vocabulary_size):

                    token, _ = token_tuple
                    writefile.write(token + '\n')

    def sentence_preprocessing(self, sentence, language, l_th, h_th):

        sentence_tokens = sentence.strip().split('\t')

        if not self.filter_out_sentences(sentence_tokens, l_th, h_th):
            return None
        else:
            re_sentence = []
            for word in sentence_tokens:
                if word.isdigit():
                    re_sentence.extend(self.transform_digits(word, language).split())
                else:
                    if language == 'el':
                        if self.detect_en.match(word):
                            re_sentence.append("<ENGLISH_TERM>")
                        # elif self.detect_undefined_words.match(word):
                        #     re_sentence.append("<UNDEFINED_WORD>")
                        else:
                            re_sentence.append(word.lower())
                    elif language == 'zh':
                        if self.detect_en.match(word):
                            re_sentence.append("<ENGLISH_TERM>")
                        # elif self.detect_undefined_words.match(word):
                        #     re_sentence.append("<UNDEFINED_WORD>")
                        else:
                            re_sentence.append(word.lower())
                    else:
                        # else is english language
                        # if self.detect_undefined_words.match(word):
                        #     re_sentence.append("<UNDEFINED_WORD>")
                        # else:
                        re_sentence.append(word.lower())

            # before returning the tokenized sentence, check if undefined or foreign content is more thatn 40%
            # of the total sentence. If it is so, just return empty list.
            length, trash = 0, 0
            for word in re_sentence:
                length += 1
                if word == "<UNDEFINED_WORD>" or word == "<ENGLISH_TERM>":
                    trash += 1

            return re_sentence if trash / length < 0.5 else None

    @staticmethod
    def transform_digits(num, language):
        en_dict = {'0': 'zero', '1': 'one', '2': 'two', '3': 'three', '4': 'four', '5': 'five', '6': 'six',
                   '7': 'seven', '8': 'eight', '9': 'nine'}
        el_dict = {'0': 'μηδέν', '1': 'ένα', '2': 'δύο', '3': 'τρία', '4': 'τέσσερα', '5': 'πέντε', '6': 'έξι',
                   '7': 'επτά', '8': 'οκτώ', '9': 'εννέα'}
        zh_dict = {'0': '零', '1': '一', '2': '二', '3': '三', '4': '四', '5': '五', '6': '六', '7': '七', '8': '八', '9': '九'}

        num_string = ""
        try:
            for ch in num:
                if language == 'el':
                    num_string += el_dict[ch] + ' '
                elif language == 'en':
                    num_string += en_dict[ch] + ' '
                else:
                    num_string += zh_dict[ch] + ' '
        except KeyError:
            num_string = "<UNDEFINED_WORD>"

        return num_string.strip(' ')

    @staticmethod
    def filter_out_sentences(sentence, l_th, h_th):

        length = len(sentence)

        if l_th < length < h_th:
            return sentence
        else:
            return None

    @staticmethod
    def count_sentences(dir, file):

        with open(dir + "/" + file, 'r', encoding='utf-8') as readfile:
            cnt = 0
            for _ in readfile:
                cnt += 1
                if cnt % 100000 == 0:
                    print("sentence no: " + str(cnt))

            print('total sentences number: ' + str(cnt))

    @staticmethod
    def count_tokens(dir, files):

        for file in files:
            language = file[:2]
            cnt = 0
            with open(dir + "/" + file, 'r', encoding='utf-8') as readfile:

                for line in readfile:
                    tokens = line.strip().split('\t')
                    cnt += len(tokens)

            print("Total tokens for language " + language + ": " + str(cnt))


if __name__ == "__main__":
    custom_tokenizer = TokenizeAndExtractVocabulary("data", "tokenized_data", "vocabularies")
    # custom_tokenizer.extract_vocabulary("zh__article_sentences_tokenized.txt", vocabulary_size=100000)
    # custom_tokenizer.tokenize_file(["el__article_contents_cleaned.txt"])
    custom_tokenizer.count_tokens("tokenized_data", ["el__article_sentences_tokenized.txt",
                                                     "en__article_sentences_tokenized.txt",
                                                     "zh__article_sentences_tokenized.txt"])
