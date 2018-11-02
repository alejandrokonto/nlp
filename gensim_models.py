import gensim
from tokenization_n_vocabulary_extraction import TokenizeAndExtractVocabulary


# def sentences_generator(file, l_th, h_th):
#     language = file[:2]
#     with open(my_tokenizer.token_fl_dir + '/' + file, 'r', encoding='utf-8') as readfile:
#         cnt = 0
#         for line in readfile:
#             cnt += 1
#             if cnt % 100000 == 0:
#                 print("Processing line: " + str(cnt))
#
#             re_line = my_tokenizer.sentence_preprocessing(line, language, l_th, h_th)
#             if re_line:
#                 yield re_line
#             else:
#                 continue


class MySentences(object):
    def __init__(self, dirname, filename, l_th, h_th):
        self.dirname = dirname
        self.file = filename
        self.l_th = l_th
        self.h_th = h_th
        self.my_tokenizer = TokenizeAndExtractVocabulary("data", self.dirname, "vocabularies")

    def __iter__(self):
        language = self.file[:2]
        with open(self.dirname + '/' + self.file, 'r', encoding='utf-8') as readfile:

            print("Iteration is now beginning. This may take a while...")
            cnt = 0
            cnt_used_lines = 0
            for line in readfile:
                cnt += 1
                if cnt % 100000 == 0:
                    print("Processing line: " + str(cnt))

                re_line = self.my_tokenizer.sentence_preprocessing(line, language, self.l_th, self.h_th)
                if re_line:
                    cnt_used_lines += 1
                    yield re_line
                else:
                    continue
            print("Lines actually used by Word2Vec algorithm: " + str(cnt_used_lines))


# create tokenized data iterator
sentences_iterator = MySentences("tokenized_data", "zh__article_sentences_tokenized.txt", 4, 10000)

# model = gensim.models.Word2Vec(sentences=sentences_iterator,
#                                size=200, min_count=5, max_vocab_size=50000, workers=24, iter=10)

model = gensim.models.FastText(sentences=sentences_iterator, size=200, window=5, min_count=5, workers=8,
                               max_vocab_size=80000, min_n=2, max_n=2, iter=10)

model.save("gensim_fast_text_models/chinese_wiki_14")