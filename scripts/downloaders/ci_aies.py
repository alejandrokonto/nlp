"""

author: pwxcoo
date: 2018-08-02
description: 多线程抓取下载词语并保存

"""

import requests, csv
from bs4 import BeautifulSoup
import glob
import time
import numpy as np
from multiprocessing.dummy import Pool as ThreadPool
import os


def downloader(url):
    """
    下载词语并保存
    """
    res = []
    try:
        response = requests.get(url)

        if response.status_code != 200:
            print(f'{url} is failed!')
            return len(res)

        print(f'{url} is parsing')
        html= response.content.decode(encoding='utf-8')

    except Exception as e:
        with open('../data/error.csv', mode='a+', encoding='utf-8', newline='') as error_file:
            csv.writer(error_file).writerows([1, e, url])
        print(f'{url} is failed! {e}')

    with open('../data/aies_dict_pinyin_page.html', mode='a+', encoding='utf-8', newline='') as txt_file:
        txt_file.write(html)

    return len(res)


def download_n_return_html(base_url, href):
    res = []
    try:
        url = base_url + href
        response = requests.get(url)

        if response.status_code != 200:
            print(f'{url} is failed!')
            return len(res)

        print(f'{url} is parsing')
        html = response.content.decode(encoding='utf-8')
    except Exception as e:
        with open('../data/error.csv', mode='a+', encoding='utf-8', newline='') as error_file:
            csv.writer(error_file).writerows([1, e, url])
        print(f'{url} is failed! {e}')

    return html


def download_n_write_txt(base_url, href):
    res = []
    url = ''
    try:
        url = base_url + href
        print(url)
        response = requests.get(url)

        if response.status_code != 200:
            print(f'{url} is failed!')
            return len(res)

        print(f'{url} is parsing')
        html = response.content.decode(encoding='utf-8', errors='ignore')
    except Exception as e:
        with open('../data/error.csv', mode='a+', encoding='utf-8', newline='') as error_file:
            csv.writer(error_file).writerows([1, e, url])
        print(f'{url} is failed! {e}')

    with open('../data/aies.cn/word_pages/' + href[1:] + '.txt', mode='w', encoding='utf-8', newline='') as txt_file:
        txt_file.write(html)

    return html


def extract_pinyin_list_links(file_):
    html_txt = ""
    with open("../data/"+file_, 'r', encoding='utf-8') as file:
        for line in file.readlines():
            html_txt += line

    html = BeautifulSoup(html_txt, 'html.parser')
    pinyin_list = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
                   'U', 'V', 'W', 'X', 'Y', 'Z']
    hrefs = []
    for pinyin in pinyin_list:
        pinyin_block = html.find(id=pinyin)
        a_elements = pinyin_block.find_all('a') if pinyin_block else []
        hrefs += [a.get('href') for a in a_elements]

    return hrefs


def extract_char_list_links(dir):
    files = glob.glob('../data' + dir + "/*.txt")
    hrefs = []
    for file in files:
        print("Processing file: " + file)
        html_txt = ""
        with open("../data/" + file, 'r', encoding='utf-8') as file_:
            for line in file_.readlines():
                html_txt += line

        html = BeautifulSoup(html_txt, 'html.parser')

        a_elements = html.find_all('a')
        if a_elements:
            a_elements_filtered = []
            for a in a_elements:
                if a.get('target'):
                    a_elements_filtered.append(a)

            hrefs += [a.get('href') for a in a_elements_filtered]

    return hrefs


def extract_word_list_links(dir):
    files = glob.glob('../data' + dir + "/*.txt")
    hrefs=[]
    for file in files:
        print("Processing file: " + file)
        html_txt = ""
        with open("../data/" + file, 'r', encoding='utf-8') as file_:
            for line in file_.readlines():
                html_txt += line

        html = BeautifulSoup(html_txt, 'html.parser')
        div_elements = html.find_all('div', 'more')
        if div_elements:
            for div_el in div_elements:
                for a_el in div_el.contents:
                    if "查看全部" in a_el.string:
                        hrefs.append(a_el.get('href'))
                        print(a_el.get('href'))

    return hrefs


if __name__ == '__main__':

    page_hrefs = np.load('../data/word_links.npy')
    pages_to_download = len(page_hrefs)

    print('start downloading WORD pages')
    base_url = 'http://cidian.aies.cn/'

    processed_links_files = glob.glob('../data/aies.cn/word_pages/*.txt')
    processed_links = ["?" + os.path.basename(file_dir)[:-4] for file_dir in processed_links_files]

    cnt = 0
    for href in page_hrefs:
        cnt += 1
        print("Checking page " + str(cnt) + " of " + str(pages_to_download) + "...")
        if href in processed_links:
            print("Already processed..")
            continue
        else:
            time.sleep(np.random.random())
            download_n_write_txt(base_url=base_url, href=href)
            print("Successfully downloaded and written")