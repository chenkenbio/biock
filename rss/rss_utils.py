#!/usr/bin/env python3

import argparse, os, sys, time
import requests, bs4
from bs4 import BeautifulSoup
from collections import OrderedDict
#import warnings, json, gzip

#import numpy as np
#import pandas as pd
#import matplotlib.pyplot as plt
#from scipy.stats import pearsonr, spearmanr


HEADERS = {'User-Agent' : 'Mozilla/5.0 (Windows; U; Windows NT 6.1; en-US; rv:1.9.1.6) Gecko/20091201 Firefox/3.5.6'} 

class Article(object):
    def __init__(self, title, link, doi, pub_date, authors):
        self.title = title
        self.link = link 
        self.doi = doi 
        self.pub_date = pub_date 
        self.authors = authors 
    def as_rss(self):
        rss_meta = "<item xmlns:rdf=\"{link}\"><title>{title}</title><link>{link}</link><pubDate>{pub_date}</pubDate><author>{authors}</author></item>".format(link=self.link, title=self.title, pub_date=self.pub_date, authors=self.authors)
        return rss_meta

def parse_oup(base_url="https://academic.oup.com/NAR/advance-articles", n_pages=2, headers=HEADERS, prefix="https://academic.oup.com"):
    """
    NAR: 2
    """
    articles = OrderedDict()
    for page in range(1, n_pages + 1):
        url = "{}?page={}".format(base_url, page)
        print("- {}\n  Parsing {}".format(time.asctime(), url), file=sys.stderr, flush=True)
        r = requests.get(url, headers=headers)
        soup = BeautifulSoup(r.text, "lxml")
        items = soup.find_all('div', class_='al-article-items')
        for item in items:
            title = item.find('h5', class_='al-title').a.text.strip()
            url = "https://academic.oup.com" + item.find('h5', class_='al-title').a.attrs['href'].strip()
            authors = '; '.join([a.text for a in item.find('div', class_='al-authors-list').select('a')])
            date = item.find('span', class_='sri-date al-pub-date').text
            doi = item.find('div', class_='al-citation-list').a.text
            if doi not in articles:
                articles[doi] = Article(title=title, link=url, doi=doi, pub_date=date, authors=authors)
    print("  {} articles fetched. {}\n\n".format(len(articles), time.asctime()), file=sys.stderr, flush=True)
    return articles

def generate_bioinfo(title="Bioinformatics Advance Access", out=sys.stderr):
    articles = parse_oup(base_url="https://academic.oup.com/bioinformatics/advance-articles", n_pages=4)
    print('<rss xmlns:prism="http://purl.org/rss/1.0/modules/prism/" version="2.0">', file=out)
    print('  <channel>', file=out)
    print('    <title>{}</title>'.format(title), file=out)
    print('    <link>http://academic.oup.com/bioinformatics</link>', file=out)
    print('    <description> </description>', file=out)
    print('    <language>en-us</language>', file=out)
    print('    <pubDate>{}</pubDate>'.format(articles[next(iter(articles))].pub_date), file=out)
    print('    <lastBuildDate>{}</lastBuildDate>'.format(time.asctime()), file=out)
    print('    <generator>Chen Ken</generator>', file=out)
    for item in articles:
        print("    " + articles[item].as_rss(), file=out)
    print('  </channel>', file=out)
    print('</rss>', file=out)


def generate_nar(title="Nucleic Acids Research Advance Access", out=sys.stderr):
    articles = parse_oup(base_url="https://academic.oup.com/NAR/advance-articles", n_pages=2)
    print('<rss xmlns:prism="http://purl.org/rss/1.0/modules/prism/" version="2.0">', file=out)
    print('  <channel>', file=out)
    print('    <title>{}</title>'.format(title), file=out)
    print('    <link>http://academic.oup.com/nar</link>', file=out)
    print('    <description> </description>', file=out)
    print('    <language>en-us</language>', file=out)
    print('    <pubDate>{}</pubDate>'.format(articles[next(iter(articles))].pub_date), file=out)
    print('    <lastBuildDate>{}</lastBuildDate>'.format(time.asctime()), file=out)
    print('    <generator>Chen Ken</generator>', file=out)
    for item in articles:
        print("    " + articles[item].as_rss(), file=out)
    print('  </channel>', file=out)
    print('</rss>', file=out)

def get_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('journal', choices=('NAR', 'nar', 'bioinformatics', 'BIOINFORMATICS'))
    p.add_argument('--out', required=True)
    p.add_argument('--time-interval', default=3600, type=int)
    p.add_argument('--seed', type=int, default=2020)
    return p.parse_args()


if __name__ == "__main__":
    args = get_args()
    print("# {}".format(args))
    while True:
        if args.journal.lower() == 'nar':
            with open(args.out, 'w') as out:
                generate_nar(out)
        elif args.journal.lower() == 'bioinformatics':
            with open(args.out, 'w') as out:
                generate_bioinfo(out)
        time.sleep(args.time_interval)

