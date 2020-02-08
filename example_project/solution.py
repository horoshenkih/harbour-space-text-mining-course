from argparse import ArgumentParser
from copy import deepcopy
import json
import random
import spacy
from rouge import Rouge
from tqdm import tqdm
import numpy as np
from collections import defaultdict
import functools
import operator
from gensim.models import Word2Vec
import streamlit as st
import requests
import bs4


nlp = spacy.load("en_core_web_lg", disable=["tagger", "ner", "parser"])
nlp.add_pipe(nlp.create_pipe('sentencizer'))


class BaselineSummarizer:
    def __init__(self, n_sentences=5):
        self._n_sentences = n_sentences

    def __call__(self, text):
        doc = nlp(text)
        sentences = list(doc.sents)
        idx = list(range(len(sentences)))
        random.shuffle(idx)
        return "\n".join([sentences[i].text for i in sorted(idx[:self._n_sentences])])


class Summarizer:
    def __init__(self, n_sentences=5):
        self._n_sentences = n_sentences

    def __call__(self, text):
        doc = nlp(text)
        sentences = [sent for sent in doc.sents if sent.has_vector]
        sentences_rank = sorted(
            range(len(sentences)),
            key=lambda i: doc.similarity(sentences[i]),
            reverse=True
        )
        i2s = {}
        for r in sentences_rank[:self._n_sentences]:
            i2s[r] = sentences[r]
        return "\n".join([sentences[i].text for i in sorted(i2s.keys())])


class TrainedSummarizer:
    def __init__(self, n_sentences=5, dim=100):
        self._n_sentences = n_sentences
        self._model = None
        self._dim = dim

    def fit(self, train, train_size=None):
        if train_size is not None:
            train = deepcopy(train)
            random.shuffle(train)
            train = train[:train_size]

        train_tokens = []
        for train_item in tqdm(train, desc="tokenize"):
            train_tokens.append([t.text for t in nlp(train_item)])

        self._model = Word2Vec(sentences=train_tokens, size=self._dim, iter=10)

    def average_vector(self, tokens):
        vectors = [self._model.wv[t.text] for t in tokens if t.text in self._model.wv]
        if vectors:
            return functools.reduce(operator.add, vectors)
        else:
            return np.zeros(self._dim)

    def __call__(self, text):
        doc = nlp(text)
        sentences = [sent for sent in doc.sents if sent.has_vector]
        doc_vector = self.average_vector(doc)
        sentences_rank = sorted(
            range(len(sentences)),
            key=lambda i: np.dot(doc_vector, self.average_vector(sentences[i])),
            reverse=True
        )
        i2s = {}
        for r in sentences_rank[:self._n_sentences]:
            i2s[r] = sentences[r]
        return "\n".join([sentences[i].text for i in sorted(i2s.keys())])


def compare_summarizers(data, summarizers):
    # construct rouge metric function ROUGE-1 F
    compute_rouge = Rouge(metrics=["rouge-1"], stats=["f"])

    def get_score(reference, hypothesis):
        """
        Compute ROUGE-1 F score
        :param reference: true summary
        :param hypothesis: predicted summary
        :return: the value of ROUGE-1 F
        """
        return compute_rouge.get_scores(hypothesis, reference)[0]["rouge-1"]["f"]

    # Compare summarizers on the part of the validation dataset.
    # Dataset is a list of dicts, each dict has two keys: "document" and "summary".
    validation = deepcopy(data["validation"])
    if args.validation_size is None:
        validation_size = len(validation)
    else:
        validation_size = args.validation_size
        # NB: always shuffle the data!
        random.shuffle(validation)

    # A document is a text of news articles separated by special token "|||||".
    # For proper sentence segmentation we need to clean up the data.
    def clean_document(text):
        return "\n".join(text.split("|||||"))

    print("Compute scores on the validation dataset")
    scores = defaultdict(list)

    for i in tqdm(range(validation_size)):
        document = clean_document(validation[i]["document"])
        true_summary = validation[i]["summary"]

        for summarizer_name, summarizer in summarizers.items():
            summary = summarizer(document)
            scores[summarizer_name].append(get_score(true_summary, summary))

    for summarizer_name in summarizers:
        print("Score of '{}' is {}".format(summarizer_name, np.mean(scores[summarizer_name])))


if __name__ == '__main__':
    p = ArgumentParser()
    p.add_argument("solution", type=int, choices=(1, 2, 3))
    p.add_argument("--multi-news-json", default="multi_news.json")
    p.add_argument("-s", "--seed", type=int, default=0, help="random seed")
    p.add_argument("-n", "--validation-size", type=int)
    p.add_argument("-t", "--train-size", type=int)
    args = p.parse_args()
    random.seed(args.seed)

    if args.solution == 3:
        # show demo
        summarizer = Summarizer(n_sentences=5)
        st.title("TechCrunch sentence summarization demo")
        url = st.text_input("TechCrunch URL", "")

        response = requests.get(url)
        soup = bs4.BeautifulSoup(response.text, "html.parser")
        items = soup.find("div", {"class": "article-content"}).findAll("p")

        raw_html = "\n".join(map(str, items))
        import re

        def cleanhtml(raw_html):
            cleanr = re.compile('<.*?>')
            cleantext = re.sub(cleanr, '', raw_html)
            return cleantext

        summary = summarizer(cleanhtml(raw_html))
        st.subheader('Summary')
        st.write(summary)

        st.subheader('Article')
        st.markdown(raw_html, unsafe_allow_html=True)
    else:
        # evaluate quality of summarizers

        # read the data from multi_news.json
        with open(args.multi_news_json) as f:
            print("Read multi news data from", args.multi_news_json)
            data = json.load(f)

        summarizers = dict()
        summarizers["baseline"] = BaselineSummarizer(n_sentences=5)
        summarizers["spaCy vectors"] = Summarizer(n_sentences=5)
        if args.solution == 2:
            print("Train summarizer")
            summarizers["trained vectors"] = TrainedSummarizer(n_sentences=5)
            summarizers["trained vectors"].fit([e["document"] for e in data["train"]], train_size=args.train_size)

        compare_summarizers(data, summarizers)
