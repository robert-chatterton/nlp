# feel free to include more imports as needed here
# these are the ones that we used for the base model
import numpy as np
import sys
from collections import Counter
import math
import nltk

"""
Robert Chatterton
NUID 001456770
hw3_sentiment.py
"""

"""
Cite your sources here:
Speech and Language Processing (3rd Ed.)
"""

"""
Implement your functions that are not methods of the Sentiment Analysis class here
"""
def generate_tuples_from_file(training_file_path):
    # given a file, generates the tuples of the form: (ID, document, label)
    tuples = []
    file = open(training_file_path, "r", encoding="utf8")
    for line in file:
        if len(line.strip()) == 0:
            continue
        s = line.split()
        tuples.append((s[0], ' '.join(ele for ele in s[1:len(s) - 1]), s[len(s) - 1]))
    file.close()
    return tuples

def precision(gold_labels, classified_labels, positive_label="1", negative_label="0"):
    # determines the precision of this model
    truePos, falsePos = 0, 0
    for l in range(len(classified_labels)):
        if classified_labels[l] == positive_label and gold_labels[l] == positive_label:
            truePos += 1
        elif classified_labels[l] == positive_label and gold_labels[l] == negative_label:
            falsePos += 1
    return truePos / (truePos + falsePos)

def recall(gold_labels, classified_labels, positive_label="1", negative_label="0"):
    # determines the recall of this model
    truePos, falseNeg = 0, 0
    for l in range(len(classified_labels)):
        if classified_labels[l] == positive_label and gold_labels[l] == positive_label:
            truePos += 1
        elif classified_labels[l] == negative_label and gold_labels[l] == positive_label:
            falseNeg += 1
    return truePos / (truePos + falseNeg)

def f1(gold_labels, classified_labels):
    # determines the f1 score of this model
    p = precision(gold_labels, classified_labels)
    r = recall(gold_labels, classified_labels)
    if p == 0 and r == 0:
        return 0
    return (2 * p * r) / (p + r)

"""
implement your SentimentAnalysis class here
"""
class SentimentAnalysis:
    
    def __init__(self):
        self.POS_LABEL = "1"
        self.NEG_LABEL = "0"
        self.labels = [self.NEG_LABEL, self.POS_LABEL]

    def train(self, examples):
        # bag of words with two classes: positive and negative
        self.bag = {}
        counts = {}
        # initialize dictionaries
        for l in self.labels:
            self.bag[l] = {}
            counts[l] = 0
        # for each example tuple, determine what its vocabulary is
        for e in examples:
            s_tokens = e[1].split()
            if e[2] == self.POS_LABEL:
                counts[self.POS_LABEL] += 1
                for s in s_tokens:
                    if s not in self.bag[self.POS_LABEL]:
                        self.bag[self.POS_LABEL][s] = 1
                    else:
                        self.bag[self.POS_LABEL][s] += 1
            elif e[2] == self.NEG_LABEL:
                counts[self.NEG_LABEL] += 1
                for s in s_tokens:
                    if s not in self.bag[self.NEG_LABEL]:
                        self.bag[self.NEG_LABEL][s] = 1
                    else:
                        self.bag[self.NEG_LABEL][s] += 1
        # nDoc -> number of documents to train on
        nDoc = counts[self.POS_LABEL] + counts[self.NEG_LABEL]
        # self.pc -> P(c) for both classes
        self.pc = {self.NEG_LABEL:(counts[self.NEG_LABEL] / nDoc), self.POS_LABEL:(counts[self.POS_LABEL] / nDoc)}
        # self.vocab -> vocabulary for this sentiment analyzer
        self.vocab = {}
        for label in self.bag:
            for word in self.bag[label]:
                if word not in self.vocab:
                    self.vocab[word] = self.bag[label][word]
                else:
                    self.vocab[word] += self.bag[label][word]

    def score(self, data):
        sent = data.split()
        hashy = {}
        for l in self.labels:
            # wc -> total word count
            wc = 0
            for w in self.bag[l]:
                wc += self.bag[l][w]
            hashy[l] = self.pc[l]
            for s in sent:
                # is this word in the vocabulary at all? if not, don't bother
                if s in self.vocab:
                    # is this word in the vocabulary for this class?
                    if s in self.bag[l]:
                        hashy[l] *= ((self.bag[l][s] + 1) / (wc + len(self.vocab)))
                    else:
                        hashy[l] *= (1 / (wc + len(self.vocab)))
        return hashy

    def classify(self, data):
        hashy = self.score(data)
        # take the greater of the two values and classify based on their key (label)
        if hashy[self.POS_LABEL] > hashy[self.NEG_LABEL]:
            return self.POS_LABEL
        else:
            return self.NEG_LABEL

    def featurize(self, data):
        ret = []
        for s in data.split():
            ret.append((s, True))
        return ret

    def __str__(self):
        return "\nNaive Bayes - bag-of-words baseline\n"


class SentimentAnalysisImproved:

    def __init__(self):
        self.POS_LABEL = "1"
        self.NEG_LABEL = "0"
        self.labels = [self.NEG_LABEL, self.POS_LABEL]
        self.stemmer = nltk.stem.snowball.SnowballStemmer("english")

    def train(self, examples):
        # bag of words with two classes: positive and negative
        self.bag = {}
        counts = {}
        # initialize dictionaries
        for l in self.labels:
            self.bag[l] = {}
            counts[l] = 0
        # for each example tuple, determine what its vocabulary is
        for e in examples:
            # case-fold
            # use nltk.word_tokenize() to improve document tokenization
            s_tokens = nltk.word_tokenize(e[1].lower())
            if e[2] == self.POS_LABEL:
                counts[self.POS_LABEL] += 1
                for s in s_tokens:
                    # use NLTK's snowball stemmer to better process tokens
                    s = self.stemmer.stem(s)
                    if s not in self.bag[self.POS_LABEL]:
                        self.bag[self.POS_LABEL][s] = 1
                    else:
                        self.bag[self.POS_LABEL][s] += 1
            elif e[2] == self.NEG_LABEL:
                counts[self.NEG_LABEL] += 1
                for s in s_tokens:
                    # use NLTK's snowball stemmer to better process tokens
                    s = self.stemmer.stem(s)
                    if s not in self.bag[self.NEG_LABEL]:
                        self.bag[self.NEG_LABEL][s] = 1
                    else:
                        self.bag[self.NEG_LABEL][s] += 1
        # nDoc -> number of documents to train on
        nDoc = counts[self.POS_LABEL] + counts[self.NEG_LABEL]
        # self.pc -> P(c) for both classes
        self.pc = {self.NEG_LABEL:(counts[self.NEG_LABEL] / nDoc), self.POS_LABEL:(counts[self.POS_LABEL] / nDoc)}
        # self.vocab -> vocabulary for this sentiment analyzer
        self.vocab = {}
        for label in self.bag:
            for word in self.bag[label]:
                if word not in self.vocab:
                    self.vocab[word] = self.bag[label][word]
                else:
                    self.vocab[word] += self.bag[label][word]

    def score(self, data):
        # tokenization done by nltk rather than just str.split()
        sent = nltk.word_tokenize(data)
        hashy = {}
        for l in self.labels:
            # wc -> total word count
            wc = 0
            for w in self.bag[l]:
                wc += self.bag[l][w]
            hashy[l] = np.log(self.pc[l])
            for s in sent:
                # use NLTK's snowball stemmer to better process tokens
                s = self.stemmer.stem(s)
                # is this word in the vocabulary at all? if not, don't bother
                if s in self.vocab:
                    # is this word in the vocabulary for this class
                    if s in self.bag[l]:
                        # adding the natural logs of the ratios seems to produce better results
                        hashy[l] += np.log((self.bag[l][s] + 1) / (wc + len(self.vocab)))
                    else:
                        hashy[l] += np.log(1 / (wc + len(self.vocab)))
        return hashy
    
    def classify(self, data):
        # first normalization of the data (case-folding)
        hashy = self.score(data.lower())
        # choose the label with a higher probability
        if hashy[self.POS_LABEL] > hashy[self.NEG_LABEL]:
            return self.POS_LABEL
        else:
            return self.NEG_LABEL

    def featurize(self, data):
        ret = []
        for s in nltk.word_tokenize(data.lower()):
            # use NLTK's snowball stemmer to better process tokens
            s = self.stemmer.stem(s)
            ret.append((s, True))
        return ret

    def __str__(self):
        return "\nNaive Bayes - bag-of-words improved\n"


    def describe_experiments(self):
        s = """
        Changes made:
        - case-folded data before classification and during training
        - used nltk's word_tokenize() for tokenization instead of str.split()
        - utilized nltk's SnowballStemmer in order to better process tokens
        - when scoring, calculated the natural log of the percentages and added them
        """
        return s


def main():

    training = sys.argv[1]
    testing = sys.argv[2]

    training_tups = generate_tuples_from_file(training)
    testing_tups = generate_tuples_from_file(testing)
    
    sa = SentimentAnalysis()
    print(sa)
    # do the things that you need to with your base class
    sa.train(training_tups)

    # report precision, recall, f1
    gold = []
    classified = []
    for t in testing_tups:
        gold.append(t[2])
        classified.append(sa.classify(t[1]))
    p = precision(gold, classified)
    r = recall(gold, classified)
    f = f1(gold, classified)
    print("precision score: \t", p)
    print("recall score: \t\t", r)
    print("f1 score: \t\t", f, "\n")
    
    improved = SentimentAnalysisImproved()
    print(improved)
    # do the things that you need to with your improved class
    improved.train(training_tups)
    
    # report final precision, recall, f1 (for your best model)
    goldI = []
    classifiedI = []
    for t in testing_tups:
        goldI.append(t[2])
        classifiedI.append(improved.classify(t[1]))
    pi = precision(goldI, classifiedI)
    ri = recall(goldI, classifiedI)
    fi = f1(goldI, classifiedI)
    print("precision score: \t", pi)
    print("recall score: \t\t", ri)
    print("f1 score: \t\t", fi, "\n")

    print(improved.describe_experiments())


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage:", "python hw3_sentiment.py training-file.txt testing-file.txt")
        sys.exit(1)

    main()