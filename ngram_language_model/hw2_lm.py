import matplotlib.pyplot as plt
import sys
import numpy as np
import random as rd

"""
Robert Chatterton
NUID 001456770
"""

# Feel free to implement helper functions

class LanguageModel:
    # constants to define pseudo-word tokens
    # access via self.UNK, for instance
    UNK = "<UNK>"
    SENT_BEGIN = "<s>"
    SENT_END = "</s>"

    def __init__(self, n_gram, is_laplace_smoothing):
        """Initializes an untrained LanguageModel
        Parameters:
          n_gram (int): the n-gram order of the language model to create
          is_laplace_smoothing (bool): whether or not to use Laplace smoothing
        """
        self.n = n_gram
        self.laplace = is_laplace_smoothing
        self.tokens = []
        self.sentences = []
        self.vocab = {self.UNK:0}
        self.ngrams = {}
        self.possibilities = {}

    def train(self, training_file_path):
        """Trains the language model on the given data. Assumes that the given data
        has tokens that are white-space separated, has one sentence per line, and
        that the sentences begin with <s> and end with </s>
        Parameters:
          training_file_path (str): the location of the training data to read
        Returns:
        None
        """
        path = training_file_path
        f = open(path, "r")
        for line in f:
            self.sentences.append(line)
        f.close()
       
        # create tokens
        for s in self.sentences:
            self.tokens = self.tokens + s.split()
        
        # create vocabulary in the form 'token:occurances'
        for w in self.tokens:
            if w in self.vocab:
                self.vocab[w] += 1
            else:
                self.vocab[w] = 1

        # turn tokens with count 1 into UNK
        rmList = []
        for w in self.vocab:
            if self.vocab[w] == 1:
                self.vocab[self.UNK] += self.vocab[w]
                rmList.append(w)
        for key in rmList:
            self.vocab.pop(key)
        if self.vocab[self.UNK] == 0:
            self.vocab.pop(self.UNK)
        
        # create n-grams, assign them to self.ngrams
        grams = []
        # create list of ngrams
        for s in self.sentences:
            words = s.split()
            wordsUNK = []
            for w in words:
                if not w in self.vocab:
                    w = self.UNK
                wordsUNK.append(w)
            grams += self.makeNGrams(wordsUNK)
        # create dict of ngrams
        for g in grams:
            if g in self.ngrams:
                self.ngrams[g] += 1
            else:
                self.ngrams[g] = 1
        # generate probability lists
        self.findPossibilities(grams)

    def score(self, sentence):
        """Calculates the probability score for a given string representing a single sentence.
        Parameters:
          sentence (str): a sentence with tokens separated by whitespace to calculate the score of
      
        Returns:
          float: the probability value of the given string for this model
        """
        split = sentence.split()
        newSentence = []
        for w in split:
            if not w in self.vocab:
                w = self.UNK
            newSentence.append(w)
        sn = self.makeNGrams(newSentence)
        _prob = 1
        for n_ns in sn:
            if n_ns in self.ngrams:
                wc = self.ngrams[n_ns]
            else:
                wc = 0
            if self.n == 1:
                l = len(self.tokens)
            else:
                n = n_ns.split()
                if n[len(n) - self.n] in self.vocab:
                    l = self.vocab[n[len(n) - self.n]]
                else:
                    l = self.vocab[self.UNK]
            _prob *= self.prob(wc, l, self.laplace)
        return _prob
    
    def makeNGrams(self, words):
        # creates n-grams based off of each sentence, provided as a list of words. 
        # Returns a list of words.
        grams = []
        for w in range(len(words[:len(words) - self.n + 1])):
            g = []
            for i in range(self.n):
                g.append(words[w + i])
            w = ""
            if len(g) == 1:
                w = g[0]
            else:
                w = g[0]
                for i in range(len(g))[1:]:
                    w = w + " " + g[i]
            grams.append(w)
        return grams
    
    def prob(self, wc, corpusLength, lap):
        # calculates probability based off of the given word count, corpus length, and whether
        # or not laplace smoothing is involved.
        if lap:
            V = len(self.vocab)
            #print(V)
            return float((wc + 1) / (corpusLength + V))
        else:
            return float(wc / corpusLength)
        
    def generate_sentence(self):
        """Generates a single sentence from a trained language model using the Shannon technique.
      
        Returns:
          str: the generated sentence
        """
        sent = [self.SENT_BEGIN]
        for i in range(self.n - 2):
            sent.append(self.SENT_BEGIN)
        
        prev = self.SENT_BEGIN
        while True:
            if self.n == 1:
                word = rd.choice(list(self.possibilities))
            else:
                word = rd.choice(self.possibilities[prev])
                prev = word
            sent.append(word) 
            if word == self.SENT_END:
                break
        
        if self.n - 2 > 0: 
            for i in range(self.n - 2):
                sent.append(self.SENT_END)
                
        sentence = sent[0]
        for s in sent[1:]:
            sentence += " " + s
        return sentence

    def generate(self, n):
        """Generates n sentences from a trained language model using the Shannon technique.
        Parameters:
          n (int): the number of sentences to generate
      
        Returns:
          list: a list containing strings, one per generated sentence
        """
        strings = []
        for i in range(n):
            strings.append(self.generate_sentence())
        return strings
    
    def findPossibilities(self, ngrams):
        for ng in ngrams:
            words = ng.split()
            if self.n == 1:
                if words[0] not in self.possibilities and not words[0] == self.SENT_BEGIN:
                    self.possibilities[words[0]] = ""
            else:
                if words[0] not in self.possibilities:
                    self.possibilities[words[0]] = [words[1]]
                else:
                    self.possibilities[words[0]] = self.possibilities[words[0]] + [words[1]]
    
    def getScores(self, path):
        scores = []
        sent = []
        # get testing file data
        f = open(path, "r")
        for line in f:
            sent.append(line)
        f.close()
        # get scores of each sentence
        for s in sent:
            scores.append(self.score(s))
        return scores
    
    def makeGraphs(self, testpath1, testpath2, filename):
        testScores = self.getScores(testpath1)
        myTestScores = self.getScores(testpath2)
        overallMin = min(testScores + myTestScores)
        series_to_plot = [testScores, myTestScores]
        min_exponent = np.floor(np.log10(np.abs(overallMin)))
        plt.hist(series_to_plot, bins=np.logspace(min_exponent, 0), label=[testpath1, testpath2], stacked=True)
        plt.title("Count of sentences with a given probability for %i-grams" % self.n)
        plt.xlabel("Probability")
        plt.ylabel("Count")
        plt.xscale('log')
        plt.legend()
        plt.savefig(filename, bbox_inches="tight")

def main():
    training_path = sys.argv[1]
    testing_path1 = sys.argv[2]
    testing_path2 = sys.argv[3]
    
    lm = LanguageModel(1, True)
    lm.train(training_path)
    lm2 = LanguageModel(2, True)
    lm2.train(training_path)
    sents = lm.generate(50)
    sents2 = lm2.generate(50)
    # if you want to see the created sentences, uncomment lines below
#     print("\nUNIGRAM GENERATIONS:\n")
#     for s in sents:
#         print(s)
#     print("\nBIGRAM GENERATIONS:\n")
#     for s in sents2:
#         print(s)
        
    # make graphs
    # in order for the bigram language model histogram to generate without overlap,
    # it must be run by itself (i.e. comment out line 256)
    print("Creating unigram language model histogram...")
    lm.makeGraphs(testing_path1, testing_path2, "hw2-unigram-histogram.pdf")
    print("Unigram language model histogram created.")
    print("Creating bigram language model histogram...")
    lm2.makeGraphs(testing_path1, testing_path2, "hw2-bigram-histogram.pdf")
    print("Bigram language model histogram created.")
    
if __name__ == '__main__':
    
    # make sure that they've passed the correct number of command line arguments
    if len(sys.argv) != 4:
        print("Usage:", "python hw2_lm.py training_file.txt textingfile1.txt textingfile2.txt")
        sys.exit(1)

    main()