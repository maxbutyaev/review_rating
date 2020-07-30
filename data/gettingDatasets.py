import pandas as pd
import os, string
import re
from bs4 import BeautifulSoup
import nltk
import time
start_time = time.time()

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer,PorterStemmer

datadir = os.path.abspath(os.curdir) + '/'

#This directories are too heavy for uploading to github, you can take the files from https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
testnegdir = datadir + 'test/neg'
testposdir = datadir + 'test/pos'
trainnegdir = datadir + 'train/neg'
trainposdir = datadir + 'train/pos'

ps = PorterStemmer()
lemmatizer = WordNetLemmatizer()
stopwordsset = set(stopwords.words("english"))


def get_wordnet_pos(word):
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)


def tokenize(sentence):
    # Deleting html-tags and lowering text
    sentence = BeautifulSoup(sentence).get_text().lower()
    # Getting words from sentence
    words = re.sub('[^a-z]', ' ', sentence).split()
    # deleting unimportant words
    important_words = [word for word in words if word not in stopwordsset]
    # lemmatizing words
    lemmas = [lemmatizer.lemmatize(word, get_wordnet_pos(word)) for word in important_words]
    # stemming words to shorten the vocabulary
    tokens = [ps.stem(word) for word in lemmas]
    tokened_sentence = ' '.join(tokens)
    # getting sentence made of important lemmatized and stemmed words
    return (tokened_sentence)


def txttoDF(datafolder):
    tempx = 0
    # gives ready DataFrame with ids, marks, comments and tokened comments(lemmatized, stemmed, without HTML-tags and stopwords)
    dikt = {'ids': [], 'marks': [], 'comments': [], 'tokened_comments': []}
    files = [f for f in os.listdir(datafolder)]
    for file in files:
        with open(datafolder + '/' + file, encoding='utf-8') as tempfile:
            temptext = ''
            for line in tempfile:
                temptext = temptext + line
            dikt['comments'] += [temptext]
            dikt['tokened_comments'] += [tokenize(temptext)]
        tempid = int(str(file).split('_')[0])
        tempmark = int(str(file).split('_')[1].split('.')[0])
        dikt['ids'] += [tempid]
        if tempx % 500 == 0:
            print(tempx,"--- %s seconds ---" % (time.time() - start_time))
        tempx += 1
        dikt['marks'] += [tempmark]
    DF = pd.DataFrame(data=dikt)
    DF = DF.sort_values('ids')
    return (DF)


testneg = txttoDF(testnegdir)
testneg.to_csv(path_or_buf=datadir + '/' + 'testneg.csv', encoding='utf-8', sep=';', index=False)
testpos = txttoDF(testposdir)
testpos.to_csv(path_or_buf=datadir + '/' + 'testpos.csv', encoding='utf-8', sep=';', index=False)
trainpos = txttoDF(trainposdir)
trainpos.to_csv(path_or_buf=datadir + '/' + 'trainpos.csv', encoding='utf-8', sep=';', index=False)
trainneg = txttoDF(trainnegdir)
trainneg.to_csv(path_or_buf=datadir + '/' + 'trainneg.csv', encoding='utf-8', sep=';', index=False)
