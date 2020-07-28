import re

import nltk
import numpy as np
from bs4 import BeautifulSoup

from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer, PorterStemmer
import joblib
from translate import Translator

try:
  stopwordsset = set(stopwords.words("english"))
except:
  nltk.download('stopwords')
  nltk.download('wordnet')
  nltk.download('punkt')
  nltk.download('averaged_perceptron_tagger')
  stopwordsset = set(stopwords.words("english"))

from review_rating import settings

ps = PorterStemmer()
lemmatizer = WordNetLemmatizer()

data_dir = settings.BASE_DIR + '\\data\\'


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
    num_tokens = len(tokens)
    tokened_sentence = ' '.join(tokens)
    # getting sentence made of important lemmatized and stemmed words
    return tokened_sentence, num_tokens


def rate(text, language):
    translator_to_en = Translator(to_lang='en', from_lang=language)
    translator_from_en = Translator(from_lang='en', to_lang=language)
    comment = translator_to_en.translate(text)
    vec = joblib.load(data_dir + 'vectorizer.json')
    tokened_sentence, num_tokens = tokenize(comment)
    vec_comment = vec.transform([tokened_sentence]).astype(np.uint8).toarray()
    rfr = joblib.load(data_dir + 'JLmodel_1450_6.json')
    rating = round(rfr.predict(vec_comment)[0] * 9 + 1)
    comment_str = (translator_from_en.translate('Your comment:') + ' "%s"' % text)
    rate_str = (translator_from_en.translate('I suppose, you meant') + ' %i/10' % int(rating))
    if num_tokens<10:
        warn_str=translator_from_en.translate('Warning! Your comment is too short, rating may be inaccurate. Write a detailed movie review for more accuracy.')
    else:
        warn_str=''
    if rating >= 5:
        is_positive = True
    else:
        is_positive = False
    return comment_str, rate_str, warn_str, is_positive
