import nltk
import re
from nltk.corpus import stopwords
from gensim.models import Word2Vec, KeyedVectors
from gensim.test.utils import get_tmpfile

# Dataset cleaning

data = open('corpus.txt').read().lower()
data = re.sub('[^a-zA-Z]', ' ', data)
data = re.sub(r'\s+', ' ', data)

# Preparing the dataset

all_sentences = nltk.sent_tokenize(data)

all_words = [nltk.word_tokenize(sent) for sent in all_sentences]

# Removing Stop Words

#for i in range(len(all_words)):
	#all_words[i] = [w for w in all_words[i] if w not in stopwords.words('english')]

word2vec = Word2Vec(all_words, iter = 70, min_count = 2, compute_loss = True)

vocabulary = word2vec.wv.vocab

filename = "word2vec.bin"
word2vec.save_word2vec_format(filename, binary = True)

sim_words = word2vec.wv.most_similar(positive = ['attorney', 'election'])
print("Words similar to 'attorney for election': ", sim_words)

