from gensim.models import Word2Vec, KeyedVectors

filename = "word2vec.bin"
word2vec = KeyedVectors.load_word2vec_format(filename, binary = True)

while True:

	inputStr = input("Type some word(s): ")
	words = []

	if " " in inputStr:
		words = inputStr.split(" ")
		if "" in words:
			words.remove("")

	else:
		words.append(inputStr)

	sim_words = word2vec.wv.most_similar(positive = words)

	lst1 = []
	for w, s in sim_words:
		lst1.append(w)

	print("Suggestions:\n", lst1)