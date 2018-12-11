'''
Xu, Jian 2018.11.18
Remove stop words and etc.
'''
import csv
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, punkt
import re
from nltk import WordNetLemmatizer

def tokenize_subset(file_path):
	csv_file = open(file_path,'r')
	reader = csv.reader(csv_file)

	# return this
	tokens = []

	for i,line in enumerate(reader):
		if i == 0:
			continue

		filtered_description = tokenize_line(line)	
		
		for f in filtered_description:
			if f not in tokens:
				tokens.append(f)

	return tokens

def tokenize_line(line):
	# print ('Raw Input: ',line[1])
	description = line[1]
	# remove puntuation
	while description.find('-') >= 0:
		description = description.replace('-',' ')
	description_depunc = re.sub(r'[^\w\s]','',description)
	# print ('After Puctuation: ',description_depunc)

	# un-capitalize all letters
	description_decap = ''
	for l in description_depunc:
		description_decap += l.lower()
	# print ('After Regularization: ', description_decap)

	# remove stop words
	stop_words = set(stopwords.words('english'))
	word_tokens = word_tokenize(description_decap)
	filtered_description = [w for w in word_tokens if not w in stop_words]
	# print ('After Stop Words Remeval and Vectorization: ',filtered_description)
	
	wnl = WordNetLemmatizer()
	filtered_description = [wnl.lemmatize(t) for t in filtered_description]

	filtered_description = list(set(filtered_description))
	# print ('After lemmatization and Deduplication: ', filtered_description)

	return filtered_description