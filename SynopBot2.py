import re
import sys
import time

import requests
import pandas as pd
import PyPDF4

from bs4 import BeautifulSoup, SoupStrainer
from collections import Counter
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from os import listdir, path
from os.path import isfile, join
from nltk.corpus import wordnet

stop_words = set(stopwords.words('english'))
normalizer = WordNetLemmatizer()
support_nouns = ['db', 'database', 'admin', 'tech', 'support', 'users', 'user', 'rights', 'access', 'sign', 'signin', 'sso', 'error', 'errors', 'setup']
services_nouns = ['report', 'reports', 'dashboard', 'budgets', 'drilldown', 'train', 'training', 'teach']

blog_url = 'https://synoptixsoftware.com/blog/page/'
e_learning_site = 'https://elearning.synoptixsoftware.com/all-courses'
file_path = '/Users/rramage/Desktop/SynML/SynoptixML/synopbot/doc/'
blogs = {}
elearning = []


def parse_blogs():
	global blogs
	links = parse_blog_links()
	for link in links:
		web_res = requests.get(link)
		print('parsing', link, web_res.status_code)
		if web_res.status_code == 200:
			web_content = web_res.content
			row_strainer = SoupStrainer('div', class_='main container py-5')
			soup = BeautifulSoup(web_content, 'html.parser', parse_only=row_strainer)
			blogs[soup.find('h1').getText()] = {'content': soup.getText(), 'url': link}


def parse_pdf():
	onlyFiles = [f for f in listdir(file_path) if isfile(join(file_path, f)) and f.endswith('.pdf')]
	for f in onlyFiles:
		file_obj = open(file_path + f, 'rb')
		reader = PyPDF4.PdfFileReader(file_obj, strict=False)
		for p in range(reader.getNumPages()):
			text = reader.getPage(p).extractText()
			if text.lower().startswith('document version history') or text.lower().startswith('version history'):
				break
			text = re.sub(r'\d+', ' ', text).strip()
			text = re.sub(r'\W+', ' ', text).strip()
			blogs[f] = {'content': text, 'url': f}


def parse_blog_links():
	blog_links = set([])
	count = 1
	web_res = requests.get(blog_url + str(count))
	while web_res.status_code != 404:
		print('finding links')
		web_content = web_res.content
		row_strainer = SoupStrainer('div', class_='entry-summary')
		soup = BeautifulSoup(web_content, 'html.parser', parse_only=row_strainer)
		for link in soup.select('.card-body a'):
			blog_links.add(link.attrs.get("href"))
		count += 1
		web_res = requests.get(blog_url + str(count))
	return blog_links


def parse_elearning():
	web_res = requests.get(e_learning_site)
	web_content = web_res.content
	row_strainer = SoupStrainer('div', class_=re.compile(r'ld-course-list-items*'))
	soup = BeautifulSoup(web_content, 'html.parser', parse_only=row_strainer)
	global elearning
	for cap in soup.select('.caption'):
		class_res = requests.get(cap.select(".ld_course_grid_button a")[0].attrs.get("href"))
		class_content = class_res.content
		class_strainer = SoupStrainer('div', class_=re.compile(r'ld-tab-content*'))
		class_soup = BeautifulSoup(class_content, 'html.parser', parse_only=class_strainer)
		course_descrip = ''
		for p in class_soup.select('p'):
			course_descrip += '<br>' + p.getText()
		blogs[cap.select(".entry-title")[0].getText()] = {'content':course_descrip, 'url':cap.select(".ld_course_grid_button a")[0].attrs.get("href")}


def get_part_of_speech(word):
	probable_part_of_speech = wordnet.synsets(word)
	pos_counts = Counter()
	pos_counts["n"] = len([item for item in probable_part_of_speech if item.pos() == "n"])
	pos_counts["v"] = len([item for item in probable_part_of_speech if item.pos() == "v"])
	pos_counts["a"] = len([item for item in probable_part_of_speech if item.pos() == "a"])
	pos_counts["r"] = len([item for item in probable_part_of_speech if item.pos() == "r"])
	most_likely_part_of_speech = pos_counts.most_common(1)[0][0]
	return most_likely_part_of_speech


def preprocess_text(text):
	cleaned = re.sub(r'\W+', ' ', text).lower()
	tokenized = word_tokenize(cleaned)
	global stop_words
	filtered = [w for w in tokenized if w not in stop_words]
	# for word in filtered:
		# print(word, get_part_of_speech(word))
	normalized = " ".join([normalizer.lemmatize(token, get_part_of_speech(token)) for token in filtered])
	return normalized


def classify_blogs():
	parse_pdf()
	parse_blogs()
	parse_elearning()
	processed_blogs = [preprocess_text(blog['content']) for blog in blogs.values()]

	# initialize and fit TfidfVectorizer
	vectorizer = TfidfVectorizer(norm=None, ngram_range=(2, 2))
	print('scoring')
	tfidf_scores = vectorizer.fit_transform(processed_blogs)

	# get vocabulary of terms
	feature_names = vectorizer.get_feature_names()
	columns = [c for c in blogs.keys()]
	# create pandas DataFrame with tf-idf scores
	try:
		url_ref = []
		for blog_k, blog_v in blogs.items():
			url_ref.append([blog_k, blog_v.get('url')])
		url_df = pd.DataFrame(url_ref, columns=['name', 'url'])
		df_tf_idf = pd.DataFrame(tfidf_scores.T.todense(), index=feature_names, columns=columns)
		return df_tf_idf, url_df
		#======================================================================
		#               Creates synonym pairing of n-grams
		#======================================================================
		# classified = dict()
		# for col in columns:
		# 	print(col)
		# 	subjects = [data for data in df_tf_idf[col].sort_values(ascending=False).iteritems()][:5]
		# 	extendedSubjects = []
		# 	for subject in subjects:
		# 		alter = dict()
		# 		split = subject[0].split(' ')
		# 		for word in split:
		# 			syno = set([])
		# 			for syn in wordnet.synsets(word):
		# 				for l in syn.lemmas():
		# 					syno.add(l.name().replace('_', ' '))
		# 			alter[word] = syno
		# 		for w in alter.get(split[0]):
		# 			extendedSubjects.extend([w + ' ' + w2 for w2 in alter[split[1]]])
		# 	classified[col] = extendedSubjects
		# 	print(subjects)
		# # print(classified)
		#======================================================================
	except:
		pass


if not path.exists('synStore.h5'):
	dataStore = pd.HDFStore('synStore.h5')
	data_out = classify_blogs()
	dataStore['support'] = data_out[0]
	dataStore['urls'] = data_out[1]

if len(sys.argv) == 2:
	userInput = sys.argv[1]
	vectorizer = TfidfVectorizer(norm=None, ngram_range=(1, 2))
	user_scores = vectorizer.fit_transform([preprocess_text(userInput)])
	user_feature_names = vectorizer.get_feature_names()
	#==============================================================
	#                   DEBUG: POS tagging
	#==============================================================
	# tagged_input = pos_tag(userInput.strip().split())
	# print('='*40)
	# for t in tagged_input:
	# 	print(t)
	# print('='*40)
	#==============================================================
	dataStore = pd.HDFStore('OLD_synStore.h5')
	df = dataStore['support']
	urls = dataStore['urls']
	dataStore.close()
	resp_dict = dict()
	classified = dict()
	e_learn = dict()
	response = 'Based on your question here is what I\'ve found that might be of use to you:\n'
	for col in df.columns:
		subjects = [data for data in df[col].sort_values(ascending=False).iteritems()][:5]
		classified[col] = subjects
	for seg in user_feature_names:
		rating = 0
		for k, v in classified.items():
			for subj in v:
				if subj[0].count(seg) > 0:
					# print(subj[0])
					out = urls[(urls.name.isin([k]))]
					url = out.iloc[0][1]
					if url.count('elearning') > 0:
						e_learn[seg] = url #+ ' : ' + str(rating)
					elif subj[1] > rating:
						rating = subj[1]
						resp_dict[seg] = url #+ ' : ' + str(rating)
				#============================================================
				#          Shows all matches, not just the highest
				#============================================================
				# if subj[0].count(seg) > 0:
				# 	rating = subj[1]
				# 	out = urls[(urls.name.isin([k]))]
				# 	resp_dict[seg +':'+out.iloc[0][1]] = str(rating)
				#============================================================

	for k, v in resp_dict.items():
		response += k + ' can be found at ' + ((file_path + v) if v.endswith('.pdf') else v) + '\n'
	if len(e_learn.keys()) > 0:
		response += '\nWe also have the following training opportunities you might be interested in\n'
		for k, v in e_learn.items():
			response += f'{k} at {v}\n'
	print(response)

