from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from bs4 import BeautifulSoup, SoupStrainer
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from os import path
import os.path
import random
import requests
import re
import csv
import sys

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
support_nouns = ['db', 'database', 'admin', 'tech', 'support', 'users', 'user', 'rights', 'access', 'sign', 'signin', 'sso', 'error', 'errors', 'setup']
services_nouns = ['report', 'reports', 'dashboard', 'budgets', 'drilldown', 'train', 'training', 'teach']
training_doc = '/Users/rramage/Desktop/SynML/SynoptixML/synopbot/doc/helptraining.html'
forum_csv = '/Users/rramage/Desktop/SynML/SynoptixML/synopbot/doc/forum_2.csv'
e_learning_site = 'https://elearning.synoptixsoftware.com/all-courses'
reg = re.compile(r'(<[/a-zA-Z]+>)')
elearning = []


def get_joke():
	with open('/Users/rramage/Desktop/SynML/SynoptixML/synopbot/doc/jokes.txt', 'r') as file:
		lines = file.readlines()
	return lines[random.randrange(len(lines))]


def get_responses():
	if path.exists('/Users/rramage/Desktop/SynML/SynoptixML/synopbot/doc/responses.csv'):
		responses = []
		lower_responses = []
		with open('/Users/rramage/Desktop/SynML/SynoptixML/synopbot/doc/responses.csv') as csv_file:
			reader = csv.reader(csv_file, delimiter=',')
			for row in reader:
				responses.append(row[1])
		with open('/Users/rramage/Desktop/SynML/SynoptixML/synopbot/doc/lower_responses.csv') as csv_file:
			reader = csv.reader(csv_file, delimiter=',')
			for row in reader:
				lower_responses.append(row[1])
		return responses, lower_responses
	else:
		parsing = parse_html()
		with open('/Users/rramage/Desktop/SynML/SynoptixML/synopbot/doc/responses.csv', 'w') as response_file:
			response_writer = csv.writer(response_file, delimiter=',', quotechar='"')
			for i in range(len(parsing[0])):
				response_writer.writerow([i, parsing[0][i]])
		with open('/Users/rramage/Desktop/SynML/SynoptixML/synopbot/doc/lower_responses.csv', 'w') as response_file:
			response_writer = csv.writer(response_file, delimiter=',', quotechar='"')
			for i in range(len(parsing[1])):
				response_writer.writerow([i, parsing[1][i]])
		return parsing


def scrub_text(text):
	tokenized_response = word_tokenize(text)
	filtered_sentence = [w for w in tokenized_response if not w in stop_words]
	lemmatized_sentence = [lemmatizer.lemmatize(w) for w in filtered_sentence]
	return ' '.join(lemmatized_sentence)


def parse_html():
	row_strainer = SoupStrainer('p')
	soup = BeautifulSoup(open(training_doc), 'html.parser', parse_only=row_strainer)
	return clean_text([seg for seg in soup])


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
		elearning.append(
			f'<h3>{cap.select(".entry-title")[0].getText()}</h3><p>{course_descrip}<a href="{cap.select(".ld_course_grid_button a")[0].attrs.get("href")}" target="blank"><b>More details here</b></a></p>')


def get_course_responses():
	if path.exists('/Users/rramage/Desktop/SynML/SynoptixML/synopbot/doc/e_learning_responses.csv'):
		responses = []
		lower_responses = []
		with open('/Users/rramage/Desktop/SynML/SynoptixML/synopbot/doc/e_learning_responses.csv') as csv_file:
			reader = csv.reader(csv_file, delimiter=',')
			for row in reader:
				row_data = row[1]
				responses.append(row_data)
				lower_responses.append(row_data.lower())
		return responses, lower_responses
	else:
		parse_elearning()
		responses = []
		lower_responses = []
		for course in elearning:
			responses.append(course)
			lower_responses.append(course.lower())
		with open('/Users/rramage/Desktop/SynML/SynoptixML/synopbot/doc/e_learning_responses.csv', 'w') as e_learning_file:
			response_writer = csv.writer(e_learning_file, delimiter=',', quotechar='"')
			for i in range(len(responses)):
				response_writer.writerow([i, responses[i]])
		return responses, lower_responses


def clean_text(text):
	responses = []
	lower_responses = []
	for seg in text:
		clean_seg = re.sub(r'(<[/a-zA-Z]+>)', '', seg.getText()).strip()
		if clean_seg != '':
			responses.append(clean_seg)
			lower_responses.append(clean_seg.lower())
	with open(forum_csv) as csv_file:
		reader = csv.reader(csv_file, delimiter=',')
		for row in reader:
			if len(row) == 2:
				responses.append(row[1])
				lower_responses.append(row[1].lower())
	# print(len(responses))
	# print(len(lower_responses))
	return responses, lower_responses


def process_question(text):
	return text.lower().strip()


def chat(user_start=''):
	if user_start == '':
		user_msg = input('What can I help with today?\n')
	else:
		user_msg = user_start
	if user_msg.lower() == 'tell me a joke':
		print(get_joke())
	else:
		print(get_response(user_msg))


def get_adjectives(input):
	adjectives = []
	for msg in input:
		if msg[1].startswith('JJ'):
			adjectives.append(msg[0])
	return adjectives


def get_verbs(input):
	verbs = []
	for msg in input:
		if msg[1] == 'VB':
			verbs.append(msg[0])
	return verbs


def get_nouns(input):
	nouns = []
	prev_noun = False
	for msg in input:
		if msg[1].startswith('NN'):
			strip = msg[0].strip('?')
			if strip not in ['help', 'Help']:
				if msg[1] != 'NNS':
					strip += 's' if not strip.endswith('s') and not strip.endswith('ing') else ''
				if prev_noun:
					prev = nouns.pop().strip('s')
					strip = prev + ' ' + strip
				nouns.append(strip)
				prev_noun = True
		else:
			prev_noun = False
	return nouns


def compare_overlap(user, criteria):
	similiar_count = 0
	for word in user:
		split = word.split()
		for seg in split:
			if seg.lower() in criteria:
				similiar_count += 1
	return similiar_count


def get_response(user_input):
	responses = get_responses()
	course_responses = get_course_responses()
	to_user_responses = responses[0]
	check_against_responses = responses[1]
	stop_keys = ['stop', 'bye', 'done', 'quit', 'no']
	vectorizer = TfidfVectorizer()

	tagged_input = pos_tag(user_input.strip().split())

	user_nouns = get_nouns(tagged_input)
	# print(user_nouns)

	# user_verbs = get_verbs(tagged_input)
	# user_adjectives = get_adjectives(tagged_input)

	check_against_course = course_responses[1]
	check_against_course.append(user_input.lower())
	course_vectors = vectorizer.fit_transform(check_against_course)
	course_similarities = cosine_similarity(course_vectors[-1], course_vectors)
	course_index = course_similarities.argsort()[0][-2]
	recommended_course = ''

	if course_similarities[0][course_index] > 0.2:
		selected_course = course_responses[0][course_index]
		recommended_course = f'We also have an elearning course we think would be a good fit to help with this if you\'re interested.\n{selected_course}'

	check_against_responses.append(user_input.lower())
	response_vectors = vectorizer.fit_transform(check_against_responses)

	# compute cosine similarity betweeen the user message tf-idf vector and the different response tf-idf vectors:
	cosine_similarities = cosine_similarity(response_vectors[-1], response_vectors)

	# print(cosine_similarities)
	# get the index of the most similar response to the user message:
	similar_response_index = cosine_similarities.argsort()[0][-2]

	# removes the user response from list so the sizes don't get mismatched
	check_against_responses.pop()
	best_response = ''
	support_count = compare_overlap(user_nouns, support_nouns)
	services_count = compare_overlap(user_nouns, services_nouns)
	department = 'support staff, or submit a question on our support forum here http://synoptix.com/forum' if support_count > services_count else 'services department'

	if cosine_similarities[0][similar_response_index] < 0.4:
		if re.match(r'.*(train[ing|er]*).*', user_input.lower()):
			if len(user_nouns) > 0:
				return f'If you\'re looking for training in {user_nouns[0]} I can get you in touch with one of our technical experts.'
			else:
				return f'If you\'re looking for training, I can get you in touch with one of our technical experts.'
		elif re.match(r'.*[help]*.*(database|server|admin|migrat[e|ion]*).*(help)*.*', user_input.lower()):
			return 'Let\'s get you in touch with out technical <b><a href="https://synoptixsoftware.com/support" target="blank">support staff</a></b>.'
		else:
			if len(user_nouns) > 0:
				best_response = f'If you need help with {user_nouns[0]},'
			best_response += f' I would recommend contacting our {department}.  {recommended_course}<br>\nFor the time being though this may be of some help to you:<br>'
	try:
		best_response += to_user_responses[similar_response_index]
	except Exception as e:
		best_response = 'I\'m not sure how to answer that sorry.'
	return best_response


#
if len(sys.argv) == 2:
	chat(sys.argv[1])
else:
	chat()
