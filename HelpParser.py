from bs4 import BeautifulSoup, SoupStrainer
import re
import csv

training_doc = './doc/HelpTraining.html'
forum_csv = './doc/forum_data.csv'
reg = re.compile(r'(<[/a-zA-Z]+>)')
elearning = ['Our ELearning "Introduction to Report Designer: Accounting Cell Properties" course covers accounting cells which are at the core of financial report building. Accounting Cells are at the core of financial report building. The cell properties allow you to control and adjust how much information is displayed in your report. In this course, you will learn how to use these cell properties. Each area is reviewed and discussed to build foundational knowledge of Synoptix’s Report Designer. You can enroll in this course <a href="https://elearning.synoptixsoftware.com/courses/introduction-to-report-designing-accounting-cell-properties">here</a>.',
             'Our ELearning "Introduction to Report Designer: Formatting Menu" course will familiarize you with formatting options and icons of report writing. .  It is the ideal preliminary course to acquaint users with the report designer. We advise starting here and then proceeding to our courses on the Report Designer Palette & Cell Properties. you can enroll <a>here</a>',
             'Our ELearning "Introduction to Report Designer: Reporting Palette" course will help familiarize you with the vital areaas and properties of the reporting']

def get_responses():
	return parse_html()


def parse_html():
	row_strainer = SoupStrainer('p')
	soup = BeautifulSoup(open(training_doc), 'html.parser', parse_only=row_strainer)
	return clean_text([seg for seg in soup])


def clean_text(text):
	responses = []
	lower_responses = []
	for course in elearning:
		responses.append(course)
		lower_responses.append(course.lower())
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
