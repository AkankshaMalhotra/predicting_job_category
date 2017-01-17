#!/usr/bin/python
import pandas as pd
import easygui as g
import numpy as np
import nltk
import random
import string
import textmining
import sys

#Loading csv file
def loadcsv(filename):
	t=[]
	titles = pd.read_csv(filename, sep=',', usecols=['Title'], squeeze=True)
	desc = pd.read_csv(filename, sep=',', usecols=['FullDescription'], squeeze=True)
	cat = pd.read_csv(filename, sep=',', usecols=['Category'], squeeze=True)
	category=(cat.values).tolist()
	d=pd.concat([titles,desc],axis=1)
	x=(d.values).tolist()
	for i in range(len(x)):
		 t.append(" ".join(x[i]))
	return [t,category]


#Cleaning Data by removing stopwords and punctuations and converting the words to lower case
def cleanData(corpus):
	punctuation=string.punctuation
	from nltk.corpus import stopwords
	english_stops = set(stopwords.words('english'))
	for i in range(len(corpus)):    
			corpus[i]=corpus[i].lower()
			corpus[i]=corpus[i].split()
			corpus[i] = [word for word in corpus[i] if word not in english_stops]	
			corpus[i] = [word for word in corpus[i] if word not in punctuation]	
	return(corpus)

#Creation of Bag of words using termDocument Matrix
def termDocumentMatrix(vocab):
	tdm = textmining.TermDocumentMatrix()
	vocab=np.array(vocab)
	for row in vocab:
		tdm.add_doc(" ".join(row))
	tdm.write_csv('words.csv', cutoff=1)


#Using multinomial naive bayes algorithm fitting the training set and calculating score
def naivetrain(cat):
	wer = pd.read_csv("words.csv")
	length=int(len(wer)*0.67)
	from sklearn.naive_bayes import MultinomialNB
	clf = MultinomialNB()
	clf.fit(wer.iloc[:length], cat[:length])
	scor=clf.score(wer.iloc[(length+1):],cat[(length+1):])
	return(scor)

#Using multinomial naive bayes algorithm fitting the training set and predicting job category
def naivetrain2(cat):
	wer = pd.read_csv("words.csv")
	length=len(wer)
	from sklearn.naive_bayes import MultinomialNB
	clf = MultinomialNB()
	scor=0.0
	clf.fit(wer.iloc[:length-1], cat[:length-1])
	y=clf.predict(wer.iloc[(length-1)])
	return(y)

	

def main():
	msg ="What do you want to do?"
	title = "Job Category Extraction Using Naive Bayes"
	choices = ["Training and testing of already present data", "Job Category Prediction from user's input"]
	choice = g.choicebox(msg, title, choices)
	if (str(choice) == choices[1]):
		tile=g.enterbox(msg='Enter Job Title', title='Job Title ', default='', strip=True)
		desc=g.enterbox(msg='Enter Job Description', title='Job Description ', default='', strip=True)
		if(not title) or (not desc):
			sys.exit(0)
		dt=[]		
		dt.append(tile)
		dt.append(desc)
		dt="".join(dt)
		filename=raw_input("Enter file name along with path")
		info=loadcsv(filename)
		info[0].append(dt)
		clean=cleanData(info[0])
		termDocumentMatrix(clean)
		c=naivetrain2(info[1])
		g.msgbox("Job Category is:"+str(c))
	if str(choice)==choices[0]:
		filename=raw_input("Enter file name along with path")
		info=loadcsv(filename)
		clean=cleanData(info[0])
		termDocumentMatrix(clean)
		c=naivetrain(info[1])
		g.msgbox("Accuracy of Model:"+str(c))

main()


