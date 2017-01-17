#!/usr/bin/python
import pandas as pd
import easygui as g
import numpy as np
import nltk
import math
import random
import string
import textmining
import sys
import csv

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


def termDocumentMatrix(vocab):
	tdm = textmining.TermDocumentMatrix()
	big=[]
	vocab=np.array(vocab)
	for i in range(len(vocab)):
		tdm.add_doc(" ".join(vocab[i]))
	for row in tdm.rows(cutoff=1):
		big.append(row)
	wm=np.array(big[1:])
	return(wm)



def splitDataset(wm,cat):
	trainSize = int(len(wm) * .67)
	trainSet = []
	x=[]
	index=75
	i=0
	testSet=[]
	categtrain=[]
	categtest=[]
	while len(trainSet) < trainSize:
		while(index in x):
			index = random.randrange(len(wm))
		x.append(index)
		trainSet.append(wm[index])
		categtrain.append(cat[index])
	print(len(x))
	t=[]
	v=trainSize-1
	x.sort()
	for i in xrange(len(wm)-1,0,-1):
		if(v>-1 and i==x[v]):
			x.pop()
			v=v-1
		else:
			t.append(i)
	for i in t:
	        testSet.append(wm[i])
		categtest.append(cat[i])
	return([trainSet,testSet,categtrain,categtest])


def naivetrain(wm,cat):
	dataSet=splitDataset(wm,cat)
	from sklearn.naive_bayes import MultinomialNB
	clf = MultinomialNB()
	clf.fit(dataSet[0], dataSet[2])
	scor=clf.score(dataSet[1],dataSet[3])
	return scor

def naivetrain2(wm,cat):
	from itertools import repeat
	length=len(wm)
	from sklearn.naive_bayes import MultinomialNB
	clf = MultinomialNB()
	clf.fit(dataSet[0], dataSet[2])
	x=clf.predict(wm[:length-1],cat)
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
		wm=termDocumentMatrix(clean)
		c=naivetrain2(wm,info[1])
		g.msgbox("Job Category is:"+str(c))
	if str(choice)==choices[0]:
		filename=raw_input("Enter file name along with path")
		info=loadcsv(filename)
		clean=cleanData(info[0])
		wm=termDocumentMatrix(clean)
		c=naivetrain(wm,info[1])
		g.msgbox("Accuracy of Model:"+str(c))

main()


