#!/usr/bin/env python3
"""
Sagar Joshi N0774756 - Chatbot
"""

#  Initialise different libraries
import aiml
import sys
from flask import Flask 
from flask import render_template
import os
import nltk
import numpy as np
import random
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

#Initialise AIML File
kernel = aiml.Kernel()
kernel.learn("mybot-basic.xml")

###########################################################

#Open's Question and Answer Text file
file = open("mybot.txt", "r")
questionList = []
answerList = []

for line in file:
    fields = line.split("$")  
    questionList.append(fields[0].lower())
    answerList.append(fields[1])

#Initiates WordNetLemmatizer from NLTK Library
wnlemmatizer = nltk.stem.WordNetLemmatizer()

#Lemmatizes list of words
def perform_lemmatization(tokens):
    return [wnlemmatizer.lemmatize(token) for token in tokens]

#Removes punctuation from text
punctuation_removal = dict((ord(punctuation), None) for punctuation in string.punctuation)

#Tokenizes, Lemmatizes and Removes Punctuation From The Sentence
def get_processed_text(document):
    return perform_lemmatization(nltk.word_tokenize(document.lower().translate(punctuation_removal)))

#######################################################################################

# Main loop
def findAnswer(userinput):
    #pre-process user input and determine response agent (if needed)
    responseAgent = 'query'
    #activate selected response agent
    if responseAgent == 'query':
        answer = kernel.respond(userinput)
    #post-process the answer for commands
    if answer[0] == '#':
        params = answer[1:].split('$')
        cmd = int(params[0])
        if cmd == 0:
            return(params[1])
            sys.exit()
        elif cmd == 99:
            botlux_response = '' 
            questionList.append(userinput) 

            #Initiates tfidvectorizer which vectorises the questions
            word_vectorizer = TfidfVectorizer(tokenizer=get_processed_text, stop_words='english')
            all_word_vectors = word_vectorizer.fit_transform(questionList)

            #Finds cosine similarity and word vectors
            similar_vector_values = cosine_similarity(all_word_vectors[-1], all_word_vectors)
            similar_sentence_number = similar_vector_values.argsort()[0][-2]
            
            #Flattens the retrieved cosine similarity
            matched_vector = similar_vector_values.flatten()
            matched_vector.sort()
            vector_matched = matched_vector[-2]

            #If vector = 0 then bot hasn't found an answer
            if vector_matched == 0:
                botlux_response = botlux_response + "I am sorry, I could not understand you"
                return botlux_response
            else:
                botlux_response = botlux_response + answerList[similar_sentence_number]
                return botlux_response
    else:
        return(answer)
    
###################################################################################

#Initialise Web UI
app = Flask(__name__)

@app.route("/",endpoint = 'index1')
def index1():
    return render_template("index.html")

@app.route("/<query>", endpoint = 'index2') #You have to return the answer here
def index2(query):
    return findAnswer(query)

if __name__ == "__main__":
    app.run(debug=True)
