#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
David Isoka N0764763
"""
#similarity works but not aiml

#######################################################
# Initialise Wikipedia agent
#######################################################
import wikipedia

#######################################################
# Initialise weather agent
#######################################################
import json, requests
#insert your personal OpenWeathermap API key here if you have one, and want to use this feature
APIkey = "5403a1e0442ce1dd18cb1bf7c40e776f" 

# Initialise similarity component
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import numpy as np
import io
import random
import string
import warnings
#######################################################
#  Initialise AIML agent
#######################################################
import aiml
# Create a Kernel object. No string encoding (all I/O is unicode)
kern = aiml.Kernel()
kern.setTextEncoding(None)
# Use the Kernel's bootstrap() method to initialize the Kernel. The
# optional learnFiles argument is a file (or list of files) to load.
# The optional commands argument is a command (or list of commands)
# to run after the files are loaded.
# The optional brainFile argument specifies a brain file to load.
kern.bootstrap(learnFiles="aiml-responses.xml")
#######################################################
#Opening and reading Question and Answer Text file
file = open("fashionQA.txt", "r")
raw = file.read()
raw = raw.lower() # converts to lowercase
#nltk.download('punkt') #for first time running
#nltk.download('wordnet') #for first time running

tokens_sentence = nltk.sent_tokenize(raw) #converts to list of sentences
tokens_word = nltk.word_tokenize(raw) #converts to list of words

#preprocessing
nltklemma = nltk.stem.WordNetLemmatizer()

def lemtokens(tokens):
        return [nltklemma.lemmatize(token) for token in tokens]
remove_puct_dict = dict((ord(punct), None) for punct in string.punctuation)
def lemnormalize(text):
    return lemtokens(nltk.word_tokenize(text.lower().translate(remove_puct_dict)))

smalltalk_inputs = ('hello', 'hi', 'whats up', 'hey')
smalltalk_responses = ['hi', 'hello', 'I am glad u r talking to me!']
def greeting(sentence):
    for word in sentence.split():
        if word.lower() in smalltalk_inputs:
            return random.choice(smalltalk_responses)

def response(user_input):
    chatbot_response = ''
    tokens_sentence.append(user_input)
    vec_tfidf = TfidfVectorizer(tokenizer=lemnormalize, stop_words='english')
    tfidf = vec_tfidf.fit_transform(tokens_sentence)
    value_cos = cosine_similarity(tfidf[-1], tfidf)
    idx = value_cos.argsort()[0][-2]
    cos_flat = value_cos.flatten()
    cos_flat.sort()
    tfidf_val = cos_flat[-2]
    if (tfidf_val == 0):
        chatbot_response = chatbot_response + "I am sorry, I couldnt understand you"
        return(chatbot_response)
    else:
        chatbot_response = chatbot_response + tokens_sentence[idx]
        return(chatbot_response)
  
flag=True
while(flag==True):
    user_input = input(">Human: ")
    user_input = user_input.lower()
    if(user_input!='bye'):
        if(user_input=='thanks'or user_input=='thank you'):
            flag=False
            print("You're welcome..")
        else:
            if(greeting(user_input)!=None):
                print("R.A.F: "+greeting(user_input))
            else:
                print("R.A.F: ",end="")
                print(response(user_input))
                tokens_sentence.remove(user_input)
    else:
        flag=False
        print("R.A.F: Bye")
# Welcome user
#######################################################
print("Hi there my name is R.A.F, I am a chatbot that knows everything there is to know about fashion designers, go ahead, test my knowledge!")
#######################################################
# Main loop
#######################################################
while True:
    #get user input
    try:
        userInput = input(">Human: ")
    except (KeyboardInterrupt, EOFError) as e:
        print("Bye!")
        break
    #pre-process user input and determine response agent (if needed)
    responseAgent = 'aiml'
    #activate selected response agent
    if responseAgent == 'aiml':
        answer = kern.respond(userInput)
    #post-process the answer for commands
    if answer[0] == '#':
        params = answer[1:].split('$')
        cmd = int(params[0])
        if cmd == 0:
            print(params[1])
            break
        elif cmd == 1:
            try:
                wSummary = wikipedia.summary(params[1], sentences=3,auto_suggest=False)
                print(wSummary)
            except:
                print("Sorry, I do not know that. Be more specific!")
       # elif cmd == 2:
            
    else:
        print(answer)
