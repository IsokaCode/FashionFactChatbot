#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
David Isoka N0764763 chatbot
"""

#######################################################
import wikipedia # Initialise Wikipedia agent

# For initialising similarity component
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import io
import random
import string
import warnings
import aiml # Initialise AIML agent
from PIL import Image
import tensorflow as tf

# tensorflow keras model
model = tf.keras.models.load_model("/mnt/c/dev/uni/ai/fashion_CNN_model.h5")

# Create a Kernel object. No string encoding (all I/O is unicode)
kern = aiml.Kernel()
kern.setTextEncoding(None)
# Use the Kernel's bootstrap() method to initialize the Kernel. The
# optional learnFiles argument is a file (or list of files) to load.
kern.bootstrap(learnFiles="aiml-responses.xml")

#Opening and reading Question and Answer Text file
file = open("/mnt/c/dev/uni/ai/fashionQA.txt", "r")
raw = file.read()
raw = raw.lower() # converts to lowercase
#nltk.download('punkt') #for first time running
#nltk.download('wordnet') #for first time running

#tokenization
tokens_sentence = nltk.sent_tokenize(raw) #converts to list of sentences
tokens_word = nltk.word_tokenize(raw) #converts to list of words

#preprocessing of similarity components
nltklemma = nltk.stem.WordNetLemmatizer()

def lemtokens(tokens):
        return [nltklemma.lemmatize(token) for token in tokens]
remove_puct_dict = dict((ord(punct), None) for punct in string.punctuation)
def lemnormalize(text):
    return lemtokens(nltk.word_tokenize(text.lower().translate(remove_puct_dict)))

smalltalk_inputs = ('hello', 'hi', 'whats up', 'hey')
smalltalk_responses = ['hi', 'hello', 'I am glad you are talking to me!']

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

# preprocessing
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
              'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

def image_classifier(userInput):

    # preprocessing piplining data similar to how the model was trained
    DATA_DIR = "/mnt/c/dev/uni/ai/my_clothing_dataset/"
    IMG_SIZE = 28
    FILE_END = ".png"
    img_path = DATA_DIR + userInput + FILE_END # for absolute path
   
    img = tf.keras.preprocessing.image.load_img(img_path, color_mode="grayscale", target_size=(IMG_SIZE, IMG_SIZE))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    
    img_array = tf.expand_dims(img_array, 0) 
    
    prediction = model.predict(img_array)

    score = tf.nn.softmax(prediction[0])
 
    return ("This item of clothing is likely a {}  and i am {:.2f} percent confident."
    .format(class_names[np.argmax(prediction[0])], 100 * np.max(score)))
    
############################################################

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
print("Hi there my name is R.A.F, I am a chatbot that knows everything there is to know about fashion designers, i can also identify clothing items so go ahead, test my knowledge!")
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
        elif cmd == 2:
            try:
                predict = image_classifier(params[1])
                print(predict)
            except:
                print("Sorry, I do not recognize that brand. Please be more specific!")

    else:
        print(answer)