import nltk
from nltk.corpus import stopwords
import random
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import aiml
from Seq2Seq import predict

class Chatbot():
    def __init__(self):
        self.kern = aiml.Kernel()
        self.kern.setTextEncoding(None)
        self.kern.bootstrap(learnFiles="corpuses.xml")
        self.finalexecution()
        pass

    def lemtokens(self,tokenwords):
        return [self.nltklem.lemmatize(token) for token in tokenwords]

    def lemnormal(self,sentence):
        return self.lemtokens(nltk.word_tokenize(sentence.lower().translate(self.remove_puct_dict)))

    def finalexecution(self):
        v = """
        Thomas => {}
        piers => {}
        ray => {}
        leo => {}
        arthur => {}
        newton => {}
        einstein => {}
        galileo => {}
        faraday => {}
        darwin => {}
        tesla => {}
        historian => p1
        scientist => p2
        researcher => p3
        archiologist => p4
        be_in => {}
        """
        folval = nltk.Valuation.fromstring(v)
        grammar_file = 'simple-sem.fcfg'
        objectCounter = 0
        # read text file
        f = open('corpus.txt', 'r', errors='ignore')
        crude = f.read()
        crude = crude.lower()
        # nltk.download('punkt')
        # nltk.download('wordnet')

        self.tokens_sentence = nltk.sent_tokenize(crude)
        self.tokens_word = nltk.word_tokenize(crude)

        self.stop_words = set(stopwords.words('english'))
        self.tokens_word = [w for w in self.tokens_word if not w in self.stop_words]

        self.nltklem = nltk.stem.WordNetLemmatizer()

        self.remove_puct_dict = dict((ord(punct), None) for punct in string.punctuation)

        self.smalltalk_inputs = ('hello', 'hi', 'whats up', 'hey')
        self.smalltalk_output = ['hi', 'hello', 'I am glad u r talking to me!']

        banner = True
        print("Alex:Hi i am Alex, I am your personal assistant, please let me know in few words why are you here?, If you want to exit, type Bye!")

        while (banner == True):
            self.user_input = input()
            self.user_input = self.user_input.lower()
            # pre-process user input and determine response agent (if needed)
            responseAgent = 'aiml'
            # activate selected response agent
            if responseAgent == 'aiml':
                answer = self.kern.respond(self.user_input)
            if answer[0] == '#':
                params = answer[1:].split('$')
                cmd = int(params[0])
            if cmd == 99:
                # post-process the answer for commands
                if (self.user_input != 'bye'):
                    if (self.user_input == 'thanks' or self.user_input == 'thank you'):
                        banner = False
                        print("Alex: You are welcome")
                    else:
                        words = [word for word in self.user_input.split() if word.lower() in self.smalltalk_inputs]
                        if (words != []):
                            for word in self.user_input.split():
                                if word.lower() in self.smalltalk_inputs:
                                    print("Alex: " + random.choice(self.smalltalk_output))
                        else:
                            self.tokens_sentence.append(self.user_input)
                            stop_words = set(stopwords.words('english'))
                            self.tokens_word = self.tokens_word + [w for w in self.tokens_word if not w in stop_words]
                            final_words = list(set(self.tokens_word))
                            print("Alex :", end="")
                            chatbot_result = ''
                            vec_tfidf_count = TfidfVectorizer(tokenizer=self.lemnormal)
                            value_tfidf = vec_tfidf_count.fit_transform(self.tokens_sentence)
                            value_cos = cosine_similarity(value_tfidf[-1], value_tfidf)
                            idx = value_cos.argsort()[0][-2]
                            cos_flat = value_cos.flatten()
                            cos_flat.sort()
                            tfidf_val_count = cos_flat[-2]
                            if (tfidf_val_count == 0):
                                chatbot_result = chatbot_result + "I am sorry, I couldnt understand the context"
                                print(chatbot_result)
                            else:
                                chatbot_result = chatbot_result + self.tokens_sentence[idx]
                                print(chatbot_result)
                            self.tokens_sentence.remove(self.user_input)
                else:
                    banner = False
                    print("Alex  : Bye, take care have a nice day")
            elif cmd == 4:  # x IS AN y
                o = 'o' + str(objectCounter)
                objectCounter += 1
                folval['o' + o] = o  # insert constant
                if len(folval[params[1]]) == 1:  # clean up if necessary
                    if ('',) in folval[params[1]]:
                        folval[params[1]].clear()
                folval[params[1]].add((o,))  # insert type of plant information
                if len(folval["be_in"]) == 1:  # clean up if necessary
                    if ('',) in folval["be_in"]:
                        folval["be_in"].clear()
                folval["be_in"].add((o, folval[params[2]]))  # insert location
            elif cmd == 5:  # Are there any x in y
                g = nltk.Assignment(folval.domain)
                m = nltk.Model(folval.domain, folval)
                sent = 'some ' + params[1] + ' are_in ' + params[2]
                results = nltk.evaluate_sents([sent], grammar_file, m, g)[0]
                if results[0][2] == True:
                    print("Yes.")
                else:
                    print("No.")
            elif cmd == 6:  # IS x AN y
                g = nltk.Assignment(folval.domain)
                m = nltk.Model(folval.domain, folval)
                sent = 'all ' + params[1] + ' are_in ' + params[2]
                results = nltk.evaluate_sents([sent], grammar_file, m, g)[0]
                if results[0][2] == True:
                    print("Yes.")
                else:
                    print("No.")
            elif cmd == 7:  # ARE THERE ANY ...
                g = nltk.Assignment(folval.domain)
                m = nltk.Model(folval.domain, folval)
                e = nltk.Expression.fromstring("be_in(x," + params[1] + ")")
                sat = m.satisfiers(e, "x", g)
                if len(sat) == 0:
                    print("None.")
                else:
                    # find satisfying objects in the valuation dictionary,
                    # and print their type names
                    sol = folval.values()
                    for so in sat:
                        for k, v in folval.items():
                            if len(v) > 0:
                                vl = list(v)
                                if len(vl[0]) == 1:
                                    for i in vl:
                                        if i[0] == so:
                                            print(k)
            elif cmd == 10:
                result = predict(user_input)
                
if __name__ == '__main__':
    obj = Chatbot()
