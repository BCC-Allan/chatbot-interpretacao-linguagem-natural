import random
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk

nltk.download('popular', quiet=True)  # for downloading packages
nltk.download('punkt')
nltk.download('wordnet')

example1 = "./depressao.txt"
file1 = open(example1, "r")
raw = file1.read()
sent_tokens = nltk.sent_tokenize(raw)  # lista de sentenças

# lemarization
lemmer = nltk.stem.WordNetLemmatizer()


# WordNet dicionário semântico em inglês, incluído na nltk.
def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]


# removendo a pontuação
sem_pontuacao = dict((ord(punct), None) for punct in string.punctuation)


def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(sem_pontuacao)))


GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up", "hey",)
GREETING_RESPONSES = ["hi", "hey", "*nods*", "hi there", "hello", "I am glad! You are talking to me"]


# e escolher de forma randômica, uma resposta em um array,
# baseado na entrada que está em outro array.
def greeting(sentence):
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)


def response(user_response):
    robo_response = ''
    sent_tokens.append(user_response)

    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)

    idx = vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if req_tfidf == 0:
        robo_response = robo_response + "I am sorry! I don't understand you"
        sent_tokens.remove(user_response)
        return robo_response
    else:
        robo_response = robo_response + sent_tokens[idx]
        sent_tokens.remove(user_response)
        return robo_response


flag = True
print("ROBO: My name is Robo. I will answer your queries about Chatbots. If you want to exit, type Bye!")
while flag:
    user_response = input()
    user_response = user_response.lower()
    if user_response != 'bye':
        if user_response == 'thanks' or user_response == 'thank you':
            flag = False
            print("ROBO: You are welcome..")
        else:
            if greeting(user_response) is not None:
                print("ROBO: " + greeting(user_response))
            else:
                print("ROBO: ", end="")
                print(response(user_response))

    else:
        flag = False
        print("ROBO: Bye! take care..")
