import pandas as pd
import json
import nltk
from preprocessing_document import lemmatize_tokenize_normalize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def demonstration_tfidf():
    """
    Função somente para a demonstração de como funciona a tokenização e o tfidf

    """

    tfidfvectorizer = TfidfVectorizer(tokenizer=lemmatize_tokenize_normalize)
    frase = 'What is the difference between astronomy and astrology?'
    frase_dois = 'Do I need an  expensive telescope to enjoy astronomy?'
    tfidf = tfidfvectorizer.fit_transform([frase, frase_dois])
    tfidf_tokens = tfidfvectorizer.get_feature_names()

    print(lemmatize_tokenize_normalize(frase))
    print(lemmatize_tokenize_normalize(frase_dois))

    df_tfidfvect = pd.DataFrame(data=tfidf.toarray(), index=['Doc1', 'Doc2'], columns=tfidf_tokens)
    print(df_tfidfvect)


def tf_idf_vector_from_sentence_list(sentences: list, question):
    # incluindo a pergunta do usuario no corpo, para que assim possa ser calculado o tf-idf dela
    sentences.append(question)
    tfidf_vectorizer = TfidfVectorizer(tokenizer=lemmatize_tokenize_normalize)
    tfidf_vectors = tfidf_vectorizer.fit_transform(sentences)
    user_question_vector = tfidf_vectors[-1]
    similarity_matrix = cosine_similarity(user_question_vector, tfidf_vectors)

    # magia negra para encontrar o mais similar na matriz
    idx = similarity_matrix.argsort()[0][-2]
    flat = similarity_matrix.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    # fim da magica

    if req_tfidf == 0:
        print("deu ruim")
    else:
        print(sentences[idx])


with open('./resources/faq_astronomia.json', 'r') as json_file:
    data = json.load(json_file)['faq']

df = pd.DataFrame(data)
tf_idf_vector_from_sentence_list(df.question.tolist(), 'how i invent the telescope at work?')
