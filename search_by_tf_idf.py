from typing import Optional, List

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


def get_idf_vector_from_sentence_list(corpus: List[str], document: str):
    # incluindo a pergunta do usuario no corpo, para que assim possa ser calculado o tf-idf dela
    corpus.append(document)
    tfidf_vectorizer = TfidfVectorizer(tokenizer=lemmatize_tokenize_normalize)  # faz a vetorização usando tf idf
    tfidf_vectors = tfidf_vectorizer.fit_transform(corpus)
    return tfidf_vectors


def get_most_similar_index_from_vectors(vectorize_corpus, vectorize_document) -> Optional[int]:
    similarity_matrix = cosine_similarity(vectorize_document, vectorize_corpus)

    # magia negra para encontrar o mais similar na matriz da cosine_similarity
    index = similarity_matrix.argsort()[0][-2]
    flat = similarity_matrix.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    # fim da magica

    if req_tfidf == 0:
        print("Erro não foi encontrado similar")
        return None
    return index


def search_most_similar_document(corpus: List[str], document: str) -> str:
    """
      Faz a busca de documento mais similar chamando as funções para vetorizar com tf idf e fazer a
      cosine_similarity
    """
    tfidf_vectors = get_idf_vector_from_sentence_list(corpus, document)
    vectorized_document = tfidf_vectors[-1]  # extrai o vertor da pergunta
    corpus.remove(document)
    index = get_most_similar_index_from_vectors(tfidf_vectors, vectorized_document)
    return corpus[index]


