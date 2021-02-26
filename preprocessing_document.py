"""
 Nesse arquivo existem funções para o pre tratamento da string,
 retirando stop words, normalizando e lemmatizando
"""

from nltk.stem import WordNetLemmatizer
from typing import List, Tuple
import nltk
from nltk.corpus import wordnet, stopwords
import string


def tag_swich_case(nltk_tag: str):
    """
        Faz a tradução de POS tags do nltk para o do wordnet
        tags são classificações da palavra
    """
    tokens_map = {
        'J': wordnet.ADJ,
        'V': wordnet.VERB,
        'N': wordnet.NOUN,
        'R': wordnet.ADV
    }
    # user try catch é mais rapido que um default ou um if
    try:
        return tokens_map[nltk_tag[0]]
    except KeyError:
        return wordnet.ADV  # default


def tokenize_without_stopwords(text: str) -> List[str]:
    stop_words = stopwords.words("english")
    return [word_ for word_ in nltk.word_tokenize(text) if word_ not in stop_words]


def words_with_pos_tag(text: str) -> List[Tuple[str, str]]:
    """
    Exemplo retorno
    [('the', 'n'), ('cat', 'n'), ('is', 'v'), ('sitting', 'v'), ('with', 'n')]
    """

    words_with_pos_nltk = nltk.pos_tag(tokenize_without_stopwords(text))
    return [(word_and_tag[0], tag_swich_case(word_and_tag[1])) for word_and_tag in words_with_pos_nltk]


def normalize_sentence(text: str) -> str:
    dict_pontuacoes_unicode_para_none = dict((ord(punct), None) for punct in string.punctuation)
    return text.lower().translate(dict_pontuacoes_unicode_para_none)


def lemmatize_tokenize_normalize(text: str) -> List[str]:
    """
    lemmatiza, transforma em array de palavras e retira stop words
    """
    normalized_text = normalize_sentence(text)
    words_tokenized_with_pos = words_with_pos_tag(normalized_text)
    lemmatizer = WordNetLemmatizer()

    final = []

    for word_, tag_ in words_tokenized_with_pos:
        # lematiza cara uma das palavras ajustadas para a usa POS tag
        final.append(lemmatizer.lemmatize(word_, tag_))

    return final
