import pandas as pd
import json
import nltk

from search_by_tf_idf import search_most_similar_document


def load_json_corpus() -> pd.DataFrame:
    with open('./resources/faq_astronomia.json', 'r') as json_file:
        data = json.load(json_file)['faq']

    df = pd.DataFrame(data)
    return df


def bot_inicial_speak():
    print("ROBO: Olá eu sou o robo loco da astonomia, por favor digite somente perguntas.")
    print("ROBO: Todas as perguntas devem ser feitas em ingles.")
    print("ROBO: digite bye para sair")


def dowload_libs():
    print("Um momento, verificando a necessidade de baixar conteúdo adicional...")
    nltk.download('popular', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)


def get_response(user_question: str):
    pandas_dataframe = load_json_corpus()

    aswer = ""

    if SEARCH_MODE == 'question':
        # aqui vamos fazer a pesquisa só usando as perguntas como corpus
        question_list = pandas_dataframe.question.tolist()
        similar_question = search_most_similar_document(question_list, user_question)
        # pega a responsta da pergunta similar
        aswer = pandas_dataframe.query(f"question == '{similar_question}'").iloc[0]['aswer']
    elif SEARCH_MODE == 'aswer':
        aswer_list = pandas_dataframe.aswer.tolist()
        # pega a resposta mais similar com a pergunta do usuario
        aswer = search_most_similar_document(aswer_list, user_question)

    return aswer


def main():
    dowload_libs()
    bot_inicial_speak()

    while True:
        user_question = input().lower()
        if user_question == 'bye':
            exit(0)

        print("ROBO: Processando resposta...")
        print("ROBO:", end="")
        print(get_response(user_question))


if __name__ == '__main__':

    # isso altera se o bot vai pesquisar se sua pergunta é semelhante a alguma no corpus de perguntas
    # ou se ele vai pesquisar se sua pergunta é semelhante a uma resposta no corpus de respostas
    SEARCH_MODE = 'question'  # question OU aswer
    main()
