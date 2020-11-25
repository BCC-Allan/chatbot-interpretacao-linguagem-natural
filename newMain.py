import pandas as pd
import json

from search_by_tf_idf import search_most_similar_document


def load_json_corpus() -> pd.DataFrame:
    with open('./resources/faq_astronomia.json', 'r') as json_file:
        data = json.load(json_file)['faq']

    df = pd.DataFrame(data)
    return df
    # print(search_most_similar_document(df.question.tolist(), 'how i invent the telescope?'))


def bot_inicial_speak():
    print("ROBO: Olá eu sou o robo loco da astonomia, por favor digite somente perguntas.")
    print("ROBO: Todas as perguntas devem ser feitas em ingles.")
    print("ROBO: digite bye para sair")


def main():
    bot_inicial_speak()
    pandas_dataframe = load_json_corpus()
    question_list = pandas_dataframe.question.tolist()  # aqui vamos fazer a pesquisa só usando as perguntas como corpus

    while True:
        user_response = input().lower()
        if user_response == 'bye':
            exit(0)

        print("ROBO: Processando resposta...")
        print("ROBO:", end="")
        similar_question = search_most_similar_document(question_list, user_response)
        aswer = pandas_dataframe.query(f"question == '{similar_question}'").iloc[0]['aswer']
        print(aswer)


if __name__ == '__main__':
    main()

