"""
 Esse arquivo extrai perguntas e resposta do site  http://www.seasky.org/astronomy/astronomy-faq.html
 e converte para json
"""
import requests
from bs4 import BeautifulSoup
import json


def node_list_to_text_list(node_list):
    return [node.text for node in node_list]


response = requests.get('http://www.seasky.org/astronomy/astronomy-faq.html')
dom = BeautifulSoup(response.text, 'html.parser')

question_list = dom.select('.header2')
aswer_list = dom.select('.header2 + br + p')

question_text_list = node_list_to_text_list(question_list)
aswer_text_list = node_list_to_text_list(aswer_list)

final_list = [{'question': pair[0], 'aswer': pair[1]} for pair in zip(question_text_list, aswer_text_list)]

final_json = {'faq': final_list}

with open('./resources/faq_astronomia.json', 'w') as json_file:
    json.dump(final_json, json_file, indent=2)

print("Processo de extração finalizado")
print("Arquivo gerado em /resources/faq_astronomia.json")
