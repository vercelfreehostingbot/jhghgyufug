# Debug: inspeciona estabilidade de verificar_informacao_suficiente()
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from verificacao_llm import verificar_informacao_suficiente
load_dotenv()

llm_verificador = ChatOpenAI(
        model='gpt-4o-mini',
        temperature=0,
    )

pergunta = 'meu pedido chegou amassado, posso trocar por outro produto de valor maior pagando a diferença?'

for i in range(10):
    resultado = verificar_informacao_suficiente(llm_verificador, pergunta)

    print(f"\n--- Execução {i + 1} ---")
    print(f"Precisa de mais informação? {resultado}")