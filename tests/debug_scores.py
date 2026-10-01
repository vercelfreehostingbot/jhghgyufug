import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI

from inicializacao import carregar_prompt, carregar_base_conhecimento
from busca_semantica import buscar_contexto
from verificacao_llm import verificar_grounding

pergunta = "meu pedido foi extraviado"

system_prompt = carregar_prompt()
retriever = carregar_base_conhecimento()
llm_chat = ChatOpenAI(model='gpt-4o-mini', temperature=0.3)
llm_verificador = ChatOpenAI(model='gpt-4o-mini', temperature=0)

contexto = buscar_contexto(retriever, pergunta)
print(f"===== CONTEXTO ({len(contexto)} caracteres) =====\n{contexto}\n")

mensagem_com_contexto = f'{pergunta}\n\nInformações relevantes:\n{contexto}'
messages = [
    ('system', system_prompt),
    ('human', mensagem_com_contexto)
]
resposta = llm_chat.invoke(messages)
print(f"===== RESPOSTA DO MAX =====\n{resposta.content}\n")

veredito = verificar_grounding(llm_verificador, pergunta, contexto, resposta.content)
print(f"===== VEREDITO DO GROUNDING =====\n{veredito}")