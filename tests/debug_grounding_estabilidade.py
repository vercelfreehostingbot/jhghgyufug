# Debug: inspeciona scores de similaridade dos chunks retornados pelo retriever
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from test_eval_set import gerar_resposta_max
from verificacao_llm import verificar_grounding
from busca_semantica import buscar_contexto
load_dotenv()

from inicializacao import carregar_base_conhecimento, carregar_prompt

system_prompt = carregar_prompt()
retriever = carregar_base_conhecimento()

llm_chat = ChatOpenAI(
        model='gpt-4o-mini',
        temperature=0,
    )

llm_verificador = ChatOpenAI(
        model='gpt-4o-mini',
        temperature=0
    )

pergunta = 'meu pedido chegou amassado, posso trocar por outro produto de valor maior pagando a diferença?'

contexto = buscar_contexto(retriever, pergunta)

for i in range(10):
    reply = gerar_resposta_max(llm_chat, system_prompt, contexto, pergunta)
    grounding_falhou = verificar_grounding(llm_verificador, pergunta, contexto, reply)

    print(f"\n--- Execução {i + 1} ---")
    print(f"Resposta: {reply}")
    print(f"Grounding falhou? {grounding_falhou}")