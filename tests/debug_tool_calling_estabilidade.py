import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from inicializacao import carregar_prompt, carregar_base_conhecimento
from busca_semantica import buscar_contexto
from verificacao_llm import verificar_grounding
from tools import todas_as_tools

load_dotenv()

CLIENTE_ID = 8
PERGUNTA = 'meu pedido atrasado no Sul, teve alguma novidade?'
NUM_EXECUCOES = 10

system_prompt = carregar_prompt()
retriever = carregar_base_conhecimento()

llm_chat = ChatOpenAI(model='gpt-4o-mini', temperature=0.3, streaming=False)
llm_chat_com_tools = llm_chat.bind_tools(todas_as_tools)
llm_verificador = ChatOpenAI(model='gpt-4o-mini', temperature=0)

contexto = buscar_contexto(retriever, PERGUNTA)
mensagem_com_contexto = f'{PERGUNTA}\n\nInformações relevantes:\n{contexto}\n\nO cliente_id do cliente atual é {CLIENTE_ID}.'

for i in range(1, NUM_EXECUCOES + 1):
    messages = [SystemMessage(content=system_prompt), HumanMessage(content=mensagem_com_contexto)]

    resposta_decisao = llm_chat_com_tools.invoke(messages)
    tool_foi_chamada = bool(resposta_decisao.tool_calls)

    if tool_foi_chamada:
        messages.append(resposta_decisao)
        for chamada in resposta_decisao.tool_calls:
            tool_chamada = next(t for t in todas_as_tools if t.name == chamada['name'])
            resultado = tool_chamada.invoke(chamada['args'])
            messages.append(ToolMessage(content=resultado, tool_call_id=chamada['id']))

    reply = llm_chat.invoke(messages).content

    marcador_transfer = 'TRANSFER_HUMANO' in reply.upper()
    grounding_falhou = None if marcador_transfer else verificar_grounding(llm_verificador, PERGUNTA, contexto, reply)

    print(f'--- Execução {i} ---')
    print(f'Tool chamada: {tool_foi_chamada}')
    print(f'Marcador TRANSFER_HUMANO: {marcador_transfer}')
    print(f'Grounding falhou: {grounding_falhou}')
    print(f'Reply: {reply}')
    print()