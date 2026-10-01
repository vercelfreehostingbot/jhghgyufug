import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from inicializacao import carregar_prompt, carregar_base_conhecimento
from busca_semantica import buscar_contexto
from tools import todas_as_tools

CLIENTE_ID = 8  # cliente com histórico real já existente no banco (ver sessão de debug anterior)
NUM_REPETICOES = 5

PERGUNTAS = {
    'sinal_explicito_de_retomada': [
        'e aí, teve alguma novidade sobre aquilo?',
        'vocês resolveram aquele problema que eu falei?',
    ],
    'dado_completo_sem_sinal_de_retomada': [
        'meu pedido atrasado no Sul, teve alguma novidade?',
        'qual o prazo de entrega para o sul?',
    ],
    'pergunta_nova_sem_relacao_com_historico': [
        'qual o horário de atendimento?',
        'meu pedido chegou amassado, o que eu faço?',
    ],
    'referencia_explicita_a_conversa_anterior': [
        'você lembra o que eu perguntei da última vez?',
        'a resposta que você me deu antes ainda vale?',
    ],
}

system_prompt = carregar_prompt()
retriever = carregar_base_conhecimento()

llm_chat = ChatOpenAI(model='gpt-4o-mini', temperature=0.3)
llm_chat_com_tools = llm_chat.bind_tools(todas_as_tools)


def tool_foi_chamada(pergunta):
    contexto = buscar_contexto(retriever, pergunta)
    mensagem_com_contexto = f'{pergunta}\n\nInformações relevantes:\n{contexto}\n\nO cliente_id do cliente atual é {CLIENTE_ID}.'
    messages = [SystemMessage(content=system_prompt), HumanMessage(content=mensagem_com_contexto)]

    resposta_decisao = llm_chat_com_tools.invoke(messages)
    return bool(resposta_decisao.tool_calls)


def main():
    resumo = defaultdict(lambda: defaultdict(int))

    for categoria, perguntas in PERGUNTAS.items():
        print(f'=== Categoria: {categoria} ===\n')

        for pergunta in perguntas:
            chamadas = 0

            for i in range(1, NUM_REPETICOES + 1):
                chamou = tool_foi_chamada(pergunta)
                chamadas += int(chamou)
                print(f'  [{i}/{NUM_REPETICOES}] "{pergunta}" -> tool chamada: {chamou}')

            resumo[categoria][pergunta] = chamadas
            print(f'  Total: {chamadas}/{NUM_REPETICOES} chamou a tool\n')

    print('=== RESUMO GERAL ===\n')
    for categoria, perguntas in resumo.items():
        print(f'{categoria}:')
        for pergunta, chamadas in perguntas.items():
            print(f'  "{pergunta}": {chamadas}/{NUM_REPETICOES}')
        print()


if __name__ == '__main__':
    main()