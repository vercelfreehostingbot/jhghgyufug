import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

import json

from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from inicializacao import carregar_prompt, carregar_base_conhecimento, carregar_indice_saudacoes, carregar_indice_saida
from busca_semantica import buscar_contexto, eh_saudacao, eh_intencao_saida
from verificacao_llm import verificar_grounding, verificar_informacao_suficiente

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def carregar_eval_set():
    caminho = os.path.join(BASE_DIR, 'tests', 'eval_set.json')
    with open(caminho, 'r', encoding='utf-8') as f:
        return json.load(f)


def gerar_resposta_max(llm_chat, system_prompt, contexto, pergunta):
    mensagem_com_contexto = f'{pergunta}\n\nInformações relevantes:\n{contexto}'
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=mensagem_com_contexto)
    ]
    resposta = llm_chat.invoke(messages)
    return resposta.content


def avaliar_is_greeting(pergunta, vectorstore_saudacoes, **kwargs):
    return eh_saudacao(vectorstore_saudacoes, pergunta)


def avaliar_should_find_context(pergunta, retriever, **kwargs):
    return bool(buscar_contexto(retriever, pergunta).strip())


def avaliar_needs_more_information(pergunta, llm_verificador, **kwargs):
    return verificar_informacao_suficiente(llm_verificador, pergunta)


def avaliar_grounding_should_fail(pergunta, retriever, llm_chat, llm_verificador, system_prompt, **kwargs):
    contexto = buscar_contexto(retriever, pergunta)
    if not contexto.strip():
        return None  # não testável: sem contexto não há resposta para avaliar grounding

    reply = gerar_resposta_max(llm_chat, system_prompt, contexto, pergunta)

    if 'TRANSFER_HUMANO' in reply.upper():
        return True  # equivalente a bloqueio, mesmo resultado final para o usuário

    return verificar_grounding(llm_verificador, pergunta, contexto, reply)

def avaliar_contem_texto_proibido(pergunta, retriever, llm_chat, system_prompt, texto_proibido, **kwargs):
    contexto = buscar_contexto(retriever, pergunta)
    if not contexto.strip():
        return None

    reply = gerar_resposta_max(llm_chat, system_prompt, contexto, pergunta)

    return texto_proibido.upper() in reply.upper()


def avaliar_resposta_contem(pergunta, retriever, llm_chat, system_prompt, texto_esperado, **kwargs):
    contexto = buscar_contexto(retriever, pergunta)
    if not contexto.strip():
        return None

    reply = gerar_resposta_max(llm_chat, system_prompt, contexto, pergunta)

    return texto_esperado.upper() in reply.upper()


def avaliar_intercepta_em(pergunta, vectorstore_saida, vectorstore_saudacoes, llm_verificador, retriever, **kwargs):
    if eh_intencao_saida(vectorstore_saida, pergunta):
        return 'saida'

    if eh_saudacao(vectorstore_saudacoes, pergunta):
        return 'saudacao'

    if verificar_informacao_suficiente(llm_verificador, pergunta):
        return 'needs_more_information'

    contexto = buscar_contexto(retriever, pergunta)
    if not contexto.strip():
        return 'sem_contexto'

    return 'passou'


# Mapa: nome do campo em "checks" -> função que sabe avaliá-lo.
# O nome do campo já diz o que está sendo verificado, sem precisar de um
# campo "tipo_verificacao" separado duplicando essa informação.
AVALIADORES = {
    'is_greeting': avaliar_is_greeting,
    'should_find_context': avaliar_should_find_context,
    'needs_more_information': avaliar_needs_more_information,
    'grounding_should_fail': avaliar_grounding_should_fail,
    'contem_texto_proibido': avaliar_contem_texto_proibido,
    'resposta_contem': avaliar_resposta_contem,
    'intercepta_em': avaliar_intercepta_em,
}


def rodar_caso(caso, **contexto_execucao):
    pergunta = caso['pergunta']
    checks = caso['checks']

    if len(checks) != 1:
        raise ValueError(f'Caso deve ter exatamente 1 check, encontrado: {list(checks.keys())}')

    nome_check = next(iter(checks))
    esperado = checks[nome_check]

    if nome_check not in AVALIADORES:
        raise ValueError(f'Check desconhecido: {nome_check}')

    avaliador = AVALIADORES[nome_check]

    argumentos_extras = {}
    if 'texto_proibido' in caso:
        argumentos_extras['texto_proibido'] = caso['texto_proibido']
    if 'texto_esperado' in caso:
        argumentos_extras['texto_esperado'] = caso['texto_esperado']

    resultado_real = avaliador(pergunta, **contexto_execucao, **argumentos_extras)

    return nome_check, esperado, resultado_real


def main():
    print('Carregando Max para avaliação...')
    system_prompt = carregar_prompt()
    retriever = carregar_base_conhecimento()
    vectorstore_saudacoes = carregar_indice_saudacoes()
    vectorstore_saida = carregar_indice_saida()

    llm_chat = ChatOpenAI(model='gpt-4o-mini', temperature=0.3)
    llm_verificador = ChatOpenAI(model='gpt-4o-mini', temperature=0)

    contexto_execucao = {
        'retriever': retriever,
        'vectorstore_saudacoes': vectorstore_saudacoes,
        'vectorstore_saida': vectorstore_saida,
        'llm_chat': llm_chat,
        'llm_verificador': llm_verificador,
        'system_prompt': system_prompt,
    }

    casos = carregar_eval_set()
    total = len(casos)
    passou = 0
    falhou_esperado = 0
    falhou_inesperado = 0
    nao_testavel = 0

    print(f'\nRodando {total} casos...\n')

    for i, caso in enumerate(casos, start=1):
        nome_check, esperado, resultado_real = rodar_caso(caso, **contexto_execucao)

        if resultado_real is None:
            nao_testavel += 1
            status = 'NÃO TESTÁVEL (contexto vazio)'
        else:
            bateu = resultado_real == esperado
            if bateu:
                passou += 1
                status = 'PASSOU'
            elif caso.get('limitacao_conhecida'):
                falhou_esperado += 1
                status = 'FALHOU (limitação conhecida)'
            else:
                falhou_inesperado += 1
                status = 'FALHOU (INESPERADO)'

        print(f'[{i}/{total}] [{nome_check}] {status}')
        print(f'  Pergunta: {caso["pergunta"]}')
        print(f'  Esperado: {esperado} | Obtido: {resultado_real}')
        print('---')

    print('\n=== RESUMO ===')
    print(f'Total: {total}')
    print(f'Passou: {passou}')
    print(f'Falhou (limitação conhecida): {falhou_esperado}')
    print(f'Falhou (INESPERADO): {falhou_inesperado}')
    print(f'Não testável: {nao_testavel}')


if __name__ == '__main__':
    main()