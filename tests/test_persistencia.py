import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage

from inicializacao import carregar_prompt, carregar_base_conhecimento, carregar_indice_saudacoes, carregar_indice_saida
from db.models import Cliente, Conversa, Mensagem
from db.session import obter_session
from tools import todas_as_tools
from processamento import processar_pergunta

from datetime import datetime


def criar_cliente_e_conversa(session):
    cliente = Cliente(nome='Cliente Teste Persistencia', email=f'teste_persistencia_{datetime.now().timestamp()}@email.com')
    session.add(cliente)
    session.commit()

    conversa = Conversa(cliente_id=cliente.id)
    session.add(conversa)
    session.commit()

    return cliente, conversa


def montar_estado(session, cliente, conversa, system_prompt, retriever, vectorstore_saudacoes, vectorstore_saida, llm_chat, llm_chat_com_tools, llm_verificador):
    return {
        'session': session,
        'conversa': conversa,
        'cliente': cliente,
        'messages': [SystemMessage(content=system_prompt)],
        'tentativas_sem_contexto': 0,
        'retriever': retriever,
        'vectorstore_saudacoes': vectorstore_saudacoes,
        'vectorstore_saida': vectorstore_saida,
        'llm_chat': llm_chat,
        'llm_chat_com_tools': llm_chat_com_tools,
        'llm_verificador': llm_verificador,
    }


def contar_mensagens(session, conversa_id):
    return session.query(Mensagem).filter_by(conversa_id=conversa_id).count()


def teste_resposta_normal_persiste_pergunta_e_resposta(componentes):
    print('--- Teste 1: resposta normal persiste pergunta E resposta ---')

    with obter_session() as session:
        cliente, conversa = criar_cliente_e_conversa(session)
        estado = montar_estado(session, cliente, conversa, **componentes)

        resultado = processar_pergunta('qual o prazo de entrega para o sul?', estado)
        total_mensagens = contar_mensagens(session, conversa.id)

    print(f'Tipo de retorno: {resultado["tipo"]} (esperado: resposta)')
    print(f'Transferiu: {resultado.get("transferiu")} (esperado: False)')
    print(f'Mensagens persistidas: {total_mensagens} (esperado: 2 — pergunta + resposta)')

    passou = resultado['tipo'] == 'resposta' and resultado.get('transferiu') is False and total_mensagens == 2
    print('PASSOU' if passou else 'FALHOU')
    print()


def teste_grounding_falhou_persiste_so_pergunta(componentes):
    print('--- Teste 2: transferência por grounding_falhou persiste só a pergunta ---')

    with obter_session() as session:
        cliente, conversa = criar_cliente_e_conversa(session)
        estado = montar_estado(session, cliente, conversa, **componentes)

        resultado = processar_pergunta('posso trocar meu pedido por um produto de valor maior pagando a diferença?', estado)
        total_mensagens = contar_mensagens(session, conversa.id)

    print(f'Tipo de retorno: {resultado["tipo"]}')
    print(f'Motivo: {resultado.get("motivo")}')
    print(f'Mensagens persistidas: {total_mensagens} (esperado: 1 — só a pergunta)')

    passou = resultado['tipo'] == 'resposta' and resultado.get('transferiu') is True and total_mensagens == 1
    print('PASSOU' if passou else 'FALHOU (nota: esse caso depende de o pipeline transferir de fato — ver aviso abaixo)')
    print()


def teste_sem_contexto_nao_persiste_nada(componentes):
    print('--- Teste 3: sem_contexto não persiste nada (pergunta nem chega a ser salva) ---')

    with obter_session() as session:
        cliente, conversa = criar_cliente_e_conversa(session)
        estado = montar_estado(session, cliente, conversa, **componentes)

        resultado = processar_pergunta('copa do mundo fifa', estado)
        total_mensagens = contar_mensagens(session, conversa.id)

    print(f'Tipo de retorno: {resultado["tipo"]} (esperado: sem_contexto)')
    print(f'Mensagens persistidas: {total_mensagens} (esperado: 0)')

    passou = resultado['tipo'] == 'sem_contexto' and total_mensagens == 0
    print('PASSOU' if passou else 'FALHOU')
    print()


def teste_saudacao_nao_persiste_nada(componentes):
    print('--- Teste 4: saudação não persiste nada ---')

    with obter_session() as session:
        cliente, conversa = criar_cliente_e_conversa(session)
        estado = montar_estado(session, cliente, conversa, **componentes)

        resultado = processar_pergunta('oi', estado)
        total_mensagens = contar_mensagens(session, conversa.id)

    print(f'Tipo de retorno: {resultado["tipo"]} (esperado: saudacao)')
    print(f'Mensagens persistidas: {total_mensagens} (esperado: 0)')

    passou = resultado['tipo'] == 'saudacao' and total_mensagens == 0
    print('PASSOU' if passou else 'FALHOU')
    print()


def main():
    print('Carregando componentes do Max...')
    componentes = {
        'system_prompt': carregar_prompt(),
        'retriever': carregar_base_conhecimento(),
        'vectorstore_saudacoes': carregar_indice_saudacoes(),
        'vectorstore_saida': carregar_indice_saida(),
        'llm_chat': ChatOpenAI(model='gpt-4o-mini', temperature=0.3, streaming=True),
        'llm_verificador': ChatOpenAI(model='gpt-4o-mini', temperature=0),
    }
    componentes['llm_chat_com_tools'] = componentes['llm_chat'].bind_tools(todas_as_tools)

    print()
    teste_resposta_normal_persiste_pergunta_e_resposta(componentes)
    teste_grounding_falhou_persiste_so_pergunta(componentes)
    teste_sem_contexto_nao_persiste_nada(componentes)
    teste_saudacao_nao_persiste_nada(componentes)


if __name__ == '__main__':
    main()