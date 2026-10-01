import os
import logging

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.makedirs(os.path.join(BASE_DIR, 'logs'), exist_ok=True)

logging.basicConfig(
    filename=os.path.join(BASE_DIR, 'logs', 'app.log'),
    level=logging.WARNING,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logging.captureWarnings(True)

from inicializacao import carregar_prompt, carregar_base_conhecimento, carregar_indice_saudacoes, carregar_indice_saida
from db.models import Cliente, Conversa
from db.session import obter_session
from tools import todas_as_tools
from processamento import processar_pergunta
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage

load_dotenv()


def identificar_cliente(session):
    email = input('[Simulando login do site] Email do cliente: ').strip()

    cliente = session.query(Cliente).filter_by(email=email).first()

    if cliente:
        print(f'[Sistema]: Bem-vindo de volta, {cliente.nome}!')
    else:
        nome = input('[Sistema]: Cliente novo. Qual o seu nome? ').strip()
        cliente = Cliente(nome=nome, email=email)
        session.add(cliente)
        session.commit()
        print(f'[Sistema]: Cadastro criado, {cliente.nome}!')

    return cliente


def main():
    print('Carregando Max...')
    system_prompt = carregar_prompt()
    retriever = carregar_base_conhecimento()
    vectorstore_saudacoes = carregar_indice_saudacoes()
    vectorstore_saida = carregar_indice_saida()

    llm_chat = ChatOpenAI(model='gpt-4o-mini', temperature=0.3, streaming=True)
    llm_chat_com_tools = llm_chat.bind_tools(todas_as_tools)
    llm_verificador = ChatOpenAI(model='gpt-4o-mini', temperature=0)

    with obter_session() as session:
        cliente = identificar_cliente(session)
        conversa = Conversa(cliente_id=cliente.id)
        session.add(conversa)
        session.commit()

        estado = {
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

        print('Max: Olá! Sou o Max, assistente da XYZ Entregas. Como posso ajudar?')

        while True:
            pergunta = input('Você: ')
            resultado = processar_pergunta(pergunta, estado)

            if resultado['tipo'] == 'saida':
                print('Max: Até mais!')
                break

            if resultado['tipo'] == 'saudacao':
                print('Max: Olá! Como posso te ajudar hoje?')
                print()
                continue

            if resultado['tipo'] == 'needs_more_information':
                print('Max: Para te ajudar melhor, preciso de mais alguns detalhes. Você pode informar sua região e, se possível, há quantos dias fez o pedido?')
                print()
                continue

            if resultado['tipo'] == 'sem_contexto':
                if resultado['transferiu']:
                    print('Max: Não consegui entender sua solicitação. Vou te transferir para um atendente.')
                    print('[Sistema]: Transferindo...')
                    break
                print('Max: Não entendi muito bem sua pergunta. Você pode explicar de outra forma, com mais detalhes sobre seu pedido?')
                print()
                continue

            if resultado['tipo'] == 'resposta':
                if resultado['motivo'] == 'transfer_humano':
                    print('Max: Aguarde, vou transferir para um atendente.')
                    print('[Sistema]: Transferindo...')
                    break

                if resultado['motivo'] == 'grounding_falhou':
                    print('Max: Não tenho essa informação específica no momento, vou te transferir para um atendente humano que pode te ajudar melhor.')
                    print('[Sistema]: Transferindo...')
                    break

                print(f"Max: {resultado['reply']}")
                print()


if __name__ == '__main__':
    main()