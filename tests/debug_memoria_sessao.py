import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage

from memoria_sessao import resumir_se_necessario


def montar_conversa_falsa():
    mensagens = [SystemMessage(content='Você é o Max, assistente da XYZ Entregas.')]

    trocas = [
        ('Meu pedido está atrasado, moro no Sul.', 'Para pedidos no Sul, o prazo padrão é de 3 a 4 dias úteis.'),
        ('Já passou desse prazo.', 'Nesse caso, aguarde até 48 horas adicionais antes de abrir uma investigação formal.'),
        ('E se eu não receber depois dessas 48h?', 'Após esse prazo, se o pedido ainda não constar como entregue, uma investigação será iniciada.'),
        ('Como eu abro essa investigação?', 'Você pode entrar em contato com o suporte informando o número do pedido.'),
        ('Meu número de pedido é 12345.', 'Anotado. Mais alguma coisa que eu possa ajudar?'),
        ('Qual o prazo pra reembolso, caso aconteça?', 'O reembolso é processado em até 5 dias úteis após a confirmação.'),
    ]

    for pergunta, resposta in trocas:
        mensagens.append(HumanMessage(content=pergunta))
        mensagens.append(AIMessage(content=resposta))

    return mensagens


def montar_conversa_falsa_com_tool_call():
    mensagens = montar_conversa_falsa()

    mensagens.append(HumanMessage(content='Teve alguma novidade sobre isso?'))

    ai_com_tool_call = AIMessage(
        content='',
        tool_calls=[{'name': 'buscar_historico_anterior', 'args': {'cliente_id': 8}, 'id': 'call_teste_debug'}]
    )
    mensagens.append(ai_com_tool_call)
    mensagens.append(ToolMessage(content='Nenhum histórico de conversa anterior encontrado.', tool_call_id='call_teste_debug'))
    mensagens.append(AIMessage(content='Ainda não temos novidades sobre seu pedido.'))

    return mensagens


def imprimir_mensagens(titulo, mensagens):
    print(f'--- {titulo} ({len(mensagens)} mensagens) ---')
    for i, m in enumerate(mensagens):
        tipo = type(m).__name__
        conteudo = m.content if m.content else '(sem conteúdo)'
        print(f'  [{i}] {tipo}: {conteudo[:80]}')
    print()


def rodar_teste(nome, mensagens, llm_resumo):
    print(f'=== {nome} ===\n')
    imprimir_mensagens('Antes', mensagens)

    estado = {'messages': mensagens}
    resumir_se_necessario(estado, llm_resumo)

    imprimir_mensagens('Depois', estado['messages'])

    for m in estado['messages']:
        if isinstance(m, SystemMessage) and m.content.startswith('Resumo da conversa'):
            print('Resumo gerado:')
            print(f'  {m.content}\n')


def main():
    llm_resumo = ChatOpenAI(model='gpt-4o-mini', temperature=0)

    rodar_teste('Teste 1: conversa simples acima do limiar', montar_conversa_falsa(), llm_resumo)
    rodar_teste('Teste 2: conversa com par AIMessage(tool_calls)->ToolMessage perto do corte', montar_conversa_falsa_com_tool_call(), llm_resumo)


if __name__ == '__main__':
    main()