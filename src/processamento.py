from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

from busca_semantica import buscar_contexto, eh_saudacao, eh_intencao_saida
from verificacao_llm import verificar_grounding, verificar_informacao_suficiente
from memoria_sessao import resumir_se_necessario
from db.models import Mensagem
from tools import todas_as_tools


def processar_pergunta(pergunta, estado):
    if eh_intencao_saida(estado['vectorstore_saida'], pergunta):
        return {'tipo': 'saida'}

    if eh_saudacao(estado['vectorstore_saudacoes'], pergunta):
        return {'tipo': 'saudacao'}

    if verificar_informacao_suficiente(estado['llm_verificador'], pergunta):
        return {'tipo': 'needs_more_information'}

    contexto = buscar_contexto(estado['retriever'], pergunta)

    if not contexto.strip():
        estado['tentativas_sem_contexto'] += 1
        transferiu = estado['tentativas_sem_contexto'] >= 2
        return {'tipo': 'sem_contexto', 'transferiu': transferiu}

    estado['tentativas_sem_contexto'] = 0

    mensagem_cliente = Mensagem(conversa_id=estado['conversa'].id, remetente='cliente', conteudo=pergunta)
    estado['session'].add(mensagem_cliente)
    estado['session'].commit()

    mensagem_com_contexto = f'{pergunta}\n\nInformações relevantes:\n{contexto}\n\nO cliente_id do cliente atual é {estado["cliente"].id}.'
    estado['messages'].append(HumanMessage(content=mensagem_com_contexto))
    resumir_se_necessario(estado, estado['llm_verificador'])

    resposta_decisao = estado['llm_chat_com_tools'].invoke(estado['messages'])
    tool_calls = resposta_decisao.tool_calls

    if tool_calls:
        estado['messages'].append(resposta_decisao)
        for chamada in tool_calls:
            tool_chamada = next(t for t in todas_as_tools if t.name == chamada['name'])
            resultado = tool_chamada.invoke(chamada['args'])
            estado['messages'].append(ToolMessage(content=resultado, tool_call_id=chamada['id']))

    reply = ''
    for chunk in estado['llm_chat'].stream(estado['messages']):
        texto = chunk.content
        if texto:
            reply += texto

    if 'TRANSFER_HUMANO' in reply.upper():
        return {'tipo': 'resposta', 'reply': reply, 'transferiu': True, 'motivo': 'transfer_humano', 'tool_calls': tool_calls}

    grounding_falhou = verificar_grounding(estado['llm_verificador'], pergunta, contexto, reply)

    if grounding_falhou:
        return {'tipo': 'resposta', 'reply': reply, 'transferiu': True, 'motivo': 'grounding_falhou', 'tool_calls': tool_calls}

    estado['messages'].append(AIMessage(content=reply))

    mensagem_max = Mensagem(conversa_id=estado['conversa'].id, remetente='max', conteudo=reply)
    estado['session'].add(mensagem_max)
    estado['session'].commit()

    return {'tipo': 'resposta', 'reply': reply, 'transferiu': False, 'motivo': None, 'tool_calls': tool_calls}