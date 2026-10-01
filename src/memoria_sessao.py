from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage

LIMIAR_MENSAGENS = 10
MENSAGENS_RECENTES_MANTIDAS = 4


def _contar_mensagens_de_conversa(mensagens):
    return [m for m in mensagens if isinstance(m, (HumanMessage, AIMessage))]


def _calcular_ponto_de_corte(mensagens):
    contaveis_vistas = 0
    indice_corte = len(mensagens)

    for i in range(len(mensagens) - 1, 0, -1):
        if isinstance(mensagens[i], (HumanMessage, AIMessage)):
            contaveis_vistas += 1
        indice_corte = i
        if contaveis_vistas >= MENSAGENS_RECENTES_MANTIDAS:
            break

    while indice_corte > 1 and isinstance(mensagens[indice_corte], ToolMessage):
        indice_corte -= 1

    return indice_corte


def _formatar_mensagens_para_resumo(mensagens):
    linhas = []
    for m in mensagens:
        if isinstance(m, HumanMessage):
            linhas.append(f'Cliente: {m.content}')
        elif isinstance(m, AIMessage) and m.content:
            linhas.append(f'Max: {m.content}')
        elif isinstance(m, ToolMessage):
            linhas.append(f'[Resultado de ferramenta]: {m.content}')
    return '\n'.join(linhas)


def resumir_se_necessario(estado, llm_resumo):
    mensagens = estado['messages']
    conversa = _contar_mensagens_de_conversa(mensagens[1:])

    if len(conversa) <= LIMIAR_MENSAGENS:
        return

    indice_corte = _calcular_ponto_de_corte(mensagens)
    mais_antigas = mensagens[1:indice_corte]
    recentes = mensagens[indice_corte:]

    if not mais_antigas:
        return

    texto_para_resumir = _formatar_mensagens_para_resumo(mais_antigas)

    prompt = f"""Resuma a conversa de suporte abaixo entre um cliente e o assistente Max, preservando fatos relevantes (região, tipo de problema relatado, políticas já explicadas, decisões já tomadas). Seja objetivo, sem repetir frases literais desnecessariamente.

Conversa:
{texto_para_resumir}

Resumo:"""

    resumo = llm_resumo.invoke(prompt).content
    mensagem_resumo = SystemMessage(content=f'Resumo da conversa até este ponto: {resumo}')

    estado['messages'] = [mensagens[0], mensagem_resumo] + recentes