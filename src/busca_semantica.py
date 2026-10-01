def buscar_contexto(retriever, pergunta):
    docs = retriever.invoke(pergunta)
    return '\n'.join([doc.page_content for doc in docs])

def eh_saudacao(vectorstore_saudacoes, pergunta):
    resultados = vectorstore_saudacoes.similarity_search_with_relevance_scores(pergunta, k=1)
    _, score = resultados[0]
    return score >= 0.85

def eh_intencao_saida(vectorstore_saida, pergunta):
    resultados = vectorstore_saida.similarity_search_with_relevance_scores(pergunta, k=1)
    _, score = resultados[0]
    return score >= 0.85