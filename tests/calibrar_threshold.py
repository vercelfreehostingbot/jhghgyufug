import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from dotenv import load_dotenv
load_dotenv()

from inicializacao import carregar_base_conhecimento

# Perguntas com resposta clara e específica na base, cobrindo capítulos
# diferentes -- servem pra ver onde o filho CORRETO tende a pontuar.
QUERIES_RELEVANTES = [
    "qual o prazo pro sul?",
    "meu pedido está atrasado, o que eu faço?",
    "meu pedido foi extraviado",
    "quanto tempo demora pra receber o reembolso?",
    "qual o horário de atendimento?",
    "recebi um produto com avaria",
    "posso cancelar meu pedido antes de ser despachado?",
    "qual o prazo de entrega pra portugal?",
]

# Já medida: faixa de ruído de fundo pra uma query totalmente fora do domínio.
QUERY_FORA_DO_DOMINIO = "copa do mundo fifa"

TOP_N_EXIBIDO = 6


def rodar_query(vectorstore, query):
    resultados = vectorstore.similarity_search_with_relevance_scores(query, k=28)
    scores = [score for _, score in resultados]

    print(f"\n===== \"{query}\" =====")
    print(f"Maior score: {max(scores):.4f} | Menor score: {min(scores):.4f}")
    print(f"Top {TOP_N_EXIBIDO}:")
    for doc, score in resultados[:TOP_N_EXIBIDO]:
        print(f"  {score:.4f} | {doc.page_content[:90]}")


def main():
    retriever = carregar_base_conhecimento()
    vectorstore = retriever.vectorstore

    for query in QUERIES_RELEVANTES:
        rodar_query(vectorstore, query)

    rodar_query(vectorstore, QUERY_FORA_DO_DOMINIO)


if __name__ == '__main__':
    main()