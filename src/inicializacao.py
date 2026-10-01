import os
import json
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from extracao_pdf import extrair_palavras, identificar_perfis, montar_chunks, montar_documents_parent_child

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class RetrieverPaiFilho(BaseRetriever):
    vectorstore: object
    pais: dict
    k: int = 4
    score_threshold: float = 0.70

    def _get_relevant_documents(self, query, *, run_manager=None):
        resultados = self.vectorstore.similarity_search_with_relevance_scores(query, k=self.k)

        pais_encontrados = []
        ids_ja_adicionados = set()

        for doc_filho, score in resultados:
            if score < self.score_threshold:
                continue
            doc_id = doc_filho.metadata['doc_id']
            if doc_id in ids_ja_adicionados:
                continue
            ids_ja_adicionados.add(doc_id)
            pais_encontrados.append(self.pais[doc_id])

        return pais_encontrados


def carregar_prompt():
    with open(os.path.join(BASE_DIR, 'prompts', 'system.txt'), 'r', encoding='utf-8') as f:
        return f.read()


def salvar_pais(pais, caminho):
    dados = {doc_id: doc.page_content for doc_id, doc in pais.items()}
    with open(caminho, 'w', encoding='utf-8') as f:
        json.dump(dados, f, ensure_ascii=False)


def carregar_pais(caminho):
    with open(caminho, 'r', encoding='utf-8') as f:
        dados = json.load(f)
    return {doc_id: Document(page_content=texto) for doc_id, texto in dados.items()}


def carregar_base_conhecimento():
    caminho_pdf = os.path.join(BASE_DIR, 'data', 'politicas_xyz.pdf')
    caminho_indice = os.path.join(BASE_DIR, 'data', 'faiss_index')
    caminho_metadata = os.path.join(BASE_DIR, 'data', 'faiss_metadata.txt')
    caminho_pais = os.path.join(BASE_DIR, 'data', 'parent_docstore.json')

    data_modificacao_atual = str(os.path.getmtime(caminho_pdf))
    embeddings = OpenAIEmbeddings()

    indice_existe = os.path.exists(caminho_indice)
    metadata_existe = os.path.exists(caminho_metadata)
    pais_existe = os.path.exists(caminho_pais)

    if indice_existe and metadata_existe and pais_existe:
        with open(caminho_metadata, 'r') as f:
            data_modificacao_salva = f.read().strip()

        if data_modificacao_salva == data_modificacao_atual:
            print('Índice já atualizado, carregando do disco...')
            vectorstore = FAISS.load_local(caminho_indice, embeddings, allow_dangerous_deserialization=True)
            pais = carregar_pais(caminho_pais)
            return RetrieverPaiFilho(vectorstore=vectorstore, pais=pais)

    print('Recalculando embeddings...')
    palavras = extrair_palavras(caminho_pdf)
    perfis = identificar_perfis(palavras)
    chunks = montar_chunks(palavras, perfis)
    pais, documentos_filhos = montar_documents_parent_child(chunks)

    print(f"\n===== {len(chunks)} CHUNKS -> {len(pais)} PAIS, {len(documentos_filhos)} FILHOS =====")

    vectorstore = FAISS.from_documents(documentos_filhos, embeddings)

    vectorstore.save_local(caminho_indice)
    with open(caminho_metadata, 'w') as f:
        f.write(data_modificacao_atual)
    salvar_pais(pais, caminho_pais)

    return RetrieverPaiFilho(vectorstore=vectorstore, pais=pais)


def carregar_indice_saudacoes():
    exemplos_saudacao = ['olá', 'oi', 'oii', 'bom dia', 'boa tarde', 'boa noite', 'tudo bem', 'e aí', 'opa', 'salve']
    documentos_saudacao = [Document(page_content=texto) for texto in exemplos_saudacao]
    embeddings = OpenAIEmbeddings()
    return FAISS.from_documents(documentos_saudacao, embeddings)

def carregar_indice_saida():
    exemplos_saida = ['sair', 'quero sair', 'quero encerrar', 'encerrar conversa', 'tchau', 'até mais', 'pode encerrar', 'finalizar atendimento']
    documentos_saida = [Document(page_content=texto) for texto in exemplos_saida]
    embeddings = OpenAIEmbeddings()
    return FAISS.from_documents(documentos_saida, embeddings)
