import os
import pdfplumber
from collections import Counter

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def extrair_palavras(caminho_pdf):
    palavras = []

    with pdfplumber.open(caminho_pdf) as pdf:
        for pagina in pdf.pages:
            tabelas = pagina.find_tables()
            palavras_pagina = pagina.extract_words(extra_attrs=['size', 'fontname'])

            if not tabelas:
                palavras.extend(palavras_pagina)
                continue

            tabela = tabelas[0]
            top_tabela, bottom_tabela = tabela.bbox[1], tabela.bbox[3]

            ja_inseriu_tabela = False
            for palavra in palavras_pagina:
                dentro_da_tabela = top_tabela <= palavra['top'] <= bottom_tabela

                if not dentro_da_tabela:
                    palavras.append(palavra)
                elif not ja_inseriu_tabela:
                    palavras.append({'tabela': tabela.extract(), 'top': top_tabela})
                    ja_inseriu_tabela = True

    return palavras


def identificar_perfis(palavras):
    palavras_com_fonte = [p for p in palavras if 'size' in p]
    contagem = Counter((round(p['size'], 1), p['fontname']) for p in palavras_com_fonte)
    perfil_corpo = contagem.most_common(1)[0][0]
    tamanho_corpo = perfil_corpo[0]

    perfis_maiores = sorted(
        [perfil for perfil in contagem if perfil != perfil_corpo and perfil[0] > tamanho_corpo],
        key=lambda perfil: perfil[0],
        reverse=True
    )

    perfil_run_in = perfis_maiores[-1] if perfis_maiores else None
    headings = perfis_maiores[:-1] if perfis_maiores else []

    outros_perfis = [
        perfil for perfil in contagem
        if perfil != perfil_corpo and perfil != perfil_run_in and perfil not in headings
    ]

    return {
        'corpo': perfil_corpo,
        'headings': headings,
        'run_in': perfil_run_in,
        'outros': outros_perfis
    }

def formatar_linhas_tabela(tabela, titulo_secao):
    cabecalho = tabela[0]
    linhas_formatadas = []

    for linha in tabela[1:]:
        partes = [f"{cabecalho[i]}: {valor}" for i, valor in enumerate(linha) if valor]
        linhas_formatadas.append(f"{titulo_secao}. " + '. '.join(partes) + '.')

    return linhas_formatadas

def montar_chunks(palavras, perfis):
    nivel_chunk = perfis['headings'][-1]
    niveis_contexto = perfis['headings'][:-1]
    nivel_h2 = niveis_contexto[-1] if niveis_contexto else None

    chunks = []
    chunk_atual = None
    titulo_por_nivel = {nivel: '' for nivel in niveis_contexto}
    corpo_por_nivel = {nivel: [] for nivel in niveis_contexto}
    perfil_anterior = None
    aguardando_chunk = True
    teve_h3_no_h2_atual = True

    filhos_h2_orfao = []
    filho_atual = {'run_in': None, 'texto': []}

    def contexto_atual_formatado(excluir_h2=False):
        niveis = niveis_contexto[:-1] if excluir_h2 else niveis_contexto
        resultado = []
        for nivel in niveis:
            texto = titulo_por_nivel[nivel]
            if corpo_por_nivel[nivel]:
                texto += ' ' + ' '.join(corpo_por_nivel[nivel])
            if texto:
                resultado.append(texto.strip())
        return resultado

    def fechar_filho_atual():
        nonlocal filho_atual
        if filho_atual['texto']:
            destino = chunk_atual['filhos'] if (chunk_atual and teve_h3_no_h2_atual) else filhos_h2_orfao
            destino.append(filho_atual)
        filho_atual = {'run_in': None, 'texto': []}

    def fechar_h2_sem_h3():
        nonlocal filhos_h2_orfao
        if not teve_h3_no_h2_atual and titulo_por_nivel.get(nivel_h2):
            filhos = []
            if corpo_por_nivel.get(nivel_h2):
                filhos.append({'run_in': None, 'texto': corpo_por_nivel[nivel_h2]})
            filhos.extend(filhos_h2_orfao)
            chunks.append({
                'contexto': contexto_atual_formatado(excluir_h2=True),
                'titulo': titulo_por_nivel[nivel_h2],
                'filhos': filhos
            })
        filhos_h2_orfao = []

    def inserir_filhos_tabela(dados_tabela):
        titulo_secao_atual = chunk_atual['titulo'] if chunk_atual else titulo_por_nivel.get(nivel_h2, '')
        linhas = formatar_linhas_tabela(dados_tabela, titulo_secao_atual)
        destino = chunk_atual['filhos'] if (chunk_atual and teve_h3_no_h2_atual) else filhos_h2_orfao
        for linha in linhas:
            destino.append({'run_in': None, 'texto': [linha]})

    for palavra in palavras:
        if 'tabela' in palavra:
            fechar_filho_atual()
            inserir_filhos_tabela(palavra['tabela'])
            perfil_anterior = None
            continue

        chave = (round(palavra['size'], 1), palavra['fontname'])

        if chave == nivel_chunk:
            teve_h3_no_h2_atual = True
            if chave == perfil_anterior:
                chunk_atual['titulo'] += ' ' + palavra['text']
            else:
                fechar_filho_atual()
                if chunk_atual:
                    chunks.append(chunk_atual)
                chunk_atual = {
                    'contexto': contexto_atual_formatado(),
                    'titulo': palavra['text'],
                    'filhos': []
                }
                aguardando_chunk = False

        elif chave in niveis_contexto:
            if chave == nivel_h2 and chave != perfil_anterior:
                fechar_filho_atual()
                fechar_h2_sem_h3()
                teve_h3_no_h2_atual = False

            if chave == perfil_anterior:
                titulo_por_nivel[chave] += ' ' + palavra['text']
            else:
                titulo_por_nivel[chave] = palavra['text']
                corpo_por_nivel[chave] = []
            aguardando_chunk = True

        elif chave == perfis['run_in']:
            if chave == perfil_anterior:
                filho_atual['run_in'] += ' ' + palavra['text']
            else:
                fechar_filho_atual()
                filho_atual['run_in'] = palavra['text']
            aguardando_chunk = False

        else:
            if aguardando_chunk:
                ultimo_nivel = max(titulo_por_nivel, key=lambda n: niveis_contexto.index(n))
                corpo_por_nivel[ultimo_nivel].append(palavra['text'])
            else:
                filho_atual['texto'].append(palavra['text'])

        perfil_anterior = chave

    fechar_filho_atual()
    if chunk_atual:
        chunks.append(chunk_atual)
    fechar_h2_sem_h3()

    return chunks

def buscar_perfil_por_texto(palavras, textos_procurados):
    encontrados = {texto: [] for texto in textos_procurados}
    for palavra in palavras:
        for texto in textos_procurados:
            if texto in palavra['text']:
                perfil = (round(palavra['size'], 1), palavra['fontname'])
                encontrados[texto].append((palavra['text'], perfil))
    return encontrados


def explorar_tabelas(caminho_pdf):
    with pdfplumber.open(caminho_pdf) as pdf:
        for num_pagina, pagina in enumerate(pdf.pages, start=1):
            tabelas = pagina.extract_tables()
            if tabelas:
                print(f"\n===== PÁGINA {num_pagina}: {len(tabelas)} TABELA(S) ENCONTRADA(S) =====")
                for i, tabela in enumerate(tabelas, start=1):
                    print(f"\n--- Tabela {i} ({len(tabela)} linhas) ---")
                    for linha in tabela:
                        print(linha)

def testar_extracao_com_tabela(caminho_pdf):
    palavras = extrair_palavras(caminho_pdf)

    for i, item in enumerate(palavras):
        if 'tabela' in item:
            print(f"\n===== MARCADOR DE TABELA ENCONTRADO NO ÍNDICE {i} =====")
            print(f"Palavra anterior: {palavras[i - 1]['text']}")
            print(f"Palavra seguinte: {palavras[i + 1]['text']}")
            print(f"Dados brutos da tabela: {item['tabela']}")

if __name__ == '__main__':
    MOSTRAR_CHUNKS = True
    MOSTRAR_BUSCA_DIRIGIDA = False
    MOSTRAR_TABELAS = True

    caminho_pdf = os.path.join(BASE_DIR, 'data', 'politicas_xyz.pdf')

    palavras = extrair_palavras(caminho_pdf)
    perfis = identificar_perfis(palavras)

    print(f"\n===== PERFIS IDENTIFICADOS =====")
    print(f"Corpo: {perfis['corpo']}")
    print(f"Headings (maior -> menor): {perfis['headings']}")
    print(f"Run-in: {perfis['run_in']}")
    print(f"Outros (candidato a tabela): {perfis['outros']}")

    chunks = montar_chunks(palavras, perfis)

    if MOSTRAR_CHUNKS:
        print(f"\n===== {len(chunks)} CHUNKS CRIADOS =====")
        for i, chunk in enumerate(chunks, start=1):
            print(f"\n--- Chunk {i} ---")
            print(f"Contexto: {chunk['contexto']}")
            print(f"Título: {chunk['titulo']}")
            print(f"{len(chunk['filhos'])} filho(s):")
            for j, filho in enumerate(chunk['filhos'], start=1):
                texto_filho = ' '.join(filho['texto'])
                print(f"  Filho {j} | run-in: {filho['run_in']} | {len(texto_filho)} caracteres")
                print(f"    {texto_filho[:150]}...")

    if MOSTRAR_BUSCA_DIRIGIDA:
        alvos = buscar_perfil_por_texto(palavras, ['PRAZOS', 'INTERNACIONAIS', 'EXTRAVIO'])
        for texto, ocorrencias in alvos.items():
            print(f"\n'{texto}' encontrado em:")
            for texto_real, perfil in ocorrencias:
                print(f"  {texto_real} -> {perfil}")

    if MOSTRAR_TABELAS:
        explorar_tabelas(caminho_pdf)