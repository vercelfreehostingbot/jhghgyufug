import os
import pdfplumber
from collections import Counter

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
caminho_pdf = os.path.join(BASE_DIR, 'data', 'politicas_xyz.pdf')

with pdfplumber.open(caminho_pdf) as pdf:
    palavras = []
    for pagina in pdf.pages:
        palavras.extend(pagina.extract_words(extra_attrs=['size', 'fontname']))

perfis = Counter((round(p['size'], 1), p['fontname']) for p in palavras)

print(f"\n===== {len(perfis)} PERFIS DE FONTE ENCONTRADOS =====")
for perfil, contagem in perfis.most_common():
    print(f"{perfil} -> {contagem} palavras")

print("\n===== EXEMPLOS POR PERFIL =====")
for perfil, _ in perfis.most_common():
    exemplos = [p['text'] for p in palavras if (round(p['size'], 1), p['fontname']) == perfil][:8]
    print(f"\n{perfil}: {exemplos}")