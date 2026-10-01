# Experimentos técnicos — Retrieval e Alucinação

Este documento registra uma investigação prática sobre os parâmetros centrais do pipeline de RAG do Max: `chunk_size`, `k` (top-k retrieval), `score_threshold` e o comportamento de alucinação do modelo mesmo com contexto correto disponível. O objetivo foi entender, com testes reais, os trade-offs de cada decisão — não apenas usar valores padrão.

## Índice

- [1. Por que RAG neste projeto](#1-por-que-rag-neste-projeto)
- [2. `chunk_size`: calibrando pelo conteúdo da base de conhecimento](#2-chunk_size-calibrando-pelo-conteúdo-da-base-de-conhecimento)
- [3. `k` (top-k retrieval): por que ele mascarava o problema](#3-k-top-k-retrieval-por-que-ele-mascarava-o-problema)
- [4. Inferência não fundamentada por ausência de cobertura na base de conhecimento](#4-inferência-não-fundamentada-por-ausência-de-cobertura-na-base-de-conhecimento)
- [5. `score_threshold`: filtrando por relevância em vez de um `k` fixo](#5-score_threshold-filtrando-por-relevância-em-vez-de-um-k-fixo)
- [6. Limitação do `score_threshold`: transferência prematura sem contexto](#6-limitação-do-score_threshold-transferência-prematura-sem-contexto)
- [7. Bug de marcador de controle confundido com linguagem natural](#7-bug-de-marcador-de-controle-confundido-com-linguagem-natural)
- [8. Grounding verification: bloqueando inferências não fundamentadas](#8-grounding-verification-bloqueando-inferências-não-fundamentadas)
- [9. Persistência do índice FAISS: eliminando reprocessamento desnecessário](#9-persistência-do-índice-faiss-eliminando-reprocessamento-desnecessário)
- [10. Separando logs técnicos da interface do usuário](#10-separando-logs-técnicos-da-interface-do-usuário)
- [11. Detecção de saudação: evitando penalizar transferência por conversa social](#11-detecção-de-saudação-evitando-penalizar-transferência-por-conversa-social)
- [12. Ambiguidade entre políticas semanticamente próximas: chunking, retrieval e geração](#12-ambiguidade-entre-políticas-semanticamente-próximas-chunking-retrieval-e-geração)
- [13. Eval set automatizado: da validação manual a um script reproduzível](#13-eval-set-automatizado-da-validação-manual-a-um-script-reproduzível)
- [14. Separação de responsabilidades: reorganizando `main.py` em módulos](#14-separação-de-responsabilidades-reorganizando-mainpy-em-módulos)
- [15. Conclusões gerais](#15-conclusões-gerais)
- [16. Próximos passos identificados (não implementados ainda)](#16-próximos-passos-identificados-não-implementados-ainda)

---

# Parte 2 — Depois da migração para base em PDF: evoluções e novas dependências

- [17. Migração da base de conhecimento para PDF: extração estrutural via metadados de fonte](#17-migração-da-base-de-conhecimento-para-pdf-extração-estrutural-via-metadados-de-fonte)
- [18. Ambiguidade nacional/internacional: escopo de `needs_more_information` vs. query rewriting/HyDE](#18-ambiguidade-nacionalinternacional-escopo-de-needs_more_information-vs-query-rewritinghyde)
- [19. Query rewriting e HyDE: resultado negativo — nenhuma técnica encontrou cenário real de uso](#19-query-rewriting-e-hyde-resultado-negativo--nenhuma-técnica-encontrou-cenário-real-de-uso)
- [20. Few-shot prompting sistemático: mapeamento do eval set, correção de bugs estruturais e estabilização por camada](#20-few-shot-prompting-sistemático-mapeamento-do-eval-set-correção-de-bugs-estruturais-e-estabilização-por-camada)
- [21. Memória de conversa e tool calling: da primeira integração de banco de dados no Max à decisão de arquitetura para retomada de assunto](#21-memória-de-conversa-e-tool-calling-da-primeira-integração-de-banco-de-dados-no-max-à-decisão-de-arquitetura-para-retomada-de-assunto)
- [22. Transferência inesperada pós-tool-calling: separando duas causas distintas e uma regra de negócio desatualizada](#22-transferência-inesperada-pós-tool-calling-separando-duas-causas-distintas-e-uma-regra-de-negócio-desatualizada)
- [23. Validação pré-merge: refatoração para testabilidade, cobertura de persistência e o critério real de acionamento da tool](#23-validação-pré-merge-refatoração-para-testabilidade-cobertura-de-persistência-e-o-critério-real-de-acionamento-da-tool)
- [24. Memória de sessão: resumo progressivo para conter o crescimento ilimitado de `messages`](#24-memória-de-sessão-resumo-progressivo-para-conter-o-crescimento-ilimitado-de-messages)
- [25. Structured output: implementação, investigação de instabilidade e reversão](#25-structured-output-implementação-investigação-de-instabilidade-e-reversão)

## 1. Por que RAG neste projeto

O Max responde perguntas sobre políticas de uma empresa fictícia de entregas. Um LLM genérico não tem conhecimento sobre essas políticas — sem RAG, ele responderia com base em suposições (alucinação) ou se recusaria a responder. RAG resolve isso buscando, a cada pergunta, apenas o trecho relevante da base de conhecimento e injetando esse trecho no prompt, em vez de:

- fazer fine-tuning (caro, lento, precisa retreinar a cada mudança de política);
- ou colar a base inteira no prompt (caro em tokens, e o modelo perde precisão com excesso de informação irrelevante).

## 2. `chunk_size`: calibrando pelo conteúdo da base de conhecimento

**Teste:** medi o tamanho real de cada bloco de política no arquivo de conhecimento (~150–200 caracteres por regra) e comparei com o `chunk_size=500` usado inicialmente.

**Resultado:** com 500, os chunks gerados (400–436 caracteres) misturavam 2–3 tópicos distintos em um único chunk — por exemplo, dados institucionais, prazos de entrega e política de reembolso no mesmo bloco. Reduzindo para `chunk_size=200`, os chunks passaram a corresponder a uma única regra de negócio por vez.

**Conclusão:** `chunk_size` deveria ser definido a partir do tamanho natural das unidades de sentido do conteúdo, não copiado de um exemplo genérico. Chunks grandes demais geram contexto ruidoso (informação irrelevante misturada); chunks pequenos demais podem fragmentar uma regra no meio — um bloco de política com mais de 200 caracteres, por exemplo, é dividido em dois chunks distintos, separando uma condição da sua consequência.

## 3. `k` (top-k retrieval): por que ele mascarava o problema

`.as_retriever()` sem parâmetros usa um valor padrão do LangChain, `k=4` — não declarado explicitamente em nenhum lugar do código-fonte original.

**Observação:** como a base de conhecimento deste projeto tem apenas ~6–7 blocos de política, `k=4` recuperava quase a base inteira em qualquer pergunta. Isso mascarava a fragmentação causada por `chunk_size`: mesmo quando um chunk relevante vinha cortado, a informação faltante costumava aparecer em outro chunk vizinho, também recuperado.

**Teste com `k=2`:** reduzindo o valor, ficou mais fácil observar quando um chunk relevante ficava de fora da resposta.

**Conclusão:** o efeito de `chunk_size` e `k` é interdependente, e o tamanho da base de conhecimento determina se um problema fica visível ou escondido. Um `chunk_size` fragmentado combinado com um `k` proporcionalmente baixo em uma base grande (milhares de documentos) deixaria muito mais informação relevante de fora do que em uma base pequena como esta.

**Nota:** o `k=2` foi usado aqui apenas como teste de diagnóstico, não como configuração final — o objetivo era tornar visível o efeito da fragmentação que `k=4` estava mascarando. Esse experimento evidenciou as limitações de depender só de um número fixo de documentos, motivando a adoção de um `score_threshold` (limiar de similaridade), descrito na seção seguinte.

## 4. Inferência não fundamentada por ausência de cobertura na base de conhecimento

Pergunta de teste:

> "Meu pedido está atrasado só um pouco, ainda não chegou mas também não sumiu, o que eu faço?"

No momento desse experimento, esse cenário — atraso simples, sem extravio — ainda não estava coberto explicitamente na base de conhecimento, que só definia regras para "extravio" e para "pedido que consta como entregue mas não recebido".

**Resultado:** o modelo combinou dois fatos reais (prazos de entrega por região + regra de pedido não recebido) para gerar uma recomendação plausível ("aguarde mais um pouco"), que não está escrita em nenhum lugar da base. Isso persistiu através de três formulações diferentes de instrução no *system prompt* — proibição direta, checagem explícita de "posso responder isso?", e restrição literal contra combinar informações de contextos diferentes.

**Hipótese inicial:** como primeira tentativa para entender esse comportamento, foi testada também a execução com `temperature=0`, para verificar se a resposta era consequência de uma geração mais aleatória. O comportamento permaneceu inalterado, indicando que a temperatura não era a causa do problema. Isso é consistente com o funcionamento desse parâmetro, que influencia o nível de aleatoriedade durante a geração da resposta, mas não impede que o modelo realize inferências a partir das informações disponíveis.

**Causa raiz identificada:** a investigação indicou que o modelo não estava inventando um fato aleatório; estava realizando uma inferência lógica a partir de fatos reais, algo que, na prática, não era tratado pelo modelo como uma "invenção". Em vez de responder apenas com base no que estava explicitamente documentado, o modelo extrapolava o contexto recuperado para preencher uma lacuna da base de conhecimento com uma recomendação plausível.

**Observação complementar em outro modelo:** após os experimentos do Max, um comportamento semelhante foi observado de forma independente em uma conversa com outro modelo (Claude Sonnet), fora do escopo direto deste projeto. Ao explicar por que a segunda chamada de verificação era "mais barata", o modelo combinou dois fatos reais e documentados (o custo do risco existe; o custo de API é comparativamente menor) para concluir que esse custo seria "desprezível" — uma quantificação que não era sustentada pelas premissas disponíveis. Esse comportamento observado em outro modelo sugere que esse tipo de inferência pode não estar restrito a uma implementação ou modelo específico, embora essa observação isolada não seja suficiente para estabelecer uma conclusão geral.

**Resolução aplicada (cobertura de conteúdo):** a lacuna que originou esse caso específico — ausência de uma regra para "atraso dentro do prazo" — foi fechada adicionando um bloco explícito ao `politicas.txt`:

```
Política de atraso (dentro do prazo):
- Se o pedido ainda não chegou mas o prazo de entrega da região não foi ultrapassado, é esperado que o pedido ainda esteja a caminho
- Nenhuma ação é necessária até o fim do prazo estimado
- Após o prazo da região ser ultrapassado, aplica-se a política de pedido não recebido
```

Repetindo a mesma pergunta de teste após a adição, o novo chunk foi recuperado corretamente pelo retriever, e a resposta do Max passou a ser fundamentada no conteúdo real, sem inferência: *"Se o seu pedido ainda não chegou, mas o prazo de entrega não foi ultrapassado, é esperado que ele esteja a caminho. Recomendo que aguarde um pouco mais."*

**Limitação residual observada:** mesmo fundamentada, a resposta permanece genérica ("aguarde um pouco mais"), porque o sistema não coleta nem retém dados específicos do pedido (região, data de compra) durante a conversa — não há como calcular "faltam X dias" sem essa informação. Essa limitação é diferente da alucinação original: aqui o conteúdo está correto e ancorado no contexto, apenas não é personalizado. Fica documentada como próximo passo (Seção 16), ligada ao estudo futuro de `structured output`, não à cobertura de conteúdo em si — cobrir mais regras no `politicas.txt` não resolveria a falta de dado específico do cliente.

**Nota metodológica:** esta resolução ataca apenas o caso específico testado, não o problema estrutural. Para lacunas ainda não identificadas ou cobertas, o grounding verification (Seção 8) continua sendo a única camada implementada neste projeto capaz de mitigar, de forma geral, alucinações por combinação de fatos — mas essa mitigação é parcial, não uma garantia: a própria Seção 8 documenta um falso negativo em 7 casos testados com o verificador `gpt-4o-mini`. Nenhuma camada implementada neste projeto elimina o risco por completo.

## 5. `score_threshold`: filtrando por relevância em vez de um `k` fixo

Como observado na investigação sobre `k` (Seção 3), um `k` fixo sempre retorna o mesmo número de chunks, mesmo quando nem todos são relevantes. A alternativa testada foi o `search_type='similarity_score_threshold'` do LangChain, que descarta qualquer chunk abaixo de um limiar mínimo de relevância, usando `k` apenas como teto máximo.

Com o `score_threshold` ativo, o papel do `k` muda: deixa de ser o principal filtro de relevância e passa a atuar apenas como teto máximo de chunks retornados, já que o threshold descarta antecipadamente qualquer chunk abaixo do limiar de relevância. Por esse motivo, o valor final adotado foi `k=4` — testado e confirmado no cenário de "meu pedido foi extraviado", onde 4 chunks distintos, todos genuinamente relevantes, passaram no filtro de relevância ao mesmo tempo.

**Observação técnica importante:** o FAISS, por padrão, mede distância L2 (onde menor = mais parecido), enquanto o `score_threshold` do LangChain espera um score de relevância normalizado entre 0 e 1 (onde maior = mais relevante). O LangChain faz essa conversão internamente — o valor de threshold configurado deve ser pensado nessa segunda escala.

**Metodologia de calibração:** em vez de escolher um valor por estimativa, testei o score de relevância retornado para quatro tipos de pergunta:

| Tipo de pergunta | Exemplo | Score mais alto observado |
|---|---|---|
| Específica e relevante | "prazo de entrega para o sul" | 0.85 |
| Difusa mas relevante | "meu pedido foi extraviado" | 0.72–0.77 |
| Fora do domínio | "copa do mundo fifa" | 0.63–0.66 |
| Fora do domínio | "como fazer miojo" | ~0.60 |

**Resultado:** existe uma margem de separação real (cerca de 6 a 12 pontos percentuais) entre o pior caso relevante (~0.72) e o pior caso fora do domínio (~0.66). Com base nisso, o threshold foi calibrado em `0.68` — posicionado dentro dessa margem, testado e replicado em múltiplas execuções com as mesmas perguntas.

**Configuração final:**
```python
retriever = vectorstore.as_retriever(
    search_type='similarity_score_threshold',
    search_kwargs={'score_threshold': 0.68, 'k': 4}
)
```

**Limitação identificada:** o threshold é um valor fixo calibrado empiricamente com um conjunto pequeno de perguntas de teste, não uma constante matemática. Ele reflete um trade-off consciente entre dois erros possíveis — deixar passar uma pergunta fora do domínio ou cortar contexto relevante em perguntas mais amplas —, não uma "resposta certa" universal. Em produção, um conjunto de teste maior (um eval set mais robusto) seria necessário para validar esse valor com mais confiança.

**Caso-limite descoberto posteriormente — frases de controle de conversa:** ao testar o comando "sair" (que encerra o programa via correspondência exata de texto, não via LLM), a variação "quero sair" foi testada por curiosidade e revelou um comportamento inesperado: o retriever encontrou o chunk "Horário de atendimento" com score suficiente para passar no threshold — aparentemente por proximidade semântica fraca em torno da palavra "atendimento" — mesmo a pergunta não tendo relação real de conteúdo com horário de funcionamento. Com esse contexto irrelevante em mãos, o LLM interpretou "quero sair" como um pedido de encerramento de atendimento e respondeu com o marcador de transferência, seguindo (de forma tecnicamente correta, mas indesejada) as regras do *system prompt*.

Testando variações semelhantes ("não quero mais falar", "quero ir embora"), nenhuma reproduziu o padrão — ambas geraram contexto vazio, caindo no fallback normal de reformulação. Com apenas essas duas variações testadas, os dados não são suficientes para concluir que o caso é isolado — apenas que essas duas frases específicas não bateram no mesmo ponto cego. Outras variações não testadas ("cansei", "não aguento mais isso", "quero ir daqui") poderiam, em tese, coincidir com algum chunk por proximidade semântica da mesma forma que "quero sair" coincidiu. O que os três testes confirmam com mais segurança é que a calibração original do `score_threshold` (tabela acima) testou perguntas de negócio versus perguntas fora do domínio, mas nunca testou uma terceira categoria — frases de controle de conversa (encerrar, cancelar, desistir) — que ficou fora do conjunto de calibração original. Não implementada correção para esse caso específico; fica documentado como exemplo de que mesmo uma calibração validada com múltiplos testes pode ter pontos cegos em categorias de entrada não antecipadas, e que a extensão real desse ponto cego permanece desconhecida sem um teste mais sistemático dessa categoria (ver eval set, Seção 16).

## 6. Limitação do `score_threshold`: transferência prematura sem contexto

O `score_threshold` resolve o problema de trazer chunks irrelevantes, mas introduz um efeito colateral: quando nenhum chunk atinge o limiar, o contexto retornado fica vazio, e o *system prompt* instrui o modelo a usar o marcador de transferência (`###TRANSFER_HUMANO###`) nesse caso. Isso significa que qualquer pergunta ambígua, mal formulada ou genuinamente fora do domínio resultava em transferência **imediata** para um atendente humano, sem nenhuma chance de o cliente reformular a pergunta.

**Por que isso é um problema de produto, não só técnico:** transferir para atendimento humano tem custo real — tempo de fila, carga de trabalho do atendente, e perda de contexto (o atendente não tem acesso ao histórico da conversa com o Max). Tratar "não encontrei contexto relevante" como sinônimo de "preciso de um humano" descarta casos em que o problema era simplesmente uma pergunta mal formulada, resolvível com um pedido de esclarecimento.

**Estratégia adotada:** um contador de tentativas sem contexto (`tentativas_sem_contexto`), controlado inteiramente pelo código — não pelo modelo, para evitar depender da confiabilidade do LLM em "lembrar" quantas vezes uma regra já foi aplicada.

- Na primeira vez que uma pergunta não retorna contexto relevante, o Max responde com uma mensagem fixa pedindo para o cliente reformular ou detalhar a pergunta, sem chamar o LLM.
- Se a tentativa seguinte também não retornar contexto, a transferência para atendente humano é acionada diretamente pelo código.
- Se, em qualquer momento, uma pergunta retornar contexto válido, o contador é reiniciado — o "crédito" de tentativas é renovado.

```python
if not contexto.strip():
    tentativas_sem_contexto += 1

    if tentativas_sem_contexto >= 2:
        print('Max: Não consegui entender sua solicitação. Vou te transferir para um atendente.')
        print('[Sistema]: Transferindo...')
        break

    print('Max: Não entendi muito bem sua pergunta. Você pode explicar de outra forma, com mais detalhes sobre seu pedido?')
    continue

tentativas_sem_contexto = 0
```

**Por que o controle ficou no código, e não no prompt:** essa decisão segue a mesma lição da Seção 7 — contar tentativas ou aplicar uma regra de forma consistente é um tipo de lógica que um LLM pode falhar em seguir de forma confiável ao longo de uma conversa longa. Colocando o contador como uma variável Python comum, o comportamento fica determinístico e não depende da interpretação do modelo.

**Conclusão:** um mecanismo de recuperação (`score_threshold`) que descarta contexto irrelevante precisa de uma camada de decisão adicional para não converter automaticamente "sem contexto" em "transferir para humano". Separar essas duas coisas — dar ao cliente uma chance de reformular antes de escalar — reduz transferências desnecessárias sem comprometer o fallback para casos genuinamente fora do escopo do assistente.

## 7. Bug de marcador de controle confundido com linguagem natural

Ao testar o `score_threshold` com uma pergunta fora do domínio (sem nenhum chunk retornado), o modelo deveria responder com o marcador de controle `TRANSFERIR_HUMANO`, definido no *system prompt*, para acionar a transferência para um atendente humano.

**Resultado observado:** o modelo gerou `TRANSFIRIR_HUMANO` (com erro de grafia — "transfIrir" em vez de "transfErir"). Como a checagem no código (`if 'TRANSFERIR_HUMANO' in reply`) busca a string exata, a condição não foi satisfeita, e o fluxo de transferência não foi acionado.

**Causa raiz:** o próprio *system prompt* usa, em outras regras, o verbo "transfira" (imperativo correto de "transferir", com "i"). O modelo aparentemente generalizou esse padrão de conjugação por cima do marcador de controle, que deveria ser reproduzido literalmente, e não interpretado como parte do texto em português.

**Correção aplicada:**
- Substituição do marcador por um token que não se pareça com uma palavra natural do idioma: `###TRANSFER_HUMANO###`.
- Checagem no código tornada mais tolerante a variações, verificando apenas o núcleo do token em maiúsculas: `if 'TRANSFER_HUMANO' in reply.upper()`.

**Conclusão:** confiar na reprodução exata de uma palavra-chave de controle por um LLM é frágil, especialmente quando essa palavra se assemelha a vocabulário comum do idioma usado no restante do prompt. Marcadores de controle devem ser visualmente distintos de linguagem natural, e a validação no código deve ser tolerante a pequenas variações de grafia.

## 8. Grounding verification: bloqueando inferências não fundamentadas

A Seção 4 documentou um problema que ficou em aberto por toda a investigação: quando o contexto recuperado é relacionado à pergunta, mas não cobre exatamente o cenário descrito, o modelo tende a preencher a lacuna combinando fatos reais em uma inferência não autorizada (ex.: "aguarde mais um pouco"). Nem `score_threshold`, nem o contador de tentativas resolvem esse caso — os dois só agem quando o contexto está **vazio**, e aqui o contexto existe, só está incompleto.

**Abordagens consideradas:** duas estratégias foram avaliadas antes da implementação.

| | Segunda chamada ao modelo (LLM-as-judge) | Validação estruturada em código |
|---|---|---|
| Custo de API | Alto (dobra chamadas) | Baixo |
| Latência | Maior | Menor |
| Complexidade de implementação | Menor | Maior (exige formato de citação rígido e verificável) |
| Robustez | Maior | Menor (depende do modelo seguir o formato exigido) |

A decisão foi pela primeira abordagem, priorizando segurança sobre custo operacional: para um chatbot de atendimento, o custo de uma resposta incorreta (reputação, retrabalho) supera o custo de uma chamada extra de API.

**Implementação:** depois que a resposta do Max (`reply`) é gerada, uma segunda chamada ao modelo — com um prompt isolado, sem as regras de atendimento do Max — verifica se a resposta contém alguma afirmação não presente literalmente no contexto.

```python
def verificar_grounding(llm, contexto, resposta):
    prompt_verificacao = f"""Você é um verificador de fatos. Analise se a resposta abaixo usa APENAS informações presentes no contexto fornecido, sem inferências ou combinações não explícitas.

Contexto:
{contexto}

Resposta a verificar:
{resposta}

A resposta contém alguma afirmação, recomendação ou instrução que NÃO está literalmente escrita no contexto acima? Responda apenas SIM ou NÃO."""

    verificacao = llm.invoke(prompt_verificacao)
    return 'SIM' in verificacao.content.upper()
```

**Por que o prompt de verificação é isolado do *system prompt* do Max:** reaproveitar o mesmo prompt (com suas 12 regras de atendimento) geraria instruções concorrentes — "seja o Max, atendente empático" e "seja um verificador crítico" ao mesmo tempo. Um prompt dedicado, sem outras responsabilidades, evita esse conflito.

**Por que a checagem de grounding vem depois da checagem de `TRANSFER_HUMANO`:** se o modelo já respondeu com o marcador de transferência, `reply` não contém uma afirmação factual a ser verificada — rodar o grounding nesse caso seria uma chamada de API desperdiçada.

**Convenção usada nos testes:** `verificar_grounding` retorna `True` quando a resposta é considerada **não fundamentada** (contém algo fora do contexto, e deve ser bloqueada), e `False` quando a resposta está corretamente fundamentada e pode ser exibida. Nos termos usados a seguir, um **falso negativo** é quando o verificador retorna `False` (deixa passar) para uma resposta que, na verdade, continha uma inferência não fundamentada.

**Resultados de teste:**

| Pergunta | Contexto recuperado | Grounding | Avaliação |
|---|---|---|---|
| "atraso simples" (Seção 4) | Parcial, sem regra explícita | Bloqueou | ✅ Correto |
| "prazo pro sul" | Específico e completo | Passou | ✅ Correto |
| "meu pedido foi extraviado" (4 chunks) | Múltiplas fontes legítimas | Passou | ✅ Correto (combinação válida) |
| "6 dias sem receber reembolso" | Comparação numérica explícita | Transferiu antes do grounding (o próprio Max reconheceu a lacuna) | ✅ Correto |
| "reembolso demorando um pouco mais" | Prazo real + convite a sugestão genérica | Passou, mas a resposta continha "acione o suporte" — não fundamentado | ❌ Falso negativo |
| "reenvio sem atualização de status" | Mesmo padrão do caso anterior | Bloqueou | ✅ Correto |
| "sem código de rastreamento" | Mesmo padrão | Bloqueou | ✅ Correto |

**Limitação identificada:** de 7 perguntas testadas, 6 tiveram o comportamento esperado e 1 vazou uma inferência (uma sugestão de ação genérica, não um dado inventado). Tentativas de reproduzir esse mesmo padrão em outras perguntas estruturalmente parecidas não repetiram a falha — sugerindo um caso isolado, não uma falha sistemática. A causa provável é que o verificador usa o mesmo modelo (e portanto os mesmos vieses) que gera a resposta original: uma sugestão como "acione o suporte" pode não ser reconhecida como violação por parecer bom senso de atendimento, em vez de uma invenção factual explícita. Isso é consistente com uma limitação conhecida da técnica de LLM-as-judge — usar o mesmo modelo (ou modelo da mesma família) para gerar e verificar tende a ter pontos cegos correlacionados.

**Teste de confirmação da hipótese:** para avaliar se a causa era mesmo o modelo do verificador (e não um problema aleatório), a mesma pergunta que gerou o falso negativo foi testada novamente, trocando apenas o modelo usado em `verificar_grounding` de `gpt-4o-mini` para `gpt-4o` (mantendo o `gpt-4o-mini` como gerador das respostas do Max). Com o modelo mais forte como verificador, o mesmo caso que antes passava (`False`) foi corretamente bloqueado (`True`). O resultado reforçou a hipótese de que o falso negativo estava relacionado à capacidade do modelo utilizado como verificador, e não a um problema na lógica de validação ou no prompt — um único teste não é suficiente para confirmar isso de forma definitiva, mas é evidência consistente a favor da explicação.

**Investigação alternativa — ajustar o prompt em vez de trocar o modelo:** antes de aceitar a troca de modelo como única solução, foi investigado se o falso negativo poderia ser corrigido reformulando o prompt de verificação, mantendo `gpt-4o-mini` nos dois papéis. Essa investigação expôs uma tensão real entre dois tipos de erro opostos, testada com casos controlados (contexto e resposta escritos manualmente, fora do fluxo normal do Max, para isolar o comportamento do verificador de qualquer variação na geração da resposta):

| Versão do prompt | Falso positivo (paráfrase/cálculo numérico legítimo, ex.: "3-4 dias" → "72-96 horas") | Falso negativo ("acione o suporte" não fundamentado) |
|---|---|---|
| Original — "literalmente escrita no contexto" | Sim — bloqueava indevidamente respostas corretas mas reformuladas | Não — bloqueava corretamente |
| Sem "literalmente", com exemplos de reformulação/cálculo aceitáveis | Não — passava corretamente | Sim — voltou a deixar passar |
| Duas perguntas (paráfrase vs. ação inventada) na mesma chamada | Não — passava corretamente | Sim — a resposta da primeira pergunta contaminou o julgamento da segunda, mesmo sendo perguntas nominalmente separadas |
| Duas perguntas em chamadas separadas (uma só para "ação inventada") | Não — passava corretamente | Não — bloqueava corretamente |

A quarta versão (chamadas separadas) foi a única que resolveu os dois lados simultaneamente — mas ao custo de **duas** chamadas de verificação por resposta, em vez de uma, dobrando o custo dessa etapa. Descartada pelo mesmo critério de custo já aplicado às outras decisões de arquitetura do projeto: o ganho de precisão não justificou dobrar uma chamada que já é a segunda de duas por interação (geração + verificação).

**Conclusão desta investigação:** o comportamento do verificador com um único LLM barato, numa única chamada, parece ter uma restrição real de "escolha forçada" entre tolerar paráfrase e bloquear inferência indevida — pelo menos dentro das quatro formulações testadas aqui. Isso não prova que nenhuma formulação de prompt resolveria os dois problemas numa única chamada (o espaço de formulações possíveis é maior do que o testado), mas as tentativas feitas sugerem que o custo de engenharia para encontrá-la pode superar o de simplesmente aceitar a limitação documentada ou pagar por uma segunda chamada. A versão final do projeto manteve o prompt original ("literalmente"), priorizando o lado do erro considerado mais custoso para o negócio: bloquear uma resposta correta por engano custa uma transferência desnecessária (mesmo tipo de atrito de UX documentado na Seção 11); deixar passar uma inferência custa informação incorreta chegando ao cliente — o mesmo raciocínio de custo assimétrico já usado para justificar a adoção do grounding verification em si (comparação de abordagens, acima).

**Decisão sobre uso em produção:** apesar do ganho de confiabilidade observado com o `gpt-4o`, a configuração final do projeto permaneceu utilizando `gpt-4o-mini` em ambos os papéis. O custo por token do `gpt-4o` é cerca de 17x maior (US$ 2,50/US$ 10,00 vs. US$ 0,15/US$ 0,60 por milhão de tokens, entrada/saída), e o objetivo educacional do projeto, somado à natureza pontual do falso negativo encontrado (1 em 7 testes), não justificou o custo extra. Em um ambiente de produção real, essa escolha poderia ser diferente, considerando o impacto financeiro de respostas incorretas e a criticidade do domínio — nesse caso, o custo do risco (reputação, retrabalho) pode superar o custo da chamada mais cara. A troca de modelo permanece documentada como uma opção validada, disponível para revisão caso a taxa de falso negativo se mostre maior em uso real.

**Conclusão:** grounding verification reduz de forma significativa a taxa de alucinação por combinação de fatos — um problema que nenhuma outra camada (prompt engineering, `temperature`, `score_threshold`) havia conseguido bloquear. Ainda assim, não elimina o problema por completo: uma segunda camada de verificação com o mesmo modelo que gerou a resposta carrega parte dos mesmos vieses, então o resultado deve ser tratado como redução de risco, não garantia absoluta. Um verificador com modelo mais forte reduz esse viés, em evidência observada num teste controlado, mas ao custo de uma chamada de API significativamente mais cara — uma decisão de trade-off entre segurança e custo operacional, não uma correção "gratuita".

**Possível evolução futura:** a escolha de modelo não precisa ser a mesma para os dois papéis. Uma arquitetura mais madura poderia manter um modelo econômico (`gpt-4o-mini`) para gerar respostas — a etapa de maior volume de chamadas — e reservar um modelo mais robusto apenas para a etapa crítica de verificação, que ocorre uma vez por resposta. Isso concentraria o custo mais alto exatamente onde a segurança importa mais, em vez de pagar o mesmo prêmio em toda a interação.

**Instabilidade do veredito mesmo com `temperature=0`, entrada idêntica:** durante a investigação da Seção 12, um caso do eval set (`grounding_should_fail` para "meu pedido foi extraviado") passou a falhar de forma intermitente entre execuções do `test_eval_set.py`, sem nenhuma mudança de código relacionada. A hipótese inicial foi a variação de fraseado do `llm_chat` (`temperature=0.3`) — testada isolando a variável: rodando `gerar_resposta_max` e `verificar_grounding` em laço, 5 vezes seguidas, para a mesma pergunta e contexto.

Com `temperature=0.3` no gerador, o texto da resposta variou levemente entre execuções (troca de "se" por "caso", pequenas diferenças de pontuação), e o veredito do grounding também variou (4x bloqueou, 1x não) — consistente com a hipótese inicial. Repetindo o mesmo teste com `temperature=0` no `llm_chat`, o texto da resposta ficou **idêntico** nas 5 execuções — mas o veredito do `verificar_grounding` (cujo `llm_verificador` já era `temperature=0` desde a Seção 8 original) ainda variou: 4x `True`, 1x `False`, com a mesma pergunta, mesmo contexto e mesma resposta em todas as chamadas.

**Causa:** `temperature=0` reduz drasticamente a aleatoriedade de um LLM, mas não é uma garantia absoluta de determinismo bit-a-bit entre chamadas de API separadas — variações na forma como a inferência é processada no backend (operações de ponto flutuante em lote, paralelizadas entre GPUs) podem produzir pequenas diferenças de saída mesmo para uma entrada idêntica. Isso é documentado pela própria OpenAI como comportamento esperado da API, não um bug do projeto.

**Implicação:** o falso negativo de 1 em 7 casos já documentado acima pode não ser explicado inteiramente pela capacidade do `gpt-4o-mini` como verificador — parte da taxa de erro observada pode ser instabilidade inerente à infraestrutura de inferência, presente em qualquer modelo, inclusive um mais forte. Isso também significa que nenhum volume de testes manuais prova ausência de variação — apenas reduz a incerteza sobre a taxa dela, nunca a elimina.

**Mitigação avaliada e descartada por ora — retry com maioria de votos:** chamar `verificar_grounding` N vezes (N ímpar, ex. 3) para o mesmo caso e decidir pelo veredito mais frequente reduziria o risco de uma única chamada instável decidir o resultado:

```python
def verificar_grounding_com_retry(llm, pergunta, contexto, resposta, tentativas=3):
    resultados = [
        verificar_grounding(llm, pergunta, contexto, resposta)
        for _ in range(tentativas)
    ]
    votos_falha = sum(resultados)
    return votos_falha > tentativas / 2
```

**Decisão consciente de não implementar:** isso triplicaria o custo da etapa de verificação (já a segunda chamada de LLM por interação), para mitigar uma instabilidade observada apenas em teste deliberado de execuções repetidas, não em uso normal do projeto. Mesmo critério de custo vs. risco já aplicado a outras decisões deste documento (Seções 8, 11, 13, 16). Fica documentado como opção validada, caso a taxa de instabilidade se mostre mais frequente ou relevante em uso real.

## 9. Persistência do índice FAISS: eliminando reprocessamento desnecessário

Nas versões anteriores do projeto, `carregar_base_conhecimento()` recalculava o índice FAISS do zero a cada execução — carregando o `politicas.txt`, quebrando em chunks e gerando embeddings via API da OpenAI para cada um deles, mesmo quando nada havia mudado desde a última vez. Isso levava entre 10 e 15 segundos por execução, um custo que cresceria proporcionalmente ao tamanho da base de conhecimento em um cenário de produção real.

**Estratégia adotada:** salvar o índice em disco após o primeiro cálculo, e nas execuções seguintes, carregar esse índice já pronto — recalculando do zero apenas quando o `politicas.txt` for alterado. Para detectar essa alteração, a data de modificação do arquivo (`os.path.getmtime`) é salva junto com o índice, em um arquivo de metadados separado; a cada execução, essa data é comparada com a data atual do arquivo antes de decidir qual caminho seguir.

```python
def carregar_base_conhecimento():
    caminho_politicas = os.path.join(BASE_DIR, 'data', 'politicas.txt')
    caminho_indice = os.path.join(BASE_DIR, 'data', 'faiss_index')
    caminho_metadata = os.path.join(BASE_DIR, 'data', 'faiss_metadata.txt')

    data_modificacao_atual = str(os.path.getmtime(caminho_politicas))
    embeddings = OpenAIEmbeddings()

    indice_existe = os.path.exists(caminho_indice)
    metadata_existe = os.path.exists(caminho_metadata)

    if indice_existe and metadata_existe:
        with open(caminho_metadata, 'r') as f:
            data_modificacao_salva = f.read().strip()

        if data_modificacao_salva == data_modificacao_atual:
            vectorstore = FAISS.load_local(caminho_indice, embeddings, allow_dangerous_deserialization=True)
            return vectorstore.as_retriever(
                search_type='similarity_score_threshold',
                search_kwargs={'score_threshold': 0.68, 'k': 4}
            )

    loader = TextLoader(caminho_politicas, encoding='utf-8')
    documentos = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    chunks = splitter.split_documents(documentos)
    vectorstore = FAISS.from_documents(chunks, embeddings)

    vectorstore.save_local(caminho_indice)
    with open(caminho_metadata, 'w') as f:
        f.write(data_modificacao_atual)

    return vectorstore.as_retriever(
        search_type='similarity_score_threshold',
        search_kwargs={'score_threshold': 0.68, 'k': 4}
    )
```

**Por que a checagem de data precisa vir antes de qualquer processamento:** o objetivo da persistência é evitar trabalho desnecessário. Se a comparação de datas acontecesse depois de já ter carregado o arquivo e gerado os embeddings, o tempo e o custo que se queria evitar já teriam sido gastos antes da decisão ser tomada.

**Por que a data de modificação precisa ser salva em um arquivo próprio:** `vectorstore.save_local()` gera dois arquivos (`index.faiss`, com os vetores, e `index.pkl`, com o texto original de cada chunk) — nenhum dos dois guarda informação sobre quando o arquivo de origem foi editado, porque esse não é o propósito deles. Um terceiro arquivo, criado especificamente para esse controle, foi necessário.

**Sobre `allow_dangerous_deserialization=True`:** esse parâmetro é uma confirmação explícita, exigida pelo LangChain, de que a origem do arquivo carregado é confiável — o formato de serialização usado (`pickle`) pode, em tese, executar código arbitrário se o arquivo carregado vier de uma fonte não verificada. Como o índice é gerado pelo próprio projeto, na própria máquina, esse risco não se aplica aqui; a confirmação existe para casos onde um índice fosse compartilhado ou baixado de terceiros.

**Resultado medido:** para confirmar o ganho real (e não apenas a percepção de "ficou mais rápido"), o tempo de cada etapa foi medido com `time.time()` em uma execução com o índice já persistido:

| Etapa | Tempo |
|---|---|
| Imports das bibliotecas (langchain, faiss, etc.) | 2.73s |
| Criar `OpenAIEmbeddings()` | 0.70s |
| `FAISS.load_local()` | 0.17s |
| Total até a saudação do Max aparecer | 3.64s |

**Descoberta inesperada:** o tempo ainda percebido como "não tão rápido quanto esperado" (uns 5-6 segundos, na sensação inicial) não vinha mais do FAISS ou dos embeddings — a persistência eliminou esse gargalo com sucesso (`load_local()` levou apenas 0.17s). O tempo restante é dominado pelos **imports das bibliotecas** (2.73s, mais de 75% do tempo total), uma etapa anterior a qualquer lógica de persistência, comum a qualquer projeto que use LangChain e não relacionada ao tamanho da base de conhecimento.

**Conclusão:** a persistência resolveu o problema real que motivou a mudança — o reprocessamento repetido de embeddings, que escalaria mal com uma base de conhecimento maior. O tempo de import das bibliotecas é um custo fixo e comum ao framework, não um sintoma do problema original, e não vale a pena otimizar mais a fundo para um projeto deste porte. Medir antes de continuar otimizando evitou gastar esforço perseguindo um gargalo que já não existia mais.

## 10. Separando logs técnicos da interface do usuário

O projeto acumulou dois tipos de aviso técnico ao longo do desenvolvimento: um `DeprecationWarning` do `langchain-community` (emitido pelo módulo `warnings` do Python no momento do import) e um `WARNING` interno do LangChain quando `score_threshold` não encontra nenhum chunk relevante (emitido pelo módulo `logging`). Nas primeiras versões, cada um foi resolvido de forma pontual, com filtros que **descartavam** a mensagem por completo (`warnings.filterwarnings('ignore', ...)` e `logging.getLogger(...).setLevel(logging.ERROR)`) — suficiente para manter o terminal limpo, mas às custas de perder qualquer rastro desses eventos.

**Problema com a abordagem de descarte:** silenciar um aviso o torna invisível também para quem desenvolve o projeto. Se um comportamento inesperado começasse a gerar avisos com mais frequência em uso real, não haveria como perceber, porque a mensagem nunca chega a existir em lugar nenhum.

**Estratégia adotada:** em vez de descartar, os avisos passaram a ser **redirecionados** para um arquivo de log (`logs/app.log`), mantendo o terminal visível ao usuário limpo, mas preservando o histórico para consulta e depuração.

```python
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.makedirs(os.path.join(BASE_DIR, 'logs'), exist_ok=True)

logging.basicConfig(
    filename=os.path.join(BASE_DIR, 'logs', 'app.log'),
    level=logging.WARNING,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logging.captureWarnings(True)
```

**Por que `logging.captureWarnings(True)` era necessário:** os módulos `warnings` e `logging` são sistemas independentes no Python, que não se comunicam por padrão. Essa função cria uma ponte, redirecionando o que passaria pelo `warnings` (como o `DeprecationWarning`) para dentro do sistema de `logging`, permitindo que os dois tipos de aviso — de origens diferentes — sejam capturados pela mesma configuração de arquivo.

**Por que essa configuração precisa vir antes dos imports do LangChain:** o `DeprecationWarning` é disparado no exato momento em que `from langchain_community.vectorstores import FAISS` é executado. Se a configuração de log viesse depois dessa linha, o aviso já teria sido descartado (ou impresso no terminal) antes de existir qualquer lugar para redirecioná-lo.

**Por que a pasta `logs/` é criada em código, e não apenas documentada como pré-requisito:** ela foi adicionada ao `.gitignore` (arquivos de log são artefatos de execução, não código-fonte, e crescem a cada uso). Isso significa que ela nunca existirá automaticamente ao clonar o repositório. Documentar "crie a pasta antes de rodar" transferiria ao usuário uma responsabilidade que o próprio programa pode cumprir de forma confiável com uma linha (`os.makedirs(..., exist_ok=True)`), sem custo perceptível em qualquer execução.

**Resultado:** ambos os tipos de aviso passaram a ser registrados em `logs/app.log`, com data, hora e nível de severidade, sem aparecer no terminal:

```
2026-07-11 21:51:36,916 - WARNING - .../main.py:16: DeprecationWarning: `langchain-community` is being sunset...
2026-07-11 21:52:40,706 - WARNING - No relevant docs were retrieved using the relevance score threshold 0.68
```

Vale notar que o segundo aviso passou a ser capturado sem precisar de nenhum filtro específico por módulo (como o `logging.getLogger('langchain_core.vectorstores')` usado anteriormente) — a configuração de `level=logging.WARNING` no `basicConfig` já captura qualquer aviso desse nível ou mais grave, de qualquer origem que use o sistema `logging`, tornando a solução mais genérica e resiliente a avisos futuros ainda não identificados.

**Conclusão:** existe uma diferença prática entre "silenciar" e "redirecionar" um aviso técnico. Descartar é apropriado quando a mensagem é comprovadamente irrelevante; redirecionar para um log é mais apropriado quando a mensagem pode ter valor de diagnóstico futuro, mesmo não sendo destinada ao usuário final. Separar canais de saída — interface do usuário via `print()`, diagnóstico técnico via `logging` em arquivo — é uma prática comum em aplicações reais, especialmente à medida que um projeto de terminal evolui para algo servido como aplicação (API, interface web).

## 11. Detecção de saudação: evitando penalizar transferência por conversa social

Testando o fluxo manualmente, foi observado que uma saudação simples ("ola") gerava contexto vazio na busca (nenhum chunk do `politicas.txt` é relevante para um cumprimento) e disparava o contador `tentativas_sem_contexto` da mesma forma que uma pergunta genuinamente fora do domínio. Em uma sequência de duas saudações seguidas — comportamento humano plausível em qualquer atendimento real — o cliente seria transferido para um atendente sem ter feito nenhuma pergunta de negócio.

**Por que isso é um problema de produto:** o fallback de contador foi desenhado para capturar perguntas fora do escopo do assistente, não para penalizar conversa social sem conteúdo. Tratar as duas situações da mesma forma gera transferências desnecessárias logo no início do atendimento.

**Abordagens descartadas antes da solução final:**

- **Lista de palavras-chave fixas** (`if pergunta in ['oi', 'ola', ...]`) — falha diante de variações informais de escrita ("oii", "olar", "eae"), pelo mesmo motivo que uma correspondência de texto exata já havia se mostrado frágil no bug do marcador `TRANSFERIR_HUMANO` (Seção 7).
- **Filtro por tamanho da mensagem** — descartado ao se considerar o contra-exemplo "meu pedido ta atrasado", que tem tamanho comparável a uma saudação estendida, mas é uma pergunta de negócio legítima. Tamanho não correlaciona de forma confiável com a distinção que importa.
- **Detecção de mensagens fragmentadas ou incompletas** (ex.: cliente envia a mensagem sem querer, no meio de digitar) — considerada, mas não implementada. Diferente de saudação, que segue um padrão finito e reconhecível, um fragmento não tem um conjunto fixo de exemplos comparáveis; julgar se uma frase está "gramaticalmente completa" exige um tipo de julgamento semântico mais próximo do que motivou o uso de LLM no grounding verification (Seção 8), o que reintroduziria custo de chamada por mensagem. Fica documentado como limitação conhecida, não resolvida nesta versão.

**Solução adotada:** uma segunda base vetorial, pequena e independente da base de conhecimento principal, criada a partir de uma lista de exemplos de saudação. A pergunta do cliente é comparada contra essa base antes de qualquer busca no `politicas.txt`; se a similaridade for alta o suficiente, a mensagem é tratada como saudação — respondida com uma mensagem fixa (sem chamar o LLM) e sem contar como tentativa falha.

```python
def carregar_indice_saudacoes():
    exemplos_saudacao = ['olá', 'oi', 'oii', 'bom dia', 'boa tarde', 'boa noite', 'tudo bem', 'e aí', 'opa', 'salve']
    documentos_saudacao = [Document(page_content=texto) for texto in exemplos_saudacao]
    embeddings = OpenAIEmbeddings()
    return FAISS.from_documents(documentos_saudacao, embeddings)

def eh_saudacao(vectorstore_saudacoes, pergunta):
    resultados = vectorstore_saudacoes.similarity_search_with_relevance_scores(pergunta, k=1)
    _, score = resultados[0]
    return score >= 0.85
```

**Por que a checagem precisa vir antes de `buscar_contexto`, e não dentro de `verificar_grounding`:** o `verificar_grounding` só é alcançado após a resposta do LLM já ter sido gerada, dentro do fluxo normal. Como uma saudação gera contexto vazio, ela já é interceptada pelo bloco `if not contexto.strip()` (via `continue`) antes de a execução chegar perto do LLM ou do verificador — qualquer regra colocada dentro de `verificar_grounding` para tratar saudação nunca seria executada para esse caso.

**Por que a resposta à saudação é fixa no código, e não gerada pelo LLM:** mesmo princípio já aplicado ao fallback de "não entendi" (Seção 6) — uma saudação não exige raciocínio, então gerar a resposta via `llm.stream()` seria custo desnecessário para uma tarefa totalmente previsível.

**Calibração do `threshold` (0.85):** o valor inicial de 0.75 foi testado e rejeitado com dado real — "meu pedido ta atrasado" obteve score 0.7657, acima do valor testado, o que classificaria incorretamente uma pergunta de negócio como saudação. Elevando para 0.85, a margem de segurança contra falsos positivos se sustentou em múltiplos testes:

| Frase | Score | Classificação esperada | Resultado |
|---|---|---|---|
| "oi" | 1.0000 | Saudação | ✅ |
| "oie" | 0.9094 | Saudação (variação) | ✅ |
| "bom diaa" | 0.9569 | Saudação (variação) | ✅ |
| "hello" | 0.7962 | Saudação | ❌ (abaixo do threshold) |
| "meu pedido ta atrasado" | 0.7657 | Não-saudação | ✅ |
| "qual o prazo" | 0.7413 | Não-saudação | ✅ |
| "oi, meu pedido atrasou" (mensagem mista) | 0.7756 | Não-saudação | ✅ |

**Limitação aceita conscientemente:** com 0.85, saudações em outro idioma ("hello") ou gírias regionais não incluídas na lista de exemplos ficam abaixo do threshold e não são reconhecidas como saudação — o cliente recebe o fallback de "não entendi, pode reformular?" na primeira tentativa. Essa foi uma escolha deliberada: abaixar o threshold para cobrir esses casos reduziria a margem de segurança contra falsos positivos em perguntas de negócio curtas, que é o risco mais custoso dos dois. Quando um caso específico se mostrou relevante o suficiente para justificar tratamento (a gíria "salve", mais comum no contexto brasileiro do que "hello"), a solução adotada foi ampliar a lista de exemplos de referência, não reduzir o threshold — isso resolveu o caso sem comprometer a margem de segurança já validada.

**Limitação de escopo — saudação, não small talk completo:** o mecanismo cobre apenas cumprimentos ("oi", "bom dia", "salve"), não a categoria mais ampla de small talk usada em sistemas de diálogo (que também inclui despedidas, agradecimentos e perguntas de cortesia como "tudo bem?" fora do contexto de abertura). Ampliar a lista de exemplos manualmente para cobrir cada variação teria retorno decrescente — sempre existiriam casos não previstos. A alternativa mais robusta seria um LLM julgando se a mensagem é social ou tem intenção de negócio, mas isso reintroduziria uma chamada de API por mensagem para resolver um risco de baixo custo: uma mensagem social não reconhecida apenas aciona o fallback de "pode reformular?" (Seção 6), não gera informação incorreta. Mesmo critério de custo vs. risco já aplicado à fragmentação (acima) e ao query rewriting (Seção 16). Essa avaliação de custo também é específica ao contexto de estudos: em um produto real, um pequeno atrito de UX repetido em escala (muitos clientes recebendo o fallback por engano) tem custo agregado de experiência que poderia justificar a chamada extra — a decisão de mantê-la barata aqui reflete o volume de uso atual, não uma conclusão permanente sobre o valor da chamada.

**Teste de ambiguidade semântica:** como a palavra "salve" também pode ser usada como verbo ("salve meu número de rastreamento"), esse cenário foi testado deliberadamente antes de considerar a solução validada. O resultado (score 0.7854, abaixo do threshold) confirmou que o embedding distingue corretamente a interjeição isolada do verbo em contexto de frase — a comparação por similaridade captura a estrutura semântica da frase completa, não apenas a presença da palavra.

**Conclusão:** o mesmo mecanismo de embedding usado para RAG de negócio pode ser reaproveitado, de forma barata, para classificar categorias de mensagem que não são sobre conteúdo de negócio (como saudações) — evitando tanto correspondência de texto frágil (listas fixas) quanto o custo de uma chamada de LLM completa para uma tarefa que não exige julgamento complexo. A calibração do threshold seguiu a mesma metodologia usada em `score_threshold` (Seção 5): testar categorias antagônicas, medir a margem real entre elas, e tratar exceções conhecidas ampliando a base de exemplos em vez de comprometer a margem de segurança já validada.

## 12. Ambiguidade entre políticas semanticamente próximas: chunking, retrieval e geração

Este caso começou como um problema aparentemente simples — uma resposta que "misturava regras de negócio" — mas a investigação revelou que o problema real atravessava três camadas independentes do pipeline (chunking, retrieval, prompt) e só ficou visível depois de corrigir cada uma, uma de cada vez.

### 12.1 Sintoma inicial: chunking por caractere fragmentando unidades de negócio

Pergunta de teste:

> "meu pedido foi extraviado, o que eu faço?"

A base de conhecimento tem duas políticas distintas para cenários parecidos:

- **Política de reenvio** — aplicável quando o cliente já relata ou confirma o extravio.
- **Política para pedido constando como entregue, mas não recebido** — aplicável quando o extravio ainda não foi confirmado.

Com `RecursiveCharacterTextSplitter` e `chunk_size=200` (Seção 2), os chunks recuperados vinham corretos individualmente, mas o LLM combinou trechos de ambas as políticas em uma única resposta, sem indicar qual das duas de fato se aplicava ao relato do cliente.

**Primeira hipótese testada e descartada:** aumentar `chunk_size`. Rejeitada porque isso reintroduziria o problema original da Seção 2 — misturar múltiplos tópicos em um único chunk.

**Causa real identificada:** o `RecursiveCharacterTextSplitter`, por operar em contagem de caracteres, não respeita a fronteira lógica de uma seção — uma política pode ser cortada no meio, entre uma condição e sua consequência, mesmo com `chunk_size` calibrado para o caso comum (Seção 2). Isso não havia aparecido antes porque as políticas testadas até então cabiam confortavelmente dentro de 200 caracteres; políticas mais longas (com múltiplos bullets) expunham a fragilidade.

### 12.2 Migração para chunking por seção

Como a base de conhecimento já é naturalmente estruturada — cada política separada por linha em branco — a solução adotada foi abandonar o splitter genérico em favor de um split manual por seção:

```python
def dividir_por_secao(texto):
    blocos = texto.split('\n\n')
    return [bloco.strip() for bloco in blocos if bloco.strip()]
```

Cada bloco resultante vira um `Document` inteiro, sem dependência de `chunk_size`/`chunk_overlap`. Isso elimina por completo o risco de uma política ser cortada no meio, independente do seu tamanho.

**Trade-off aceito:** essa solução só funciona porque a base é pequena e uniformemente estruturada (uma política = um parágrafo). Não escalaria para documentos sem essa convenção de formatação — nesse caso, uma solução mais geral seria necessária (ex.: chunking semântico via embeddings, ou marcação explícita de seções via metadados).

### 12.3 Ambiguidade residual: scores quase empatados entre duas políticas distintas

Corrigido o chunking, o sintoma original (resposta misturando políticas) desapareceu — mas o Max ainda escolhia, em algumas execuções, a política errada por completo (a de "pedido não recebido" em vez de "reenvio"). Investigando os scores de relevância (`similarity_search_with_relevance_scores`, já na escala 0-1 usada pelo `score_threshold`, diferente da distância L2 padrão do FAISS — ver nota na Seção 5), os dois chunks vinham com uma diferença de apenas ~0.0002 entre si (0.7927 vs. 0.7925), dentro da margem de ruído do embedding.

**Causa identificada:** as duas políticas compartilham vocabulário de superfície ("extravio", "pedido", "investigação"), sem que o texto original deixasse explícito, de forma textual, a condição que diferencia uma da outra. O embedding não tinha sinal suficiente para discriminar entre "extravio relatado pelo cliente" e "extravio ainda não confirmado" — uma distinção óbvia para um leitor humano, mas pouco marcada lexicalmente no texto original.

### 12.4 Reformulação da base de conhecimento com condições de aplicabilidade explícitas

A correção foi reescrever o título de cada política ambígua para incluir, entre parênteses, sua condição de aplicabilidade de forma redundante e explícita — uma técnica de *chunk enrichment*, escolhida por aproveitar o peso que muitos modelos de embedding dão às primeiras tokens de um texto:

```
Política de reenvio (aplicável quando o cliente relata ou confirma que o pedido foi extraviado, ou quando recebeu um item incorreto):
...

Política para pedido constando como entregue no sistema, mas que o cliente afirma não ter recebido (situação de extravio ainda não confirmado, diferente de quando o cliente já relata diretamente o extravio):
...
```

O mesmo padrão foi aplicado ao par "Política de atraso" / "pedido não recebido", que tinha o mesmo tipo de sobreposição.

### 12.5 Regressão temporária: fragmentação reintroduzida pelo texto mais longo

Reindexando a base reformulada, o retrieval voltou a trazer fragmentos de uma linha só — o mesmo sintoma da Seção 12.1, mas agora causado indiretamente: os títulos mais longos (com a condição de aplicabilidade) empurraram o tamanho de algumas seções para além do que o `RecursiveCharacterTextSplitter` ainda estava configurado a aceitar, já que a migração completa para chunking por seção (12.2) só havia sido validada com o texto original, mais curto.

Isso não chegou a ser uma falha nova — foi a confirmação, com um segundo caso real, de que qualquer dependência residual de `chunk_size` continuava frágil frente a mudanças de conteúdo. A correção definitiva já estava em 12.2; esse passo intermediário serviu como evidência de que não havia mais como adiar a migração completa.

### 12.6 Regra de seleção de política no system prompt

Com chunking por seção e texto reformulado, os chunks passaram a chegar completos e com condição de aplicabilidade explícita — mas o LLM de geração ainda não tinha instrução de como *usar* essa condição para decidir qual política seguir quando mais de uma aparecia no contexto. O `system.txt` já continha uma regra genérica ("cada política deve ser tratada como independente"), mas nenhuma regra de desempate.

Adicionada uma seção nova ao prompt:

```
## Seleção da política aplicável

* Quando mais de uma política do contexto tratar de situações parecidas, cada uma delas terá uma condição de aplicabilidade descrita entre parênteses em seu título.
* Compare o que o cliente relatou explicitamente com a condição de cada política e aplique apenas aquela cuja condição corresponda ao relato do cliente.
* Se o cliente afirmar diretamente que o pedido foi extraviado, isso conta como o relato exigido pela política correspondente — não é necessário aguardar confirmação adicional para aplicá-la.
* Ignore políticas cuja condição de aplicabilidade não corresponda ao que foi relatado, mesmo que estejam presentes no contexto.
```

### 12.7 Conflito entre o novo raciocínio exigido e o verificador de grounding

Com a regra acima, o Max passou a escolher a política correta — mas a resposta era bloqueada pelo `verificar_grounding` (Seção 8), que classificava a resposta como não fundamentada.

**Causa raiz:** a resposta correta começava com "Como você relatou que seu pedido foi extraviado...", uma ponte lógica entre o que o cliente disse (não presente no `contexto`, só na `pergunta`) e a condição de aplicabilidade escrita no chunk. O prompt do verificador (Seção 8) proibia qualquer afirmação "não literalmente escrita no contexto" — sem exceção — e também nunca recebia a pergunta original como parâmetro, então não tinha como saber que essa inferência era legítima. As duas camadas do sistema (o prompt principal, que exige esse tipo de raciocínio; o verificador, que o proíbe sem exceção) estavam, na prática, em conflito direto.

Esse caso é um exemplo concreto do risco já registrado na Seção 8 ("um falso negativo em 7 casos testados") — só que aqui na direção oposta: um **falso positivo** de bloqueio, causado não por falta de robustez do verificador, mas por uma mudança em outra camada do sistema que o verificador não foi atualizado para acompanhar.

### 12.8 Correção do verificador: exceção para inferência legítima + passagem da pergunta original

```python
def verificar_grounding(llm, pergunta, contexto, resposta):
    prompt_verificacao = f"""Você é um verificador de fatos. Analise se a resposta abaixo é sustentada pelo contexto fornecido.

Pergunta do cliente:
{pergunta}

Contexto:
{contexto}

Resposta a verificar:
{resposta}

A resposta pode aplicar uma política do contexto com base na condição de aplicabilidade dela, desde que essa condição corresponda a algo que o cliente afirmou na pergunta. Isso NÃO é considerado inferência indevida.

O que NÃO é permitido: a resposta inventar dados sobre o pedido (região, prazo, datas, status) que não foram mencionados nem pelo cliente nem pelo contexto, ou combinar políticas de forma que nenhuma delas sustente isoladamente a afirmação.

A resposta contém alguma afirmação, recomendação ou instrução que não seja sustentada pelo contexto ou pelo que o cliente relatou? Responda apenas SIM ou NÃO."""

    verificacao = llm.invoke(prompt_verificacao)
    return 'SIM' in verificacao.content.upper()
```

A mudança de assinatura (`pergunta` como novo parâmetro) exigiu também alterar o ponto de chamada em `main.py`, passando a pergunta original recebida do `input()`.

### 12.9 Falha em obter justificativa: reordenação para raciocínio antes do veredito

Ao adicionar debug temporário pedindo justificativa (formato "SIM/NÃO, depois explique"), o modelo ignorou a instrução secundária e respondeu apenas o veredito — um comportamento consistente com o já observado em outros pontos do projeto (Seção 7): instruções de formato posicionadas depois da pergunta principal têm menor adesão em modelos menores.

Correção: inverter a ordem, pedindo raciocínio **antes** do veredito (uma forma de *chain-of-thought* aplicada ao próprio verificador, não só ao gerador de respostas), e extrair o veredito apenas da última linha da resposta — não com `'SIM' in texto_completo`, já que o raciocínio em si pode conter a palavra "sim" em outro sentido, o que geraria falso positivo na extração:

```python
    verificacao = llm.invoke(prompt_verificacao)
    linhas = verificacao.content.strip().split('\n')
    veredito = linhas[-1] if linhas else verificacao.content
    return 'SIM' in veredito.upper()
```

Com essa mudança, o raciocínio do verificador ficou visível e passou a confirmar exatamente o motivo do bloqueio anterior — evidência direta, não só inferida, da causa raiz identificada em 12.7.

### 12.10 Validação com múltiplas execuções

Como os scores de retrieval entre as duas políticas continuavam próximos (12.3 não elimina a proximidade, só reduz o risco de inversão via reformulação de texto), o caso foi testado três vezes seguidas antes de ser considerado resolvido. Nas três execuções, o Max escolheu a política correta e o verificador liberou a resposta, com raciocínio consistente nas três — evidência de estabilidade, não de uma correção que dependia de sorte em uma única chamada.

### 12.11 Conclusão da seção

Esse caso demonstra que um sintoma observado em uma única camada (a resposta final) pode ter causa em qualquer uma das camadas anteriores do pipeline — chunking, texto da base de conhecimento, retrieval, prompt de geração, ou prompt de verificação — e que corrigir uma camada pode expor ou até criar uma nova falha em outra, se as duas não forem tratadas como um sistema acoplado. Em particular, o verificador de grounding (Seção 8) não é uma camada estática e neutra: instruções adicionadas ao prompt principal (como a regra de seleção de política, 12.6) podem entrar em conflito direto com regras já existentes no verificador, exigindo revisão coordenada das duas camadas, não apenas da que gerou a resposta.

## 13. Eval set automatizado: da validação manual a um script reproduzível

A Seção 16 já identifica um "eval set mais robusto" como próximo passo. Esta seção documenta a implementação de uma primeira versão funcional, motivada diretamente pela investigação da Seção 12: validar manualmente cada mudança de prompt digitando a mesma pergunta no terminal deixou de ser viável a partir do momento em que uma única correção (12.8, 12.9) passou a ter potencial de afetar qualquer resposta do sistema, não só o caso de teste em foco.

### 13.1 Estrutura do eval set

Um arquivo `tests/eval_set.json` reúne casos de teste no formato:

```json
{
  "pergunta": "meu pedido foi extraviado, o que eu faço?",
  "checks": { "grounding_should_fail": false },
  "limitacao_conhecida": false
}
```

Cada caso testa exatamente um aspecto do pipeline (`is_greeting`, `should_find_context`, `needs_more_information` ou `grounding_should_fail`) — não uma resposta completa. Essa decisão de design evita ambiguidade sobre o que causou uma falha: um caso testando `grounding_should_fail` não também afirma nada sobre `should_find_context`, mesmo que ambos dependam do mesmo retriever internamente.

O campo `limitacao_conhecida` distingue uma falha aceita conscientemente (documentada em seções anteriores, como o falso negativo da Seção 8 ou o caso "hello" da Seção 11) de uma regressão real — permitindo que o script reporte as duas categorias separadamente, sem misturar "funcionando como esperado, mas com limitação conhecida" com "quebrou".

### 13.2 Script de execução

`tests/run_evals.py` carrega o JSON, inicializa os mesmos componentes usados em produção (`retriever`, `llm_chat`, `llm_verificador`, `system_prompt`) e despacha cada caso para uma função avaliadora correspondente ao seu tipo de `check`, via um dicionário de despacho:

```python
AVALIADORES = {
    'is_greeting': avaliar_is_greeting,
    'should_find_context': avaliar_should_find_context,
    'needs_more_information': avaliar_needs_more_information,
    'grounding_should_fail': avaliar_grounding_should_fail,
}
```

**Por que despacho por dicionário, e não uma cadeia de `if/elif`:** o nome do campo em `checks` já identifica o tipo de verificação, sem precisar de um campo redundante (`tipo_verificacao`) duplicando essa informação — e adicionar um novo tipo de check no futuro (Seção 13.7) significa acrescentar uma entrada ao dicionário, não editar uma cadeia condicional existente.

### 13.3 Bug exposto pela primeira execução: assinatura desatualizada entre módulos

A primeira tentativa de rodar o script falhou com `TypeError`, porque `avaliar_grounding_should_fail` ainda chamava `verificar_grounding(llm, contexto, resposta)` — a assinatura anterior à correção da Seção 12.8, que passou a exigir `pergunta` como parâmetro adicional. O script de avaliação havia sido escrito contra a versão anterior da função e não foi atualizado junto com a mudança em produção.

**Lição:** um eval set que chama as mesmas funções usadas em produção (em vez de reimplementar a lógica) tem a vantagem de testar o comportamento real, mas herda o mesmo risco de qualquer outro consumidor de uma função: uma mudança de assinatura precisa ser propagada a todos os pontos de chamada, incluindo os de teste — não só os de produção, que são os que costumam vir à mente primeiro.

### 13.4 Um segundo verificador com o mesmo tipo de falha, exposto pelo eval set

Testando manualmente uma pergunta deliberadamente fora de escopo ("posso trocar meu pedido por outro produto de valor maior pagando a diferença?"), o `verificar_informacao_suficiente` (que decide se falta região/prazo/data antes mesmo de buscar contexto) classificou incorretamente como "falta informação" — presumindo a existência de uma política de troca condicionada a região, quando na verdade a base de conhecimento não cobre trocas de produto de forma alguma.

**Causa identificada:** o mesmo padrão da Seção 12.9 — o prompt desse verificador tinha exemplos cobrindo atraso, extravio, reembolso e horário, mas nenhum exemplo próximo de "pergunta sobre algo que a empresa simplesmente não oferece". Sem esse sinal, o modelo generalizou pelo padrão mais parecido ("situação + problema do pedido → falta dado"), ignorando que a limitação real não era falta de dado, e sim ausência de política sobre o assunto.

Aplicada a mesma correção estrutural de 12.8/12.9 — raciocínio explícito antes do veredito, e um exemplo novo cobrindo esse tipo de caso:

```
"posso trocar meu pedido por outro produto de valor maior?" → NÃO (a resposta não depende de região, prazo ou data — depende de existir ou não uma política de troca, o que é um tipo de limitação diferente)
```

O ajuste melhorou a qualidade do raciocínio do modelo (a justificativa passou a mencionar explicitamente que "políticas de extravio geralmente são aplicáveis independentemente dessas variáveis"), mas **não corrigiu por completo** o caso de troca de produto, que continuou retornando incorretamente `needs_more_information: True`.

### 13.5 Decisão consciente de não reordenar o pipeline para corrigir o caso

A correção estrutural desse caso exigiria que `verificar_informacao_suficiente` tivesse acesso ao conteúdo da base de conhecimento (via retrieval) antes de decidir se falta um dado do pedido ou se o assunto simplesmente não é coberto — hoje ele julga apenas a partir do texto da pergunta, isolado.

Isso exigiria mover essa checagem para depois do retrieval no `main.py`, o que tem três custos: (a) toda pergunta passaria a pagar uma chamada de embedding/busca vetorial antes da checagem, mesmo as que hoje são filtradas de forma barata sem retrieval; (b) o prompt do verificador cresceria com os chunks recuperados embutidos, aumentando tokens de entrada em toda chamada, não só nas afetadas; (c) latência adicional no caminho crítico.

Dado que esse verificador já cobre corretamente os cenários mais prováveis de uso real (atraso, cancelamento, prazo), e a pergunta que expôs a falha é sobre um cenário de negócio que a empresa fictícia sequer oferece (troca de produto — a XYZ Entregas é uma transportadora, não um varejista), a decisão foi documentar como limitação conhecida em vez de reordenar o pipeline agora — o mesmo critério de custo vs. risco já aplicado em outras decisões deste documento (Seções 8, 11, 16).

### 13.6 Resultado da primeira execução completa

Com as duas correções acima aplicadas (13.3, 13.4) e um caso novo adicionado ao JSON documentando a limitação de 13.5, a primeira execução completa do eval set (25 casos) retornou:

| Resultado | Quantidade |
|---|---|
| Passou | 22 |
| Falhou (limitação conhecida) | 3 |
| Falhou (inesperado) | 0 |
| Não testável | 0 |

As 3 falhas esperadas correspondem a limitações já documentadas neste arquivo: reconhecimento de saudação em inglês (Seção 11), o falso negativo de grounding do caso "reembolso demorando um pouco mais" (Seção 8), e o caso de troca de produto (Seção 13.5). Zero falhas inesperadas confirma que as mudanças de chunking, prompt e verificador desta sessão (Seção 12) não regrediram nenhum comportamento validado anteriormente por este eval set.

### 13.7 Limitação identificada no próprio eval set

Nenhum dos quatro tipos de check implementados valida qual política de negócio foi de fato aplicada na resposta final — apenas se contexto foi encontrado, se a resposta é fundamentada, e se falta informação. O caso central da Seção 12 (Max escolhendo a política errada entre duas candidatas) não teria sido capturado por este eval set: a resposta seguindo a política incorreta (12.1-12.3) também estava tecnicamente "fundamentada" no contexto (o Chunk 1 é texto real da base) e também "encontrava contexto" — ambos os checks existentes teriam retornado `PASSOU` mesmo com a resposta de negócio incorreta.

Fica documentado como próximo passo: um quinto tipo de check (ex.: `resposta_deve_conter` / `resposta_nao_deve_conter`, testando substring esperada ou proibida na resposta final) cobriria esse tipo de caso, mas não foi implementado nesta versão — decisão consistente com o restante deste documento, de tratar evolução do eval set como parte do próximo passo já listado na Seção 16, não como algo a resolver no mesmo ciclo em que o eval set básico foi criado.

## 14. Separação de responsabilidades: reorganizando `main.py` em módulos

Com a adição de persistência, grounding verification e detecção de saudação, `main.py` acumulou seis funções de propósitos distintos além do próprio loop de conversa — carregamento de prompt, carregamento e persistência da base de conhecimento, carregamento do índice de saudação, busca de contexto, detecção de saudação e verificação de grounding. Testar novas funcionalidades (como um eval set automatizado) sobre esse arquivo único tornaria a leitura progressivamente mais difícil.

**Critério usado para dividir:** não foi "uma função por arquivo", nem apenas "o que usa o quê" — o critério foi agrupar funções que compartilham o mesmo domínio do problema e o mesmo momento de execução no fluxo.

- **`inicializacao.py`** — `carregar_prompt`, `carregar_base_conhecimento`, `carregar_indice_saudacoes`. Todas rodam uma única vez, antes do loop de conversa começar, preparando recursos que serão reutilizados.
- **`busca_semantica.py`** — `buscar_contexto`, `eh_saudacao`. Rodam a cada mensagem do cliente; ambas fazem o mesmo tipo de operação (comparação de embedding contra um vectorstore), apenas contra bases diferentes.
- **`verificacao_llm.py`** — `verificar_grounding`. Também roda a cada mensagem, mas por um mecanismo distinto: julgamento via chamada ao LLM, não busca vetorial. Separado das funções de busca semântica mesmo rodando na mesma etapa do fluxo, porque o tipo de operação é fundamentalmente diferente.
- **`main.py`** — a função `main()`, o loop de conversa, e a configuração de logging. A configuração de logging foi mantida aqui, não extraída para um arquivo próprio: hoje é pequena o suficiente (5 linhas) para não prejudicar a leitura, mas é candidata natural a um módulo separado se passar a registrar eventos proativos do próprio código, além dos avisos de biblioteca que captura hoje.

**Regra de import entre arquivos de um mesmo projeto:** cada arquivo precisa dos próprios imports das bibliotecas que usa para *criar* objetos (`ChatOpenAI(...)`, `FAISS.from_documents(...)`, `Document(...)`) — não existe herança de import entre arquivos Python. Uma função que apenas *usa* um objeto já pronto, recebido como parâmetro (ex.: `llm.invoke(...)`, `vectorstore.similarity_search(...)`), não precisa importar a classe daquele objeto — só precisa que ele já tenha sido criado em algum lugar antes de chegar até ali. Por esse motivo, `busca_semantica.py` e `verificacao_llm.py` não têm nenhum import de LangChain: as funções neles só chamam métodos de objetos que `inicializacao.py` e `main.py` já criaram.

**Consequência colateral observada:** a primeira execução após a reorganização gerou uma pasta `__pycache__/` dentro de `src/` — comportamento automático do Python ao importar módulos locais pela primeira vez (compila cada arquivo importado para bytecode e armazena em cache, para acelerar execuções futuras se o arquivo não mudar). Adicionada ao `.gitignore` pelo mesmo motivo que `faiss_index/` e `logs/`: artefato gerado, regenerável, sem valor de código-fonte.

**Conclusão:** dividir por domínio do problema e momento de execução, em vez de por tamanho de arquivo ou ordem de criação, produziu uma estrutura onde cada módulo pode ser lido (e futuramente testado) de forma isolada. Isso também deixou mais explícito um limite que já existia implicitamente no código: funções de busca semântica e de verificação via LLM têm custos e mecanismos de falha diferentes (Seções 5 e 8), e agora vivem em arquivos diferentes que refletem essa diferença.

**Regressão descoberta após a divisão — ordem de configuração de logging:** dias após a reorganização, o `DeprecationWarning` do `langchain-community` (Seção 10) voltou a aparecer no terminal, em vez de ir para `logs/app.log`. A causa: `main.py` importava `inicializacao.py` — que contém `from langchain_community.vectorstores import FAISS`, o gatilho do aviso — **antes** de `logging.basicConfig()` e `logging.captureWarnings(True)` serem executados. Quando todo o código vivia em um único arquivo, a ordem "configurar logging primeiro, importar LangChain depois" era natural; ao separar em módulos, os imports locais (`from inicializacao import ...`) ficaram no topo do arquivo por convenção, antes da configuração de logging que vinha logo abaixo — invertendo, sem intenção, a ordem que fazia a captura funcionar.

**Correção:** mover toda a configuração de `BASE_DIR`, criação da pasta `logs/`, `logging.basicConfig()` e `logging.captureWarnings(True)` para **antes** dos imports locais em `main.py`.

**Lição:** dividir código em módulos preserva a lógica de cada função, mas não preserva automaticamente a *ordem relativa* de efeitos colaterais que dependiam de sequência (como configurar um sistema de logging antes de qualquer import que possa disparar um aviso). Esse tipo de regressão é silencioso — o programa continua funcionando, só o comportamento observável (o que aparece no terminal vs. no log) muda — por isso só foi percebido ao rodar o programa normalmente, não por erro ou teste automatizado.

## 15. Conclusões gerais

- RAG reduz alucinação, mas não a elimina — mesmo com contexto correto recuperado, o modelo pode combinar fatos legítimos de formas não autorizadas pelo negócio.
- Instruções em linguagem natural no *system prompt* têm um teto de eficácia: proibições, checagens explícitas e restrições literais foram testadas e nenhuma bloqueou o comportamento por completo.
- `chunk_size` e `k` não devem ser avaliados isoladamente — o tamanho da base de conhecimento determina se os efeitos de cada um ficam visíveis ou escondidos.
- Um `score_threshold` calibrado com dados reais é mais robusto que um `k` fixo, mas ainda depende de uma escolha de engenharia dentro de uma margem, não de um valor absoluto.
- Marcadores de controle (tokens especiais usados para acionar lógica no código) precisam ser distintos de linguagem natural, e a validação correspondente no código deve tolerar variações — nenhuma reprodução de texto por um LLM deve ser considerada 100% garantida.
- Regras que dependem de contagem ou estado ao longo da conversa (como "quantas vezes isso já aconteceu") são mais confiáveis quando controladas por código determinístico do que quando delegadas inteiramente ao modelo.
- Grounding verification com uma segunda chamada ao mesmo modelo reduz drasticamente, mas não elimina, alucinação por inferência — porque o verificador herda parte dos vieses do modelo que está verificando. Um modelo mais forte no papel de verificador comprovadamente reduz esse viés, mas a decisão de adotá-lo é uma escolha de custo, não uma correção óbvia.
- Otimização de performance deve ser guiada por medição, não por sensação: o gargalo percebido nem sempre é o gargalo real, e resolver o problema errado consome tempo sem resultado.
- Silenciar um aviso técnico e redirecioná-lo para um log são decisões diferentes: a primeira descarta informação, a segunda a preserva para diagnóstico sem expô-la à interface do usuário.
- Embeddings são úteis além da recuperação de conteúdo de negócio: classificar tipo de mensagem (saudação vs. pergunta real) é uma aplicação barata da mesma técnica, desde que a margem entre categorias seja validada com casos antagônicos reais, não presumida.
- Separar código por domínio do problema e momento de execução — não por tamanho de arquivo — facilita leitura e testagem isolada; funções que apenas usam um objeto já criado não precisam reimportar a biblioteca que o originou.
- Chunking, texto da base de conhecimento, retrieval, prompt de geração e prompt de verificação são camadas acopladas, não independentes: corrigir uma pode expor — ou até criar — um conflito latente em outra que não foi revisada em conjunto. Um sintoma observado na resposta final pode ter causa em qualquer camada anterior do pipeline.
- Um eval set automatizado, mesmo simples, é mais confiável do que validação manual repetida a partir do momento em que uma mudança de prompt passa a ter potencial de efeito colateral amplo — mas precisa ser mantido sincronizado com mudanças de assinatura nas funções de produção que ele testa, e seus tipos de check definem exatamente o que ele é capaz (e incapaz) de detectar.

## 16. Próximos passos identificados (não implementados ainda)

Ordenados pela sequência de cobertura planejada, não pela ordem de descoberta.

- **Frameworks de avaliação automatizada (evolução do eval set manual):** ferramentas como Promptfoo, DeepEval e RAGAS geram e avaliam casos de teste em maior escala — incluindo red-teaming automatizado (variações adversariais, tentativas de quebrar o sistema) e métricas de RAG específicas (fidelidade, relevância). Diferente do eval set automatizado já implementado (Seção 13), essas ferramentas ajudam a gerar volume e variação de casos, mas ainda dependem de julgamento humano para definir categorias de risco relevantes ao domínio (como a categoria "frases de controle de conversa" descoberta na Seção 5, com "quero sair"). Não adotadas agora por serem um nível de sofisticação maior do que o projeto exige neste estágio — exigem configuração de framework externo e fazem mais sentido como evolução na fase de observabilidade do roadmap de estudos (LangSmith, tracing), quando também se torna possível um segundo nível de evolução: alimentar o eval set com feedback de uso real em produção (perguntas de clientes, sinais implícitos de insatisfação como pedir transferência logo após uma resposta), em vez de depender apenas de casos pensados manualmente antes do deploy.
- **Eval set mais robusto:** uma primeira versão foi implementada (Seção 13), com 25 casos cobrindo saudação, recuperação de contexto, necessidade de mais informação e grounding, rodando via `tests/run_evals.py` contra as mesmas funções usadas em produção. Duas pendências reais continuam em aberto: (a) ampliar o número de perguntas cobertas, incluindo mais variações de pergunta específica, difusa e fora do domínio, para determinar se a taxa de falso negativo do verificador `gpt-4o-mini` (Seção 8) justifica o custo extra do `gpt-4o` em uso real; (b) um quinto tipo de check validando qual política de negócio foi de fato aplicada na resposta final — gap identificado na Seção 13.7, já que os quatro tipos de check atuais não capturariam o caso da Seção 12 (política errada escolhida) se ele se repetisse hoje, porque uma resposta com a política errada ainda pode estar "fundamentada" e "com contexto encontrado". Planejado antes do few-shot, para que os exemplos escolhidos sejam informados por casos mapeados sistematicamente, não apenas pelos que já surgiram por acaso durante os testes manuais.
- **Few-shot prompting:** parcialmente endereçado de forma pontual — a seção "Seleção da política aplicável" adicionada ao `system.txt` (Seção 12.6), o prompt de `verificar_grounding` (Seção 12.8) e o prompt de `verificar_informacao_suficiente` (Seção 13.4) já incorporam exemplos concretos e explícitos, cada um adicionado reativamente no momento em que uma falha específica foi identificada. O que ainda não foi feito é uma aplicação sistemática dessa técnica: escolher exemplos representativos a partir do eval set completo (Seção 13), cobrindo classes de ambiguidade ainda não mapeadas, em vez de reagir apenas aos casos que já surgiram durante os testes manuais. Continua dependendo do eval set mais robusto (item acima) para essa etapa sistemática.
- **Query rewriting (contextual retrieval) e HyDE (Hypothetical Document Embeddings):** `buscar_contexto` recebe apenas a pergunta atual, isolada do histórico da conversa — diferente do LLM de geração, que recebe `messages` completo. Em uma sequência como "meu pedido atrasou" seguida de "e já faz 5 dias", a segunda busca vetorial usaria só a frase vaga, sem termos que o embedding relacione bem ao `politicas.txt`, mesmo a pergunta fazendo sentido no contexto da conversa. Duas técnicas resolvem esse tipo de problema por caminhos diferentes: *query rewriting* reformula a pergunta do usuário (com base no histórico) antes de gerar o embedding de busca; *HyDE* gera uma resposta hipotética para a pergunta e usa o embedding dessa resposta hipotética na busca, em vez do embedding da pergunta em si — a ideia é que uma resposta hipotética tende a ser semanticamente mais próxima de um chunk real (que também é texto de afirmação) do que uma pergunta pura. Ambas exigem uma chamada de LLM adicional antes do retriever. **Decisão consciente de não implementar nenhuma agora:** o objetivo deste projeto é educacional, e o custo extra por mensagem não se justifica na base de conhecimento atual (pequena, ~7 blocos). A decisão entre as duas técnicas — ou nenhuma — depende de um dado que só existe com uma base maior: qual o comportamento real de retrieval multi-turno em um volume de conteúdo mais próximo de produção. Fica planejado reavaliar as duas quando a base de conhecimento for expandida (ver item "base de conhecimento maior via PDF" no roadmap de estudos), testando-as sob a mesma base, em vez de decidir com a base pequena atual. Essa decisão é específica ao estágio de estudos do projeto: em um cenário de produção com volume real de conversas multi-turno, o custo por chamada deixaria de ser o critério dominante frente ao impacto de uma busca ruim na experiência do cliente, e a decisão provavelmente pesaria a favor de implementar uma das duas técnicas.
> Reavaliado com a base em PDF nas Seções 18 e 19 — conclusão final: nenhuma das duas técnicas encontrou cenário real de uso na arquitetura atual do Max.
- **Crescimento ilimitado do histórico de mensagens:** `messages` acumula toda a conversa (`HumanMessage` e `AIMessage`) sem nenhum mecanismo de limite, e a lista inteira é reenviada ao modelo a cada nova pergunta. Isso gera dois problemas reais em conversas longas: custo cumulativo crescente por mensagem (a N-ésima pergunta reenvia todas as N-1 anteriores), e risco de exceder a janela de contexto máxima do modelo, o que causaria falha na chamada. É uma limitação já ativa hoje, não apenas hipotética — mas invisível no padrão de uso atual, porque as sessões de teste realizadas até aqui nunca foram longas o suficiente para o sintoma se manifestar de forma perceptível (nem em custo, nem em erro de janela excedida).
  Duas abordagens comuns resolvem isso, cada uma com trade-off diferente: **janela de mensagens recentes** (`ConversationBufferWindowMemory` do LangChain, ou truncamento manual — mecanicamente a mesma solução, via biblioteca ou código próprio), que mantém só as últimas N mensagens sem custo de chamada adicional, mas corre o risco de descartar informação relevante mencionada fora da janela (ex.: região do cliente, dita no início de uma conversa longa); e **memória com resumo periódico**, em que uma chamada ao LLM condensa o histórico acumulado quando um limite é atingido (não a cada mensagem), preservando mais contexto relevante ao custo de uma chamada extra periódica.
  Não implementado agora: testar isso de forma significativa exigiria criar cenários de conversa longa manualmente, o que não compensa o esforço no estágio atual do projeto — mais sensato avaliar quando houver uma base de conhecimento maior e um processo de testes automatizados em vigor (eval set, e possivelmente os frameworks descritos acima), em vez de simular manualmente conversas extensas agora. Corresponde à Fase 3 do roadmap de estudos ("Memória de conversa"). Assim como as demais decisões de custo deste documento, essa é uma escolha calibrada para o volume de uso de um projeto de estudos; em produção, com conversas mais longas e recorrentes, o mesmo problema deixaria de ser tolerável e a implementação de uma das duas abordagens passaria a ser necessária, não opcional.
> Atualização: a Seção 21 implementou memória de conversa, mas por um mecanismo diferente do descrito aqui — histórico entre sessões via `buscar_historico_anterior` (tool calling + banco de dados), não janela/resumo da lista `messages` em uso dentro de uma única sessão. O problema específico descrito neste item (crescimento ilimitado de `messages` dentro de uma sessão longa) foi resolvido na Seção 24, via resumo progressivo combinado com janela de mensagens recentes.
- **Detecção de mensagens fragmentadas ou incompletas:** cenário identificado na Seção 11, mas não implementado por exigir julgamento semântico (provavelmente via LLM), reintroduzindo custo de chamada por mensagem recebida.
- **Cálculo de prazo restante personalizado (ex.: "falta 1 dia até o pedido entrar em atraso"):** identificado ao testar a nova regra de "atraso dentro do prazo" — a resposta do Max, mesmo fundamentada, é genérica ("aguarde mais um pouco"), porque o sistema não coleta nem retém dados específicos do pedido (região, data de compra) durante a conversa. Resolver isso exigiria o modelo perguntar essas informações e, mais importante, extraí-las de forma estruturada (não só texto livre) para permitir um cálculo real de data. Não implementado agora por abrir escopo novo (extração estruturada + lógica de cálculo), fora do que uma regra de conteúdo ou prompt resolveria sozinho. Fica planejado para quando `structured output` (Pydantic, JSON mode) for estudado, conforme o roadmap de estudos.
> Atualização: implementado, investigado a fundo e **revertido** na Seção 25 — a extração estruturada e o cálculo funcionaram, mas a integração no pipeline elevou a taxa de transferência desnecessária para um cenário específico (pedido ainda não postado) de ~4% para 30-55%, um custo de confiabilidade desproporcional ao ganho (personalização de um caso raro e de baixo risco). Decisão registrada como estudo de caso de reversão, não como pendência em aberto.

---

# Parte 2 — Depois da migração para base em PDF: evoluções e novas dependências

## 17. Migração da base de conhecimento para PDF: extração estrutural via metadados de fonte

Este experimento realiza o item "base de conhecimento maior via PDF", mencionado na Seção 16 como parte do roadmap de estudos ligado a query rewriting e HyDE — a decisão de reavaliar essas técnicas "quando a base de conhecimento for expandida" pressupõe justamente o trabalho documentado aqui.

### Contexto

Todas as seções anteriores (1-16) documentam o comportamento do Max contra a base de conhecimento original, um `politicas.txt` de formato controlado, com seções separadas por linha em branco. Esta seção documenta o início da investigação de escala: substituir essa base por um PDF de 6 páginas (`politicas_xyz.pdf`), mais próximo de um documento real de produção — texto desordenado, capítulos com formatação inconsistente, uma tabela, e um capítulo "piloto" com estrutura visivelmente diferente do resto.

O trabalho foi conduzido numa branch separada (`experimento-base-pdf`), preservando `politicas.txt` e o eval set de 26 casos intactos em `main` como referência de comparação.

### Por que o splitter antigo não se aplica

O `dividir_por_secao` original depende de `\n\n` como separador de seção — uma convenção que só existia porque o `.txt` foi escrito manualmente por mim, seguindo esse padrão. Um PDF processado por `pdfplumber.extract_text()` não preserva essa convenção: quebras de linha refletem a geometria da página, não a estrutura lógica do conteúdo. A inspeção da saída produzida por `extract_text()` confirmou essa hipótese antes da implementação do pipeline de chunking.

### Extração: pdfplumber

A biblioteca pdfplumber foi adotada porque separa `extract_text()`/`extract_words()` (texto corrido) de `extract_tables()` (extração geométrica de tabelas) — necessário porque o documento contém uma tabela de prazos por região (Capítulo 1).

### Descoberta da hierarquia de heading: abordagem por metadado de fonte, não regex

Duas abordagens foram avaliadas para identificar títulos de seção:

- **Regex sobre padrão de numeração** ("1.1", "2.2"): descartada. O documento tem inconsistência real de formatação — a seção "3.3" aparece com ponto final ("3.3.") e um mini-título antes dela, diferente do padrão das demais seções do mesmo capítulo.
- **Metadados de fonte** (tamanho + nome, via `extract_words(extra_attrs=['size', 'fontname'])`): escolhida. Sinal mais robusto porque não depende de como a numeração foi digitada.

Metodologia de descoberta: extração de todas as palavras do documento com metadado de fonte, agrupamento por `(tamanho, fonte)` via `collections.Counter`, e inspeção de exemplos por perfil para confirmar a hipótese antes de qualquer regra de classificação. Resultado: 6 perfis distintos —

| Perfil | Papel |
|---|---|
| (18.0, Helvetica-Bold) | Título do documento (H1) |
| (15.0, Helvetica-Bold) | Título de capítulo (H2) |
| (12.0, Helvetica-Bold) | Seção numerada (H3) |
| (11.0, Helvetica-BoldOblique) | Run-in (subtítulo sem numeração própria, ex: "Modalidade expressa") |
| (10.0, Helvetica) | Corpo do texto |
| (9.0, Helvetica) | Conteúdo da tabela (candidato a tratamento via `extract_tables()`) |

Essa distribuição evidencia que a hierarquia estrutural do documento está codificada principalmente nos metadados tipográficos, e não em convenções textuais como numeração ou espaçamento.

A classificação foi implementada de forma dinâmica (perfil mais frequente = corpo; perfis maiores que o corpo, ordenados, = headings; o menor dos "maiores que o corpo" = run-in), sem valores hardcoded — decisão deliberada para que a lógica se adapte a mudanças futuras na base sem recalibração manual. O objetivo não foi resolver a extração para este PDF especificamente, mas para uma classe de documentos com convenções tipográficas semelhantes.

### Estratégia de chunking: parent-child retrieval

Avaliadas três abordagens de chunking estrutura-consciente: (1) reaplicar o splitter por tamanho fixo dentro de cada H3, (2) parent-child retrieval (child pequeno para embedding/busca, H3 inteiro como contexto retornado ao LLM), (3) chunking semântico ou assistido por LLM. A opção 3 foi descartada por custo/latência desnecessários dado que o documento tem estrutura de fonte confiável. Além do custo computacional, utilizar uma LLM para segmentar um documento cuja estrutura já pode ser inferida diretamente da tipografia introduziria uma fonte adicional de variabilidade sem benefício proporcional. Escolhida a opção 2 (parent-child), com H3 como limite do chunk-pai e run-in como limite do child.

### Casos de borda identificados durante a implementação do agrupamento (`montar_chunks`)

1. **Título fragmentado por palavra**: a lógica inicial abria um novo chunk a cada palavra de um heading multi-palavra, em vez de reconhecer a sequência como um único título. Corrigido comparando o perfil da palavra atual com o da anterior, concatenando quando idênticos.
2. **H1/H2 virando chunk vazio**: capítulos e o título do documento, sem texto de corpo entre eles e o primeiro H3, geravam chunks sem conteúdo útil. Corrigido tratando apenas o heading de menor nível (H3) como limite de chunk; H1/H2 passaram a se acumular como contexto anexado ao próximo H3.
3. **Contexto acumulando histórico indevidamente**: `contexto_atual` era uma lista que só crescia — chunks de capítulos posteriores carregavam todos os capítulos anteriores no contexto, não só o relativo. Corrigido trocando para um dicionário indexado por nível de heading, com sobrescrita (não acúmulo) a cada novo H2/H1.
4. **Texto órfão entre H2 e o primeiro H3** (ex: parágrafo de abertura do Capítulo 2) vazava para o chunk da última seção do capítulo anterior. Corrigido com uma flag de estado (`aguardando_chunk`) que redireciona esse texto para o contexto do heading superior em vez do `chunk_atual` desatualizado.
5. **Capítulo sem nenhuma seção numerada (Capítulo 6, piloto)**: por não ter H3 algum, seu conteúdo nunca disparava criação de chunk — o texto ficava preso em uma variável nunca lida, e seu run-in sobrescrevia inadvertidamente o `run_in_ativo` do último chunk H3 válido (contaminação cross-capítulo, sem erro visível). Corrigido com uma regra geral (não hardcoded para o Capítulo 6 especificamente): ao fechar um H2 que nunca teve H3 associado, seu título e corpo acumulados são promovidos a um chunk de fallback.

### Limitação identificada nesta etapa (resolvida na etapa seguinte, abaixo)

Na primeira versão do agrupamento, `run_in_ativo` capturava apenas o último run-in encontrado antes do chunk fechar — em seções com múltiplos run-ins (ex: 3.2, ou o próprio Capítulo 6, que tem três subtítulos em maiúscula), apenas o último era preservado no metadado do chunk-pai. Aceitável para o nível de contexto (pai), mas era justamente o problema que o split fino por run-in (child) precisava resolver.

### Split fino do child por run-in: completando o parent-child retrieval

Com a hierarquia de chunks-pai (H3) validada, a etapa seguinte implementou a metade que faltava do parent-child retrieval: subdividir o `texto` de cada chunk-pai em uma lista de **filhos**, um por run-in — em vez de uma lista única de palavras com apenas o último `run_in_ativo` registrado.

**Estrutura de dados adotada:** cada chunk-pai passou a carregar `'filhos': [...]`, uma lista de blocos `{'run_in': título_ou_None, 'texto': [...]}`. O primeiro filho de cada chunk-pai tem `run_in: None` — representa o texto de abertura da seção, antes de qualquer sub-assunto (run-in) aparecer. Cada run-in encontrado fecha o filho anterior e abre um novo, usando o mesmo princípio de "variável de trabalho que sempre existe" já usado para `chunk_atual` na etapa de heading.

**Bug exposto por essa mudança — Capítulo 6 voltou a perder conteúdo:** a primeira versão do split fino zerou os filhos do Capítulo 6 (`0 filho(s)`), reintroduzindo — por um mecanismo novo — o mesmo tipo de perda de dado já corrigido na etapa anterior para esse capítulo. Causa: a flag `aguardando_chunk` só virava `False` ao encontrar um H3, e o Capítulo 6 não tem H3 nenhum; como resultado, todo o texto de corpo do capítulo (mesmo depois de um run-in) continuava caindo no branch de "contexto acumulado" em vez de virar `texto` de um filho. Corrigido fazendo o run-in também destravar `aguardando_chunk`, e ajustando `fechar_h2_sem_h3()` para promover o texto de abertura acumulado (antes do primeiro run-in) ao primeiro filho do chunk de fallback, em vez de descartá-lo. `corpo_por_nivel` também foi convertido de string concatenada para lista de palavras, para servir tanto o contexto de chunks normais quanto o texto de filhos do chunk de fallback com a mesma estrutura de dado.

**Resultado validado:** rodando o agrupamento atualizado contra os 15 chunks do documento, o Capítulo 6 passou a produzir corretamente 4 filhos (texto de abertura + os três run-ins em maiúscula — "PRAZOS INTERNACIONAIS", "EXTRAVIO EM ENVIO INTERNACIONAL...", "REEMBOLSO — TRIBUTOS E TAXAS ALFANDEGÁRIAS"), e seções com múltiplos run-ins em capítulos normais (1.1, 1.2, 2.2, 3.2, 4.1, 5.2) também passaram a preservar todos os seus run-ins como filhos distintos, em vez de só o último.

Com isso, o parent-child retrieval está estruturalmente completo: cada chunk-pai carrega contexto + título + uma lista de filhos, prontos para virar unidades individuais de embedding/busca, com o pai disponível para ser devolvido como contexto completo ao LLM na geração.

### Extração e reconciliação da tabela por posição

Com o parent-child validado, a última peça pendente era o tratamento da tabela de prazos por região (Capítulo 1) — identificada como caso crítico por ser, em tese, um dos maiores volumes de pergunta real de usuário (prazo por região).

**Exploração inicial:** `page.extract_tables()` foi testado isoladamente antes de qualquer lógica de reconciliação, confirmando a forma bruta dos dados — uma lista de linhas, cada linha uma lista de células, com o cabeçalho (`['Região', 'Prazo padrão', 'Prazo expresso']`) na primeira posição.

**Formatação de cada linha como child independente:** decisão já estabelecida em sessão anterior — cada linha da tabela vira um child próprio (não a tabela inteira), para que o embedding de cada região carregue sinal quase puro daquela região, evitando a diluição semântica que aconteceria se as quatro regiões fossem embutidas num único vetor.

```python
def formatar_linhas_tabela(tabela, titulo_secao):
    cabecalho = tabela[0]
    linhas_formatadas = []

    for linha in tabela[1:]:
        partes = [f"{cabecalho[i]}: {valor}" for i, valor in enumerate(linha) if valor]
        linhas_formatadas.append(f"{titulo_secao}. " + '. '.join(partes) + '.')

    return linhas_formatadas
```

Cada linha formatada usa o cabeçalho como rótulo por célula (`"Região: Sul. Prazo padrão: 3 a 4 dias úteis. Prazo expresso: 2 dias úteis."`), prefixada com o título da seção de origem (*chunk enrichment*, mesma técnica já usada na Seção 12.4) — garantindo que cada linha seja autossuficiente mesmo isolada do chunk-pai.

**Atualização (integração em produção):** o prefixo de título deixou de ser aplicado dentro de `formatar_linhas_tabela` e foi centralizado na camada de conversão pai-filho (`formatar_texto_filho`), passando a cobrir também o filho de abertura de seção, que até então não tinha esse tratamento — ver "Correção de retrieval: título ausente no filho de abertura de seção", mais abaixo.

**Decisão consciente de manter a numeração do título ("1.1") no prefixo:** identificado como ruído semântico puro para o embedding (o projeto não usa essa numeração como referência ou citação em nenhum outro ponto do pipeline), mas mantido por decisão deliberada de custo-benefício — o ganho de removê-lo (poucos tokens) não justificou o custo de engenharia associado (uma implementação sem dependência nova adicionava complexidade ao código para um problema de impacto mínimo). Avaliada e descartada a opção via `import re`, especificamente pelo princípio já aplicado no restante do projeto de não empilhar imports desnecessários quando o ganho é marginal.

**Reconciliação por posição:** o problema central — `extract_words()` e `extract_tables()` operam de forma cega um ao outro, então usá-los em paralelo sem reconciliação duplicaria o conteúdo da tabela (uma vez bagunçado, dentro do texto corrido; outra vez estruturado). A solução adotada usa `page.find_tables()` (variante de `extract_tables()` que expõe `.bbox`, a caixa delimitadora da tabela) para obter a faixa vertical (`top`/`bottom`) onde a tabela vive na página, e comparar essa faixa contra a posição (`top`) de cada palavra devolvida por `extract_words()`.

```python
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
```

Palavras fora da faixa da tabela entram normalmente na lista; palavras dentro da faixa são descartadas, e um marcador único (`{'tabela': ..., 'top': ...}`) é inserido no lugar delas, na posição correta da sequência — preservando a ordem de leitura do documento sem duplicar conteúdo.

**Decisão de arquitetura — reconciliação na função de extração, formatação em `montar_chunks`:** uma primeira tentativa formatava e "fingia" a tabela como uma palavra de run-in já dentro de `extrair_palavras()`, mas essa função não tem acesso ao título da seção corrente (informação que só `montar_chunks()` acumula durante seu próprio processamento). A solução final mantém `extrair_palavras()` responsável só por marcar onde a tabela está, e delega a `montar_chunks()` — que já sabe o título da seção ativa a cada ponto do loop — a responsabilidade de formatar e inserir os filhos correspondentes.

```python
def inserir_filhos_tabela(dados_tabela):
    titulo_secao_atual = chunk_atual['titulo'] if chunk_atual else titulo_por_nivel.get(nivel_h2, '')
    linhas = formatar_linhas_tabela(dados_tabela, titulo_secao_atual)
    destino = chunk_atual['filhos'] if (chunk_atual and teve_h3_no_h2_atual) else filhos_h2_orfao
    for linha in linhas:
        destino.append({'run_in': None, 'texto': [linha]})
```

No loop principal de `montar_chunks`, o marcador de tabela é interceptado **antes** de qualquer outra checagem (`if 'tabela' in palavra:`), já que ele não tem os campos `size`/`fontname` que o restante do loop assume presentes — tentar acessá-los geraria `KeyError`.

**Bug exposto pela integração — funções auxiliares assumindo lista homogênea:** ao rodar a integração completa, `identificar_perfis()` (escrita antes da existência do marcador de tabela) quebrou com `KeyError: 'size'`, porque itera a lista de `palavras` inteira presumindo que todo item tem esse campo — suposição que deixou de ser verdadeira no momento em que `extrair_palavras()` passou a inserir um tipo de item diferente na mesma lista. Corrigido filtrando o marcador logo no início da função (`palavras_com_fonte = [p for p in palavras if 'size' in p]`), antes de qualquer contagem. Padrão a ter em mente: qualquer função futura que itere `palavras` diretamente precisa da mesma proteção, já que a lista deixou de ser homogênea.

**Resultado validado:** rodando o pipeline completo contra o PDF, o chunk da seção 1.1 passou de 2 para 6 filhos — o filho de abertura, as 4 linhas de tabela (uma por região, com o prefixo "1.1 Prazos de entrega por região" preservado) inseridas exatamente na posição onde a tabela aparece no documento original, e o filho "Modalidade expressa" na sequência correta logo em seguida. Os demais 14 chunks permaneceram idênticos aos já validados, confirmando que a mudança não teve efeito colateral fora da seção que de fato contém tabela.

Com isso, as duas pendências da Seção 17 (split fino por run-in e tratamento de tabela) estão implementadas e validadas.

### Resultado final

Ao final desta etapa, o pipeline passou a reconstruir automaticamente a hierarquia lógica do documento a partir dos metadados tipográficos — incluindo tratamento dedicado para conteúdo tabular, reconciliado por posição para não duplicar informação — produzindo uma estrutura parent-child completa (chunk-pai por seção H3, filhos por run-in e por linha de tabela), sem depender de convenções de formatação específicas do documento-fonte nem de valores hardcoded.

### Integração em produção

O pipeline validado em `tests/debug_extracao_pdf.py` foi promovido para `src/extracao_pdf.py`, mantendo a lógica de extração e chunking sem alterações de comportamento, e adicionando duas novas responsabilidades: conversão dos chunks pai-filho para `Document`s do LangChain, e a correção de retrieval descrita a seguir.

### Correção de retrieval: título ausente no filho de abertura de seção

O filho de abertura de cada seção (texto antes do primeiro run-in, `run_in: None`) não carregava nenhuma identificação da seção no próprio `page_content` — diferente das linhas de tabela, que já embutiam o título (seção anterior). Isso criava um ponto cego real: perguntas genéricas sobre a mecânica de uma regra (ex.: "os feriados contam como dia útil no prazo?") tendiam a cair nesse filho "solto", sem nenhuma palavra-chave da seção ajudando o embedding a ancorá-lo semanticamente.

A correção centralizou o prefixo de título — antes só aplicado à tabela — para todo filho sem run-in, movendo essa responsabilidade para a camada de conversão (`formatar_texto_filho`), e removendo o prefixo manual de dentro de `formatar_linhas_tabela` para não duplicar.

```python
def formatar_texto_filho(filho, titulo_secao):
    texto = ' '.join(filho['texto'])
    if filho['run_in']:
        return f"{filho['run_in']}: {texto}"
    return f"{titulo_secao}. {texto}"
```

**Efeito colateral aceito e documentado:** o `page_content` do chunk-pai passou a ter o título ligeiramente duplicado (título do chunk + título repetido dentro do primeiro filho concatenado a ele). Irrelevante na prática, já que o pai nunca é embedado — só é devolvido como contexto para o LLM, onde a repetição não tem custo funcional.

### Arquitetura de retrieval: ParentDocumentRetriever descartado em favor de retriever próprio

Testado inicialmente com `langchain_classic.retrievers.ParentDocumentRetriever` + `LocalFileStore`, mas descartado antes de ir para produção:

- `langchain_classic` é um pacote de compatibilidade separado, introduzido na reorganização do LangChain 1.0, com suporte limitado a correções de segurança até dezembro de 2026 — não é uma base recomendada para código novo.
- A documentação atual do LangChain não cataloga mais `ParentDocumentRetriever` como estratégia de retrieval; trata "Retriever" como uma interface (`BaseRetriever`) a ser implementada por projeto, não uma classe pronta para cada padrão.
- Confirmado por precedente real: a migração do pacote `langchain-mongodb` para LangChain 1.0 substituiu o uso de `ParentDocumentRetriever` por uma subclasse própria de `BaseRetriever`.

Alternativas mapeadas antes da decisão final:

| Abordagem | Onde é usada | Trade-off |
|---|---|---|
| `ParentDocumentRetriever` (langchain_classic) | Legado, ainda funcional | Pacote em manutenção, não desenvolvimento ativo |
| `AutoMergingRetriever` (LlamaIndex) | Outro framework, ativamente mantido | Exigiria migrar o projeto inteiro de framework por causa de um único componente; comportamento de merge diferente (só promove o pai quando um número mínimo de filhos do mesmo pai aparece nos resultados, não qualquer um) |
| Contexto do pai embutido no metadata do filho | Técnica sem dependência extra | Duplica o texto do pai N vezes dentro do índice vetorial (irrelevante em 6 páginas; escalaria mal em bases grandes) |
| **Retriever próprio (`RetrieverPaiFilho`, escolhida)** | `BaseRetriever` do `langchain_core` (núcleo estável) | Mais código para manter, mas controle total e zero dependência de pacote legado |

**Implementação escolhida:** `RetrieverPaiFilho`, subclasse de `langchain_core.retrievers.BaseRetriever`. Busca os `k` filhos mais similares via `similarity_search_with_relevance_scores`, filtra por `score_threshold`, e deduplica pais (necessário porque múltiplos filhos — ex.: duas linhas da mesma tabela — podem apontar para o mesmo pai). Persistência dos pais feita via JSON simples (`salvar_pais`/`carregar_pais`), substituindo `LocalFileStore`/`create_kv_docstore` sem introduzir dependência nova.

```python
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
```

### Recalibração do score_threshold

O `score_threshold=0.68`, calibrado originalmente para os chunks maiores do `politicas.txt` (uma seção inteira por chunk), não generalizou para a granularidade menor dos filhos pai-filho. Confirmado por regressão no eval set: a pergunta fora do domínio "copa do mundo fifa" passou a retornar contexto (score 0.6889, acima do threshold antigo), quando na base anterior ficava em 0.63–0.66.

**Diagnóstico.** Um dump completo de scores (`k=28`, toda a base) para essa mesma query mostrou a distribuição inteira comprimida entre 0.598 e 0.689 — uma faixa muito mais estreita do que o esperado para conteúdo sem nenhuma relação com a pergunta. Isso é consistente com anisotropia de embeddings: qualquer par de vetores compartilha um "chão de ruído" na similaridade de cosseno, independente de relação semântica real; com chunks menores, esse ruído de fundo passa a representar uma fração maior do score total, comprimindo a faixa inteira para mais perto do threshold antigo.

**Recalibração empírica.** Rodado o mesmo dump (`k=28`) para 8 perguntas com resposta clara e específica, cobrindo capítulos diferentes da base. Resultado: pior caso do lado "relevante" (filho correto) em 0.7987; melhor caso do lado "ruído" (fora do domínio) em 0.6889 — uma janela real, porém mais estreita que a anterior (~0.11 contra ~0.19 da base antiga, já que o teto do sinal real também caiu, não só o chão do ruído subiu). Escolhido `score_threshold=0.70`: fica com folga de margem tanto acima do chão de ruído medido (evita ficar colado em 0.6889) quanto abaixo do pior caso relevante medido (0.7987).

### Investigação de regressão aparente no eval set (caso "meu pedido foi extraviado")

Após a recalibração, o eval set completo passou a apresentar apenas uma falha inesperada: `grounding_should_fail` para a pergunta "meu pedido foi extraviado" (esperado `False`, obtido `True`).

Hipóteses testadas e descartadas, nessa ordem:

1. **Ambiguidade introduzida pelo Capítulo 6** (a pergunta, sem mencionar região, poderia agora ser interpretada como internacional, já que a regra de extravio internacional contradiz a doméstica). Descartada: `verificar_informacao_suficiente()` chamado isoladamente sobre essa pergunta retornou `False` — o verificador não considera a pergunta ambígua.
2. **Escalonamento para atendente humano** (o Max poderia, em algumas execuções, decidir transferir em vez de responder, o que o código do eval set trata como equivalente a "grounding falhou"). Descartada: 20 execuções isoladas de geração de resposta produziram texto idêntico, sem o marcador `TRANSFER_HUMANO` em nenhuma delas.
3. **Instabilidade conhecida do veredito de grounding** (documentada na Seção 8 — o verificador pode divergir entre execuções mesmo sobre o mesmo contexto, por variação de fraseado do `llm_chat`, que roda com `temperature=0.3`). Confirmada por eliminação: reexecutar o eval set completo do zero não reproduziu a falha. Taxa observada: 1 falha em 27 tentativas (~3.7%) somando todas as reproduções manuais e as duas rodadas completas do eval set.

**Conclusão:** não é uma regressão da migração para PDF, nem do retriever novo, nem do threshold recalibrado — é uma instância da instabilidade já conhecida e documentada, agora com uma taxa de ocorrência aproximada medida. Nenhuma mudança de código ou de eval set necessária para esse caso.

### Resultado final da integração

Eval set (26 casos) rodando limpo contra a base em PDF, com `score_threshold=0.70` e o `RetrieverPaiFilho` em produção — encerrando o item "base de conhecimento maior via PDF" do roadmap da Seção 16.

### Próximos passos

- Nenhuma pendência estrutural em aberto para a extração, o retriever ou a calibração de threshold contra a base em PDF.
- Retomar, nesta ordem, os itens da Seção 16 que dependiam de uma base maior: (1) reavaliar query rewriting/HyDE, agora com base heterogênea o suficiente (múltiplos capítulos, tabela, regras contraditórias entre si) para o teste fazer sentido; (2) few-shot prompting sistemático, usando o eval set expandido; (3) decisão sobre detecção de mensagem fragmentada (opcional).
- Memória/histórico de conversa continua adiado até o projeto passar a usar banco de dados (Fase 3 do roadmap de estudos), sem mudança nessa decisão nesta etapa.
> Atualização: banco de dados e memória de conversa via tool calling implementados na Seção 21.
- O item (1) acima passou por um desvio de escopo antes de chegar à implementação de fato — ver Seção 18.

---

## 18. Ambiguidade nacional/internacional: escopo de `needs_more_information` vs. query rewriting/HyDE

Ao abrir a branch `experimento-query-rewriting-hyde` para reavaliar as duas técnicas (item planejado desde a Seção 16), a primeira pergunta prática — "quais casos do eval set cada técnica deveria atacar?" — expôs um problema mais fundamental antes mesmo de qualquer código de rewriting/HyDE ser escrito.

### O problema: duas regras contraditórias sobre extravio, sem forma de saber qual aplicar

O Capítulo 6 (Envios Internacionais, piloto — Portugal e Estados Unidos) contradiz o Capítulo 2 na regra de extravio: no fluxo doméstico, o relato direto do cliente já é suficiente para acionar reenvio; no fluxo internacional, é necessário aguardar confirmação formal da transportadora parceira, com prazo de até 15 dias úteis. A pergunta `"meu pedido foi extraviado"`, sem menção de região ou país, não contém informação suficiente para saber qual das duas regras se aplica.

**Por que isso não é um problema que query rewriting ou HyDE deveriam resolver.** As duas técnicas, diante de uma pergunta subespecificada, só têm um caminho possível: gerar uma versão mais específica da pergunta (rewriting) ou um documento hipotético que a responda (HyDE) — e nenhuma das duas consegue gerar texto genuinamente ambíguo. Ou seja, a técnica não resolveria a ambiguidade, apenas a esconderia atrás de uma adivinhação implícita do LLM, tornando o erro mais difícil de rastrear do que já é hoje.

**Decisão:** tratar a ambiguidade nacional/internacional como um caso de `needs_more_information` (o mesmo mecanismo já usado para "meu pedido está atrasado há 1 semana" — Seção 13.4), não como um problema de retrieval. Isso desloca o problema para a camada correta do pipeline: em vez de tentar melhorar a busca para um caso em que a busca não tem informação suficiente para decidir, a pergunta ambígua é interceptada antes de chegar ao retriever, e o cliente é solicitado a especificar o dado que falta.

Essa decisão reduz o escopo do experimento de rewriting/HyDE: o grupo de casos ligados à ambiguidade de capítulo sai do escopo dessas duas técnicas inteiramente. O experimento de rewriting/HyDE (Seção 19, a ser escrita) passa a focar apenas em casos de descasamento de registro (linguagem casual/verbosa do cliente vs. texto formal do PDF), que é o tipo de problema que as duas técnicas de fato endereçam.

### Extensão do verificador: o dado que falta depende do assunto da pergunta

A primeira tentativa de correção tratou "nacional/internacional" como uma extensão direta do eixo já existente de "região" em `verificar_informacao_suficiente()` — mas uma leitura mais cuidadosa do Capítulo 6 mostrou que a granularidade necessária **depende do assunto da pergunta, não é uniforme**:

- **Prazo de entrega:** muda entre Portugal (10–15 dias úteis) e Estados Unidos (8–12 dias úteis) — saber apenas "internacional" não é suficiente, é necessário o país específico.
- **Extravio:** a regra muda apenas entre nacional e internacional — o procedimento é o mesmo para os dois países cobertos. Saber "internacional", sem o país, já é suficiente.
- **Taxas alfandegárias e outros assuntos não geográficos:** não dependem de região, país ou prazo — o dado nacional/internacional é irrelevante.

Uma primeira versão do prompt, ao tratar erroneamente "meu pedido internacional foi extraviado" como insuficiente (pedindo o país mesmo sem necessidade), foi corrigida após confronto direto com o texto do PDF — reforçando que decisões desse tipo devem ser verificadas contra a fonte, não apenas contra a intuição sobre como a ambiguidade "deveria" funcionar.

O prompt final de `verificar_informacao_suficiente()` (`src/verificacao_llm.py`) resolve o dado que falta por tópico da pergunta, com exemplos contrastantes explícitos (mesma frase-base, "internacional" vs. país específico vs. país não coberto) para fixar a distinção — incluindo o caso de um país fora da cobertura (ex.: França), tratado como limitação de escopo (falta de política), não como falta de dado, e portanto fora da responsabilidade deste verificador.

### Regressão descoberta durante a validação: região doméstica não encerrava a ambiguidade

Rodar o eval set contra a versão corrigida revelou uma segunda falha, não prevista: `"qual o prazo de entrega para o sul?"` — um caso que já passava antes da mudança — passou a retornar `needs_more_information: True` incorretamente.

**Diagnóstico.** O prompt, na primeira correção, só especificava o que contava como suficiente *quando o pedido é internacional* ("é necessário saber qual dos dois países"); nunca afirmava explicitamente que mencionar uma região doméstica (Sul, Sudeste, etc.) resolve a pergunta "nacional ou internacional?" por si só. O modelo tinha que inferir essa regra apenas a partir de um exemplo isolado no prompt, e não generalizou de forma confiável — o restante do bloco de instrução falava quase inteiramente de internacional, então o exemplo isolado não teve peso suficiente.

**Correção.** Adicionada uma frase explícita à regra de PRAZO DE ENTREGA: se o cliente já menciona uma região doméstica, assume-se pedido nacional, sem necessidade de perguntar se é nacional ou internacional. Regra determinística confirmada com o usuário antes da implementação: uma região brasileira mencionada explicitamente (ex. "sul") nunca é ambígua com um destino internacional, porque um cliente com pedido internacional não formularia a pergunta dessa forma.

**Armadilha de processo durante a correção:** a frase de correção foi validada primeiro isoladamente, via um script de debug (`tests/debug_informacao_suficiente.py`, criado nesta etapa para inspecionar o `Raciocínio` bruto do verificador, não só o veredito parseado) — mas a edição nunca foi de fato salva no arquivo de produção (`src/verificacao_llm.py`). O eval set continuou falhando no mesmo caso após um commit que, na aparência, já deveria conter a correção. Identificado ao comparar o conteúdo real do arquivo em produção contra o texto validado no debug script — nenhum dos dois estava desatualizado por engano de lógica, a divergência era puramente entre "testado" e "salvo". Reforça a mesma lição já registrada em outras seções deste documento (Seção 13.3): validar uma mudança isoladamente não garante que ela chegou ao caminho de execução real; `git diff` do arquivo de produção é uma etapa necessária antes de re-rodar qualquer suíte de teste, não apenas depois de uma falha inesperada.

### Instabilidade descartada como causa: diferença de LLM entre chamadas, não do modelo em si

Durante a mesma investigação, duas invocações manuais sucessivas da mesma pergunta (`"qual o prazo de entrega para o sul?"`) produziram vereditos divergentes — o que, à primeira vista, sugeria mais uma instância da instabilidade de veredito já documentada (Seção 8 addendum, agora também observada em `verificar_grounding`). Investigação descartou essa hipótese: as duas chamadas usavam LLMs diferentes (`llm_chat`, `temperature=0.3`, copiado por engano do padrão de outro script de debug; vs. `llm_verificador`, `temperature=0`, o que produção de fato usa). Confirmado, nos call sites reais de `main.py`, que tanto `verificar_informacao_suficiente()` quanto `verificar_grounding()` são chamados com `llm_verificador`. Refeito o teste com o LLM correto (`temperature=0`), o resultado ficou consistente entre execuções — não há evidência de instabilidade nova neste verificador.

### Fechando a lacuna de cobertura exposta pela mudança: dois tipos de check novos no eval set

A mudança em `needs_more_information` expôs uma lacuna já latente no eval set (item "eval set mais robusto", Seção 16): os quatro tipos de check existentes (Seção 13) testam cada etapa do pipeline isoladamente, nunca a sequência real de decisão do `main.py` (`eh_saudacao` → `verificar_informacao_suficiente` → `buscar_contexto` → geração → `verificar_grounding`). Isso significa que os 4 casos existentes construídos em torno de "meu pedido foi extraviado" (nos checks `should_find_context`, `grounding_should_fail`, `needs_more_information` e `contem_texto_proibido`) continuam válidos como testes de função isolada, mas nenhum deles confirma que, no fluxo real de produção, essa pergunta de fato para em `needs_more_information` antes de chegar ao retriever.

Duas lacunas foram fechadas juntas nesta etapa, por serem parte do mesmo item já pendente na Seção 16 (não uma expansão de escopo nova):

**`intercepta_em` — teste de interceptação de pipeline.** Nova função `avaliar_intercepta_em()`, que replica a sequência real de decisão do `main.py` (não uma versão reinventada dela) e retorna em qual etapa o fluxo parou (`'saudacao'`, `'needs_more_information'`, `'sem_contexto'` ou `'passou'`), em vez de um booleano isolado. Caso adicionado: `"meu pedido foi extraviado, o que eu faço?"` deve interceptar em `needs_more_information`.

**`resposta_contem` — validação da política aplicada.** Fecha o gap já identificado na Seção 13.7 (nenhum dos checks existentes valida qual política de negócio foi de fato aplicada na resposta final — uma resposta com a política errada ainda pode estar "fundamentada" e "com contexto encontrado"). Nova função `avaliar_resposta_contem()`, espelhando `avaliar_contem_texto_proibido()` com a lógica invertida (asserção de presença, não de ausência). Caso adicionado: `"qual o prazo do meu pedido para os Estados Unidos?"` deve conter `"8"` na resposta (o prazo de 8–12 dias úteis específico dos EUA, que não aparece em nenhuma outra regra de prazo da base — funcionando como assinatura de que a política correta foi aplicada, e não a doméstica ou a de Portugal).

O terceiro item da Seção 16 relacionado ("ampliar o número de perguntas cobertas") foi deliberadamente deixado de fora desta etapa — decisão de deixá-lo crescer organicamente conforme novos casos surgirem durante os próprios testes de rewriting/HyDE, em vez de um exercício em lote descolado de casos reais.

### Resultado final

Eval set expandido para 28 casos, rodando limpo: 26 passaram, 2 falhas esperadas (limitações já documentadas — reconhecimento de saudação em inglês, Seção 11; caso de reembolso verboso, Seção 8), 0 falhas inesperadas — incluindo os dois casos novos, que passaram já na primeira execução após a implementação.

### Próximos passos

- Escopo de rewriting/HyDE confirmado como restrito a casos de descasamento de registro — a ambiguidade de capítulo já está coberta por `needs_more_information`.
- Item "eval set mais robusto" da Seção 16 parcialmente fechado: pendências (a) interceptação de pipeline e (b) validação de política aplicada, resolvidas; pendência (c) ampliação do volume de perguntas, deixada para crescimento orgânico.
- Seguir para a implementação de fato de query rewriting e HyDE contra os casos de descasamento de registro mapeados no eval set — ver Seção 19.

## 19. Query rewriting e HyDE: resultado negativo — nenhuma técnica encontrou cenário real de uso

Com o escopo já restrito pela Seção 18 (só casos de descasamento de registro, não ambiguidade de capítulo), o passo seguinte foi decidir entre as duas técnicas antes de implementar qualquer uma.

### Descarte de HyDE antes da implementação

HyDE (gerar um documento hipotético de política e embedar esse texto, em vez da pergunta do cliente) resolve um tipo de gap específico: quando a pergunta do usuário é estruturalmente muito diferente do tipo de texto da fonte (ex.: busca em linguagem natural contra artigos científicos densos, ou contra código-fonte). Num chatbot de atendimento, a pergunta do cliente e a política já são o mesmo tipo de discurso — alguém descrevendo uma situação de pedido/entrega, contra um texto que também fala de situações de pedido/entrega, só que em registro mais formal. A distância é de vocabulário e verborragia, não de gênero textual — exatamente o tipo de gap que query rewriting ataca, não HyDE.

Reforço vindo de experiência prática de SAC em e-commerce: perguntas estruturalmente muito distantes de uma declaração de política (parágrafos longos, cheios de cláusulas e ressalvas, tentando deixar claro que "já tentou de tudo") tendem, no atendimento real, a ser pedidos implícitos de falar com um atendente humano, não perguntas que um bot deveria resolver sozinho — cenário que o sistema já cobre via `TRANSFER_HUMANO` e escalonamento por reclamação repetida (política, Seção 5.3), não via retrieval melhor. HyDE foi descartado sem implementação, com esse raciocínio documentado como decisão de escopo, não como omissão.

### Implementação de query rewriting

`reescrever_query()`, adicionada a `src/verificacao_llm.py`: reescreve a pergunta do cliente para um registro mais formal, preservando todo conteúdo factual, com duas restrições centrais no prompt — não inventar dado concreto (número, região, data) que o cliente não mencionou, e preservar deliberadamente vaguidão de tempo/quantidade em vez de resolvê-la com um valor específico. Essa segunda restrição existe porque LLMs, ao serem instruídos a "esclarecer" uma pergunta, tendem a resolver ambiguidade proativamente — um risco de alucinação estrutural do próprio ato de reescrever, não um bug pontual.

Decisão consciente de não adicionar uma verificação de fidelidade pós-reescrita (no espírito de `verificar_grounding()`) antes de ter evidência de que o prompt sozinho falha na prática — mesma filosofia de "corrigir o que quebra, não adicionar rede de segurança preventiva sem dado que justifique o custo" já aplicada em outras partes do projeto.

### Rodada 1 — casos de registro casual/verboso: nenhum ganho de score

Testados os 6 casos originalmente mapeados como descasamento de registro (`"meu pedido ta atrasado"`, a versão verbosa/hesitante do mesmo caso, o caso de reembolso verboso já marcado como limitação conhecida, entre outros), comparando o score de melhor match antes e depois da reescrita (`tests/debug_query_rewriting.py`, criado nesta etapa).

**Resultado:** nenhum dos 6 casos estava perto do `score_threshold` de 0.70 antes da reescrita (todos entre 0.78–0.82). A reescrita, ao normalizar hedging e vaguidão, na prática **piorou** o score em 4 dos 6 casos — a hipótese é que palavras coloquiais "descartadas" pela reescrita tinham, por acaso, alguma sobreposição lexical com o texto da política, e a versão mais enxuta perdeu esse acidente positivo sem ganhar nada em troca, já que não havia gap semântico real a corrigir. Inspeção manual do texto reescrito não mostrou nenhum dado inventado nesses 6 casos — o prompt de fidelidade se comportou como esperado, só que sem ter um problema real pra resolver.

### Rodada 2 — vocabulário fora do domínio e fragmentos curtos: ganho real de score, mas nunca exercido em produção

Levantada uma segunda hipótese: talvez o gap real não estivesse em registro casual/verboso, e sim em perguntas que evitam por completo o vocabulário do domínio (sinônimo, metáfora, gíria) ou são fragmentos muito curtos. Dez casos novos construídos cobrindo essas categorias (ex.: `"sumiu meu troço, o que faço?"`, `"vc pode me ajudar? tipo, faz tempo q n chega naaada"`, `"e aí, cadê?"`).

**Resultado, primeira leitura:** 6 dos 10 casos originais scoraram abaixo do threshold de 0.70 (faixa 0.66–0.70) — cairiam no fallback "não entendi sua pergunta" em produção. A reescrita resgatou 5 desses 6 para acima do threshold (destaque: um caso subiu de 0.6856 para 0.7687). Só um fragmento extremo (`"e aí, cadê?"`) permaneceu abaixo mesmo após a reescrita, e ainda piorou ligeiramente — hipótese: fragmento curto demais, sem nenhum verbo/substantivo do domínio para a reescrita se ancorar.

**Resultado, segunda leitura — o dado que muda tudo.** Reexecutado o mesmo teste passando cada pergunta primeiro por `verificar_informacao_suficiente()`, replicando a ordem real do pipeline (`needs_more_information` roda antes de `buscar_contexto`, Seção 18) em vez de calcular o score isoladamente. **Todos os 10 casos foram interceptados por `needs_more_information` antes de chegar ao retriever** — incluindo os dois fragmentos mais curtos, que uma leitura manual do prompt do verificador sugeria que escapariam (por não descreverem "uma situação específica"), mas que na prática o modelo classificou como insuficientes mesmo assim.

### Conclusão

Nenhuma das duas rodadas produziu um cenário de produção onde query rewriting muda o resultado que o cliente recebe:

- Rodada 1: o score já era suficiente sem rewriting — não havia problema a resolver.
- Rodada 2: o score era de fato insuficiente, e rewriting o corrigia — mas a pergunta nunca chega a esse estágio do pipeline, porque `needs_more_information` já intercepta antes.

`needs_more_information`, na forma como foi estendido na Seção 18, funciona como uma rede de segurança mais ampla do que o previsto inicialmente: não intercepta só ambiguidade nacional/internacional explícita, intercepta qualquer pergunta vaga o suficiente para "parecer" carente de mais dado — o que, na prática, cobre o mesmo espaço de perguntas problemáticas que rewriting foi desenhado para resgatar.

**Decisão:** query rewriting e HyDE descartados como funcionalidades de produção nesta etapa do projeto, documentados como resultado negativo. `reescrever_query()` e os scripts de debug (`tests/debug_query_rewriting.py`) mantidos no repositório como evidência reproduzível do experimento, sem integração em `main.py`. A conclusão é específica à arquitetura atual do Max — um sistema sem `needs_more_information` tão abrangente, ou com um verificador de suficiência mais restrito, poderia ter um cenário real de uso para rewriting que este não tem.

### Próximos passos

- Few-shot prompting sistemático, usando o eval set de 28 casos já validado.
- Detecção de mensagem fragmentada permanece opcional, adiada — decisão de retomar apenas ao preparar o Max para uma configuração de apresentação, não de estudo.
- Próxima fase do roadmap de estudos após consolidar Max: banco de dados (SQL/PostgreSQL/vector DBs), antes da fase de memória/tool calling/agents.
> Atualização: banco de dados implementado e memória/tool calling concluídos na Seção 21; próxima fase real do roadmap passa a ser structured output, MCP e agents.
## 20. Few-shot prompting sistemático: mapeamento do eval set, correção de bugs estruturais e estabilização por camada

Com query rewriting e HyDE descartados (Seção 19), o próximo passo do roadmap era few-shot prompting sistemático: em vez de reagir pontualmente a casos que quebravam (como já havia acontecido em `verificar_informacao_suficiente`, na seção "Seleção da política aplicável" de `system.txt`, e na granularidade por tópico da Seção 18), usar o eval set de 28 casos já validado para mapear classes de ambiguidade de forma deliberada e escolher exemplos representativos de cada uma — não só cobrir o próximo caso que aparecesse.

A decisão de adiar esse trabalho até depois da migração da base para PDF (Seção 17) foi deliberada: escrever exemplos few-shot em cima de uma estrutura de retrieval que ainda ia mudar (chunking pai-filho, granularidade diferente do `politicas.txt` original) arriscaria precisar reescrever tudo depois. Com a base estável e o eval set de 28 casos passando limpo (Seção 18), essa condição estava satisfeita.

### Auditoria do eval set: revisitando expectativas antigas, não só comportamento novo

Antes de escrever qualquer exemplo few-shot, cada grupo de casos do eval set foi revisitado individualmente, testando contra o comportamento real do pipeline (não só a leitura do código) — isso revelou que nem toda "limitação conhecida" documentada continuava sendo, de fato, uma limitação real, e que nem todo caso sem `limitacao_conhecida` estava, na prática, cobrindo o estágio certo do pipeline.

**Casos que se mostraram fora do escopo do few-shot (decisão determinística, não geração):** o grupo de saudação (`is_greeting`) e o grupo de contexto encontrado (`should_find_context`) são resolvidos inteiramente fora do LLM — o primeiro por similarity search contra um vectorstore de exemplos com threshold 0.85, o segundo pelo `score_threshold` do `RetrieverPaiFilho`. Nenhum dos dois passa por um modelo generativo na decisão, então few-shot não se aplica a eles — a mesma lógica que já havia descartado o caso adversarial `"salve meu número de rastreamento"` como candidato a exemplo.

**Reembolso verboso — expectativa desatualizada, não bug do sistema.** O caso `"meu reembolso está demorando um pouco mais que o esperado..."`, documentado como `limitacao_conhecida: true` (`grounding_should_fail: true`), foi testado e o Max respondeu citando quase literalmente a Seção 4.3 do PDF ("acionar o suporte informando o número do protocolo de aprovação original"). Comparando com o texto-fonte, a resposta estava corretamente fundamentada — `verificar_grounding` retornando `False` (não falhou) estava certo. A causa raiz: a Seção 4.3 não existia na base de conhecimento antiga (`politicas.txt`) e foi adicionada só na migração para PDF — o eval set nunca foi atualizado para refletir essa nova cobertura. Corrigido para `grounding_should_fail: false`, `limitacao_conhecida: false`.

**Atraso leve sem contexto suficiente — gap de cobertura, não de comportamento.** O caso `"meu pedido está atrasado só um pouco, ainda não chegou mas também não sumiu, o que eu faço?"` (grupo `grounding_should_fail`) foi testado via `main.py` e revelou ser interceptado por `needs_more_information` antes de chegar à geração — comportamento correto (a pergunta não informa região nem tempo decorrido), mas nunca documentado como tal. Mantido o teste original de `grounding_should_fail` como validação isolada da função (útil por si só), e adicionado um segundo caso com `intercepta_em: "needs_more_information"`, documentando o comportamento real do pipeline ponta a ponta — mesmo padrão dual já usado para os dois casos de extravio sem indicação de país.

### Bug estrutural encontrado durante a auditoria: `"quero sair"` nunca era reconhecido

Testando o grupo `should_find_context`, o caso `"quero sair"` (`limitacao_conhecida: true`) revelou a causa raiz: `main.py` comparava `pergunta.lower() == 'sair'` (string exata), herdada da primeira linha de código do projeto, ~2 meses atrás, criada como placeholder e nunca generalizada. `"quero sair"` não batia com essa comparação e caía no fallback padrão de "não entendi".

**Correção adotada — mesmo padrão arquitetural do `eh_saudacao`:** em vez de expandir a comparação de string com mais variações (frágil por natureza), foi criado um vectorstore de intenção de saída (`carregar_indice_saida()`, em `inicializacao.py`) com exemplos como `'sair'`, `'quero sair'`, `'tchau'`, `'encerrar conversa'`, e uma função `eh_intencao_saida()` (`busca_semantica.py`) que faz `similarity_search_with_relevance_scores` com o mesmo threshold de 0.85 já validado para saudação. Integrado em `main.py` como primeira checagem do loop, antes até da saudação.

**Benefício de custo, não só de robustez:** por ser resolvido via embedding (uma passada de inferência para gerar o vetor, sem geração token a token) em vez de uma chamada completa de chat/completion, o custo é ordens de grandeza menor que uma chamada ao `gpt-4o-mini` — e, como a pergunta do cliente já seria embedded de qualquer forma para o retriever principal caso não fosse saída, o custo incremental dessa nova checagem é mínimo.

Testado com sucesso via `main.py` real. Caso do eval set corrigido para `intercepta_em: "saida"`, `limitacao_conhecida: false` — e `avaliar_intercepta_em()`, em `tests/test_eval_set.py`, precisou ser atualizado para incluir esse novo estágio como primeira etapa checada (a função replica a sequência real de `main.py`, então uma mudança na ordem real do pipeline exige a mesma mudança na função de avaliação — um gap que, se não corrigido, faria o novo caso reportar falha inesperada mesmo com o comportamento real correto).

### Escrevendo o few-shot: três exemplos representativos, não reação a um único caso

Com o eval set auditado, três classes de ambiguidade foram escolhidas para exemplos few-shot na nova seção `## Exemplos`, adicionada ao final de `system.txt`:

1. **Nenhuma política aplicável.** Caso "troca por produto de valor maior, pagando a diferença" — nenhuma seção do PDF cobre esse cenário. O exemplo ensina a não generalizar por analogia com políticas parecidas (reenvio de item incorreto, reembolso por avaria) quando a condição de aplicabilidade não corresponde, e a nomear a lacuna explicitamente em vez de inferir uma solução.
2. **Seleção entre políticas sobrepostas.** Extravio relatado diretamente pelo cliente (Seção 2.3, dispensa espera de 48h) vs. atraso confirmado (Seção 2.2, exige aguardar 48h antes de investigação) — mesmo tema, condições de aplicabilidade diferentes, risco de misturar as duas.
3. **Regra nacional vs. internacional na geração.** Extravio para Portugal — a regra doméstica (relato direto já basta) não pode ser aplicada a um destino internacional, que exige confirmação formal da transportadora parceira (Capítulo 6). Estende para a camada de geração a mesma distinção que a Seção 18 já havia mapeado para `needs_more_information`.

### Instabilidade real: identificando a camada certa para resolver, não só o sintoma

O caso de "troca por produto de valor maior" (limitação conhecida em `needs_more_information: false`) foi revisitado sob o argumento de que não é um caso raro — é um tipo de pergunta plausível em qualquer fluxo real de atendimento de e-commerce, o que descartou tratá-lo como limitação aceitável.

**Teste de estabilidade de `verificar_informacao_suficiente()`:** 10 execuções da mesma pergunta retornaram 6x `False` / 3x `True` (`verificar_informacao_suficiente` com `temperature=0`) — instabilidade real, mesmo com um exemplo few-shot quase idêntico já presente no prompt dessa função (`"posso trocar meu pedido por outro produto de valor maior?"` → NÃO). Testado também com `model_kwargs={"seed": 42}` — a variação persistiu (3/10 ainda divergentes), descartando o `seed` como solução suficiente para esse caso.

**Diagnóstico:** o critério de `verificar_informacao_suficiente()` ("falta um dado do pedido — região, prazo, data?") não é o critério certo para esse caso. A pergunta não carece de dado do pedido; carece de uma política que dê ao Max autoridade para decidir. Forçar essa segunda categoria a se encaixar no critério da primeira é a causa provável da instabilidade — a função está sendo usada fora do seu propósito, não mal calibrada.

**Correção: não mexer em `verificar_informacao_suficiente()`.** A decisão de "o Max tem autoridade para resolver isso sozinho, ou precisa reconhecer o limite e escalar?" já pertence semanticamente à camada de geração + `verificar_grounding()` — exatamente onde o Exemplo 1 do novo few-shot foi escrito. Testado com 10 execuções de `verificar_grounding()` para o mesmo caso, já com o `system.txt` atualizado: resposta **idêntica** e `grounding_falhou = True` em 10/10 execuções — instabilidade eliminada por completo nessa camada.

`verificar_informacao_suficiente()` mantém sua instabilidade residual documentada, mas isso deixou de ser relevante para este caso específico: ele é resolvido de forma estável na camada correta do pipeline, não na camada originalmente questionada.

### Resultado final

Eval set expandido para 30 casos. Duas das três limitações conhecidas existentes no início da sessão foram resolvidas na raiz:

- `"quero sair"`: bug estrutural de comparação de string, corrigido com detecção semântica (mesmo padrão do `eh_saudacao`).
- Troca por produto de valor maior: instabilidade real, resolvida movendo a decisão para a camada correta do pipeline (few-shot em `system.txt`, não ajuste em `verificar_informacao_suficiente()`).

Uma limitação conhecida permanece documentada e deliberadamente não tratada nesta etapa: reconhecimento de saudação em inglês (`"hello"`) — ver "Próximos passos" abaixo.

### Próximos passos

- **Tratamento de idioma (adição futura, não implementada):** a limitação do `"hello"` foi reavaliada à luz da migração para base internacional (Capítulo 6, Portugal/EUA) — a premissa original ("empresa só nacional, perguntas em inglês não deveriam acontecer") não é mais válida. Abordagem escolhida para quando isso for implementado: detecção de idioma como camada determinística e separada no início do pipeline (não instrução ao LLM), com resposta nativa no idioma detectado — não tradução (custo alto, considerado apenas para produção com precificação diferenciada) nem escalonamento automático a humano (dependeria de atendente fluente ou ferramenta de tradução não confiável). Confirmado que a base de conhecimento não precisa ser duplicada: os embeddings da OpenAI são multilíngues por natureza (retrieval cruzado pt/en, a validar empiricamente antes de confiar, no mesmo espírito da calibração de `score_threshold`), e o modelo de geração já lê contexto em português e responde em outro idioma nativamente — o trabalho seria estender os exemplos few-shot já existentes (saudação, `system.txt`) para os idiomas suportados, mais a camada de detecção.
- Próxima fase do roadmap de estudos após consolidar Max: banco de dados (SQL/PostgreSQL/vector DBs), antes da fase de memória/tool calling/agents.
> Atualização: banco de dados implementado e memória/tool calling concluídos na Seção 21; próxima fase real do roadmap passa a ser structured output, MCP e agents.

---

## 21. Memória de conversa e tool calling: da primeira integração de banco de dados no Max à decisão de arquitetura para retomada de assunto

### Contexto e motivação

Até este ponto, o Max nunca havia integrado banco de dados em produção — todo o trabalho de PostgreSQL/SQLAlchemy/pgvector (modelagem relacional, CRUD, extensão vetorial) tinha sido feito isoladamente no repositório de estudos, como aprendizado. As tabelas `clientes`, `conversas` e `mensagens` já existiam no banco `max_db`, mas nada no código do Max se conectava a elas.

Essa sessão cobre a primeira integração real de banco de dados no Max, motivada por um problema concreto: como fazer o Max reconhecer quando um cliente está retomando um assunto de uma conversa anterior (ex: "e como ficou meu extravio?"), sem repetir explicações já dadas.

### Memória de conversa: o que já existia vs. o que faltava

`main.py` já reenvia o histórico completo da conversa a cada chamada — a lista `messages` é criada uma única vez, fora do laço principal, e cresce via `.append()` a cada turno. Isso já é "memória de curto prazo" no sentido técnico correto (o modelo é stateless; é o código que reconstrói o contexto a cada chamada), mas com duas limitações reconhecidas:

1. Essa memória existe só durante a execução do processo Python — nada é persistido entre execuções diferentes do programa.
2. Não existe nenhum tratamento de crescimento de histórico (limite de tamanho, resumo, poda) — problema já identificado anteriormente e listado como pendência.

Esta sessão endereça a primeira limitação (persistência), não a segunda (crescimento/custo do histórico em memória), que segue como item futuro.

### Decisão de identificação de cliente: fora do fluxo de chat, não dentro dele

Uma pergunta central antes de qualquer implementação: como simular múltiplos clientes de forma fiel a um chatbot de e-commerce real, sem inventar um sistema de login completo?

A primeira ideia — perguntar o e-mail do cliente como parte da conversa — foi descartada por não refletir como um chatbot de atendimento real funciona: nenhum widget de chat embutido num site pergunta "qual seu e-mail?" como primeira mensagem. Em produção, se o cliente está logado no site, essa identificação já aconteceu numa camada anterior (o backend do site já sabe o `cliente_id` e passa isso para o chatbot via sessão/token); o widget "já sabe" quem é o cliente antes da primeira mensagem aparecer.

A decisão adotada foi simular essa camada de "login prévio" fora do laço de chat, não como uma pergunta do Max dentro dele: no início de `main()`, antes do `while True:`, um `input()` simula o dado que já viria de um sistema de sessão/login real. É uma simplificação consciente e documentável — o que foi simplificado (autenticação de verdade: senha, JWT, sessão criptografada) e o que foi mantido fiel (a arquitetura de *onde* a identidade do cliente entra no sistema) ficam claros o suficiente para defender em entrevista.

Número de pedido, por outro lado, continua sendo algo que o Max pergunta *dentro* da conversa quando o tópico exigir (rastreamento, atraso, extravio) — porque isso é dado de contexto pontual da pergunta, não identidade do cliente. São dois dados de natureza diferente: um identifica quem está falando (resolvido fora do chat); o outro localiza o que está sendo discutido (resolvido dentro dele).

### Decisão de arquitetura: julgamento do LLM vs. detecção determinística

O padrão já estabelecido no Max para decisões de intenção é detecção determinística via embedding — `eh_saudacao()` e `eh_intencao_saida()` comparam a pergunta contra um vectorstore pequeno de exemplos, com threshold calibrado (0.85). A pergunta natural era: por que não usar o mesmo padrão para detectar retomada de assunto?

A resposta está na natureza do espaço semântico de cada caso. Saudação e intenção de saída são espaços fechados e de baixa variabilidade — "oi", "bom dia", "tchau", "quero sair" giram em torno de um núcleo semântico compacto, cobrível com um punhado de exemplos representativos. Retomada de assunto é uma intenção aberta: "e aí, como ficou?", "vocês resolveram aquilo?", "ainda sobre meu problema", ou até implícita sem frase-gatilho nenhuma ("meu produto chegou quebrado, de novo"). Não existe um núcleo compacto para ancorar exemplos — qualquer conjunto fixo estaria sempre perseguindo casos que não previu.

Esse é um critério de decisão real, não específico deste caso: classificação por embedding/vectorstore serve bem a intenções fechadas e de baixa variabilidade; julgamento por LLM (raciocínio semântico geral) serve melhor a intenções abertas, onde o custo extra por chamada compra generalização que um conjunto fixo de exemplos não alcança.

A decisão adotada foi delegar essa detecção ao próprio LLM principal via **tool/function calling** — uma ferramenta `buscar_historico_anterior` que o modelo decide chamar ou não, com base no texto da pergunta. Esse caso de uso real (não um exemplo de tutorial genérico) foi usado como primeiro exercício prático de tool calling no roadmap de estudos.

#### Mecânica de tool calling: duas chamadas, não uma

O modelo nunca executa código — ele só decide e formata um pedido estruturado de chamada de função. Quem executa é o código Python. O ciclo completo, quando o modelo decide chamar a ferramenta, envolve duas chamadas de chat completion: uma primeira, onde o modelo recebe a pergunta e a lista de ferramentas disponíveis e decide se quer chamar alguma; e, se decidir, uma segunda, já com o resultado da função real disponível no histórico, para gerar a resposta final. Quando o modelo decide que não precisa da ferramenta, o custo permanece o de uma chamada só — o custo extra só existe quando a ferramenta é de fato acionada.

Testado isoladamente antes de integrar no `main.py` (`tests/debug_tool_calling.py`, com a função mockada retornando um histórico fixo): o modelo reconheceu corretamente uma pergunta de retomada de assunto sem palavras-chave óbvias, decidiu chamar a ferramenta com o `cliente_id` correto, e incorporou o resultado mockado de forma coerente na resposta final — validando o mecanismo antes de qualquer integração real com banco de dados.

### Reconciliando tool calling com streaming

`main.py` usa `.stream()` para gerar a resposta ao cliente, mas tool calling exige ver a resposta *completa* antes de decidir se ela contém um pedido de chamada de função — streaming complica isso, já que seria necessário remontar o pedido de função pedaço por pedaço.

A solução: a chamada de decisão (`llm_chat_com_tools.invoke(messages)`) usa `.invoke()`, não streaming — é rápida e só precisa checar `tool_calls`. A chamada de geração final continua usando `.stream()` normalmente, com um detalhe já validado antes: o texto é bufferizado por completo (`reply`) antes de ser exibido ao cliente, não impresso token a token. Essa não é uma escolha estética — é uma dependência real da arquitetura de verificação pós-geração do Max: as checagens de `TRANSFER_HUMANO` e `verificar_grounding()` rodam sobre o texto completo, e podem descartar o `reply` inteiro em favor de uma mensagem de transferência. Streaming token a token exibiria ao cliente uma resposta que o sistema ainda poderia decidir invalidar — quebrando exatamente essa proteção.

### Primeira integração de banco de dados no Max: `src/db/`

Como a integração de banco não existia no Max, o primeiro passo real foi trazer a camada de fundação — não só a função da ferramenta. Seguindo o princípio de que infraestrutura compartilhada (da qual outros módulos dependem) merece separação física dos módulos de feature (`busca_semantica.py`, `verificacao_llm.py`, que resolvem uma etapa específica do pipeline sem depender uns dos outros), foi criada uma pasta dedicada `src/db/`:

- `models.py` — as classes `Cliente`, `Conversa`, `Mensagem`, mapeando o schema já validado e existente no `max_db` (sem os `CHECK` constraints de string vazia/só espaços, que já vivem no banco — este arquivo mapeia estrutura, não a recria).
- `session.py` — `engine` e `obter_session()`, com a senha lida do `.env` via `Path(__file__).resolve().parents[2]` (dois níveis de profundidade a mais que `main.py`, que roda de `src/` e usa `load_dotenv()` sem path explícito).

Decisão explícita: `max_db` passa a ser o banco próprio e exclusivo do Max — só o código de produção deve se conectar nele, não mais os scripts do repositório de estudos, que serão removidos dali no futuro.

#### Bugs reais encontrados na integração

Três problemas surgiram ao rodar o primeiro teste de conexão (`tests/debug_conexao_db.py`), todos resolvidos por depuração incremental:

**Dependências ausentes.** `psycopg2-binary` e `sqlalchemy` nunca haviam sido instalados no `.venv` do Max — ambiente virtual completamente separado do repositório de estudos, mesmo rodando na mesma máquina.

**`UnicodeDecodeError` ao conectar — sintoma, não causa.** O erro inicial (`'utf-8' codec can't decode byte 0xe7'`) sugeria problema de encoding na senha. A investigação real (`print(repr(os.getenv("DB_PASSWORD")))`) revelou `None` — a variável nunca havia sido adicionada ao `.env` do Max (só existia no `.env` do repositório de estudos). A connection string montada usava a string literal `"None"` como senha; o PostgreSQL rejeitava a autenticação e retornava a mensagem de erro em um encoding que o `psycopg2` falhava ao decodificar. O erro de encoding era um efeito colateral de uma falha de autenticação anterior, não o problema em si.

**Import quebrando fora de `src/`.** Scripts em `tests/` seguem o padrão já estabelecido no projeto (`sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))` no topo do arquivo, antes dos demais imports) — permitindo rodar de qualquer diretório sem depender de `cd src` antes. O botão de "play" do VS Code, notavelmente, não replica esse comportamento por si só, tornando esse padrão necessário e não apenas conveniente.

### Estrutura de tools: pacote dedicado, pensando em escala

Assim como `db/`, a decisão foi por um pacote dedicado `src/tools/` — não um arquivo único `ferramentas.py` — pensando na sequência já mapeada do roadmap (MCP, agents), que provavelmente vai adicionar mais ferramentas: cada uma em seu próprio arquivo (`historico.py` para `buscar_historico_anterior`), com `__init__.py` centralizando e exportando a lista (`todas_as_tools`) consumida por `main.py`. Um padrão mais avançado — separar a definição da ferramenta (schema) da lógica de negócio que ela executa, comum em sistemas com muitas tools reutilizando lógica em contextos diferentes — foi considerado e adiado por complexidade desnecessária neste estágio, mesmo raciocínio já aplicado à decisão sobre o padrão repository em `db/`.

#### `buscar_historico_anterior`: critério de corte por tempo, não por volume

A primeira formulação cogitada — trazer só a última `Conversa` do cliente — falha num cenário real e concreto: um cliente que reclama diariamente do mesmo pedido gera múltiplas conversas separadas por sessão; filtrar por "última conversa" perderia o contexto das anteriores, mesmo sendo extremamente recentes e relevantes. Um critério alternativo — últimas N mensagens fixas — tem o problema inverso: uma única troca intensa dentro de uma janela de tempo curta poderia facilmente ultrapassar um número fixo pequeno.

A decisão final: filtrar por **janela de tempo** (últimos 30 dias), não por volume nem por sessão — com um teto de segurança de 150 mensagens existindo apenas como proteção contra volume patológico, não como critério principal. Isso resolve o cenário identificado (reclamações diárias dentro do mesmo mês continuam todas visíveis, independente de quantas sessões separadas geraram) sem introduzir a fragilidade do corte por sessão ou por contagem fixa.

O retorno da função é montado inteiramente em Python (loop e f-strings sobre os dados brutos do banco) — nenhuma chamada de LLM está envolvida nessa etapa. A síntese natural do histórico para o cliente acontece só na chamada de geração final que já existia; formatar o retorno de forma mais ou menos "pronta" muda apenas quanto trabalho de leitura o modelo faz dentro dessa chamada, não adiciona uma chamada extra em nenhum dos dois casos.

**Limitação conhecida, documentada e não resolvida nesta etapa:** o filtro por janela de tempo é, por natureza, um vetor de abuso de custo. Um cliente malicioso poderia gerar um volume alto de mensagens baratas (busca semântica de saudação/saída, não geração completa) para inflar deliberadamente o histórico trazido por `buscar_historico_anterior`, encarecendo chamadas futuras de geração sempre que a ferramenta fosse acionada. A mitigação real (rate limiting, limite de tamanho por mensagem) depende de uma camada de API/deploy ainda não implementada no roadmap — fica registrada como pendência a revisar quando essa camada existir, não resolvida agora.

### Identificação de cliente e persistência de mensagens

`identificar_cliente(session)` simula o "login do site" descrito acima: pede um e-mail, busca um `Cliente` existente por esse e-mail ou cria um novo. Uma `Conversa` nova é criada a cada execução do programa, associada a esse cliente. Essa lógica passou a viver dentro de um bloco `with obter_session() as session:` que agora envolve o `while True:` inteiro — mudança estrutural real em relação ao `main.py` anterior, necessário porque a sessão precisa permanecer aberta durante toda a conversa (não só por chamada isolada) para que a persistência de mensagens, abaixo, funcione dentro do mesmo contexto transacional.

A persistência de cada `Mensagem` (pergunta do cliente e resposta do Max) foi implementada com um timing deliberado, não simplesmente ao final do turno: a pergunta do cliente é salva e commitada assim que o contexto relevante é encontrado — **antes** de qualquer chamada ao LLM. Isso garante que a pergunta persista mesmo que a conversa seja interrompida logo depois por uma transferência para atendente humano (`TRANSFER_HUMANO` ou `grounding_falhou`). A resposta do Max, por sua vez, só é salva **depois** de passar pelas duas checagens pós-geração — porque, nesses casos de transferência, o Max nunca chega a gerar uma resposta "de verdade" para persistir.

Vale registrar explicitamente o escopo dessa cobertura: saudação, pedido de mais informação (`verificar_informacao_suficiente`) e intenção de saída usam `continue`/`break` *antes* de alcançar esse trecho de persistência — nenhum desses três caminhos gera uma linha em `Mensagem` hoje. A cobertura implementada resolve especificamente o caso de transferência após tentativa de resposta; os demais casos ficam como escopo não coberto nesta etapa, não como bug.

### Validação em produção real

Com a integração completa, o fluxo foi testado de ponta a ponta contra o `max_db` real (não mais scripts isolados de estudo):

- **Identificação de cliente:** confirmado via query direta no banco — um `Cliente` criado com nome e e-mail corretos, uma `Conversa` associada corretamente ao `cliente_id`. Login recorrente (mesmo e-mail, segunda execução) reconheceu o cliente existente sem recriar registro.
- **Tool calling — comportamento funcional, critério de acionamento revisado na Seção 22:** o critério linguístico originalmente hipotetizado aqui (referência sem antecedente resolvido na própria mensagem, ex. "teve alguma novidade **sobre isso**?", sinalizando retomada) foi contradito por testes posteriores documentados na Seção 22 — a pergunta "meu pedido atrasado no Sul, teve alguma novidade?", apesar de ter dado completo e nenhuma referência pendurada, acionou `buscar_historico_anterior` de forma consistente em todos os testes reproduzidos. O critério real que leva o modelo a chamar a ferramenta se mostrou mais amplo do que o inicialmente suposto (ver Seção 22 para a investigação completa, motivada originalmente por uma transferência inesperada causada por esse mesmo acionamento).
- **Persistência confirmada via query real:** em duas execuções de teste, a pergunta do cliente foi persistida em ambas (validando o timing "salvar antes do LLM"); a resposta do Max foi persistida apenas na execução que não terminou em transferência — exatamente o comportamento desenhado.

### Próximos passos

- Cobrir persistência de mensagens para os caminhos de saudação, pedido de mais informação e intenção de saída (hoje fora do escopo implementado).
- Mitigação do vetor de custo do filtro de 30 dias em `buscar_historico_anterior`, quando a camada de API/deploy existir.
- Migração real do FAISS para pgvector no Max (adiada desde a etapa de banco vetorial, para evitar retrabalho até este bloco de memória/tool calling estar concluído).
- Próxima fase do roadmap: structured output, MCP, agents.

## 22. Transferência inesperada pós-tool-calling: separando duas causas distintas e uma regra de negócio desatualizada

Durante um teste manual do fluxo de memória/tool calling recém-integrado, uma pergunta com dado completo ("meu pedido atrasado no Sul, teve alguma novidade?") gerou uma transferência inesperada para atendente humano — comportamento que não acontecia antes da introdução do tool calling, para o mesmo tipo de pergunta. A investigação levou por caminhos que pareciam confirmados e depois precisaram ser revisados, e terminou revelando dois fenômenos reais e independentes coexistindo no mesmo sintoma.

### Hipótese inicial: histórico vazio confundido com status do pedido

A primeira hipótese testada: quando `buscar_historico_anterior` não encontra nenhuma mensagem dentro da janela de 30 dias, ela retorna a string `"Nenhum histórico de conversa anterior encontrado nos últimos 30 dias."`. Para uma pergunta como "teve alguma novidade?", era plausível que o modelo reinterpretasse essa ausência de *histórico de conversa* como ausência de *novidade no status do pedido* — gerando uma afirmação sobre o pedido que nenhuma das duas fontes (contexto de política via RAG, ou o próprio resultado da tool) sustentava, e que `verificar_grounding()` corretamente rejeitaria.

Essa hipótese foi testada com dados reais e parcialmente descartada: repetições da mesma pergunta, já com histórico presente no banco (incluindo a resposta correta anterior), continuaram gerando transferência — o que não é compatível com uma causa que dependeria de histórico vazio.

### Lacuna de observabilidade descoberta no processo

Nenhuma das falhas observadas até esse ponto tinha o `reply` gerado pelo modelo capturado — o banco (tabela `Mensagem`) só persiste pergunta e resposta final quando a resposta passa nas duas checagens pós-geração; nas transferências, nada é persistido além da pergunta do cliente. Sem essa informação, não havia como saber, a posteriori, qual texto o modelo tinha gerado nem qual dos dois caminhos de transferência (`TRANSFER_HUMANO` no texto, ou `grounding_falhou=True`) havia sido acionado.

Um script de debug (`tests/debug_tool_calling_estabilidade.py`) foi criado para repetir a mesma pergunta N vezes de forma controlada, logando decisão de tool calling, presença do marcador `TRANSFER_HUMANO`, veredito de grounding e o `reply` completo a cada execução. Em duas baterias (10 execuções cada, 20 no total), o script nunca reproduziu a falha — resultado que, à primeira vista, pareceu confirmar apenas "instabilidade rara do modelo", mas que escondia uma diferença estrutural real: o script nunca persiste nada no banco, então nunca acumula histórico entre execuções — ao contrário da sessão manual real, onde repetir a pergunta entre execuções separadas do `main.py` faz o banco (e portanto o histórico devolvido pela tool) crescer a cada tentativa.

### Causa real #1: uma regra de negócio acionada por uma fonte de dado que não existia quando foi escrita

Adicionando um print do `reply` completo *antes* da checagem de `TRANSFER_HUMANO` (não só antes do grounding) e reproduzindo a falha com histórico já acumulado no banco (mesma pergunta e mesma resposta repetidas nele), o `reply` capturado revelou o marcador `###TRANSFER_HUMANO###` — um caminho de transferência distinto do `grounding_falhou`, com mensagem de saída diferente na interface (`"Aguarde, vou transferir para um atendente."` vs. `"Não tenho essa informação específica no momento..."`).

A causa raiz: o `system.txt` já continha, desde antes da existência de memória entre sessões, a instrução de emitir `###TRANSFER_HUMANO###` quando *"o cliente permanecer insatisfeito após duas tentativas de solução"*. Antes do tool calling, essa regra só podia ser avaliada dentro da sessão atual — não existia nenhuma fonte de dado capaz de mostrar ao modelo uma repetição *entre* sessões. Com `buscar_historico_anterior` trazendo histórico entre execuções separadas, a mesma instrução — sem nenhuma mudança de texto — passou a ser satisfeita também por repetições de teste manual entre reinicializações do programa, algo que a regra nunca tinha sido escrita para considerar.

> Atualização (Seção 23): um teste observacional posterior refinou a caracterização do critério de acionamento da tool em si. Não é ausência/presença de referência sintática pendurada (a hipótese original da Seção 21) — é se a pergunta pede uma **atualização de status ao longo do tempo** ("teve alguma novidade?") versus uma pergunta atemporal sobre política ("qual o prazo de entrega?"), mesmo as duas tendo dado geográfico igualmente completo. Ver Seção 23 para o teste e o resultado completo.

**Decisão de produto:** manter o comportamento como está, sem alteração de código ou de prompt. O raciocínio: a probabilidade de um cliente real repetir organicamente a mesma pergunta em sessões distintas dentro da janela de 30 dias é estimada como próxima de zero; e quando isso de fato acontece, o resultado (escalonamento para um atendente humano) é o comportamento desejado — não um efeito colateral a corrigir. Um cliente que já retornou múltiplas vezes com a mesma questão sem solução provavelmente está mais bem atendido por um humano do que por mais uma resposta automatizada.

### Causa real #2: uma única ocorrência de `grounding_falhou`, tratada como instabilidade já conhecida

Isolando a pergunta original — sem histórico algum, testada em três clientes genuinamente novos, com o print do `reply` já em vigor antes de qualquer checagem — o pipeline passou limpo em todas as três execuções (`grounding_falhou=False`). Combinado com o fato de que a única falha confirmada nesse caminho específico (sem histórico, veredito de grounding, não `TRANSFER_HUMANO`) foi a primeira observação que abriu esta investigação, e de que as demais falhas do meio (que não tiveram o `reply` capturado) são candidatas fortes a já pertencerem à causa #1, a contagem real de ocorrências não explicadas de `grounding_falhou` nesse cenário caiu para uma única instância isolada.

**Decisão:** tratar essa ocorrência como uma manifestação do baseline de instabilidade de grounding já documentado e aceito neste projeto (Seção 8 e seu adendo, taxa histórica observada em torno de 1/27 em geração com `temperature=0.3`), não como uma causa nova introduzida pelo tool calling. Uma única ocorrência em toda a bateria de testes não justifica investigação adicional — e replicar esse tipo de instabilidade sob demanda já era conhecido como estatisticamente difícil antes mesmo desta sessão.

### Lição sobre o processo de investigação em si

Vale registrar um padrão que se repetiu nesta sessão: uma hipótese "quase confirmada" (causa #1 explicando tudo) precisou ser corrigida a meio-caminho ao revisitar o primeiro dado coletado (a falha sem histórico, anterior à introdução do tool calling na conversa). Isso reforça um princípio já aplicado em investigações anteriores deste projeto (ex.: Seção 12, Seção 18): dados que não se encaixam na hipótese corrente não devem ser descartados silenciosamente — eles frequentemente indicam que mais de uma causa está presente no mesmo sintoma observado.

### Próximos passos

- Nenhuma ação de código pendente desta investigação — ambas as causas foram fechadas por decisão explícita (causa #1: comportamento mantido; causa #2: aceito como instabilidade conhecida).
- Cobrir persistência de mensagens para os caminhos ainda não salvos (saudação, `verificar_informacao_suficiente`, intenção de saída) — item já identificado na Seção 21, ainda pendente.
- Bateria de testes formal para validar a implementação de histórico como um todo (janela de 30 dias, teto de 150 mensagens, múltiplas conversas do mesmo cliente) antes de mergear a branch `experimento-memoria-tool-calling` na principal.

## 23. Validação pré-merge: refatoração para testabilidade, cobertura de persistência e o critério real de acionamento da tool

Antes de mergear a branch `experimento-memoria-tool-calling` na principal, restava validar formalmente o que as Seções 21 e 22 tinham confirmado apenas de forma manual e pontual: o comportamento de `buscar_historico_anterior`, a persistência de mensagens, e a ausência de regressão no restante do pipeline. Esta seção documenta essa validação e uma refatoração estrutural que ela motivou.

### Delimitando o que o eval set já cobre — e o que não cobre

Antes de rodar qualquer coisa, foi preciso confirmar uma suposição: será que o eval set (`test_eval_set.py`) já cobre, ainda que indiretamente, o fluxo de login e tool calling? A resposta, confirmada lendo o script: não. `gerar_resposta_max()` monta `messages` do zero a cada caso e chama `llm_chat.invoke(messages)` sem `bind_tools` — o LLM usado na avaliação nem tem acesso à ferramenta. O eval set roda o pipeline de RAG isolado (retrieval, `needs_more_information`, grounding), exatamente como a separação em módulos da Seção 14 desenhou. Ele continua sendo um teste de regressão válido para essa parte do sistema, mas tem cobertura zero de tool calling, persistência, ou `identificar_cliente()` — não por lacuna acidental, mas por desenho: são responsabilidades que vivem fora das funções que ele testa.

Isso definiu o escopo da validação: o eval set garante que nada do pipeline de RAG quebrou; tudo relacionado a memória entre sessões precisava de testes novos, dedicados.

### `tests/test_historico.py`: validação formal de `buscar_historico_anterior`

Cinco casos, testando a função diretamente via `.invoke({'cliente_id': ...})` (necessário porque o decorator `@tool` do LangChain transforma a função num objeto `BaseTool`, não uma função Python comum) contra um banco real, com timestamps inseridos manualmente no passado (não simulação de tempo real):

1. Cliente sem histórico algum → retorna a string de fallback esperada.
2. Janela de 30 dias → mensagem com 29 dias aparece, mensagem com 31 dias não aparece.
3. Teto de 150 mensagens → de 155 mensagens inseridas, exatamente as 150 mais recentes retornam; a mais antiga (índice 0) está ausente, a borda do corte (índice 5) está presente, a mais recente (índice 154) está presente. Testar a borda, não só os extremos, confirma que o corte acontece exatamente em 150, não "em algum lugar perto disso".
4. Múltiplas conversas do mesmo cliente → o filtro é por `Conversa.cliente_id` via `join`, não por uma `conversa_id` isolada; mensagens de uma conversa antiga aparecem numa consulta feita a partir de uma conversa nova.
5. Ordem cronológica → mensagens retornam na ordem em que aconteceram, não na ordem inversa da query (`.desc()` seguido de `.reverse()` no código).

**Resultado: 5/5 passou na primeira execução**, sem ajustes necessários.

### Refatoração de `main.py` para testabilidade: `processar_pergunta()`

Testar o comportamento de persistência exigiria simular o `while True` do `main()`, que depende de `input()` a cada iteração — não há como chamar isso programaticamente sem ou (a) reestruturar o código, ou (b) recorrer a `subprocess` injetando stdin, mais frágil e mais difícil de debugar.

Como o estágio atual do projeto prioriza aprendizado e testabilidade sobre preservar a experiência exata de terminal (Max ainda não está na fase de virar um produto apresentável de portfólio), a decisão foi extrair o corpo do loop para uma função isolada: `processar_pergunta(pergunta, estado)`, em um novo módulo `src/processamento.py` — seguindo o mesmo padrão de separação por domínio que a Seção 14 já estabeleceu para `busca_semantica.py` e `verificacao_llm.py`.

**Design da função:** recebe um dicionário `estado` mutável carregando `session`, `conversa`, `cliente`, `messages`, `tentativas_sem_contexto`, e os componentes de retrieval/LLM — em vez de dez parâmetros posicionais soltos, que precisariam viajar entre chamadas sucessivas do loop de qualquer forma. Em vez de `print()` diretamente, a função retorna um dicionário estruturado (`{'tipo': ..., 'reply': ..., 'transferiu': ..., 'motivo': ...}`); `main()` passa a apenas interpretar esse retorno e decidir o que exibir. Essa separação — decidir vs. exibir — é o que torna a função testável sem precisar capturar stdout.

**Trade-off aceito, não acidental:** extrair a lógica de geração para fora do loop principal quebrou o efeito de streaming token-a-token no terminal (`llm_chat.stream()` continua sendo usado internamente para montar o texto, mas o `reply` completo só é retornado — e portanto só impresso — depois de pronto, não pedaço a pedaço). Decisão consciente, dado o estágio atual do projeto: testabilidade pesou mais que preservar a UX de terminal.

**Regressão testada manualmente, ponta a ponta, antes de confiar na refatoração:** cliente novo passando pelos quatro caminhos principais (saudação, `needs_more_information`, resposta normal com contexto, `sem_contexto` → transferência na segunda tentativa) e cliente existente com histórico real (o cliente da investigação da Seção 22) confirmando que o caminho de tool calling sobreviveu à refatoração — sem `KeyError`, sem mudança de comportamento observável além da perda do streaming já esperada.

### `tests/test_persistencia.py`: validação formal do timing de persistência

Com `processar_pergunta()` chamável diretamente (sem `input()`), quatro casos testam o timing documentado na Seção 21 — pergunta salva antes da chamada ao LLM, resposta salva só depois de passar pelas duas checagens pós-geração:

1. Resposta normal → 2 mensagens persistidas (pergunta + resposta).
2. Transferência (`TRANSFER_HUMANO` ou `grounding_falhou` — ambos tratados como equivalentes para este teste) → 1 mensagem persistida (só a pergunta; a resposta nunca é salva quando o fluxo transfere).
3. `sem_contexto` → 0 mensagens persistidas (a pergunta nem chega a ser salva nesse caminho, por desenho).
4. Saudação → 0 mensagens persistidas.

**Resultado: 4/4 passou na primeira execução.** O Teste 2 passou pelo caminho `transfer_humano` especificamente nesta execução — o teste aceita qualquer um dos dois motivos de transferência como válido, já que ambos resultam no mesmo comportamento de persistência (só a pergunta salva).

### Regressão do eval set pós-refatoração

Rodar `test_eval_set.py` de novo após a refatoração de `main.py`/`processamento.py`, mesmo sem esses arquivos serem tocados diretamente pelo eval set: **28/29 passou**, com a única falha sendo a limitação já documentada e conhecida do caso "hello" (detecção de idioma ainda não implementada, ver Seção 20). Nenhuma regressão nova introduzida pela refatoração.

### `tests/debug_tool_calling_gatilho.py`: o critério real de acionamento da tool

A Seção 22 tinha caracterizado o acionamento de `buscar_historico_anterior` como "mais amplo do que o inicialmente suposto", sem determinar exatamente qual é o critério — só que a hipótese original (referência sintática pendurada, tipo "sobre isso") não bastava para explicar tudo.

Um script observacional (sem gabarito certo/errado — decisão do LLM não é uma regra de código para comparar contra) testou 4 categorias de pergunta, 2 perguntas por categoria, 5 repetições cada, contra o cliente com histórico real da Seção 22:

| Categoria | Resultado |
|---|---|
| Sinal explícito de retomada ("e aí, teve alguma novidade sobre aquilo?") | 5/5 e 5/5 — sempre aciona |
| Dado completo sem sinal de retomada ("...teve alguma novidade?" vs. "qual o prazo de entrega?") | 5/5 vs. 0/5 — resultado misto dentro da própria categoria |
| Pergunta nova sem relação com histórico ("qual o horário de atendimento?") | 0/5 e 0/5 — nunca aciona |
| Referência explícita a conversa anterior ("você lembra o que eu perguntei da última vez?") | 5/5 e 5/5 — sempre aciona |

Cada pergunta individual se comportou de forma **perfeitamente determinística** (0/5 ou 5/5, nunca uma taxa intermediária) — o que já é, em si, uma descoberta: o acionamento da tool não é instável ou aleatório, é consistente por pergunta.

O resultado misto da segunda categoria é o dado mais revelador: as duas perguntas têm dado geográfico igualmente completo (região Sul) e nenhuma das duas tem uma referência sintática pendurada tipo "sobre aquilo". A diferença real está na semântica do que está sendo pedido — "teve alguma novidade?" pede uma atualização de status ao longo do tempo, algo que só faz sentido responder considerando o que já foi dito antes; "qual o prazo de entrega?" é uma pergunta atemporal sobre política, sem noção de continuidade nenhuma.

**Conclusão revisada:** o critério real de acionamento não é sintático (presença de uma referência pendurada), é semântico — a tool é chamada quando a pergunta pede, implícita ou explicitamente, uma atualização sobre algo que já pode ter acontecido, independente de como a frase está estruturada. Isso é mais sofisticado do que a hipótese original da Seção 21 previa, e também mais coerente com o propósito documentado da tool (buscar contexto quando o cliente parece estar retomando um assunto) do que um "falso positivo" — a pergunta que abriu a investigação da Seção 22 ("teve alguma novidade?") estava, sob esse critério, corretamente classificada como retomada, mesmo tendo dado completo.

### Checklist final antes do merge

- [x] `test_eval_set.py` — regressão do pipeline de RAG, 28/29 (1 falha conhecida, sem relação com esta branch).
- [x] `tests/test_historico.py` — 5/5, comportamento de `buscar_historico_anterior` validado formalmente.
- [x] `tests/test_persistencia.py` — 4/4, timing de persistência validado formalmente.
- [x] `tests/debug_tool_calling_gatilho.py` — observação do critério real de acionamento, sem instabilidade detectada.
- [x] Regressão manual ponta a ponta pós-refatoração (`processamento.py`), dois perfis de cliente.

### Próximos passos

- Cobrir persistência de mensagens para os caminhos de saudação, `needs_more_information` e intenção de saída (segue pendente desde a Seção 21 — fora do escopo desta rodada de validação, que focou em confirmar o comportamento já implementado, não em expandir cobertura).
- Mitigação do vetor de custo do filtro de 30 dias em `buscar_historico_anterior`, quando a camada de API/deploy existir (pendente desde a Seção 21).
- Migração real do FAISS para pgvector no Max (adiada desde a etapa de banco vetorial).
- Merge de `experimento-memoria-tool-calling` na branch principal.
- Próxima fase do roadmap: structured output, MCP, agents.

## 24. Memória de sessão: resumo progressivo para conter o crescimento ilimitado de `messages`

Com a branch de memória entre sessões e tool calling já mergeada (Seção 23), restava um item pendente desde a Seção 16, anterior até à existência dessa branch: a lista `estado['messages']` acumula toda `HumanMessage`/`AIMessage` da conversa atual, sem nenhum mecanismo de limite, e é reenviada inteira ao modelo a cada nova pergunta — crescendo custo e latência à medida que uma conversa se alonga, com risco de eventualmente exceder a janela de contexto do modelo.

É importante não confundir esse problema com o que a Seção 21 resolveu: aquilo era memória **entre sessões** (histórico de conversas passadas, buscado sob demanda via `buscar_historico_anterior`); este é memória **dentro da sessão atual** (a lista de mensagens ativa numa única conversa em andamento). São dois problemas distintos, e só o primeiro estava resolvido antes desta seção.

### Escolha de abordagem

Duas técnicas mecanicamente diferentes resolvem esse tipo de problema:

- **Janela deslizante**: mantém só as últimas N mensagens, descartando o resto. Sem custo de chamada adicional (é só um corte de lista), mas perde informação mencionada fora da janela.
- **Resumo progressivo**: uma chamada extra ao LLM condensa o histórico antigo num resumo compacto, preservando continuidade semântica ao custo de uma chamada adicional por resumo gerado.

A decisão foi combinar as duas — o mesmo padrão que o LangChain historicamente chamava de `ConversationSummaryBufferMemory`: manter um pequeno número de mensagens recentes intactas, e resumir tudo que passa disso. Isso evita o custo de resumir cada mensagem individualmente (barato demais para compensar em conversas curtas) e evita a perda total de contexto da janela deslizante pura.

**Critério de disparo escolhido: contagem de mensagens, não contagem de tokens.** Mais simples de implementar (`len(conversa) > 10`), e considerado suficiente dado o padrão de uso esperado do Max — conversas de suporte tipicamente curtas, resolvidas ou escaladas em poucas trocas. Decisão explicitamente marcada para reavaliação futura, quando houver dados reais de clientes que permitam calibrar por token em vez de por contagem bruta (o projeto já tem `tiktoken` disponível, se for necessário).

**Parâmetros escolhidos:** resumir a cada 10 mensagens (`HumanMessage`/`AIMessage`) acumuladas, mantendo as últimas 4 intactas — o suficiente para cobrir referências imediatas tipo "aquilo que eu acabei de falar", sem deixar a janela bruta grande o bastante para anular o ganho de custo do resumo.

### Implementação: `src/memoria_sessao.py`

Novo módulo, contendo `resumir_se_necessario(estado, llm_resumo)` — deliberadamente fora de `src/tools/`, já que não é uma ferramenta exposta ao LLM via `bind_tools` (o modelo não decide chamá-la); é lógica determinística baseada em regra, chamada diretamente por `processamento.py`, no mesmo padrão de `buscar_contexto()` ou `verificar_grounding()`.

A função:
1. Filtra `estado['messages']` para contar só `HumanMessage`/`AIMessage` (ignorando `SystemMessage` e `ToolMessage` na contagem do limiar).
2. Se o total ainda não passou de 10, não faz nada.
3. Caso contrário, calcula o ponto de corte andando de trás para frente na lista, contando até acumular 4 mensagens contáveis.
4. **Proteção contra quebra de formato de API:** se o ponto de corte calculado cair exatamente sobre um `ToolMessage`, o corte é empurrado para trás (mantendo mais que 4 mensagens recentes, nunca menos) até não cair mais ali — porque a API da OpenAI exige que todo `ToolMessage` venha imediatamente após a `AIMessage(tool_calls)` correspondente na lista enviada; cortar no meio desse par quebraria a chamada.
5. O trecho mais antigo é formatado como texto simples e enviado a uma chamada de resumo, reaproveitando `estado['llm_verificador']` (já roda em `temperature=0`, adequado para resumo determinístico, sem necessidade de um componente novo).
6. O resumo retorna embrulhado num novo `SystemMessage` (não `HumanMessage` nem `AIMessage`, já que nenhuma das partes de fato "disse" aquele resumo — é contexto interpretativo, o mesmo papel do `system_prompt` original).
7. `estado['messages']` é reconstruída como: `[SystemMessage original, novo SystemMessage de resumo] + mensagens recentes intactas`.

A chamada foi integrada em `processamento.py` logo após a nova `HumanMessage` do cliente ser adicionada à lista, antes da decisão de tool calling — garantindo que, se o resumo disparar, o modelo já decide sobre chamar `buscar_historico_anterior` enxergando a lista já reduzida.

### Validação: `tests/debug_memoria_sessao.py`

Como a qualidade de um resumo gerado por LLM não tem gabarito certo/errado automatizável — precisa de leitura humana para julgar se a informação relevante foi preservada —, o script é observacional (sem `PASSOU`/`FALHOU`), montando conversas falsas inteiramente em memória (sem precisar digitar 11+ mensagens manualmente):

- **Teste 1** (13 mensagens, sem tool call): confirmou que exatamente as últimas 4 mensagens ficaram intactas, e que o resumo gerado preservou os fatos relevantes (região, natureza do atraso, prazo de 48h, processo de investigação, número do pedido) de forma fiel e sem alucinação.
- **Teste 2** (17 mensagens, com um par `AIMessage(tool_calls)` → `ToolMessage` posicionado bem no limite do corte): confirmou que a proteção contra quebra de formato funcionou como desenhado — o par ficou inteiro do lado recente, resultando em 5 mensagens recentes mantidas em vez de 4, exatamente o comportamento esperado quando o corte "puro" cairia no meio do par.

Ambos os testes passaram na primeira execução, sem necessidade de ajuste no código.

## 25. Structured output: implementação, investigação de instabilidade e reversão

Retomando o item pendente desde a Seção 16: personalizar a resposta de atraso com um cálculo real de prazo (`"falta 1 dia até o pedido entrar em atraso"`), em vez da resposta genérica que a Seção 4 já tinha diagnosticado. Esta seção documenta a implementação completa, a investigação de uma instabilidade séria encontrada em produção simulada, e a decisão final de reverter tudo — um resultado negativo, mas com valor de estudo equivalente a uma seção de sucesso: o processo de isolamento de variáveis aplicado aqui é reaproveitável em qualquer investigação futura de comportamento não determinístico do Max.

### Desenho da solução

O objetivo já vinha claro da Seção 16: extrair dados do pedido de forma estruturada (não texto livre) para permitir um cálculo de data determinístico em Python, em vez de pedir ao LLM que "calcule" — modelos erram aritmética de data com frequência, e a divisão de trabalho correta é o LLM interpretar e o código calcular.

**Fonte dos dados do pedido:** decisão errada revertida no meio da investigação. A primeira hipótese foi extrair região e data de compra da própria conversa (o cliente informando em texto livre), mas isso confundia dois conceitos: *data de compra* (o cliente sabe) e *data de confirmação de postagem* (de onde a contagem de prazo realmente começa, por regra do `politicas_xyz.pdf`, Capítulo 1). Corrigido para: uma tabela nova `pedidos` no PostgreSQL (`cliente_id`, `regiao`, `data_compra`, `data_postagem` nullable, `entregue_em` nullable), populada via seed manual — simulando o papel que um sistema de logística real ocupa, já que o Max não tem integração real com nenhum.

**Módulos novos:**
- `src/pedidos.py` — `obter_pedido_em_andamento(cliente_id)`: consulta determinística (não exposta via `bind_tools`, mesmo padrão de `buscar_contexto`/`verificar_grounding`), devolvendo o pedido mais recente ainda sem `entregue_em` (decisão de escopo: um cliente com múltiplos pedidos em andamento simultâneos não é tratado — fica documentado como limitação aceita, não como bug).
- `src/prazo.py` — schema Pydantic `PrazoPolitica` (`pergunta_sobre_prazo: bool`, `prazo_min_dias_uteis`/`prazo_max_dias_uteis: Optional[int]`), função `extrair_prazo()` usando `llm.with_structured_output()`, e as funções determinísticas `dias_uteis_decorridos()` (via `numpy.busday_count`, contando a partir do dia seguinte à postagem, conforme a regra do PDF) e `situacao_prazo()` (compara decorridos vs. prazo máximo, decisão de produto de Igor: calcular pela ponta maior do intervalo, para evitar declarar atraso prematuro).

**Integração em `processamento.py`:** a consulta ao pedido subiu para antes de `verificar_informacao_suficiente`, para que um cliente com pedido em andamento pare de ser questionado sobre região/prazo que o sistema já pode consultar sozinho (bug encontrado e corrigido durante os testes manuais — o Max chegou a perguntar região a um cliente cujo pedido já estava no banco). Uma segunda chamada de busca ao retriever (`"prazo de entrega da região {regiao}"`) foi adicionada especificamente para alimentar a extração do prazo, porque a busca pela pergunta original do cliente ("meu pedido está atrasado") não recupera a tabela de prazos do Capítulo 1 — a causa de um bug real descrito abaixo.

### Bugs encontrados e corrigidos durante a implementação

1. **Prazo inventado por falta de dado no contexto.** Na primeira integração, `extrair_prazo` recebia o mesmo `contexto` que o Max usa para responder (recuperado pela pergunta do cliente), que não contém a tabela de prazos do Capítulo 1. O modelo, mesmo assim, preencheu `prazo_max_dias_uteis` com um valor (7, o valor correto da região Nordeste) para uma região Sul, sem nenhuma base textual. Corrigido com a segunda busca dedicada, mais uma trava de código (`if pedido['regiao'].lower() not in contexto_prazo.lower(): return None`) e uma instrução explícita no prompt ("use somente valores que estejam escritos... se não estiver, deixe vazio") — a lição maior, reforçada depois por um segundo bug (abaixo), é que **a `description` de um campo Pydantic é uma sugestão ao modelo, não uma garantia**: a mesma ambiguidade textual (nenhuma condição de "só preencha se X" no campo do prazo) fez o modelo preencher os campos de prazo mesmo quando `pergunta_sobre_prazo=False`, corrigido adicionando a condição explícita na própria `description`.
2. **Resposta ambígua sobre quanto prazo resta.** O texto gerado pelo `info_prazo` trazia os números certos, mas o Max às vezes citava só o prazo total ("aguarde até 4 dias úteis") em vez do que realmente importa ("restam 2 dias"), soando como se o cliente tivesse que esperar o prazo inteiro de novo. Corrigido acrescentando uma instrução de conteúdo (não de redação) ao fim de cada frase gerada em `montar_info_prazo` — "informe ao cliente quantos dias úteis ainda restam" — validado como estável em execuções subsequentes.
3. **Regressão de tom: resposta "robótica" no caso de pedido não postado.** A primeira correção para esse caso usou uma instrução de redação literal ("logo no início da resposta, diga que..."), e o Max passou a repetir a mesma frase quase palavra por palavra em várias execuções, perdendo a variação natural de redação que aparecia nos outros casos (prazo dentro, atraso confirmado). Causa identificada por comparação direta das respostas nos dois cenários: instrução ditando *o quê* dizer dava liberdade de redação; instrução ditando *como* dizer (ordem das palavras, frase pronta) fazia o modelo copiar. Corrigido reescrevendo a instrução como fato + objetivo ("responda com naturalidade, no seu tom habitual, mas garanta que o cliente entenda que...") em vez de comando de fala.
4. **Falta de tom de acolhimento em atraso confirmado.** Levantado por julgamento de produto (experiência prévia de Igor em SAC, não por teste automatizado): a resposta ao caso de atraso confirmado aplicava a política (aguardar 48h) sem nenhum reconhecimento da frustração do cliente antes da instrução. Resolvido com uma seção nova no `system.txt` ("Tom em situações de insatisfação", escopada a atraso confirmado e extravio) e um exemplo few-shot novo, com a mesma ressalva estrutural do restante do prompt: o reconhecimento não pode virar promessa de ação que o Max não executa (a seção "Sobre ações" já existente foi referenciada explicitamente, não duplicada).

### A instabilidade: investigação de causa raiz

Depois dos ajustes acima, testes manuais repetidos do cenário "cliente com pedido não postado, pergunta sobre atraso" mostraram uma taxa de erro alta e incômoda: o Max, em vez de responder que o prazo ainda não começou a contar, às vezes aplicava a política de atraso confirmado (citando as 48h), apesar do `info_prazo` estar correto. O verificador de grounding interceptava esses casos (a política de 48h não está fundamentada quando o próprio sistema calculou que o pedido não está atrasado), então nenhum cliente real receberia a informação errada — mas o custo era uma transferência desnecessária ao atendimento humano, numa taxa bem maior que a instabilidade de linha de base já documentada (Seção 22, ~4% a `temperature=0.3`).

**Método de isolamento:** em vez de testar manualmente pelo `main.py` (lento, sujeito a erro de transcrição ao copiar do terminal — dois falsos alarmes de "espaço perdido" foram rastreados a artefatos de quebra de linha do terminal, não a bugs reais), cada variável suspeita foi isolada em um script dedicado, chamando as funções reais do projeto diretamente, com contagem automática de acerto/erro em lote (10-40 repetições por teste).

| Hipótese testada | Método | Resultado |
|---|---|---|
| Redação da pergunta do cliente | mesma pergunta vs. reformulada, mesmo contexto | sem diferença — ambas corretas em todas as repetições |
| Determinismo da busca RAG | hash MD5 do `contexto` em 10 chamadas reais | **idêntico** nas 10 — busca é 100% determinística |
| Determinismo da consulta ao banco | leitura do pedido em sessão reaproveitada vs. sessão nova, 40 chamadas | **idêntico** nas 40 |
| Chamada de `buscar_historico_anterior` (tool) | inspeção de `tool_calls` em todas as execuções reais | sempre `[]` — a tool nunca foi chamada neste cenário |
| Persistência da `Mensagem` do cliente antes da geração | teste A/B alternando com/sem escrita, mesma execução | **sem diferença** entre os grupos — taxa de erro igual nos dois |
| Índice FAISS em disco antigo/corrompido | índice reconstruído do zero a partir do PDF, reteste completo | **sem melhora** — taxa de erro manteve-se (até levemente pior) |
| Pausa entre turnos (possível rate-limit entre repetições) | 15s de espera entre cada chamada ao pipeline completo | reduziu de ~40-55% para ~30%, mas **não eliminou** |

Cada hipótese testada foi eliminada com prova (não com leitura de código apenas), exceto a última, que apontou para uma direção sem resolvê-la de vez. O padrão mais consistente encontrado: scripts com **uma única chamada real** de LLM por repetição mantiveram-se sempre entre 0-5% de erro (a mesma taxa da Seção 22); qualquer script reproduzindo o volume real de chamadas de um turno completo (busca de contexto, extração estruturada, decisão de tools, geração, verificação de grounding — até 6 chamadas reais em sequência rápida) subiu para 30-55%.

**Conclusão de causa, com o nível de confiança disponível:** a taxa de instabilidade de `temperature=0.3` já documentada (Seção 22) parece ser amplificada por volume de chamadas reais de API empilhadas dentro do mesmo turno — compatível com degradação sob carga do lado do provedor (rate-limiting ou efeito correlato, que não necessariamente levanta uma exceção explícita no código, apenas piora a qualidade da geração). **Isso não foi provado com certeza absoluta** — não há visibilidade sobre o mecanismo interno da API da OpenAI a partir do lado do cliente —, mas é a explicação mais sustentada pelo padrão de evidências coletado, depois de eliminar toda causa determinística testável no próprio código do Max.

### Decisão: reversão completa

Avaliado o trade-off: o ganho (uma resposta personalizada de prazo, para um cenário específico e de risco baixo — pedido não postado, cliente perguntando sobre atraso) era pequeno frente ao custo (uma taxa de transferência desnecessária de 30-55% nesse mesmo cenário, bem acima da linha de base aceitável do restante do sistema). Decisão de Igor: reverter toda a implementação desta seção, em vez de aceitar a complexidade nova com esse nível de confiabilidade, ou investir mais tempo perseguindo uma causa que pode nem ser corrigível do lado do projeto.

**Revertido (via `git checkout`, nada havia sido commitado):** `src/processamento.py`, `src/db/models.py` (classe `Pedido` removida), `prompts/system.txt` (seção de tom e exemplo few-shot removidos), `requirements.txt` (dependência `numpy` removida). **Apagados (arquivos nunca rastreados pelo Git):** `src/pedidos.py`, `src/prazo.py`, e todos os scripts de investigação em `tests/` criados durante este experimento.

**Deixado como está, por decisão consciente:** a tabela `pedidos` no PostgreSQL (com os 3 registros de seed) não foi removida do banco — avaliado como resíduo inofensivo (nenhum código a referencia mais, sem custo de performance nas demais tabelas), mas registrado aqui para não virar um mistério numa retomada futura do projeto. Removível a qualquer momento com `DROP TABLE pedidos;`, se a decisão for não retomar esta feature.

### Lições

- **Metodológica:** copiar saída de terminal para comparação manual é uma fonte real de falso positivo — dois "bugs" investigados nesta seção ("prazomáximo", "opedido" sem espaço) eram artefatos de quebra de linha do terminal na cópia, não erros no código gerador do texto. Comparação por hash (MD5) resolve isso sem ambiguidade e deveria ser o primeiro recurso ao suspeitar de não-determinismo de texto, antes de qualquer leitura visual.
- **Técnica:** a `description` de um campo Pydantic em `with_structured_output` é uma sugestão ao modelo, não uma garantia de comportamento — confirmado duas vezes nesta seção (prazo preenchido sem base textual; prazo preenchido mesmo com `pergunta_sobre_prazo=False`). A garantia real precisa vir de uma trava no código que consome o resultado, nunca só do texto da instrução.
- **De produto:** uma instrução textual embutida no contexto (ex.: "informe quantos dias restam") é mais frágil que uma regra no `system.txt`, mas também mais barata de testar e iterar — boa para ajuste fino de conteúdo; instruções de *redação exata* (em vez de *conteúdo obrigatório*) tendem a produzir cópia literal do exemplo, que é o efeito indesejado descrito no bug 3.
- **De engenharia/decisão:** implementar, medir com rigor, e reverter diante de um trade-off desfavorável é um resultado tão válido quanto manter uma feature que funciona — e mais defensável em entrevista do que insistir em uma implementação cujo custo de confiabilidade não foi justificado pelo ganho. A investigação de causa raiz, mesmo sem chegar a uma causa 100% determinística, teve valor: eliminou sistematicamente todas as explicações determinísticas plausíveis no próprio código, isolando a suspeita razoável em um fator externo (carga sobre a API), que está fora do que o projeto pode controlar.

### Se este item for retomado no futuro

A base de código descrita aqui (schema, funções de cálculo, integração) é reaproveitável, mas a reversão só deve ser desfeita se a instabilidade for resolvida ou aceita explicitamente antes — por exemplo, adicionando uma segunda tentativa de geração antes de transferir (em vez de transferir na primeira falha de grounding), ou reduzindo o número de chamadas reais de LLM por turno nesse caminho específico. Retomar sem alguma dessas mitigações reproduziria o mesmo problema.

### Próximos passos

- Reavaliar o critério de disparo (contagem de mensagens vs. contagem de tokens) quando houver volume real de uso para calibrar.
- Cobrir persistência de mensagens para os caminhos de saudação, `needs_more_information` e intenção de saída (pendente desde a Seção 21, ainda não relacionado a este item).
- Migração real do FAISS para pgvector no Max (adiada desde a etapa de banco vetorial).
- Próxima fase do roadmap: MCP, agents. (`structured output` foi tentado e revertido — Seção 25; retomar a ideia de base, se fizer sentido no futuro, exige primeiro resolver a instabilidade documentada lá.)

