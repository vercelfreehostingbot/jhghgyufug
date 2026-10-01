def verificar_informacao_suficiente(llm, pergunta):
    prompt = f"""Analise a pergunta do cliente abaixo.

Pergunta: {pergunta}

Considere SIM apenas se a pergunta descreve uma situação cuja resposta correta MUDA dependendo de um dado do pedido que não foi informado — e esse dado é necessário especificamente para o assunto perguntado, não em geral.

A empresa atende pedidos nacionais e, em caráter piloto, pedidos internacionais para dois países: Portugal e Estados Unidos. O dado que falta depende do assunto da pergunta:

- PRAZO DE ENTREGA: se o pedido é internacional, o prazo muda entre Portugal (10 a 15 dias úteis) e Estados Unidos (8 a 12 dias úteis) — nesse caso, saber apenas "internacional" NÃO é suficiente, é necessário saber qual dos dois países. Porém, se o cliente já menciona uma região doméstica (Sul, Sudeste, Nordeste, Norte, Centro-Oeste), isso já resolve a questão: assuma que o pedido é nacional, e não é necessário perguntar se é nacional ou internacional.

Considere NÃO se:
- A pergunta já contém o dado necessário para o assunto específico perguntado (ex: região doméstica já basta para prazo nacional; "internacional" já basta para extravio; o país específico já foi informado, quando o assunto exige).
- A pergunta é sobre um assunto que não depende de região, país de destino ou prazo.
- A pergunta é sobre algo que as políticas da empresa não cobrem (incluindo destinos internacionais fora de Portugal e Estados Unidos) — isso é falta de política, não falta de dado, e não deve ser tratado por este verificador.
- A pergunta é genérica sobre uma política, sem descrever uma situação específica do pedido do cliente.

Exemplos:
"meu pedido está atrasado há 3 dias, quero reembolso" → SIM (não sabemos a região, nem o prazo dela)
"qual o prazo pro sul" → NÃO (região doméstica já indica pedido nacional, suficiente para prazo)
"qual o prazo do meu pedido internacional" → SIM (para prazo, "internacional" sozinho não basta — precisa saber Portugal ou Estados Unidos)
"qual o prazo do meu pedido para os Estados Unidos" → NÃO (país já definido, suficiente para prazo)
"meu pedido foi extraviado" → SIM (não sabemos se é nacional ou internacional, e a regra muda entre os dois)
"meu pedido internacional foi extraviado" → NÃO (para extravio, "internacional" já basta — a regra é igual para Portugal e Estados Unidos)
"meu pedido para a França foi extraviado" → NÃO (não é falta de dado — a empresa não cobre esse destino, é uma limitação de escopo, não deste verificador)
"qual o horário de atendimento" → NÃO (não depende de nenhum dado do pedido)
"posso trocar meu pedido por outro produto de valor maior?" → NÃO (depende de existir ou não uma política de troca, não de região, país ou prazo)

Primeiro, explique em uma frase: qual é o assunto da pergunta, e esse assunto especificamente exige um dado que não foi informado?

Depois, na última linha, responda apenas com SIM ou NÃO.

Formato obrigatório:
Raciocínio: <sua explicação>
Veredito: SIM ou NÃO"""

    verificacao = llm.invoke(prompt)

    linhas = verificacao.content.strip().split('\n')
    veredito = linhas[-1] if linhas else verificacao.content
    return 'SIM' in veredito.upper()

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

Primeiro, explique em uma frase seu raciocínio: identifique qual trecho do contexto sustenta (ou não sustenta) cada afirmação da resposta.

Depois, na última linha, responda apenas com SIM (se houver algo não sustentado) ou NÃO (se tudo estiver sustentado).

Formato obrigatório:
Raciocínio: <sua explicação>
Veredito: SIM ou NÃO"""

    verificacao = llm.invoke(prompt_verificacao)

    linhas = verificacao.content.strip().split('\n')
    veredito = linhas[-1] if linhas else verificacao.content
    return 'SIM' in veredito.upper()