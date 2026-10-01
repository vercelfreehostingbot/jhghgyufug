from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

@tool
def buscar_historico_anterior(cliente_id: int) -> str:
    """Busca o histórico de conversas anteriores deste cliente, quando ele parece estar retomando um assunto já discutido antes (ex: 'e como ficou meu problema?', 'vocês resolveram aquilo?')."""
    return "Conversa anterior (15/08): cliente relatou extravio do pedido #4521 (região Sul). Foi orientado a aguardar 30 dias para reembolso automático."

llm_chat = ChatOpenAI(model='gpt-4o-mini', temperature=0)
llm_com_tools = llm_chat.bind_tools([buscar_historico_anterior])

messages = [
    SystemMessage(content="Você é o Max, assistente da XYZ Entregas. O cliente atual tem cliente_id=7."),
    HumanMessage(content="e aí, como ficou aquele problema que eu relatei?")
]

resposta = llm_com_tools.invoke(messages)

if resposta.tool_calls:
    print(f"Modelo decidiu chamar: {resposta.tool_calls}")

    messages.append(resposta)

    for chamada in resposta.tool_calls:
        resultado = buscar_historico_anterior.invoke(chamada['args'])
        messages.append(ToolMessage(content=resultado, tool_call_id=chamada['id']))

    resposta_final = llm_com_tools.invoke(messages)
    print(f"Max: {resposta_final.content}")
else:
    print(f"Max: {resposta.content}")