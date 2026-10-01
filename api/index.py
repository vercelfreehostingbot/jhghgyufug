import json
import os
import sys
from http.server import BaseHTTPRequestHandler

SRC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)


_resources = None


def _load_resources():
    global _resources
    if _resources is None:
        from dotenv import load_dotenv
        from inicializacao import (
            carregar_base_conhecimento,
            carregar_indice_saida,
            carregar_indice_saudacoes,
            carregar_prompt,
        )
        from langchain_core.messages import SystemMessage
        from langchain_openai import ChatOpenAI
        from tools import todas_as_tools

        load_dotenv()
        llm_chat = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
        _resources = {
            "messages": [SystemMessage(content=carregar_prompt())],
            "retriever": carregar_base_conhecimento(),
            "vectorstore_saudacoes": carregar_indice_saudacoes(),
            "vectorstore_saida": carregar_indice_saida(),
            "llm_chat": llm_chat,
            "llm_chat_com_tools": llm_chat.bind_tools(todas_as_tools),
            "llm_verificador": ChatOpenAI(model="gpt-4o-mini", temperature=0),
        }
    return _resources


def _json_response(handler, status, payload):
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        _json_response(self, 200, {"status": "ok"})

    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", "0"))
        if content_length > 32_000:
            _json_response(self, 413, {"error": "Mensagem muito grande."})
            return

        try:
            payload = json.loads(self.rfile.read(content_length) or b"{}")
            pergunta = str(payload.get("message", "")).strip()
            email = str(payload.get("email", "")).strip()
            nome = str(payload.get("name", "Cliente")).strip() or "Cliente"
            if not pergunta:
                _json_response(self, 400, {"error": "O campo 'message' é obrigatório."})
                return

            from processamento import processar_pergunta
            from db.models import Cliente, Conversa
            from db.session import obter_session

            resources = _load_resources()
            estado = dict(resources)
            estado["messages"] = list(resources["messages"])
            estado["tentativas_sem_contexto"] = 0
            with obter_session() as session:
                cliente = session.query(Cliente).filter_by(email=email).first() if email else None
                if cliente is None:
                    cliente = Cliente(nome=nome, email=email or None)
                    session.add(cliente)
                    session.commit()

                conversa = Conversa(cliente_id=cliente.id)
                session.add(conversa)
                session.commit()

                estado.update({
                    "session": session,
                    "conversa": conversa,
                    "cliente": cliente,
                })
                resultado = processar_pergunta(pergunta, estado)

            resultado.pop("tool_calls", None)
            _json_response(self, 200, resultado)
        except Exception:
            _json_response(self, 500, {"error": "Erro interno ao processar a mensagem."})

    def log_message(self, format, *args):
        return
