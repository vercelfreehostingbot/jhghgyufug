from datetime import datetime, timedelta
from langchain_core.tools import tool

from db.models import Mensagem, Conversa
from db.session import obter_session

@tool
def buscar_historico_anterior(cliente_id: int) -> str:
    """Busca o histórico de conversas anteriores deste cliente, quando ele parece estar retomando um assunto já discutido antes (ex: 'e como ficou meu problema?', 'vocês resolveram aquilo?')."""

    janela_data = datetime.now() - timedelta(days=30)

    with obter_session() as session:
        mensagens_recentes = (
            session.query(Mensagem)
            .join(Conversa)
            .filter(Conversa.cliente_id == cliente_id)
            .filter(Mensagem.enviada_em >= janela_data)
            .order_by(Mensagem.enviada_em.desc())
            .limit(150)
            .all()
        )
        mensagens_recentes.reverse()

        if not mensagens_recentes:
            return "Nenhum histórico de conversa anterior encontrado nos últimos 30 dias."

        texto = ""
        for msg in mensagens_recentes:
            texto += f"[{msg.enviada_em.strftime('%d/%m %H:%M')}] {msg.remetente}: {msg.conteudo}\n"

        return texto