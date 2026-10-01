# Debug: confirma conexão com o banco de dados (max_db) via SQLAlchemy
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from db.models import Cliente, Conversa, Mensagem
from db.session import obter_session

with obter_session() as session:
    clientes = session.query(Cliente).all()
    conversas = session.query(Conversa).all()
    mensagens = session.query(Mensagem).all()

    print(f"Clientes encontrados: {len(clientes)}")
    print(f"Conversas encontradas: {len(conversas)}")
    print(f"Mensagens encontradas: {len(mensagens)}")