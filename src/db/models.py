from sqlalchemy.orm import declarative_base
from sqlalchemy import Column, Integer, String, Text, TIMESTAMP, ForeignKey
from sqlalchemy.sql import func

Base = declarative_base()

class Cliente(Base):
    __tablename__ = "clientes"

    id = Column(Integer, primary_key=True)
    nome = Column(String(150), nullable=False)
    email = Column(Text, unique=True)
    criado_em = Column(TIMESTAMP(timezone=True), nullable=False, server_default=func.now())

class Conversa(Base):
    __tablename__ = "conversas"

    id = Column(Integer, primary_key=True)
    cliente_id = Column(Integer, ForeignKey("clientes.id"), nullable=False)
    iniciada_em = Column(TIMESTAMP(timezone=True), nullable=False, server_default=func.now())

class Mensagem(Base):
    __tablename__ = "mensagens"

    id = Column(Integer, primary_key=True)
    conversa_id = Column(Integer, ForeignKey("conversas.id"), nullable=False)
    remetente = Column(String(10), nullable=False)
    conteudo = Column(Text, nullable=False)
    enviada_em = Column(TIMESTAMP(timezone=True), nullable=False, server_default=func.now())