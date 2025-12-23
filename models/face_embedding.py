from sqlalchemy import Column, Integer
from sqlalchemy.orm import declarative_base
from pgvector.sqlalchemy import Vector

Base = declarative_base()

class FaceEmbedding(Base):
    __tablename__ = "face_embeddings"

    user_id = Column(Integer, primary_key=True)
    embedding = Column(Vector(512), nullable=False)