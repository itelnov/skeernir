import uuid

import bcrypt
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from sqlalchemy import (
    Column,
    Integer,
    String,
    ForeignKey,
    DateTime,
    JSON, 
    Text)


Base = declarative_base()

# Database models

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String)
    hashed_password = Column(String)
    conversations = relationship("Conversation", back_populates="user")

    @staticmethod
    def hash_password(password: str) -> str:
        return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

    @staticmethod
    def verify_password(plain_password: str, hashed_password: str) -> bool:
        return bcrypt.checkpw(plain_password.encode(), hashed_password.encode())


class Conversation(Base):
    __tablename__ = "conversations"
    id = Column(Integer, primary_key=True, index=True)
    session_uuid = Column(String, unique=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    title = Column(String)
    interrupted = Column(String, default=False)
    interrupted_value = Column(String, default="")
    state = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=func.now())
    user = relationship("User", back_populates="conversations")
    messages = relationship("Message", back_populates="conversation")
    attachments = relationship("Attachment", back_populates="conversation")


class Message(Base):
    __tablename__ = "messages"
    id = Column(Integer, primary_key=True, index=True)
    message_uuid = Column(String, unique=True, index=True)
    session_uuid = Column(String, ForeignKey("conversations.session_uuid"))
    content = Column(String)
    is_bot = Column(String, default=False)
    timestamp = Column(DateTime, default=func.now())
    conversation = relationship("Conversation", back_populates="messages")
    # Relationship to attachments - one-to-many
    attachments = relationship(
        "Attachment", 
        back_populates="message", 
        cascade="all, delete-orphan"
    )


class Attachment(Base):
    """
    Separate model to store message attachments
    Allows for flexible attachment storage with metadata
    TODO
    Potential Strategies:
        For smaller files (<1MB), store directly in the database
        For larger files, use local file storage and save the path
        Implement file size limits and validation in your API layer
    """
    __tablename__ = "message_attachments"
    
    id = Column(Integer, primary_key=True, index=True)
    attachment_uuid = Column(
        String, default=lambda: str(uuid.uuid4()), unique=True, index=True)
    message_uuid = Column(String, ForeignKey("messages.message_uuid"), nullable=False)
    session_uuid = Column(
        String, ForeignKey("conversations.session_uuid"), nullable=False)
    filename = Column(String, nullable=False)
    # MIME type of the file (e.g., 'image/jpeg', 'application/pdf')
    mime_type = Column(String, nullable=False)
    # File size in bytes
    file_size = Column(Integer, nullable=False)
    file_content = Column(Text, nullable=True)
    
    # Additional metadata as JSON to allow for flexible storage
    attachemnt_metadata = Column(JSON, nullable=True)
    # Timestamp of attachment creation
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    # Relationship back to the message
    message = relationship("Message", back_populates="attachments")
    # Relationship back to the conversation
    conversation = relationship("Conversation", back_populates="attachments")


class GraphLog(Base):
    
    __tablename__ = "graph_outputs"
    
    id = Column(Integer, primary_key=True, index=True)
    graphlog_uuid = Column(
        String, default=lambda: str(uuid.uuid4()), unique=True, index=True)
    message_uuid = Column(String, ForeignKey("messages.message_uuid"), nullable=False)
    session_uuid = Column(String, ForeignKey("conversations.session_uuid"), nullable=False)
    item_node = Column(String, nullable=False)
    item_type = Column(String, nullable=False)
    item_content = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
