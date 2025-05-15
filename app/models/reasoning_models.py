from sqlalchemy import Column, Integer, Text, Boolean, ForeignKey
from sqlalchemy.orm import relationship, declarative_base

Base = declarative_base()

class ReasoningChain(Base):
    __tablename__ = "reasoning_chains"

    id = Column(Integer, primary_key=True, index=True)
    original_poem = Column(Text, nullable=False)

    steps = relationship("ReasoningStep", back_populates="chain", cascade="all, delete-orphan")

class ReasoningStep(Base):
    __tablename__ = "reasoning_steps"

    id = Column(Integer, primary_key=True, index=True)
    chain_id = Column(Integer, ForeignKey("reasoning_chains.id"), nullable=False)
    error_poem=Column(Text, nullable=False)
    step_content = Column(Text, nullable=False)
    edited_poem=Column(Text, nullable=False)
    me_score = Column(Boolean, nullable=True)       
    ie_score = Column(Boolean, nullable=True)       
    coherence_score = Column(Integer, nullable=True)

    chain = relationship("ReasoningChain", back_populates="steps")