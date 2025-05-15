from sqlalchemy.orm import Session
from app.models import reasoning_models as models
from app.schemas import reasoning_schemas as schemas

def create_reasoning_chain(db: Session, chain: schemas.ReasoningChainCreate):
    db_chain = models.ReasoningChain(original_poem=chain.original_poem)
    db.add(db_chain)
    db.commit()
    db.refresh(db_chain)
    return db_chain

def get_reasoning_chain(db: Session, chain_id: int):
    return db.query(models.ReasoningChain).filter(models.ReasoningChain.id == chain_id).first()

def create_reasoning_step(db: Session, step: schemas.ReasoningStepCreate, chain_id: int):
    db_step = models.ReasoningStep(
        chain_id=chain_id,
        error_poem=step.error_poem,
        step_content=step.step_content,
        edited_poem=step.edited_poem,
        me_score=step.me_score,
        ie_score=step.ie_score,
        coherence_score=step.coherence_score
    )
    db.add(db_step)
    db.commit()
    db.refresh(db_step)
    return db_step

def get_steps_for_chain(db: Session, chain_id: int):
    return db.query(models.ReasoningStep).filter(models.ReasoningStep.chain_id == chain_id).all()