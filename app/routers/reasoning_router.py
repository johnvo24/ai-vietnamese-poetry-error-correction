from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.schemas import reasoning_schemas as schemas
from app.database import db
from app.controllers import reasoning_controller as controller

router = APIRouter()

@router.post("/chains/", response_model=schemas.ReasoningChain)
def create_chain(chain: schemas.ReasoningChainCreate, db: Session = Depends(db.get_db)):
    return controller.create_reasoning_chain(db, chain)

@router.get("/chains/{chain_id}", response_model=schemas.ReasoningChain)
def read_chain(chain_id: int, db: Session = Depends(db.get_db)):
    db_chain = controller.get_reasoning_chain(db, chain_id=chain_id)
    if db_chain is None:
        raise HTTPException(status_code=404, detail="Chain not found")
    return db_chain

@router.post("/chains/{chain_id}/steps/", response_model=schemas.ReasoningStep)
def create_step_for_chain(
    chain_id: int, step: schemas.ReasoningStepCreate, db: Session = Depends(db.get_db)
):
    return controller.create_reasoning_step(db, step=step, chain_id=chain_id)

@router.get("/chains/{chain_id}/steps/", response_model=list[schemas.ReasoningStep])
def read_steps_for_chain(chain_id: int, db: Session = Depends(db.get_db)):
    return controller.get_steps_for_chain(db, chain_id=chain_id)