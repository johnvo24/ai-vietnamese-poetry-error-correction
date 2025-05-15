from pydantic import BaseModel
from typing import List, Optional

class ReasoningStepBase(BaseModel):
    error_poem: str
    step_content: str
    edited_poem: str
    me_score: Optional[bool] = None
    ie_score: Optional[bool] = None
    coherence_score: Optional[int] = None

class ReasoningStepCreate(ReasoningStepBase):
    pass

class ReasoningStep(ReasoningStepBase):
    id: int
    chain_id: int

    class Config:
        orm_mode = True

class ReasoningChainBase(BaseModel):
    original_poem: str

class ReasoningChainCreate(ReasoningChainBase):
    pass

class ReasoningChain(ReasoningChainBase):
    id: int
    steps: List[ReasoningStep] = []

    class Config:
        orm_mode = True