from pydantic import BaseModel
from typing import Optional, List

class GeneratePoemRequest(BaseModel):
    model: str
    prompt: str
    # max_length: int = 50

class Step(BaseModel):
    error_poem: str
    step_content: str
    edited_poem: str
    reasoning_score: Optional[int] = None
    meaning_score: Optional[bool] = None
    imagery_score: Optional[bool] = None

class ChainRequest(BaseModel):
    original_poem: str
    steps: List[Step]