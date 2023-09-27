import time
from typing import Dict, List, Literal, Optional, Union
from pydantic import BaseModel, Field

class TokenCheckRequest(BaseModel):
    model: str
    prompt: str
    max_tokens: int

class TokenCheckResponse(BaseModel):
    fits: bool
    token_count: int
    context_length: int