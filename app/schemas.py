from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(min_length=3)
    top_k: int = Field(default=3, ge=1, le=10)


class Source(BaseModel):
    document: str
    chunk_id: int


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]
