from pydantic import BaseModel

class QuestionRequest(BaseModel):
    # Request schema for the Question Answering endpoint (/ask).
    question: str

class DocumentRequest(BaseModel):
    # Request schema for the Addition Answering endpoint (/add).
    text: str
