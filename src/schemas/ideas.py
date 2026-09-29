from typing import List, Optional
from pydantic import BaseModel, Field


class UserIdeaInput(BaseModel):
    idea: str = Field(..., min_length=5, description="Natural language invention or research idea")
    domain: Optional[str] = Field(None, description="Optional domain context, e.g., AI/ML, Agriculture, NLP")
    title: Optional[str] = Field(None, description="Optional title for the idea")


class ExtractedFeatures(BaseModel):
    title: str = Field(..., description="Concise descriptive title of the idea")
    description: str = Field(..., description="Refined description of the proposed concept")
    domain: str = Field("Artificial Intelligence & Machine Learning", description="Domain classification")
    technologies: List[str] = Field(default_factory=list, description="Core technologies mentioned (e.g. LLMs, RAG, Computer Vision)")
    technical_features: List[str] = Field(default_factory=list, description="Specific technical components and mechanisms")
    methods: List[str] = Field(default_factory=list, description="Algorithmic or procedural methods employed")
    inputs: List[str] = Field(default_factory=list, description="Input data types, sensors, or formats")
    outputs: List[str] = Field(default_factory=list, description="Outputs, predictions, decisions, or generated artifacts")
    key_components: List[str] = Field(default_factory=list, description="Architectural modules or system subsystems")
