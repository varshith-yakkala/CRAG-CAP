from dataclasses import dataclass, field
from typing import Optional, Dict, Any

@dataclass
class RetrievedDocument:
    doc_id: str
    text: str
    title: Optional[str] = ""
    score: Optional[float] = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
