"""
Pydantic models for crawled property data.
Enforces schema on every crawled page.
"""

from __future__ import annotations
import hashlib
import json
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field, computed_field, model_validator


class PropertyDocument(BaseModel):
    """
    Canonical data model for a single Prime Lands property listing.

    All fields validated at construction. The content_hash field enables
    deduplication across crawl sessions.

    Attributes:
        property_id: Unique identifier derived from URL slug
        title: Property listing headline
        address: Full address string
        price: Price as displayed (LKR string)
        bedrooms: Count of bedrooms (None if not listed)
        bathrooms: Count of bathrooms (None if not listed)
        sqft: Floor area in square feet (None if not listed)
        amenities: List of amenity strings
        agent: Listing agent name
        url: Canonical source URL
        description: Full property description
        images: List of image URLs
        property_type: residential/commercial/land
        crawled_at: ISO timestamp of crawl
        content_hash: SHA256 of title+description for dedup
    """

    property_id: str = Field(..., min_length=1)
    title: str = Field(..., min_length=1)
    address: str = Field(default="")
    price: str = Field(default="")
    bedrooms: Optional[int] = Field(default=None, ge=0)
    bathrooms: Optional[int] = Field(default=None, ge=0)
    sqft: Optional[float] = Field(default=None, ge=0)
    amenities: list[str] = Field(default_factory=list)
    agent: str = Field(default="")
    url: str = Field(..., min_length=1)
    description: str = Field(default="")
    images: list[str] = Field(default_factory=list)
    property_type: str = Field(default="residential")
    crawled_at: str = Field(
        default_factory=lambda: datetime.now().isoformat()
    )
    content_hash: str = Field(default="")

    @model_validator(mode="after")
    def compute_content_hash(self) -> "PropertyDocument":
        """Compute SHA256 hash of content for deduplication."""
        content = f"{self.title}|{self.description}"
        self.content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
        return self

    def to_rag_text(self) -> str:
        """
        Generate the full text representation used for chunking and embedding.

        Returns:
            Formatted multi-field text document
        """
        parts = [
            f"# {self.title}",
            f"**Address:** {self.address}" if self.address else "",
            f"**Price:** {self.price}" if self.price else "",
            f"**Bedrooms:** {self.bedrooms}" if self.bedrooms is not None else "",
            f"**Bathrooms:** {self.bathrooms}" if self.bathrooms is not None else "",
            f"**Area:** {self.sqft} sqft" if self.sqft else "",
            f"**Agent:** {self.agent}" if self.agent else "",
            "",
            "## Description",
            self.description,
            "",
            "## Amenities",
            "\n".join(f"- {a}" for a in self.amenities) if self.amenities else "N/A",
        ]
        return "\n".join(p for p in parts if p is not None)
    
    def to_jsonl(self) -> str:
        """Serialize to JSONL format."""
        return self.model_dump_json()
