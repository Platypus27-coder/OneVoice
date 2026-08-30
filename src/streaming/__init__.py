"""Streaming ASR alignment and semantic commit support."""

from .semantic_commit import (
    RollingHypothesisAssembler,
    SemanticCommitController,
    StablePrefixAligner,
)
from .session import RollingAudioEvent, RollingUtteranceSession

# Public V2 name from the design document; keep the descriptive legacy class
# name as the implementation and compatibility import.
StreamingSession = RollingUtteranceSession

__all__ = [
    "RollingHypothesisAssembler", "SemanticCommitController", "StablePrefixAligner",
    "RollingAudioEvent", "RollingUtteranceSession", "StreamingSession",
]
