"""Intelligence layer: Pydantic packet/response shapes, hashing,
validators, critic state, and the swarm-facing service.

Steps 0.4 (packet + responses + hashing), 0.5 (validators + critic
state), and 0.6 (IntelligenceService + DAFTransport) are all exported
here. The deprecated ``intelligence_layer.py`` and
``new_intelligence_service.py`` have been removed.
"""

from csf_recommendation_engine.domain.intelligence.critic_state import (
    CriticState,
    CriticStateSnapshot,
)
from csf_recommendation_engine.domain.intelligence.hashing import (
    ENVELOPE_FIELDS_EXCLUDED_FROM_HASH,
    attach_packet_hash,
    canonical_json,
    compute_packet_hash,
)
from csf_recommendation_engine.domain.intelligence.intelligence_service import (
    IntelligenceService,
)
from csf_recommendation_engine.domain.intelligence.packet import (
    PACKET_SCHEMA_VERSION,
    Candidate,
    Eligibility,
    EntityDossier,
    ExposureContext,
    FeedbackNote,
    FitSignals,
    IntelligencePacket,
    IntentTags,
    MarketBriefing,
    PolicyBundle,
    RankerScores,
    RequestContext,
)
from csf_recommendation_engine.domain.intelligence.responses import (
    CoverageCoachResponse,
    InstrumentResolutionResponse,
    MarketReaderResponse,
    MatchStrategistResponse,
    ProfilerResponse,
    RecommendationRationale,
    RecommenderExplainerResponse,
    TaggerResponse,
)
from csf_recommendation_engine.domain.intelligence.transport import (
    DAFTransport,
    ExecuteResult,
    TransportError,
)
from csf_recommendation_engine.domain.intelligence.validators import (
    Decision,
    PacketVocab,
    ValidationResult,
    apply_critic_mode,
    build_packet_vocab,
    validate_citations,
    validate_eligibility,
    validate_response,
)

__all__ = [
    # packet.py
    "PACKET_SCHEMA_VERSION",
    "Candidate",
    "Eligibility",
    "EntityDossier",
    "ExposureContext",
    "FeedbackNote",
    "FitSignals",
    "IntelligencePacket",
    "IntentTags",
    "MarketBriefing",
    "PolicyBundle",
    "RankerScores",
    "RequestContext",
    # responses.py
    "CoverageCoachResponse",
    "InstrumentResolutionResponse",
    "MarketReaderResponse",
    "MatchStrategistResponse",
    "ProfilerResponse",
    "RecommendationRationale",
    "RecommenderExplainerResponse",
    "TaggerResponse",
    # hashing.py
    "ENVELOPE_FIELDS_EXCLUDED_FROM_HASH",
    "attach_packet_hash",
    "canonical_json",
    "compute_packet_hash",
    # validators.py
    "Decision",
    "PacketVocab",
    "ValidationResult",
    "apply_critic_mode",
    "build_packet_vocab",
    "validate_citations",
    "validate_eligibility",
    "validate_response",
    # critic_state.py
    "CriticState",
    "CriticStateSnapshot",
    # transport.py
    "DAFTransport",
    "ExecuteResult",
    "TransportError",
    # intelligence_service.py
    "IntelligenceService",
]
