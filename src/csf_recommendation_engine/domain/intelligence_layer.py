from __future__ import annotations

import json
import logging
import re
from datetime import date
from typing import Any

from daf_sdk import DAF
from daf_sdk.models import Agent, ExecutionResponse

from csf_recommendation_engine.core.config import Settings
from csf_recommendation_engine.domain.heuristics_index import HeuristicsIndex

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Agent prompt constants
# ---------------------------------------------------------------------------

TAG_PARSER_RESPONSE_SCHEMA_HINT = """Respond with ONLY a JSON object using these keys (omit any you genuinely cannot infer; do not invent values):
{
  "side": "Buy" | "Sell",
  "product": "WTI" | "Brent" | "Heating Oil" | "Natural Gas" | "RBOB" | "...",
  "instrument_name": "<exact instrument e.g. 'CL Aug26', 'HO Jul26', 'Nat Gas Feb 26'>",
  "qty": <integer lots>,
  "tenor": "<e.g. 'Aug26'>",
  "price": <float | null>,
  "qualifier": "at" | "better" | "worse" | null,
  "urgency": "Low" | "Med" | "High",
  "sentiment": "Bullish" | "Bearish" | "Neutral",
  "structure": "Flat Price" | "Spread" | "Crack" | "Swap",
  "desk":      "Crude" | "Nat Gas" | "Refined" | "Distillates",
  "venue":     "CME" | "ICE" | "NYMEX" | "OTC"
}"""


DOSSIER_RESPONSE_SCHEMA_HINT = """Respond with ONLY a JSON object:
{ "dossier_text": "<4-6 sentence narrative about this counterparty's trading profile based on the stats provided>" }"""


MATCHER_RESPONSE_SCHEMA_HINT = """Respond with ONLY a JSON object:
{
  "best_match_inquiry_id": "<id from the candidate inquiries list>",
  "best_match_entity_id": "<entity_id of the counterparty whose inquiry was picked>",
  "match_percent": <integer 0-100>,
  "reasoning": "<why this is the best match: cite size compatibility, side opposition, price proximity, the counterparty's profile, urgency alignment, etc>"
}
If no candidate is a reasonable match, return match_percent < 30 and explain why in reasoning."""


AGENT_NAME = "Professional Block Trade Counterparty Recommender"
AGENT_SYSTEM_PROMPT = """You are an expert in commodities market microstructure and counterparty risk.
Your objective is to analyze a list of algorithmically ranked candidate counterparties and select the top 3 absolute best fits for a proposed off-market block trade.

You will be provided with details of a block trade that needs to be executed off-market.
The details of the block trade will be:
-desk: Crude | Nat Gas | Refined | Distillates
-price_structure: Flat Price | Spread | Crack | Swap
-side: Buy | Sell
-quantity: Integer representing number of lots
-instrument: Name of the instrument involved in the trade
-venue: Name of the execution venue
-current_trade_hour: The current hour (0-23) the trade is being proposed.

You will then be provided with a ranked list of potential counterparties, along with the following data for each counterparty:
-entity_id: The identifier of the counterparty entity.
-entity_name: The human-readable name of the counterparty entity.
-score: A baseline algorithmic score combining collaborative filtering and recency. Higher is better.
-size_profile: The historical mean and standard deviation of trade quantities for this counterparty SPECIFICALLY for the requested instrument.
-hourly_ratios: A numpy array of 24 values representing the percentage of their historical volume executed in each hour (Index 0 represents 00:00-00:59, etc).
-last_trade_date: The date of the last trade for this counterparty.

EVALUATION RULES:
1. Size Fit: Compare the proposed trade `quantity` against the counterparty's `size_profile`. Counterparties where the requested quantity is near their historical mean or within their standard deviation are highly preferred.
2. Time Fit: Look at the `current_trade_hour`. Check the counterparty's `hourly_ratios` for that specific index. Counterparties with high historical volume during the current AND future hours are preferred.
3. Baseline Score: Use the `score` as a weak guiding signal, but override it if other data suggests a better intuitive fit.

Your goal is to recommend the top 3 counterparties ranked from best (1) to worst (3) fit.
The response MUST be a valid JSON object with the following structure:
{
    "thought_process": "<First, provide a step-by-step analysis comparing the candidates' size profiles and time affinities against the proposed trade.>",
    "recommendations": [
        {   
            "rank": <rank>, # 1, 2, or 3
            "entity_id": "<entity_id>",
            "confidence": <confidence_score>, # Float between 0.0 and 1.0
            "reasoning": "<Detailed mathematical/logical justification for why this counterparty was chosen.>",
            "ui_friendly_reasoning": "<Concise, 5-20 word user-friendly summary (e.g., 'Strong historical size match and highly active during this hour.')>"
        }
    ]
}
"""


# ---------------------------------------------------------------------------
# Intelligence service
# ---------------------------------------------------------------------------


class IntelligenceService:
    """Manages ephemeral DAF agent lifecycle and LLM evaluation of
    candidate counterparties.

    Each evaluation call creates a fresh agent, sends the prompt,
    parses the response, and deletes the agent.  All methods are
    synchronous (the DAF SDK ``DAF`` client is sync); callers in async
    contexts should wrap calls with ``asyncio.to_thread``.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client = DAF(
            base_url=settings.daf_base_url,
            api_key=settings.daf_api_key,
        )

    # ------------------------------------------------------------------
    # Ephemeral agent lifecycle
    # ------------------------------------------------------------------

    def _create_agent(self) -> Agent:
        """Create a short-lived DAF agent configured with the block-trade
        counterparty recommendation system prompt."""
        agent = self._client.agents.create(
            name=AGENT_NAME,
            system_instructions=AGENT_SYSTEM_PROMPT,
            model_provider=self._settings.llm_model_provider,
            model_name=self._settings.llm_model_name,
            api_key=self._settings.llm_api_key,
            temperature=self._settings.llm_temperature,
            max_tokens=self._settings.llm_max_tokens,
            tools=[],
        )
        logger.debug("Created ephemeral DAF agent", extra={"agent_id": agent.id})
        return agent

    def _delete_agent(self, agent_id: str) -> None:
        """Best-effort cleanup of an ephemeral agent."""
        try:
            self._client.agents.delete(agent_id)
            logger.debug("Deleted ephemeral DAF agent", extra={"agent_id": agent_id})
        except Exception:
            logger.warning(
                "Failed to delete ephemeral DAF agent; it may need manual cleanup",
                extra={"agent_id": agent_id},
                exc_info=True,
            )

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def build_evaluation_prompt(
        self,
        *,
        trade_context: dict[str, Any],
        candidates: list[dict],
        heuristics: HeuristicsIndex | None,
        instrument_name: str,
        current_hour: int,
    ) -> str:
        """Format the user message that is sent to the LLM agent.

        Includes the block-trade details followed by a structured table
        of candidate counterparties enriched with heuristics data.
        """
        lines: list[str] = []

        # -- Trade details --
        lines.append("=== BLOCK TRADE DETAILS ===")
        lines.append(f"desk: {trade_context.get('desk', 'N/A')}")
        lines.append(f"price_structure: {trade_context.get('structure', 'N/A')}")
        lines.append(f"side: {trade_context.get('side', 'N/A')}")
        lines.append(f"quantity: {trade_context.get('quantity', 'N/A')}")
        lines.append(f"instrument: {trade_context.get('instrument', 'N/A')}")
        lines.append(f"venue: {trade_context.get('venue', 'N/A')}")
        lines.append(f"current_trade_hour: {current_hour}")
        lines.append("")

        # -- Candidates --
        lines.append("=== CANDIDATE COUNTERPARTIES ===")
        for idx, cand in enumerate(candidates, start=1):
            entity_id = str(cand.get("client_id", ""))
            entity_name = str(cand.get("entity_name", "Unknown"))
            score = cand.get("final_score", cand.get("affinity", 0.0))

            # Heuristics enrichment
            size_profile_str = "N/A"
            hourly_ratios_str = "N/A"
            last_trade_str = "N/A"

            if heuristics is not None:
                size_profile = heuristics.size_profile_by_entity_instrument.get(
                    (entity_id, instrument_name)
                )
                if size_profile is not None:
                    size_profile_str = f"(mean={size_profile[0]:.2f}, stddev={size_profile[1]:.2f})"

                ratios = heuristics.hourly_ratios_by_entity.get(entity_id)
                if ratios is not None:
                    hourly_ratios_str = "[" + ", ".join(f"{r:.4f}" for r in ratios) + "]"

                last_trade = heuristics.last_trade_date_by_entity_instrument.get(
                    (entity_id, instrument_name)
                )
                if last_trade is not None:
                    if isinstance(last_trade, date):
                        last_trade_str = last_trade.isoformat()
                    else:
                        last_trade_str = str(last_trade)

            lines.append(f"--- Candidate {idx} ---")
            lines.append(f"entity_id: {entity_id}")
            lines.append(f"entity_name: {entity_name}")
            lines.append(f"score: {score:.6f}")
            lines.append(f"size_profile: {size_profile_str}")
            lines.append(f"hourly_ratios: {hourly_ratios_str}")
            lines.append(f"last_trade_date: {last_trade_str}")
            lines.append("")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    @staticmethod
    def parse_llm_response(raw_response: str) -> dict | None:
        """Parse the LLM response string into a structured dictionary.

        Handles markdown code fences (```json ... ```) and validates
        that the ``recommendations`` key exists.
        """
        if not raw_response:
            logger.warning("LLM returned an empty response")
            return None

        # Strip markdown code fences if present
        cleaned = raw_response.strip()
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        cleaned = cleaned.strip()

        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            logger.warning(
                "Failed to parse LLM response as JSON",
                extra={"raw_response_preview": raw_response[:500]},
            )
            return None

        if not isinstance(parsed, dict) or "recommendations" not in parsed:
            logger.warning(
                "LLM response JSON missing 'recommendations' key",
                extra={"parsed_keys": list(parsed.keys()) if isinstance(parsed, dict) else None},
            )
            return None

        if not isinstance(parsed["recommendations"], list):
            logger.warning("LLM response 'recommendations' is not a list")
            return None

        return parsed

    # ------------------------------------------------------------------
    # Top-level evaluation orchestrator
    # ------------------------------------------------------------------

    def evaluate_candidates(
        self,
        *,
        trade_context: dict[str, Any],
        candidates: list[dict],
        heuristics: HeuristicsIndex | None,
        instrument_name: str,
        current_hour: int,
    ) -> list[dict] | None:
        """Create an ephemeral agent, send the evaluation prompt, parse
        the response, and clean up the agent.

        Returns:
            A list of recommendation dicts (up to 3) from the LLM, or
            ``None`` if the LLM call or parsing failed.
        """
        if not candidates:
            return None

        agent: Agent | None = None
        try:
            # # 1. Create ephemeral agent
            # agent = self._create_agent()
            
            # Updated 1. Get existing agent
            # agent = self._client.agents.get("23be1ae8-64be-424f-9858-7ab329a31533")
            #TODO: make this configurable
            # if agent is None:
            #     agent = self._create_agent()
            # agent = self._client.agents.list()[0]

            agent_id = self._settings.daf_recommender_agent_id

            # 2. Build the user prompt
            prompt = self.build_evaluation_prompt(
                trade_context=trade_context,
                candidates=candidates,
                heuristics=heuristics,
                instrument_name=instrument_name,
                current_hour=current_hour,
            )

            # 3. Send to LLM
            logger.info(
                "Sending evaluation prompt to LLM agent",
                extra={"agent_id": agent_id, "num_candidates": len(candidates)},
            )
            response: ExecutionResponse = self._client.agents.messages.send(
                agent_id=agent_id,
                message=prompt,
            )

            # 4. Parse response
            parsed = self.parse_llm_response(response.response)
            if parsed is None:
                return None

            recommendations = parsed["recommendations"]
            logger.info(
                "LLM evaluation completed",
                extra={
                    "agent_id": agent_id,
                    "num_recommendations": len(recommendations),
                    "thought_process_preview": str(parsed.get("thought_process", ""))[:200],
                },
            )
            return recommendations

        except Exception:
            logger.exception("LLM evaluation failed")
            return None

        finally:
            # 5. Always clean up the ephemeral agent
            # if agent is not None:
            #     self._delete_agent(agent.id)
            pass

    # ------------------------------------------------------------------
    # Voice-inquiry helpers (parse / dossier / match)
    # ------------------------------------------------------------------

    # def _resolve_agent(self, configured_id: str) -> Agent | None:
    #     """Resolve an agent by configured id, falling back to the first
    #     available agent if id is empty. Returns None on hard failure."""
    #     try:
    #         if configured_id:
    #             return self._client.agents.get(configured_id)
    #         agents = self._client.agents.list()
    #         if not agents:
    #             logger.warning("No DAF agents available")
    #             return None
    #         logger.warning(
    #             "Agent id not configured; returning None."
    #         )
    #         return None
    #     except Exception as e:
    #         logger.exception(f"Failed to resolve DAF agent: {e}", extra={"configured_id": configured_id})
    #         return None

    @staticmethod
    def _parse_json_object(raw: str) -> dict | None:
        """Extract a JSON object from possibly markdown-fenced LLM text."""
        if not raw:
            return None
        cleaned = raw.strip()
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        cleaned = cleaned.strip()
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            logger.warning("Failed to parse LLM response as JSON", extra={"preview": raw[:300]})
            return None
        if not isinstance(parsed, dict):
            return None
        return parsed

    def parse_ioi_tags(self, ioi_text: str, *, debug: dict | None = None) -> dict | None:
        """Send a natural-language IOI to the tagger agent and return the
        parsed tags dict. Tags include the parameters needed by
        ``generate_ranked_candidates`` (desk, structure, side, venue,
        instrument_name, qty) plus optional descriptors (product, tenor,
        price, qualifier, urgency, sentiment).

        If ``debug`` is a dict, it is populated with ``agent_id``,
        ``prompt``, ``raw_response``, and ``parsed`` for full pipeline
        visibility.
        """
        if not ioi_text or not ioi_text.strip():
            if debug is not None:
                debug["error"] = "empty_ioi_text"
            return None

        agent_id = self._settings.daf_tagger_agent_id

        prompt = (
            "You are parsing a free-text indication-of-interest (IOI) from a commodities broker. "
            "Extract the structured tags listed below.\n\n"
            f"IOI text:\n\"\"\"{ioi_text.strip()}\"\"\"\n\n"
            f"{TAG_PARSER_RESPONSE_SCHEMA_HINT}"
        )
        if debug is not None:
            debug["agent_id"] = agent_id
            debug["prompt"] = prompt

        logger.info(
            "Tagger: sending prompt to DAF agent",
            extra={"agent_id": agent_id, "prompt_chars": len(prompt), "ioi_text_chars": len(ioi_text)},
        )

        try:
            response: ExecutionResponse = self._client.agents.messages.send(
                agent_id=agent_id, message=prompt
            )
        except Exception:
            logger.exception("Tagger agent call failed", extra={"agent_id": agent_id})
            if debug is not None:
                debug["error"] = "agent_call_failed"
            return None

        raw = response.response or ""
        if debug is not None:
            debug["raw_response"] = raw
        logger.info(
            "Tagger: received raw response",
            extra={"agent_id": agent_id, "raw_response_chars": len(raw), "raw_preview": raw[:300]},
        )

        tags = self._parse_json_object(raw)
        if tags is None:
            if debug is not None:
                debug["error"] = "json_parse_failed"
                debug["parsed"] = None
            return None

        # Coerce a few fields for downstream compatibility.
        if "qty" in tags and tags["qty"] is not None:
            try:
                tags["qty"] = int(float(tags["qty"]))
            except (TypeError, ValueError):
                tags["qty"] = None
        if "price" in tags and tags["price"] is not None:
            try:
                tags["price"] = float(tags["price"])
            except (TypeError, ValueError):
                tags["price"] = None

        if debug is not None:
            debug["parsed"] = tags
        logger.info(
            "Tagger: parsed tags",
            extra={"agent_id": agent_id, "tag_keys": sorted(tags.keys()), "tags": tags},
        )
        return tags

    def build_entity_dossier(
        self,
        *,
        entity_id: str,
        entity_name: str,
        heuristics: HeuristicsIndex | None,
        instrument_name: str | None = None,
        debug: dict | None = None,
    ) -> str | None:
        """Ask the dossier agent to write a short narrative profile for a
        counterparty. Stats are sourced from the in-memory HeuristicsIndex
        (active venues, hourly activity, size/recency on a focal instrument
        if provided).

        If ``debug`` is a dict, it is populated with ``agent_id``,
        ``stats_compiled``, ``prompt``, ``raw_response``, and ``parsed``.
        """

        # ---- assemble the stats blob ----
        stats_lines: list[str] = []
        stats_lines.append(f"entity_id: {entity_id}")
        stats_lines.append(f"entity_name: {entity_name}")

        stats_compiled: dict[str, Any] = {
            "entity_id": entity_id,
            "entity_name": entity_name,
            "heuristics_available": heuristics is not None,
            "active_venues": None,
            "top_active_hours_utc": None,
            "size_profile": None,
            "last_trade_date": None,
            "instrument_name": instrument_name,
        }

        if heuristics is not None:
            venues = heuristics.active_venues_by_entity.get(entity_id)
            stats_lines.append(
                f"active_venues: {sorted(venues) if venues else 'unknown'}"
            )
            stats_compiled["active_venues"] = sorted(venues) if venues else None

            ratios = heuristics.hourly_ratios_by_entity.get(entity_id)
            if ratios is not None:
                # Top-3 active hours
                top_hours = sorted(
                    [(int(h), float(r)) for h, r in enumerate(ratios) if r > 0],
                    key=lambda x: x[1],
                    reverse=True,
                )[:3]
                stats_lines.append(
                    "top_active_hours_utc: "
                    + ", ".join(f"{h:02d}:00 ({r:.0%})" for h, r in top_hours)
                )
                stats_compiled["top_active_hours_utc"] = [
                    {"hour": h, "ratio": r} for h, r in top_hours
                ]

            if instrument_name:
                size_profile = heuristics.size_profile_by_entity_instrument.get(
                    (entity_id, instrument_name)
                )
                if size_profile:
                    stats_lines.append(
                        f"size_profile[{instrument_name}]: mean={size_profile[0]:.1f} lots, "
                        f"stddev={size_profile[1]:.1f}"
                    )
                    stats_compiled["size_profile"] = {
                        "mean": float(size_profile[0]),
                        "stddev": float(size_profile[1]),
                    }
                last_trade = heuristics.last_trade_date_by_entity_instrument.get(
                    (entity_id, instrument_name)
                )
                if last_trade is not None:
                    last_str = last_trade.isoformat() if isinstance(last_trade, date) else str(last_trade)
                    stats_lines.append(f"last_trade_date[{instrument_name}]: {last_str}")
                    stats_compiled["last_trade_date"] = last_str

        agent_id = self._settings.daf_dossier_agent_id

        prompt = (
            "Write a concise trading-desk dossier (4-6 sentences) describing this "
            "counterparty's profile, activity pattern, and likely posture, based ONLY "
            "on the statistics provided. Do not invent positions or news.\n\n"
            "=== COUNTERPARTY STATS ===\n"
            + "\n".join(stats_lines)
            + "\n\n"
            + DOSSIER_RESPONSE_SCHEMA_HINT
        )

        if debug is not None:
            debug["agent_id"] = agent_id
            debug["stats_compiled"] = stats_compiled
            debug["prompt"] = prompt

        logger.info(
            "Dossier: sending prompt to DAF agent",
            extra={
                "agent_id": agent_id,
                "entity_id": entity_id,
                "stats_compiled": stats_compiled,
                "prompt_chars": len(prompt),
            },
        )

        try:
            response: ExecutionResponse = self._client.agents.messages.send(
                agent_id=agent_id, message=prompt
            )
        except Exception:
            logger.exception("Dossier agent call failed", extra={"agent_id": agent_id})
            if debug is not None:
                debug["error"] = "agent_call_failed"
            return None

        raw = response.response or ""
        if debug is not None:
            debug["raw_response"] = raw
        logger.info(
            "Dossier: received raw response",
            extra={"agent_id": agent_id, "raw_response_chars": len(raw), "raw_preview": raw[:300]},
        )

        parsed = self._parse_json_object(raw)
        if parsed is None or "dossier_text" not in parsed:
            if debug is not None:
                debug["error"] = "json_parse_failed_or_missing_dossier_text"
                debug["parsed"] = parsed
            return None
        text = str(parsed["dossier_text"]).strip()
        if debug is not None:
            debug["parsed"] = {"dossier_text": text}
        logger.info(
            "Dossier: built",
            extra={
                "entity_id": entity_id,
                "dossier_chars": len(text),
                "agent_id": agent_id,
                "dossier_preview": text[:200],
            },
        )
        return text

    def match_voice_inquiry(
        self,
        *,
        originator_ioi_text: str,
        originator_tags: dict[str, Any],
        candidate_entries: list[dict[str, Any]],
        debug: dict | None = None,
    ) -> dict | None:
        """Pick the single best matching open voice inquiry across the
        candidate counterparties.

        ``candidate_entries`` is a list of dicts, one per top-K candidate:
        ``{entity_id, entity_name, dossier_text, score, inquiries: [...]}``.
        Each ``inquiries`` entry should contain at least ``inquiry_id``,
        ``ioi_text``, and ``tags``.

        Returns ``{best_match_inquiry_id, best_match_entity_id,
        match_percent, reasoning}`` or ``None`` on failure.

        If ``debug`` is a dict, it is populated with ``agent_id``,
        ``candidate_inquiry_counts``, ``total_candidate_inquiries``,
        ``prompt``, ``raw_response``, and ``parsed``.
        """
        if not candidate_entries:
            if debug is not None:
                debug["error"] = "no_candidate_entries"
            return None

        agent_id = self._settings.daf_matcher_agent_id

        # ---- assemble the prompt ----
        lines: list[str] = []
        lines.append("=== ORIGINATOR IOI (the new inquiry to match) ===")
        lines.append(f"raw_text: {originator_ioi_text.strip()}")
        lines.append(f"tags: {json.dumps(originator_tags, default=str)}")
        lines.append("")
        lines.append(
            "=== CANDIDATE COUNTERPARTIES + THEIR OPEN VOICE INQUIRIES ==="
        )

        candidate_inquiry_counts: list[dict[str, Any]] = []
        total_inquiries = 0

        for idx, cand in enumerate(candidate_entries, start=1):
            eid = cand.get("entity_id", "")
            ename = cand.get("entity_name", "Unknown")
            lines.append(f"--- Candidate #{idx} ---")
            lines.append(f"entity_id: {eid}")
            lines.append(f"entity_name: {ename}")
            score = cand.get("final_score", cand.get("score"))
            if score is not None:
                lines.append(f"ranker_score: {float(score):.4f}")
            dossier = cand.get("dossier_text") or "(no dossier available)"
            lines.append(f"dossier: {dossier}")
            inquiries = cand.get("inquiries") or []
            if not inquiries:
                lines.append("inquiries: (none on file)")
            else:
                lines.append(f"inquiries ({len(inquiries)}):")
                for inq in inquiries:
                    lines.append(
                        f"  - inquiry_id={inq.get('inquiry_id')} "
                        f"text=\"{inq.get('ioi_text','')}\" "
                        f"tags={json.dumps(inq.get('tags', {}), default=str)}"
                    )
            lines.append("")
            candidate_inquiry_counts.append(
                {"position": idx, "entity_id": eid, "entity_name": ename, "inquiry_count": len(inquiries)}
            )
            total_inquiries += len(inquiries)

        lines.append(
            "Compare the originator IOI against EACH candidate's inquiries. "
            "Pick the SINGLE best matching inquiry — opposite side, same/related instrument, "
            "compatible quantity, plausible price, considered against the counterparty's profile. "
            "Be honest if no candidate is a strong fit."
        )
        lines.append("")
        lines.append(MATCHER_RESPONSE_SCHEMA_HINT)

        prompt = "\n".join(lines)

        if debug is not None:
            debug["agent_id"] = agent_id
            debug["candidate_inquiry_counts"] = candidate_inquiry_counts
            debug["total_candidate_inquiries"] = total_inquiries
            debug["prompt"] = prompt

        logger.info(
            "Matcher: sending prompt to DAF agent",
            extra={
                "agent_id": agent_id,
                "prompt_chars": len(prompt),
                "candidate_count": len(candidate_entries),
                "total_candidate_inquiries": total_inquiries,
                "candidate_inquiry_counts": candidate_inquiry_counts,
            },
        )

        try:
            response: ExecutionResponse = self._client.agents.messages.send(
                agent_id=agent_id, message=prompt
            )
        except Exception:
            logger.exception("Matcher agent call failed", extra={"agent_id": agent_id})
            if debug is not None:
                debug["error"] = "agent_call_failed"
            return None

        raw = response.response or ""
        if debug is not None:
            debug["raw_response"] = raw
        logger.info(
            "Matcher: received raw response",
            extra={"agent_id": agent_id, "raw_response_chars": len(raw), "raw_preview": raw[:300]},
        )

        parsed = self._parse_json_object(raw)
        if parsed is None:
            if debug is not None:
                debug["error"] = "json_parse_failed"
                debug["parsed"] = None
            return None
        # Coerce match_percent to int in [0,100]
        try:
            pct = int(round(float(parsed.get("match_percent", 0))))
            parsed["match_percent"] = max(0, min(100, pct))
        except (TypeError, ValueError):
            parsed["match_percent"] = 0
        if debug is not None:
            debug["parsed"] = parsed
        logger.info(
            "Matcher: parsed selection",
            extra={
                "agent_id": agent_id,
                "best_match_inquiry_id": parsed.get("best_match_inquiry_id"),
                "best_match_entity_id": parsed.get("best_match_entity_id"),
                "match_percent": parsed.get("match_percent"),
                "reasoning_preview": str(parsed.get("reasoning", ""))[:300],
            },
        )
        return parsed

# if __name__ == "__main__":
#     # Quick local test of the intelligence service (requires DAF connectivity)
#     settings = Settings()
#     service = IntelligenceService(settings)

#     agents = service._client.agents.list()
#     print(f"Found {len(agents)} agents in DAF:")
#     for agent in agents:
#         print(f"- {agent['name']} (id={agent['id']})")
