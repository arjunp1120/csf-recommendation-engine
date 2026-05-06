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
2. Time Fit: Look at the `current_trade_hour`. Check the counterparty's `hourly_ratios` for that specific index. Counterparties with high historical volume during the current hour are preferred.
3. Baseline Score: Use the `score` as a strong guiding signal, but override it if the raw size/time data suggests a better intuitive fit.

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
            agent = self._client.agents.list()[0]

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
                extra={"agent_id": agent.id, "num_candidates": len(candidates)},
            )
            response: ExecutionResponse = self._client.agents.messages.send(
                agent_id=agent.id,
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
                    "agent_id": agent.id,
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
