"""Streamlit UI demo for IntelligenceService.tag_ioi().

Run with:
    streamlit run streamlit_ioi_demo.py

Calls the rebuilt async IntelligenceService (plan Step 0.6) via
asyncio.run() since Streamlit's main script is sync. Each "Parse tags"
or "Regenerate" click is one independent call to the S1 IOI Tagger
agent via the DAF execute endpoint.
"""

import asyncio

from dotenv import load_dotenv

load_dotenv()

import streamlit as st

from csf_recommendation_engine.core.config import get_settings
from csf_recommendation_engine.domain.intelligence.intelligence_service import (
    IntelligenceService,
)


TAG_FIELDS: list[tuple[str, str]] = [
    ("side", "Side (Buy / Sell / Either / None)"),
    ("product", "Product (WTI, Brent, RBOB, HO, NG, ...)"),
    ("qty", "Quantity (lots — integer or range)"),
    ("tenor", "Tenor"),
    ("price", "Price (float or range)"),
    ("qualifier", "Qualifier (at / better / worse)"),
    ("urgency", "Urgency (Low / Med / High)"),
    ("sentiment", "Sentiment (Bullish / Bearish / Neutral)"),
]


@st.cache_resource
def get_service() -> IntelligenceService:
    return IntelligenceService(get_settings())


def _tag_ioi_sync(service: IntelligenceService, text: str):
    """Run the async tagger call from a sync Streamlit handler."""
    return asyncio.run(service.tag_ioi(text))


st.set_page_config(page_title="IOI Tag Parser", layout="centered")
st.title("IOI Tag Parser")
st.caption(
    "Enter free-form Indication-of-Interest text. The intelligence service "
    "extracts structured tags using the configured DAF Tagger agent."
)

default_example = "Mike at Vitol buying 50 lots WTI Mar26 at 71.20, working order"
ioi_text = st.text_area(
    "Natural-language IOI",
    value=default_example,
    height=120,
    placeholder="e.g. Mike at Vitol buying 50 lots WTI Mar26 at 71.20, working order",
)

submit = st.button("Parse tags", type="primary", disabled=not ioi_text.strip())

if submit:
    service = get_service()
    with st.spinner("Calling DAF Tagger agent..."):
        tagger_response = _tag_ioi_sync(service, ioi_text.strip())

    if tagger_response is None:
        st.error("Failed to parse tags. Check logs / agent configuration.")
    else:
        st.success("Tags extracted")

        tag_dict = tagger_response.model_dump()
        cols = st.columns(2)
        for idx, (key, label) in enumerate(TAG_FIELDS):
            with cols[idx % 2]:
                value = tag_dict.get(key, "")
                st.text_input(
                    label,
                    value="" if value is None else str(value),
                    disabled=True,
                    key=f"out_{key}",
                )

        with st.expander("Raw response (full TaggerResponse)"):
            st.json(tag_dict)
