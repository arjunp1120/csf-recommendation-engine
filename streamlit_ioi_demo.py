"""Streamlit demo for the /voice_inquiry/* endpoints.

Run with:
    streamlit run streamlit_ioi_demo.py

This demo talks to a running FastAPI backend (default http://localhost:8000)
over HTTP — it does NOT spin up its own IntelligenceService, because the
match pipeline needs the same app_state, DB pool, champion model, and
heuristics index that the API server bootstraps at startup.

Tabs:
  - 🎯 Match     POST /voice_inquiry/match  (full pipeline)
  - 🏷️  Tags     POST /voice_inquiry/tags   (tag parser in isolation)
  - 📚 Inquiries Local view of voice_inquiries.json
  - 🧠 Dossiers  Local view of voice_inquiry_dossiers.json
  - 🔧 Raw I/O   Last request + response JSON for both endpoints
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import httpx
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from csf_recommendation_engine.core.config import get_settings


# ── Constants ──────────────────────────────────────────────────────────────

settings = get_settings()

EXAMPLE_IOIS: list[str] = [
    "Mike at Vitol buying 50 lots WTI Mar26 at 71.20, working order",
    "Trafigura looking to sell 25 lots Brent Apr26 spread, urgent",
    "Shell wants to lift 100 lots HO Aug26 at market, no time constraint",
    "Glencore selling 75 lots Nat Gas Feb26 at 3.45 or better",
    "Chevron buy 30 RBOB Jun26 spread, urgent",
]

TAG_KEYS: list[str] = [
    "side", "product", "instrument_name", "qty", "tenor", "price",
    "qualifier", "urgency", "sentiment", "structure", "desk", "venue",
]

SCORE_KEYS: list[str] = [
    "final_score", "lightfm_normalized", "size_score", "recency_score", "time_score",
]


# ── Tag comparison helpers (side-by-side colored diff) ─────────────────────

GREEN = "#22c55e"
RED = "#ef4444"
GRAY = "#9ca3af"


def _compare_tag(key: str, o: Any, m: Any) -> str:
    """Return 'match' | 'mismatch' | 'na' for one tag pair.

    Domain-aware:
      - 'side': OPPOSITE sides (Buy↔Sell) count as a MATCH, since that is
        what enables a trade. Same sides count as a mismatch.
      - 'qty':  match if the smaller is at least 50% of the larger (i.e.
        within roughly 2×).
      - 'price': match if the absolute difference is within 5% of the
        larger absolute value.
      - everything else: case-insensitive string equality.
    Missing/empty values on either side return 'na'.
    """
    if o is None or m is None or o == "" or m == "":
        return "na"
    if key == "side":
        no, nm = str(o).strip().lower(), str(m).strip().lower()
        return "match" if (no, nm) in {("buy", "sell"), ("sell", "buy")} else "mismatch"
    if key == "qty":
        try:
            qo, qm = float(o), float(m)
            if max(qo, qm) == 0:
                return "match"
            return "match" if (min(qo, qm) / max(qo, qm)) >= 0.5 else "mismatch"
        except (TypeError, ValueError):
            return "na"
    if key == "price":
        try:
            po, pm = float(o), float(m)
            denom = max(abs(po), abs(pm)) or 1.0
            return "match" if abs(po - pm) / denom <= 0.05 else "mismatch"
        except (TypeError, ValueError):
            return "na"
    return "match" if str(o).strip().lower() == str(m).strip().lower() else "mismatch"


def compare_tags(orig: dict, matched: dict) -> dict[str, str]:
    return {k: _compare_tag(k, orig.get(k), matched.get(k)) for k in TAG_KEYS}


def render_tag_card(key: str, value: Any, status: str) -> str:
    """HTML for one bordered tag card. Use with st.markdown(unsafe_allow_html=True)."""
    color = {"match": GREEN, "mismatch": RED, "na": GRAY}[status]
    icon = {"match": "✓ MATCH", "mismatch": "✗ MISMATCH", "na": "— N/A"}[status]
    display = "—" if value is None or value == "" else str(value)
    return (
        f'<div style="border:2px solid {color}; padding:6px 10px; '
        f'border-radius:6px; margin-bottom:6px; background-color: rgba(0,0,0,0.02);">'
        f'<div style="font-size:0.7rem; color:{color}; font-weight:700; letter-spacing:0.05em">{icon}</div>'
        f'<div style="margin-top:2px"><b>{key}:</b> {display}</div>'
        f'</div>'
    )


# ── HTTP helpers ───────────────────────────────────────────────────────────

def _post(base_url: str, path: str, payload: dict, *, timeout: float) -> tuple[httpx.Response, float]:
    url = f"{base_url.rstrip('/')}{path}"
    t0 = time.perf_counter()
    with httpx.Client(timeout=timeout) as client:
        r = client.post(url, json=payload)
    return r, time.perf_counter() - t0


def call_tags_endpoint(base_url: str, ioi_text: str, originator_entity_id: str):
    payload = {"ioi_text": ioi_text, "originator_entity_id": originator_entity_id}
    return _post(base_url, "/voice_inquiry/tags", payload, timeout=120.0), payload


def call_match_endpoint(
    base_url: str, ioi_text: str, originator_entity_id: str, top_k: int | None
):
    payload: dict[str, Any] = {"ioi_text": ioi_text, "originator_entity_id": originator_entity_id}
    if top_k:
        payload["top_k"] = int(top_k)
    return _post(base_url, "/voice_inquiry/match", payload, timeout=300.0), payload


def health_check(base_url: str) -> tuple[bool, str]:
    try:
        with httpx.Client(timeout=5.0) as client:
            r = client.get(f"{base_url.rstrip('/')}/health/live")
        return r.status_code == 200, f"HTTP {r.status_code} — {r.text[:120]}"
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)


# ── Disk helpers (admin tabs) ──────────────────────────────────────────────

def read_inquiries() -> list[dict[str, Any]]:
    p = Path(settings.voice_inquiries_path)
    if not p.exists():
        return []
    try:
        data = json.loads(p.read_text(encoding="utf-8") or "[]")
        return data if isinstance(data, list) else []
    except json.JSONDecodeError:
        return []


def write_inquiries(items: list[dict[str, Any]]) -> None:
    p = Path(settings.voice_inquiries_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(items, indent=2), encoding="utf-8")


def read_dossiers() -> dict[str, str]:
    p = Path(settings.voice_inquiry_dossiers_path)
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8") or "{}")
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        return {}


def write_dossiers(d: dict[str, str]) -> None:
    p = Path(settings.voice_inquiry_dossiers_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(d, indent=2), encoding="utf-8")


# ── Session state ──────────────────────────────────────────────────────────

st.set_page_config(page_title="Voice Inquiry Demo", layout="wide")

for k, default in (
    ("last_match_response", None),
    ("last_match_request", None),
    ("last_match_elapsed", None),
    ("last_tags_response", None),
    ("last_tags_request", None),
    ("last_tags_elapsed", None),
):
    if k not in st.session_state:
        st.session_state[k] = default


# ── Sidebar ────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("⚙️ Configuration")
    api_base = st.text_input("API base URL", value="http://127.0.0.1:8000")
    default_top_k = st.slider("Default top_k", 1, 10, int(settings.voice_inquiry_top_k))

    if st.button("🩺 Health check", use_container_width=True):
        ok, msg = health_check(api_base)
        (st.success if ok else st.error)(msg)

    st.divider()
    with st.expander("Configured agent IDs"):
        st.code(
            "tagger:      " + (settings.daf_tagger_agent_id or "(empty)") + "\n"
            "dossier:     " + (settings.daf_dossier_agent_id or "(empty)") + "\n"
            "matcher:     " + (settings.daf_matcher_agent_id or "(empty)") + "\n"
            "recommender: " + (settings.daf_recommender_agent_id or "(empty)")
        )
    with st.expander("Storage paths"):
        st.code(
            "inquiries: " + str(settings.voice_inquiries_path) + "\n"
            "dossiers:  " + str(settings.voice_inquiry_dossiers_path)
        )
    with st.expander("LLM/DAF connection"):
        st.code(
            "daf_base_url:      " + settings.daf_base_url + "\n"
            "llm_model:         " + f"{settings.llm_model_provider}/{settings.llm_model_name}" + "\n"
            "llm_temperature:   " + str(settings.llm_temperature)
        )


# ── Main ───────────────────────────────────────────────────────────────────

st.title("🎙️ Voice Inquiry Demo")
st.caption(
    "End-to-end demo of `/voice_inquiry/match` (Tagger → Ranker → Dossier → Matcher) "
    "and `/voice_inquiry/tags` (parser in isolation)."
)

tab_match, tab_tags, tab_inquiries, tab_dossiers, tab_raw = st.tabs(
    ["🎯 Match", "🏷️ Tags only", "📚 Inquiries", "🧠 Dossiers", "🔧 Raw I/O"]
)


# ═══════════════════════════════════════════════════════════════════════════
# Tab: full match pipeline
# ═══════════════════════════════════════════════════════════════════════════
with tab_match:
    st.subheader("Full match pipeline")
    st.caption(
        "Pipeline: Tagger agent → RecommendRequest → LightFM ranker + heuristics rerank → "
        "per-candidate dossier (cache or DAF) → Matcher agent picks best open inquiry → "
        "originator IOI is persisted to disk."
    )

    inquiries_on_disk = read_inquiries()
    known_entities = sorted({inq.get("entity_id") for inq in inquiries_on_disk if inq.get("entity_id")})

    col_in, col_cfg = st.columns([2, 1])
    with col_in:
        quick = st.selectbox(
            "Quick-pick example IOI",
            options=["(custom)"] + EXAMPLE_IOIS,
            index=1,
            key="match_quick_pick",
        )
        default_text = "" if quick == "(custom)" else quick
        ioi_text = st.text_area(
            "IOI text",
            height=120,
            value=default_text,
            key="match_ioi_text",
        )
    with col_cfg:
        st.markdown("**Originator**")
        st.caption("Excluded from candidate pool")
        if known_entities:
            choice = st.selectbox(
                "From known inquiries",
                options=known_entities + ["(custom)"],
                key="match_origin_choice",
            )
            originator_entity_id = (
                st.text_input("Custom originator ID", "", key="match_origin_custom")
                if choice == "(custom)"
                else choice
            )
        else:
            originator_entity_id = st.text_input("Originator entity ID", "", key="match_origin_only")
        top_k = st.number_input(
            "top_k", min_value=1, max_value=20, value=int(default_top_k), key="match_top_k"
        )

    can_run = bool((ioi_text or "").strip()) and bool((originator_entity_id or "").strip())
    if st.button("▶️ Run match pipeline", type="primary", disabled=not can_run, use_container_width=True):
        with st.spinner("Calling /voice_inquiry/match — Tagger → Ranker → Dossier → Matcher..."):
            try:
                (resp, elapsed), payload = call_match_endpoint(
                    api_base, ioi_text.strip(), originator_entity_id.strip(), int(top_k)
                )
                st.session_state.last_match_request = payload
                st.session_state.last_match_elapsed = elapsed
                if resp.status_code != 200:
                    st.session_state.last_match_response = {
                        "_error": True,
                        "status_code": resp.status_code,
                        "body": resp.text,
                    }
                else:
                    st.session_state.last_match_response = resp.json()
            except Exception as exc:  # noqa: BLE001
                st.session_state.last_match_response = {"_error": True, "exception": str(exc)}
                st.session_state.last_match_elapsed = None

    data = st.session_state.last_match_response
    if data is not None:
        st.divider()
        if data.get("_error"):
            st.error(
                f"Request failed: HTTP {data.get('status_code')} {data.get('body', '')} "
                f"{data.get('exception', '')}"
            )
        else:
            elapsed = st.session_state.last_match_elapsed or 0.0
            st.success(f"✅ Pipeline completed in {elapsed:.2f}s")

            # ── Best match header ─────────────────────────────────────────
            bm = data.get("best_match")
            if bm:
                pct = int(bm.get("match_percent") or 0)
                hcol1, hcol2 = st.columns([1, 4])
                with hcol1:
                    st.metric("Match", f"{pct}%")
                with hcol2:
                    st.markdown(
                        f"**Picked inquiry:** `{bm.get('best_match_inquiry_id')}` "
                        f"on entity `{bm.get('best_match_entity_id')}`"
                    )
                    st.progress(min(max(pct, 0), 100) / 100.0)
                with st.expander("🧠 Matcher reasoning", expanded=True):
                    st.write(bm.get("reasoning") or "_(empty reasoning)_")
            else:
                st.warning("No best_match returned — matcher failed, returned None, or no candidates.")

            # ── Side-by-side comparison ──────────────────────────────────
            matched_inq = data.get("matched_inquiry")
            appended = data.get("appended_inquiry") or {}
            if matched_inq and appended:
                st.subheader("🆚 Side-by-side comparison")
                st.caption(
                    "🟢 green = compatible for a trade • 🔴 red = incompatible • ⚪ gray = "
                    "either side missing. Rules: **side** matches when OPPOSITE (Buy↔Sell), "
                    "**qty** within ~2×, **price** within ±5%, all other tags exact (case-insensitive)."
                )
                orig_tags = appended.get("tags") or {}
                m_tags = matched_inq.get("tags") or {}
                statuses = compare_tags(orig_tags, m_tags)

                n_match = sum(1 for s in statuses.values() if s == "match")
                n_mismatch = sum(1 for s in statuses.values() if s == "mismatch")
                n_na = sum(1 for s in statuses.values() if s == "na")
                mc1, mc2, mc3, mc4 = st.columns(4)
                mc1.metric("🟢 Compatible", n_match)
                mc2.metric("🔴 Incompatible", n_mismatch)
                mc3.metric("⚪ N/A", n_na)
                mc4.metric("Tags compared", len(TAG_KEYS))

                left, right = st.columns(2)
                with left:
                    st.markdown("##### 📨 Your new inquiry")
                    st.caption(
                        f"`{appended.get('inquiry_id', '—')}` • entity "
                        f"`{appended.get('entity_id', '—')}` • "
                        f"{(appended.get('created_at') or '')[:19]}"
                    )
                    st.info(appended.get("ioi_text") or "_(no text)_")
                    for key in TAG_KEYS:
                        st.markdown(
                            render_tag_card(key, orig_tags.get(key), statuses[key]),
                            unsafe_allow_html=True,
                        )
                with right:
                    st.markdown("##### 🎯 Best-matched existing inquiry")
                    st.caption(
                        f"`{matched_inq.get('inquiry_id', '—')}` • entity "
                        f"`{matched_inq.get('entity_id', '—')}` • "
                        f"{(matched_inq.get('created_at') or '')[:19]}"
                    )
                    st.success(matched_inq.get("ioi_text") or "_(no text)_")
                    for key in TAG_KEYS:
                        st.markdown(
                            render_tag_card(key, m_tags.get(key), statuses[key]),
                            unsafe_allow_html=True,
                        )
            elif bm and not matched_inq:
                st.warning(
                    f"Matcher picked inquiry `{bm.get('best_match_inquiry_id')}` on entity "
                    f"`{bm.get('best_match_entity_id')}`, but it was not found in any "
                    "candidate's open inquiries (matcher may have hallucinated). "
                    "Side-by-side comparison unavailable."
                )

            # ── Parsed tags ───────────────────────────────────────────────
            tags = data.get("parsed_tags") or {}
            with st.expander("📋 Parsed tags from originator IOI", expanded=False):
                tcols = st.columns(3)
                for i, key in enumerate(TAG_KEYS):
                    with tcols[i % 3]:
                        val = tags.get(key)
                        st.text_input(
                            key,
                            value="" if val is None else str(val),
                            disabled=True,
                            key=f"match_tag_disp_{key}",
                        )
                st.json(tags)

            # ── Candidates ────────────────────────────────────────────────
            cands = data.get("candidates") or []
            st.subheader(f"🎯 Top-{len(cands)} candidates")
            best_inq_id = (bm or {}).get("best_match_inquiry_id")
            best_eid = (bm or {}).get("best_match_entity_id")

            for idx, cand in enumerate(cands, start=1):
                eid = cand.get("entity_id")
                ename = cand.get("entity_name", "Unknown")
                fscore = cand.get("final_score")
                fscore_str = f"{fscore:.4f}" if isinstance(fscore, (int, float)) else "—"
                is_winner = eid == best_eid
                label = f"#{idx} — {ename} (`{eid}`) — score {fscore_str}"
                if is_winner:
                    label = "🏆 " + label
                with st.expander(label, expanded=is_winner):
                    # Score breakdown
                    scols = st.columns(len(SCORE_KEYS))
                    for sc, key in zip(scols, SCORE_KEYS):
                        v = cand.get(key)
                        sc.metric(
                            key,
                            f"{v:.4f}" if isinstance(v, (int, float)) else "—",
                        )

                    # Dossier
                    st.markdown("**Dossier**")
                    dossier_text = cand.get("dossier_text")
                    if dossier_text:
                        st.write(dossier_text)
                    else:
                        st.caption("_(no dossier)_")

                    # Inquiries on file for this candidate
                    inquiries = cand.get("inquiries") or []
                    st.markdown(f"**Open inquiries ({len(inquiries)})**")
                    if not inquiries:
                        st.caption("_(none on file)_")
                    for inq in inquiries:
                        inq_id = inq.get("inquiry_id")
                        is_pick = inq_id == best_inq_id
                        prefix = "🎯 " if is_pick else "• "
                        with st.container(border=True):
                            st.markdown(f"{prefix}`{inq_id}`" + ("  — **picked**" if is_pick else ""))
                            st.text(inq.get("ioi_text") or "")
                            with st.expander("tags", expanded=False):
                                st.json(inq.get("tags") or {})

            # ── Persisted inquiry ─────────────────────────────────────────
            with st.expander("💾 Appended inquiry (now on disk)", expanded=False):
                st.json(data.get("appended_inquiry") or {})


# ═══════════════════════════════════════════════════════════════════════════
# Tab: tags-only
# ═══════════════════════════════════════════════════════════════════════════
with tab_tags:
    st.subheader("Tag parser in isolation")
    st.caption("Sends just the Tagger agent call — no ranking, no dossier, no matcher, nothing persisted.")

    quick_t = st.selectbox(
        "Quick-pick example IOI",
        options=["(custom)"] + EXAMPLE_IOIS,
        index=1,
        key="tags_quick",
    )
    default_t = "" if quick_t == "(custom)" else quick_t
    ioi_t = st.text_area("IOI text", height=120, value=default_t, key="tags_ioi")
    oid_t = st.text_input(
        "Originator entity ID (schema-required, unused by /tags)",
        value="DUMMY-TAGS-ONLY",
        key="tags_oid",
    )

    if st.button(
        "🏷️ Parse tags",
        type="primary",
        disabled=not (ioi_t or "").strip(),
        use_container_width=True,
        key="tags_run",
    ):
        with st.spinner("Calling /voice_inquiry/tags..."):
            try:
                (resp, elapsed), payload = call_tags_endpoint(
                    api_base, ioi_t.strip(), oid_t.strip() or "DUMMY-TAGS-ONLY"
                )
                st.session_state.last_tags_request = payload
                st.session_state.last_tags_elapsed = elapsed
                if resp.status_code != 200:
                    st.session_state.last_tags_response = {
                        "_error": True,
                        "status_code": resp.status_code,
                        "body": resp.text,
                    }
                else:
                    st.session_state.last_tags_response = resp.json()
            except Exception as exc:  # noqa: BLE001
                st.session_state.last_tags_response = {"_error": True, "exception": str(exc)}
                st.session_state.last_tags_elapsed = None

    tdata = st.session_state.last_tags_response
    if tdata is not None:
        st.divider()
        if tdata.get("_error"):
            st.error(
                f"Request failed: HTTP {tdata.get('status_code')} {tdata.get('body', '')} "
                f"{tdata.get('exception', '')}"
            )
        else:
            elapsed = st.session_state.last_tags_elapsed or 0.0
            st.success(f"✅ Tagger returned in {elapsed:.2f}s")
            tags = tdata.get("parsed_tags") or {}
            tcols = st.columns(3)
            for i, key in enumerate(TAG_KEYS):
                with tcols[i % 3]:
                    val = tags.get(key)
                    st.text_input(
                        key,
                        value="" if val is None else str(val),
                        disabled=True,
                        key=f"tags_disp_{key}",
                    )

            with st.expander("🤖 Agent prompt sent", expanded=False):
                st.code(tdata.get("prompt") or "", language="markdown")
            with st.expander("📜 Raw agent response", expanded=False):
                st.code(tdata.get("raw_response") or "", language="json")
            with st.expander("🛠 Full debug payload", expanded=False):
                st.json(tdata.get("debug") or {})


# ═══════════════════════════════════════════════════════════════════════════
# Tab: inquiries admin (local disk view)
# ═══════════════════════════════════════════════════════════════════════════
with tab_inquiries:
    st.subheader("Voice inquiries on disk")
    st.caption(f"Path: `{settings.voice_inquiries_path}`")

    inquiries = read_inquiries()
    m1, m2, m3 = st.columns(3)
    m1.metric("Total inquiries", len(inquiries))
    m2.metric("Distinct entities", len({inq.get("entity_id") for inq in inquiries}))
    m3.metric("Most recent", (inquiries[-1].get("created_at", "—")[:19] if inquiries else "—"))

    entity_ids = sorted({inq.get("entity_id", "") for inq in inquiries if inq.get("entity_id")})
    filt_eid = st.selectbox("Filter by entity_id", options=["(all)"] + entity_ids, key="inq_filter")
    filtered = inquiries if filt_eid == "(all)" else [i for i in inquiries if i.get("entity_id") == filt_eid]
    st.caption(f"Showing {len(filtered)} of {len(inquiries)}")

    for inq in reversed(filtered):
        inq_id = inq.get("inquiry_id", "")
        with st.expander(
            f"`{inq_id}` — entity={inq.get('entity_id', '?')} — "
            f"{(inq.get('created_at') or '')[:19]}"
        ):
            st.markdown("**IOI text:**")
            st.text(inq.get("ioi_text") or "")
            st.markdown("**Tags:**")
            st.json(inq.get("tags") or {})
            if st.button("🗑️ Delete from disk", key=f"del_inq_{inq_id}"):
                write_inquiries([i for i in inquiries if i.get("inquiry_id") != inq_id])
                st.toast(f"Deleted {inq_id}")
                st.rerun()


# ═══════════════════════════════════════════════════════════════════════════
# Tab: dossiers admin (local disk view)
# ═══════════════════════════════════════════════════════════════════════════
with tab_dossiers:
    st.subheader("Cached entity dossiers")
    st.caption(f"Path: `{settings.voice_inquiry_dossiers_path}`")

    dossiers = read_dossiers()
    m1, m2 = st.columns(2)
    m1.metric("Cached entities", len(dossiers))
    m2.metric("Total chars", sum(len(v) for v in dossiers.values()))

    if not dossiers:
        st.info("Cache is empty. Dossiers are built lazily on first `/voice_inquiry/match` call per entity.")

    for eid, text in dossiers.items():
        with st.expander(f"`{eid}` — {len(text)} chars"):
            st.write(text)
            cols = st.columns([1, 5])
            with cols[0]:
                if st.button("🗑️ Delete", key=f"del_dos_{eid}"):
                    write_dossiers({k: v for k, v in dossiers.items() if k != eid})
                    st.toast(f"Deleted dossier for {eid} — next match will rebuild it")
                    st.rerun()


# ═══════════════════════════════════════════════════════════════════════════
# Tab: raw I/O
# ═══════════════════════════════════════════════════════════════════════════
with tab_raw:
    st.subheader("Last request/response payloads")
    st.caption("Useful for copying into curl, Postman, or bug reports.")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### `/voice_inquiry/match`")
        if st.session_state.last_match_elapsed is not None:
            st.caption(f"Elapsed: {st.session_state.last_match_elapsed:.2f}s")
        st.markdown("**Request**")
        st.json(st.session_state.last_match_request or {})
        st.markdown("**Response**")
        st.json(st.session_state.last_match_response or {})
    with c2:
        st.markdown("### `/voice_inquiry/tags`")
        if st.session_state.last_tags_elapsed is not None:
            st.caption(f"Elapsed: {st.session_state.last_tags_elapsed:.2f}s")
        st.markdown("**Request**")
        st.json(st.session_state.last_tags_request or {})
        st.markdown("**Response**")
        st.json(st.session_state.last_tags_response or {})
