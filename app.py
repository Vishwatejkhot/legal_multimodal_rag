import sys
import datetime
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st
from image_pipeline.image_handler import handle_uploaded_bytes
from agent.react_agent import run_stream
from agent.timeline_extractor import extract_timeline, flag_deadline_issues
from scorer.xgboost_predictor import predict_evidence_strength
from config import FAISS_INDEX_PATH, BM25_INDEX_PATH, XGBOOST_MODEL_PATH

st.set_page_config(
    page_title="JusticeAI — UK Legal Evidence Analysis",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
/* ── Reset & Base ── */
section[data-testid="stMain"] { background: #F7F8FC !important; }
section[data-testid="stSidebar"] { background: #1B2B6B !important; border-right: none; }
section[data-testid="stSidebar"] * { color: #E8EDF8 !important; }
section[data-testid="stSidebar"] .stMarkdown p { color: #B8C4E0 !important; }
section[data-testid="stSidebar"] hr { border-color: #2D4099 !important; }
div[data-testid="stFileUploaderDropzone"] { background: #F0F4FF !important; border: 2px dashed #C5D0F0 !important; border-radius: 10px !important; }
div[data-testid="stFileUploaderDropzone"]:hover { border-color: #1B2B6B !important; background: #E8EFFE !important; }
.stTextArea textarea { border: 1.5px solid #D1D8F0 !important; border-radius: 10px !important; background: #FAFBFF !important; font-size: 0.95rem !important; resize: none !important; }
.stTextArea textarea:focus { border-color: #1B2B6B !important; box-shadow: 0 0 0 3px rgba(27,43,107,0.1) !important; }
button[kind="primary"] { background: #1B2B6B !important; border-radius: 10px !important; font-weight: 600 !important; letter-spacing: 0.02em !important; padding: 0.65rem 2rem !important; font-size: 1rem !important; border: none !important; transition: background 0.2s !important; }
button[kind="primary"]:hover { background: #243792 !important; }
.stTabs [data-baseweb="tab-list"] { background: #ECEFFE; border-radius: 10px; padding: 4px; gap: 2px; }
.stTabs [data-baseweb="tab"] { border-radius: 8px; font-weight: 500; color: #4A5568; padding: 8px 20px; }
.stTabs [aria-selected="true"] { background: #1B2B6B !important; color: white !important; }
.stExpander { border: 1px solid #E2E8F0 !important; border-radius: 10px !important; }
div[data-testid="stAlert"] { border-radius: 10px !important; }
footer { display: none !important; }
#MainMenu { display: none !important; }
header[data-testid="stHeader"] { background: transparent !important; }

/* ── Header ── */
.jai-hero {
    background: linear-gradient(135deg, #1B2B6B 0%, #1E3A8A 50%, #1D4ED8 100%);
    border-radius: 16px;
    padding: 32px 40px;
    margin-bottom: 28px;
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.jai-hero-left h1 { color: white; font-size: 2.2rem; font-weight: 800; margin: 0 0 6px 0; letter-spacing: -0.02em; }
.jai-hero-left p  { color: #93C5FD; font-size: 0.95rem; margin: 0; }
.jai-hero-badges  { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 16px; }
.jai-badge { background: rgba(255,255,255,0.12); color: white; border: 1px solid rgba(255,255,255,0.2); border-radius: 20px; padding: 4px 12px; font-size: 0.78rem; font-weight: 500; }

/* ── Section labels ── */
.section-label { font-size: 0.75rem; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; color: #6B7CB8; margin-bottom: 8px; }

/* ── Cards ── */
.jai-card { background: white; border: 1px solid #E2E8F0; border-radius: 14px; padding: 24px; box-shadow: 0 1px 3px rgba(0,0,0,0.04), 0 4px 12px rgba(27,43,107,0.04); }

/* ── Strength badge ── */
.strength-wrap { border-radius: 14px; padding: 24px; text-align: center; color: white; }
.strength-wrap .s-label { font-size: 0.72rem; font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase; opacity: 0.85; margin-bottom: 8px; }
.strength-wrap .s-value { font-size: 3rem; font-weight: 800; line-height: 1; }
.strength-wrap .s-conf  { font-size: 0.85rem; opacity: 0.8; margin-top: 8px; }
.s-high   { background: linear-gradient(135deg, #065F46, #059669); }
.s-medium { background: linear-gradient(135deg, #92400E, #D97706); }
.s-low    { background: linear-gradient(135deg, #7F1D1D, #DC2626); }
.s-unknown{ background: linear-gradient(135deg, #374151, #6B7280); }

/* ── Conflict alert ── */
.conflict-alert { background: #FFF5F5; border: 1px solid #FED7D7; border-left: 4px solid #E53E3E; border-radius: 10px; padding: 16px 20px; margin-bottom: 16px; }
.conflict-alert .ca-title { font-weight: 700; color: #C53030; font-size: 0.82rem; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 10px; }
.conflict-alert .ca-item { color: #742A2A; font-size: 0.88rem; padding: 4px 0; border-bottom: 1px solid #FEB2B2; }
.conflict-alert .ca-item:last-child { border-bottom: none; }

/* ── Source chip ── */
.chip { display: inline-flex; align-items: center; gap: 5px; background: #EEF2FF; color: #3730A3; border-radius: 6px; padding: 4px 10px; font-size: 0.78rem; font-weight: 500; margin: 3px; }
.chip-case { background: #F0FDF4; color: #166534; }
.chip-cpr  { background: #FFFBEB; color: #92400E; }
.chip-sent { background: #FDF4FF; color: #6B21A8; }

/* ── Chunk card ── */
.chunk { background: white; border: 1px solid #E2E8F0; border-radius: 10px; padding: 14px 16px; margin-bottom: 10px; transition: box-shadow 0.15s; }
.chunk:hover { box-shadow: 0 4px 12px rgba(27,43,107,0.08); }
.chunk-h  { border-left: 3px solid #059669; }
.chunk-m  { border-left: 3px solid #D97706; }
.chunk-l  { border-left: 3px solid #DC2626; }
.chunk-top { display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 8px; gap: 8px; }
.chunk-src { font-size: 0.76rem; color: #4A5568; flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-weight: 500; }
.chunk-pct { font-size: 0.72rem; font-weight: 700; padding: 2px 8px; border-radius: 4px; white-space: nowrap; flex-shrink: 0; }
.cp-h { background: #D1FAE5; color: #065F46; }
.cp-m { background: #FEF3C7; color: #92400E; }
.cp-l { background: #FEE2E2; color: #7F1D1D; }
.chunk-txt { font-size: 0.84rem; color: #2D3748; line-height: 1.65; }
.tag-img    { font-size: 0.7rem; background: #DBEAFE; color: #1E40AF; padding: 1px 6px; border-radius: 4px; margin-right: 5px; }
.tag-rerank { font-size: 0.7rem; background: #EDE9FE; color: #5B21B6; padding: 1px 6px; border-radius: 4px; margin-right: 5px; }

/* ── Step progress ── */
.step-row  { display: flex; align-items: center; gap: 12px; padding: 7px 0; font-size: 0.88rem; color: #2D3748; }
.dot-done  { width: 10px; height: 10px; border-radius: 50%; background: #059669; flex-shrink: 0; }
.dot-run   { width: 10px; height: 10px; border-radius: 50%; background: #D97706; flex-shrink: 0; animation: blink 1s ease-in-out infinite; }
.dot-wait  { width: 10px; height: 10px; border-radius: 50%; background: #CBD5E0; flex-shrink: 0; }
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.3} }

/* ── Timeline ── */
.tl-row { display: flex; gap: 14px; }
.tl-dot-col { display: flex; flex-direction: column; align-items: center; }
.tl-dot { width: 12px; height: 12px; border-radius: 50%; background: #1B2B6B; margin-top: 3px; flex-shrink: 0; }
.tl-dot-warn { background: #D97706; }
.tl-line { width: 2px; background: #E2E8F0; flex: 1; margin: 4px 0; min-height: 20px; }
.tl-body { flex: 1; padding-bottom: 20px; }
.tl-date  { font-size: 0.72rem; font-weight: 700; color: #1B2B6B; letter-spacing: 0.06em; text-transform: uppercase; }
.tl-event { font-size: 0.9rem; color: #2D3748; margin-top: 2px; }
.tl-legal { font-size: 0.8rem; color: #D97706; margin-top: 3px; font-weight: 500; }
.deadline-breach { background: #FFF5F5; border: 1px solid #FED7D7; border-radius: 10px; padding: 14px 16px; margin-bottom: 12px; }
.deadline-ok     { background: #F0FDF4; border: 1px solid #BBF7D0; border-radius: 10px; padding: 14px 16px; margin-bottom: 12px; }

/* ── Sidebar elements ── */
.sb-logo { font-size: 1.4rem; font-weight: 800; color: white !important; letter-spacing: -0.01em; }
.sb-sub  { font-size: 0.78rem; color: #93C5FD !important; margin-top: 2px; }
.sb-section { font-size: 0.68rem; font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase; color: #6B8FCE !important; margin: 16px 0 8px 0; }
.sb-status  { display: flex; align-items: center; gap: 8px; padding: 8px 12px; background: rgba(255,255,255,0.07); border-radius: 8px; margin-bottom: 6px; font-size: 0.83rem; }
.sb-dot-g   { width: 8px; height: 8px; border-radius: 50%; background: #34D399; flex-shrink: 0; }
.sb-dot-r   { width: 8px; height: 8px; border-radius: 50%; background: #F87171; flex-shrink: 0; }
.sb-dot-y   { width: 8px; height: 8px; border-radius: 50%; background: #FBBF24; flex-shrink: 0; }
.sb-hist { background: rgba(255,255,255,0.07); border-radius: 8px; padding: 10px 12px; margin-bottom: 6px; border-left: 3px solid #4F6ECC; }
.sb-hist-q { font-size: 0.82rem; color: #E8EDF8 !important; font-weight: 500; }
.sb-hist-m { font-size: 0.72rem; color: #93AED8 !important; margin-top: 3px; }
.sb-step { font-size: 0.82rem; color: #B8C4E0 !important; padding: 3px 0; }
</style>
""", unsafe_allow_html=True)

# ── Session state ──────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []
if "last_result" not in st.session_state:
    st.session_state.last_result = None

# ── Helpers ────────────────────────────────────────────────────────────────────
def _indexes_ready():
    return Path(FAISS_INDEX_PATH).exists() and Path(BM25_INDEX_PATH).exists()

def _xgboost_ready():
    return Path(XGBOOST_MODEL_PATH).exists()

def _strength_cls(label):
    return {"High": "s-high", "Medium": "s-medium", "Low": "s-low"}.get(label, "s-unknown")

def _chunk_cls(conf):
    if conf >= 70: return "chunk-h", "cp-h"
    if conf >= 40: return "chunk-m", "cp-m"
    return "chunk-l", "cp-l"

def _file_icon(name):
    ext = Path(name).suffix.lower()
    return {"pdf": "📄", ".pdf": "📄", ".docx": "📝",
            ".jpg": "🖼", ".jpeg": "🖼", ".png": "🖼", ".webp": "🖼"}.get(ext, "📎")

def _source_chip(src):
    if "Case Law" in src or "BAILII" in src:
        return f'<span class="chip chip-case">⚖ {src[:55]}</span>'
    if "CPR" in src or "HMCTS" in src:
        return f'<span class="chip chip-cpr">📜 {src[:55]}</span>'
    if "Sentencing" in src:
        return f'<span class="chip chip-sent">⚖ {src[:55]}</span>'
    return f'<span class="chip">📋 {src[:55]}</span>'

def _export(query, result, strength):
    ts = datetime.datetime.now().strftime("%d %B %Y %H:%M")
    lines = [f"JUSTICEAI ANALYSIS REPORT — {ts}", "=" * 60,
             f"\nQUERY:\n{query}\n",
             f"EVIDENCE STRENGTH: {strength['label']} ({strength['confidence']}% confidence)\n"]
    if result["conflicts"]:
        lines.append("CONFLICTS:\n" + "\n".join(f"• {c}" for c in result["conflicts"]) + "\n")
    lines.append("ANALYSIS:\n" + result["answer"])
    lines.append("\nSOURCES:\n" + "\n".join(f"• {s}" for s in result["sources"]))
    return "\n".join(lines)


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sb-logo">⚖️ JusticeAI</div>', unsafe_allow_html=True)
    st.markdown('<div class="sb-sub">UK Legal Evidence Analysis</div>', unsafe_allow_html=True)
    st.divider()

    st.markdown('<div class="sb-section">System Status</div>', unsafe_allow_html=True)
    idx_ok = _indexes_ready()
    xg_ok  = _xgboost_ready()
    idx_dot = "sb-dot-g" if idx_ok else "sb-dot-r"
    xg_dot  = "sb-dot-g" if xg_ok  else "sb-dot-y"
    st.markdown(f'<div class="sb-status"><div class="{idx_dot}"></div>Legal index {"ready" if idx_ok else "not built"}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sb-status"><div class="{xg_dot}"></div>XGBoost model {"ready" if xg_ok else "not trained"}</div>', unsafe_allow_html=True)
    if not idx_ok:
        st.caption("Run: `python scripts/build_index.py`")

    st.divider()
    st.markdown('<div class="sb-section">How to Use</div>', unsafe_allow_html=True)
    for step in ["1. Upload evidence files (PDF, images, letters)", "2. Type your legal question", "3. Click Analyse Evidence", "4. Review law, conflicts & timeline"]:
        st.markdown(f'<div class="sb-step">{step}</div>', unsafe_allow_html=True)

    st.divider()
    if st.session_state.history:
        st.markdown('<div class="sb-section">Recent Analyses</div>', unsafe_allow_html=True)
        for h in reversed(st.session_state.history[-5:]):
            st.markdown(f"""
<div class="sb-hist">
  <div class="sb-hist-q">{h['query'][:65]}{"…" if len(h['query'])>65 else ""}</div>
  <div class="sb-hist-m">{h['time']} · {h['strength']} · {h['files']} file(s)</div>
</div>""", unsafe_allow_html=True)
        if st.button("Clear history", use_container_width=True):
            st.session_state.history = []
            st.rerun()

    st.divider()
    st.markdown('<div style="font-size:0.72rem;color:#6B8FCE;text-align:center">Processing is local · No data stored externally</div>', unsafe_allow_html=True)


# ── Hero header ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="jai-hero">
  <div class="jai-hero-left">
    <h1>⚖️ JusticeAI</h1>
    <p>Multimodal legal evidence analysis for UK caseworkers</p>
    <div class="jai-hero-badges">
      <span class="jai-badge">GPT-4o Vision</span>
      <span class="jai-badge">FAISS + BM25 + RRF</span>
      <span class="jai-badge">Cross-Encoder Re-ranking</span>
      <span class="jai-badge">XGBoost Evidence Scoring</span>
      <span class="jai-badge">UK Legislation · Case Law · CPR</span>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)


# ── Input form ─────────────────────────────────────────────────────────────────
col_upload, col_gap, col_query = st.columns([10, 1, 10])

with col_upload:
    st.markdown('<div class="section-label">📁 Upload Evidence Files</div>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "upload",
        type=["jpg", "jpeg", "png", "webp", "pdf", "docx"],
        accept_multiple_files=True,
        label_visibility="collapsed",
        help="Accepts photos, scanned letters, PDFs, Word documents",
    )
    if uploaded_files:
        st.markdown(
            "".join(f'<span class="chip">{_file_icon(f.name)} {f.name}</span>' for f in uploaded_files),
            unsafe_allow_html=True,
        )
        st.caption(f"{len(uploaded_files)} file(s) ready")

with col_query:
    st.markdown('<div class="section-label">💬 Legal Question</div>', unsafe_allow_html=True)
    query = st.text_area(
        "query",
        placeholder=(
            "e.g. Does this eviction notice comply with Section 21 of the Housing Act 1988?\n\n"
            "e.g. Does this evidence support a housing disrepair claim?"
        ),
        height=148,
        label_visibility="collapsed",
    )
    with st.expander("Example questions"):
        examples = [
            "Is this Section 21 notice legally valid?",
            "Does this support a disrepair claim under the Landlord and Tenant Act 1985?",
            "Is the deposit protection compliant with the Deregulation Act 2015?",
            "What are the tenant's rights given this evidence?",
        ]
        for q in examples:
            if st.button(q, key=q[:30], use_container_width=True):
                st.session_state["_preset"] = q
                st.rerun()

if "_preset" in st.session_state and not query.strip():
    query = st.session_state.pop("_preset")

st.markdown("<br>", unsafe_allow_html=True)
analyse_btn = st.button("🔍  Analyse Evidence", type="primary", use_container_width=True)


# ── Analysis ───────────────────────────────────────────────────────────────────
if analyse_btn:
    if not query.strip():
        st.error("Please enter a legal question.")
        st.stop()
    if not _indexes_ready():
        st.error("Legal index not built — run `python scripts/build_index.py` first.")
        st.stop()

    extra_chunks = []
    step_ph = st.empty()

    def show_steps(steps):
        dot_map = {"done": "dot-done", "run": "dot-run", "wait": "dot-wait"}
        rows = "".join(
            f'<div class="step-row"><div class="{dot_map[s]}"></div><span>{l}</span></div>'
            for l, s in steps
        )
        step_ph.markdown(f'<div class="jai-card" style="margin-top:16px">{rows}</div>', unsafe_allow_html=True)

    steps = [
        ("Processing uploaded files", "run"),
        ("Searching legal index  (FAISS + BM25 + RRF + Cross-Encoder)", "wait"),
        ("Detecting evidence conflicts", "wait"),
        ("Generating answer (streaming)", "wait"),
        ("Scoring evidence strength (XGBoost)", "wait"),
    ]
    show_steps(steps)

    # Step 1 — files
    if uploaded_files:
        for f in uploaded_files:
            try:
                chunks = handle_uploaded_bytes(f.read(), f.name, question=query)
                extra_chunks.extend(chunks)
            except Exception as e:
                st.warning(f"Could not read **{f.name}**: {e}")

    steps[0] = (f"Files processed — {len(extra_chunks)} chunks extracted", "done")
    steps[1] = ("Searching legal index  (FAISS + BM25 + RRF + Cross-Encoder)", "run")
    show_steps(steps)

    # Step 2-3 — retrieve + conflicts
    try:
        result, answer_stream = run_stream(query, extra_chunks=extra_chunks or None)
    except Exception as e:
        st.error(f"Analysis error: {e}")
        st.stop()

    steps[1] = (f"Legal index searched — {len(result['chunks'])} chunks re-ranked", "done")
    steps[2] = (f"Conflict detection — {len(result['conflicts'])} conflict(s) found", "done")
    steps[3] = ("Generating answer (streaming)", "run")
    show_steps(steps)

    # Step 4 — stream answer
    st.markdown("---")
    st.markdown("#### ✍️ Generating Answer")
    try:
        full_answer = st.write_stream(answer_stream)
    except Exception as e:
        st.error(f"Streaming error: {e}")
        full_answer = ""
    result["answer"] = full_answer
    st.markdown("---")

    steps[3] = ("Answer generated", "done")
    steps[4] = ("Scoring evidence strength (XGBoost)", "run")
    show_steps(steps)

    # Step 5 — XGBoost
    try:
        strength = predict_evidence_strength(result)
    except Exception:
        strength = {"label": "Unknown", "confidence": 0, "probabilities": {}}

    # Timeline
    try:
        timeline_events = extract_timeline(result["chunks"])
        all_text = " ".join(c["text"] for c in result["chunks"])
        deadline_issues = flag_deadline_issues(timeline_events, all_text)
    except Exception:
        timeline_events, deadline_issues = [], []

    steps[4] = (f"Evidence strength: {strength['label']} ({strength['confidence']}% confidence)", "done")
    show_steps(steps)

    st.session_state.last_result = {
        "query": query, "result": result, "strength": strength,
        "timeline": timeline_events, "deadline_issues": deadline_issues,
        "files": [f.name for f in (uploaded_files or [])],
    }
    st.session_state.history.append({
        "query": query, "strength": strength["label"],
        "files": len(uploaded_files) if uploaded_files else 0,
        "time": datetime.datetime.now().strftime("%H:%M"),
    })


# ── Results ────────────────────────────────────────────────────────────────────
cached = st.session_state.last_result
if cached:
    result          = cached["result"]
    strength        = cached["strength"]
    query_text      = cached["query"]
    files           = cached.get("files", [])
    timeline_events = cached.get("timeline", [])
    deadline_issues = cached.get("deadline_issues", [])

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### Analysis Results")

    r1, r2, r3 = st.columns([2, 5, 3])

    with r1:
        sc = _strength_cls(strength["label"])
        st.markdown(f"""
<div class="strength-wrap {sc}">
  <div class="s-label">Evidence Strength</div>
  <div class="s-value">{strength["label"]}</div>
  <div class="s-conf">{strength["confidence"]}% confidence</div>
</div>""", unsafe_allow_html=True)

    with r2:
        st.markdown(f'<div class="jai-card"><div class="section-label">Query</div><div style="font-size:1rem;font-weight:600;color:#1B2B6B;line-height:1.5">{query_text}</div></div>', unsafe_allow_html=True)
        if files:
            st.markdown(
                '<div class="jai-card" style="margin-top:12px"><div class="section-label">Evidence Files</div>' +
                "".join(f'<span class="chip">{_file_icon(fn)} {fn}</span>' for fn in files) +
                "</div>", unsafe_allow_html=True,
            )

    with r3:
        probs = strength.get("probabilities", {})
        if probs:
            st.markdown('<div class="section-label">Strength Breakdown</div>', unsafe_allow_html=True)
            colors = {"High": "#059669", "Medium": "#D97706", "Low": "#DC2626"}
            for lvl, pct in probs.items():
                st.markdown(f'<div style="display:flex;justify-content:space-between;font-size:0.82rem;margin-bottom:2px"><span>{lvl}</span><b>{pct:.0f}%</b></div>', unsafe_allow_html=True)
                st.markdown(f'<div style="background:#E2E8F0;border-radius:4px;height:6px;margin-bottom:8px"><div style="background:{colors.get(lvl,"#6B7280")};width:{pct}%;height:6px;border-radius:4px"></div></div>', unsafe_allow_html=True)

    if result["conflicts"]:
        items = "".join(f'<div class="ca-item">• {c}</div>' for c in result["conflicts"])
        st.markdown(f'<div class="conflict-alert"><div class="ca-title">⚠ {len(result["conflicts"])} Conflict(s) Detected in Evidence</div>{items}</div>', unsafe_allow_html=True)

    if result.get("low_confidence_count", 0) > 0:
        st.warning(f"⚠️ {result['low_confidence_count']} chunk(s) below 40% confidence — treat related conclusions with caution.")

    tab1, tab2, tab3, tab4 = st.tabs(["📋 Legal Analysis", "📚 Sources & Law", "🔍 Evidence Chunks", "📅 Timeline"])

    with tab1:
        st.markdown(result["answer"])
        st.markdown("---")
        st.download_button(
            "⬇️ Download Full Report",
            data=_export(query_text, result, strength),
            file_name=f"justiceai_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.txt",
            mime="text/plain",
            use_container_width=True,
        )

    with tab2:
        if not result["sources"]:
            st.info("No sources retrieved. Ensure `build_index.py` has been run.")
        else:
            st.markdown('<div class="section-label">All Sources</div>', unsafe_allow_html=True)
            st.markdown("".join(_source_chip(s) for s in result["sources"]), unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            groups = {
                "UK Legislation": [s for s in result["sources"] if "Legislation" in s or "Act" in s or "ukpga" in s],
                "Case Law":       [s for s in result["sources"] if "Case Law" in s],
                "Court Procedure":[s for s in result["sources"] if "CPR" in s or "HMCTS" in s],
                "Sentencing":     [s for s in result["sources"] if "Sentencing" in s],
            }
            for grp, srcs in groups.items():
                if srcs:
                    st.markdown(f"**{grp}**")
                    for s in srcs:
                        st.markdown(f"- `{s}`")

    with tab3:
        chunks = result.get("chunks", [])
        if not chunks:
            st.info("No evidence chunks available.")
        else:
            st.caption(f"{len(chunks)} chunks retrieved, re-ranked, and scored")
            for chunk in chunks[:8]:
                conf = chunk.get("confidence", 0)
                bc, pc = _chunk_cls(conf)
                img_tag    = '<span class="tag-img">📷 image</span>' if chunk.get("from_image") else ""
                rerank_tag = f'<span class="tag-rerank">↑ {chunk["rerank_score"]:.2f}</span>' if "rerank_score" in chunk else ""
                src     = chunk.get("source", "unknown")[:75]
                preview = chunk["text"][:480] + ("…" if len(chunk["text"]) > 480 else "")
                st.markdown(f"""
<div class="chunk {bc}">
  <div class="chunk-top">
    <div class="chunk-src">{img_tag}{rerank_tag}{src}</div>
    <div class="chunk-pct {pc}">{conf:.0f}%</div>
  </div>
  <div class="chunk-txt">{preview}</div>
</div>""", unsafe_allow_html=True)

    with tab4:
        if not timeline_events:
            st.info("No dates could be extracted from this evidence.")
        else:
            if deadline_issues:
                st.markdown('<div class="section-label">Statutory Deadline Checks</div>', unsafe_allow_html=True)
                for issue in deadline_issues:
                    cls    = "deadline-breach" if issue["breach"] else "deadline-ok"
                    icon   = "⚠️" if issue["breach"] else "✅"
                    verdict= "BREACH" if issue["breach"] else "Compliant"
                    st.markdown(f"""
<div class="{cls}">
  <b>{icon} {verdict}</b> — {issue["rule"]}<br>
  <span style="font-size:0.83rem;color:#555">{issue["start"]} → {issue["end"]} = <b>{issue["gap_days"]} days</b> (required: {issue["required_days"]})</span>
</div>""", unsafe_allow_html=True)

            st.markdown(f'<div class="section-label">{len(timeline_events)} Event(s) Extracted</div>', unsafe_allow_html=True)
            for i, ev in enumerate(timeline_events):
                is_last  = i == len(timeline_events) - 1
                has_note = bool(ev.get("legal_note"))
                dot_cls  = "tl-dot-warn" if has_note else "tl-dot"
                line_html= "" if is_last else '<div class="tl-line"></div>'
                st.markdown(f"""
<div class="tl-row">
  <div class="tl-dot-col"><div class="tl-dot {dot_cls}"></div>{line_html}</div>
  <div class="tl-body">
    <div class="tl-date">{ev["date_str"]}</div>
    <div class="tl-event">{ev["event"]}</div>
    {f'<div class="tl-legal">⚠ {ev["legal_note"]}</div>' if has_note else ""}
  </div>
</div>""", unsafe_allow_html=True)
