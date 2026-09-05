"""
Remunilytics — prospect portal.

Token-gated, single-company view with anonymised peer context and
click-through-to-source on every disclosed figure.

Run locally:
    streamlit run portal/app.py
Then open:  http://localhost:8501/?token=<token from portal/tokens.json>
"""

import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go

from data_layer import (
    load_all, resolve_token, parse_source_link, peer_alias_map, anonymise,
    scope, latest_grant_year, latest_per_company, provenance_summary,
    dedupe_duplicate_plans, exclude_deferred_bonus_plans,
    exclude_buyout_replacement_awards, parse_fiscal_year,
    prefer_latest_ar_vintage, TIER_LABEL,
)
from source_render import has_box, render_citation

st.set_page_config(
    page_title="Remunilytics",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  #MainMenu, footer, [data-testid="stToolbar"] {visibility: hidden;}
  [data-testid="stHeader"] {height: 0; min-height: 0; visibility: hidden;}
  .block-container {padding-top: 2.2rem; max-width: 1180px;}
  .rl-brand {font-size:.72rem; letter-spacing:.18em; text-transform:uppercase;
             color:#8A94A6; font-weight:600;}
  .rl-title {font-size:2.05rem; font-weight:700; line-height:1.15; margin:.1rem 0 .2rem 0;}
  .rl-sub   {color:#5A6478; font-size:.95rem; margin-bottom:.4rem;}
  .rl-card  {background:#FFFFFF; border:1px solid #E6E9EF; border-left:3px solid #1F4E79;
             border-radius:8px; padding:1rem 1.1rem; height:100%;}
  .rl-card h4 {margin:0 0 .35rem 0; font-size:.72rem; letter-spacing:.09em;
               text-transform:uppercase; color:#8A94A6; font-weight:700;}
  .rl-big   {font-size:1.85rem; font-weight:700; color:#1F4E79; line-height:1.1;}
  .rl-note  {color:#5A6478; font-size:.85rem; margin-top:.3rem;}
  .rl-flag  {border-left-color:#C55A11 !important;}
  .rl-flag .rl-big {color:#C55A11;}
  .rl-pill  {display:inline-block; padding:.12rem .5rem; border-radius:99px;
             font-size:.68rem; font-weight:600; letter-spacing:.03em;}
  .rl-t3 {background:#E8F3EA; color:#2F6B3A;}
  .rl-t2 {background:#EAF0F8; color:#2E5E93;}
  .rl-t1 {background:#F3F1EA; color:#8A7A3A;}
  .rl-src a {text-decoration:none; font-size:.82rem; color:#2E75B6; font-weight:600;}
  .rl-src a:hover {text-decoration:underline;}
  .rl-foot {color:#8A94A6; font-size:.78rem; border-top:1px solid #E6E9EF;
            padding-top:.8rem; margin-top:2rem;}
</style>
""", unsafe_allow_html=True)

OWN = "#1F4E79"
PEER = "#B9C4D4"
ACCENT = "#C55A11"


# ── Helpers ───────────────────────────────────────────────────────────────────
def card(title, big, note="", flag=False):
    cls = "rl-card rl-flag" if flag else "rl-card"
    st.markdown(
        f'<div class="{cls}"><h4>{title}</h4>'
        f'<div class="rl-big">{big}</div>'
        f'<div class="rl-note">{note}</div></div>',
        unsafe_allow_html=True,
    )


def src_link(value, prefix="Source: "):
    """Render a click-to-source link with a precision badge."""
    url, label = parse_source_link(value)
    if not url:
        return ""
    return f'<span class="rl-src">{prefix}<a href="{url}" target="_blank">{label} ↗</a></span>'


def source_view(row, extraction_type: str, key_suffix: str = ""):
    """Show the exact highlighted block of the report a row was taken from.

    Renders nothing unless a Tier 3 citation with a resolved bounding box exists
    for this row, so tabs degrade quietly to their page-level source link.
    """
    fname = row.get("file_name")
    cid = row.get("source_chunk_id")
    if not (pd.notna(fname) and has_box(str(fname), cid, extraction_type)):
        return
    key = f"fullpage_{extraction_type}_{cid}_{row.name}{key_suffix}"
    with st.expander("See this in the Annual Report"):
        crop_path, pg = render_citation(str(fname), cid, extraction_type,
                                        crop=True, zoom=2.2)
        if crop_path:
            st.image(crop_path,
                     caption=f"Extracted from page {pg} of the Annual Report",
                     width="stretch")
        # Flat checkbox rather than a nested expander (Streamlit disallows
        # expanders inside expanders).
        if st.checkbox("Show the full page for context", key=key):
            full_path, _ = render_citation(str(fname), cid, extraction_type,
                                           crop=False, zoom=1.7)
            if full_path:
                st.image(full_path, width="stretch")


def tier_pill(tier):
    if tier not in TIER_LABEL:
        return ""
    name, tip = TIER_LABEL[tier]
    cls = {3: "rl-t3", 2: "rl-t2", 1: "rl-t1"}[tier]
    return f'<span class="rl-pill {cls}" title="{tip}">{name}</span>'


def fmt_money(v, ccy="GBP"):
    if v is None or pd.isna(v):
        return "n/a"
    sym = {"GBP": "£", "USD": "$", "EUR": "€"}.get(str(ccy).upper(), "")
    v = float(v)
    return f"{sym}{v/1_000_000:.2f}m" if abs(v) >= 1_000_000 else f"{sym}{v/1_000:,.0f}k"


def fmt_pct(v, dp=0):
    return "n/a" if v is None or pd.isna(v) else f"{float(v):.{dp}f}%"


def pctile(value, arr):
    arr = np.asarray([a for a in arr if pd.notna(a)], dtype=float)
    if len(arr) == 0 or pd.isna(value):
        return None
    return round(100.0 * (arr < float(value)).sum() / len(arr))


def ordinal(n):
    if n is None:
        return "n/a"
    n = int(n)
    suf = "th" if 11 <= n % 100 <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suf}"


def canonical_label(m):
    return {
        "eps": "EPS", "revenue": "Revenue", "return_on_capital": "Return on Capital",
        "tsr_relative": "TSR (Relative)", "tsr_absolute": "TSR (Absolute)",
        "cashflow": "Cash Flow", "cash_conversion": "Cash Conversion",
        "esg": "ESG / Sustainability", "strategic": "Strategic", "margin": "Margin",
        "ebit": "EBIT", "ebitda": "EBITDA", "pbt": "PBT", "pat": "PAT",
        "rote": "RoTE", "cet1_ratio": "CET1", "nav_per_share": "NAV/share",
        "net_interest_income": "Net Interest Income", "other": "Other",
        "restricted_time_based": "Restricted (time-based)",
    }.get(str(m), str(m).replace("_", " ").title())


# ── Token gate ────────────────────────────────────────────────────────────────
params = st.query_params
token = params.get("token")
if isinstance(token, list):
    token = token[0] if token else None

cfg = resolve_token(token)
if not cfg:
    st.markdown('<div class="rl-brand">Remunilytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="rl-title">This link isn\'t valid</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="rl-sub">Portal links are personal and tied to a single company. '
        'Please use the link from your email, or get in touch and we\'ll send a fresh one.</div>',
        unsafe_allow_html=True,
    )
    st.stop()

COMPANY = cfg["company_name"]
PEERS = [p for p in cfg.get("peer_companies", []) if p != COMPANY]
PEER_BASIS = cfg.get("peer_basis", "Selected peer group")
SHOW_PAY_BENCH = bool(cfg.get("show_pay_benchmarking", False))
UNIVERSE = [COMPANY] + PEERS

data = load_all()
alias = peer_alias_map(PEERS)

# ── Header ────────────────────────────────────────────────────────────────────
left, right = st.columns([3, 1])
with left:
    st.markdown('<div class="rl-brand">Remunilytics · Prepared for</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="rl-title">{COMPANY}</div>', unsafe_allow_html=True)
    # Deliberately company-level, not per-recipient: a single link is shared
    # with multiple officers at the same company, so naming one of them here
    # would be wrong (or stale) for everyone else who opens the same link.
    st.markdown(
        f'<div class="rl-sub">Executive remuneration structure, benchmarked '
        f'against {len(PEERS)} anonymised peers.</div>',
        unsafe_allow_html=True,
    )
with right:
    st.markdown(
        f'<div class="rl-card"><h4>Peer set</h4>'
        f'<div style="font-size:1.5rem;font-weight:700;color:#1F4E79">{len(PEERS)}</div>'
        f'<div class="rl-note">{PEER_BASIS}</div></div>',
        unsafe_allow_html=True,
    )

st.write("")

# ── Slices ────────────────────────────────────────────────────────────────────
ltip_all = scope(data, "ltip", UNIVERSE)
stip_all = scope(data, "stip", UNIVERSE)
pay_all = scope(data, "pay", UNIVERSE)
pol_all = scope(data, "policy", UNIVERSE)

own_year = latest_grant_year(ltip_all, COMPANY)
# latest_per_company also breaks ties when the same grant_year is described
# across multiple AR vintages (see its docstring) — no separate call needed.
ltip_latest = latest_per_company(ltip_all, "grant_year")
# Defensive filters for known upstream extraction gaps that predate their
# fixes in existing (not-yet-re-extracted) data:
ltip_latest = exclude_deferred_bonus_plans(ltip_latest)  # DABP is STIP, not LTIP
# A recruitment buy-out/replacement award mirrors a PREVIOUS employer's plan
# terms for one named individual — not this company's own LTIP design.
ltip_latest = exclude_buyout_replacement_awards(ltip_latest)
# Some ARs describe one grant twice (a policy table AND a granted-awards
# table), producing two near-identically-worded plans with the same metrics
# and weights. Left in, this silently doubles weight-sum totals — dedupe once
# here so every view downstream (cards, counts, the mix chart) is protected.
ltip_latest = dedupe_duplicate_plans(ltip_latest)
ltip_own = ltip_latest[ltip_latest["company_name"] == COMPANY]
ltip_primary = ltip_own[ltip_own.get("is_sub_metric", pd.Series(False, index=ltip_own.index)).fillna(False) == False] \
    if "is_sub_metric" in ltip_own.columns else ltip_own

pol_latest = latest_per_company(pol_all, "financial_year")
pay_latest = latest_per_company(pay_all, "financial_year")

TABS = ["Overview", "Long-Term Incentive", "Annual Bonus", "Policy"]
if SHOW_PAY_BENCH:
    TABS.append("Single Figure")
TABS.append("Sources")
tabs = st.tabs(TABS)
T = dict(zip(TABS, tabs))

# ══════════════════════════════════════════════════════════════════════════════
# OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
with T["Overview"]:
    st.markdown("#### At a glance")

    own_n = len(ltip_primary)
    peer_counts = (
        ltip_latest[(ltip_latest["company_name"] != COMPANY)]
        .assign(_sub=lambda d: d.get("is_sub_metric", pd.Series(False, index=d.index)).fillna(False))
        .query("_sub == False")
        .groupby("company_name").size()
    )
    peer_med_n = float(peer_counts.median()) if len(peer_counts) else float("nan")

    own_mix = set(ltip_primary["canonical_metric"].dropna()) if "canonical_metric" in ltip_primary.columns else set()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        delta = "" if pd.isna(peer_med_n) else f"Peer median {peer_med_n:.0f}"
        card("LTIP metrics", f"{own_n}", delta,
             flag=(not pd.isna(peer_med_n) and own_n > peer_med_n + 1))
    with c2:
        pol_own = pol_latest[pol_latest["company_name"] == COMPANY]
        ceo = pol_own[pol_own["position"].astype(str).str.contains("chief exec|CEO", case=False, na=False)]
        ltip_max = ceo["ltip_max_percentage"].dropna()
        peers_max = pol_latest[(pol_latest["company_name"] != COMPANY) &
                               pol_latest["position"].astype(str).str.contains("chief exec|CEO", case=False, na=False)]["ltip_max_percentage"].dropna()
        if len(ltip_max):
            p = pctile(ltip_max.iloc[0], peers_max.tolist())
            card("CEO LTIP opportunity", f"{ltip_max.iloc[0]:.0f}%",
                 f"of salary · {ordinal(p)} pctile vs peers" if p is not None else "of salary")
        else:
            card("CEO LTIP opportunity", "n/a", "not disclosed in latest policy")
    with c3:
        yrs = pd.to_numeric(ltip_primary.get("performance_period_years"), errors="coerce").dropna()
        card("Performance period (LTIP)", f"{yrs.mode().iloc[0]:.0f} yrs" if len(yrs) else "n/a",
             f"Grant year {own_year}" if own_year else "")
    with c4:
        prov = provenance_summary([ltip_own, stip_all[stip_all.company_name == COMPANY],
                                   pol_own, pay_latest[pay_latest.company_name == COMPANY]])
        exact = prov.get(3, 0)
        card("Traceable data points", f"{prov['total']}",
             f"{exact} cited to exact disclosure" if exact else "all linked to source page")

    st.write("")
    st.markdown("#### What stands out in your LTIP")

    insights = []
    if own_mix:
        peer_mix = ltip_latest[ltip_latest["company_name"] != COMPANY]
        if "canonical_metric" in peer_mix.columns:
            usage = (peer_mix.dropna(subset=["canonical_metric"])
                     .groupby("canonical_metric")["company_name"].nunique())
            n_peers = max(peer_mix["company_name"].nunique(), 1)
            for metric, cnt in usage.sort_values(ascending=False).items():
                share = cnt / n_peers
                if share >= 0.5 and metric not in own_mix:
                    insights.append(
                        (f"{canonical_label(metric)} is common among peers",
                         f"{cnt} of {n_peers} peers ({share:.0%}) include it in their LTIP "
                         f"— worth knowing for benchmarking conversations.", True))
            for metric in own_mix:
                cnt = int(usage.get(metric, 0))
                if cnt / n_peers <= 0.25:
                    # Lead with the point of difference itself, not a hedge
                    # about why it might matter — this page is meant to open
                    # a conversation, not read as the tool flagging a problem.
                    phrase = ("most don't weight it in their LTIP" if cnt == 0
                              else f"only {cnt} of {n_peers} peers also weight it")
                    insights.append(
                        (f"{canonical_label(metric)} is distinctive to your LTIP",
                         f"A point of difference from peer practice — {phrase}.", False))
    if not pd.isna(peer_med_n) and own_n > peer_med_n + 1:
        insights.append(("Your LTIP carries more measures than most peers",
                         f"{own_n} metrics vs a peer median of {peer_med_n:.0f}. "
                         "A longer scorecard spreads focus across more targets.", True))

    if insights:
        for title, body, flag in insights[:5]:
            st.markdown(
                f'<div class="rl-card {"rl-flag" if flag else ""}" style="margin-bottom:.6rem">'
                f'<div style="font-weight:700;font-size:1rem;color:#22303F">{title}</div>'
                f'<div class="rl-note">{body}</div></div>', unsafe_allow_html=True)
    else:
        st.info("Your LTIP metric mix sits close to the peer group on every measure we track.")

    st.caption("Peers are anonymised. Every figure in this portal links to the page of the "
               "Annual Report it was taken from — see the Sources tab.")

# ══════════════════════════════════════════════════════════════════════════════
# LTIP
# ══════════════════════════════════════════════════════════════════════════════
with T["Long-Term Incentive"]:
    if ltip_own.empty:
        st.info("No LTIP data on file for this company.")
    else:
        # These conditions were published BEFORE the award was made. Label the
        # nature of the DISCLOSURE, not the award's current status: by the time a
        # reader sees this the grant has very likely happened (typically announced
        # via a PDMR/grant RNS), so a present-tense "not yet granted" badge would
        # be factually wrong and invite an easy correction from a reward team.
        _statuses = (ltip_own["grant_status"].dropna().unique().tolist()
                     if "grant_status" in ltip_own.columns else [])
        _is_announced = _statuses == ["announced"]
        # Prefer the date the report actually stated. Only ever populated when
        # the report said it (see grant_date_is_stated) — never inferred.
        _dates = (ltip_own["grant_date"].dropna().unique().tolist()
                  if "grant_date" in ltip_own.columns else [])
        _grant_date = _dates[0] if len(_dates) == 1 else None
        _timing_notes = (ltip_own["grant_timing_note"].dropna().unique().tolist()
                         if "grant_timing_note" in ltip_own.columns else [])
        _timing_note = _timing_notes[0] if len(_timing_notes) == 1 else None

        if _is_announced and _grant_date:
            _badge = f"Granted {_grant_date}"
        elif _is_announced:
            _badge = "Disclosed ahead of grant"
        else:
            _badge = f"Granted {_grant_date}" if _grant_date else ""
        st.markdown(f"#### Your {own_year} LTIP"
                    + (f" &nbsp;<span class='rl-pill rl-t1'>{_badge}</span>" if _badge else ""),
                    unsafe_allow_html=True)
        if _is_announced and _grant_date:
            st.caption(f"Performance conditions as published in the Annual Report, for awards "
                       f"the company stated would be made in {_grant_date}.")
        elif _is_announced and _timing_note:
            # Extracted notes vary between a short fragment ("following the
            # 2026 AGM") and a full sentence ("Awards will be granted
            # following...") depending on how the report itself phrases it —
            # render as its own standalone sentence rather than assuming it
            # grammatically completes a template, which broke for full clauses.
            _note = _timing_note.strip().rstrip(".")
            _note = (_note[:1].upper() + _note[1:]) if _note else _note
            st.caption(f"Performance conditions as published in the Annual Report. {_note}. "
                       f"No exact grant date has been disclosed.")
        elif _is_announced:
            st.caption("Performance conditions as published in the Annual Report, for an award "
                       "that had not yet been made at the reporting date. The award itself may "
                       "have been granted subsequently.")
        show = ltip_own.copy()
        for _, r in show.iterrows():
            wt = r.get("weight_percentage")
            wt_s = f"{wt:.0f}%" if pd.notna(wt) else "—"
            pending = str(r.get("targets_pending", "")).lower() in ("true", "1")
            thr, strc = r.get("threshold_value"), r.get("stretch_value")
            tgt = r.get("target_value")
            parts = []
            for lbl, val in [("Threshold", thr), ("Target", tgt), ("Max", strc)]:
                if pd.notna(val) and str(val).strip() and str(val).strip().lower() != "nan":
                    parts.append(f"<b>{lbl}</b> {val}")
            rng = " &nbsp;·&nbsp; ".join(parts) if parts else (
                "<i>Targets not yet disclosed at the date of the report</i>" if pending else "—")
            t = r.get("source_attribution_tier")
            t = int(t) if pd.notna(t) else None
            # Per-card status badge only when the year mixes granted and announced
            # plans (the all-announced case is already stated in the heading).
            _card_status = ""
            if not _is_announced and str(r.get("grant_status", "")).lower() == "announced":
                _card_status = " <span class='rl-pill rl-t1'>Disclosed ahead of grant</span>"
            st.markdown(
                f'<div class="rl-card" style="margin-bottom:.55rem">'
                f'<div style="display:flex;justify-content:space-between;align-items:baseline;gap:1rem">'
                f'  <div style="font-weight:700;font-size:1rem;color:#22303F">{r.get("metric_name")}{_card_status}</div>'
                f'  <div style="font-size:1.15rem;font-weight:700;color:#1F4E79">{wt_s}</div>'
                f'</div>'
                f'<div class="rl-note">{rng}</div>'
                f'<div style="margin-top:.45rem">{tier_pill(t)} &nbsp; {src_link(r.get("source_link"))}</div>'
                f'</div>', unsafe_allow_html=True)

            source_view(r, "ltip")

        # Peer metric-mix comparison
        st.write("")
        st.markdown("#### Metric mix vs peers")
        if "canonical_metric" in ltip_latest.columns:
            # Rows with a real weight but no canonical classification (a gap in
            # our metric-name rules, not a data absence) must NOT be dropped —
            # doing so silently shrinks a company's bar below 100% and misreads
            # as "this company only weights X% of its LTIP". Bucket them as
            # "Other" instead so the stack always reflects the true total.
            mix_src = ltip_latest.copy()
            has_weight = mix_src["weight_percentage"].notna()
            mix_src["canonical_metric"] = mix_src["canonical_metric"].where(
                mix_src["canonical_metric"].notna() | ~has_weight, "other"
            )
            mix = (mix_src.dropna(subset=["canonical_metric"])
                   .groupby(["company_name", "canonical_metric"])["weight_percentage"]
                   .sum().reset_index())
            # A company whose LTIP is genuinely 100% restricted/time-based (all
            # captured metrics carry an explicit zero weight — e.g. binary
            # pass/fail underpins with no % split, common for RSAs) would
            # otherwise show as a blank bar indistinguishable from "no LTIP
            # data on file at all". Give it one distinct labeled segment
            # instead so "different plan design" doesn't read as "missing data".
            totals = mix.groupby("company_name")["weight_percentage"].sum()
            zero_weight_companies = totals[totals == 0].index
            if len(zero_weight_companies):
                mix = mix[~mix["company_name"].isin(zero_weight_companies)]
                mix = pd.concat([mix, pd.DataFrame({
                    "company_name": zero_weight_companies,
                    "canonical_metric": "restricted_time_based",
                    "weight_percentage": 100.0,
                })], ignore_index=True)
            mix = anonymise(mix, alias, COMPANY, own_label=COMPANY)
            # "other"/"restricted_time_based" always drawn last (top of stack)
            # regardless of weight, so they don't reshuffle the recognised
            # metric ordering.
            metrics_order = (mix[~mix["canonical_metric"].isin(["other", "restricted_time_based"])]
                             .groupby("canonical_metric")["weight_percentage"]
                             .sum().sort_values(ascending=False).index.tolist())
            for tail in ("other", "restricted_time_based"):
                if tail in mix["canonical_metric"].values:
                    metrics_order.append(tail)
            fig = go.Figure()
            for m in metrics_order:
                sub = mix[mix["canonical_metric"] == m]
                fig.add_bar(
                    x=sub["display_name"], y=sub["weight_percentage"],
                    name=canonical_label(m),
                    marker_line_width=0,
                    marker_color=("#C7CDD6" if m == "other"
                                 else "#8A94A6" if m == "restricted_time_based"
                                 else None),
                )
            order = [COMPANY] + [alias[p] for p in sorted(PEERS) if p in alias]
            fig.update_layout(
                barmode="stack", height=480,
                # A peer with zero data (e.g. no LTIP scheme) still needs its
                # own labeled x-axis slot -- but Plotly's categorical autorange
                # only spans categories that have actual trace data, so an
                # empty category trailing at the END of the array (unlike one
                # sandwiched between two with data) gets silently cut off
                # without an explicit range forcing the full domain to show.
                xaxis={"categoryorder": "array", "categoryarray": order,
                      "range": [-0.5, len(order) - 0.5]},
                yaxis_title="Weighting (%)",
                # Bottom margin has to fit BOTH the angled x-axis labels and the
                # legend below them — too small and Plotly clips the tick text.
                margin=dict(l=10, r=10, t=10, b=130),
                # Plotly reverses legend order by default for stacked bars (last
                # trace added, i.e. top of the stack, shows first) — "normal"
                # keeps it matching the order traces were added, so "Other"
                # (added last, deliberately) lists last rather than first.
                legend=dict(orientation="h", traceorder="normal",
                           yanchor="top", y=-0.32, x=0),
                plot_bgcolor="white", paper_bgcolor="white",
            )
            fig.update_xaxes(tickangle=-30)
            st.plotly_chart(fig, width='stretch')
            _caption = f"Most recent disclosed grant per company. Peer set: {PEER_BASIS}."
            # A peer with literally no bar (not even the grey "Restricted"
            # segment) has no LTIP data on file at all — call that out
            # explicitly so it doesn't read as missing data or a chart bug.
            _blank_peers = [alias[p] for p in PEERS
                           if p in alias and alias[p] not in mix["display_name"].values]
            if _blank_peers:
                _caption += (f" {', '.join(sorted(_blank_peers))}: no LTIP scheme identified "
                            f"in their Annual Report — not missing data.")
            st.caption(_caption)

# ══════════════════════════════════════════════════════════════════════════════
# ANNUAL BONUS
# ══════════════════════════════════════════════════════════════════════════════
with T["Annual Bonus"]:
    s_own = stip_all[stip_all["company_name"] == COMPANY].copy()
    if s_own.empty:
        st.info("No annual bonus data on file for this company.")
    else:
        s_own["_yr"] = parse_fiscal_year(s_own["financial_year"])
        # Same AR-vintage duplication risk as LTIP (see prefer_latest_ar_vintage
        # docstring) -- group on the already-parsed numeric year, not the raw
        # text label, since "FY9" vs "FY10" would sort wrong lexically.
        s_own = prefer_latest_ar_vintage(s_own, year_col="_yr")
        years = sorted(s_own["_yr"].dropna().unique(), reverse=True)
        yr = st.selectbox("Financial year", years, format_func=lambda y: f"FY{int(y)}")
        d = s_own[s_own["_yr"] == yr]
        payout = d["total_bonus_payout"].dropna()
        if len(payout):
            st.markdown(f"**Overall outcome:** {payout.iloc[0]:.1f}% of maximum")
        for _, r in d.iterrows():
            wt = r.get("weight_percentage")
            wt_s = f"{wt:.0f}%" if pd.notna(wt) else "—"
            tgt, act = r.get("target_value"), r.get("actual_performance")
            out = r.get("actual_payout_percentage")
            bits = []
            if pd.notna(tgt) and str(tgt).strip().lower() not in ("nan", "not disclosed", ""):
                bits.append(f"<b>Target</b> {tgt}")
            if pd.notna(out):
                bits.append(f"<b>Outcome</b> {out:.1f}%")
            head = " &nbsp;·&nbsp; ".join(bits) if bits else "—"
            t = r.get("source_attribution_tier")
            t = int(t) if pd.notna(t) else None
            st.markdown(
                f'<div class="rl-card" style="margin-bottom:.55rem">'
                f'<div style="display:flex;justify-content:space-between;align-items:baseline;gap:1rem">'
                f'  <div style="font-weight:700;font-size:1rem;color:#22303F">{r.get("metric_name")}</div>'
                f'  <div style="font-size:1.15rem;font-weight:700;color:#1F4E79">{wt_s}</div>'
                f'</div><div class="rl-note">{head}</div>'
                f'<div style="margin-top:.45rem">{tier_pill(t)} &nbsp; {src_link(r.get("source_link"))}</div>'
                f'</div>', unsafe_allow_html=True)
            source_view(r, "stip")
            if pd.notna(act) and len(str(act)) > 160:
                with st.expander("Committee's assessment"):
                    st.write(str(act))

# ══════════════════════════════════════════════════════════════════════════════
# POLICY
# ══════════════════════════════════════════════════════════════════════════════
with T["Policy"]:
    p_own = pol_latest[pol_latest["company_name"] == COMPANY]
    if p_own.empty:
        st.info("No policy data on file for this company.")
    else:
        st.markdown("#### Maximum opportunity vs peers")
        st.caption("Policy maxima as a percentage of salary — structural design, not pay levels.")
        peers_pol = pol_latest[pol_latest["company_name"] != COMPANY]
        for field, label in [("annual_bonus_max_percentage", "Annual bonus maximum"),
                             ("ltip_max_percentage", "LTIP maximum"),
                             ("shareholding_guideline_percentage", "Shareholding guideline")]:
            if field not in pol_latest.columns:
                continue
            ceo_mask = pol_latest["position"].astype(str).str.contains("chief exec|CEO", case=False, na=False)
            own_v = p_own[p_own["position"].astype(str).str.contains("chief exec|CEO", case=False, na=False)][field].dropna()
            peer_v = peers_pol[peers_pol["position"].astype(str).str.contains("chief exec|CEO", case=False, na=False)][field].dropna()
            if not len(own_v) or not len(peer_v):
                continue
            ov = float(own_v.iloc[0])
            fig = go.Figure()
            fig.add_box(x=peer_v.tolist(), name="Peers", marker_color=PEER,
                        boxpoints="all", jitter=.5, pointpos=0, hoverinfo="x")
            fig.add_scatter(x=[ov], y=["Peers"], mode="markers", name=COMPANY,
                            marker=dict(color=ACCENT, size=15, line=dict(color="white", width=2)))
            fig.update_layout(height=150, showlegend=False,
                              margin=dict(l=10, r=10, t=28, b=10),
                              title=dict(text=f"{label} — you: {ov:.0f}% ({ordinal(pctile(ov, peer_v.tolist()))} pctile)",
                                         font=dict(size=13)),
                              xaxis_title="% of salary", plot_bgcolor="white", paper_bgcolor="white")
            st.plotly_chart(fig, width='stretch')

        st.write("")
        st.markdown("#### Your policy detail")
        for _, r in p_own.iterrows():
            t = r.get("source_attribution_tier")
            t = int(t) if pd.notna(t) else None
            st.markdown(
                f'<div class="rl-card" style="margin-bottom:.55rem">'
                f'<div style="font-weight:700;font-size:1rem;color:#22303F">'
                f'{r.get("executive_name")} — {r.get("position")}</div>'
                f'<div class="rl-note">'
                f'Bonus max <b>{fmt_pct(r.get("annual_bonus_max_percentage"))}</b> &nbsp;·&nbsp; '
                f'LTIP max <b>{fmt_pct(r.get("ltip_max_percentage"))}</b> &nbsp;·&nbsp; '
                f'Shareholding <b>{fmt_pct(r.get("shareholding_guideline_percentage"))}</b></div>'
                f'<div style="margin-top:.45rem">{tier_pill(t)} &nbsp; {src_link(r.get("source_link"))}</div>'
                f'</div>', unsafe_allow_html=True)
            source_view(r, "policy")

# ══════════════════════════════════════════════════════════════════════════════
# SINGLE FIGURE (opt-in per recipient)
# ══════════════════════════════════════════════════════════════════════════════
if SHOW_PAY_BENCH:
    with T["Single Figure"]:
        pay_own = pay_latest[pay_latest["company_name"] == COMPANY]
        if pay_own.empty:
            st.info("No single-figure data on file for this company.")
        else:
            st.markdown("#### Single figure of total remuneration")
            for _, r in pay_own.iterrows():
                t = r.get("source_attribution_tier")
                t = int(t) if pd.notna(t) else None
                ccy = r.get("total_compensation_currency", "GBP")
                st.markdown(
                    f'<div class="rl-card" style="margin-bottom:.55rem">'
                    f'<div style="display:flex;justify-content:space-between;align-items:baseline">'
                    f'  <div style="font-weight:700;font-size:1rem;color:#22303F">'
                    f'  {r.get("executive_name")} — {r.get("position")}</div>'
                    f'  <div style="font-size:1.2rem;font-weight:700;color:#1F4E79">'
                    f'  {fmt_money(r.get("total_compensation_amount"), ccy)}</div></div>'
                    f'<div class="rl-note">'
                    f'Salary {fmt_money(r.get("salary_amount"), ccy)} &nbsp;·&nbsp; '
                    f'Bonus {fmt_money(r.get("annual_bonus_amount"), ccy)} &nbsp;·&nbsp; '
                    f'LTIP {fmt_money(r.get("ltip_amount"), ccy)}</div>'
                    f'<div style="margin-top:.45rem">{tier_pill(t)} &nbsp; {src_link(r.get("source_link"))}</div>'
                    f'</div>', unsafe_allow_html=True)
                source_view(r, "executive_pay")
            st.caption("Shown for your company only.")

# ══════════════════════════════════════════════════════════════════════════════
# SOURCES
# ══════════════════════════════════════════════════════════════════════════════
with T["Sources"]:
    st.markdown("#### Where every number comes from")
    st.write(
        "Every figure in this portal is extracted from a published Annual Report and "
        "carries a link back to the page it came from. Nothing here is modelled, "
        "estimated or survey-derived."
    )
    own_frames = [
        ltip_own,
        stip_all[stip_all["company_name"] == COMPANY],
        pol_latest[pol_latest["company_name"] == COMPANY],
        pay_latest[pay_latest["company_name"] == COMPANY],
    ]
    prov = provenance_summary(own_frames)
    c1, c2, c3 = st.columns(3)
    with c1:
        card("Exact citation", f"{prov.get(3,0)}", "traced to the specific disclosure block")
    with c2:
        card("Page-level", f"{prov.get(2,0)}", "matched to the page carrying the value")
    with c3:
        card("Section-level", f"{prov.get(1,0)}", "points to the report section")

    st.write("")
    rows = []
    for label, df in [("Long-term incentive", own_frames[0]), ("Annual bonus", own_frames[1]),
                      ("Policy", own_frames[2]), ("Single figure", own_frames[3])]:
        if df is None or df.empty:
            continue
        for _, r in df.iterrows():
            url, lab = parse_source_link(r.get("source_link"))
            if not url:
                continue
            t = r.get("source_attribution_tier")
            rows.append({
                "Facet": label,
                "Item": r.get("metric_name") or r.get("executive_name") or "—",
                "Precision": TIER_LABEL.get(int(t), ("—", ""))[0] if pd.notna(t) else "—",
                "Source": url,
            })
    if rows:
        st.dataframe(
            pd.DataFrame(rows), width='stretch', hide_index=True,
            column_config={"Source": st.column_config.LinkColumn("Source", display_text="Open ↗")},
        )

st.markdown(
    '<div class="rl-foot">Remunilytics · Data extracted from published Annual Reports. '
    'Peer identities are anonymised in this view. Prepared as an introduction to our approach — '
    'not remuneration advice.</div>', unsafe_allow_html=True)
