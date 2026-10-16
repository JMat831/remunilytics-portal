"""
Design "moving parts" -- a countable measure of how complex a company's
executive incentive design is, built ONLY from the portal's own data
definitions (standing_ltip_view, stip_view/stip_chart_rows, ceo_for_display),
so every count matches what the LTIP, Annual Bonus and Policy tabs show.

The UK Corporate Governance Code (Provision 40) asks remuneration committees to
address simplicity and clarity. A client-facing measure should be facts the
company can check against its own report, so the portal shows COUNTS, never a
blended score; the internal screen (analysis_design_complexity.py) derives its
0-100 score from these same counts.

Components (each one a fact on another tab):
  ltip_vehicles      LTIP vehicles running (PSP, RSP, VCP...)        LTIP tab
  ltip_measures      performance measures across those vehicles       LTIP tab
                     (a time-based restricted award has none)
  ltip_sub_measures  nested sub-measures (e.g. per-comparator TSR,    LTIP tab
                     a category split into components)
  conditions         underpins / gateways / modifiers / kickers:      LTIP + Bonus
                     one per LTIP vehicle that has any, plus one if
                     the bonus has any
  bonus_measures     measures in the bonus scorecard (most recent     Bonus tab
                     year with weightings; a "(combined weighting)"
                     category counts through its members)

Context shown alongside, NOT counted (release mechanics, near-universal in the
UK, and a missing value must not blank the total): CEO bonus deferral (% and
years), post-employment shareholding (years), LTIP performance period.
"""

import re

import numpy as np
import pandas as pd

from data_layer import ceo_for_display

COMPONENTS = [
    ("ltip_vehicles", "LTIP vehicles running", "Long-Term Incentive"),
    ("ltip_measures", "LTIP performance measures", "Long-Term Incentive"),
    ("ltip_sub_measures", "Nested sub-measures", "Long-Term Incentive"),
    ("conditions", "Underpins, gateways and modifiers", "Long-Term Incentive / Annual Bonus"),
    ("bonus_measures", "Annual bonus measures", "Annual Bonus"),
]
CONTEXT = [
    ("bonus_deferral", "CEO bonus deferral", "Policy"),
    ("post_employment", "Post-employment shareholding", "Policy"),
    ("performance_period", "LTIP performance period", "Long-Term Incentive"),
]

# "kicker" = TSR kicker (Babcock); "multiplier" = bonus multipliers. Committee
# DISCRETION is deliberately not counted -- nearly every plan mentions it.
_CONDITION_RE = re.compile(r"\bunderpins?\b|\bgateways?\b|\bmodifiers?\b|\bkickers?\b|\bmultipliers?\b", re.I)
_NEGATED_RE = re.compile(r"\b(?:no|not subject to (?:any|an)?|without (?:any|an)?)\s*(?:performance\s+)?"
                         r"(?:underpins?|gateways?|modifiers?)", re.I)
# the verb ("objectives underpin delivery of the plan") is not a condition
_VERB_RE = re.compile(r"underpins?\s+(?:the|our|its|their|delivery|delivering|everything)", re.I)
_COMBINED_RE = re.compile(r"combined weighting", re.I)


def _is_sub(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower().isin(["true", "1", "1.0"])


def _has_condition(text: str) -> bool:
    if not isinstance(text, str) or not text:
        return False
    cleaned = _VERB_RE.sub(" ", _NEGATED_RE.sub(" ", text))
    return bool(_CONDITION_RE.search(cleaned))


def ltip_primary_measures(ltip_view: pd.DataFrame) -> pd.DataFrame:
    """Primary performance measures in a standing-LTIP frame: not sub-metrics,
    and not the placeholder row a time-based restricted award is stored as.
    The single definition behind both the Overview 'LTIP metrics' card and the
    moving-parts count, so the two can never disagree."""
    if ltip_view.empty:
        return ltip_view
    prim = ltip_view[~_is_sub(ltip_view.get("is_sub_metric", pd.Series(False, index=ltip_view.index)))]
    if "canonical_metric" in prim.columns:
        prim = prim[prim["canonical_metric"].fillna("") != "restricted_time_based"]
    return prim


def moving_parts(company: str, ltip_view: pd.DataFrame, bonus_rows: pd.DataFrame,
                 pol_latest: pd.DataFrame, ltip_design: pd.DataFrame = None) -> dict:
    """Counts + the items behind each count for one company.

    ltip_view   standing_ltip_view(...) output (any scope that includes company)
    bonus_rows  stip_chart_rows(stip_view(...)) output
    pol_latest  latest policy rows
    ltip_design standing_ltip_view(..., ceo_only=False): used when the CEO has no
                current award but the plan exists (a departing CEO), so the
                company's DESIGN is still counted
    Returns {component: count or None, 'items': {component: [str]}, 'total': int or None,
             'notes': [str]}. A component is None when the data can't support it;
    the total is None if any component is."""
    out = {"items": {k: [] for k, _, _ in COMPONENTS}, "context": {}, "notes": []}
    s = ltip_view[ltip_view["company_name"] == company] if not ltip_view.empty else ltip_view
    if s.empty and ltip_design is not None and not ltip_design.empty:
        s = ltip_design[ltip_design["company_name"] == company]
        if not s.empty:
            out["notes"].append("the CEO has no current LTIP award; counted from the plan as granted to other executives")
    b = bonus_rows[bonus_rows["company_name"] == company] if not bonus_rows.empty else bonus_rows
    ceo, _ = ceo_for_display(pol_latest, company)

    # LTIP: "no rows" is only a genuine zero if policy shows no LTIP either.
    policy_has_ltip = bool(len(ceo)) and pd.to_numeric(ceo.get("ltip_max_percentage"), errors="coerce").fillna(0).gt(0).any()
    if s.empty:
        if policy_has_ltip:
            for k in ("ltip_vehicles", "ltip_measures", "ltip_sub_measures"):
                out[k] = None
            out["notes"].append("LTIP disclosed in policy but its plan detail is not on file")
        else:
            for k in ("ltip_vehicles", "ltip_measures", "ltip_sub_measures"):
                out[k] = 0
            out["notes"].append("no LTIP on file")
        ltip_conditions = 0
    else:
        plans = list(dict.fromkeys(s["plan_name"].fillna("(unnamed plan)")))
        out["ltip_vehicles"] = len(plans)
        out["items"]["ltip_vehicles"] = plans
        prim = ltip_primary_measures(s)
        out["ltip_measures"] = int(len(prim))
        out["items"]["ltip_measures"] = prim["metric_name"].fillna("").tolist()
        subs = s[_is_sub(s.get("is_sub_metric", pd.Series(False, index=s.index)))]
        out["ltip_sub_measures"] = int(len(subs))
        out["items"]["ltip_sub_measures"] = subs["metric_name"].fillna("").tolist()
        ltip_conditions = 0
        for plan, g in s.groupby(s["plan_name"].fillna("(unnamed plan)"), sort=False):
            txt = (g["metric_name"].fillna("") + " " + g.get("additional_conditions", pd.Series("", index=g.index)).fillna("")
                   + " " + g.get("measurement_method", pd.Series("", index=g.index)).fillna(""))
            if any(_has_condition(t) for t in txt):
                ltip_conditions += 1
                out["items"]["conditions"].append(f"{plan} (LTIP)")

    # Bonus
    if b.empty:
        out["bonus_measures"] = None
        out["notes"].append("no annual bonus scorecard on file")
        bonus_condition = 0
    else:
        rows = b[b.get("canonical_metric", pd.Series("", index=b.index)).fillna("") != "undisclosed"]
        names = rows["metric_name"].fillna("")
        members = rows[~names.str.contains(_COMBINED_RE)]
        counted = members if len(members) else rows
        out["bonus_measures"] = int(len(counted))
        out["items"]["bonus_measures"] = counted["metric_name"].fillna("").tolist()
        btxt = b["metric_name"].fillna("") + " " + b.get("additional_conditions", pd.Series("", index=b.index)).fillna("")
        bonus_condition = int(any(_has_condition(t) for t in btxt))
        if bonus_condition:
            hit = b[btxt.map(_has_condition)]["metric_name"].fillna("").tolist()
            out["items"]["conditions"].append(f"Annual bonus ({hit[0][:60]})" if hit else "Annual bonus")
    out["conditions"] = ltip_conditions + bonus_condition

    # Context (not counted)
    d = pd.to_numeric(ceo.get("bonus_deferral_percentage"), errors="coerce").dropna() if len(ceo) else pd.Series(dtype=float)
    y = pd.to_numeric(ceo.get("bonus_deferral_period_years"), errors="coerce").dropna() if len(ceo) else pd.Series(dtype=float)
    out["context"]["bonus_deferral"] = (None if d.empty else
                                        "none" if d.max() == 0 else
                                        f"{d.max():g}%" + (f" for {y.max():g} yrs" if len(y) and y.max() > 0 else ""))
    pe = pd.to_numeric(ceo.get("post_employment_shareholding_years"), errors="coerce").dropna() if len(ceo) else pd.Series(dtype=float)
    out["context"]["post_employment"] = None if pe.empty else ("none" if pe.max() == 0 else f"{pe.max():g} yrs")
    pp = pd.to_numeric(s.get("performance_period_years"), errors="coerce").dropna() if not s.empty else pd.Series(dtype=float)
    out["context"]["performance_period"] = None if pp.empty else f"{pp.mode().iloc[0]:g} yrs"

    vals = [out[k] for k, _, _ in COMPONENTS]
    out["total"] = None if any(v is None for v in vals) else int(sum(vals))
    return out


def parts_frame(companies, ltip_view, bonus_rows, pol_latest, ltip_design=None) -> pd.DataFrame:
    """One row per company: each component, the total, and notes."""
    rows = []
    for c in companies:
        mp = moving_parts(c, ltip_view, bonus_rows, pol_latest, ltip_design)
        rows.append({"company_name": c, **{k: mp[k] for k, _, _ in COMPONENTS},
                     "total": mp["total"], **{f"ctx_{k}": mp["context"].get(k) for k, _, _ in CONTEXT},
                     "notes": "; ".join(mp["notes"])})
    return pd.DataFrame(rows)


def midrank_pctile(value, arr):
    """Identical to portal/app.py pctile: peers below count fully, level half."""
    arr = np.asarray([a for a in arr if pd.notna(a)], dtype=float)
    if len(arr) == 0 or value is None or pd.isna(value):
        return None
    v = float(value)
    return round(100.0 * ((arr < v).sum() + 0.5 * (arr == v).sum()) / len(arr))
