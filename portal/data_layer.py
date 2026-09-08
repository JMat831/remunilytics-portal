"""
Data access layer for the Remunilytics prospect portal.

Responsibilities
----------------
* Load + cache the five dataframes
* Parse the Excel HYPERLINK formulas in `source_link` into usable (url, label)
* Resolve a token to its company + peer group
* Anonymise peers consistently ("Peer A", "Peer B", ...) within a session
* Provide provenance quality so the UI can be honest about source precision
"""

import os
import re
import json
import pandas as pd
import streamlit as st

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATAFRAMES = os.path.join(BASE, "data", "dataframes")
TOKENS_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tokens.json")

DATASETS = {
    "ltip": "all_companies_ltip.csv",
    "ltip_consensus": "all_companies_ltip_with_consensus.csv",
    "stip": "all_companies_stip.csv",
    "pay": "all_companies_executive_pay.csv",
    "policy": "all_companies_policy.csv",
}

_HYPERLINK_RE = re.compile(r'=HYPERLINK\("([^"]+)"\s*,\s*"([^"]+)"\)')


def parse_source_link(value):
    """Turn '=HYPERLINK("url#page=N","p.N of AR")' into (url, label).

    Returns (None, None) when absent/unparseable so callers can degrade quietly.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None, None
    m = _HYPERLINK_RE.match(str(value).strip())
    if not m:
        s = str(value).strip()
        return (s, "View source") if s.lower().startswith("http") else (None, None)
    return m.group(1), m.group(2)


def source_page(value):
    """Extract the page number from a source link, if present."""
    url, _ = parse_source_link(value)
    if not url:
        return None
    m = re.search(r"#page=(\d+)", url)
    return int(m.group(1)) if m else None


TIER_LABEL = {
    3: ("Exact", "Cited to the specific disclosure block in the report"),
    2: ("Page", "Matched to the page containing this value"),
    1: ("Section", "Points to the start of the remuneration report section"),
}


def tier_of(row):
    t = row.get("source_attribution_tier")
    try:
        return int(t) if pd.notna(t) else None
    except (TypeError, ValueError):
        return None


def _attach_canonical_metric(df: pd.DataFrame) -> pd.DataFrame:
    """Derive `canonical_metric` live from the pipeline's own classifier.

    all_companies_ltip.csv doesn't carry this column (only the consensus join
    does, and that file lags). Classifying here keeps the portal aligned with the
    current METRIC_NAME_RULES instead of whatever was true when the join last ran.
    """
    if df.empty or "metric_name" not in df.columns:
        return df
    if "canonical_metric" in df.columns and df["canonical_metric"].notna().any():
        return df
    try:
        import sys
        if BASE not in sys.path:
            sys.path.insert(0, BASE)
        from enrich_ltip_with_consensus import classify_ltip_metric
    except Exception:
        return df
    df = df.copy()
    df["canonical_metric"] = df["metric_name"].map(
        lambda n: classify_ltip_metric(n)[0] if pd.notna(n) else None
    )
    return df


@st.cache_data(show_spinner=False)
def load_all():
    """Load every dataframe once per server process."""
    out = {}
    for key, fname in DATASETS.items():
        path = os.path.join(DATAFRAMES, fname)
        if not os.path.exists(path):
            out[key] = pd.DataFrame()
            continue
        df = pd.read_csv(path)
        # Normalise a couple of fields we rely on downstream
        if "company_name" in df.columns:
            df["company_name"] = df["company_name"].astype(str).str.strip()
        if key == "ltip":
            df = _attach_canonical_metric(df)
        out[key] = df
    return out


@st.cache_data(show_spinner=False)
def load_tokens():
    """Tokens are a bearer-secret access list, so they're never committed to
    the (public) deploy repo. Production reads them from Streamlit's private
    Secrets manager (`tokens_json`, a JSON string pasted into the app's
    Settings -> Secrets); local dev falls back to portal/tokens.json.
    """
    try:
        raw = st.secrets.get("tokens_json")
    except Exception:
        raw = None
    if raw:
        try:
            return json.loads(raw)
        except (TypeError, ValueError):
            pass
    if not os.path.exists(TOKENS_JSON):
        return {}
    with open(TOKENS_JSON, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_token(token: str):
    """Return the config dict for a token, or None if invalid."""
    if not token:
        return None
    return load_tokens().get(token.strip())


# ──────────────────────────────────────────────────────────────────────────────
# Peer anonymisation
# ──────────────────────────────────────────────────────────────────────────────

def peer_alias_map(peers):
    """Stable 'Peer A/B/C...' labels.

    Sorted by name so labels are deterministic across reloads, but the ordering
    carries no signal (not by size or performance) so a recipient can't infer
    identity from position alone.
    """
    letters = [chr(ord("A") + i) for i in range(26)]
    return {c: f"Peer {letters[i]}" if i < 26 else f"Peer {i + 1}"
            for i, c in enumerate(sorted(peers))}


def anonymise(df: pd.DataFrame, alias: dict, own_company: str,
              own_label: str = None) -> pd.DataFrame:
    """Replace company_name with alias labels; own company keeps its real name."""
    if df.empty:
        return df
    out = df.copy()
    own_label = own_label or own_company
    out["display_name"] = out["company_name"].map(
        lambda c: own_label if c == own_company else alias.get(c, "Peer")
    )
    out["is_own"] = out["company_name"] == own_company
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Scoped slices
# ──────────────────────────────────────────────────────────────────────────────

def scope(data: dict, key: str, companies) -> pd.DataFrame:
    """Rows for the given companies only — the hard boundary of the portal."""
    df = data.get(key)
    if df is None or df.empty or "company_name" not in df.columns:
        return pd.DataFrame()
    return df[df["company_name"].isin(list(companies))].copy()


def latest_grant_year(df: pd.DataFrame, company: str):
    d = df[df["company_name"] == company]
    if d.empty or "grant_year" not in d.columns:
        return None
    yrs = pd.to_numeric(d["grant_year"], errors="coerce").dropna()
    return int(yrs.max()) if len(yrs) else None


def parse_fiscal_year(series: pd.Series) -> pd.Series:
    """Extract a comparable 4-digit year from a fiscal-year label in ANY of
    the formats the extraction prompts document: 'FY2026', '2026', 'FY26',
    'F26' (Wizz Air omits the Y), 'FY24/25' or 'FYx/xx' (month-of-year-end
    plus a slash and the last two digits). Takes the LAST digit group found
    -- the "ending" year for a slash range, or the only group otherwise --
    and treats a bare 1-2 digit group as 20xx.

    A bare 4-digit-only regex (the original implementation) silently drops
    every row for any company using a 2-digit format: verified against real
    data, that emptied Policy for 11 companies and Pay for 32, including
    several with live tokens (AO World, Diageo, Kainos, Moonpig, Sage, Wizz
    Air among others) -- their Policy/Single-Figure/Annual-Bonus tabs looked
    like "no data on file" when the data was there, just unparseable.
    """
    def parse_one(s):
        if pd.isna(s):
            return None
        groups = re.findall(r"\d+", str(s))
        if not groups:
            return None
        last = groups[-1]
        return int(last[-4:]) if len(last) >= 4 else 2000 + int(last)
    return series.map(parse_one)


def latest_per_company(df: pd.DataFrame, year_col: str) -> pd.DataFrame:
    """Keep only each company's most recent year of rows.

    Also breaks ties when that latest year is described by more than one AR
    vintage on file (see prefer_latest_ar_vintage) — otherwise two mentions
    of the same year from different source documents both survive, and a
    metric worded differently between them looks like a real duplicate/gap
    rather than the same fact re-extracted. Verified against real data: 2
    company/year combos in Policy, 15 in Pay, before this fix.
    """
    if df.empty or year_col not in df.columns:
        return df
    d = df.copy()
    d["_yr"] = parse_fiscal_year(d[year_col])
    d = d.dropna(subset=["_yr"])
    if d.empty:
        return d
    maxes = d.groupby("company_name")["_yr"].transform("max")
    d = d[d["_yr"] == maxes]
    d = prefer_latest_ar_vintage(d, year_col="_yr")
    return d.drop(columns=["_yr"])


_DEFERRED_BONUS_RE = re.compile(
    r"deferred annual bonus|\bdabp\b|deferred bonus plan", re.IGNORECASE
)


def exclude_deferred_bonus_plans(df: pd.DataFrame) -> pd.DataFrame:
    """Drop plan rows that are a Deferred Annual Bonus Plan (DABP), not an LTIP.

    Mandatorily-deferred annual bonus shares are a settlement of an already-
    earned STIP payment, not a forward-looking LTIP vehicle — the extraction
    prompt now excludes these going forward (see config.py), but existing
    extractions predating that fix still carry them (e.g. GSK's 2026 grant
    shows a "Deferred Annual Bonus Plan (DABP)" row at 100% weight). This is a
    portal-side mirror of the same, already-validated exclusion rule.
    """
    if df.empty or "plan_name" not in df.columns:
        return df
    mask = ~df["plan_name"].astype(str).str.contains(_DEFERRED_BONUS_RE, na=False)
    return df[mask]


_BUYOUT_AWARD_RE = re.compile(
    r"buy-?out award|replacement award|replacement of\b|catch-up award", re.IGNORECASE
)


def exclude_buyout_replacement_awards(df: pd.DataFrame) -> pd.DataFrame:
    """Drop one-off awards tied to a named individual's unusual circumstance
    -- real, disclosed compensation, but not the company's repeatable LTIP
    design.

    When a new executive forfeits unvested awards from a PREVIOUS employer to
    join, the hiring company sometimes grants a mirroring "buy-out" award on
    similar terms to compensate. Found via Moonpig Group: two 2026-dated
    plans literally named "Buy-out award (replacement of Autotrader 2024/2025
    PSP), ... for Catherine Faiers" -- their metrics/weights mirror Auto
    Trader's plan design, not Moonpig's, and are tied to one named individual,
    not a repeatable policy choice. Left in, they inflated Moonpig's
    2026 grant to 200% weight and crowded out the company's own genuine LTIP
    plan (weight not-yet-disclosed for the new year) as the "latest" one
    shown -- exactly the same failure shape as AOIP for AO World, just
    triggered by an unrelated cause (a recruitment mechanic, not a
    mis-named deferred bonus). A Glass Lewis research report on Moonpig
    doesn't reference these at all, consistent with them being a one-off
    individual item rather than part of the standing remuneration policy.

    Also catches "catch-up award" -- the same underlying pattern via a
    different mechanic (a reappointment gap, not a recruitment buy-out).
    Found via Kainos Group: "2025 Performance Share Plan (PSP) -- 'FY25 PSP'
    (catch-up award for Brendan Mooney following reappointment as CEO)",
    granted 23 June 2025 alongside the genuine, ongoing "FY26 PSP" grant --
    verified against the raw extraction text (not hallucinated: real grant
    date, real named individual, real metrics) and against a Glass Lewis
    research report, which references only the standard FY2026/FY2027
    grants and never this one-off, consistent with the same "not part of
    the standing policy" pattern as Moonpig's buy-out awards.
    """
    if df.empty or "plan_name" not in df.columns:
        return df
    mask = ~df["plan_name"].astype(str).str.contains(_BUYOUT_AWARD_RE, na=False)
    return df[mask]


_CFO_ONLY_RE = re.compile(r"\bCFO only\b", re.IGNORECASE)
_AMBIGUOUS_ROLE_SCOPED_RE = re.compile(r"US[- ]based\b|US executive director", re.IGNORECASE)


def exclude_other_executive_only_awards(df: pd.DataFrame, pol_df: pd.DataFrame = None) -> pd.DataFrame:
    """Drop a plan explicitly scoped to a role other than the Group CEO --
    it isn't part of the CEO's own LTIP design, and every other CEO-anchored
    figure in this portal (the "CEO LTIP opportunity" card,
    `apply_ltip_quantum_weighting`'s Policy split) would be silently
    contradicted by pooling it in anyway.

    "CFO only" is unambiguous and always excluded regardless of the CEO's
    own Policy split -- by definition it can never be the CEO's own award,
    even once the CEO separately has a genuine restricted element of their
    own (Vistry Group has both: a CEO-specific Restricted Share Award
    confirmed by a later RNS, AND a separate CFO-only one from the same AR
    -- cross-checking only "does the CEO have some populated
    rsp_max_percentage" would wrongly let the CFO-only one back in once the
    CEO's own was added).

    A "US executive director(s)" / "US-based" scope is genuinely ambiguous,
    though -- it might describe a different, separately-Policy'd individual
    (BAE Systems: "President and Chief Executive Officer, BAE Systems,
    Inc." heads a distinct US-employees LTIP with its own larger, hybrid
    opportunity, confirmed via a Glass Lewis research report to be entirely
    separate from the Group CEO's UK-employees, performance-shares-only
    policy) or it might genuinely BE the CEO (Smith & Nephew: "Restricted
    Share Plan (RSP) - US Executive Directors" applies because that CEO,
    Deepak Nath, is himself a US executive director -- confirmed by his own
    Policy row disclosing a matching 300/125 psp/rsp split). Resolved by
    cross-checking `_true_ceo_policy_rows(pol_df)`: only excluded when that
    company's own CEO Policy row shows no matching rsp_max_percentage at
    all. Without `pol_df`, only the unambiguous "CFO only" phrasing is
    excluded, matching the previous, narrower behaviour.
    """
    if df.empty or "plan_name" not in df.columns:
        return df
    plan_names = df["plan_name"].astype(str)
    # Unconditional: "CFO only" can never be the CEO's own award, regardless
    # of whether the CEO separately has some other restricted element of
    # their own (found via Vistry Group, which has *both* -- a CEO-specific
    # Restricted Share Award confirmed by a later RNS, AND a separately
    # CFO-only one from the same AR; checking only "does the CEO have some
    # populated rsp_max_percentage" wrongly let the CFO-only one back in
    # once the CEO's own was added).
    drop_mask = plan_names.str.contains(_CFO_ONLY_RE, na=False)

    if pol_df is not None and not pol_df.empty:
        ambiguous = plan_names.str.contains(_AMBIGUOUS_ROLE_SCOPED_RE, na=False)
        if ambiguous.any():
            ceo_pol = _true_ceo_policy_rows(pol_df)
            ceo_has_rsp = set(ceo_pol.loc[ceo_pol["rsp_max_percentage"].notna(), "company_name"])
            drop_mask = drop_mask | (ambiguous & ~df["company_name"].isin(ceo_has_rsp))

    return df[~drop_mask]


def dedupe_duplicate_plans(df: pd.DataFrame) -> pd.DataFrame:
    """Drop plans that are the same grant captured twice under different names.

    Some ARs describe one grant in two places (e.g. a policy/implementation
    table and a "share awards granted" table), and each mention gets extracted
    as its own plan — same metrics, same weights, slightly different wording
    (e.g. "Adjusted EPS" vs "Adjusted Earnings Per Share (EPS)"). Left alone,
    any weight-sum aggregation (metric-mix chart, LTIP metric counts) silently
    doubles, tripling a company's apparent total past 100%.

    This is intentionally an EXACT-match rule to keep false positives at zero:
    within the same (company, grant_year), if two different plan_names have an
    identical multiset of (canonical_metric, weight_percentage) across their
    primary (non-sub) metrics, only the first plan_name (by row order) is kept.
    Genuinely distinct plans — e.g. a Performance Share Award alongside a
    Restricted Share Award with different metrics/weights — never match and are
    both kept untouched.
    """
    if df.empty or "plan_name" not in df.columns or "canonical_metric" not in df.columns:
        return df

    primary = df[df.get("is_sub_metric", pd.Series(False, index=df.index)).fillna(False) == False] \
        if "is_sub_metric" in df.columns else df

    keep_plan_keys = set()   # (company_name, grant_year, plan_name) to retain
    seen_signatures = {}     # (company_name, grant_year) -> {signature: kept_plan_name}

    for (co, gy, pn), g in primary.groupby(["company_name", "grant_year", "plan_name"], dropna=False):
        sig = frozenset(
            (m, w) for m, w in zip(g["canonical_metric"], g["weight_percentage"])
            if pd.notna(m)
        )
        group_key = (co, gy)
        bucket = seen_signatures.setdefault(group_key, {})
        if sig and sig in bucket:
            continue  # duplicate of an already-kept plan for this company/year
        if sig:
            bucket[sig] = pn
        keep_plan_keys.add((co, gy, pn))

    mask = df.apply(
        lambda r: (r["company_name"], r.get("grant_year"), r["plan_name"]) in keep_plan_keys,
        axis=1,
    )
    return df[mask]


def prefer_latest_ar_vintage(df: pd.DataFrame, year_col: str = "grant_year") -> pd.DataFrame:
    """When the same (company, year) is described by more than one AR
    vintage on file (e.g. AO World's grant_year=2022 Value Creation Plan
    appears in both the 2025 AR and the 2026 AR, worded slightly differently
    each time), keep only rows from the most recently re-extracted AR.

    Unlike dedupe_duplicate_plans (an exact metric/weight match, deliberately
    conservative), this triggers on file_name alone — two mentions of the
    same year from different source documents are always the same underlying
    grant/plan re-described, never two genuinely distinct ones, so there is
    no false-positive risk in preferring the newer one. Complements
    dedupe_duplicate_plans: without this, a metric worded differently across
    vintages (so it fails the exact-match test) survives as an unwanted
    duplicate — one copy correctly classified, one falling into "Other".
    Verified against real data: affects 63 company/year combos in LTIP alone,
    19 in STIP -- pass the relevant year column ("grant_year" for LTIP,
    "financial_year" for STIP/Policy/Pay) for each.
    """
    if df.empty or "file_name" not in df.columns or year_col not in df.columns:
        return df
    latest_file = df.groupby(["company_name", year_col])["file_name"].transform("max")
    return df[df["file_name"] == latest_file]


def _true_ceo_policy_rows(pol_df: pd.DataFrame) -> pd.DataFrame:
    """Best-available proxy for "the Group CEO's own Policy row" per company.

    A plain "chief exec" match can catch more than one person -- e.g. BAE
    Systems discloses Policy figures for both "Chief Executive" (the actual
    Group CEO, Charles Woodburn, UK-based) and "President and Chief
    Executive Officer, BAE Systems, Inc." (a different, named individual
    heading the US subsidiary, confirmed by a Glass Lewis research report to
    sit under an entirely separate US-employees LTIP policy with its own,
    larger, hybrid opportunity). The subsidiary/regional title is reliably
    the longer, more qualified one -- keep only the shortest matching title
    per company as the best proxy for the group-level role.
    """
    ceo_pol = pol_df[pol_df["position"].astype(str).str.contains("chief exec|CEO", case=False, na=False)].copy()
    if ceo_pol.empty:
        return ceo_pol
    shortest_len = ceo_pol.groupby("company_name")["position"].transform(
        lambda s: s.astype(str).str.len().min()
    )
    return ceo_pol[ceo_pol["position"].astype(str).str.len() == shortest_len]


def apply_ltip_quantum_weighting(ltip_df: pd.DataFrame, pol_df: pd.DataFrame) -> pd.DataFrame:
    """Rescale a hybrid PSP+RSP award's metric weights by their true share of
    total LTIP opportunity, for the metric-mix chart only.

    A company granting a 300%-of-salary Performance Share Plan AND a separate
    150%-of-salary Restricted Share Plan in the same year does not have "200%
    of LTIP" -- it has one blended award that is 2/3 PSP and 1/3 RSP by value.
    Left unscaled, each plan's own weights (which sum to ~100% within that
    plan) stack on top of each other, producing bars up to 200% and silently
    overstating every metric's true share of the award (found via Burberry
    Group, whose Policy already discloses the 300/150 split -- see
    psp_max_percentage/rsp_max_percentage below).

    Only applies when Policy discloses BOTH halves of the split for the CEO
    and the RSP is currently active, with mechanism "additive" (both
    elements granted in full every year, e.g. Burberry's fixed 300/150) or
    "substitutive" (a fixed trade-off within one overall cap, e.g.
    Antofagasta: Policy states "minimum of 70%" performance / "maximum of
    30%" restricted, and both FY2025 and FY2026 were confirmed granted at
    exactly that 70/30 split -- a Glass Lewis research report was needed to
    confirm this isn't a genuine either/or alternative, which the existing
    raw-sum-near-100 check below already guards against regardless). For a
    substitutive plan, the disclosed `ltip_max_percentage` is used as the
    true total rather than psp_max + rsp_max -- Antofagasta's psp_max (300)
    is a ceiling assuming zero restricted is used, not the actual granted
    amount, so summing it with rsp_max (90) would overstate the total to
    390% instead of the real 300% cap. Only triggers when the plans cleanly
    separate into exactly one RSP-like and one PSP-like plan_name; anything
    messier (three plans, or a disclosed split with no RSP-named plan among
    the grant's rows) is left untouched rather than guessed at.
    """
    if ltip_df.empty or "plan_name" not in ltip_df.columns:
        return ltip_df
    if pol_df.empty or "psp_max_percentage" not in pol_df.columns:
        return ltip_df

    ceo_pol = _true_ceo_policy_rows(pol_df)
    split = ceo_pol[
        ceo_pol["psp_max_percentage"].notna()
        & ceo_pol["rsp_max_percentage"].notna()
        & (ceo_pol["rsp_status"].astype(str).str.lower() == "active")
        & ceo_pol["ltip_hybrid_mechanism"].astype(str).str.lower().isin(["additive", "substitutive"])
    ].drop_duplicates(subset="company_name", keep="last").set_index("company_name")
    if split.empty:
        return ltip_df

    out_frames = []
    for company, grp in ltip_df.groupby("company_name", sort=False):
        if company not in split.index:
            out_frames.append(grp)
            continue
        psp_max = split.loc[company, "psp_max_percentage"]
        rsp_max = split.loc[company, "rsp_max_percentage"]
        # Use the disclosed overall cap as the true total, not psp_max +
        # rsp_max: for an "additive" plan the two always sum to it anyway,
        # but for a "substitutive" one (Antofagasta: performance ranges
        # 210-300% trading off against restricted's 0-90%, always summing
        # to a fixed 300% cap) psp_max is a ceiling assuming zero restricted
        # is used, not the actual granted amount -- summing the two would
        # overstate the total (390% instead of 300%) and understate the
        # restricted share (23.1% instead of the confirmed 30%).
        total = split.loc[company, "ltip_max_percentage"] if "ltip_max_percentage" in split.columns else None
        if pd.isna(total) or not total:
            total = psp_max + rsp_max
        if not total:
            out_frames.append(grp)
            continue
        plan_names = grp["plan_name"].astype(str)
        metric_names = grp["metric_name"].astype(str) if "metric_name" in grp.columns else pd.Series("", index=grp.index)
        # The RSP element is usually its own plan_name (Burberry, WPP, Smith &
        # Nephew) but sometimes bundled as one metric row inside a single
        # shared plan_name alongside the weighted PSP metrics (Hunting's
        # "2024 HPSP" names both elements in one plan, splitting them out
        # only at the metric_name level) -- check both. Also covers a
        # non-restricted "second element" that Policy still discloses as an
        # additive psp/rsp-style split (Diageo's SESOP share options, whose
        # performance conditions were removed, making it price/time-driven
        # like an RSP even though it isn't literally restricted shares).
        is_rsp = (
            (plan_names.str.contains(r"\brsp\b|restricted share", case=False, regex=True)
             & ~plan_names.str.contains("performance", case=False))
            | metric_names.str.contains(r"\brsp\b|restricted share|time-based restricted award|\bsesop\b",
                                        case=False, regex=True)
        )
        is_psp = ~is_rsp
        if grp.loc[is_rsp, "plan_name"].nunique() != 1 or grp.loc[is_psp, "plan_name"].nunique() != 1:
            out_frames.append(grp)
            continue
        # Both sides must independently already read as a complete award: the
        # PSP metrics must sum to ~100% on their own, and the RSP side must
        # either sum to ~100% (an explicit weighted restricted award, e.g.
        # Hunting) or to 0 (pure pass/fail underpins with no weight at all,
        # e.g. Burberry). A plan whose "PSP" component is actually a partial
        # kicker inside one blended award (e.g. Harworth's Core RSP Award
        # plus two 16.5%-weighted outperformance kickers, raw PSP sum 33) is
        # not this additive two-plan pattern at all -- scaling it would
        # invent a false total instead of reporting the real one.
        psp_raw = grp.loc[is_psp, "weight_percentage"].sum()
        rsp_raw = grp.loc[is_rsp, "weight_percentage"].sum()
        if not (90 <= psp_raw <= 110) or not (rsp_raw == 0 or 90 <= rsp_raw <= 110):
            out_frames.append(grp)
            continue
        # rsp_share from rsp_max directly, psp_share as the remainder of the
        # true total -- NOT psp_max / total, since for a substitutive plan
        # psp_max is a ceiling (fully performance, zero restricted used),
        # not the actual granted amount.
        rsp_share = rsp_max / total
        psp_share = 1 - rsp_share
        psp_rows = grp.loc[is_psp].copy()
        psp_rows["weight_percentage"] = psp_rows["weight_percentage"] * psp_share
        rsp_row = grp.loc[is_rsp].iloc[[0]].copy()
        rsp_row["metric_name"] = "Restricted (time-based) award"
        if "canonical_metric" in rsp_row.columns:
            rsp_row["canonical_metric"] = "restricted_time_based"
        rsp_row["weight_percentage"] = rsp_share * 100.0
        out_frames.append(pd.concat([psp_rows, rsp_row], ignore_index=True))
    return pd.concat(out_frames, ignore_index=True) if out_frames else ltip_df


def provenance_summary(frames) -> dict:
    """Aggregate source-precision counts across the frames shown to a recipient."""
    counts = {3: 0, 2: 0, 1: 0, 0: 0}
    for df in frames:
        if df is None or df.empty or "source_attribution_tier" not in df.columns:
            if df is not None and not df.empty:
                counts[0] += len(df)
            continue
        t = pd.to_numeric(df["source_attribution_tier"], errors="coerce")
        counts[3] += int((t == 3).sum())
        counts[2] += int((t == 2).sum())
        counts[1] += int((t == 1).sum())
        counts[0] += int(t.isna().sum())
    counts["total"] = counts[3] + counts[2] + counts[1] + counts[0]
    return counts
