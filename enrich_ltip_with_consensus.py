"""
Enrich the LTIP dataset with matched consensus forecast values.

For each row in all_companies_ltip.csv:
  1. Normalise the free-text 'metric_name' to a canonical metric (revenue / eps /
     ebit / pbt / ebitda / pat / nii / rote / roe / cet1_ratio / nav_per_share /
     tsr / roce / roic / cash / esg / other) and a qualifier (adjusted /
     underlying / etc.).
  2. Classify 'measurement_method' as one of: absolute, cumulative, average,
     cagr, relative, qualitative.
  3. Look up the consensus value for the right ticker / canonical / year:
       - absolute      → consensus[performance_end_year]
       - cumulative    → SUM(consensus[end−N+1 … end])
       - average       → MEAN(consensus[end−N+1 … end])
       - cagr / growth → base = consensus[end−N], outer = consensus[end];
                         implied CAGR = (outer/base)^(1/N) − 1
       - relative / qualitative → skipped (cannot derive from consensus alone)

Output:
  data/dataframes/all_companies_ltip_with_consensus.csv

Does NOT overwrite all_companies_ltip.csv.
"""

import os
import re
import pandas as pd
from collections import Counter
from config import get_paths

paths = get_paths()


def parse_year(val):
    """Coerce an LTIP performance_end_year value (int, '2027', 'FY27', 'FY2027',
    'FY2024/25' etc.) into a plain integer year. Returns None if unparseable."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    try:
        # Already a clean number (or numeric string)
        return int(float(val))
    except (ValueError, TypeError):
        pass
    s = str(val).strip()
    m = re.search(r'(20\d{2})/(\d{2})(?!\d)', s)
    if m:
        return max(int(m.group(1)), 2000 + int(m.group(2)))
    years = [int(m) for m in re.findall(r"(20\d{2})", s)]
    if years:
        return max(years)  # 'FY2026' → 2026
    m = re.search(r"(?:FY|HY|H[12]|Q[1-4])\s*['‘’]?\s*(\d{2})", s, re.IGNORECASE)
    if m:
        return 2000 + int(m.group(1))
    return None



def parse_base_year_from_method(method_text, end_year: int) -> tuple:
    """Extract an explicit (base_year, n_years) from a measurement_method string.

    Handles patterns like:
      "from FY 2023/24 to FY 2026/27"         → (2024, 3)
      "from FY2024 to FY2027"                  → (2024, 3)
      "FY24–FY27 (CAGR)"                       → (2024, 3)
      "FY2023 baseline"                         → (2023, end_year - 2023)
      "1 April 2024 to 31 March 2027"          → (2024, 3)  base = year ending before start
      "1 January 2025 to 31 December 2027"     → (2024, 3)  base = year before period start

    Returns (base_year: int, n_years: int) or (None, None) if no pattern found.
    The returned n_years is the count of years between base and end
    (i.e. end_year - base_year), regardless of performance_period_years.
    """
    if not method_text or (isinstance(method_text, float) and pd.isna(method_text)):
        return None, None
    s = str(method_text).strip()

    def _max_year(y4_str, y2_str=None):
        """Return the later calendar year from a 'YYYY/YY' or plain 'YYYY' token."""
        y4 = int(y4_str)
        if y2_str:
            y2 = 2000 + int(y2_str)
            return max(y4, y2)
        return y4

    # Pattern: "from FY YYYY/YY" or "FY YYYY/YY" as a from-anchor
    m = re.search(r'\bfrom\s+FY\s*(\d{4})/(\d{2})\b', s, re.IGNORECASE)
    if m:
        base = _max_year(m.group(1), m.group(2))
        return base, end_year - base

    # Pattern: "from FY YYYY" (4-digit)
    m = re.search(r"\bfrom\s+FY\s*[''']?\s*(\d{4})\b", s, re.IGNORECASE)
    if m:
        base = int(m.group(1))
        return base, end_year - base

    # Pattern: "from FY YY" (2-digit)
    m = re.search(r"\bfrom\s+FY\s*[''']?\s*(\d{2})\b", s, re.IGNORECASE)
    if m:
        base = 2000 + int(m.group(1))
        return base, end_year - base

    # Pattern: "FYYYY–FYYYY" or "FY YY–FY YY" range (dash/en-dash/em-dash)
    m = re.search(r"\bFY\s*[''']?\s*(\d{2,4})\s*[–—\-]+\s*FY\s*[''']?\s*(\d{2,4})\b", s, re.IGNORECASE)
    if m:
        raw1, raw2 = m.group(1), m.group(2)
        y1 = (2000 + int(raw1)) if len(raw1) == 2 else int(raw1)
        return y1, end_year - y1

    # Pattern: "FY YYYY/YY" as first FY token (e.g. "Growth from FY 2023/24 to…")
    m = re.search(r'\bFY\s*(\d{4})/(\d{2})\b', s, re.IGNORECASE)
    if m:
        base = _max_year(m.group(1), m.group(2))
        return base, end_year - base

    # Pattern: "YYYY baseline" or "FY YYYY baseline"
    m = re.search(r'\b(?:FY\s*)?(\d{4})\s+baseline\b', s, re.IGNORECASE)
    if m:
        base = int(m.group(1))
        return base, end_year - base

    # Pattern: calendar date range "D Month YYYY to D Month YYYY"
    # Base = the fiscal year that ends on or before the start date.
    # We use start_year - 1 for start months Jan-Mar (March year-end convention),
    # start_year otherwise — but since we don't have the FY end month here, use
    # a simpler rule: base = the 4-digit year in the start date minus 1 if start
    # month is Jan-Jun, else the start year itself.
    # This is intentionally conservative; the fiscal_year_end_month enhancement
    # (planned) will make it exact.
    MONTHS = {'january':1,'february':2,'march':3,'april':4,'may':5,'june':6,
              'july':7,'august':8,'september':9,'october':10,'november':11,'december':12}
    m = re.search(
        r'\b(\d{1,2})\s+(january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{4})\b',
        s, re.IGNORECASE)
    if m:
        start_month = MONTHS[m.group(2).lower()]
        start_year  = int(m.group(3))
        # If performance starts in Jan-Jun, the base fiscal year typically ended
        # in the prior calendar year (e.g. April start → base = year ending March before)
        base = start_year - 1 if start_month <= 6 else start_year
        return base, end_year - base

    return None, None


LTIP_PATH      = os.path.join(paths["dataframes_path"], "all_companies_ltip.csv")
CONSENSUS_PATH = os.path.join(paths["dataframes_path"], "all_companies_consensus_long.csv")
OUTPUT_PATH    = os.path.join(paths["dataframes_path"], "all_companies_ltip_with_consensus.csv")


# ────────────────────────────────────────────────────────────────────────────────
# LTIP metric_name → canonical
# ────────────────────────────────────────────────────────────────────────────────
# Order matters: more specific patterns first to avoid false positives
# (e.g. ROCE must be checked before ROE).
METRIC_NAME_RULES = [
    # Disqualifiers FIRST — margin / ratio metrics look like financial metrics
    # but cannot be joined to absolute consensus values
    ("margin",              [r"\bmargin\b", r"as a percentage of",
                             r"\b%\s*of\b", r"as a % of"]),

    # A hybrid plan's restricted/time-based element (e.g. Diageo's SESOP,
    # Vistry's CFO-only RSA, BAE's US-executive RSA) is written by the LTIP
    # extraction prompt using this exact standardised phrase ("Time-based
    # restricted award (no performance conditions)" / "(subject to
    # underpin)") specifically so it can be told apart from a genuine
    # performance-conditioned metric. Must come before the ESG rule below —
    # the "(subject to underpin)" variant would otherwise be claimed by
    # ESG's bare `underpin` trigger and vanish into the wrong bucket instead
    # of getting its own distinct "Restricted (time-based)" treatment.
    ("restricted_time_based", [r"time-based restricted award"]),

    # Joinable to consensus
    ("eps",                 [r"\beps\b", r"earnings per share"]),
    ("ebitda",              [r"\bebitda\b"]),
    # PBIT / "profit before interest and tax" IS EBIT — must be matched here,
    # BEFORE the broader pbt "profit before ... tax" rule below would claim it.
    ("ebit",                [r"\bebit\b", r"\bpbit\b", r"profit before interest",
                             r"operating profit", r"operating income"]),
    # Allow words between "profit before" and "tax" so variants like "profit
    # before exceptional items and tax" (Diageo) are caught, not just the exact
    # phrase. Safe because the EBIT rule above already took the interest case.
    ("pbt",                 [r"\bpbt\b", r"\bpbet\b", r"profit before.*\btax\b",
                             r"pre[- ]?tax profit"]),
    ("pat",                 [r"\bpat\b", r"profit after tax", r"net income",
                             r"net profit", r"attributable profit"]),
    ("net_interest_income", [r"net interest income", r"\bnii\b"]),
    ("nav_per_share",       [r"nav per share", r"net asset value per share",
                             r"\bnta per share\b"]),
    # ESG before revenue — carbon/co₂/intensity patterns are unambiguous and
    # must not be overridden by \brevenue\b appearing elsewhere in the name
    # (e.g. "CO₂ Revenue Intensity Reduction" → esg, not revenue)
    ("esg",                 [r"\besg\b", r"sustainability", r"carbon",
                             r"emission", r"\bco2\b", r"co₂", r"intensity",
                             r"diversity", r"\bdiverse\b", r"gender", r"female",
                             r"\bwomen\b", r"safety", r"energy transition",
                             r"environmental", r"\bwaste\b",
                             r"\benergy\b.*(?:reduction|saving|efficien)",
                             r"governance", r"underpin"]),

    ("revenue",             [r"\brevenue\b", r"\bsales\b", r"\bturnover\b",
                             r"total income"]),
    # Bank-specific — joinable to consensus where published
    ("rote",                [r"\brote\b", r"return on tangible equity"]),
    ("cet1_ratio",          [r"\bcet1\b", r"capital ratio"]),

    # Value Creation Plan (VCP) — mechanically distinct from absolute TSR:
    # typically pure share-price/market-cap growth against fixed hurdles,
    # no dividend component, funded via a value-creation curve rather than a
    # standard threshold/target/max vesting scale. Must precede tsr_absolute
    # -- VCP metric text often also contains "share price growth" (an
    # absolute-TSR trigger phrase below), and lumping the two together would
    # hide a genuinely distinctive, uncommon plan design as if it were an
    # ordinary absolute-TSR PSP (found via AO World's VCP22). Requires "share
    # price" alongside "value creation" specifically -- "value creation" ALONE
    # is a generic label other companies use for unrelated metrics (Shell's
    # "Intrinsic Value Creation" is actually FCF-per-share growth, nothing to
    # do with a share-price-hurdle plan, and must not be swept in here).
    ("value_creation_plan", [r"share price.*value creation",
                             r"value creation.*share price"]),

    # TSR — explicit absolute first; everything else defaults to relative.
    # Share-price measures are absolute market measures, but the phrases must be
    # SPECIFIC: a bare "share price" would wrongly capture relative-TSR metrics
    # that merely define TSR as "share price plus dividends".
    ("tsr_absolute",        [r"absolute.*\btsr\b", r"\btsr\b.*absolute",
                             r"absolute total shareholder return",
                             r"absolute.*total.*return",
                             r"share price growth", r"share price performance",
                             r"share price vs", r"share price target"]),
    ("tsr_relative",        [r"\btsr\b", r"total shareholder return",
                             r"total accounting return", r"\btar\b"]),

    # Capital returns — ROCE / ROIC / ROE combined (not joinable to consensus)
    ("return_on_capital",   [r"\broce\b", r"\broace\b", r"\brnoa\b", r"\bronoa\b",
                             r"return on capital employed",
                             r"return on average capital",
                             r"return on capital", r"return on net operating",
                             r"\broic\b", r"return on invested capital",
                             r"\broe\b", r"return on equity"]),

    # Cash metrics — cash_conversion must precede cashflow (more specific first)
    ("cash_conversion",     [r"cash conversion", r"cash conversion ratio",
                             r"operating cash conversion",
                             r"working capital.*efficiency"]),
    ("cashflow",            [r"free cash flow", r"\bfcf\b",
                             r"cash generation", r"operating cash flow",
                             # one-word "cashflow" is common and was being missed
                             r"\bcash flow\b", r"\bcashflow\b",
                             r"cash remittances",
                             r"daily cash", r"cash from operations",
                             r"net cash"]),

    # Strategic — non-financial / qualitative / people / customer measures
    ("strategic",           [r"strategic", r"cultural", r"culture",
                             r"engagement", r"customer.*score",
                             r"customer satisfaction", r"customer care",
                             r"customer service", r"net promoter",
                             r"\bnps\b", r"trustpilot",
                             r"science and innovation", r"regulatory event",
                             r"business integrity", r"non[-\s]financial",
                             r"\bpeople\b"]),
]


def classify_ltip_metric(metric_name: str):
    """Return (canonical_metric, qualifier) for an LTIP metric_name string."""
    if not metric_name or pd.isna(metric_name):
        return None, None
    name = str(metric_name).lower()

    # Qualifier — preserve LTIP's flavour so we can match the right consensus variant
    qualifier = None
    if "adjusted" in name:
        qualifier = "adjusted"
    elif "underlying" in name:
        qualifier = "underlying"
    elif "reported" in name:
        qualifier = "reported"
    elif "comparable" in name:
        qualifier = "comparable"
    elif "core" in name:
        qualifier = "core"
    elif "benchmark" in name:
        qualifier = "benchmark"
    elif "headline" in name:
        qualifier = "headline"

    # Canonical metric
    canonical = None
    for label, patterns in METRIC_NAME_RULES:
        if any(re.search(p, name) for p in patterns):
            canonical = label
            break

    return canonical, qualifier


# ────────────────────────────────────────────────────────────────────────────────
# measurement_method → metric_type
# ────────────────────────────────────────────────────────────────────────────────
def classify_metric_type(metric_name: str, measurement_method: str) -> str:
    """Determine how the LTIP target should be evaluated against consensus."""
    name   = str(metric_name or "").lower()
    method = str(measurement_method or "").lower()

    # If the NAME explicitly contains growth/cagr, that always wins
    if "cagr" in name or "compound annual growth" in name or " growth" in name:
        return "cagr"

    if "cagr" in method or "compound annual growth" in method:
        return "cagr"
    if "cumulative" in method or "total" in method and "cumulative" in method:
        return "cumulative"
    if "cumulative" in method:
        return "cumulative"
    if "average" in method or "averaged" in method:
        return "average"
    if "relative to peer" in method or "relative to a peer" in method or "peer group" in method:
        return "relative"
    if any(tok in method for tok in [
        "qualitative", "discretionary", "binary", "pass/fail",
        "assessed holistically", "committee", "compliance assessment",
    ]):
        return "qualitative"
    if any(tok in method for tok in [
        "final year", "end of period", "at end of", "ending",
        "year value", "final measurement",
    ]):
        return "absolute"

    # TSR — relative unless the name explicitly says absolute
    if re.search(r"\btsr\b|total shareholder return|total accounting return", name):
        if "absolute" in name:
            return "absolute"
        return "relative"

    return "absolute"  # safest default


# ────────────────────────────────────────────────────────────────────────────────
# Consensus lookup
# ────────────────────────────────────────────────────────────────────────────────
# Qualifier preference for picking among multiple consensus rows on the same
# (ticker, canonical, year). LTIP's qualifier is matched first, then we fall
# back through this order.
QUALIFIER_FALLBACK = ["underlying", "adjusted", "reported", "comparable",
                      "core", "headline", "normalised", "statutory", None]


def pick_consensus_value(df, ltip_qualifier, eps_basis_pref=None):
    """From a slice of consensus rows, pick the best one based on qualifier
    preference. Returns the row's dict or None."""
    if df.empty:
        return None

    # Build preference order: LTIP's own qualifier first, then fallback list
    pref = [ltip_qualifier] + [q for q in QUALIFIER_FALLBACK if q != ltip_qualifier]

    for q in pref:
        candidate = df[df["qualifier"].fillna("") == (q or "")]
        if not candidate.empty:
            # For EPS, optionally prefer basis (basic > diluted if unspecified)
            if eps_basis_pref is not None and "eps_basis" in candidate.columns:
                basis_match = candidate[candidate["eps_basis"].fillna("") == eps_basis_pref]
                if not basis_match.empty:
                    return basis_match.iloc[0].to_dict()
            return candidate.iloc[0].to_dict()
    return df.iloc[0].to_dict()


def lookup_year(con_slice, year, ltip_qualifier):
    """Find a consensus value for a specific year."""
    year_slice = con_slice[con_slice["period_year"] == year]
    return pick_consensus_value(year_slice, ltip_qualifier)


def lookup_range(con_slice, start_year, end_year, ltip_qualifier):
    """Return a list of consensus values for each year in [start_year, end_year]."""
    values = []
    for y in range(int(start_year), int(end_year) + 1):
        match = lookup_year(con_slice, y, ltip_qualifier)
        if match and match.get("consensus_value") is not None:
            values.append((y, match["consensus_value"], match.get("unit"),
                           match.get("currency"), match.get("qualifier")))
    return values


# ────────────────────────────────────────────────────────────────────────────────
# Main enrichment
# ────────────────────────────────────────────────────────────────────────────────
def main():
    if not os.path.exists(LTIP_PATH):
        raise FileNotFoundError(f"LTIP file not found: {LTIP_PATH}")
    if not os.path.exists(CONSENSUS_PATH):
        raise FileNotFoundError(
            f"Consensus long file not found: {CONSENSUS_PATH}. "
            f"Run aggregate_consensus.py first."
        )

    ltip = pd.read_csv(LTIP_PATH, encoding="utf-8-sig")
    con  = pd.read_csv(CONSENSUS_PATH, encoding="utf-8-sig")

    # Only annual periods are useful for LTIP joins
    con = con[con["period_type"].fillna("").str.lower().isin(["annual", ""])].copy()

    # Coerce consensus_value to numeric — LLM extractions occasionally yield
    # strings like "N/A" or "1,234" that break arithmetic in cumulative/average.
    # Strip commas first so values like "1,234" parse to 1234, then drop rows
    # that still can't be coerced.
    con["consensus_value"] = (
        con["consensus_value"]
        .astype(str)
        .str.replace(",", "", regex=False)
    )
    con["consensus_value"] = pd.to_numeric(con["consensus_value"], errors="coerce")
    before = len(con)
    con = con.dropna(subset=["consensus_value"]).copy()
    dropped = before - len(con)
    if dropped:
        print(f"  [!] Dropped {dropped} consensus rows with non-numeric values")

    print(f"Loaded {len(ltip)} LTIP rows and {len(con)} annual consensus rows")

    # Output columns
    out_cols = {
        "canonical_metric": [],
        "ltip_qualifier":   [],
        "metric_type":      [],
        "consensus_outer_value": [],
        "consensus_outer_year":  [],
        "consensus_base_value":  [],
        "consensus_base_year":   [],
        "consensus_implied_cagr":      [],
        "consensus_implied_cumulative": [],
        "consensus_implied_average":    [],
        "consensus_unit":     [],
        "consensus_currency": [],
        "match_quality":      [],
    }

    stats = Counter()

    for _, row in ltip.iterrows():
        ticker      = row.get("ticker_bb")
        metric_name = row.get("metric_name")
        method      = row.get("measurement_method")
        end_year    = row.get("performance_end_year")
        period_yrs  = row.get("performance_period_years")

        canonical, ltip_qual = classify_ltip_metric(metric_name)
        metric_type          = classify_metric_type(metric_name, method)

        out_cols["canonical_metric"].append(canonical)
        out_cols["ltip_qualifier"].append(ltip_qual)
        out_cols["metric_type"].append(metric_type)

        # Defaults
        outer_val = outer_year = base_val = base_year = None
        implied_cagr = implied_cum = implied_avg = None
        unit = currency = None
        match_quality = None

        # Skip when no canonical match or no joinable metric
        UNJOINABLE = {
            "tsr_relative", "tsr_absolute", "return_on_capital",
            "cash_conversion", "cashflow", "esg", "strategic", "margin",
        }
        if not canonical:
            match_quality = "no_canonical_metric"
            stats[match_quality] += 1
        elif canonical in UNJOINABLE:
            match_quality = f"metric_not_in_consensus_{canonical}"
            stats[f"unjoinable_{canonical}"] += 1
        elif metric_type in {"relative", "qualitative"}:
            match_quality = f"unjoinable_{metric_type}"
            stats[match_quality] += 1
        else:
            end_year_int = parse_year(end_year)

            # Fallback: infer end year from grant_year + performance_period_years when
            # the AR described the grant prospectively without stating the end date.
            if end_year_int is None and not pd.isna(ticker):
                try:
                    grant_yr   = row.get("grant_year")
                    infer_p    = int(float(period_yrs)) if not pd.isna(period_yrs) else None
                    infer_g    = int(float(grant_yr))   if grant_yr is not None and not pd.isna(grant_yr) else None
                    if infer_g and infer_p:
                        end_year_int = infer_g + infer_p
                except (ValueError, TypeError):
                    pass

            if pd.isna(ticker) or end_year_int is None:
                match_quality = "missing_ticker_or_year"
                stats[match_quality] += 1
                # fall through to append defaults
                out_cols["consensus_outer_value"].append(outer_val)
                out_cols["consensus_outer_year"].append(outer_year)
                out_cols["consensus_base_value"].append(base_val)
                out_cols["consensus_base_year"].append(base_year)
                out_cols["consensus_implied_cagr"].append(implied_cagr)
                out_cols["consensus_implied_cumulative"].append(implied_cum)
                out_cols["consensus_implied_average"].append(implied_avg)
                out_cols["consensus_unit"].append(unit)
                out_cols["consensus_currency"].append(currency)
                out_cols["match_quality"].append(match_quality)
                continue

            # Does this company have ANY consensus data?
            company_has_consensus = (con["ticker_bb"] == ticker).any()

            # Slice consensus to this company + canonical metric
            con_slice = con[
                (con["ticker_bb"] == ticker)
                & (con["canonical_metric"] == canonical)
                & con["period_year"].notna()
            ]

            if con_slice.empty:
                if company_has_consensus:
                    match_quality = "metric_not_in_company_consensus"
                else:
                    match_quality = "no_consensus_data_for_company"
                stats[match_quality] += 1
            else:
                try:
                    period_int = int(float(period_yrs)) if not pd.isna(period_yrs) else 3
                except (ValueError, TypeError):
                    period_int = 3  # default to 3yr if unparseable

                if metric_type == "absolute":
                    m = lookup_year(con_slice, end_year_int, ltip_qual)
                    if m:
                        outer_val   = m["consensus_value"]
                        outer_year  = end_year_int
                        unit        = m.get("unit")
                        currency    = m.get("currency")
                        match_quality = "absolute_matched"
                    else:
                        match_quality = "no_consensus_for_year"

                elif metric_type == "cagr":
                    # Try to extract an explicit base year from the measurement
                    # method text (e.g. "from FY 2023/24 to FY 2026/27").
                    # Fall back to the formula end_year - period if not found.
                    parsed_base, parsed_n = parse_base_year_from_method(method, end_year_int)
                    if parsed_base and parsed_n and parsed_n > 0:
                        base_year_int = parsed_base
                        cagr_n        = parsed_n
                    else:
                        base_year_int = end_year_int - period_int
                        cagr_n        = period_int

                    m_outer = lookup_year(con_slice, end_year_int, ltip_qual)
                    m_base  = lookup_year(con_slice, base_year_int, ltip_qual)
                    if m_outer:
                        outer_val  = m_outer["consensus_value"]
                        outer_year = end_year_int
                        unit       = m_outer.get("unit")
                        currency   = m_outer.get("currency")
                    if m_base:
                        base_val  = m_base["consensus_value"]
                        base_year = base_year_int
                    if (outer_val is not None and base_val is not None
                            and base_val > 0 and cagr_n > 0):
                        implied_cagr = (outer_val / base_val) ** (1 / cagr_n) - 1
                        match_quality = "cagr_matched"
                    elif outer_val is not None:
                        match_quality = "cagr_outer_only"
                    else:
                        match_quality = "cagr_no_match"

                elif metric_type == "cumulative":
                    start = end_year_int - period_int + 1
                    vals = lookup_range(con_slice, start, end_year_int, ltip_qual)
                    if vals:
                        implied_cum = sum(v[1] for v in vals)
                        outer_year  = end_year_int
                        unit        = vals[-1][2]
                        currency    = vals[-1][3]
                        match_quality = (
                            "cumulative_matched_full" if len(vals) == period_int
                            else f"cumulative_partial_{len(vals)}_of_{period_int}"
                        )
                    else:
                        match_quality = "cumulative_no_match"

                elif metric_type == "average":
                    start = end_year_int - period_int + 1
                    vals = lookup_range(con_slice, start, end_year_int, ltip_qual)
                    if vals:
                        implied_avg = sum(v[1] for v in vals) / len(vals)
                        outer_year  = end_year_int
                        unit        = vals[-1][2]
                        currency    = vals[-1][3]
                        match_quality = (
                            "average_matched_full" if len(vals) == period_int
                            else f"average_partial_{len(vals)}_of_{period_int}"
                        )
                    else:
                        match_quality = "average_no_match"

                stats[match_quality] += 1

        out_cols["consensus_outer_value"].append(outer_val)
        out_cols["consensus_outer_year"].append(outer_year)
        out_cols["consensus_base_value"].append(base_val)
        out_cols["consensus_base_year"].append(base_year)
        out_cols["consensus_implied_cagr"].append(implied_cagr)
        out_cols["consensus_implied_cumulative"].append(implied_cum)
        out_cols["consensus_implied_average"].append(implied_avg)
        out_cols["consensus_unit"].append(unit)
        out_cols["consensus_currency"].append(currency)
        out_cols["match_quality"].append(match_quality)

    enriched = ltip.copy()
    for col, vals in out_cols.items():
        enriched[col] = vals

    enriched.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    # Summary
    print(f"\n{'=' * 70}")
    print("LTIP <-> CONSENSUS JOIN SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Total LTIP rows:   {len(ltip)}")
    print(f"  Output written to: {OUTPUT_PATH}\n")

    print("  Match quality breakdown:")
    for status, count in stats.most_common():
        pct = 100 * count / len(ltip)
        print(f"    {status:<45} {count:>5}  ({pct:.0f}%)")

    # Show successful matches by canonical
    matched_mask = enriched["match_quality"].str.contains("matched", na=False)
    if matched_mask.any():
        print(f"\n  Successful matches by canonical metric:")
        for canonical, count in enriched.loc[matched_mask, "canonical_metric"].value_counts().items():
            print(f"    {canonical:<25} {count:>5}")


if __name__ == "__main__":
    main()
