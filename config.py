# ============================================================================
# LLM PROVIDER CONFIGURATION
# ============================================================================

# LLM Provider Selection
LLM_PROVIDER = "anthropic"  # Options: "openai" or "anthropic"

# Model Selection based on provider
LLM_MODELS = {
    "openai": {
        "model": "gpt-4o",
        "max_tokens": 4000,
        "temperature": 0.0
    },
    "anthropic": {
        "model": "claude-sonnet-5",
        "max_tokens": 4000,
        "temperature": None  # claude-sonnet-5 rejects the temperature param; omit it
    }
}

# ============================================================================
# PIPELINE CONFIGURATION
# ============================================================================

# File Paths
PATHS = {
    "base_path": r"C:\Users\jonat\OneDrive\Remunilytics",
    "extractions_path": r"C:\Users\jonat\OneDrive\Remunilytics\data\extractions", 
    "schemas_path": r"C:\Users\jonat\OneDrive\Remunilytics\schemas",
    "dataframes_path": r"C:\Users\jonat\OneDrive\Remunilytics\data\dataframes",
    "metadata_path": r"C:\Users\jonat\OneDrive\Remunilytics\data\metadata",  # NEW
    "pdfs_path": r"C:\Users\jonat\OneDrive\Remunilytics\data\metadata\pdfs",  # NEW
    "temp_pdf": "temp_chunks.pdf",
}

# RAG Configuration
RAG_CONFIG = {
    "chunk_size": 1000,
    "chunk_overlap": 100,   
    "retrieval_k": 50,
    "llm_review_max_chars": 25000,  # Max characters for LLM review source text
    "search_queries": {
        "ltip": "LTIP Performance Share Plan PSP metrics targets weights vesting earnings per share EPS revenue growth EBIT profit TSR shareholder return carbon emissions diversity ESG environmental social governance underpin modifier threshold stretch exceptional measurement period relative peers percentage of award Restricted Share Plan RSP Restricted Share Award RSA Restricted Stock Unit RSU restricted stock award conditional shares time-based award hybrid LTIP no performance conditions continued employment nominee account holding period",
        
        "stip": "annual bonus short term incentive STIP targets achievement performance conditions salary bonus payout percentage achieved actual results financial performance operational targets customer satisfaction revenue profit EBITDA cash flow scorecard strategic objectives personal performance committee assessment management objectives",
        
        "executive_pay": "salary benefits pension bonus LTIP total remuneration Executive Directors single figure CEO CFO total compensation fixed variable pay allowances",

        "policy": "remuneration policy implementation base salary Executive Director CEO CFO annual bonus incentive opportunity maximum percentage LTIP award deferral deferred shares shareholding guideline requirement post-employment holding period pension contribution benefits policy table salary review increase effective date workforce alignment market data new policy current policy remuneration committee new appointment appointed recruit recruitment package incoming"
    }
}

# Schema Extraction Configuration
SCHEMA_CONFIG = {
    "max_content_length": 80000,  # was 45000 — bumped to prevent late-index chunks being silently truncated
    "include_table_processing": True
}

# Pipeline Flow:
# Chunk Retrieval (100+ chunks) → Cache Check (avoids API costs) → RAG CONFIG Search Query (filters) → 50 relevant chunks → Ordered Context (by chunk index) → Text Prompt Analysis → Schema Conversion → Structured JSON

# ============================================================================
# PROMPTS
# ============================================================================

LTIP_PROMPT_TEMPLATE = """
You are an AI specialised in analysing executive remuneration reports.
Carefully extract the details - including metrics, corresponding targets, any conditions (e.g. underpin/modifier) 
and measurement method (e.g. 'Average over period', 'Total over period', 'Final year' (pertaining to fundamental metrics like Earnings per Share), 
'Relative to peers' (e.g. Relative TSR)) - of the various plans/schemes from the provided context. 
Ignore plans which have vested (may have columns called 'Achieved' and 'Vesting' in relevant tables).
Don't mix elements of different plans - use information in close proximity.
Bear in mind the LTIP may have a specific name like 'Performance Share Plan', 'Hybrid Plan' or 'Restricted Share Plan'.
Look out for abbreviations like 'PSP' (for 'Performance Share Plan') for example.

EXCLUDE DEFERRED ANNUAL BONUS AWARDS: Shares representing the MANDATORY DEFERRAL of an
already-earned annual bonus (e.g. 'Deferred Annual Bonus Plan' / 'DABP', 'Deferred Bonus
Plan', 'Bonus deferred into shares') are part of the SHORT-TERM incentive (annual bonus) and
must NOT be listed as an LTIP plan — even when the report tables them alongside LTIP awards
under a heading such as 'LTI grants' or 'Share awards granted in the year'. They carry no
forward-looking performance conditions because the performance (the bonus) has already been
earned; a time-based holding period alone does not make them an LTIP. This is DIFFERENT from a
Restricted Share Plan / Restricted Share Award (RSA) — OR ITS US-STYLE EQUIVALENT, a Restricted
Stock Unit (RSU) / Restricted Stock Award — which IS a forward-looking long-term award and should
still be captured: the test is whether the shares are settlement of a PAST bonus (exclude) versus
a new forward-looking grant (include). US-listed or US-influenced companies (and non-UK
subsidiaries within a UK group) commonly say "Restricted Stock Unit (RSU)" where a UK company
would say "Restricted Share" — treat these as the SAME category of award; do not let the
terminology difference cause you to miss it.

ALWAYS CAPTURE RESTRICTED/TIME-BASED AWARDS AS THEIR OWN PLAN ENTRY: A Restricted Share Plan,
Restricted Share Award (RSA), Restricted Stock Unit (RSU), Restricted Stock Award, or other
time-based award that vests primarily on continued employment — even if subject to a single
underpin/gateway condition (a pass/fail check, NOT a graduated threshold-to-maximum performance
scale) — is still a genuine LTIP vehicle and MUST be listed as its own plan entry for that grant
year, even though it has no scaled metric to measure. Do not omit it just because it lacks a
performance table, and do not exclude it merely because it says "subject to underpin" — an
underpin is a gate, not a performance condition in the PSP sense. Represent it with a single
metric, e.g.:
  - Metric Name: "Time-based restricted award (no performance conditions)" — or, if an underpin
    applies, "Time-based restricted award (subject to underpin)" and note the underpin in
    Additional Condition
  - Weight: 100
  - Threshold/Target/Stretch/Exceptional: leave blank
  - Measurement Method: "Time-based — vests on continued employment"
This applies whether the RSA/RSU is a separate plan alongside a PSP (a hybrid structure) or the
company's only long-term award that year.
Also, the key details may be contained in a complex table or graphic.
The relevant fiscal years may have the format FYxx (the last two digits of the year after 'FY'), 
FY20xx (all four digits of the year after 'FY') or FYx/xx (month of the year end, forward slash 
and the last two digits of the year) or FYx/20xx (month of the year end, forward slash and all four digits of the year).
If multiple years are referenced, list each year's LTIP metrics separately, clearly indicating which year they belong to.

IMPORTANT: Please provide complete information for ALL metrics. For each metric extract:
- Description
- Weight
- Performance thresholds (Threshold, Target, Stretch, Maximum if applicable)
- Vesting levels for each performance threshold
- Any conditions or modifiers applied
- Measurement Method
- Source Chunk: the number from the [CHUNK n | ...] marker of the chunk you took this
  metric's details from. If the metric's details span more than one chunk, cite the chunk
  containing its targets/weights table. Always cite a chunk number that actually appears in
  the context above — never invent one.

CRITICAL: When extracting metrics, pay careful attention to fiscal year indicators (FY24/FY26 vs FY25/FY27).
Do NOT mix metrics from different grant years. Each plan's metrics must come from content
explicitly mentioning that plan's performance period.

NO DUPLICATE PLANS: A single grant is often described in more than one place in the report
(e.g. a forward-looking policy/implementation table AND a "share awards granted in the year"
table). List each grant's metrics ONCE, consolidating detail from all sections into a single
plan entry for that (grant year, plan name). Do NOT emit the same grant's metrics twice as
separate plans just because they appear in two tables — this double-counts weights (a plan
would wrongly sum to ~200%). If two tables describe the same grant, merge them; only create
separate plans when they are genuinely different awards or different grant years.

MIXED PERFORMANCE HORIZONS: Some plans measure different metrics over different periods
(e.g. an Executive Incentive Plan with some financial metrics measured over 3 years and
others over 1 year). When metrics within the SAME plan have different performance periods,
state each metric's own performance period and end year explicitly next to that metric
(e.g. "measured over 3 years, ending FY2027" vs "measured over 1 year, FY2025"). Do not
apply a single plan-level period to metrics that are actually measured over a different horizon.

SUB-METRIC WEIGHTS: When a metric (e.g. a Non-Financial or ESG element) is split into
sub-components, the sub-component weights must SUM to the parent metric's weight — do NOT
give each sub-component the full parent weight (e.g. a 25% non-financial element split into
four components is ~6.25% each, not 25% each).

GRANT YEAR IDENTIFICATION RULES:
- grant_year is always the CALENDAR year in which the award is (or will be) made.
- Plans ending in FY26 were typically granted in 2024 (calendar year)
- Plans ending in FY27 were typically granted in 2025 (calendar year)
- Plans ending in FY28 were typically granted in 2026 (calendar year)
- If performance metrics reference different end years (e.g. FY26 vs FY27), they belong to SEPARATE plans
- For a NOT-YET-GRANTED award described as "to be made in the year ending [date]", work out the
  calendar year the award will actually be made in — NOT the fiscal year label. Use the grant
  date of the equivalent prior-year award as the pattern where the report gives one. Example:
  a June year-end company whose prior award was granted 3 September 2024 (within "the year
  ended 30 June 2025") discloses conditions for awards "to be made in the year ending 30 June
  2026" — that award will be granted around September 2025, so grant_year = 2025 (NOT 2026).

UPCOMING / NOT-YET-GRANTED AWARDS (IMPORTANT):
The test for including a plan is whether its MEASURES AND WEIGHTINGS ARE DISCLOSED — NOT
whether the award has formally been granted yet. Reports routinely set out the performance
conditions for a future award under headings such as "Performance conditions for long-term
incentive awards TO BE MADE in the year ending [date]", "Implementation of policy for [year]",
or "[Year] LTIP awards". These MUST be captured as their own plan/year. Do not skip a table
merely because the award has not yet been made or the share numbers are not yet known.

Two distinct situations arise — capture BOTH, and label them:

(a) TARGETS ALREADY QUANTIFIED for the future award. The report gives the actual
    threshold / midpoint / target / maximum figures (e.g. a table with "Weighting (% total)"
    and Threshold/Midpoint/Maximum rows). Record the metrics, weights AND those real target
    values exactly as stated. Mark it "Grant Status: announced".

(b) TARGETS NOT YET SET for the future award (e.g. "targets will be published in the RNS
    notifying the market of the grant", "to be determined", "disclosed in next year's report").
    Still list EVERY metric and its weight, write the targets explicitly as
    "Targets: to be determined", and leave Threshold, Target, Stretch and Exceptional blank.
    Mark it "Grant Status: announced".

Awards that HAVE been formally granted (a grant date and/or number of shares awarded is
stated) are "Grant Status: granted".

GRANT DATE: Report the grant date whenever the text states one, for BOTH granted and
announced awards, quoting it as given — a full date ("On 3 September 2024, the Executive
Directors received awards...") or just month and year where that is all that is specified
("DLTIP awards to be made in September 2025 will comprise..."). Such timing statements often
sit in the narrative on a DIFFERENT page from the targets table, so look for them across the
surrounding commentary, not only in the table itself. If — and only if — no date is stated
anywhere, write "Grant Date: not stated". NEVER infer a date from the previous year's grant
timing or from the fiscal year label; an absent date is more useful to us than a guessed one.

- NEVER copy or carry forward the targets from an earlier grant year into a later grant. If a
  later grant's targets are not explicitly stated in the text, they are pending — do not invent
  or infer them from a prior year's plan.
- When a company describes a CHANGE to measures or weightings for an upcoming grant, use the
  NEW/REVISED measures and weights that will apply — not the prior weights they are being
  compared against. Prefer a dedicated measures-and-weightings table over narrative
  descriptions of the change (e.g. use "25% on free cash flow", not the "increased from 20%
  on average cash conversion" it is compared to).

Try to use contiguous chunk indices as much as possible for each individual plan.

{context}
---
Answer the question based on the above context: {question}
---
Provide the answer in the following format:
- **Year (in which the awards were/will be granted):** [Calendar year]
  - **Plan Name:** [If specified]
  - **Grant Status:** [granted | announced] — "granted" if a grant date or number of shares
    awarded is stated; "announced" if the conditions are disclosed for an award not yet made
  - **Grant Date:** [date exactly as stated in the report, e.g. "3 September 2024" or
    "September 2025"; or "not stated" if the report gives no date — never infer one]
  - **Performance Period (no. of years):** [If specified (usually 3 years for most LTIPs)]
  - **Metrics:**
    - Metric 1: [Description, Weight, Threshold, Target, Stretch and Exceptional target (if applicable), Additional Condition (e.g. underpin/modifier) if applicable, Measurement Method, Source Chunk: n]
    - Metric 2: [Description, Weight, Threshold, Target, Stretch and Exceptional target (if applicable), Additional Condition (e.g. underpin/modifier) if applicable, Measurement Method, Source Chunk: n]
    - [etc.]
"""   

LTIP_QUESTION = """
For all LONG-TERM incentive plans of the last one or two years — including both awards already
GRANTED and awards ANNOUNCED but not yet granted (e.g. "performance conditions for awards to be
made in the year ending ...") — what are the performance metrics and corresponding weights,
targets, conditions (if any) and measurement methods for each plan?
Focus only on plans which have NOT YET vested, i.e. paid out.
"""

STIP_PROMPT_TEMPLATE = """
You are an AI specialised in analysing executive remuneration reports.
Carefully extract the details of the CEO's SHORT-TERM incentive plans (annual bonus plans, STIP) from the provided context.
Focus specifically on the Chief Executive Officer's bonus structure and performance.
Extract information for BOTH the completed year (with actual performance and payouts) AND the upcoming year (targets only), if both are present.
Don't mix elements of different plans or different years - use information in close proximity.
Bear in mind the STIP may have a specific name like 'Annual Bonus Plan', 'Short-Term Incentive Plan' or simply 'Bonus'.
Look out for abbreviations like 'STIP' (for 'Short-Term Incentive Plan') for example.
The key details may be contained in complex tables showing targets vs actual performance (for completed year) or just targets (for upcoming year).
The relevant fiscal years may have the format FYxx (the last two digits of the year after 'FY'), 
FY20xx (all four digits of the year after 'FY') or FYx/xx (month of the year end, forward slash 
and the last two digits of the year) or FYx/20xx (month of the year end, forward slash and all four digits of the year).

IMPORTANT: Please provide complete information for ALL CEO bonus metrics. For each metric, extract:
- Description
- Weight
- Performance thresholds (Threshold, Target, Stretch, Maximum if applicable)
- Vesting levels for each performance threshold
- For COMPLETED year: Actual performance achieved and corresponding payout percentage
- For UPCOMING year: Only targets (no actual performance or payouts available)
- Any conditions or modifiers applied
- Measurement method
- Source Chunk: the number from the [CHUNK n | ...] marker of the chunk you took this
  metric's details from. If the details span more than one chunk, cite the chunk containing
  its targets/outcome table. Always cite a chunk number that actually appears in the context
  above — never invent one.

SPECIAL HANDLING FOR QUALITATIVE METRICS: For metrics like "Strategic measures", "Personal goals", "Individual objectives", or similar qualitative assessments:
- Provide a concise summary of the key performance areas assessed
- Include any sub-metrics or scoring frameworks mentioned
- Note the overall assessment outcome and payout percentage achieved
- If detailed tables or extensive narrative assessments are provided, summarise the main points rather than reproducing everything

CRITICAL: Focus specifically on the CEO's bonus plan. Clearly separate completed year results from upcoming year plans.
Look for tables showing 'Target vs Actual' or 'Performance vs Achievement' for the completed year.
Look for forward-looking target tables for the upcoming year's bonus structure.

{context}
---
Answer the question based on the above context: {question}
---
Provide the answer in the following format:
- **Year (completed financial year):** [Calendar/Fiscal year]
  - **Plan Name:** [If specified]
  - **Plan Status:** Completed (with actual results)
  - **CEO Metrics:** 
    - Metric 1: [Description, Weight, Threshold, Target, Stretch/Maximum (if applicable), Actual Performance Achieved, Payout Percentage, Additional Condition (if applicable), Source Chunk: n]
    - Metric 2: [Description, Weight, Threshold, Target, Stretch/Maximum (if applicable), Actual Performance Achieved, Payout Percentage, Additional Condition (if applicable), Source Chunk: n]
    - [For qualitative metrics: Include summary of assessment areas and overall outcome]
    - [etc.]
  - **Total CEO Bonus Payout:** [Overall percentage achieved, if specified]

- **Year (upcoming financial year):** [Calendar/Fiscal year] (if available)
  - **Plan Name:** [If specified]
  - **Plan Status:** Prospective (targets only)
  - **CEO Metrics:** 
    - Metric 1: [Description, Weight, Threshold, Target, Stretch/Maximum (if applicable), Actual Performance: N/A (prospective), Payout Percentage: N/A (prospective), Additional Condition (if applicable), Source Chunk: n]
    - Metric 2: [Description, Weight, Threshold, Target, Stretch/Maximum (if applicable), Actual Performance: N/A (prospective), Payout Percentage: N/A (prospective), Additional Condition (if applicable), Source Chunk: n]
    - [etc.]
"""

STIP_QUESTION = """
For the CEO's SHORT-TERM incentive plans (annual bonus plans), extract information for both:
1. The most recent COMPLETED financial year showing the CEO's performance metrics, targets, actual performance achieved and payout percentages
2. The UPCOMING financial year showing the CEO's bonus structure and targets (if disclosed)
Focus specifically on the Chief Executive Officer's bonus plan and distinguish between completed plans (with results) and prospective plans (targets only).
For any qualitative or strategic metrics, provide summaries of the assessment areas and outcomes.
"""

EXECUTIVE_PAY_PROMPT_TEMPLATE = """
You are an AI specialised in analysing executive remuneration reports.
Carefully extract the executive directors' total remuneration breakdown from the provided context.
Focus on the "single figure" remuneration tables that show the complete compensation for each executive director.
Look for tables showing individual compensation components: salary, benefits, pension, bonus, LTIP/long-term incentives, and total compensation.
Extract information for ALL executive directors listed (typically CEO, CFO and other Executive Directors).
The key details are usually contained in detailed remuneration tables, often called "Single figure for total remuneration" or similar.
The relevant fiscal years may have the format FYxx (the last two digits of the year after 'FY'), 
FY20xx (all four digits of the year after 'FY') or FYx/xx (month of the year end, forward slash 
and the last two digits of the year) or FYx/20xx (month of the year end, forward slash and all four digits of the year).

IMPORTANT: Please provide complete information for ALL executive directors. For each director, extract:
- Full name and position/title
- Salary (base salary)
- Benefits (benefits in kind, allowances, etc.)
- Pension (pension contributions or pension-related payments)
- Annual Bonus (short-term incentive payments for the year)
- LTIP (long-term incentive plan vesting/payments for the year)
- Other compensation (if any)
- Total compensation (sum of all components)
- Currency (if specified)
- Pro-rata indicator: Note if the executive only served part of the year (new appointment, departure or role change mid-year)
- Pro-rata notes: Brief explanation if pro-rata applies (e.g., "Appointed 1 July 2024", "Promoted from CFO to CEO on 1 March 2024")
- Source Chunk: the number from the [CHUNK n | ...] marker of the chunk containing the
  single-figure table row you took this director's figures from. Always cite a chunk number
  that actually appears in the context above — never invent one.

Extract data for both the current year and previous year for comparison, if available.

CRITICAL: Focus on actual payments made or due for the completed financial year.
Look for tables with clear monetary amounts (£, $, €, etc.) showing what each executive actually received.
Distinguish between different executives - do not mix compensation data between individuals.

{context}
---
Answer the question based on the above context: {question}
---
Provide the answer in the following format:
- **Financial Year:** [Year]
  - **Executive Director 1:**
    - **Name:** [Full name]
    - **Position:** [Title, e.g., Chief Executive Officer]
    - **Salary:** [Amount and currency]
    - **Benefits:** [Amount and currency]
    - **Pension:** [Amount and currency]
    - **Annual Bonus:** [Amount and currency]
    - **LTIP:** [Amount and currency]
    - **Other:** [Amount and currency, if any]
    - **Total:** [Amount and currency]
    - **Source Chunk:** [n]
  - **Executive Director 2:**
    - **Name:** [Full name]
    - **Position:** [Title]
    - **Salary:** [Amount and currency]
    - **Benefits:** [Amount and currency]
    - **Pension:** [Amount and currency]
    - **Annual Bonus:** [Amount and currency]
    - **LTIP:** [Amount and currency]
    - **Other:** [Amount and currency, if any]
    - **Total:** [Amount and currency]
    - **Source Chunk:** [n]
  - [Additional directors as applicable]

- **Previous Financial Year:** [Year] (if available)
  - [Same format as above for comparison]
"""

EXECUTIVE_PAY_QUESTION = """
What are the detailed remuneration breakdowns for all executive directors for the most recent completed financial year?
Extract the individual compensation components (salary, benefits, pension, bonus, LTIP, total) for each executive director.
Flag any pro-rata payments where an executive served only part of the year or changed roles mid-year.
Include previous year data for comparison if available.
"""

POLICY_PROMPT_TEMPLATE = """
You are a UK executive remuneration analyst reviewing the Directors' Remuneration Policy section of an Annual Report.
Carefully extract the remuneration policy details for each Executive Director from the provided context.
Focus on the policy that applies to the current/upcoming financial year.

CRITICAL RULES:
1. For executive_name, ALWAYS use the person's ACTUAL NAME (e.g., "Jason Windsor", "Milena Mondini de Focatiis") 
   - NEVER use titles like "CEO" or "CFO" in the executive_name field
   - If only a title is mentioned, search the context for the corresponding name
2. If an executive's salary has recently changed, use the NEW salary amount (the one currently in effect)
3. Express all opportunity percentages as plain numbers (e.g., 200 for 200% of salary, not "200%")
4. For shareholding guidelines, express as percentage of base salary (e.g., 400 for 400%)

FIELD DEFINITIONS:
- base_salary_amount: The executive's current/approved base salary (numeric amount)
- base_salary_currency: Currency code (GBP, USD, EUR)
- annual_bonus_max_percentage: Maximum annual bonus as % of base salary (e.g., 200 means 200% of salary)
- bonus_deferral_percentage: What % of earned bonus must be deferred (e.g., 50 means 50% of bonus)
- bonus_deferral_period_years: Number of years the deferred bonus is held
- ltip_max_percentage: Maximum LTIP award as % of base salary (e.g., 350 means 350% of salary) —
  the single highest achievable opportunity. Two different hybrid mechanisms affect this:
    - ADDITIVE hybrid (both elements granted in full, simultaneously, every year — e.g. a
      Performance Share Award of 425% PLUS a separate Restricted Share Award of 100%): this is
      the TOTAL, i.e. 525%.
    - SUBSTITUTIVE hybrid (the restricted element is granted "in lieu of"/"instead of" a portion
      of the performance element, usually at a discount — e.g. "up to 62.5% of salary in
      Restricted Shares in lieu of PSP, reducing PSP down to 125%"): this is simply the
      PERFORMANCE-ONLY maximum (e.g. 250%), NOT the sum — using the restricted element here
      REDUCES the performance element rather than adding to it.
- psp_max_percentage: the maximum PERFORMANCE-conditioned element (Performance Share Plan/Award)
  as % of salary, ASSUMING the restricted element is not used. Only set when a restricted-element
  provision exists at all (whether or not currently used).
- rsp_max_percentage: the maximum RESTRICTED/time-based element (Restricted Share Plan/Award,
  vesting on continued employment, no performance conditions or only a pass/fail underpin) as %
  of salary — the ceiling IF used. IMPORTANT: set this whenever the report states a maximum for
  this element, EVEN IF the company says it does not currently intend to use it (e.g. "the
  ability to grant Restricted Shares would remain, although not anticipated to be used") — a
  dormant/unused provision still has a real stated maximum and must be captured. Only leave both
  psp_max_percentage and rsp_max_percentage blank when there is no restricted-element provision
  in the policy at all.
- rsp_status: state "active" if the restricted element is actually being granted (current or
  recent awards exist), or "provision_unused" if the policy permits it but the company states no
  current intention to grant it / no such award has been made. Only relevant when
  rsp_max_percentage is set.
- ltip_hybrid_mechanism: state "additive" or "substitutive" per the definitions above. Only
  relevant when rsp_max_percentage is set. These often differ by individual executive (e.g. CEO
  vs CFO may have different % figures, but the mechanism type is usually the same for both).
- shareholding_guideline_percentage: Required shareholding as % of base salary (e.g., 400 means 400%)
- post_employment_shareholding_years: Years shareholding must be maintained after leaving
- policy_change_summary: Brief description of any notable policy changes
- Source Chunk: the number from the [CHUNK n | ...] marker of the chunk containing the policy
  table row these figures came from. Always cite a chunk number that actually appears in the
  context above — never invent one.

Look for sections covering "Implementation of Policy", "Remuneration Policy", "Base salary",
"Annual bonus", "LTIP opportunity", "Bonus deferral" and "Shareholding guidelines".
The key details may be contained in policy tables or narrative descriptions. Hybrid LTIP splits
are often stated per-executive in an "Implementation" table (e.g. "Jonny will be granted a
Performance Share Award (PSA) of 425% of salary and a Restricted Share Award (RSA) of 100% of
salary").

{context}
---
Answer the question based on the above context: {question}
---
Provide the answer in the following format for EACH Executive Director:
- **Executive Director:** [FULL NAME - not title]
  - **Position:** [Title, e.g., Chief Executive Officer]
  - **Financial Year:** [Year the policy applies to]
  - **Base Salary:** [Amount and currency, with effective date if mentioned]
  - **Annual Bonus Opportunity:** [Maximum % of base salary]
  - **Bonus Deferral:** [% of bonus deferred, for how many years]
  - **LTIP Opportunity:** [Maximum % of base salary]
  - **LTIP Split (only if a restricted-element provision exists):** [Performance element % of salary] + [Restricted element % of salary] — [Mechanism: additive/substitutive] — [Status: active/provision_unused] — omit this whole line if there is no restricted-element provision at all
  - **Shareholding Guideline:** [% of base salary required]
  - **Post-Employment Shareholding:** [Years]
  - **Policy Changes:** [Summary of any changes, or "No changes" if none]
  - **Source Chunk:** [n]
"""

POLICY_QUESTION = """
For each Executive Director, extract their remuneration policy details including:
1. Their FULL NAME (not just title) and position
2. Current/approved base salary (amount and currency)
3. Maximum annual bonus opportunity (as % of base salary)
4. Bonus deferral requirement (% of bonus deferred, and for how many years)
5. Maximum LTIP opportunity (as % of base salary)
6. Shareholding guideline (as % of base salary)
7. Post-employment shareholding period (years)
8. Any notable policy changes for the upcoming year

Focus on the current/upcoming financial year policy. Use actual names, not titles.
"""

LTIP_CONVERSION_PROMPT = """
Convert the following detailed LTIP (Long-Term Incentive Plan) analysis to JSON format according to the provided schema.

IMPORTANT:
- Preserve ALL the detailed information from the text analysis
- Maintain separation between different plan years
- Include all metrics, weights, targets and conditions exactly as described
- Do not lose any numerical values or specific details

PENDING TARGETS:
- If a metric's targets are described as pending / to be determined / to be disclosed in a
  later announcement, still emit the metric with its grant_year, metric_name and
  weight_percentage, set "targets_pending": true, and leave threshold/target/stretch/exceptional
  value fields null. NEVER copy targets from an earlier grant year into a later one.
- For metrics with fully disclosed targets, set "targets_pending": false (or omit it).

GRANT STATUS:
- Copy the plan's "Grant Status" line into the plan-level "grant_status" field: "granted" or
  "announced". If the text analysis gives no status, infer: "announced" when the plan is
  described as an award to be made in a future period, otherwise "granted".
- grant_status is INDEPENDENT of targets_pending. An announced award can have fully quantified
  targets (grant_status "announced", targets_pending false) — emit those real target values.

GRANT DATE:
- Copy the plan's "Grant Date" line into the plan-level "grant_date" field verbatim (e.g.
  "3 September 2024", "September 2025") and set "grant_date_is_stated": true.
- If the line says "not stated", or the analysis notes the date was assumed/expected/inferred
  from a prior year, leave "grant_date" null and set "grant_date_is_stated": false.
  Do NOT manufacture a date.

SOURCE ATTRIBUTION:
- Each metric in the text analysis is tagged with a "Source Chunk: n". Copy that integer into
  the metric's "source_chunk_id" field. If a metric has no chunk tag, omit the field (leave null).

Schema: {schema}

Text Analysis to Convert:
{text_content}

Return only valid JSON that matches the schema exactly, preserving all the detailed information from the analysis.
"""

STIP_CONVERSION_PROMPT = """
Convert the following detailed STIP (Short-Term Incentive Plan) analysis to JSON format according to the provided schema.

IMPORTANT: 
- Preserve ALL the detailed information from the text analysis
- Maintain separation between completed year results and upcoming year targets
- Include all metrics, weights, targets, actual performance, and payout percentages exactly as described
- For qualitative metrics, preserve the summary assessments provided
- Do not lose any numerical values or specific details

SOURCE ATTRIBUTION:
- Each item in the text analysis is tagged with a "Source Chunk: n". Copy that integer into
  the corresponding "source_chunk_id" field. If an item has no chunk tag, omit the field.

Schema: {schema}

Text Analysis to Convert:
{text_content}

Return only valid JSON that matches the schema exactly, preserving all the detailed information from the analysis.
"""

EXECUTIVE_PAY_CONVERSION_PROMPT = """
Convert the following detailed executive remuneration analysis to JSON format according to the provided schema.

IMPORTANT: 
- Preserve ALL the detailed information from the text analysis
- Maintain separation between different executives
- Include all compensation components (Salary, Benefits, Pension, Bonus, LTIP, Total) exactly as described
- Preserve currency information and exact amounts
- Set is_pro_rata to true if the executive served only part of the year or changed roles mid-year
- Include pro_rata_notes explaining the circumstance (e.g., "Appointed 1 July 2024")
- Do not lose any numerical values or specific details

SOURCE ATTRIBUTION:
- Each item in the text analysis is tagged with a "Source Chunk: n". Copy that integer into
  the corresponding "source_chunk_id" field. If an item has no chunk tag, omit the field.

Schema: {schema}

Text Analysis to Convert:
{text_content}

Return only valid JSON that matches the schema exactly, preserving all the detailed information from the analysis.
"""


POLICY_CONVERSION_PROMPT = """
Convert the following remuneration policy analysis to JSON format according to the provided schema.

CRITICAL RULES:
1. executive_name MUST be the person's ACTUAL NAME (e.g., "Jason Windsor"), NOT their title
   - If you see "CEO" or "CFO" as the name, look for the actual name in the text
   - If truly unavailable, use "Name Not Disclosed - [Title]"
2. All percentage fields should be plain numbers without % symbol (e.g., 200 not "200%")
3. financial_year should be the year the policy applies to (e.g., "2025")
4. Leave fields as null if information is genuinely not available - do not guess

FIELD MAPPINGS:
- base_salary_amount → numeric value only
- base_salary_currency → "GBP", "USD", "EUR", etc.
- annual_bonus_max_percentage → number (e.g., 200 for 200%)
- bonus_deferral_percentage → number (e.g., 50 for 50%)
- bonus_deferral_period_years → number (e.g., 3)
- ltip_max_percentage → number (e.g., 350 for 350%). For an ADDITIVE hybrid this is the TOTAL
  of performance + restricted elements; for a SUBSTITUTIVE hybrid this is the performance-only
  maximum (NOT the sum) — copy exactly what the text analysis states.
- psp_max_percentage / rsp_max_percentage → numbers, from the "LTIP Split" line if present.
  Copy rsp_max_percentage even when Status is "provision_unused" — a dormant maximum is still a
  real number to capture. Leave both null when the text analysis has no LTIP Split line at all.
- rsp_status → "active" or "provision_unused", copied from the LTIP Split line's Status.
- ltip_hybrid_mechanism → "additive" or "substitutive", copied from the LTIP Split line's
  Mechanism.
- shareholding_guideline_percentage → number (e.g., 400 for 400%)
- post_employment_shareholding_years → number (e.g., 2)
- source_chunk_id → the integer from the "Source Chunk" line for that executive

Schema: {schema}

Text Analysis to Convert:
{text_content}

Return ONLY valid JSON matching the schema. No markdown, no explanation.
"""

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_llm_config():
    return LLM_MODELS[LLM_PROVIDER]

def get_paths():
    return PATHS

def get_rag_config():
    return RAG_CONFIG

def get_schema_config():
    return SCHEMA_CONFIG

def get_ltip_prompt():
    return LTIP_PROMPT_TEMPLATE

def get_ltip_question():
    return LTIP_QUESTION

def get_stip_prompt():
    return STIP_PROMPT_TEMPLATE

def get_stip_question():
    return STIP_QUESTION

def get_executive_pay_prompt():
    return EXECUTIVE_PAY_PROMPT_TEMPLATE

def get_executive_pay_question():
    return EXECUTIVE_PAY_QUESTION

def get_policy_prompt():
    return POLICY_PROMPT_TEMPLATE

def get_policy_question():
    return POLICY_QUESTION

def get_ltip_conversion_prompt():
    return LTIP_CONVERSION_PROMPT

def get_stip_conversion_prompt():
    return STIP_CONVERSION_PROMPT

def get_executive_pay_conversion_prompt():
    return EXECUTIVE_PAY_CONVERSION_PROMPT

def get_policy_conversion_prompt():
    return POLICY_CONVERSION_PROMPT

def print_config_summary():
    """Print a summary of current configuration"""
    print("🔧 CURRENT CONFIGURATION")
    print("=" * 50)
    print(f"LLM Provider: {LLM_PROVIDER}")
    print(f"LLM Model: {LLM_MODELS[LLM_PROVIDER]}")
    print(f"RAG Retrieval K: {RAG_CONFIG['retrieval_k']}")
    print(f"Chunk Size: {RAG_CONFIG['chunk_size']}")
    print(f"Max Content Length: {SCHEMA_CONFIG['max_content_length']}")
    print("=" * 50)