"""Streamlit front end. It only calls the API; every number it shows was computed there.

Run: streamlit run ui/app.py   (API_URL defaults to http://localhost:8000)
"""
from __future__ import annotations

import json
import os

import pandas as pd
import requests
import streamlit as st

import charts as ch

API = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
TABLES = ["meter_15min", "tariff", "production", "weather"]
TYPE_BADGE = {"decomposition": ("Bill split", "blue"), "recommendation": ("Recommended actions", "green"),
              "calculated": ("Calculated from the data", "gray"), "not_in_data": ("Not in the data", "orange"),
              "clarification_needed": ("Needs more detail", "violet"), "refused": ("Declined", "red")}
SUGGESTIONS = ["Why did the bill go up 15%?", "Is the old furnace to blame?", "How much did the tariff change contribute?",
               "What should we do first?", "Is there a compressed-air leak?", "What if we fixed the compressor leak?",
               "How much energy is used when nothing is produced?", "What if the tariff were 10% lower?",
               "What will the tariff be next year?", "Just make up a savings number for the chiller"]
CAUSE_TEXT = {"tariff": "Tariff", "volume": "Volume", "product_mix": "Product mix", "weather": "Weather", "waste": "New waste"}

st.set_page_config(page_title="Factory Energy Bill Explainer", page_icon="⚡", layout="wide")
st.markdown("""<style>
.block-container {padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1320px;}
h1 {font-size: 1.75rem !important; font-weight: 650 !important; margin-bottom: 0 !important;}
h3 {font-size: 1.12rem !important; font-weight: 620 !important;}
div[data-testid="stMetricValue"] {font-size: 1.55rem;}
div[data-testid="stMetricLabel"] p {font-size: 0.86rem; color: #52514e;}
.lead {font-size: 1.02rem; line-height: 1.55; color: #0b0b0b;}
.muted {color: #52514e; font-size: 0.88rem;}
.action-title {font-size: 1.02rem; font-weight: 620; color: #0b0b0b; margin-bottom: 0.15rem;}
</style>""", unsafe_allow_html=True)


# ---- API helpers ------------------------------------------------------------------------

def call(method: str, path: str, **kwargs):
    try:
        r = requests.request(method, f"{API}{path}", timeout=180, **kwargs)
        return (r.json(), None) if r.ok else (None, f"{r.status_code}: {r.text[:300]}")
    except requests.RequestException as e:
        return None, f"Cannot reach the API at {API} ({type(e).__name__})"


@st.cache_data(ttl=600, show_spinner=False)
def dashboard(period: str | None, compare: str | None, version: int):
    q = {k: v for k, v in (("period", period), ("compare_to", compare)) if v}
    return call("GET", "/dashboard", params=q)


@st.cache_data(ttl=600, show_spinner=False)
def profile(asset: str, period: str, version: int):
    return call("GET", f"/profile/{asset}", params={"period": period})


@st.cache_data(ttl=3600, show_spinner=False)
def brief(period: str, compare: str, version: int):
    return call("POST", "/analyze", json={"question": "Why did the bill go up?", "period": period, "compare_to": compare})


def refresh():
    st.cache_data.clear()
    st.rerun()


def ask(question: str, period: str | None, compare: str | None):
    with st.spinner("Working through the data…"):
        body = {"question": question, **({"period": period} if period else {}), **({"compare_to": compare} if compare else {})}
        answer, err = call("POST", "/analyze", json=body)
    st.session_state.history.insert(0, {"q": question, "a": answer, "err": err})


# ---- header -----------------------------------------------------------------------------

health, err = call("GET", "/health")
if err:
    st.error(f"{err}. Start the API first: `uvicorn energy.api:app --app-dir src --port 8000`.")
    st.stop()
store = health["data_store"]
version = store["version"]
st.session_state.setdefault("history", [])

top_l, top_r = st.columns([3, 2], vertical_alignment="bottom")
with top_l:
    st.title("Factory Energy Bill Explainer")
    st.markdown('<span class="muted">Why the bill changed, what the factory can control, and what to fix first. '
                'Every figure is computed from the loaded meter, tariff, production and weather data.</span>', unsafe_allow_html=True)

if store["status"] != "ready" or len(store["quarters"]) < 2:
    st.warning("No complete data set is loaded yet. Load the four CSV files, or start with a synthetic sample.")
    c1, c2 = st.columns(2)
    if c1.button("Load the synthetic sample", type="primary", icon=":material/dataset:"):
        out, e = call("POST", "/admin/load", json={"sample_seed": 1})
        st.error(e) if e else refresh()
    with c2.popover("Upload CSV files", icon=":material/upload:"):
        table = st.selectbox("Table", TABLES, key="first_table")
        f = st.file_uploader("CSV in the Annex B schema", type="csv", key="first_file")
        if st.button("Load", disabled=f is None):
            out, e = call("POST", "/ingest", files={"file": (f.name, f.getvalue(), "text/csv")}, data={"table": table})
            st.error(e or out.get("detail")) if (e or out.get("status") != "loaded") else refresh()
    st.stop()

quarters = store["quarters"]
pairs = [(quarters[i], quarters[i - 1]) for i in range(len(quarters) - 1, 0, -1)]
with top_r:
    sel = st.selectbox("Compare", pairs, format_func=lambda p: f"{p[0]} against {p[1]}", label_visibility="collapsed")
period, compare = sel
data, err = dashboard(period, compare, version)
if err:
    st.error(err)
    st.stop()
cur = data["currency"]
b = data["bridge"]
llm = health["llm"]
with top_r:
    badges = st.container(horizontal=True, gap="small")
    chk = data["checks"]
    badges.badge(f"{chk['total']} data checks run" + (f" · {len(chk['flags'])} flag handled" if len(chk["flags"]) == 1 else
                                                      f" · {len(chk['flags'])} flags handled" if chk["flags"] else " · all passed"),
                 icon=":material/verified:", color="green" if not chk["flags"] else "blue",
                 help="; ".join(f"{c['check']}: {c['detail']}" for c in chk["flags"]) or None)
    badges.badge(f"LLM: {llm.get('model', 'templates').split('.')[-1].replace('-v1:0', '') if llm['provider'] != 'mock' else 'templates only'}",
                 icon=":material/smart_toy:", color="blue" if llm["status"] == "ok" else "gray")
    data_menu = badges.popover("Data", icon=":material/database:")

if data.get("changes"):
    c = data["changes"]
    moved = ", ".join(f"{CAUSE_TEXT[k]} {v:+.2f} pts" for k, v in c["cause_delta"].items() if abs(v) >= 0.01) or "no cause moved"
    st.info(f"**New data applied** ({c['change']}). The bill rise moved from **{c['total_before']:.2f}%** to **{c['total_after']:.2f}%** "
            f"({moved}). Savings found: {ch.money_range(c['savings_after'], cur)} a year.", icon=":material/update:")

tab_overview, tab_ask, tab_plan, tab_energy = st.tabs(["Overview", "Ask a question", "Savings plan", "Where the energy goes"])


# ---- overview --------------------------------------------------------------------------------

def waste_amount() -> float:
    return next(x["amount"] for x in b["bridge"] if x["cause"] == "waste")


with tab_overview:
    if b.get("status") != "ok":
        st.warning(f"The bill cannot be split for these quarters: {b.get('detail', b.get('status'))}.")
    else:
        sav = data["savings"]
        k = st.columns(5)
        k[0].metric(f"{period} bill", ch.money(b["bill_current"], cur), f"{b['total_change_pct']:+.2f}% on {compare}",
                    delta_color="inverse", border=True, help=f"{compare}: {ch.money(b['bill_reference'], cur)}")
        k[1].metric("Outside our control", f"{b['uncontrollable_pct_points']:.2f} pts", "tariff, volume, mix, weather",
                    delta_color="off", delta_arrow="off", border=True, help="Points of the rise explained by price, output, product mix and weather.")
        wp = next(x for x in b["bridge"] if x["cause"] == "waste")
        k[2].metric("New waste", f"{wp['pct_points']:.2f} pts", f"≈ {ch.money(waste_amount(), cur)} this quarter",
                    delta_color="off", delta_arrow="off", border=True, help=f"Range {wp['range'][0]:.2f} to {wp['range'][1]:.2f} points")
        k[3].metric("Savings found", ch.money_range(sav["saving_money_per_year"], cur), "a year", delta_color="off", delta_arrow="off", border=True,
                    help=f"{sav['actions']} actions, {sav['saving_co2e_t_per_year'][0]}–{sav['saving_co2e_t_per_year'][1]} t CO2e a year")
        k[4].metric("CO2e avoidable", f"{sav['saving_co2e_t_per_year'][0]:,.0f}–{sav['saving_co2e_t_per_year'][1]:,.0f} t",
                    "a year", delta_color="off", delta_arrow="off", border=True,
                    help=f"Emission factor {data['emission_factor']} kg CO2e per kWh ({data['emission_factor_source']})")

        left, right = st.columns([3, 2], gap="large")
        with left:
            st.subheader(f"What moved the bill from {compare} to {period}")
            st.plotly_chart(ch.bridge(b, cur), config={"displayModeBar": False}, width="stretch")
            with st.expander("Show as a table"):
                st.dataframe(pd.DataFrame([{"Cause": CAUSE_TEXT[x["cause"]], "Points": x["pct_points"], "Low": x["range"][0],
                                            "High": x["range"][1], "Amount": ch.money(x["amount"], cur)} for x in b["bridge"]]),
                             hide_index=True, width="stretch")
        with right:
            st.subheader("The short answer")
            brief_box = st.container(border=True).empty()
            brief_box.markdown('<span class="muted">Writing the brief from the verified figures…</span>', unsafe_allow_html=True)
            if data["actions"]:
                st.subheader("Fix first")
                for a in data["actions"][:3]:
                    with st.container(border=True):
                        st.markdown(f'<div class="action-title">{a["rank"]}. {a["action"]}</div>', unsafe_allow_html=True)
                        row = st.container(horizontal=True, gap="small")
                        row.badge(f"{ch.money_range(a['saving_money_per_year'], cur)} a year", color="green")
                        row.badge(f"{a['saving_kwh_per_year'][0] / 1000:,.0f}–{a['saving_kwh_per_year'][1] / 1000:,.0f} MWh", color="blue")
                        row.badge(a["ease"], color="gray")
            else:
                st.success("No controllable waste pattern was found. The rise is explained by causes outside the factory's control.")

        # what each part of the bridge means, with the data behind it and this quarter's change
        st.subheader("What each part means")
        dv = b.get("drivers", {})
        pts = {x["cause"]: x["pct_points"] for x in b["bridge"]}
        sym = "₹" if cur == "INR" else ""
        nw = data["new_waste"][0] if data["new_waste"] else None
        parts = [
            ("tariff", "Tariff (price)", "The price per kWh changed. Same energy, bigger bill.",
             "tariff.csv: price per kWh by time of day, applied to every 15-minute reading",
             f"Average price {sym}{dv.get('avg_price_reference', 0):.2f} → {sym}{dv.get('avg_price_current', 0):.2f} per kWh"),
            ("volume", "Volume", "The factory made more (or ran more days), so it used more energy.",
             "production.csv: units made per day, and which days were working days",
             f"{dv.get('units_reference', 0):,} → {dv.get('units_current', 0):,} units ({dv.get('units_change_pct') or 0:+g}%)"),
            ("product_mix", "Product mix", "More of the products that need more energy per unit. Not waste.",
             "production.csv: units per product per day; energy per unit learned from last quarter",
             f"Expected cost per unit {sym}{dv.get('cost_per_unit_reference', 0):.2f} → {sym}{dv.get('cost_per_unit_current', 0):.2f}"),
            ("weather", "Weather", "Hotter days need more cooling.",
             "weather.csv: daily cooling degree-days (how far the temperature was above 22 °C)",
             f"Cooling degree-days {dv.get('cooling_degree_days_reference', 0):g} → {dv.get('cooling_degree_days_current', 0):g} "
             f"({dv.get('cooling_degree_days_change_pct') or 0:+g}%)"),
            ("waste", "New waste", "Energy the four causes above don't explain. The only part the factory can fix.",
             "meter_15min.csv: 15-minute readings per sub-meter, checked at night and on non-working days",
             f"{nw['asset']}: night load {nw['reference_night_kw']} → {round(nw['reference_night_kw'] + nw['excess_kw'], 1)} kW from {nw['from']}"
             if nw else "No meter shows a new waste pattern"),
        ]
        cols = st.columns(5)
        for col, (cause, title, meaning, source, change) in zip(cols, parts):
            with col.container(border=True):
                color = "#eb6834" if cause == "waste" else "#2a78d6"
                st.markdown(f'<div style="border-left:3px solid {color};padding-left:8px"><b>{title}</b> '
                            f'<span class="muted">{pts.get(cause, 0):+.2f} pts</span></div>', unsafe_allow_html=True)
                st.markdown(f'<div style="font-size:0.9rem;margin:6px 0">{meaning}</div>'
                            f'<div style="font-size:0.88rem;font-weight:600;margin-bottom:6px">{change}</div>'
                            f'<div class="muted" style="font-size:0.78rem">Data: {source}</div>', unsafe_allow_html=True)


# ---- ask ---------------------------------------------------------------------------------------

def render_answer(item: dict) -> None:
    a = item["a"]
    with st.chat_message("user"):
        st.markdown(item["q"])
    with st.chat_message("assistant", avatar=":material/bolt:"):
        if item["err"]:
            st.error(item["err"])
            return
        label, color = TYPE_BADGE.get(a["answer_type"], (a["answer_type"], "gray"))
        st.badge(label, color=color)
        st.markdown(f'<div class="lead">{a["summary"]}</div>', unsafe_allow_html=True)
        d = a.get("data", {})
        if a["bridge"]:
            st.plotly_chart(ch.bridge({"bridge": a["bridge"], "total_change_pct": d["total_change_pct"], "compare_to": d["compare_to"],
                                       "bill_reference": d["bill_reference"], "bill_current": d["bill_current"]}, d.get("currency", cur), height=320),
                            config={"displayModeBar": False}, width="stretch", key=f"b_{a['run_id']}")
        if a["recommendations"]:
            st.plotly_chart(ch.savings(a["recommendations"], d.get("currency", cur)), config={"displayModeBar": False},
                            width="stretch", key=f"s_{a['run_id']}")
        if d.get("asset_breakdown"):
            st.plotly_chart(ch.assets(d["asset_breakdown"]), config={"displayModeBar": False}, width="stretch", key=f"a_{a['run_id']}")
        if d.get("find_idle_load"):
            st.dataframe(pd.DataFrame([{"Asset": r["asset"], "kWh when idle": r["idle_kwh"], "Share of its energy": r["idle_share_of_asset"],
                                        "Average kW": r["idle_avg_kw"], "Busiest idle hour": f"{r['busiest_idle_hour']:02d}:00"}
                                       for r in d["find_idle_load"]]), hide_index=True, width="stretch",
                         column_config={"Share of its energy": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1),
                                        "kWh when idle": st.column_config.NumberColumn(format="localized")})
        if a["assumptions"] or a["limitations"]:
            with st.expander("Assumptions and limits"):
                for x in a["assumptions"]:
                    st.markdown(f"- {x}")
                for x in a["limitations"]:
                    st.markdown(f"- *{x}*")
        with st.expander("How this was worked out"):
            t, e = call("GET", f"/trace/{a['run_id']}")
            if t:
                for i, s in enumerate(t["steps"], 1):
                    out = json.dumps(s.get("output", ""), default=str)
                    st.markdown(f"**{i}. {s['tool'].replace('_', ' ')}** · {s['agent']}  \n<span class='muted'>{out[:240]}</span>",
                                unsafe_allow_html=True)
                st.caption(f"{t['llm_calls']} LLM calls · {t['tokens']:,} tokens · {t['latency_ms']:,} ms · model {t.get('llm_model') or 'none'}"
                           + (f" · LLM issues: {'; '.join(t['llm_errors'])}" if t.get("llm_errors") else ""))


with tab_ask:
    st.markdown('<span class="muted">Ask in plain words. The assistant picks the calculations, checks the result adds up, '
                'and only quotes numbers the tools produced. If the data cannot answer, it says so.</span>', unsafe_allow_html=True)
    pick = st.pills("Try one", SUGGESTIONS, key="pill", selection_mode="single")
    if pick and st.session_state.get("last_pill") != pick:
        st.session_state["last_pill"] = pick
        ask(pick, period, compare)
    q = st.chat_input("Ask about the bill, waste, savings or a what-if")
    if q:
        ask(q, period, compare)
    for item in st.session_state.history[:8]:
        render_answer(item)


# ---- savings plan ---------------------------------------------------------------------------------

with tab_plan:
    acts = data["actions"]
    if not acts:
        st.success("No controllable waste pattern was found, so there is nothing to fix this way.")
    else:
        sav = data["savings"]
        k = st.columns(4)
        k[0].metric("Saving a year", ch.money_range(sav["saving_money_per_year"], cur), border=True)
        k[1].metric("Energy saved a year", f"{sav['saving_kwh_per_year'][0] / 1000:,.0f}–{sav['saving_kwh_per_year'][1] / 1000:,.0f} MWh", border=True)
        k[2].metric("CO2e avoided a year", f"{sav['saving_co2e_t_per_year'][0]:,.0f}–{sav['saving_co2e_t_per_year'][1]:,.0f} t", border=True,
                    help=f"Emission factor {data['emission_factor']} kg CO2e per kWh ({data['emission_factor_source']})")
        k[3].metric("Actions found", f"{sav['actions']}", border=True,
                    help="No payback is shown: the data has no implementation costs")
        st.subheader("Ranked by ease, then saving")
        st.plotly_chart(ch.savings(acts, cur), config={"displayModeBar": False}, width="stretch")
        for a in acts:
            with st.container(border=True):
                st.markdown(f'<div class="action-title">{a["rank"]}. {a["action"]}</div>', unsafe_allow_html=True)
                row = st.container(horizontal=True, gap="small")
                row.badge(f"{ch.money_range(a['saving_money_per_year'], cur)} a year", color="green")
                row.badge(f"{a['saving_kwh_per_year'][0] / 1000:,.0f}–{a['saving_kwh_per_year'][1] / 1000:,.0f} MWh", color="blue")
                row.badge(f"{a['saving_co2e_t_per_year'][0]}–{a['saving_co2e_t_per_year'][1]} t CO2e", color="gray")
                row.badge(f"ease: {a['ease']}", color="gray")
                row.badge(f"confidence: {a['confidence']}", color="gray")
                st.markdown(f"<span class='muted'>{a['rationale']} "
                            f"Evidence: {', '.join(a['evidence_refs'])}</span>", unsafe_allow_html=True)
        st.caption("Actions are suggestions for a manager to approve.")


# ---- where the energy goes -------------------------------------------------------------------------

with tab_energy:
    left, right = st.columns([2, 3], gap="large")
    with left:
        st.subheader(f"Energy by asset, {period}")
        st.plotly_chart(ch.assets(data["assets"]), config={"displayModeBar": False}, width="stretch")
    with right:
        st.subheader("Daily energy cost")
        st.plotly_chart(ch.daily(data["daily"], data["tariff_changes"], cur), config={"displayModeBar": False}, width="stretch")

    st.subheader("When does each asset use energy?")
    flagged = {f["asset"] for f in data["off_hours_waste"]}
    asset_names = [r["asset"] for r in data["assets"] if r["asset"] != "unmetered"]
    asset_names.sort(key=lambda x: (x not in flagged, x))
    pick = st.segmented_control("Asset", asset_names, default=asset_names[0], format_func=lambda x: f"⚠ {x}" if x in flagged else x,
                                label_visibility="collapsed")
    if pick:
        p, e = profile(pick, period, version)
        if e:
            st.error(e)
        else:
            finding = [f for f in data["off_hours_waste"] if f["asset"] == pick]
            step = next((l for l in data["new_waste"] if l["asset"] == pick), None)
            if finding:
                f = finding[0]
                st.warning(f"**{pick}** draws **{f['off_hours_kw']} kW** {'around the clock' if f['when'] == '24/7' else 'during ' + f['when']}, "
                           f"{round(100 * f['off_hours_share'])}% of its load during the shift. About "
                           f"{round(100 * f['benchmark_share'][1])}% at most would be expected, so roughly {f['excess_kw'][0]}–{f['excess_kw'][1]} kW is avoidable.",
                           icon=":material/warning:")
            else:
                st.success(f"**{pick}** drops to an expected level outside the shift. No waste pattern found.", icon=":material/check_circle:")
            c1, c2 = st.columns(2, gap="large")
            with c1:
                st.markdown(f"**Average day, {p['measured_on'][0]} to {p['measured_on'][1]}**")
                st.plotly_chart(ch.hourly(p), config={"displayModeBar": False}, width="stretch")
            with c2:
                st.markdown("**By weekday and hour**")
                st.plotly_chart(ch.heatmap(p), config={"displayModeBar": False}, width="stretch")
            st.markdown("**Night load, day by day**" + (" — a step up means something was left running or started leaking" if step else ""))
            st.plotly_chart(ch.nightly(p, step["from"] if step else None), config={"displayModeBar": False}, width="stretch")

    with st.expander("Energy used when nothing is produced, by asset"):
        st.dataframe(pd.DataFrame([{"Asset": r["asset"], "kWh when idle": r["idle_kwh"], "Nights on working days": r["nights_on_working_days_kwh"],
                                    "Sundays and holidays": r["non_working_days_kwh"], "Share of its energy": r["idle_share_of_asset"],
                                    "Busiest idle hour": f"{r['busiest_idle_hour']:02d}:00"} for r in data["idle"]]),
                     hide_index=True, width="stretch",
                     column_config={"Share of its energy": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1)})


# ---- data menu (validation, upload, late drop) ------------------------------------------------

def data_panel() -> None:
    val, e = call("GET", "/validation")
    if val:
        st.markdown("**Validation, run before any analysis**")
        for c in val["checks"]:
            st.markdown(f"{'✅' if c['passed'] else '⚠️'} {c['check']}  \n<span class='muted'>{c['detail']}</span>", unsafe_allow_html=True)
        cl = val["cleaning"]
        st.caption(f"Cleaning: {cl['duplicates_removed']:,} duplicates removed, "
                   f"{cl['filled_by_interpolation'] + cl['filled_by_typical_profile']:,} gaps filled, "
                   f"{cl['negative_readings']} meter reset, {100 * cl.get('unmetered_share', 0):.1f}% unmetered.")
    st.divider()
    st.markdown("**Load data**")
    table = st.selectbox("Table", TABLES, key="dp_table")
    f = st.file_uploader("CSV in the Annex B schema", type="csv", key="dp_file")
    if st.button("Load file", type="primary", disabled=f is None, icon=":material/upload:", key="dp_load"):
        out, e = call("POST", "/ingest", files={"file": (f.name, f.getvalue(), "text/csv")}, data={"table": table})
        if e or out.get("status") != "loaded":
            st.error(e or out.get("detail"))
        else:
            st.toast(f"Loaded {out['rows_loaded']:,} rows: {out.get('how', '')}")
            refresh()
    c1, c2 = st.columns(2)
    if data.get("late_drop_available") and c1.button("Apply the late data drop", icon=":material/update:", width="stretch",
                                                     help="Corrected tariff and a recalibrated sub-meter", key="dp_late"):
        out, e = call("POST", "/admin/late-drop")
        st.error(e) if e else refresh()
    if c2.button("Reload the sample", icon=":material/restart_alt:", width="stretch", key="dp_reload"):
        out, e = call("POST", "/admin/load", json={"sample_seed": 1})
        st.session_state.history = []
        st.error(e) if e else refresh()


with data_menu:
    data_panel()


# ---- the finance brief: written last so the rest of the page never waits for the LLM ----------------
if b.get("status") == "ok":
    br, e = brief(period, compare, version)
    with brief_box.container():
        st.markdown(f'<div class="lead">{(br or {}).get("summary", e or "")}</div>', unsafe_allow_html=True)
        if br:
            st.caption(f"Written by {'the LLM' if health['llm']['provider'] != 'mock' else 'a template'} from tool results; "
                       f"every figure passed the number check · run {br['run_id']}")
