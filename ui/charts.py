"""Plotly figures for the dashboard, light theme. They only draw numbers the API returned."""
from __future__ import annotations

import plotly.graph_objects as go

INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"  # validated categorical slots 1-3
BLUE_RAMP = ["#f0efec", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'
CAUSE_NAME = {"tariff": "Tariff", "volume": "Volume", "product_mix": "Product mix", "weather": "Weather", "waste": "New waste"}
WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def money(x: float, currency: str = "INR") -> str:
    if currency == "INR":
        return f"₹{x / 1e5:,.1f} lakh"
    return f"{currency} {x:,.0f}"


def money_range(r, currency: str = "INR") -> str:
    if currency == "INR":
        return f"₹{r[0] / 1e5:,.1f}–{r[1] / 1e5:,.1f} lakh"
    return f"{currency} {r[0]:,.0f}–{r[1]:,.0f}"


def _layout(fig: go.Figure, height: int, **kw) -> go.Figure:
    fig.update_layout(
        height=height, margin=dict(l=8, r=8, t=36, b=8), paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
        font=dict(family=FONT, size=13, color=INK_2), hoverlabel=dict(bgcolor="white", font=dict(family=FONT, color=INK)),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, font=dict(color=INK_2)), **kw)
    fig.update_xaxes(showgrid=False, linecolor=AXIS, tickfont=dict(color=INK_2), zeroline=False)
    fig.update_yaxes(gridcolor=GRID, gridwidth=1, linecolor=AXIS, tickfont=dict(color=MUTED), zeroline=False)
    return fig


def bridge(b: dict, currency: str, height: int = 400) -> go.Figure:
    """Waterfall of the change in percentage points: causes outside the factory's control in blue,
    new (controllable) waste in orange, the total in ink. Whiskers show each cause's range."""
    order = [x for x in b["bridge"] if x["cause"] != "waste"]
    order.sort(key=lambda x: -abs(x["pct_points"]))
    order += [x for x in b["bridge"] if x["cause"] == "waste"]
    labels, bases, values, colors, err_hi, err_lo, hover, text = [], [], [], [], [], [], [], []
    run = 0.0
    for x in order:
        v = x["pct_points"]
        labels.append(CAUSE_NAME.get(x["cause"], x["cause"]))
        bases.append(run if v >= 0 else run + v)
        values.append(abs(v))
        run += v
        colors.append(ORANGE if x["cause"] == "waste" else BLUE)
        top = run if v >= 0 else run - v
        err_hi.append(max(x["range"][1] - v, 0))
        err_lo.append(max(v - x["range"][0], 0))
        text.append(f"{v:+.2f}")
        hover.append(f"<b>{labels[-1]}</b><br>{v:+.2f} points (range {x['range'][0]:.2f} to {x['range'][1]:.2f})"
                     f"<br>about {money(abs(x.get('amount', 0)), currency)}<extra></extra>")
    total = b["total_change_pct"]
    labels.append("Total change")
    bases.append(0)
    values.append(total)
    colors.append(INK_2)
    err_hi.append(0)
    err_lo.append(0)
    text.append(f"+{total:.2f}%" if total >= 0 else f"{total:.2f}%")
    hover.append(f"<b>Total change</b><br>{total:+.2f}% of the {b['compare_to']} bill<br>"
                 f"{money(b['bill_reference'], currency)} → {money(b['bill_current'], currency)}<extra></extra>")
    fig = go.Figure(go.Bar(x=labels, y=values, base=bases, marker=dict(color=colors, cornerradius=4), width=0.62,
                           text=text, textposition="outside", textfont=dict(color=INK, size=13), cliponaxis=False,
                           error_y=dict(type="data", symmetric=False, array=[h if i < len(order) else 0 for i, h in enumerate(err_hi)],
                                        arrayminus=[0] * len(labels), color=INK_2, thickness=1.2, width=5),
                           hovertemplate=hover, showlegend=False))
    for name, color in (("Outside the factory's control", BLUE), ("Controllable waste", ORANGE)):
        fig.add_trace(go.Bar(x=[None], y=[None], marker_color=color, name=name, showlegend=True))
    fig = _layout(fig, height, barmode="overlay", bargap=0.3)
    fig.update_yaxes(title=dict(text=f"Points of the {b['compare_to']} bill", font=dict(color=MUTED, size=12)),
                     rangemode="tozero", range=[0, max(total, run) * 1.15])
    return fig


def savings(actions: list[dict], currency: str) -> go.Figure:
    """Saving range per action (low to high), in ranked order, first at the top."""
    acts = list(reversed(actions))
    names = [f"#{a['rank']} {a['action'].split(' (')[0]}" for a in acts]
    lo = [a["saving_money_per_year"][0] for a in acts]
    hi = [a["saving_money_per_year"][1] for a in acts]
    fig = go.Figure(go.Bar(
        y=names, x=[h - l for l, h in zip(lo, hi)], base=lo, orientation="h", marker=dict(color=BLUE, cornerradius=4), width=0.5,
        text=[f"{money_range(a['saving_money_per_year'], currency)} a year · {a['ease']}" for a in acts], textposition="outside", textfont=dict(color=INK, size=12), cliponaxis=False,
        hovertemplate=[f"<b>{a['action']}</b><br>{money_range(a['saving_money_per_year'], currency)} a year<br>"
                       f"{a['saving_kwh_per_year'][0]:,}–{a['saving_kwh_per_year'][1]:,} kWh · "
                       f"{a['saving_co2e_t_per_year'][0]}–{a['saving_co2e_t_per_year'][1]} t CO2e<extra></extra>" for a in acts]))
    fig = _layout(fig, 90 + 70 * len(acts), showlegend=False)
    fig.update_xaxes(title=dict(text="Saving a year", font=dict(color=MUTED, size=12)), rangemode="tozero",
                     range=[0, max(hi) * 1.9], tickformat="~s", showgrid=True, gridcolor=GRID)
    fig.update_yaxes(showgrid=False, tickfont=dict(color=INK, size=13))
    if currency == "INR":
        fig.update_xaxes(tickvals=[v for v in range(0, int(max(hi) * 1.9), 200_000)],
                         ticktext=[f"₹{v / 1e5:g}L" for v in range(0, int(max(hi) * 1.9), 200_000)])
    return fig


def assets(rows: list[dict]) -> go.Figure:
    rows = sorted(rows, key=lambda r: (r["asset"] == "unmetered", -r["kwh"]))[::-1]
    fig = go.Figure(go.Bar(
        y=[r["asset"].replace("_", " ") for r in rows], x=[100 * r["share_of_main"] for r in rows], orientation="h",
        marker=dict(color=[MUTED if r["asset"] == "unmetered" else BLUE for r in rows], cornerradius=4), width=0.6,
        text=[f"{100 * r['share_of_main']:.1f}%" for r in rows], textposition="outside", textfont=dict(color=INK), cliponaxis=False,
        hovertemplate=[f"<b>{r['asset']}</b><br>{r['kwh']:,.0f} kWh · {100 * r['share_of_main']:.1f}% of main"
                       + (f"<br>{r['change_pct']:+.1f}% vs previous quarter" if r.get("change_pct") is not None else "") + "<extra></extra>"
                       for r in rows]))
    fig = _layout(fig, 360, showlegend=False)
    fig.update_xaxes(title=dict(text="Share of the main meter (%)", font=dict(color=MUTED, size=12)), showgrid=True,
                     gridcolor=GRID, range=[0, max(100 * r["share_of_main"] for r in rows) * 1.2])
    fig.update_yaxes(showgrid=False, tickfont=dict(color=INK))
    return fig


def daily(d: dict, tariff_changes: list[str], currency: str) -> go.Figure:
    y = [c / 1e5 for c in d["cost"]] if currency == "INR" else d["cost"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d["dates"], y=y, mode="lines", line=dict(color=BLUE, width=2), name="Daily cost",
                             hovertemplate="%{x}<br>" + ("₹%{y:.2f} lakh" if currency == "INR" else "%{y:,.0f}") + "<extra></extra>"))
    off = [i for i, w in enumerate(d["working"]) if not w]
    fig.add_trace(go.Scatter(x=[d["dates"][i] for i in off], y=[y[i] for i in off], mode="markers", name="Sunday or holiday",
                             marker=dict(color=SURFACE, size=8, line=dict(color=BLUE, width=2)), hoverinfo="skip"))
    for t in tariff_changes:
        fig.add_vline(x=t, line=dict(color=INK_2, width=1))
        fig.add_annotation(x=t, y=1, yref="paper", text="Tariff rise", showarrow=False, xanchor="left", yanchor="bottom",
                           font=dict(color=INK_2, size=12))
    fig = _layout(fig, 300)
    fig.update_yaxes(title=dict(text="₹ lakh a day" if currency == "INR" else "Cost a day", font=dict(color=MUTED, size=12)))
    return fig


def hourly(p: dict) -> go.Figure:
    """Average load by hour, working vs non-working days, against the band the asset should drop to."""
    h = p["hours"]
    lo, hi = p["expected_off_hours_kw"]
    s0, s1 = p["shift_hours"]
    fig = go.Figure()
    fig.add_vrect(x0=s0 - 0.5, x1=s1 - 0.5, fillcolor="#f0efec", line_width=0, layer="below")
    fig.add_annotation(x=(s0 + s1) / 2 - 0.5, y=1, yref="paper", text="Production shift", showarrow=False, yanchor="bottom",
                       font=dict(color=MUTED, size=12))
    fig.add_trace(go.Scatter(x=h + h[::-1], y=[hi] * 24 + [lo] * 24, fill="toself", fillcolor="rgba(27,175,122,0.15)",
                             line=dict(width=0), name="Expected outside the shift", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=h, y=p["working_day_kw"], mode="lines+markers", line=dict(color=BLUE, width=2), marker=dict(size=6),
                             name="Working days", hovertemplate="%{x}:00 · %{y:.1f} kW<extra>Working days</extra>"))
    fig.add_trace(go.Scatter(x=h, y=p["non_working_day_kw"], mode="lines+markers", line=dict(color=ORANGE, width=2),
                             marker=dict(size=6), name="Sundays and holidays",
                             hovertemplate="%{x}:00 · %{y:.1f} kW<extra>Sundays and holidays</extra>"))
    fig = _layout(fig, 330)
    fig.update_xaxes(title=dict(text="Hour of day", font=dict(color=MUTED, size=12)), tickvals=list(range(0, 24, 2)),
                     ticktext=[f"{x:02d}:00" for x in range(0, 24, 2)], range=[-0.5, 23.5])
    fig.update_yaxes(title=dict(text="Average kW", font=dict(color=MUTED, size=12)), rangemode="tozero")
    return fig


def heatmap(p: dict) -> go.Figure:
    z = p["heatmap_kw"]
    fig = go.Figure(go.Heatmap(z=z, x=[f"{x:02d}" for x in range(24)], y=WEEKDAYS, xgap=2, ygap=2,
                               colorscale=[[i / (len(BLUE_RAMP) - 1), c] for i, c in enumerate(BLUE_RAMP)],
                               colorbar=dict(title=dict(text="kW", font=dict(color=MUTED)), thickness=10, outlinewidth=0),
                               hovertemplate="%{y} %{x}:00 · %{z:.1f} kW<extra></extra>"))
    fig = _layout(fig, 290)
    fig.update_yaxes(autorange="reversed", showgrid=False, tickfont=dict(color=INK_2))
    fig.update_xaxes(title=dict(text="Hour of day", font=dict(color=MUTED, size=12)))
    return fig


def nightly(p: dict, step_from: str | None) -> go.Figure:
    n = p["nightly_kw"]
    fig = go.Figure(go.Scatter(x=n["dates"], y=n["kw"], mode="lines", line=dict(color=BLUE, width=2), name="Night load (00:00-05:00)",
                               hovertemplate="%{x}<br>%{y:.1f} kW<extra></extra>"))
    if step_from:
        fig.add_vline(x=step_from, line=dict(color=ORANGE, width=2))
        fig.add_annotation(x=step_from, y=1, yref="paper", text=f"Stepped up on {step_from}", showarrow=False, xanchor="left",
                           yanchor="bottom", font=dict(color=ORANGE, size=12))
    fig = _layout(fig, 260, showlegend=False)
    fig.update_yaxes(title=dict(text="Average kW, 00:00-05:00", font=dict(color=MUTED, size=12)), rangemode="tozero")
    return fig
