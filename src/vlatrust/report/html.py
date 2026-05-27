"""Render a Scorecard as a single self-contained HTML file (inline CSS, no deps).

One human-facing artifact, openable straight in a browser, no external requests.
The headline Trust-Shift is shown with its confidence tier; every supporting
panel (fragility, calibration, reliability gaps, abstention) is laid out so a
reader sees *why* the score is what it is. An optional ``disclaimer`` banner is
rendered prominently — used to mark mock/synthetic input so a demo report is
never mistaken for an empirical measurement.
"""

from __future__ import annotations

from html import escape

from ..core.types import Scorecard

__all__ = ["scorecard_to_html"]


def _fmt(x: float | None, nd: int = 3) -> str:
    if x is None:
        return "<span class='na'>N/A</span>"
    return f"{x:.{nd}f}"


def _pair(ci) -> str:
    if ci is None:
        return "<span class='na'>N/A</span>"
    return f"[{ci[0]:.3f}, {ci[1]:.3f}]"


def scorecard_to_html(
    sc: Scorecard, *, title: str = "vlatrust Scorecard", disclaimer: str | None = None
) -> str:
    ts = sc.trust_shift
    score_txt = "N/A (no usable confidence)" if ts is None else f"{ts:.3f}"
    score_class = (
        "na" if ts is None else ("good" if ts >= 0.66 else "warn" if ts >= 0.33 else "bad")
    )

    banner = ""
    if disclaimer:
        banner = f"<div class='disclaimer'>⚠ {escape(disclaimer)}</div>"

    # fragility panel
    frag_rows = ""
    if sc.fragility is not None:
        for c in sc.fragility.curves:
            star = " ★" if c.modality == sc.fragility.most_fragile_modality else ""
            frag_rows += (
                f"<tr><td>{escape(c.modality)}{star}</td>"
                f"<td>{_fmt(c.fragility)}</td><td>{escape(c.mechanism)}</td></tr>"
            )
    frag_panel = (
        f"<table><tr><th>modality</th><th>fragility</th><th>mechanism</th></tr>{frag_rows}</table>"
        if frag_rows
        else "<p class='na'>no collapse curves</p>"
    )

    # calibration / reliability / conformal
    cal = sc.calibration
    cal_panel = (
        f"<p>ECE {_fmt(cal.ece)} · inverse-Brier {_fmt(cal.inverse_brier)} · n={cal.n}</p>"
        if cal is not None
        else "<p class='na'>uncalibrated (no usable confidence)</p>"
    )
    rel = sc.reliability
    if rel is not None:
        rel_panel = (
            f"<p>Δsucc {_fmt(rel.delta_succ)} {_pair(rel.delta_succ_ci)} "
            f"<b>{'claimable' if rel.succ_claimable else 'not claimable'}</b></p>"
            f"<p>Δcov {_fmt(rel.delta_cov)} {_pair(rel.delta_cov_ci)} "
            f"<b>{'claimable' if rel.cov_claimable else 'not claimable'}</b></p>"
        )
    else:
        rel_panel = "<p class='na'>no reference/target split</p>"
    conf = sc.conformal
    conf_panel = (
        f"<p>α={_fmt(conf.alpha, 2)} · q̂={_fmt(conf.q_hat)} · coverage={_fmt(conf.coverage)} · "
        f"abstention={_fmt(conf.abstention_rate)}{' · weighted' if conf.weighted else ''}</p>"
        if conf is not None
        else "<p class='na'>no conformal calibration</p>"
    )

    notes = "".join(f"<li>{escape(n)}</li>" for n in sc.notes)

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{escape(title)}</title>
<style>
  :root {{ color-scheme: light dark; }}
  body {{ font: 15px/1.5 system-ui, sans-serif; margin: 0; padding: 2rem; max-width: 880px; }}
  h1 {{ font-size: 1.4rem; margin: 0 0 .25rem; }}
  .sub {{ color: #888; margin-bottom: 1.5rem; }}
  .disclaimer {{ background: #fff3cd; color: #664d03; border: 1px solid #ffe69c;
    padding: .6rem .9rem; border-radius: 8px; margin-bottom: 1.25rem; font-weight: 600; }}
  .score {{ font-size: 3rem; font-weight: 700; line-height: 1; }}
  .score.good {{ color: #1a7f37; }} .score.warn {{ color: #9a6700; }}
  .score.bad {{ color: #cf222e; }} .score.na, .na {{ color: #888; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 1rem; margin-top: 1.5rem; }}
  .card {{ border: 1px solid #8884; border-radius: 10px; padding: 1rem; }}
  .card h2 {{ font-size: .95rem; margin: 0 0 .5rem; text-transform: uppercase; letter-spacing: .04em; color: #888; }}
  table {{ border-collapse: collapse; width: 100%; font-size: .9rem; }}
  th, td {{ text-align: left; padding: .25rem .5rem; border-bottom: 1px solid #8883; }}
  ul {{ margin: .25rem 0; padding-left: 1.1rem; }} code {{ font-size: .85rem; }}
</style></head>
<body>
{banner}
<h1>{escape(title)}</h1>
<div class="sub">Trust-Shift · confidence source: <code>{escape(sc.confidence_source.value)}</code>
 · physically valid: <b>{"yes" if sc.hard_valid else "no"}</b></div>
<div class="score {score_class}">{escape(score_txt)}</div>
<div class="grid">
  <div class="card"><h2>Fragility (collapse)</h2>{frag_panel}</div>
  <div class="card"><h2>Calibration</h2>{cal_panel}</div>
  <div class="card"><h2>Reliability gap</h2>{rel_panel}</div>
  <div class="card"><h2>Conformal abstention</h2>{conf_panel}</div>
</div>
<div class="card" style="margin-top:1rem"><h2>Notes</h2><ul>{notes}</ul></div>
</body></html>
"""
