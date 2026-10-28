/* If they pit now: what the stop costs, where they'd rejoin, the order around them after the stop, and their win,
   podium and points chances now and after (the race outcome model asked again). Shared by f1.html (real races) and
   f1-25.html (your game): PitNow.html(answer, name, nameOf, colourOf, you), PitNow.cell(rejoin) for the tables. */
(() => {
  "use strict";
  const esc = s => String(s ?? "").replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const pc = v => v == null ? "–" : v > 0 && v < 0.005 ? "<1%" : v > 0.995 && v < 1 ? ">99%" : Math.round(v * 100) + "%";
  const UNDER = { green: "", SC: " under the safety car", VSC: " under the VSC" };
  const TYRE = { SOFT: "soft", MEDIUM: "medium", HARD: "hard", "INTER/WET": "wet-weather" };

  function html(w, name, nameOf, colourOf, you = false) {
    if (!w) return "";
    const who = you ? "You" : esc(name), them = you ? "you" : "them";
    const parts = [w.ahead ? `${w.ahead.gap.toFixed(1)} s behind ${esc(nameOf(w.ahead.driver))}` : "still in the lead",
                   w.behind ? `${w.behind.gap.toFixed(1)} s ahead of ${esc(nameOf(w.behind.driver))}` : null].filter(Boolean);
    const range = w.range[0] !== w.range[1] ? ` P${Math.min(...w.range)} to P${Math.max(...w.range)}, depending on the stop.` : "";
    const cost = w.under === "red" ? "Under the red flag the cars wait in the pit lane, so new tyres cost no time."
      : `A stop costs about ${Math.round(w.cost)} s here${UNDER[w.under] || ""}.`;
    let out = `<p class="why-line why-window"><b>${cost}</b> ${who} would rejoin P${w.position}: ${parts.join(", ")}.${range}</p>`;
    // The order after the stop: two cars ahead, the driver, two behind (the gap from the driver to each). Ahead or
    // behind goes by the place: a gap under 0.05 s is rounded to 0.0
    const rows = [...w.around.filter(a => a.position < w.position), { driver: w.driver, position: w.position, me: true },
                  ...w.around.filter(a => a.position > w.position)];
    out += `<div class="pn-order" aria-label="The order after the stop">` + rows.map(r => `<div class="pn-r${r.me ? " me" : ""}"><b>P${r.position}</b>` +
      `<i style="background:${esc(colourOf(r.driver))}"></i><span>${esc(nameOf(r.driver))}${r.me ? " (after the stop)" : ""}</span>` +
      `<em>${r.me ? "" : (r.position > w.position ? "+" : "−") + Math.abs(r.gap).toFixed(1) + " s"}</em></div>`).join("") + `</div>`;
    // The race outcome: now and if they pit now
    const c = w.chances, way = (a, b, up) => Math.abs(b - a) < (up === "place" ? 1 : 0.005) ? "" : (up === "place" ? b < a : b > a) ? "up" : "dn";
    out += `<div class="pn-chg">` + [["Win", c.win], ["Podium", c.podium], ["Points", c.points]].map(([n, [a, b]]) =>
      `<span>${n} <b>${pc(a)}</b> → <b class="${way(a, b)}">${pc(b)}</b></span>`).join("") +
      `<span>Likely <b>P${c.likely[0]}</b> → <b class="${way(c.likely[0], c.likely[1], "place")}">P${c.likely[1]}</b></span></div>` +
      `<p class="note">On new ${TYRE[w.tyre] || "fresh"} tyres (the tyre model's pick), if nobody else pits on this lap. The stop's cost is
       learned from 2022-2025; the race outcome model is asked again with ${them} moved there.</p>`;
    return out;
  }

  // In the tables: "P5" and who they'd be behind
  function cell(r) {
    if (!r) return "<span></span>";
    const text = r.ahead ? `${r.ahead.gap.toFixed(1)} s behind ${r.ahead.driver}` : "still leading";
    return `<span class="rj" title="If they pit now: P${r.position}, ${esc(text)}">P${r.position}<small>${esc(text)}</small></span>`;
  }

  document.head.insertAdjacentHTML("beforeend", `<style>
.rj{text-align:right;font-family:var(--f-num);line-height:1.05;white-space:nowrap}
.rj small{display:block;color:#9aa0ae;font-size:11px;font-family:var(--f-ui);overflow:hidden;text-overflow:ellipsis}
.pn-order{display:grid;gap:3px;margin:8px 0;font-size:14px}
.pn-r{display:grid;grid-template-columns:34px 4px minmax(0,1fr) 56px;gap:8px;align-items:center;padding:4px 8px;border-radius:4px;background:#1d1f2b}
.pn-r b{font-family:var(--f-num);font-weight:600} .pn-r i{align-self:stretch;border-radius:2px}
.pn-r em{font-style:normal;font-family:var(--f-num);text-align:right;color:#c9cfdb}
.pn-r.me{background:#2a2c3a;box-shadow:inset 0 0 0 1px #e9ecf2}
.pn-chg{display:flex;flex-wrap:wrap;gap:6px 16px;font-size:14px;margin:6px 0 0;color:#c9cfdb}
.pn-chg b{font-family:var(--f-num);color:#e9ecf2} .pn-chg b.up{color:#22d36b} .pn-chg b.dn{color:#ff8a8a}
</style>`);

  window.PitNow = { html, cell };
})();
