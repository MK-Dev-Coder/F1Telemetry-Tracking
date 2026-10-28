/* Car against car, shared by f1.html (real races) and f1-25.html (your game):
   - the battle forecast: how many laps until the car behind is within 1 second of the car ahead (their pace)
   - the undercut threat: if the car behind pits now and the car ahead answers next lap, does it come out ahead?
   - the strategy battle: two drivers side by side, with the gap between them lap by lap.
   Battle.forecast(b) and Battle.undercut(u) are table cells, Battle.html(...) the panel's lines, Battle.duel(...) the
   side by side (with the rows of Battle.facts(...)). b = a driver's "battle" from the server (against the car just ahead). */
(() => {
  "use strict";
  const esc = s => String(s ?? "").replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const pc = v => v == null ? "–" : v > 0 && v < 0.005 ? "<1%" : v > 0.995 && v < 1 ? ">99%" : Math.round(v * 100) + "%";
  const laps = n => `${n} lap${n === 1 ? "" : "s"}`;
  const rate = v => `${v >= 0 ? "+" : "−"}${Math.abs(v).toFixed(2)} s a lap`;

  // In the tables: when the car behind gets within 1 s
  function forecast(b) {
    if (!b) return "<span></span>";
    const [main, small] = b.gap <= 1 ? ["Within 1 s", "now"]
      : b.laps != null ? [`in ${laps(b.laps)}`, rate(b.closing)]
      : b.closing == null ? ["–", b.racing === false ? "not under green" : "no pace yet"]
      : [b.closing < -0.05 ? "dropping back" : "not closing", rate(b.closing)];
    return `<span class="bf${b.gap <= 1 ? " near" : b.laps != null && b.laps <= 5 ? " soon" : ""}" title="${esc(main)} (${esc(small)})">${esc(main)}<small>${esc(small)}</small></span>`;
  }

  // The undercut chance as a meter
  function undercut(u) {
    if (!u) return `<span class="dim">–</span>`;
    return `<span class="uc ${u.chance >= 0.5 ? "hot" : u.chance >= 0.2 ? "warm" : ""}" title="Comes out ahead in ${pc(u.chance)} of the real undercuts like this">` +
      `<span class="uc-m"><i style="width:${u.chance * 100}%"></i></span><b>${pc(u.chance)}</b></span>`;
  }

  // The panel: the car ahead and the car behind, in words. me = the driver's own battle (against the car ahead),
  // them = the car behind's battle (against this driver). you: the player (F1 25).
  function html(me, them, name, nameOf, you = false) {
    const who = you ? "You" : esc(name), are = you ? "are" : "is", pits = you ? "you pit" : `${esc(name)} pits`;
    const lines = [];
    if (me) {
      const ahead = esc(nameOf(me.ahead)), u = me.undercut;
      let t = `<b>${who} ${are} ${me.gap.toFixed(1)} s behind ${ahead}.</b> `;
      t += me.gap <= 1 ? "Within striking distance (1 s) now."
        : me.laps != null ? `${who} ${you ? "are" : "is"} gaining ${me.closing.toFixed(2)} s a lap: within 1 s in about ${laps(me.laps)}.`
        : me.closing == null ? (me.racing === false ? "No forecast under the safety car, VSC or red flag."
          : "No forecast yet: it needs a few green laps of pace.")
        : me.closing < -0.05 ? `${ahead} is pulling away, ${Math.abs(me.closing).toFixed(2)} s a lap.` : "Not closing in before the end.";
      if (u && u.chance >= 0.05) t += ` <b>Undercut:</b> if ${pits} now and ${ahead} pits next lap, ${you ? "you'd" : `${esc(name)} would`}
        come out ahead in ${pc(u.chance)} of the real undercuts like this (one usually gains ${u.gain.toFixed(1)} s).`;
      else if (u) t += ` An undercut on ${ahead} is unlikely to work: the gap is bigger than undercuts usually gain (${u.gain.toFixed(1)} s).`;
      lines.push(`<p class="why-line why-window">${t}</p>`);
    }
    if (them) {
      const behind = esc(nameOf(them.driver)), u = them.undercut;
      let t = `<b>${behind} is ${them.gap.toFixed(1)} s behind ${you ? "you" : esc(name)}.</b> `;
      t += them.gap <= 1 ? "Within striking distance (1 s) now."
        : them.laps != null ? `Gaining ${them.closing.toFixed(2)} s a lap: within 1 s in about ${laps(them.laps)}.`
        : them.closing == null ? "" : them.closing < -0.05 ? `${who} ${are} pulling away.` : "Not closing in before the end.";
      if (u && u.chance >= 0.05) t += ` <b>Undercut threat:</b> if ${behind} pits now and ${pits} next lap, ${behind} comes out ahead in
        ${pc(u.chance)} of the real undercuts like this.`;
      lines.push(`<p class="why-line ${u && u.chance >= 0.5 ? "why-box" : "why-stay"}">${t}</p>`);
    }
    if (!lines.length) return "";
    return lines.join("") + `<p class="note">The forecast uses each car's pace over its last 5 green laps. The undercut is learned from
      the real undercuts of 2022-2025: one gains more when the car ahead's tyres are worn.</p>`;
  }

  // The strategy battle's side by side, from the page's own lap data for the two drivers (A, B, as the server sends
  // them for a lap). o = { tyre(compound) html, tyreName(compound), pct(v), soon (the pit stop row's label),
  // outcome (true: with the likely finish) }
  function facts(A, B, d, o) {
    const best = x => { const t = Object.entries(x.next_tyre || {}).sort((p, q) => q[1] - p[1])[0]; return t ? `${o.tyre(t[0])} ${o.pct(t[1])}` : "–"; };
    const age = x => x.compound ? `${o.tyre(x.compound)} ${esc(o.tyreName(x.compound))}${x.tyre_age != null ? `, ${x.tyre_age} laps old` : ""}` : "–";
    const pit = x => x.pitting_now ? "in the pits" : o.pct(x.pit_chance);
    const prize = { win: "win", podium: "podium", points: "points" };
    const finish = x => x.outcome ? `P${x.outcome.likely} <span class="dim">(${prize[x.outcome.for]} ${o.pct(x.outcome[x.outcome.for])})</span>` : "–";
    const rows = [["Place", `P${A.position ?? "–"}`, `P${B.position ?? "–"}`], ["Tyre", age(A), age(B)],
      ["Stops", String(d.stops[d.a]?.length ?? 0), String(d.stops[d.b]?.length ?? 0)], [o.soon, pit(A), pit(B)],
      ["Next tyre", best(A), best(B)], ["If they pit now", A.rejoin ? `P${A.rejoin.position}` : "–", B.rejoin ? `P${B.rejoin.position}` : "–"]];
    if (o.outcome) rows.push(["Likely finish", finish(A), finish(B)]);
    return rows;
  }

  // The strategy battle. d = the server's answer; o = { name(code), colour(code), tyre(compound, age) html,
  // total (laps in the race), facts: [[label, html for a, html for b], ...] from the page's own lap data, you (code),
  // head: false when the page shows the two drivers itself }
  function duel(d, o) {
    const A = d.a, B = d.b, n = d.now, nm = c => (c === o.you ? "You" : esc(o.name(c)));
    let top = o.head === false ? "" : `<div class="du-head">${[A, B].map(c => `<span><i style="background:${esc(o.colour(c))}"></i>${esc(o.name(c))}${c === o.you ? " (you)" : ""}</span>`).join("<em>against</em>")}</div>`;
    if (n) {
      const front = nm(n.front), back = nm(n.back), isAre = n.back === o.you ? "are" : "is";
      let t = `<b>${back} ${isAre} ${n.gap.toFixed(1)} s behind ${n.front === o.you ? "you" : front}</b>` +
        (n.between ? ` (${n.between} car${n.between === 1 ? "" : "s"} between ${o.you && [n.front, n.back].includes(o.you) ? "you" : "them"}).` : ".");
      t += n.gap <= 1 ? " Within 1 s now." : n.laps != null ? ` Gaining ${n.closing.toFixed(2)} s a lap: within 1 s in about ${laps(n.laps)}.`
        : n.closing == null ? "" : n.closing < -0.05 ? ` The gap grows ${Math.abs(n.closing).toFixed(2)} s a lap.` : " Not closing in before the end.";
      if (n.undercut && n.undercut.chance >= 0.05) t += ` <b>Undercut:</b> if ${back === "You" ? "you pit" : back + " pits"} now and ${n.front === o.you ? "you pit" : front + " pits"} next lap,
        ${back === "You" ? "you'd" : back + " would"} come out ahead in ${pc(n.undercut.chance)} of the real undercuts like this.`;
      top += `<p class="why-line why-window">${t}</p>`;
    }
    if (o.facts?.length) top += `<div class="du-facts"><span></span><b>${esc(o.name(A))}</b><b>${esc(o.name(B))}</b>` +
      o.facts.map(([label, a, b]) => `<span>${esc(label)}</span><span>${a}</span><span>${b}</span>`).join("") + `</div>`;
    top += `<h4>Gap between them, lap by lap</h4>` + gapChart(d, o);
    top += `<h4>Tyres</h4><div class="du-st">` + [A, B].map(c => `<span>${esc(o.name(c))}</span>` +
      Stints.bar(d.stints[c] || [], o.total, { colour: o.colours, now: d.lap, going: true })).join("") + `</div>`;
    return top;
  }

  // The gap of B behind A on every lap (above the line: A ahead), with each pit stop marked in the driver's colour
  function gapChart(d, o) {
    const pts = d.laps.filter(l => l.gap != null);
    if (pts.length < 2) return `<p class="note">The gap shows after two laps.</p>`;
    const W = 560, H = 190, L = 40, R = 10, T = 12, Bm = 22, total = Math.max(o.total, d.lap);
    const lim = Math.min(40, Math.max(5, ...pts.map(p => Math.abs(p.gap)))), clip = v => Math.max(-lim, Math.min(lim, v));
    const x = lap => L + (lap - 1) / Math.max(total - 1, 1) * (W - L - R), y = v => T + (lim - clip(v)) / (2 * lim) * (H - T - Bm);
    const path = pts.map((p, k) => `${k ? "L" : "M"}${x(p.lap).toFixed(1)},${y(p.gap).toFixed(1)}`).join("");
    const ticks = [-lim, -lim / 2, 0, lim / 2, lim].map(v => `<text x="${L - 6}" y="${y(v) + 4}" text-anchor="end">${v === 0 ? "0" : (v > 0 ? "+" : "−") + Math.abs(Math.round(v))}</text>` +
      `<line x1="${L}" x2="${W - R}" y1="${y(v)}" y2="${y(v)}" class="${v === 0 ? "zero" : ""}"/>`).join("");
    const band = `<rect x="${L}" width="${W - L - R}" y="${y(1)}" height="${y(-1) - y(1)}" class="near"/>`;
    const stops = [d.a, d.b].flatMap(c => (d.stops[c] || []).map(lap => `<line x1="${x(lap)}" x2="${x(lap)}" y1="${T}" y2="${H - Bm}" ` +
      `style="stroke:${esc(o.colour(c))}" class="stop"><title>${esc(o.name(c))} pits on lap ${lap}</title></line>`)).join("");
    const lapsAxis = [1, Math.round(total / 2), total].map(l => `<text x="${x(l)}" y="${H - 6}" text-anchor="middle">${l === 1 ? "L1" : l}</text>`).join("");
    const last = pts.at(-1);
    return `<svg class="du-chart" viewBox="0 0 ${W} ${H}" role="img" aria-label="The gap between ${esc(o.name(d.a))} and ${esc(o.name(d.b))} on every lap: ` +
      `${Math.abs(last.gap).toFixed(1)} s on lap ${last.lap}">${band}${ticks}${stops}<path d="${path}"/>` +
      `<circle cx="${x(last.lap)}" cy="${y(last.gap)}" r="3.5"/>${lapsAxis}` +
      `<text x="${W - R}" y="${T + 10}" text-anchor="end" class="lab">${esc(o.name(d.a))} ahead</text>` +
      `<text x="${W - R}" y="${H - Bm - 4}" text-anchor="end" class="lab">${esc(o.name(d.b))} ahead</text></svg>` +
      `<p class="note">Seconds between them at the end of each lap; the band is within 1 s. A line marks each pit stop.</p>`;
  }

  document.head.insertAdjacentHTML("beforeend", `<style>
.bf{text-align:right;font-family:var(--f-num);line-height:1.05;white-space:nowrap}
.bf small{display:block;color:#9aa0ae;font-size:11px;font-family:var(--f-ui)}
.bf.soon{color:#ffd166} .bf.near{color:#ff8a8a}
.uc{display:grid;grid-template-columns:minmax(0,1fr) 40px;gap:8px;align-items:center}
.uc-m{position:relative;height:8px;border-radius:4px;background:#2a2c3a;overflow:hidden}
.uc-m i{position:absolute;inset:0 auto 0 0;background:#8a96a8;border-radius:4px}
.uc.warm .uc-m i{background:#ffd166} .uc.hot .uc-m i{background:#ff6b6b}
.uc b{font-family:var(--f-num);font-weight:600;text-align:right}
.du-head{display:flex;flex-wrap:wrap;align-items:center;gap:8px 12px;font-size:16px;font-weight:700;margin:2px 0 8px}
.du-head span{display:inline-flex;align-items:center;gap:8px} .du-head i{width:4px;height:18px;border-radius:2px}
.du-head em{font-style:normal;font-weight:400;font-size:14px;color:#9aa0ae}
.du-facts{display:grid;grid-template-columns:minmax(110px,auto) 1fr 1fr;gap:6px 12px;align-items:center;font-size:14px;margin:10px 0;color:#e9ecf2}
.du-facts > b{font-size:13px;color:#c9cfdb;text-transform:uppercase;letter-spacing:.04em} .du-facts > span:nth-child(3n+1){color:#9aa0ae}
.du-chart{width:100%;height:auto;display:block;background:#15151e;border-radius:4px}
.du-chart text{fill:#9aa0ae;font:12px var(--f-num)} .du-chart text.lab{font-family:var(--f-ui)}
.du-chart line{stroke:#2a2c3a} .du-chart line.zero{stroke:#5a5e70} .du-chart line.stop{stroke-width:2;opacity:.9}
.du-chart rect.near{fill:rgba(255,209,102,.10)}
.du-chart path{fill:none;stroke:#e9ecf2;stroke-width:2} .du-chart circle{fill:#e9ecf2}
.du-st{display:grid;grid-template-columns:auto minmax(0,1fr);gap:8px 10px;align-items:center;font-size:14px}
</style>`);

  window.Battle = { forecast, undercut, html, duel, facts };
})();
