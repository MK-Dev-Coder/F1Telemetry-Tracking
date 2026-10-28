/* The race engineer's track briefing, shared by f1.html (a real race: from the races at its track before it, so a
   replay gives no spoilers) and f1-25.html (your game race: real F1 at that track, at this race's length, beside the
   game's own plan). Engineer.render(el, d, o) puts in el a hero card (the track, the engineer's call and today's
   conditions, which can be changed) and one part of the briefing. d = the server's /engineer/{circuit} answer.
   o = { face(code, colour), onChange({temp, rain}), part ("extra", "track", "plans", "winners" or "prediction": only
   that part; none = all), extra (html for the "extra" part, e.g. the game's own plan), reveal (show the results of a
   past race), what (a few words on what it's for) }. Engineer.ui has the pieces the game's plan is drawn with. */
(() => {
  "use strict";
  const esc = s => String(s ?? "").replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const TY = { SOFT: ["S", "Soft", "var(--soft)"], MEDIUM: ["M", "Medium", "var(--medium)"], HARD: ["H", "Hard", "var(--hard)"],
    INTERMEDIATE: ["I", "Intermediate", "var(--inter)"], WET: ["W", "Wet", "var(--wet)"], UNKNOWN: ["?", "Unknown", "#5b6676"] };
  const LETTER = { S: "SOFT", M: "MEDIUM", H: "HARD", I: "INTERMEDIATE", W: "WET" };
  const pct = v => v == null ? "–" : v > 0 && v < 0.005 ? "<1%" : Math.round(v * 100) + "%";
  const n1 = v => v == null ? "–" : String(Math.round(v * 10) / 10);
  const r0 = v => v == null ? "–" : String(Math.round(v));
  const ring = c => { const [l, n, col] = TY[c] || TY.UNKNOWN; return `<span class="eg-chip" style="--c:${col}" title="${n}">${l}</span>`; };
  const planChips = plan => plan.split("-").map(l => ring(LETTER[l] || "UNKNOWN")).join("");
  const plain = plan => plan.split("-").map(l => (TY[LETTER[l]] || TY.UNKNOWN)[1]).join(" → ");
  const planStints = (plan, stops, laps) => { const c = plan.split("-").map(l => LETTER[l] || "UNKNOWN"), ends = [...stops, laps];
    return c.map((comp, k) => ({ compound: comp, from: k ? ends[k - 1] + 1 : 1, to: ends[k] ?? laps })); };
  const bar = (stints, laps) => Stints.bar(stints, laps, { colour: c => TY[c] || TY.UNKNOWN, now: null, going: false });
  const lapsOf = d => d.game_laps || d.laps;
  const factor = d => d.game_laps ? d.laps / d.game_laps : 1;
  // How long a tyre lasts today, in this race's laps (in a game race, scaled to its length): {life, range}
  const lifeOf = (d, c) => { const x = d.tyres?.[c] || d.wet?.[c];
    if (!x) return null;
    if (x.game) return { life: x.game.life, range: x.game.range };
    return "life_today" in x ? { life: x.life_today, range: x.range_today } : { life: x.life, range: x.range }; };

  // A lap meter: a block for each lap of the race, solid up to the shortest usual stint, lighter over the laps a
  // set is usually boxed after, and a line at the usual life
  function meter(c, v, laps) {
    const p = x => `${Math.max(0, Math.min(100, x / laps * 100))}%`, [a, b] = v.range || [];
    return `<span class="eg-mw" style="--c:${(TY[c] || TY.UNKNOWN)[2]}" role="img" aria-label="Usually boxed after ${r0(a)} to ${r0(b)} laps">` +
      `<span class="eg-meter" style="--n:${laps}">` + (a != null ? `<i class="safe" style="width:${p(a)}"></i>` +
      `<i class="win" style="left:${p(a)};width:calc(${p(b ?? a)} - ${p(a)})"></i>` : "") + `</span>` +
      (v.life != null ? `<b class="eg-mk" style="left:${p(v.life)}"></b>` : "") + `</span>`;
  }
  function axis(laps) {
    const step = laps > 40 ? 10 : 5, ticks = [];
    for (let k = 0; k <= laps; k += step) ticks.push(k);
    if (laps - ticks.at(-1) >= step / 2) ticks.push(laps);
    return `<span class="eg-axis" aria-hidden="true">${ticks.map(k => `<span style="left:${k / laps * 100}%">${k}</span>`).join("")}</span>`;
  }

  /* ---- the hero: the track, the engineer's call, today ---- */
  function skeleton(el) {
    el.innerHTML = `<section class="eg-hero" aria-label="The briefing">
      <div class="eg-h-track"></div>
      <div class="eg-h-call" aria-live="polite"></div>
      <div class="eg-h-today" role="group" aria-label="Today's conditions">
        <small>Today</small>
        <label class="eg-t"><output class="eg-tout">–</output><span>track</span>
          <input type="range" class="eg-temp" min="10" max="60" step="1" aria-label="Track temperature"></label>
        <span class="eg-usual"></span>
        <span class="eg-seg" role="group" aria-label="Weather"><button type="button" class="eg-dry" aria-pressed="true">Dry</button><button type="button" class="eg-wet" aria-pressed="false">Rain</button></span>
      </div></section><div class="eg-out" aria-live="polite"></div>`;
    let wait = null;
    const temp = el.querySelector(".eg-temp");
    temp.oninput = () => { el.querySelector(".eg-tout").textContent = `${temp.value} °C`; clearTimeout(wait);
      wait = setTimeout(() => el._eg.onChange({ temp: +temp.value, rain: el._eg.rain }), 150); };
    el.querySelector(".eg-dry").onclick = () => el._eg.onChange({ temp: el._eg.temp, rain: false });
    el.querySelector(".eg-wet").onclick = () => el._eg.onChange({ temp: el._eg.temp, rain: true });
  }

  function hero(el, d, o) {
    const n = d.races?.length || 0, flag = window.RacePicker ? RacePicker.flag(RacePicker.place(d.circuit)[0]) : "";
    el.querySelector(".eg-h-track").innerHTML = `<small>Race engineer</small><b class="eg-h-name">${flag}<span>${esc(d.circuit)}</span></b>` +
      (d.empty ? `<span>No race here before this one in the data.</span>`
        : `<span><b>${lapsOf(d)} laps</b>${d.game_laps ? ` · ${d.distance}% of the real ${d.laps}` : ""}</span>` +
          `<span>${n} race${n === 1 ? "" : "s"} here, ${d.races[0].slice(0, 4)}–${d.races.at(-1).slice(0, 4)}</span>`) +
      (o.what ? `<em>${esc(o.what)}</em>` : "");
    el.querySelector(".eg-h-call").innerHTML = d.empty
      ? `<small>The engineer's call</small><p class="eg-why">Nothing to brief from yet: the data starts in 2022.</p>` : call(d);
    el.querySelector(".eg-usual").textContent = d.empty ? "" : `usually ${d.usual_temp} °C here`;
  }

  // The plan most finishers used here, checked against how long its tyres last today
  function call(d) {
    if (!d.plans.length) return `<small>The engineer's call</small><p class="eg-why">No dry race here to learn a plan from.</p>`;
    const laps = lapsOf(d), top = d.plans[0], stops = top.stop_laps.map(s => Math.max(1, Math.round(s / factor(d))));
    const stints = planStints(top.plan, stops, laps), best = [...d.plans].sort((a, b) => b.wins - a.wins || b.top5 - a.top5)[0];
    // A stint before a stop that is longer than its tyre lasts today: box earlier. (Not the last stint: it runs to the
    // flag, and the stints that ended in a stop say nothing about that.)
    const late = stints.slice(0, -1).map(s => ({ s, t: lifeOf(d, s.compound) }))
      .find(({ s, t }) => t && t.range?.[1] != null && t.life != null && s.to - s.from + 1 > t.range[1] + 0.5);
    const tyre = c => TY[c][1].toLowerCase();
    const note = d.rain ? `<p class="eg-flag wet">Rain: this is the dry plan. On a wet track it's the intermediates or wets (Tyres here).</p>`
      : late ? `<p class="eg-flag">On a ${d.track_temp} °C track the ${tyre(late.s.compound)} is usually boxed after ${r0(late.t.range[0])}–${r0(late.t.range[1])}
          laps, so lap ${late.s.to} is late: box around lap <b>${r0(late.s.from - 1 + late.t.life)}</b>.</p>`
      : `<p class="eg-flag ok">Today's tyres can do it: each stint before a stop fits how long its tyre lasts on a ${d.track_temp} °C track.</p>`;
    return `<small>The engineer's call · from real F1 here</small>` +
      `<div class="eg-call"><span class="eg-chips">${planChips(top.plan)}</span><b>${esc(plain(top.plan))}</b>` +
      `<span>${stops.length ? `box around lap <b>${stops.join(" and ")}</b>` : "no stop"}</span></div>${bar(stints, laps)}` +
      `<p class="eg-why">${pct(top.share)} of the finishers ran it · ${top.wins} win${top.wins === 1 ? "" : "s"} · ${top.top5} top-5` +
      (best.plan !== top.plan && best.wins > top.wins ? ` · most wins: ${esc(plain(best.plan))} (${best.wins})` : "") + `</p>${note}`;
  }

  /* ---- the parts ---- */
  // The track's numbers in one strip, with each race here as a pill: a safety car, a VSC or neither, and rain
  function strip(d) {
    const sc = d.safety_car, w = d.weather, n = d.races.length;
    const pills = (d.history || []).map(x => { const k = x.sc ? "sc" : x.vsc ? "vsc" : "clear";
      return `<i class="eg-pill ${k}${x.wet || x.rain ? " rain" : ""}" title="${x.year}: ${x.sc ? "safety car" : x.vsc ? "virtual safety car" : "no safety car"}` +
        `${x.wet ? ", a wet race" : x.rain ? ", some rain" : ""}, track ${x.track} °C">${String(x.year).slice(2)}</i>`; }).join("");
    const cell = (k, v, s) => `<div class="eg-st"><small>${k}</small><b>${v}</b><span>${s}</span></div>`;
    return `<section class="eg-strip" aria-label="The track">` +
      cell("A pit stop costs", `${n1(d.pit_loss.green)} s`, `${n1(d.pit_loss.sc)} s under the safety car · VSC ${n1(d.pit_loss.vsc)} s`) +
      `<div class="eg-st"><small>Safety car or VSC</small><b>${sc.either} of ${sc.of}</b><span class="eg-pills">${pills}</span>` +
      `<span class="eg-leg"><i class="sc"></i>safety car <i class="vsc"></i>VSC <i class="rain"></i>rain</span></div>` +
      cell("Pole sitter won", `${d.pole_won} of ${n}`, d.places_moved != null ? `drivers move ${d.places_moved} places on average` : "") +
      cell("Usual track", `${w.track} °C`, `${w.coolest}–${w.hottest} °C here · air ${w.air} °C`) +
      cell("Rain", `${w.rain_races} of ${n}`, "races here with rain") + `</section>`;
  }

  function tyres(d) {
    const laps = lapsOf(d);
    const row = (c, v, sub, tag) => { const [, name, col] = TY[c];
      return `<div class="eg-tr${v?.life == null ? " none" : ""}" style="--c:${col}"><span class="eg-tn">${ring(c)}<b>${name}</b></span>` +
        (v?.life == null ? `<span class="eg-big">–</span><span class="eg-tx">${sub}</span>`
          : `<span class="eg-big">${r0(v.life)}<small>laps</small></span>${meter(c, v, laps)}${tag || ""}` +
            `<span class="eg-tx">Usually boxed after <b>${r0(v.range[0])}–${r0(v.range[1])} laps</b>${sub ? ` · ${sub}` : ""}</span>`) + `</div>`; };
    const dry = ["SOFT", "MEDIUM", "HARD"].map(c => { const x = d.tyres[c];
      if (x.life_today == null) return row(c, null, `only ${x.stints} stint${x.stints === 1 ? "" : "s"} here ended in a stop under green: too few`);
      const chg = Math.round(x.change * 100);
      return row(c, lifeOf(d, c), `wears ${x.wear_today.toFixed(3)} s a lap${x.wear_here ? "" : " (all tracks)"} · ${x.stints} stints`,
        `<span class="eg-chg${chg < 0 ? " dn" : chg > 0 ? " up" : ""}">${chg ? `${chg < 0 ? "−" : "+"}${Math.abs(chg)}%` : "usual"}<small>${chg ? "against a usual day" : "a usual day"}</small></span>`); }).join("");
    const wet = d.wet ? ["INTERMEDIATE", "WET"].map(c => row(c, lifeOf(d, c), `${d.wet[c].stints} stints in ${d.wet.races} wet races`,
      `<span class="eg-chg"><b>all</b><small>tracks</small></span>`)).join("") : "";
    return `<section class="eg-box"><header class="eg-hd"><h2>Tyres here</h2><p>How many laps a set lasts ${d.game_laps ? `in your ${laps}-lap race` : "here"}
        on a ${d.track_temp} °C track: one block for each lap of the race.</p></header>` +
      `<div class="eg-trs"><div class="eg-tr ax"><span></span><span></span>${axis(laps)}</div>${dry}` +
      (wet ? `<p class="eg-line"><b>Rain:</b> on a wet track it's the intermediates and wets. How long they lasted in the wet races, at every
        track (each rain is different: take them as a guide).</p>${wet}` : "") + `</div>` +
      `<details class="eg-how"><summary>How it's worked out</summary><p>Tyre life: the stints that ended in a stop under green at this track (the
        middle half of them as the usual range). Wear: how much slower each lap gets, for each lap on the set, with the fuel burn (${d.fuel} s a lap)
        taken out. On a hotter day than usual the tyres wear faster, as learned from every track (medium +${d.temperature_effect.MEDIUM} s, hard
        +${d.temperature_effect.HARD} s, soft +${d.temperature_effect.SOFT} s a lap for each °C), so they last less (at most 40% either way).
        A shorter game race wears the tyres as much faster, so they last that many laps less.</p></details></section>`;
  }

  function plans(d) {
    if (!d.plans.length) return `<section class="eg-box"><header class="eg-hd"><h2>Strategies here</h2><p>No dry race here to learn from.</p></header></section>`;
    const laps = lapsOf(d), most = Math.max(...d.plans.map(p => p.share));
    const rows = d.plans.map((p, k) => { const stops = p.stop_laps.map(s => Math.max(1, Math.round(s / factor(d))));
      return `<div class="eg-pl${k ? "" : " call"}"><span class="eg-chips">${planChips(p.plan)}</span>${bar(planStints(p.plan, stops, laps), laps)}` +
        `<span class="eg-share"><b>${pct(p.share)}</b><i><b style="width:${p.share / most * 100}%"></b></i></span>` +
        `<span class="eg-ns"><span class="eg-n" data-k="Box around">${stops.length ? `lap ${stops.join(", ")}` : "no stop"}</span>` +
        `<span class="eg-n" data-k="Places">${p.gained == null ? "–" : `${p.gained > 0 ? "+" : p.gained < 0 ? "−" : ""}${Math.abs(p.gained)}`}</span>` +
        `<span class="eg-n" data-k="Wins · top 5">${p.wins} · ${p.top5}</span></span></div>`; }).join("");
    return `<section class="eg-box"><header class="eg-hd"><h2>Strategies here</h2><p>What the finishers of the dry races here did${d.game_laps
        ? `, with the stops scaled to your ${laps} laps` : ""}. The most used, at the top, is the engineer's call.</p></header>` +
      `<div class="eg-pls"><div class="eg-pl hd"><span>Plan</span>${Stints.axis(laps)}<span>Finishers</span><span>Box around</span>` +
      `<span>Places gained</span><span>Wins · top 5</span></div>${rows}</div></section>`;
  }

  function winners(d, o) {
    const rows = [...d.winners].reverse().map(w => `<div class="eg-wn"><span class="eg-yr">${w.year}</span>` +
      `<span class="eg-who">${o.face(w.driver, w.colour)}<span>${esc(w.name)}</span></span>` +
      `<span class="eg-grid${w.grid === 1 ? " pole" : ""}">${w.grid === 1 ? "From pole" : `From P${w.grid ?? "–"}`}</span>` +
      bar(w.stints, w.stints.at(-1)?.to || d.laps) +
      `<span class="eg-n">${esc(plain(w.plan))}${w.wet ? ' <em class="eg-tag">wet</em>' : ""}</span></div>`).join("");
    return `<section class="eg-box"><header class="eg-hd"><h2>The winners here</h2><p>How each winner's race went, tyre by tyre (the real race's laps).</p></header>` +
      `<div class="eg-wns">${rows || '<p class="eg-note">No winners in the data.</p>'}</div></section>`;
  }

  function prediction(d, o) {
    const p = d.prediction;
    if (!p) return "";
    const cell = (k, v) => `<span class="eg-pc ${k}"><i><b style="width:${(v || 0) * 100}%"></b></i><em>${pct(v)}</em></span>`;
    const head = `<div class="eg-pr hd"><span></span><span>Driver</span><span>Win</span><span>Podium</span><span class="pts">Points</span><span class="r">Likely</span></div>`;
    const row = (x, i, c, sub) => `<div class="eg-pr"><span class="r">${i + 1}</span><span class="nm">${o.face(x.driver, x.colour)}<span>${esc(x.name)}<small>${esc(sub)}</small></span></span>` +
      cell("win", c.win) + cell("podium", c.podium) + `<span class="pts">${cell("points", c.points)}</span><span class="r">P${c.likely}</span></div>`;
    if (p.kind === "next") {
      const fav = p.drivers[0];
      return `<section class="eg-box"><header class="eg-hd"><h2>Race prediction</h2><p>${esc(p.name)}, ${esc(p.version)}: the race preview model (trained on every race before it), the top 10 by win chance.</p></header>` +
        `<p class="eg-line"><b>Favourite:</b> ${esc(fav.name)} (${esc(fav.team)}), ${pct(fav.win)} to win, ${pct(fav.podium)} for a podium. One race has a lot of luck in it.</p>` +
        `<div class="eg-pred">${head}${p.drivers.slice(0, 10).map((x, i) => row(x, i, x, x.team)).join("")}</div></section>`;
    }
    const b = "before qualifying", fav = p.drivers[0], won = p.drivers.find(x => x.final === 1), show = p.kind === "past" || o.reveal;
    const whose = p.kind === "this" ? "this race" : `the last race here (${esc(p.race_id.slice(0, 4))})`;
    return `<section class="eg-box"><header class="eg-hd"><h2>Race prediction</h2><p>What the race preview said ${b} ${whose}, trained only on the races before it.</p></header>` +
      `<p class="eg-line"><b>Favourite:</b> ${esc(fav.name)} at ${pct(fav[b].win)}.` + (show && won ? ` <b>Winner:</b> ${esc(won.name)}, who had ${pct(won[b].win)}.` : "") +
      (!show ? ` Tick "Show what happened" for the result.` : "") + `</p>` +
      `<div class="eg-pred">${head}${p.drivers.slice(0, 8).map((x, i) => row(x, i, x[b], `${x.team}${show && x.final ? ` · finished P${x.final}` : ""}`)).join("")}</div></section>`;
  }

  function render(el, d, o) {
    if (!el.querySelector(".eg-hero")) skeleton(el);            // the controls stay (a slider being moved keeps its place)
    el._eg = { onChange: o.onChange, temp: d.track_temp, rain: d.rain };
    el.querySelector(".eg-dry").setAttribute("aria-pressed", String(!d.rain)); el.querySelector(".eg-wet").setAttribute("aria-pressed", String(!!d.rain));
    hero(el, d, o);
    const out = el.querySelector(".eg-out"), temp = el.querySelector(".eg-temp"), all = o.part == null, extra = all || o.part === "extra" ? o.extra || "" : "";
    temp.disabled = !!d.empty;
    if (d.empty) { el.querySelector(".eg-tout").textContent = "–"; out.innerHTML = extra; return; }
    const w = d.weather;
    if (document.activeElement !== temp) {
      temp.min = Math.max(5, Math.floor(Math.min(w.coolest, d.track_temp) - 10)); temp.max = Math.min(65, Math.ceil(Math.max(w.hottest, d.track_temp) + 10));
      temp.value = d.track_temp; el.querySelector(".eg-tout").textContent = `${d.track_temp} °C`;
    }
    const none = `<section class="eg-box"><header class="eg-hd"><h2>Race prediction</h2><p>The race preview was only tested before the races of 2025 and 2026.</p></header></section>`;
    const parts = { track: () => strip(d) + tyres(d), plans: () => plans(d), winners: () => winners(d, o), prediction: () => prediction(d, o) || none };
    out.innerHTML = all ? extra + parts.track() + `<div class="eg-two">${parts.plans()}${parts.winners()}</div>` + prediction(d, o)
      : extra + (parts[o.part]?.() || "");
  }

  document.head.insertAdjacentHTML("beforeend", `<style>
.eg-hero{display:grid;grid-template-columns:minmax(170px,.75fr) minmax(0,1.7fr) minmax(190px,.8fr);margin:0 0 12px;border-radius:14px;overflow:hidden;
  background:radial-gradient(120% 140% at 0% 0%,#3d2800 0,#1f1a10 45%,#15151e 100%);border:1px solid #4a3a12}
.eg-hero > div{padding:16px 18px;display:flex;flex-direction:column;gap:6px;min-width:0}
.eg-hero > div + div{border-left:1px solid rgba(255,176,32,.18)}
.eg-hero > div > small{font-size:12px;font-weight:700;letter-spacing:.14em;text-transform:uppercase;color:#e9c98a}
.eg-h-name{display:flex;align-items:center;gap:10px;font-size:26px;font-weight:900;letter-spacing:.02em;text-transform:uppercase;line-height:1.05;color:#fff}
.eg-h-name span{min-width:0;overflow-wrap:anywhere}
.eg-h-track > span{font-size:14px;color:#e2e5ec} .eg-h-track > span b{font-family:var(--f-num);font-weight:600;color:#fff}
.eg-h-track em{font-style:normal;font-size:12.5px;color:#c9b68e;margin-top:auto;padding-top:6px}
.eg-call{display:flex;flex-wrap:wrap;align-items:center;gap:6px 12px}
.eg-call > b{font-size:21px;font-weight:900;color:#fff;letter-spacing:.01em}
.eg-call > span:last-child{font-size:15px;color:#e2e5ec} .eg-call > span b{font-family:var(--f-num);color:#ffb020;font-size:17px}
.eg-call .eg-chip{width:30px;height:30px;font-size:13px;border-width:3px}
.eg-h-call .stbar{height:26px;margin:2px 0}
.eg-why{margin:0;font-size:13.5px;color:#c9cfdb}
.eg-flag{margin:2px 0 0;padding:8px 12px;border-radius:8px;font-size:13.5px;line-height:1.4;background:rgba(255,138,138,.12);border:1px solid rgba(255,138,138,.35);color:#ffd6d6}
.eg-flag b{color:#fff;font-family:var(--f-num)} .eg-flag.ok{background:rgba(59,196,110,.12);border-color:rgba(59,196,110,.35);color:#c9f5d8}
.eg-flag.wet{background:rgba(47,123,255,.14);border-color:rgba(47,123,255,.4);color:#d3e2ff}
.eg-t{display:grid;grid-template-columns:auto 1fr;align-items:baseline;gap:0 8px}
.eg-t output{font-family:var(--f-num);font-size:30px;font-weight:600;color:#fff;line-height:1}
.eg-t span{font-size:13px;color:#e9c98a}
.eg-t input{grid-column:1 / -1;width:100%;margin:8px 0 0;accent-color:#ffb020}
.eg-usual{font-size:12.5px;color:#c9b68e}
.eg-seg{display:inline-flex;align-self:flex-start;background:rgba(0,0,0,.35);border-radius:999px;padding:3px;margin-top:4px}
.eg-seg button{border:0;background:none;padding:5px 16px;border-radius:999px;cursor:pointer;font-weight:700;font-size:13px;color:#ffe2a8}
.eg-seg button[aria-pressed="true"]{background:#ffb020;color:#1a1300}
.eg-seg button:focus-visible,.eg-how summary:focus-visible{outline:2px solid #ffb020;outline-offset:2px}
.eg-out{display:grid;grid-template-columns:minmax(0,1fr);gap:12px} .eg-scroll{overflow-x:auto}
.eg-strip{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));background:#15151e;border-radius:12px;overflow:hidden}
.eg-st{padding:12px 16px;display:flex;flex-direction:column;gap:3px;border-right:1px solid #24252f;min-width:0}
.eg-st small{font-size:12px;letter-spacing:.1em;text-transform:uppercase;color:#9aa0ae}
.eg-st > b{font-family:var(--f-num);font-size:22px;font-weight:600;color:#fff} .eg-st > span{font-size:13px;color:#c9cfdb}
.eg-pills{display:flex;flex-wrap:wrap;gap:4px}
.eg-pill{position:relative;width:28px;height:22px;border-radius:6px;display:grid;place-items:center;font:600 11.5px var(--f-num);font-style:normal;background:#2a2c38;color:#c9cfdb}
.eg-pill.sc{background:#ffd60a;color:#1a1300} .eg-pill.vsc{background:transparent;box-shadow:inset 0 0 0 2px #ffb020;color:#ffd27a}
.eg-pill.rain::after{content:"";position:absolute;top:-3px;right:-3px;width:8px;height:8px;border-radius:50%;background:#3fa4ff;box-shadow:0 0 0 2px #15151e}
.eg-leg{display:flex;flex-wrap:wrap;align-items:center;gap:2px 6px;font-size:11.5px !important;color:#9aa0ae !important}
.eg-leg i{width:9px;height:9px;border-radius:3px;display:inline-block} .eg-leg i.sc{background:#ffd60a} .eg-leg i.vsc{box-shadow:inset 0 0 0 2px #ffb020}
.eg-leg i.rain{border-radius:50%;background:#3fa4ff}
.eg-box{background:#15151e;border-radius:12px;padding:16px 18px}
.eg-hd{margin:0 0 12px} .eg-hd h2{margin:0 0 3px;font-size:14px;letter-spacing:.12em;text-transform:uppercase;color:#d7dae2}
.eg-hd p{margin:0;color:#9aa0ae;font-size:13.5px}
.eg-two{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:12px}
.eg-trs{display:grid;gap:4px}
.eg-tr{display:grid;grid-template-columns:150px 96px minmax(0,1fr) 92px;align-items:center;gap:2px 16px;padding:10px 12px;border-radius:10px;background:#1b1c26}
.eg-tr.ax{background:none;padding:0 12px;height:16px} .eg-tr.ax .eg-axis{grid-column:3}
.eg-tn{display:flex;align-items:center;gap:10px} .eg-tn b{font-size:15px;font-weight:900;letter-spacing:.06em;text-transform:uppercase}
.eg-chip{width:24px;height:24px;flex:none;border-radius:50%;border:2.5px solid var(--c);display:grid;place-items:center;font-weight:900;font-size:11px;color:#fff}
.eg-big{font-family:var(--f-num);font-size:28px;font-weight:600;line-height:1;color:#fff;white-space:nowrap}
.eg-big small{font-family:var(--f-ui);font-size:13px;font-weight:400;color:#c9cfdb;margin-left:5px}
.eg-tr.none .eg-big{color:#6f7484}
.eg-tx{grid-column:3 / -1;font-size:13px;color:#c9cfdb} .eg-tx b{color:#fff}
.eg-tr.none .eg-tx{color:#9aa0ae}
.eg-mw{position:relative;display:block}
.eg-meter{position:relative;display:block;height:16px;background:#2c2e3b;
  -webkit-mask:repeating-linear-gradient(90deg,#000 0 calc(100% / var(--n) * .8),transparent 0 calc(100% / var(--n)));
  mask:repeating-linear-gradient(90deg,#000 0 calc(100% / var(--n) * .8),transparent 0 calc(100% / var(--n)))}
.eg-meter i{position:absolute;top:0;bottom:0;left:0} .eg-meter .safe{background:var(--c)}
.eg-meter .win{background:color-mix(in srgb,var(--c) 42%,transparent)}
.eg-mk{position:absolute;top:-5px;bottom:-5px;width:3px;margin-left:-1.5px;border-radius:2px;background:#fff;box-shadow:0 0 0 1px #15151e}
.eg-axis{position:relative;display:block;height:16px;font:12px var(--f-num);color:#9aa0ae}
.eg-axis span{position:absolute;transform:translateX(-50%)} .eg-axis span:first-child{transform:none}
.eg-chg{display:flex;flex-direction:column;align-items:flex-end;font:600 17px var(--f-num);color:#c9cfdb;text-align:right}
.eg-chg small{font:400 11px var(--f-ui);color:#9aa0ae} .eg-chg.dn{color:#ff8a8a} .eg-chg.up{color:#7cf2a6}
.eg-how{margin-top:10px;font-size:13px;color:#9aa0ae} .eg-how summary{cursor:pointer;color:#c9cfdb;font-weight:700;width:max-content}
.eg-how p{margin:6px 0 0;line-height:1.5}
.eg-pls,.eg-wns{display:grid;gap:4px}
.eg-pl{display:grid;grid-template-columns:96px minmax(160px,1fr) 110px 100px 72px 92px;align-items:center;gap:8px 14px;padding:9px 12px;border-radius:10px;background:#1b1c26;font-size:13.5px}
.eg-pl.call{box-shadow:inset 4px 0 0 #ffb020;background:#211e16}
.eg-pl.hd{background:none;padding-block:0 2px;font-size:11.5px;letter-spacing:.08em;text-transform:uppercase;color:#9aa0ae}
.eg-chips{display:flex;gap:4px;align-items:center}
.eg-share{display:grid;grid-template-columns:44px 1fr;align-items:center;gap:8px} .eg-share > b{font:600 15px var(--f-num);color:#fff}
.eg-share i{height:6px;border-radius:3px;background:#2a2c38;position:relative;overflow:hidden}
.eg-share i b{position:absolute;inset:0 auto 0 0;background:#ffb020;border-radius:3px}
.eg-n{font-family:var(--f-num);color:#e2e5ec} .eg-ns{display:contents}
.eg-wn{display:grid;grid-template-columns:48px minmax(130px,.9fr) 92px minmax(160px,1.6fr) minmax(120px,.8fr);align-items:center;gap:8px 14px;padding:9px 12px;border-radius:10px;background:#1b1c26}
.eg-yr{font:600 13px var(--f-num);color:#c9cfdb;background:#2a2c38;border-radius:6px;padding:3px 0;text-align:center}
.eg-who{display:flex;align-items:center;gap:9px;font-weight:900;text-transform:uppercase;letter-spacing:.03em;font-size:14px;min-width:0}
.eg-who > span{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.eg-grid{font-size:13px;color:#c9cfdb} .eg-grid.pole{color:#e8c547;font-weight:700}
.eg-tag{font-style:normal;font-family:var(--f-ui);font-size:11.5px;font-weight:700;color:#7fb6ff;background:rgba(47,123,255,.16);border-radius:6px;padding:1px 6px;margin-left:4px}
.eg-line{margin:0 0 10px;padding:10px 12px 10px 14px;border-radius:8px;background:#26231a;border-left:4px solid #ffb020;color:#e9ecf2}
.eg-line b{color:#fff}
.eg-pred{display:grid;gap:4px}
.eg-pr{display:grid;grid-template-columns:26px minmax(140px,1.2fr) repeat(3,minmax(90px,1fr)) 56px;gap:10px;align-items:center;padding:6px 8px;border-radius:8px;font-size:14px}
.eg-pr:nth-child(odd){background:#1c1d27}
.eg-pr.hd{background:none;font-size:12px;letter-spacing:.08em;text-transform:uppercase;color:#9aa0ae}
.eg-pr .nm{display:flex;align-items:center;gap:8px;min-width:0;font-weight:900;text-transform:uppercase;letter-spacing:.04em}
.eg-pr .nm span{overflow:hidden;text-overflow:ellipsis;white-space:nowrap} .eg-pr .nm small{display:block;font-weight:400;text-transform:none;letter-spacing:0;color:#9aa0ae;font-size:12px}
.eg-pc{display:flex;align-items:center;gap:8px;font-family:var(--f-num)}
.eg-pc i{flex:1;height:6px;border-radius:3px;background:#2a2c38;position:relative;overflow:hidden}
.eg-pc i b{position:absolute;inset:0 auto 0 0;border-radius:3px}
.eg-pc.win i b{background:#e8c547} .eg-pc.podium i b{background:#c9cfdb} .eg-pc.points i b{background:#3bc46e}
.eg-pc em{font-style:normal;width:42px;text-align:right}
.eg-pr .r{text-align:right;font-family:var(--f-num)}
.eg-note{color:#9aa0ae;font-size:13px;margin:4px 0 0;line-height:1.5}
/* the game's plan (f1-25.html): the race as a lap line with the game's pit window, and the game against real F1 */
.eg-tl{position:relative;height:92px;margin:4px 34px 0}
.eg-tl-track{position:absolute;left:0;right:0;top:40px;height:14px;background:#2c2e3b;
  -webkit-mask:repeating-linear-gradient(90deg,#000 0 calc(100% / var(--n) * .8),transparent 0 calc(100% / var(--n)));
  mask:repeating-linear-gradient(90deg,#000 0 calc(100% / var(--n) * .8),transparent 0 calc(100% / var(--n)))}
.eg-tl-done{position:absolute;inset:0 auto 0 0;background:#5b6676}
.eg-tl-win{position:absolute;top:30px;height:34px;border-radius:7px;background:rgba(255,176,32,.2);box-shadow:inset 0 0 0 2px #ffb020}
.eg-tl-win b{position:absolute;bottom:calc(100% + 6px);left:50%;transform:translateX(-50%);white-space:nowrap;font-size:13px;font-weight:900;letter-spacing:.06em;text-transform:uppercase;color:#ffb020}
.eg-tl-pin{position:absolute;top:66px;transform:translateX(-50%);display:flex;flex-direction:column;align-items:center;white-space:nowrap;font-size:12.5px;color:#c9cfdb}
.eg-tl-pin::before{content:"";width:0;height:0;border:6px solid transparent;border-bottom:8px solid #7fd3ff;border-top:0;margin-bottom:2px}
.eg-tl-pin b{font-family:var(--f-num);color:#fff}
.eg-tl-ax{position:absolute;left:-34px;right:-34px;top:40px;height:14px;display:flex;align-items:center;justify-content:space-between;font:12px var(--f-num);color:#9aa0ae;pointer-events:none}
.eg-hd.row > div{flex:1 1 320px} .eg-rejoin{display:inline-flex;align-items:baseline;gap:8px;padding:6px 12px;border-radius:999px;background:#26231a;border:1px solid #4a3a12;font-size:13px;color:#e9c98a}
.eg-rejoin b{font:600 18px var(--f-num);color:#fff}
.eg-hd.row{display:flex;flex-wrap:wrap;justify-content:space-between;align-items:flex-start;gap:8px 16px}
.eg-h2h{display:grid;gap:6px}
.eg-h2h-r{display:grid;grid-template-columns:minmax(0,1fr) 34px minmax(0,1fr);align-items:center;gap:10px}
.eg-h2h-r.hd{font-size:11.5px;letter-spacing:.08em;text-transform:uppercase;color:#9aa0ae} .eg-h2h-r.hd span:first-child{justify-content:flex-end}
.eg-h2h-r > span{display:flex;align-items:center;gap:8px;min-width:0}
.eg-h2h-r .l{justify-content:flex-end} .eg-h2h-r > span.eg-chip{display:grid;justify-self:center;width:30px;height:30px;font-size:12px}
.eg-h2h-r .trk{flex:1;display:flex;height:14px;min-width:0} .eg-h2h-r .l .trk{justify-content:flex-end}
.eg-h2h-r i{display:block;height:100%;border-radius:4px;min-width:3px;background:var(--c)}
.eg-h2h-r .l i{background:repeating-linear-gradient(135deg,var(--c) 0 5px,color-mix(in srgb,var(--c) 50%,transparent) 5px 10px)}
.eg-h2h-r b{font:600 16px var(--f-num);color:#fff;white-space:nowrap} .eg-h2h-r small{color:#9aa0ae;font-size:12px;white-space:nowrap}
.eg-sets{display:flex;flex-wrap:wrap;gap:8px}
.eg-set{display:grid;grid-template-columns:auto auto;align-items:center;gap:0 8px;padding:7px 12px 7px 8px;border-radius:10px;background:#1b1c26}
.eg-set .eg-chip{grid-row:1 / 3} .eg-set b{font:600 17px var(--f-num);color:#fff} .eg-set small{font-size:11.5px;color:#9aa0ae}
.eg-set.on{box-shadow:inset 0 0 0 2px #7fd3ff;background:#1a2433} .eg-set.on small{color:#7fd3ff;font-weight:700}
@media (max-width:1100px){ .eg-pl{grid-template-columns:96px minmax(0,1fr) 110px} .eg-pl.hd > span:nth-child(n+4){display:none}
  .eg-pl .eg-ns{display:flex;flex-wrap:wrap;gap:2px 20px;grid-column:2 / -1}
  .eg-pl .eg-n::before{content:attr(data-k) " ";font-family:var(--f-ui);font-size:11.5px;color:#9aa0ae} }
@media (max-width:980px){ .eg-hero{grid-template-columns:minmax(0,1fr) minmax(190px,.6fr)}
  .eg-h-call{grid-column:1 / -1;grid-row:2;border-left:0 !important;border-top:1px solid rgba(255,176,32,.18)}
  .eg-two{grid-template-columns:minmax(0,1fr)} .eg-wn{grid-template-columns:48px minmax(0,1fr) 92px} .eg-wn .stbar,.eg-wn .eg-n{grid-column:1 / -1} }
@media (max-width:640px){ .eg-hero{grid-template-columns:minmax(0,1fr)} .eg-hero > div + div{border-left:0;border-top:1px solid rgba(255,176,32,.18)}
  .eg-h-call{grid-row:auto} .eg-tr{grid-template-columns:minmax(0,1fr) auto;padding:10px} .eg-tr .eg-mw,.eg-tr .eg-tx{grid-column:1 / -1} .eg-tr.ax{display:none}
  .eg-tr .eg-chg{flex-direction:row;gap:6px;align-items:baseline}
  .eg-pl{grid-template-columns:auto minmax(0,1fr)} .eg-pl .stbar,.eg-pl .eg-ns{grid-column:1 / -1} .eg-pl.hd{display:none}
  .eg-pr{grid-template-columns:22px minmax(0,1.4fr) repeat(2,minmax(70px,1fr)) 44px} .eg-pr .pts{display:none} }
</style>`);

  window.Engineer = { render, chip: ring, ui: { TY, esc, ring, plain, planStints, bar, lifeOf, r0, n1, pct } };
})();
