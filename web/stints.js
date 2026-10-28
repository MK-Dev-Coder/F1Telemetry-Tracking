/* The tyre strategy graphic of both pages (like F1 TV's): every stint as a bar along the race, in its tyre's
   colour, with a line at the lap shown. Stints.axis(total) is the lap scale for the header, Stints.bar(...) one row. */
(() => {
  "use strict";
  const css = `
.stax{position:relative;display:block;height:16px;font-family:var(--f-num);font-size:13px;letter-spacing:0;text-transform:none;color:#9aa0ae}
.stax span{position:absolute;top:0;transform:translateX(-50%);white-space:nowrap}
.stbar{position:relative;display:block;height:22px;border-radius:3px;
  background:repeating-linear-gradient(90deg,rgba(255,255,255,.07) 0 1px,transparent 1px var(--step)),#1a1b24}
.stbar i{position:absolute;top:2px;bottom:2px;border-radius:3px;background:var(--c);color:#111;font:normal 900 12px/18px var(--f-ui);
  padding-left:5px;overflow:hidden;white-space:nowrap;box-shadow:inset -2px 0 0 rgba(0,0,0,.35)}
.stbar i small{font:600 11px var(--f-num);margin-left:4px;opacity:.8}
.stbar i.now{box-shadow:inset -3px 0 0 rgba(255,255,255,.85)}          /* still on this set */
.stbar b{position:absolute;top:-6px;bottom:-6px;width:1px;background:#c9cfdb;pointer-events:none}`;
  document.head.insertAdjacentHTML("beforeend", `<style>${css}</style>`);

  const pc = (laps, total) => Math.max(0, Math.min(100, laps / Math.max(total, 1) * 100));
  const step = total => (total > 40 ? 10 : 5);

  // The lap scale: L1, every 5 or 10 laps, and the last lap (a tick too close to the last is left out)
  function axis(total) {
    const ticks = [1];
    for (let l = step(total); l < total; l += step(total)) if (total - l >= step(total) / 2) ticks.push(l);
    if (total > 1) ticks.push(total);
    return `<span class="stax" aria-hidden="true">${ticks.map(l => `<span style="left:${pc(l - 0.5, total)}%">${l === 1 ? "L1" : l}</span>`).join("")}</span>`;
  }

  // One car's stints [{compound, from, to}]. colour(compound) gives [letter, name, css colour]; now is the lap shown
  // (a line); going means the last stint is still running
  function bar(stints, total, { colour, now, going }) {
    const said = [];
    const segs = stints.map((h, k) => {
      const [letter, name, c] = colour(h.compound), laps = h.to - h.from + 1;
      said.push(`${name} laps ${h.from} to ${h.to}`);
      return `<i class="${going && k === stints.length - 1 ? "now" : ""}" style="left:${pc(h.from - 1, total)}%;width:${pc(laps, total)}%;--c:${c}" ` +
        `title="${name}: laps ${h.from}–${h.to} (${laps} lap${laps === 1 ? "" : "s"})">${letter}${laps >= 3 ? `<small>${laps}</small>` : ""}</i>`;
    }).join("");
    const line = now ? `<b style="left:${pc(now - 0.5, total)}%"></b>` : "";
    return `<span class="stbar" role="img" aria-label="${said.length ? "Stints: " + said.join(", ") : "No stints yet"}" style="--step:${pc(step(total), total)}%">${segs}${line}</span>`;
  }

  window.Stints = { axis, bar };
})();
