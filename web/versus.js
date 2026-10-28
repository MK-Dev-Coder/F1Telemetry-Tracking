/* The strategy battle's driver picker: a pop-up window with the two drivers as big cards in their team colours, VS
   between them, and every driver of the race below to pick from. Versus.open({...}) shows it, Versus.bar(a, b) is
   the page's own button (both photos and VS), Versus.face(d, "sm") a small photo for the tables. A driver = {code, first, last, team, colour, number, photo, position, out}.
   The photos are F1's own, from the address OpenF1 gives; without one (or offline) the car number shows instead. */
(() => {
  "use strict";
  const esc = s => String(s ?? "").replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const ROLE = { a: "Driver", b: "Against" };

  // A round photo in the team colour (the number underneath, for when there's no photo)
  const face = (d, cls = "") => `<span class="vs-ph ${cls}" style="--c:${esc(d.colour)}" aria-hidden="true"><span class="num">${esc(d.number ?? d.code)}</span>` +
    (d.photo ? `<img src="${esc(d.photo)}" alt="" decoding="async" onerror="this.remove()">` : "") + `</span>`;
  const where = d => [d.team, d.position ? `P${d.position}` : d.out ? "out" : null].filter(Boolean).join(" · ");

  // The page's button: both drivers, VS, and "Change"
  function bar(a, b) {
    const side = (d, r) => `<span class="vs-side${r ? " r" : ""}" style="--c:${esc(d.colour)}">${face(d)}` +
      `<span class="t"><b>${esc(d.last || d.code)}</b><small>${esc(where(d))}</small></span></span>`;
    return `${side(a)}<span class="vs-mini" aria-hidden="true">VS</span>${side(b, true)}<span class="vs-change">Change drivers</span>`;
  }

  // One of the two big cards at the top
  function slot(d, role, active) {
    return `<button type="button" class="vs-slot${active ? " on" : ""}" data-slot="${role}" aria-pressed="${active}" style="--c:${esc(d.colour)}"
      aria-label="${ROLE[role]}: ${esc(`${d.first || ""} ${d.last || d.code}`.trim())}. ${active ? "Picking now" : "Press to pick this one"}">
      <span class="vs-role">${ROLE[role]}</span><span class="vs-big" aria-hidden="true">${esc(d.number ?? "")}</span>
      ${d.photo ? `<img src="${esc(d.photo)}" alt="" decoding="async" onerror="this.remove()">` : `<span class="vs-code" aria-hidden="true">${esc(d.code)}</span>`}
      <span class="vs-name"><small>${esc(d.first || "")}</small><b>${esc(d.last || d.code)}</b><i>${esc(where(d))}</i></span></button>`;
  }

  let openNow = null;

  /* o = {drivers: [driver, ...] (in race order), a, b (codes), title, opener (gets the focus back), onPick(a, b)} */
  function open(o) {
    close();
    const by = Object.fromEntries(o.drivers.map(d => [d.code, d]));
    const st = { a: o.a, b: o.b, slot: "a" };
    const back = document.createElement("div");
    back.className = "vs-back";
    back.innerHTML = `<div class="vs-sheet" role="dialog" aria-modal="true" aria-labelledby="vsTitle">
      <header class="vs-head"><div><small>${esc(o.title || "Strategy battle")}</small><h2 id="vsTitle">Pick two drivers</h2></div>
        <button class="vs-close" type="button" aria-label="Close"><svg viewBox="0 0 24 24" width="20" height="20" aria-hidden="true">
          <path d="M6 6l12 12M18 6 6 18" stroke="currentColor" stroke-width="2.4" stroke-linecap="round"/></svg></button></header>
      <div class="vs-stage"></div>
      <p class="vs-hint" aria-live="polite"></p>
      <div class="vs-grid" role="group" aria-label="Drivers"></div>
      <footer class="vs-foot"><button type="button" class="vs-cancel">Cancel</button><button type="button" class="vs-go">Compare</button></footer>
    </div>`;
    const sheet = back.firstElementChild, stage = sheet.querySelector(".vs-stage"), grid = sheet.querySelector(".vs-grid");

    const show = changed => {
      stage.innerHTML = slot(by[st.a], "a", st.slot === "a") +
        `<span class="vs-mid"><b class="vs-badge">VS</b><button type="button" class="vs-swap" aria-label="Swap the two drivers" title="Swap">` +
        `<svg viewBox="0 0 24 24" width="18" height="18" aria-hidden="true"><path d="M7 7h12l-3.5-3.5M17 17H5l3.5 3.5" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/></svg></button></span>` +
        slot(by[st.b], "b", st.slot === "b");
      if (changed) stage.querySelector(`[data-slot="${changed}"]`)?.classList.add("pop");
      sheet.querySelector(".vs-hint").innerHTML = `Pick the <b>${st.slot === "a" ? "first driver" : "driver to compare against"}</b>, or press a card to change it.`;
      grid.innerHTML = o.drivers.map(d => { const role = d.code === st.a ? "a" : d.code === st.b ? "b" : "";
        return `<button type="button" class="vs-d${d.out ? " out" : ""}" data-code="${esc(d.code)}" style="--c:${esc(d.colour)}"${role ? ` data-pick="${role}"` : ""}
          aria-pressed="${!!role}" aria-label="${esc(`${d.first || ""} ${d.last || d.code}`.trim())}, ${esc(where(d))}${role ? `, ${ROLE[role]}` : ""}">` +
          `${d.position ? `<span class="pos">P${d.position}</span>` : ""}${role ? `<span class="tag">${ROLE[role]}</span>` : ""}` +
          `${face(d)}<b>${esc(d.last || d.code)}</b><small>${esc(d.team || "")}</small></button>`; }).join("");
    };

    const pick = code => {
      const other = st.slot === "a" ? "b" : "a";
      if (code === st[other]) [st.a, st.b] = [st.b, st.a];            // the other one: they swap
      else st[st.slot] = code;
      const changed = st.slot;
      st.slot = other;                                                  // then the other card
      show(changed);
      grid.querySelector(`[data-code="${CSS.escape(code)}"]`)?.focus({ preventScroll: true });
    };

    back.addEventListener("click", e => {
      if (e.target === back || e.target.closest(".vs-close, .vs-cancel")) return close();
      if (e.target.closest(".vs-go")) { const { a, b } = st; close(); return o.onPick(a, b); }
      if (e.target.closest(".vs-swap")) { [st.a, st.b] = [st.b, st.a]; return show(); }
      const s = e.target.closest(".vs-slot");
      if (s) { st.slot = s.dataset.slot; show(); return stage.querySelector(`[data-slot="${st.slot}"]`).focus(); }
      const d = e.target.closest(".vs-d");
      if (d) pick(d.dataset.code);
    });
    back.addEventListener("keydown", e => {
      if (e.key === "Escape") { e.preventDefault(); return close(); }
      if (e.key === "Tab") {                                  // the focus stays in the window while it's open
        const all = [...sheet.querySelectorAll("button")], first = all[0], last = all.at(-1);
        if (e.shiftKey && document.activeElement === first) { e.preventDefault(); last.focus(); }
        else if (!e.shiftKey && document.activeElement === last) { e.preventDefault(); first.focus(); }
      }
    });

    document.body.append(back);
    document.documentElement.style.overflow = "hidden";       // the page behind doesn't scroll
    openNow = { back, opener: o.opener };
    show();
    (grid.querySelector('[data-pick="a"]') || sheet.querySelector(".vs-close")).focus({ preventScroll: true });
  }

  function close() {
    if (!openNow) return;
    const { back, opener } = openNow;
    openNow = null;
    back.remove();
    document.documentElement.style.overflow = "";
    opener?.focus();
  }

  document.head.insertAdjacentHTML("beforeend", `<style>
/* the page's button */
.vs-bar{display:grid;grid-template-columns:minmax(0,1fr) auto minmax(0,1fr) auto;align-items:center;gap:12px;width:100%;margin:0 0 12px;
  padding:9px 12px;border-radius:18px;border:1px solid #2a2c3a;background:#1a1b25;color:#e9ecf2;cursor:pointer;font:inherit;text-align:left;
  transition:background .15s,border-color .15s}
.vs-bar:hover{background:#1f2130;border-color:#4a4c5a} .vs-bar:hover .vs-change{background:#fff;color:#000}
.vs-bar:focus-visible{outline:2px solid #fff;outline-offset:2px}
.vs-side{display:flex;align-items:center;gap:10px;min-width:0} .vs-side.r{flex-direction:row-reverse;text-align:right}
.vs-side .vs-ph{width:46px;height:46px;flex:none}
.vs-side .t{min-width:0;border-left:3px solid var(--c);padding-left:9px} .vs-side.r .t{border-left:0;border-right:3px solid var(--c);padding:0 9px 0 0}
.vs-side b{display:block;font-weight:900;font-size:16px;text-transform:uppercase;letter-spacing:.04em;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.vs-side small{display:block;color:#9aa0ae;font-size:13px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.vs-mini{width:36px;height:36px;border-radius:50%;background:#fff;color:#000;font-weight:900;font-size:14px;display:grid;place-items:center;
  box-shadow:0 0 0 3px rgba(225,6,50,.55)}
.vs-change{font-size:13px;font-weight:600;color:#e9ecf2;border:1px solid #3a3c4a;border-radius:999px;padding:6px 12px;white-space:nowrap;transition:background .15s,color .15s}
/* a round photo in the team colour */
.vs-ph{position:relative;display:grid;place-items:center;border-radius:50%;overflow:hidden;
  background:radial-gradient(circle at 50% 28%,color-mix(in srgb,var(--c) 75%,#fff),var(--c) 62%,color-mix(in srgb,var(--c) 60%,#000))}
.vs-ph img{position:absolute;inset:0;width:100%;height:100%;object-fit:cover;object-position:50% 0}
.vs-ph.sm{width:26px;height:26px;flex:none} .vs-ph.sm .num{font-size:11px}
.vs-ph .num{font:900 18px/1 var(--f-ui,"Titillium Web",system-ui,sans-serif);color:rgba(255,255,255,.92)}
/* the window */
.vs-back{position:fixed;inset:0;z-index:100;background:rgba(5,7,10,.7);backdrop-filter:blur(6px);display:grid;place-items:center;padding:16px;
  animation:vs-fade .16s ease-out}
.vs-sheet{width:min(760px,100%);max-height:min(92vh,900px);display:flex;flex-direction:column;background:#14151d;color:#e9ecf2;border:1px solid #2a2c3a;
  border-radius:24px;overflow:hidden;box-shadow:0 30px 80px rgba(0,0,0,.65);animation:vs-up .24s cubic-bezier(.3,1.3,.5,1);
  font-family:var(--f-ui,"Titillium Web","Segoe UI",system-ui,sans-serif)}
.vs-head{flex:none;display:flex;align-items:flex-start;justify-content:space-between;padding:20px 20px 12px 24px;
  background:radial-gradient(120% 140% at 50% 0%,rgba(225,6,50,.38),rgba(225,6,50,.08) 55%,transparent 80%)}
.vs-head small{font-size:13px;letter-spacing:.14em;text-transform:uppercase;color:#ff9aab;font-weight:700}
.vs-head h2{margin:2px 0 0;font-size:28px;font-weight:900;text-transform:uppercase;line-height:1.05}
.vs-close{flex:none;width:42px;height:42px;border-radius:50%;border:1px solid rgba(255,255,255,.2);background:rgba(255,255,255,.08);color:#fff;
  cursor:pointer;display:grid;place-items:center}
.vs-close:hover{background:rgba(255,255,255,.16)}
.vs-stage{flex:none;display:grid;grid-template-columns:minmax(0,1fr) auto minmax(0,1fr);align-items:center;gap:14px;padding:4px 24px 0}
.vs-slot{position:relative;height:200px;border-radius:20px;overflow:hidden;border:2px solid transparent;padding:0;cursor:pointer;color:#fff;
  text-align:left;font:inherit;background:linear-gradient(155deg,color-mix(in srgb,var(--c) 92%,#fff 8%) 0%,color-mix(in srgb,var(--c) 55%,#0b0b10) 62%,#0b0b10 100%);
  box-shadow:0 10px 30px rgba(0,0,0,.35);transition:border-color .15s,transform .2s,box-shadow .2s,opacity .2s}
.vs-slot:not(.on){opacity:.78} .vs-slot:hover{opacity:1;transform:translateY(-2px)}
.vs-slot.on{border-color:#fff;box-shadow:0 0 0 4px rgba(255,255,255,.14),0 14px 34px rgba(0,0,0,.45)}
.vs-slot:focus-visible{outline:3px solid #fff;outline-offset:3px}
.vs-slot.pop{animation:vs-pop .35s cubic-bezier(.3,1.5,.5,1)}
.vs-role{position:absolute;top:12px;left:14px;z-index:2;font-size:12px;font-weight:700;letter-spacing:.12em;text-transform:uppercase;
  background:rgba(0,0,0,.35);border-radius:6px;padding:2px 8px}
.vs-slot.on .vs-role{background:#fff;color:#000}
.vs-big{position:absolute;top:-8px;right:12px;font:900 120px/1 var(--f-ui,"Titillium Web",system-ui,sans-serif);color:rgba(255,255,255,.16)}
.vs-slot img{position:absolute;right:0;bottom:0;height:92%;max-width:70%;object-fit:contain;object-position:right bottom;
  filter:drop-shadow(0 8px 16px rgba(0,0,0,.45))}
.vs-code{position:absolute;right:16px;bottom:44px;font:900 64px/1 var(--f-ui,"Titillium Web",system-ui,sans-serif);color:rgba(255,255,255,.5)}
.vs-name{position:absolute;left:14px;right:38%;bottom:12px;z-index:2;display:grid;text-shadow:0 2px 8px rgba(0,0,0,.5)}
.vs-name small{font-size:14px;font-weight:600;opacity:.9}
.vs-name b{font-size:24px;font-weight:900;text-transform:uppercase;letter-spacing:.02em;line-height:1.05;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.vs-name i{font-style:normal;font-size:13px;opacity:.85;margin-top:4px}
.vs-mid{display:grid;justify-items:center;gap:10px}
.vs-badge{width:58px;height:58px;border-radius:50%;background:#fff;color:#000;font-size:20px;font-weight:900;display:grid;place-items:center;
  box-shadow:0 0 0 5px rgba(225,6,50,.6),0 10px 30px rgba(0,0,0,.5);animation:vs-beat 2.4s ease-in-out infinite}
.vs-swap{width:38px;height:38px;border-radius:50%;border:1px solid #3a3c4a;background:#1f2130;color:#e9ecf2;cursor:pointer;display:grid;place-items:center}
.vs-swap:hover{background:#fff;color:#000}
.vs-hint{flex:none;margin:12px 24px 2px;color:#9aa0ae;font-size:14px} .vs-hint b{color:#fff}
.vs-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(108px,1fr));gap:8px;padding:10px 24px 14px;overflow-y:auto;overscroll-behavior:contain}
.vs-d{position:relative;display:grid;justify-items:center;gap:3px;padding:12px 6px 10px;border-radius:14px;border:1px solid #262838;background:#1d1f2b;
  color:inherit;cursor:pointer;font:inherit;min-width:0;transition:background .15s,transform .15s,border-color .15s}
.vs-d:hover{background:#252838;transform:translateY(-2px)} .vs-d:focus-visible{outline:2px solid #fff;outline-offset:2px}
.vs-d .vs-ph{width:62px;height:62px;margin-bottom:4px}
.vs-d b{max-width:100%;font-size:14px;font-weight:900;text-transform:uppercase;letter-spacing:.04em;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.vs-d small{max-width:100%;font-size:12px;color:#9aa0ae;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.vs-d .pos{position:absolute;top:7px;left:9px;font:600 12px var(--f-num,ui-monospace,monospace);color:#c9cfdb}
.vs-d .tag{position:absolute;top:6px;right:6px;background:#e10632;color:#fff;font-size:11px;font-weight:700;letter-spacing:.06em;text-transform:uppercase;
  border-radius:6px;padding:1px 6px}
.vs-d[data-pick]{border-color:var(--c);background:color-mix(in srgb,var(--c) 20%,#1d1f2b)}
.vs-d[data-pick="b"] .tag{background:#fff;color:#000}
.vs-d.out{opacity:.5}
.vs-foot{flex:none;display:flex;justify-content:flex-end;gap:10px;padding:12px 24px 18px;border-top:1px solid #262838}
.vs-foot button{border-radius:999px;font:inherit;font-weight:700;font-size:15px;padding:10px 22px;cursor:pointer}
.vs-cancel{border:1px solid #3a3c4a;background:transparent;color:#e9ecf2} .vs-cancel:hover{background:#1f2130}
.vs-go{border:0;background:#e10632;color:#fff;min-width:140px} .vs-go:hover{background:#ff1744}
.vs-foot button:focus-visible{outline:2px solid #fff;outline-offset:2px}
@media (max-width:600px){
  .vs-back{place-items:end center;padding:0}
  .vs-sheet{width:100%;max-height:94vh;border-radius:22px 22px 0 0;border-bottom:0}
  .vs-head{padding:16px 14px 8px 16px} .vs-head h2{font-size:22px}
  .vs-stage{gap:8px;padding:2px 12px 0} .vs-slot{height:150px;border-radius:16px}
  .vs-big{font-size:84px} .vs-name b{font-size:clamp(13px,4.3vw,18px)} .vs-name small{font-size:13px} .vs-name{left:10px;right:8px}
  .vs-badge{width:44px;height:44px;font-size:15px} .vs-swap{width:32px;height:32px}
  .vs-hint{margin:10px 14px 0} .vs-grid{grid-template-columns:repeat(auto-fill,minmax(92px,1fr));padding:8px 12px 12px}
  .vs-foot{padding:10px 12px 14px} .vs-go{flex:1}
  .vs-bar{grid-template-columns:minmax(0,1fr) auto minmax(0,1fr)} .vs-change{display:none}
}
@keyframes vs-fade{from{opacity:0}}
@keyframes vs-up{from{transform:translateY(20px) scale(.97);opacity:0}}
@keyframes vs-pop{0%{transform:scale(.94)}100%{transform:none}}
@keyframes vs-beat{0%,100%{transform:scale(1)}50%{transform:scale(1.06)}}
@media (prefers-reduced-motion:reduce){.vs-back,.vs-sheet,.vs-slot.pop,.vs-badge{animation:none}}
</style>`);

  window.Versus = { open, close, bar, face };
})();
