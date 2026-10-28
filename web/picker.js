/* The race picker: a pop-up window with a tab for every season (real races) or your sessions (F1 25), each with
   its country's flag. Shared by f1.html and f1-25.html: RacePicker.open({...}) shows it, RacePicker.flag(code)
   and RacePicker.place(name) give the flag and the Grand Prix of a circuit for the pages' own buttons. */
(() => {
  "use strict";

  // Where each circuit is: the real races' circuit names and the game's tracks -> [country code, Grand Prix]
  const PLACES = {
    "Melbourne": ["au", "Australian Grand Prix"], "Shanghai": ["cn", "Chinese Grand Prix"],
    "Sakhir": ["bh", "Bahrain Grand Prix"], "Jeddah": ["sa", "Saudi Arabian Grand Prix"],
    "Suzuka": ["jp", "Japanese Grand Prix"], "Miami": ["us", "Miami Grand Prix"],
    "Imola": ["it", "Emilia Romagna Grand Prix"], "Monaco": ["mc", "Monaco Grand Prix"], "Monte Carlo": ["mc", "Monaco Grand Prix"],
    "Catalunya": ["es", "Spanish Grand Prix"], "Madring": ["es", "Spanish Grand Prix"], "Madrid": ["es", "Spanish Grand Prix"], "Montreal": ["ca", "Canadian Grand Prix"],
    "Silverstone": ["gb", "British Grand Prix"], "Spielberg": ["at", "Austrian Grand Prix"],
    "Hungaroring": ["hu", "Hungarian Grand Prix"], "Spa-Francorchamps": ["be", "Belgian Grand Prix"],
    "Zandvoort": ["nl", "Dutch Grand Prix"], "Monza": ["it", "Italian Grand Prix"], "Baku": ["az", "Azerbaijan Grand Prix"],
    "Singapore": ["sg", "Singapore Grand Prix"], "Austin": ["us", "United States Grand Prix"],
    "Mexico City": ["mx", "Mexico City Grand Prix"], "Interlagos": ["br", "São Paulo Grand Prix"],
    "Las Vegas": ["us", "Las Vegas Grand Prix"], "Lusail": ["qa", "Qatar Grand Prix"],
    "Yas Marina": ["ae", "Abu Dhabi Grand Prix"], "Yas Marina Circuit": ["ae", "Abu Dhabi Grand Prix"],
    "Paul Ricard": ["fr", "French Grand Prix"], "Le Castellet": ["fr", "French Grand Prix"],
    "Hockenheim": ["de", "German Grand Prix"], "Sochi": ["ru", "Russian Grand Prix"], "Hanoi": ["vn", "Vietnam Grand Prix"],
    "Portimao": ["pt", "Portuguese Grand Prix"], "Kuala Lumpur": ["my", "Malaysian Grand Prix"],
  };
  // The game's short and reverse layouts ("Silverstone Reverse") are at the same place
  const place = name => {
    const [, base, kind] = String(name || "").match(/^(.*?)(?: (Short|Reverse))?$/);
    const [code, gp] = PLACES[base] || [null, base || "Unknown track"];
    return [code, kind ? `${gp} (${kind.toLowerCase()})` : gp];
  };

  // Flags come from the flag-icons set; offline, the country's letters show instead
  const FLAGS = "https://cdn.jsdelivr.net/npm/flag-icons@7.2.3/flags/4x3/";
  const esc = s => String(s ?? "").replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const flag = code => code
    ? `<span class="rp-flag"><img src="${FLAGS}${esc(code)}.svg" alt="" data-code="${esc(code)}" decoding="async"></span>`
    : `<span class="rp-flag rp-noflag" aria-hidden="true"></span>`;
  document.addEventListener("error", e => {                 // error events don't bubble, but can be caught on the way down
    const img = e.target;
    if (img.tagName === "IMG" && img.closest?.(".rp-flag")) {
      const box = img.parentNode;
      box.classList.add("rp-noflag");
      box.textContent = (img.dataset.code || "").toUpperCase();
    }
  }, true);

  const css = `
.rp-backdrop{position:fixed;inset:0;z-index:100;background:rgba(5,7,10,.66);backdrop-filter:blur(6px);display:grid;
  grid-template-columns:minmax(0,1fr);place-items:center;padding:16px;animation:rp-fade .16s ease-out}
.rp-sheet{width:min(560px,100%);min-width:0;max-height:min(88vh,860px);display:flex;flex-direction:column;background:#14151d;color:#e9ecf2;
  border:1px solid #2a2c3a;border-radius:22px;overflow:hidden;box-shadow:0 30px 80px rgba(0,0,0,.6);animation:rp-up .2s ease-out;
  font-family:var(--f-ui,"Titillium Web","Segoe UI",system-ui,sans-serif)}
.rp-head{position:relative;flex:none;text-align:center;padding:34px 64px 18px;
  background:radial-gradient(120% 100% at 50% 0%,rgba(225,6,50,.45),rgba(225,6,50,.12) 55%,transparent 82%)}
.rp-close{position:absolute;top:14px;right:14px;width:44px;height:44px;border-radius:50%;border:1px solid rgba(255,255,255,.2);
  background:rgba(255,255,255,.08);color:#fff;cursor:pointer;display:grid;place-items:center}
.rp-close:hover{background:rgba(255,255,255,.16)}
.rp-logo{height:22px;display:block;margin:0 auto 14px}
.rp-title{margin:0;font-weight:900;font-size:clamp(30px,6vw,42px);letter-spacing:.01em;line-height:1;text-transform:uppercase}
.rp-tags{display:flex;gap:8px;justify-content:center;flex-wrap:wrap;margin-top:14px}
.rp-tags span{background:rgba(225,6,50,.16);color:#ff9aab;border:1px solid rgba(225,6,50,.38);border-radius:8px;padding:3px 10px;
  font-size:14px;font-weight:600}
.rp-tabs{flex:none;display:flex;gap:4px;margin:6px 20px 6px;padding:5px;background:#1f2130;border-radius:999px;overflow-x:auto;
  scrollbar-width:none}
.rp-tabs::-webkit-scrollbar{display:none}
.rp-tab{flex:1 0 auto;min-width:68px;border:0;background:transparent;color:#aab1c0;font:inherit;font-weight:600;font-size:16px;
  padding:9px 14px;border-radius:999px;cursor:pointer}
.rp-tab:hover{color:#fff}
.rp-tab[aria-selected="true"]{background:#fff;color:#15151e;box-shadow:0 2px 10px rgba(0,0,0,.35)}
.rp-list{overflow-y:auto;padding:4px 20px 22px;overscroll-behavior:contain}
.rp-month{font-size:13px;letter-spacing:.12em;text-transform:uppercase;color:#9aa0ae;font-weight:600;margin:18px 0 0 76px}
.rp-group{display:grid;grid-template-columns:62px minmax(0,1fr);gap:14px;padding:14px 0;border-bottom:1px solid #262838}
.rp-group:last-child{border-bottom:0}
.rp-when{display:flex;flex-direction:column;align-items:center;gap:5px;padding-top:4px}
.rp-when small{font-size:13px;letter-spacing:.08em;text-transform:uppercase;color:#9aa0ae;font-weight:700}
.rp-when b{width:44px;height:44px;border-radius:50%;border:1px solid #393c4e;display:grid;place-items:center;
  font-family:var(--f-num,ui-monospace,monospace);font-size:16px;font-weight:600;color:#dfe3ea}
.rp-items{display:grid;gap:10px;min-width:0}
.rp-item{display:flex;align-items:center;gap:14px;width:100%;min-height:60px;text-align:left;border:1px solid transparent;
  background:#1d1f2b;color:inherit;border-radius:14px;padding:12px 16px;font:inherit;cursor:pointer}
.rp-item:hover{background:#252838}
.rp-item[aria-current="true"]{border-color:rgba(225,6,50,.7);background:#26161e}
.rp-item .rp-text{min-width:0;flex:1}
.rp-item .t{display:block;font-weight:700;font-size:17px;line-height:1.2;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.rp-item .s{display:block;color:#9aa0ae;font-size:14px;margin-top:3px;font-family:var(--f-num,ui-monospace,monospace);
  overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.rp-chip{flex:none;font-size:12px;font-weight:700;letter-spacing:.08em;text-transform:uppercase;border-radius:6px;padding:3px 8px;
  background:#2c2f3e;color:#dfe3ea}
.rp-chip.live{background:#e10632;color:#fff}
.rp-flag{flex:none;width:30px;height:22px;border-radius:4px;overflow:hidden;background:#2c2f3e;display:inline-grid;place-items:center;
  box-shadow:0 0 0 1px rgba(255,255,255,.14);font-family:var(--f-num,ui-monospace,monospace);font-size:12px;font-weight:700;color:#dfe3ea}
.rp-flag img{width:100%;height:100%;object-fit:cover;display:block}
.rp-empty{color:#9aa0ae;text-align:center;padding:28px 12px;margin:0}
.rp-sr{position:absolute;width:1px;height:1px;overflow:hidden;clip:rect(0 0 0 0);white-space:nowrap}
/* the pages' own button that opens it, in their red title bar */
.lt-pick{display:inline-flex;align-items:center;gap:8px;max-width:min(380px,100%);background:transparent;border:0;
  border-bottom:1px solid rgba(255,255,255,.55);color:#fff;font:inherit;font-size:14px;padding:3px 0;cursor:pointer;text-align:left}
.lt-pick:hover{border-bottom-color:#fff}
.lt-pick .rp-flag{width:20px;height:15px;border-radius:2px;font-size:11px}
.lt-pick .n{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.lt-pick::after{content:"";flex:none;width:7px;height:7px;margin:0 4px 3px 2px;border-right:2px solid currentColor;
  border-bottom:2px solid currentColor;transform:rotate(45deg)}
@media (max-width:600px){
  .rp-backdrop{place-items:end center;padding:0}
  .rp-sheet{width:100%;max-height:92vh;border-radius:22px 22px 0 0;border-bottom:0}
  .rp-head{padding:28px 56px 14px}
  .rp-list,.rp-tabs{margin-inline:12px}.rp-list{padding-inline:0}
  .rp-group{grid-template-columns:54px minmax(0,1fr);gap:10px}
  .rp-when small{font-size:12px;letter-spacing:.04em}
  .rp-tab{min-width:0;padding-inline:10px}
  .rp-item{gap:12px;padding:12px}
  .rp-month{margin-left:64px}
}
@keyframes rp-fade{from{opacity:0}}
@keyframes rp-up{from{transform:translateY(18px);opacity:0}}
@media (prefers-reduced-motion:reduce){.rp-backdrop,.rp-sheet{animation:none}}`;
  document.head.insertAdjacentHTML("beforeend", `<style>${css}</style>`);

  let openNow = null;

  /* o = {title, tags (list, or a function of the tab), tabs: [{id, label}] (optional), tab, tabsLabel,
          groups(tab) -> [{month, when: {label, value}, items: [{id, flag, title, sub, chip, chipClass, current}]}],
          empty, onPick(id), opener (gets the focus back)} */
  function open(o) {
    close();
    const back = document.createElement("div");
    back.className = "rp-backdrop";
    back.innerHTML = `<div class="rp-sheet" role="dialog" aria-modal="true" aria-labelledby="rpTitle">
      <header class="rp-head">
        <button class="rp-close" type="button" aria-label="Close"><svg viewBox="0 0 24 24" width="22" height="22" aria-hidden="true">
          <path d="M6 6l12 12M18 6L6 18" stroke="currentColor" stroke-width="2.4" stroke-linecap="round"/></svg></button>
        <img class="rp-logo" src="f1-logo.png" alt="" onerror="this.remove()">
        <h2 class="rp-title" id="rpTitle">${esc(o.title)}</h2>
        <div class="rp-tags"></div>
      </header>
      ${o.tabs ? `<div class="rp-tabs" role="tablist" aria-label="${esc(o.tabsLabel || "Season")}">${o.tabs.map(t =>
        `<button class="rp-tab" role="tab" type="button" data-tab="${esc(t.id)}" aria-selected="false" tabindex="-1">${esc(t.label)}</button>`).join("")}</div>` : ""}
      <div class="rp-list"${o.tabs ? ' role="tabpanel"' : ""}></div>
    </div>`;
    const sheet = back.firstElementChild, list = sheet.querySelector(".rp-list"), tabs = [...sheet.querySelectorAll(".rp-tab")];

    const show = (tab, focusTab) => {
      tabs.forEach(b => { const on = b.dataset.tab === tab; b.setAttribute("aria-selected", String(on)); b.tabIndex = on ? 0 : -1; });
      const here = tabs.find(b => b.dataset.tab === tab);
      if (here) { here.scrollIntoView({ block: "nearest", inline: "center" }); if (focusTab) here.focus(); }
      const tags = typeof o.tags === "function" ? o.tags(tab) : o.tags || [];
      sheet.querySelector(".rp-tags").innerHTML = tags.map(t => `<span>${esc(t)}</span>`).join("");
      const groups = o.groups(tab);
      list.innerHTML = groups.map(g => (g.month ? `<div class="rp-month">${esc(g.month)}</div>` : "") +
        `<section class="rp-group"><div class="rp-when" aria-hidden="true"><small>${esc(g.when.label)}</small><b>${esc(g.when.value)}</b></div>` +
        `<div class="rp-items">${g.items.map(it => `<button class="rp-item" type="button" data-id="${esc(it.id)}"${it.current ? ' aria-current="true"' : ""}>` +
          `${flag(it.flag)}<span class="rp-text"><span class="rp-sr">${esc(g.when.label)} ${esc(g.when.value)}: </span>` +
          `<span class="t">${esc(it.title)}</span>${it.sub ? `<span class="s">${esc(it.sub)}</span>` : ""}</span>` +
          `${it.chip ? `<span class="rp-chip ${esc(it.chipClass || "")}">${esc(it.chip)}</span>` : ""}</button>`).join("")}</div></section>`).join("")
        || `<p class="rp-empty">${esc(o.empty || "Nothing here yet.")}</p>`;
      list.scrollTop = 0;
      list.querySelector('[aria-current="true"]')?.scrollIntoView({ block: "center" });
    };

    back.addEventListener("click", e => {
      if (e.target === back || e.target.closest(".rp-close")) return close();
      const tab = e.target.closest(".rp-tab");
      if (tab) return show(tab.dataset.tab);
      const item = e.target.closest(".rp-item");
      if (item) { close(); o.onPick(item.dataset.id); }
    });
    back.addEventListener("keydown", e => {
      if (e.key === "Escape") { e.preventDefault(); return close(); }
      const at = tabs.indexOf(document.activeElement);
      if (at >= 0 && ["ArrowLeft", "ArrowRight", "Home", "End"].includes(e.key)) {   // tabs: the arrow keys move along them
        e.preventDefault();
        const to = e.key === "Home" ? 0 : e.key === "End" ? tabs.length - 1 : (at + (e.key === "ArrowRight" ? 1 : -1) + tabs.length) % tabs.length;
        return show(tabs[to].dataset.tab, true);
      }
      if (e.key === "Tab") {                                  // the focus stays in the window while it's open
        const all = [...sheet.querySelectorAll("button:not([tabindex='-1'])")];
        const first = all[0], last = all.at(-1);
        if (e.shiftKey && document.activeElement === first) { e.preventDefault(); last.focus(); }
        else if (!e.shiftKey && document.activeElement === last) { e.preventDefault(); first.focus(); }
      }
    });

    document.body.append(back);
    document.documentElement.style.overflow = "hidden";       // the page behind doesn't scroll
    openNow = { back, opener: o.opener };
    show(o.tab ?? o.tabs?.[0]?.id);
    (list.querySelector('[aria-current="true"]') || tabs.find(b => b.tabIndex === 0) || sheet.querySelector(".rp-close")).focus({ preventScroll: true });
  }

  function close() {
    if (!openNow) return;
    const { back, opener } = openNow;
    openNow = null;
    back.remove();
    document.documentElement.style.overflow = "";
    opener?.focus();
  }

  window.RacePicker = { open, close, flag, place };
})();
