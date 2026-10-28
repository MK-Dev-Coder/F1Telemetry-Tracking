/* The pit stop chance in one plain sentence, for fans: what the model says, and its main reasons from SHAP.
   Shared by f1.html (real races) and f1-25.html (your game): PitWhy.html(name, chance, threshold, reasons, soon). */
(() => {
  "use strict";
  const esc = s => String(s ?? "").replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const pct = v => `${Math.round(v * 100)}%`;
  // "tyres are 7 laps old (like 28 in the real 57-lap race)" -> "tyres are 7 laps old": the sentence stays short
  const plain = text => String(text).replace(/\s*\([^)]*\)/g, "").trim();
  const both = list => list.length > 1 ? `${list.slice(0, -1).join(", ")} and ${list.at(-1)}` : list[0] || "";

  // Likely (at the alert line): the reasons pushing the chance up. Unlikely: the ones keeping it down.
  // you = the player in F1 25 ("You are likely to pit", "Your pit window is opening")
  function html(name, chance, threshold, reasons, soon = "in the next 3 laps", you = false) {
    if (chance == null) return "";
    const tone = chance >= threshold ? "box" : chance >= threshold / 2 ? "window" : "stay";
    const pick = sign => reasons.filter(r => Math.sign(r.points) === sign).slice(0, 2).map(r => esc(plain(r.text)));
    const why = tone === "stay" ? pick(-1) : pick(1);
    const [who, whose, is] = you ? ["You", "Your", "are"] : [esc(name), `${esc(name)}'s`, "is"];
    const head = { box: `${who} ${is} likely to pit ${esc(soon)}`, window: `${whose} pit window is opening`,
                   stay: `${who} ${is} unlikely to pit ${esc(soon)}` }[tone];
    const odds = tone === "stay" ? `only ${pct(chance)}` : `${pct(chance)} chance`;
    // "Norris's pit window is opening: 20% chance. Main reasons: on soft tyres and tyres are 20 laps old."
    return `<p class="why-line why-${tone}"><b>${head}:</b> ${odds}.` +
      (why.length ? ` ${why.length > 1 ? "Main reasons" : "Main reason"}: ${both(why)}.` : "") + `</p>`;
  }

  document.head.insertAdjacentHTML("beforeend", `<style>
.why-line{margin:4px 0 14px;padding:10px 12px 10px 14px;border-radius:8px;background:#1f2130;border-left:4px solid #8a96a8;
  font-size:15px;line-height:1.45;color:#e9ecf2}
.why-line b{font-weight:700;color:#fff}
.why-line.why-box{border-left-color:#e10632;background:#2a161d}
.why-line.why-window{border-left-color:#ffd60a;background:#26231a}
.why-line.why-stay{border-left-color:#3fa4ff;background:#16202d}
</style>`);

  window.PitWhy = { html };
})();
