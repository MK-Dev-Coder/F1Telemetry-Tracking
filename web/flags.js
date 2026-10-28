/* The small icons both pages use: race control's flags (as in the F1 app), the game's events, and the weather.
   Each is an inline SVG string. */
(() => {
  "use strict";
  const pole = `<path d="M5 3v18" stroke="#c9cfdb" stroke-width="2" stroke-linecap="round"/>`;
  const flag = (fill, at = 0) => `<path d="M${6 + at} ${4 + at}h12l-2.5 4.5L${18 + at} ${13 + at}H${6 + at}z" fill="${fill}" stroke="rgba(0,0,0,.4)" stroke-width=".8"/>`;
  const svg = inner => `<svg viewBox="0 0 24 24" width="24" height="24" aria-hidden="true">${inner}</svg>`;
  const chq = [0, 1, 2, 3].flatMap(c => [0, 1, 2].map(r => `<rect x="${6 + c * 3}" y="${4 + r * 3}" width="3" height="3" fill="${(c + r) % 2 ? "#111" : "#fff"}"/>`)).join("");
  const badge = (text, bg, fg) => `<svg viewBox="0 0 34 24" width="34" height="24" aria-hidden="true"><rect x="1" y="4" width="32" height="16" rx="4" fill="${bg}"/>` +
    `<text x="17" y="16.5" text-anchor="middle" font-size="${text.length > 3 ? 8.5 : 10}" font-weight="900" fill="${fg}" font-family="Titillium Web, sans-serif">${text}</text></svg>`;
  const ICON = {
    green: svg(pole + flag("#22d36b")), yellow: svg(pole + flag("#ffd60a")), red: svg(pole + flag("#ff3346")), blue: svg(pole + flag("#2f7bff")),
    yellow2: svg(pole + flag("#c9a800", -2) + flag("#ffd60a", 1)), chequered: svg(pole + chq),
    bw: svg(pole + `<path d="M6 4h12v9H6z" fill="#fff"/><path d="M6 4h12L6 13z" fill="#111"/><path d="M6 4h12v9H6z" fill="none" stroke="#8a96a8" stroke-width=".8"/>`),
    sc: badge("SC", "#ffd60a", "#15151e"), vsc: badge("VSC", "#ffd60a", "#15151e"),
    pit: svg(`<circle cx="12" cy="12" r="8.5" fill="none" stroke="#ffd60a" stroke-width="3"/><circle cx="12" cy="12" r="3" fill="#ffd60a"/>`),
    overtake: svg(`<path d="M6 13l6-6 6 6M6 19l6-6 6 6" fill="none" stroke="#7fd3ff" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"/>`),
    fastest: svg(`<circle cx="12" cy="13.5" r="7.5" fill="none" stroke="#d29bff" stroke-width="2.2"/><path d="M12 13.5V9M10 3h4" stroke="#d29bff" stroke-width="2.2" stroke-linecap="round"/>`),
    control: svg(`<rect x="5" y="3.5" width="14" height="17" rx="2" fill="none" stroke="#c9cfdb" stroke-width="1.8"/><path d="M8.5 8h7M8.5 12h7M8.5 16h4" stroke="#c9cfdb" stroke-width="1.8" stroke-linecap="round"/>`),
    // the game's events
    penalty: badge("PEN", "#ff3346", "#fff"), warning: badge("!", "#ffd60a", "#15151e"), retired: badge("OUT", "#5b6676", "#fff"),
    drs: badge("DRS", "#22d36b", "#06121f"), drsoff: badge("DRS", "#3a3c4a", "#c9cfdb"),
    ot: badge("OT", "#5ee0cf", "#06121f"), otoff: badge("OT", "#3a3c4a", "#c9cfdb"), aero: badge("AERO", "#3a3c4a", "#c9cfdb"),
    collision: svg(`<path d="M12 2.5l2 5.5 5.5-2-2.5 5 4.5 3-5.5 1 1 5.5-5-3-5 3 1-5.5-5.5-1 4.5-3-2.5-5 5.5 2z" fill="none" stroke="#ff8a8a" stroke-width="1.7" stroke-linejoin="round"/>`),
    winner: svg(`<path d="M7 4h10v4a5 5 0 0 1-10 0z" fill="none" stroke="#ffd60a" stroke-width="2"/><path d="M7 6H4a3 3 0 0 0 3 4M17 6h3a3 3 0 0 1-3 4M12 13v4M8 20h8" fill="none" stroke="#ffd60a" stroke-width="2" stroke-linecap="round"/>`),
  };
  // The weather, by the name the game gives it
  const cloud = (dy = 0, fill = "#c9cfdb") => `<path d="M7 ${17 + dy}h10a4 4 0 0 0 .3-8 5.5 5.5 0 0 0-10.6 1.2A3.5 3.5 0 0 0 7 ${17 + dy}z" fill="${fill}"/>`;
  const sun = (cx, cy, r) => `<circle cx="${cx}" cy="${cy}" r="${r}" fill="#ffd60a"/>` + [0, 45, 90, 135, 180, 225, 270, 315].map(a => {
    const c = Math.cos(a * Math.PI / 180), s = Math.sin(a * Math.PI / 180);
    return `<path d="M${(cx + c * (r + 2)).toFixed(1)} ${(cy + s * (r + 2)).toFixed(1)}L${(cx + c * (r + 4)).toFixed(1)} ${(cy + s * (r + 4)).toFixed(1)}" stroke="#ffd60a" stroke-width="1.6" stroke-linecap="round"/>`; }).join("");
  const drops = n => [8, 12, 16].slice(0, n).map(x => `<path d="M${x} 18.5l-1 3" stroke="#3fa4ff" stroke-width="2" stroke-linecap="round"/>`).join("");
  const WEATHER = {
    "Clear": svg(sun(12, 12, 4.5)), "Light cloud": svg(sun(9, 9, 3.5) + cloud(1)), "Overcast": svg(cloud(0, "#9aa0ae")),
    "Light rain": svg(cloud(-2) + drops(2)), "Heavy rain": svg(cloud(-2, "#9aa0ae") + drops(3)),
    "Storm": svg(cloud(-2, "#8a96a8") + `<path d="M13 15l-3 4h3l-2 4" fill="none" stroke="#ffd60a" stroke-width="1.8" stroke-linejoin="round"/>`),
  };
  window.Flags = { ICON, weather: name => WEATHER[name] || WEATHER.Overcast };
})();
