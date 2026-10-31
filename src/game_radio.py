"""F1 25 team radio: ask your race engineer with your voice while you drive, and hear the answer.

The F1 25 page turns your voice into text (the browser's speech recognition), sends it here, and speaks the
answer. The engineer's facts come only from your live data and the pit stop model. A local AI (Ollama, free
and offline) puts them into words like a real engineer would, so you can ask in your own words; it is given
the facts and nothing else, and its answer is checked (its pit call must match the model's, and no number or
name that isn't in the facts). Without Ollama, or if a check fails, the engineer answers from the data directly.
The engineer also calls things out without being asked: "box, box" when the model says so, the safety car,
your fastest lap, real penalties (not warnings) and the last lap.

    Set up the AI once:  ollama pull llama3.1:8b      (about 4.9 GB; any Ollama model works: F1_RADIO_MODEL)
"""
import os
import re
import threading
import time

import requests

import game_strategy
import game_telemetry

OLLAMA = os.environ.get("F1_RADIO_OLLAMA", "http://127.0.0.1:11434")
MODEL = os.environ.get("F1_RADIO_MODEL", "llama3.1:8b")
SYSTEM = """You are the race engineer of the player in the F1 25 video game, talking on the team radio during a session.
Answer the driver in one or two short sentences, the way an F1 race engineer speaks on the radio: calm, direct,
British English, no lists, no emojis, no markdown.
Use ONLY the facts you are given. If they don't answer the question, say you don't have that information.
Never make up a number, a name or a pit call. Say "box" only if the PIT CALL fact says BOX, and say "stay out"
if it says STAY OUT and the driver asks about pitting.
Only say what is true now: no forecasts, no guesses about what other drivers will do, and no opinions the facts
don't give (like "it's not critical").
If the driver asks how it's going in general, give the position and the gap first, like the plain answer.
Never call the driver by a name: on the radio the driver knows you're talking to them.
Call other drivers only by the names in the facts, exactly as written, never "he" or "she"."""
# Capitalised words an engineer may say that aren't names (anything else capitalised must be in the facts)
COMMON = {"I", "OK", "Okay", "Copy", "Box", "Stay", "Push", "Good", "Great", "Nice", "Well", "Yes", "No", "Mate",
          "Sorry", "Understood", "Fuel", "Tyres", "Battery", "Track", "Air", "Lap", "Last", "Best", "Fastest",
          "Safety", "Car", "Virtual", "Pit", "Rain", "Wet", "Dry", "Damage", "Front", "Rear", "Gap", "DRS", "ERS",
          "VSC", "SC", "P", "We", "You", "Your", "It", "That", "This", "The", "Keep", "Let", "Hold", "Bring", "Mediums",
          "Hards", "Softs", "Inters", "Wets", "Medium", "Hard", "Soft", "Overtake", "Hotlap", "Hot", "Plan", "Strategy"}


def unknown_names(answer, facts):
    known = set(re.findall(r"[A-Za-z]+", facts.lower()))
    return [w for w in re.findall(r"\b[A-Z][a-z]+\b", answer) if w not in COMMON and w.lower() not in known]


def tidy_names(answer, facts):
    """The AI knows real drivers' names and likes to talk to the driver by name ("Verstappen, ...", "..., Max.").
    The driver's own name isn't in the facts, so like any name that isn't, it's dropped where it's only used to
    talk to the driver; anywhere else it fails the check below."""
    for w in set(unknown_names(answer, facts)):
        answer = re.sub(rf",\s*{w}([.!?,])", r"\1", answer)
        answer = re.sub(rf"^{w},\s*(\w)", lambda m: m.group(1).upper(), answer)
    return answer


def names_ok(answer, facts):
    """Every name in the AI's answer must be in the facts."""
    return not unknown_names(answer, facts)

ERS_MODES = {0: "off", 1: "medium", 2: "hotlap", 3: "overtake"}
ERS_MODES_2026 = {**ERS_MODES, 3: "boost"}     # the 2026 format calls mode 3 boost (overtake mode is its own thing)
CORNERS = ["rear left", "rear right", "front left", "front right"]       # the game's order: RL, RR, FL, FR
# What each question is about: the words that point to it. A word naming the subject counts 3, a counting
# word ("how many", "left"...) or a radio check word counts 1, so "how many laps left on these tyres" is about
# the tyres and "copy, what's the gap behind?" is about the car behind. The first topic with the highest score
# wins; the radio check is last, so it only wins when nothing else is asked. "Chance" and "finish" are asked about
# other things too ("chance of rain", "laps to the finish"), so the chances come after the other topics and
# "finish" counts 1.
RADIO_CHECK = ["radio check", "radio shack", "radio chuck", "hear me", "can you hear", "copy"]   # as browsers hear it
GENERIC = {"laps", "how many", "remaining", "to go", "how long", "left", "finish", *RADIO_CHECK}
TOPICS = [
    ("pit", ["pit", "box", "stop", "strategy", "plan", "when do i come in", "undercut"]),
    ("tyres", ["tyre", "tire", "tyres", "tires", "wear", "grip", "rubber", "degradation", "deg"]),
    ("behind", ["behind", "chasing", "defend", "car behind", "who is behind"]),
    ("leader", ["leader", "leading", "the lead", "first place"]),
    ("overtake", ["overtake", "overtake mode", "push to pass", "manual override"]),      # F1 26
    ("ahead", ["ahead", "in front", "gap", "catch", "catching"]),
    ("position", ["position", "where am i", "what place", "p what", "standing"]),
    ("fuel", ["fuel", "petrol", "gas", "lift and coast"]),
    ("ers", ["ers", "battery", "deploy", "energy", "boost", "harvest", "harvesting"]),
    ("brakes", ["brake", "brakes", "break", "breaks"]),          # browsers often hear "brake" as "break"
    ("aero", ["aero", "active aero", "straight mode", "corner mode", "straight line mode", "drs"]),
    ("weather", ["weather", "rain", "rains", "raining", "temperature", "track temp", "wet", "dry", "drying", "forecast"]),
    ("laps", ["laps", "how many", "remaining", "to go", "how long", "left"]),
    ("pace", ["pace", "lap time", "last lap", "best lap", "fastest", "how fast", "quick", "slow"]),
    ("damage", ["damage", "wing", "broken", "floor", "is the car ok", "car ok"]),
    ("chances", ["chance", "chances", "win", "podium", "points", "can i win", "where will i finish", "finish",
                 "predict", "prediction"]),                    # the race outcome model
    ("radio_check", RADIO_CHECK),
]
HELP = ("You can ask about pit stops, your chances (win, podium, points), tyres, the gap ahead or behind, the leader, "
        "position, fuel, battery, brakes, weather, laps, pace, overtake mode or active aero.")
GONE = set(game_strategy.OUT.values())         # out of the race: DNF, DSQ... (they keep a place at the back)


def topic_of(text):
    text = " " + re.sub(r"[^a-z0-9 ]", " ", text.lower()) + " "
    best, most = None, 0
    for name, words in TOPICS:
        hits = sum(1 if w in GENERIC else 3 for w in words if f" {w} " in text)
        if hits > most:
            best, most = name, hits
    return best


def sentence(text):
    """The first letter in capitals, the rest as it was (str.capitalize would lowercase "Jeddah" and "F1")."""
    return text[:1].upper() + text[1:]


def next_tyre_name(d):
    """The tyre the model expects a car to fit, as the engineer says it ("the hards")."""
    best = max(d["next_tyre"].items(), key=lambda kv: kv[1])[0]
    return {"SOFT": "the softs", "MEDIUM": "the mediums", "HARD": "the hards", "INTER/WET": "inters"}.get(best, best.lower())


# The safety car events in the engineer's words (the game sends deployed, returning, returned, resume race)
NEUTRAL_CALLS = {"Safety car deployed": "Safety car, safety car. Keep the delta positive.",
                 "Safety car returning": "Safety car in this lap. Get ready for the restart.",
                 "Safety car resume race": "Green, green. Racing.",
                 "Virtual safety car deployed": "VSC, VSC. Keep the delta.",
                 "Virtual safety car returning": "VSC ending. Get ready.",
                 "Virtual safety car resume race": "Green, green. Racing."}


# F1 26: race control's word on active aero and overtake mode
RACE_CONTROL_2026 = {"PMEN": "Active aero on partial mode", "PMDI": "Full active aero again",
                     "OVEN": "Overtake mode is enabled", "OVDI": "Overtake mode is disabled"}


def wet(sample):
    """Is a forecast sample rain (the weather says so, or a 50% chance or more)?"""
    return sample["weather"] in game_strategy.RAIN or sample["rain"] >= 50


def rain_ahead(info, within=60):
    """The first forecast sample in the next minutes that is rain, and the samples looked at."""
    ahead = [f for f in info.get("forecast") or [] if 0 < f["in"] <= within]
    return next((f for f in ahead if wet(f)), None), ahead


def lap_at(info, laps, lap_now, minutes):
    """In a race: the lap the car will be on in that many minutes, at its usual pace (the middle of its clean laps)."""
    times = sorted(l["time"] for l in laps or [] if l.get("time") and not l.get("invalid") and not l.get("pitted"))
    if not times or not lap_now:
        return ""
    at, total = lap_now + round(minutes * 60 / times[len(times) // 2]), info.get("total_laps")
    return ", after the flag" if total and at > total else f", about lap {at}"


def forecast_text(info, laps=None, lap_now=None):
    """The game's forecast in the engineer's words: rain coming (and on which lap, in a race), drying up, or dry."""
    soon, ahead = rain_ahead(info)
    if not ahead:
        return ""
    if info.get("weather") in game_strategy.RAIN:
        dry = next((f for f in ahead if not wet(f)), None)
        return (f" Drying up in {dry['in']} minutes{lap_at(info, laps, lap_now, dry['in'])}." if dry
                else f" Still wet in {ahead[-1]['in']} minutes.")
    if soon:
        return f" Rain in {soon['in']} minutes{lap_at(info, laps, lap_now, soon['in'])}, {soon['rain']} percent chance."
    return f" No rain in the next {ahead[-1]['in']} minutes."


def lap_time(s):
    """1:41.2 for the screen; "1 41 point 2" is what the voice says."""
    if not s:
        return "no time", "no time"
    m, rest = divmod(round(s, 1), 60)               # rounded first: 1:59.96 is 2:00.0, not 1:60.0
    return f"{m:.0f}:{rest:04.1f}", f"{m:.0f} {rest:04.1f}".replace(".", " point ")


def name_of(row):
    return (row.get("name") or row.get("driver") or "the car").title()


def speakable(text):
    """What the voice says: lap times as an engineer reads them (1:41.2 -> "1 41 point 2")."""
    return re.sub(r"(\d+):(\d{2})\.(\d+)", r"\1 \2 point \3", text)


# Numbers the AI may write as words ("two seconds", "half a second"), with their unit, and places ("in second")
NUMBER_WORDS = {w: n for n, w in enumerate("zero one two three four five six seven eight nine ten eleven twelve thirteen "
                                           "fourteen fifteen sixteen seventeen eighteen nineteen twenty".split())}
NUMBER_WORDS |= {"thirty": 30, "forty": 40, "fifty": 50, "sixty": 60}
PLACES = {w: n for n, w in enumerate("zeroth first second third fourth fifth sixth seventh eighth ninth tenth eleventh "
                                     "twelfth thirteenth fourteenth fifteenth sixteenth seventeenth eighteenth "
                                     "nineteenth twentieth".split())}
UNITS = {"s": "s", "sec": "s", "secs": "s", "second": "s", "seconds": "s", "lap": "lap", "laps": "lap",
         "percent": "%", "%": "%", "degree": "deg", "degrees": "deg", "metre": "m", "metres": "m", "meter": "m",
         "meters": "m", "megajoule": "MJ", "megajoules": "MJ"}
TIME_WORDS = [(r"\bhalf a second\b", 0.5, 1), (r"\ba tenth\b", 0.1, 1), (r"\ba second\b", 1, 1)]   # "a second" = 1.0 s
ORDINAL = "|".join(PLACES)


def numbers_in(text):
    """Every number in a text as (value, decimals, unit): in figures, in words, "half a second"; a lap time
    (1:35.6) is one number, in seconds."""
    text, found = text.lower(), []
    for m in re.finditer(r"(\d+):(\d{2}(?:\.\d+)?)", text):
        found.append((int(m[1]) * 60 + float(m[2]), len(m[2].partition(".")[2]), "time"))
    text = re.sub(r"\d+:\d{2}(?:\.\d+)?", " ", text)
    for phrase, value, digits in TIME_WORDS:
        found += [(value, digits, "s")] * len(re.findall(phrase, text))
        text = re.sub(phrase, " ", text)
    words = re.findall(r"\d+(?:\.\d+)?|[a-z]+|%", text)
    for k, w in enumerate(words):
        if w[0].isdigit():
            value, digits = float(w), len(w.partition(".")[2])
        elif w in NUMBER_WORDS:
            value, digits = NUMBER_WORDS[w], 0
        else:
            continue
        after = words[k + 1] if k + 1 < len(words) else ""
        if after in ("tenth", "tenths"):                          # "two tenths"
            found.append((value / 10, digits + 1, "s"))
        else:
            found.append((value, digits, UNITS.get(after)))
    return found


def places_in(text):
    """The race positions a text gives: "P3", "in second", "fifth place", "you're leading" (P1)."""
    text = text.lower()
    found = [int(n) for n in re.findall(r"\bp\s?(\d{1,2})\b", text)]
    found += [PLACES[w] for w in re.findall(rf"\b(?:in|to|up to|down to|running|from)\s+({ORDINAL})\b"
                                            r"(?!\s+(?:sector|lap|gear|set|stint|corner|turn|stop))", text)]
    found += [PLACES[w] for w in re.findall(rf"\b({ORDINAL})\s+(?:place|position)\b", text)]
    return found + [1] * len(re.findall(r"\b(?:you'?re|you are|we'?re|we are)\s+(?:leading|in the lead)\b", text))


def numbers_ok(answer, facts):
    """Every number in the AI's answer, in figures or in words, must be a number of the facts with the same
    unit, at most rounded to the answer's own decimals: "27 seconds" for 27.3 seconds passes, "1.2 seconds" for
    1 and "a second" for a 10-second gap don't. Every position it gives must be one the facts give: "P2" when
    you lead doesn't pass, even if some other fact has a 2. The driver and session lines don't count."""
    data = "\n".join(line for line in facts.splitlines() if not line.startswith(("DRIVER:", "SESSION:")))
    known = numbers_in(data)
    numbers = all(any(abs(x - k) <= 0.5 * 10 ** -d + 1e-9 and u in (None, unit) for k, _, unit in known)
                  for x, d, u in numbers_in(answer))
    return numbers and set(places_in(answer)) <= set(places_in(data))


# The pit call in the AI's words must be the model's. A list of ways to say "pit" is never complete, so an
# answer about pitting must say the call itself: "box" when the model says box, "stay out" when it doesn't.
SAYS_BOX = re.compile(r"\bbox(?:ing)?\b")
SAYS_STAY = re.compile(r"\bstay(?:ing)? out\b")
SAYS_WAIT = re.compile(r"\b(?:don'?t|do not|no need to|not) (?:pit|box|come in|stop)\b|\bnot yet\b|\bhold on\b")
SAYS_PIT = re.compile(r"\bpit(?:ting)? (?:now|this lap|next lap)\b|\bcome in\b|\bbring you in\b|\binto the pits?\b"
                      r"|\btime to pit\b|\b(?:we'?ll|let'?s|we will|we're going to) pit\b")
MENTIONS_PIT = re.compile(r"\b(?:box|boxing|pit|pits|pitting|stop|stopping|come in|bring you in)\b")


def pit_call_ok(answer, box, topic):
    """BOX: an answer about pitting, the tyres or how it's going must say box, and no answer may say to stay
    out or wait. STAY OUT: no answer may say box or pit now, and one about pitting must say stay out."""
    said = answer.lower()
    if box:
        must = topic in ("pit", "tyres", None) or MENTIONS_PIT.search(said)
        return not (SAYS_STAY.search(said) or SAYS_WAIT.search(said)) and (bool(SAYS_BOX.search(said)) or not must)
    plain = SAYS_WAIT.sub(" ", said)                          # "no need to box" isn't a box call
    if SAYS_BOX.search(plain) or SAYS_PIT.search(plain):
        return False
    return bool(SAYS_STAY.search(said)) or not (topic == "pit" or MENTIONS_PIT.search(plain))


# The local AI (Ollama). Its state is checked at most every 30 s, so a page asking often costs nothing.
# "warm" = the model is loaded into the graphics card (about 10 s the first time); until then the engineer
# answers from the data straight away instead of making you wait.
_ai = {"checked": 0.0, "ready": False, "warm": False, "warming": False, "note": "", "load_error": "", "failed": 0.0}


def ai_status():
    if time.time() - _ai["checked"] > 30:
        _ai["checked"] = time.time()
        try:
            names = [m["name"] for m in requests.get(f"{OLLAMA}/api/tags", timeout=2).json().get("models", [])]
            ready = any(n in (MODEL, f"{MODEL}:latest") for n in names)          # "mistral" is "mistral:latest"
            _ai.update(ready=ready, note="" if ready else f"Ollama is running, but the model {MODEL} isn't there: "
                                                             f"run  ollama pull {MODEL}")
            # Load the model into the graphics card now, not at the first question (after a failed load, e.g.
            # not enough memory, it's tried again every 5 minutes)
            if ready and not _ai["warming"] and (not _ai["load_error"] or time.time() - _ai["failed"] > 300):
                _ai["warming"] = True
                threading.Thread(target=_warm_up, daemon=True).start()
        except (requests.RequestException, ValueError):
            _ai.update(ready=False, warm=False, note="Ollama isn't running, so the engineer answers from the data.")
    note = _ai["note"] or _ai["load_error"] or ("" if _ai["warm"] else
                                                "The AI is loading (about 10 seconds); the data answers until then.")
    # "ai" = the AI can answer now or soon; a model that failed to load can't, so the page shows the data engineer
    return {"ai": _ai["ready"] and not _ai["load_error"], "warm": _ai["warm"], "model": MODEL, "note": note}


def _warm_up():
    try:
        r = requests.post(f"{OLLAMA}/api/generate", json={"model": MODEL, "prompt": "", "keep_alive": "30m"}, timeout=120)
        r.raise_for_status()                     # e.g. not enough memory for the model: Ollama answers with an error
        _ai.update(warm=True, load_error="")
    except requests.RequestException as err:
        _ai.update(warm=False, failed=time.time(),
                   load_error=f"Ollama couldn't load {MODEL} ({err}), so the engineer answers from the data.")
    finally:
        _ai["warming"] = False


def ask_ai(question, facts, plain):
    """The AI's answer from the facts, or None if Ollama isn't there, is still loading, or doesn't answer in time."""
    status = ai_status()
    if not (status["ai"] and status["warm"]):
        return None
    body = {"model": MODEL, "stream": False, "keep_alive": "30m", "options": {"temperature": 0.2, "num_predict": 90},
            "messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": f"FACTS (live data):\n{facts}\n\nA plain answer from the data: {plain}"
                                                     f"\n\nThe driver says: {question}"}]}
    try:
        reply = requests.post(f"{OLLAMA}/api/chat", json=body, timeout=12).json()["message"]["content"]
    except (requests.RequestException, ValueError, KeyError):
        return None
    reply = re.sub(r"[*_#`]", "", reply).strip().strip('"')
    return reply[:300] or None


class Radio:
    """One engineer per session: answers questions, and remembers what it has already called out."""

    def __init__(self):
        self.lock = threading.Lock()
        self.calls = {}                       # session uid -> [{"id", "text", "say"}] called out without asking
        self.seen = {}                        # session uid -> what was already called out

    # -- answers to your questions ------------------------------------------------------------------------
    def answer(self, s, lock, text, use_ai=True):
        """The engineer's answer: the AI's words when it's there and its answer passes the checks, else the data's."""
        start = time.perf_counter()
        snap = self.snapshot(s, lock)
        plain = self.data_answer(s, lock, text, snap=snap)
        if use_ai and (text or "").strip() and plain["topic"] != "radio_check":     # "Loud and clear" needs no AI
            # A clear question gets only its own facts (so the AI can't mix up, say, laps left in the race and in
            # the tyres); a general one ("how's it going?") gets them all
            facts, box = self.facts(s, lock, only=plain["topic"], snap=snap)
            said = ask_ai(text, facts, plain["text"])
            said = said and tidy_names(said, facts)
            if (said and pit_call_ok(said, box, plain["topic"]) and numbers_ok(said, facts) and names_ok(said, facts)
                    and not re.search(r"\b(he|she|him|her|his|he's|she's)\b", said.lower())):
                return {"topic": plain["topic"], "text": said, "say": speakable(said), "source": "ai",
                        "ms": round(1000 * (time.perf_counter() - start))}
            plain["note"] = "the AI's answer didn't pass the checks" if said else ai_status()["note"]
        return {**plain, "source": "data", "ms": round(1000 * (time.perf_counter() - start))}

    def facts(self, s, lock, only=None, snap=None):
        """What the engineer knows right now, one line per topic (what the AI may use), and the pit call.
        only = one topic: just that topic's facts (with the pit call, which is always there)."""
        snap = snap or self.snapshot(s, lock)
        # The driver's name isn't given: an engineer doesn't use it, and a name that isn't in the facts fails the
        # names check, so the AI can't start every answer with "Verstappen, ..."
        info = snap[2]
        lines = ["DRIVER: you are talking to the player (no name)",
                 f"SESSION: {info.get('track', 'unknown track')}, {info.get('session_type', 'session')}"]
        modes = bool(snap[1].get("modes"))          # overtake mode: only the 2026 cars have it
        box = False
        for name, _ in TOPICS:
            if name == "radio_check" or only and name not in (only, "pit") or name == "overtake" and not modes and not only:
                continue
            a = self.data_answer(s, lock, None, topic=name, snap=snap)
            if name == "pit":
                box = a["text"].startswith("Box")
                lines.append("PIT CALL: " + ("BOX" if box else "STAY OUT"))
                if only and only != "pit":
                    continue
            lines.append(f"{name.upper()}: {a['text']}")
        return "\n".join(lines), box

    @staticmethod
    def snapshot(s, lock):
        """The live data the answers are made from, read once under the recorder's lock."""
        with lock:
            board, car, info, me_i = s.leaderboard(), s.car_now(), dict(s.info), s.player
            car["fitted_set"] = next((t for t in s.tyre_set_list() if t["fitted"]), None)
            laps = s.player_laps()
        return board, car, info, me_i, laps

    def data_answer(self, s, lock, text, topic=None, snap=None):
        topic = topic or topic_of(text or "")
        board, car, info, me_i, laps = snap or self.snapshot(s, lock)
        me = next((r for r in board if r["car"] == me_i), None)
        if topic is None and me is not None:                 # a general question: how it's going, in short
            parts = [self._position(s, lock, me, board, car, info, laps)["text"],
                     self._ahead(s, lock, me, board, car, info, laps)["text"],
                     self._tyres(s, lock, me, board, car, info, laps)["text"].split(".")[0] + "."]
            return self._say(None, " ".join(parts))
        if topic is None:
            return self._say(None, "Sorry, say again? " + HELP)
        if topic == "radio_check":
            return self._say(topic, "Loud and clear.")
        if me is None:
            return self._say(topic, "No data from the car yet. Go out on track and ask again.")
        maker = getattr(self, f"_{topic}")
        return maker(s, lock, me, board, car, info, laps)

    def _say(self, topic, text, say=None):
        return {"topic": topic, "text": text, "say": say or text}

    def _pit(self, s, lock, me, board, car, info, laps):
        if not s.is_race():
            return self._say("pit", "No pit strategy in this session. That's for the race.")
        snap = game_strategy.snapshot(s, lock)
        d = next((x for x in snap.get("drivers", []) if x["player"]), None) if snap.get("available") else None
        if d is None:
            return self._say("pit", "Strategy comes after the first lap.")
        tyre = next_tyre_name(d)
        if me["status"] == "FINISHED":
            return self._say("pit", "The race is done. No more stops.")
        if d["pitting_now"]:
            return self._say("pit", f"We're in the pits. Going onto {tyre}.")
        if d["pit_chance"] is None:                          # only when the car is out of the race
            return self._say("pit", "We're out of the race. No more strategy today.")
        chance, why = d["pit_chance"], d["reasons"][0]["text"] if d["reasons"] else ""
        because = f": {why}." if why else "."
        if d["alert"]:
            return self._say("pit", f"Box, box. Box this lap for {tyre}. Pit chance {chance:.0%}{because}")
        if chance >= snap["alert_threshold"] / 2:
            return self._say("pit", f"Pit window is opening. Chance {chance:.0%}{because} We'd go onto {tyre}.")
        return self._say("pit", f"Stay out, stay out. Pit chance only {chance:.0%}." + (f" {sentence(why)}." if why else ""))

    def _chances(self, s, lock, me, board, car, info, laps):
        """The race outcome model: the chance to win, finish on the podium and score points, and the main reason."""
        if not s.is_race():
            return self._say("chances", "Chances are for the race.")
        if me["status"] == "FINISHED":
            return self._say("chances", f"Race done. You finished P{me['position']}.")
        if me["status"] in GONE:
            return self._say("chances", "We're out of the race. No more chances today.")
        snap = game_strategy.snapshot(s, lock)
        d = next((x for x in snap.get("drivers", []) if x["player"]), None) if snap.get("available") else None
        o = d and d.get("outcome")
        if not o:
            return self._say("chances", "Chances come once every car has finished a lap.")
        pct = lambda v: "under 1%" if v < 0.005 else "over 99%" if v > 0.995 else f"{v:.0%}"
        text = (f"Likely finish P{o['likely']}. Win {pct(o['win'])}, podium {pct(o['podium'])}, "
                f"points {pct(o['points'])}.")
        if o.get("reason"):
            text += f" Main reason: {sentence(o['reason']['text'])}."
        return self._say("chances", text)

    def _tyres(self, s, lock, me, board, car, info, laps):
        wear, temps = car.get("wear"), car.get("surface_temp")
        if car["compound"] == "UNKNOWN" and not wear:
            return self._say("tyres", "No tyre data from the car yet.")
        age, last = car.get("tyre_age"), laps[-1] if laps else None
        # The game counts the lap just finished a moment after the line; the lap's own record (what the pit
        # model's "tyres are N laps old" comes from) already has it, so the two always say the same
        if (age is not None and last and last.get("tyre_age") is not None and last["compound"] == car["compound"]
                and not last["pitted"]):
            age = max(age, last["tyre_age"] + 1)
        name = "Tyres" if car["compound"] == "UNKNOWN" else f"{car['compound'].lower().capitalize()}s"
        text = name + (f", {age} lap{'' if age == 1 else 's'} old." if age is not None else ".")
        if wear:
            worst = max(range(4), key=lambda k: wear[k])
            text += f" Wear {round(sum(wear) / 4)} percent on average, most on the {CORNERS[worst]}, {round(wear[worst])} percent."
        if temps and all(t is not None for t in temps):
            hot = max(range(4), key=lambda k: temps[k])
            text += f" Hottest is the {CORNERS[hot]}, {round(temps[hot])} degrees."
        fitted = car.get("fitted_set")
        if fitted and fitted.get("laps_left") is not None:
            text += f" About {fitted['laps_left']} laps left in this set."
        return self._say("tyres", text)

    def _ahead(self, s, lock, me, board, car, info, laps):
        if me["position"] == 1:
            behind = next((r for r in board if r["position"] == 2), None)
            gap = f" {behind['interval']:.1f} seconds to {name_of(behind)} behind." if behind and behind.get("interval") else ""
            return self._say("ahead", "You're leading." + gap)
        ahead = next((r for r in board if r["position"] == me["position"] - 1), None)
        if ahead is None or me.get("interval") is None:
            return self._say("ahead", f"P{me['position']}. No gap yet.")
        text = f"{me['interval']:.1f} seconds to {name_of(ahead)}."
        trend = self._trend(s, lock, me["car"], ahead["car"])
        if trend is not None and abs(trend) >= 0.05:           # names, not "he": whoever drives the car
            text += (f" You're catching {name_of(ahead)} by {abs(trend):.1f} seconds a lap." if trend < 0
                     else f" {name_of(ahead)} is pulling away by {trend:.1f} seconds a lap.")
        return self._say("ahead", text)

    def _leader(self, s, lock, me, board, car, info, laps):
        if me["position"] == 1:
            return self._ahead(s, lock, me, board, car, info, laps) | {"topic": "leader"}
        leader = next((r for r in board if r["position"] == 1), None)
        if leader is None or me.get("gap") is None:
            return self._say("leader", f"P{me['position']}. No gap to the leader yet.")
        return self._say("leader", f"{me['gap']:.1f} seconds to the leader, {name_of(leader)}. You're P{me['position']}.")

    def _behind(self, s, lock, me, board, car, info, laps):
        # The next car still racing: retired cars keep a place at the back, with the gap they had
        behind = next((r for r in board if r["position"] == me["position"] + 1 and r["status"] not in GONE), None)
        if behind is None:
            return self._say("behind", "Nobody behind you.")
        if behind.get("interval") is None:
            return self._say("behind", f"{name_of(behind)} is behind.")
        text = f"{name_of(behind)} is {behind['interval']:.1f} seconds behind."
        trend = self._trend(s, lock, behind["car"], me["car"])
        if trend is not None and abs(trend) >= 0.05:
            text += (f" {name_of(behind)} is catching you by {abs(trend):.1f} seconds a lap." if trend < 0
                     else f" You're pulling away by {trend:.1f} seconds a lap.")
        return self._say("behind", text)

    @staticmethod
    def _trend(s, lock, chaser, target):
        """How the gap from the chasing car to the car ahead of it changed over the last lap (- = closing)."""
        with lock:
            a, b = s.laps.get(chaser, {}), s.laps.get(target, {})
            both = sorted(n for n in a if n in b and a[n].get("gap_to_leader") is not None and b[n].get("gap_to_leader") is not None
                          and a[n].get("t") is not None and b[n].get("t") is not None)
            if len(both) < 2:
                return None
            gap = lambda n: a[n]["gap_to_leader"] - b[n]["gap_to_leader"]
            return gap(both[-1]) - gap(both[-2])

    def _position(self, s, lock, me, board, car, info, laps):
        total = info.get("total_laps")
        return self._say("position", f"P{me['position']}. Lap {me['lap']}" + (f" of {total}." if total else "."))

    def _fuel(self, s, lock, me, board, car, info, laps):
        left = car.get("fuel_laps")
        if left is None:
            return self._say("fuel", "No fuel reading yet.")
        if left >= 0.2:
            return self._say("fuel", f"Fuel is fine, plus {left:.1f} laps. Push.")
        if left >= -0.2:
            return self._say("fuel", "Fuel is on target. Keep this pace.")
        return self._say("fuel", f"Fuel is short, minus {abs(left):.1f} laps. Lift and coast into the big stops.")

    def _ers(self, s, lock, me, board, car, info, laps):
        ers = car.get("ers")
        if ers is None:
            return self._say("ers", "No battery reading yet.")
        modes = ERS_MODES_2026 if s.format >= 2026 else ERS_MODES
        text = f"Battery at {min(100, round(ers / 4e6 * 100))} percent, mode {modes.get(car.get('ers_mode'), 'unknown')}."
        if car.get("ers_limit"):                             # 2026 format: the most the car may harvest per lap
            text += (f" Harvested {car['ers_harvested'] / 1e6:.1f} megajoules this lap,"
                     f" the limit is {car['ers_limit'] / 1e6:.1f} megajoules.")
        return self._say("ers", text)

    def _overtake(self, s, lock, me, board, car, info, laps):
        """F1 26: overtake mode (extra electric power, near the car ahead), and the gap to that car in a race."""
        m = car.get("modes")
        if not m:
            return self._say("overtake", "No overtake mode data. That comes with the 2026 cars, in UDP format 2026.")
        ahead = next((r for r in board if r["position"] == me["position"] - 1), None) if s.is_race() else None
        gap = f" {name_of(ahead)} ahead, {me['interval']:.1f} seconds." if ahead and me.get("interval") else ""
        if info.get("overtake_mode") is False:
            return self._say("overtake", "Race control has turned overtake mode off." + gap)
        if m["overtake_active"]:
            return self._say("overtake", "Overtake mode is on. Go for it." + gap)
        if m["overtake_available"]:
            return self._say("overtake", "Overtake mode is ready. Use it." + gap)
        if m["overtake_distance"]:
            return self._say("overtake", f"Overtake mode in {m['overtake_distance']} metres." + gap)
        return self._say("overtake", "No overtake mode right now." + gap)

    def _aero(self, s, lock, me, board, car, info, laps):
        """F1 26: active aero (straight mode on the straights, corner mode elsewhere); older cars: DRS."""
        m = car.get("modes")
        if not m:
            if car.get("drs_allowed"):
                return self._say("aero", "DRS is available.")
            far = car.get("drs_distance")
            return self._say("aero", f"DRS in {far} metres." if far else "No DRS right now.")
        text = f"Active aero in {'straight' if m['aero_mode'] else 'corner'} mode."
        if not m["aero_mode"] and m["aero_distance"]:
            text += f" Straight mode in {m['aero_distance']} metres."
        if info.get("aero") == "PARTIAL":
            text += " Race control has it on partial mode."
        return self._say("aero", text)

    def _weather(self, s, lock, me, board, car, info, laps):
        w = info.get("weather") or "unknown"
        rain = w in game_strategy.RAIN
        return self._say("weather", f"{w}. Track {info.get('track_temp')} degrees, air {info.get('air_temp')} degrees."
                         + (" It's wet, careful on the white lines." if rain else " It's dry.")
                         + forecast_text(info, laps, me["lap"] if s.is_race() else None))

    def _brakes(self, s, lock, me, board, car, info, laps):
        temps = car.get("brake_temp")
        if not temps or any(t is None for t in temps):
            return self._say("brakes", "No brake reading yet.")
        hot = max(range(4), key=lambda k: temps[k])
        return self._say("brakes", f"Hottest brake is the {CORNERS[hot]}, {round(temps[hot])} degrees.")

    def _laps(self, s, lock, me, board, car, info, laps):
        total = info.get("total_laps")
        if s.is_race() and me["status"] in GONE | {"FINISHED"}:              # the lap count stops at the flag
            return self._say("laps", "The race is over for us." if me["status"] in GONE else f"Race done. P{me['position']}.")
        if s.is_race() and total:
            left = max(0, total - me["lap"] + 1)
            return self._say("laps", "This is the last lap." if left <= 1 else f"{left} laps to go, including this one.")
        left = info.get("time_left")
        if left is None:
            return self._say("laps", "No session time yet.")
        return self._say("laps", f"{left // 60} minutes {left % 60} seconds left in the session." if left else "The session is over.")

    def _pace(self, s, lock, me, board, car, info, laps):
        last, best = lap_time(me.get("last_lap")), lap_time(me.get("best_lap"))
        fastest = min((r for r in board if r.get("best_lap")), key=lambda r: r["best_lap"], default=None)
        text, say = f"Last lap {last[0]}, best {best[0]}.", f"Last lap {last[1]}, best {best[1]}."
        if fastest and fastest["car"] != me["car"]:
            f = lap_time(fastest["best_lap"])
            text += f" Fastest is {name_of(fastest)}, {f[0]}."
            say += f" Fastest is {name_of(fastest)}, {f[1]}."
        elif fastest:
            text += " That's the fastest lap of anyone."
            say += " That's the fastest lap of anyone."
        return self._say("pace", text, say)

    def _damage(self, s, lock, me, board, car, info, laps):
        hurt = {k.replace("_", " "): v for k, v in (car.get("damage") or {}).items() if v}
        if not hurt:
            return self._say("damage", "Car looks good, no damage.")
        worst = sorted(hurt.items(), key=lambda kv: -kv[1])[:2]
        return self._say("damage", "Damage: " + ", ".join(f"{k} {v} percent" for k, v in worst) + ".")

    # -- radio calls without being asked -----------------------------------------------------------------
    def calls_for(self, s, lock, strategy=None):
        """New things to call out on the radio for this session, since the last time (kept per session)."""
        with self.lock, lock:
            seen = self.seen.setdefault(s.uid, {"events": len(s.events), "box": None, "last_lap": False, "rain": False})
            calls = self.calls.setdefault(s.uid, [])
            me = s.drivers.get(s.player, {}).get("code")
            for e in s.events[seen["events"]:]:              # events since the last check
                text = e["text"]
                if e["code"] == "SCAR":
                    if text.strip() in NEUTRAL_CALLS:            # "returned" adds nothing to "in this lap"
                        self._call(calls, NEUTRAL_CALLS[text.strip()])
                elif e["code"] == "FTLP" and me and f" {me} " in f"{text} ":
                    self._call(calls, "Fastest lap. Purple. Great job.")
                elif (e["code"] == "PENA" and e.get("car") == s.player       # real penalties, not warnings or laps
                      and e.get("penalty") in game_telemetry.REAL_PENALTIES):   # taken away (older events: no type)
                    secs = e.get("seconds")
                    self._call(calls, {4: f"{secs} second time penalty." if secs else "Time penalty.",
                                       6: "We've been disqualified."}.get(
                        e["penalty"], f"We've got a {game_telemetry.PENALTY_TYPES[e['penalty']].lower()}."))
                elif e["code"] == "RDFL":
                    self._call(calls, "Red flag, red flag. Into the pits.")
                elif e["code"] in RACE_CONTROL_2026:                 # F1 26: race control's active aero and overtake mode
                    reason = e.get("reason")
                    self._call(calls, RACE_CONTROL_2026[e["code"]] + (f" for the {reason}." if reason else "."))
                elif e["code"] == "COLL" and e.get("severity") == 2 and s.player in e.get("cars", ()):
                    self._call(calls, "That was a big hit. Check the car.")
                elif e["code"] == "CHQF":
                    pos = s.lap_now.get(s.player, {}).get("position")
                    self._call(calls, "Chequered flag." + (f" P{pos}. Good job." if pos else ""))
            seen["events"] = len(s.events)
            # Rain on the way (in the next 10 minutes, while it's dry): said once, and again after a dry spell
            soon, _ = rain_ahead(s.info, within=10)
            soon = soon if s.info.get("weather") not in game_strategy.RAIN else None
            if soon and not seen["rain"]:
                where = lap_at(s.info, s.player_laps(), s.lap_now.get(s.player, {}).get("lap"), soon["in"]) if s.is_race() else ""
                self._call(calls, f"Rain expected in {soon['in']} minutes{where}. Think about inters.")
            seen["rain"] = bool(soon)
            lap, total = s.lap_now.get(s.player, {}).get("lap"), s.info.get("total_laps")
            if s.is_race() and lap and total and lap == total and not seen["last_lap"]:
                seen["last_lap"] = True
                self._call(calls, "Last lap. Bring it home.")
        if strategy and strategy.get("available"):              # the model says box: once per stint
            d = next((x for x in strategy["drivers"] if x["player"]), None)
            if d:
                with self.lock:
                    key = (d["compound"], d["lap"] - int(d["tyre_age"] or 0))           # this set of tyres
                    if d["alert"] and seen["box"] != key:
                        seen["box"] = key
                        self._call(calls, f"Box, box. Box this lap for {next_tyre_name(d)}.")
        with self.lock:
            return calls[-10:]

    @staticmethod
    def _call(calls, text):
        # The id is the time in ms, so it keeps going up when the app restarts: an open page says only calls
        # newer than the last one it said
        number = max(int(time.time() * 1000), calls[-1]["id"] + 1 if calls else 0)
        calls.append({"id": number, "text": text, "say": text})


RADIO = Radio()
