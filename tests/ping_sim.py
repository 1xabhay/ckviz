"""Seeded synthetic daily-ping data as Frames (no zip): N participants × D days of daily pings with known engagement.

Each profile is (p_reply, last_day): the participant replies to day d's ping with probability p_reply while d <= last_day,
and never after. A reply is 1-3 messages, each answered by the bot. Returns (frames, truth) where
truth[participant] = sorted days replied to.
"""
import numpy as np
import pandas as pd

from ckviz.frames import Frames

T0 = 1_700_000_000
DAY = 86_400
DAYS = 10


def simulate(profiles: list[tuple[float, int]], days: int = DAYS, seed: int = 0) -> tuple[Frames, dict[str, list[int]]]:
    rng = np.random.default_rng(seed)
    rows, truth, speakers, convos = [], {}, [], []
    for i, (p, last) in enumerate(profiles):
        pid, bot, conv = f"p{i:02d}", f"bot:p{i:02d}", f"c{i:02d}"
        speakers += [(pid, "user"), (bot, "bot")]
        convos.append((conv, pid))
        truth[pid], prev = [], None
        for d in range(1, days + 1):
            t = T0 + (d - 1) * DAY + 9 * 3600 + i * 60
            ping = f"{conv}-d{d}"
            rows.append(dict(id=ping, speaker=bot, conversation_id=conv, reply_to=prev, timestamp=t, text=f"day {d}: which fruit today?",
                             day=d, is_ping=True, status="sent"))
            prev = ping
            if d <= last and rng.random() < p:
                truth[pid].append(d)
                for j in range(int(rng.integers(1, 4))):
                    t += int(rng.integers(60, 3600))
                    rows.append(dict(id=f"{ping}-u{j}", speaker=pid, conversation_id=conv, reply_to=prev, timestamp=t,
                                     text=" ".join(["word"] * int(rng.integers(1, 12))), day=d, is_ping=None, status="received"))
                    rows.append(dict(id=f"{ping}-b{j}", speaker=bot, conversation_id=conv, reply_to=f"{ping}-u{j}", timestamp=t + 5,
                                     text="noted, thanks", day=d, is_ping=None, status="sent"))
                    prev = f"{ping}-b{j}"
    u = pd.DataFrame(rows).set_index("id").rename(columns={"day": "meta.day_number", "is_ping": "meta.is_ping", "status": "meta.status"})
    u["meta.local_time"] = pd.to_datetime(u["timestamp"], unit="s", utc=True).dt.strftime("%Y-%m-%dT%H:%M:%S+00:00")
    sp = pd.DataFrame(speakers, columns=["id", "meta.type"]).set_index("id")
    cv = pd.DataFrame(convos, columns=["id", "meta.owner_id"]).set_index("id")
    return Frames(u, sp, cv), truth
