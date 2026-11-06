"""A tiny invented daily-ping dataset as a ConvoKit zip: two participants chatting with a bot about fruit.

Two prompt ids, a message with status "moderated", a failed bot reply. v2=False leaves out the fields a leaner export
would lack (prompt_id, sent_at and the extra file). All text and ids are made up.
"""
import io
import json
import zipfile

SPEAKERS = {"u-aaa": {"type": "user"}, "bot:u-aaa": {"type": "bot"}, "u-bbb": {"type": "user"}, "bot:u-bbb": {"type": "bot"}}

T0 = 1_700_000_000  # epoch seconds; day 1 starts here
DAY = 86_400


def _utts(v2: bool) -> list[dict]:
    rows = []

    def add(uid, user, day, sec, text, reply_to, conv, status, bot, extra=None, sp="sp1"):
        meta = {"day_number": day, "local_time": f"2024-03-0{day}T09:{sec // 60:02d}:{sec % 60:02d}", "status": status}
        if bot and v2:
            meta.update({"prompt_id": sp, "sent_at": T0 + (day - 1) * DAY + sec + 4})
        meta.update(extra or {})
        rows.append({"id": uid, "speaker": ("bot:" if bot else "") + user, "conversation_id": conv,
                     "reply_to": reply_to, "timestamp": T0 + (day - 1) * DAY + sec, "text": text, "meta": meta})

    # participant A: days 1-3, prompt sp1 then sp2 on day 3
    add("a1", "u-aaa", 1, 0, "Hello! What did you eat for lunch?", None, "a1", "sent", True, {"is_ping": True})
    add("a2", "u-aaa", 1, 30, "a pear and some toast today", "a1", "a1", "received", False)
    add("a3", "u-aaa", 1, 35, "Nice choice. Was the pear ripe yet?", "a2", "a1", "sent", True)
    add("a4", "u-aaa", 2, 10, "it was crunchy", "a3", "a1", "received", False)
    add("a5", "u-aaa", 2, 14, "Would you pick the same fruit again tomorrow?", "a4", "a1", "sent", True)
    add("a6", "u-aaa", 3, 50, "maybe a plum instead, thanks", "a5", "a1", "received", False)
    add("a7", "u-aaa", 3, 55, "Plums are great. Enjoy it!", "a6", "a1", "sent", True, sp="sp2")
    # participant B: days 1 and 3; a moderated message, a failed bot reply
    add("b1", "u-bbb", 1, 0, "Hi there! Any fruit today then?", None, "b1", "sent", True, {"is_ping": True})
    add("b2", "u-bbb", 1, 120, "some other text", "b1", "b1", "moderated", False)
    add("b3", "u-bbb", 1, 125, "Let's stay on fruit here.", "b2", "b1", "sent", True)
    add("b4", "u-bbb", 3, 300, "ok sure. kiwi was nice", "b3", "b1", "received", False)
    add("b5", "u-bbb", 3, 302, "", "b4", "b1", "failed", True, {"sent_at": None} if v2 else None)
    return rows


CONVOS = {"a1": {"owner_id": "u-aaa", "site": "c-aaa"}, "b1": {"owner_id": "u-bbb", "site": "c-bbb"}}

EXTRA = {"label": "demo"}


def export_zip(v2: bool = True, name: str = "ping_corpus_test") -> bytes:
    """The export as zip bytes, with one top-level corpus folder."""
    files = {
        "utterances.jsonl": "\n".join(json.dumps(u) for u in _utts(v2)) + "\n",
        "speakers.json": json.dumps(SPEAKERS),
        "conversations.json": json.dumps(CONVOS),
        "corpus.json": json.dumps({"export_version": 2 if v2 else 1}),
        "index.json": json.dumps({"utterances-index": {}, "speakers-index": {}, "conversations-index": {},
                                  "overall-index": {}, "version": 1}),
    }
    if v2:
        files["extra.json"] = json.dumps(EXTRA)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for fn, content in files.items():
            z.writestr(f"{name}/{fn}", content)
    return buf.getvalue()
