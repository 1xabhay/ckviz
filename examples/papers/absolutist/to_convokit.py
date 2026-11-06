"""Figshare data of Al-Mosaiwi & Johnstone (2018) -> ConvoKit corpus (forum = conversation, member = speaker + one utterance).

Inputs (CC BY 4.0): doi:10.6084/m9.figshare.4743715 (one .txt per forum member, unzipped into <raw>/texts/) and
doi:10.6084/m9.figshare.4743547 ("Stats - In an Absolute State.xlsx" saved as <raw>/stats.xlsx), whose 'Original'
sheet maps each file to its forum and condition and carries the authors' LIWC word count and absolutist indices.

    uv run python -I examples/papers/absolutist/to_convokit.py data/papers/absolutist/raw data/papers/absolutist
"""
import sys
from pathlib import Path

import pandas as pd

from ckviz.frames import Frames, to_corpus

GROUPS = {1: "control", 2: "anxiety", 3: "depression", 4: "suicidal_ideation", 5: "ptsd", 6: "schizophrenia",
          7: "eating_disorder", 8: "bpd", 9: "recovery"}
STUDY = {1: 1, 2: 1, 3: 1, 4: 1, 5: 2, 6: 2, 7: 2, 8: 2, 9: 3}


def decode(b: bytes) -> str:
    # The files are a mix of Windows-1252 (most), UTF-8 and UTF-16 (BOM); line ends are often bare CR.
    if b[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return b.decode("utf-16")
    try:
        return b.decode("utf-8-sig")
    except UnicodeDecodeError:
        return b.decode("cp1252", errors="replace")


def main(raw: Path, out: Path):
    o = pd.read_excel(raw / "stats.xlsx", sheet_name="Original")
    o["Filename"] = o["Filename"].str.strip()  # some sheet names carry a leading no-break space
    missing = [n for n in o["Filename"] if not (raw / "texts" / n).exists()]
    print("not on Figshare, dropped:", missing)
    o = o[~o["Filename"].isin(missing)]
    group = o["Condition"].map(GROUPS)
    ids = "m" + o["N"].astype(str)
    convo = "forum" + o["Forum"].astype(str)
    texts = [decode((raw / "texts" / name).read_bytes()).strip() for name in o["Filename"]]
    utts = pd.DataFrame({
        "speaker": ids, "conversation_id": convo, "reply_to": None, "timestamp": 0, "text": texts,
        "meta.group": group, "meta.file": o["Filename"],
        # the authors' LIWC2015 output, kept to check our counts against: % of words
        "meta.liwc_wc": o["WC"], "meta.liwc_absolutist": (o["Absol"] - o["AbsolRev"]).round(4), "meta.liwc_fp": o["AbsolRev"],
    }).set_index(ids.rename("id"))
    speakers = pd.DataFrame({"meta.group": group.values}, index=ids.rename("id"))
    convos = (pd.DataFrame({"id": convo, "meta.group": group, "meta.study": o["Condition"].map(STUDY), "meta.forum": o["Forum"]})
              .drop_duplicates("id").set_index("id"))
    to_corpus(Frames(utts, speakers, convos)).dump("corpus", base_path=str(out))
    print(f"{len(utts)} members, {len(convos)} forums -> {out / 'corpus'}")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
