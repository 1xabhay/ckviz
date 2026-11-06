#!/usr/bin/env python3
"""Generate examples/data/synthetic: an invented ConvoKit corpus of everyday chat. Standard library only.

    python examples/data/make_synthetic.py [out_dir]     # default: examples/data/synthetic

19 fictional users each have one conversation with a neutral assistant. Each user has an integer score in the `phq9`
meta key. The key is named that only so the example pipelines read like real analyses: the score is an arbitrary
number assigned here, measures nothing, and has no clinical meaning.

What is planted: per message, P(garden clause) rises with the score and P(bicycle clause) falls with it. Every garden
clause contains "compost" and every bicycle clause contains "bicycle", so those two words should top fighting_words
and correlate with the score. Gardening and cycling were picked because they have nothing to do with mood or health.
Everything else in a message is drawn independently of the score.
"""
import json
import random
import sys
from pathlib import Path

SEED = 20261105
N_UTTERANCES, N_USERS = 1543, 19
START = 1_700_000_000
ASSISTANT = "assistant"

GARDEN = [
    "Also I spread some compost along the fence.",
    "The compost heap was steaming again this morning.",
    "I put mulch and fresh compost around the seedlings.",
    "A neighbour lent me a trowel to shift the compost.",
    "Turning the compost took longer than it should have.",
]
BICYCLE = [
    "I fixed a puncture on my bicycle before lunch.",
    "My bicycle needs a new saddle soon.",
    "The bicycle chain slipped twice on the hill.",
    "I rode the bicycle to the shop with a full pannier.",
    "Someone borrowed my bicycle and bent the handlebars.",
]

WHEN = ["yesterday", "this morning", "last night", "on Saturday", "after work", "earlier today", "at the weekend"]
WHO = ["my cousin", "a friend", "my flatmate", "the neighbours", "my brother", "an old colleague", "my aunt"]

# topic -> (what the user says, what the assistant asks)
TOPICS = {
    "cooking": (
        ["I made a {thing} {when} and it came out better than I expected.",
         "{who} showed me an easier way to make a {thing} and I wrote it down.",
         "I burned the first {thing} {when} so I started over with less heat.",
         "We shared a {thing} with {who} and there was nothing left over."],
        ["What would you change next time you cook it?", "Who taught you to cook that?", "How long did it take to make?"],
        ["lentil soup", "vegetable curry", "lemon cake", "potato pie", "pan of noodles"],
    ),
    "commute": (
        ["The {thing} was late {when} so I read a few pages while waiting.",
         "I ran into {who} on the {thing} and we talked the whole way.",
         "They changed the {thing} timetable and now I leave ten minutes earlier.",
         "The {thing} was so full {when} that I stood the entire trip."],
        ["How long is the trip usually?", "Is there another route you could take?", "What do you do to pass the time?"],
        ["bus", "tram", "train", "ferry", "shuttle"],
    ),
    "repairs": (
        ["I finally mended the {thing} {when} with a cheap part from the hardware store.",
         "The {thing} started rattling again so I asked {who} to take a look.",
         "It took two tries to fix the {thing} but it works fine now.",
         "I watched a short video about the {thing} and then did the job myself."],
        ["Was the repair harder than you thought?", "Did you need any special tools?", "How long had it been broken?"],
        ["kitchen tap", "garage door", "washing machine", "bathroom shelf", "toaster"],
    ),
    "pets": (
        ["Our {thing} knocked a glass off the table {when} and looked quite pleased about it.",
         "I took the {thing} to the park and {who} came along too.",
         "The {thing} has decided that the laundry basket is the best place to nap.",
         "We got a new bed for the {thing} and it still sleeps on the sofa."],
        ["How old is your pet now?", "What is its favourite thing to do?", "Does it get along with visitors?"],
        ["cat", "dog", "rabbit", "parrot", "tortoise"],
    ),
    "books": (
        ["I started a {thing} {when} and could not put it down.",
         "{who} recommended a {thing} and the ending surprised me.",
         "I gave up halfway through a {thing} because the plot went nowhere.",
         "Our reading group picked a {thing} and we argued about it for an hour."],
        ["What kept you reading?", "Would you recommend it to someone else?", "What are you reading next?"],
        ["mystery novel", "travel memoir", "history of maps", "science fiction story", "short story collection"],
    ),
    "weather": (
        ["It rained {when} so I stayed in and looked for the {thing}.",
         "The wind took the {thing} right off the balcony {when}.",
         "It was warm enough {when} to leave the {thing} outside.",
         "I went out without the {thing} and of course the clouds rolled in."],
        ["What is the forecast for the rest of the week?", "Do you prefer it warmer or cooler?", "Did it change your plans?"],
        ["umbrella", "folding chairs", "sun hat", "old blanket", "raincoat"],
    ),
    "shopping": (
        ["I went looking for a {thing} {when} and came home with three other things.",
         "The market had a good {thing} at half the usual price.",
         "I returned the {thing} because it was the wrong size.",
         "{who} talked me out of buying an expensive {thing}."],
        ["Did you find what you wanted?", "Do you usually shop there?", "Was it worth the price?"],
        ["desk lamp", "rain jacket", "set of pans", "wool scarf", "bookcase"],
    ),
    "weekend": (
        ["We went to the {thing} {when} and stayed much longer than planned.",
         "I am going to the {thing} with {who} next week.",
         "The {thing} was closed when we got there so we had coffee instead.",
         "I had not been to the {thing} in years and it has hardly changed."],
        ["What was the best part?", "Would you go back?", "Who else went along?"],
        ["museum", "flea market", "lake", "botanical gardens", "town fair"],
    ),
}
ACK = ["I see.", "Thanks for telling me.", "That makes sense.", "Okay.", "Got it.", ""]

PERSONAS = ["lighthouse keeper", "puppet maker", "clock repairer", "map illustrator", "ferry pilot", "beekeeper",
            "piano tuner", "kite designer", "cheese taster", "stage carpenter", "sign painter", "tile layer",
            "hat maker", "bell ringer", "glass blower", "weather observer", "bookbinder", "canal lock keeper", "zookeeper"]


def build(seed: int = SEED) -> dict[str, str]:
    rng = random.Random(seed)
    # ponytail: only rng.random() is used, the one method Python promises is stable across versions.
    pick = lambda xs: xs[int(rng.random() * len(xs))]
    shuffled = lambda xs: sorted(xs, key=lambda _: rng.random())

    scores = shuffled([i * 3 // 2 for i in range(N_USERS)])  # 19 distinct integers across 0..27
    personas = shuffled(PERSONAS)
    speakers: dict = {ASSISTANT: {"role": "ai"}}
    conversations: dict = {}
    lines: list[str] = []
    per_convo, extra = divmod(N_UTTERANCES, N_USERS)

    for c in range(N_USERS):
        user = f"user{c + 1:02d}"
        score = scores[c]
        profile = {"age": 19 + int(rng.random() * 53), "gender": pick(["woman", "man", "non-binary person"]),
                   "persona": f"Fictional character: a {personas[c]}.", "phq9": score}
        speakers[user] = {"role": "person", **profile}
        n = per_convo + (c < extra)
        conversation_id = f"c{c + 1:02d}-000"
        conversations[conversation_id] = {"user_id": user, **profile, "n_utterances": n}

        p_garden = 0.05 + 0.40 * score / 27
        p_bicycle = 0.45 - 0.40 * score / 27
        clock, previous, topic = START + c * 86_400, None, pick(sorted(TOPICS))
        for t in range(n):
            if t % 2 == 0:  # assistant turn; moves to a new topic now and then
                if t and rng.random() < 0.2:
                    topic = pick(sorted(TOPICS))
                question = pick(TOPICS[topic][1])
                text = f"Hello, how has your week been? {question}" if t == 0 else f"{pick(ACK)} {question}".strip()
                speaker, role = ASSISTANT, "ai"
            else:
                frames, _, things = TOPICS[topic]
                said = pick(frames).format(thing=pick(things), when=pick(WHEN), who=pick(WHO))
                parts = [said[0].upper() + said[1:]]
                # Two draws, always made, so a user's stream of randomness doesn't depend on which clause fired.
                g, b = rng.random(), rng.random()
                garden, bicycle = pick(GARDEN), pick(BICYCLE)
                parts += [garden] if g < p_garden else []
                parts += [bicycle] if b < p_bicycle else []
                text, speaker, role = " ".join(parts), user, "person"
            utterance_id = f"c{c + 1:02d}-{t:03d}"
            lines.append(json.dumps({"id": utterance_id, "conversation_id": conversation_id, "speaker": speaker,
                                     "reply_to": previous, "timestamp": clock, "text": text, "meta": {"role": role}},
                                    separators=(",", ":")))
            previous = utterance_id
            clock += 20 + int(rng.random() * 120)

    corpus = {"name": "ckviz_synthetic_chat",
              "description": "Invented everyday chat between a neutral assistant and fictional users, generated by "
                             "examples/data/make_synthetic.py. The phq9 key is an arbitrary number with no clinical meaning.",
              "num_conversations": N_USERS, "num_utterances": len(lines), "num_speakers": len(speakers),
              "seed": seed, "synthetic": True}
    types = lambda d: {k: str(type(v)) for k, v in d.items()}
    index = {"utterances-index": {"role": str(str)},
             "speakers-index": {"role": str(str), **types(profile)},
             "conversations-index": types(next(iter(conversations.values()))),
             "overall-index": types(corpus), "version": 1}
    dump = lambda d: json.dumps(d, indent=2) + "\n"
    return {"utterances.jsonl": "\n".join(lines) + "\n", "speakers.json": dump(speakers),
            "conversations.json": dump(conversations), "index.json": dump(index), "corpus.json": dump(corpus)}


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "synthetic"
    out.mkdir(parents=True, exist_ok=True)
    for name, body in build().items():
        (out / name).write_bytes(body.encode())
