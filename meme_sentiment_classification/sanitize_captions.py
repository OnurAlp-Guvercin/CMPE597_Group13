import re

SANITIZE_REPLACEMENTS = {
    # narrator framing
    "meme poster is ": "",
    "the meme poster is ": "",
    "the poster is ": "",
    "poster is ": "",
    "meme creator is ": "",
    "the meme creator is ": "",
    "the person is ": "",
    "the person who wrote the post is ": "",
    "the person who made the meme is ": "",
    "the author is ": "",
    "the user is ": "",

    # reporting / explanation
    "is saying that ": "",
    "saying that ": "",
    "is saying ": "",
    "saying ": "",

    "is trying to say that ": "",
    "trying to say that ": "",
    "trying to say ": "",

    "is trying to convey that ": "",
    "trying to convey that ": "",
    "trying to convey ": "",

    "is trying to explain that ": "",
    "trying to explain that ": "",
    "trying to explain ": "",

    "is trying to show that ": "",
    "trying to show that ": "",
    "trying to show ": "",

    "is trying to express that ": "",
    "trying to express that ": "",
    "trying to express ": "",

    "is conveying the message that ": "",
    "conveying the message that ": "",

    "is conveying that ": "",
    "conveying that ": "",

    "is explaining that ": "",
    "explaining that ": "",

    "is expressing that ": "",
    "expressing that ": "",

    "is showing that ": "",
    "showing that ": "",

    "is describing how ": "",
    "describing how ": "",

    "is talking about ": "",
    "talking about ": "",

    "is commenting on ": "",
    "commenting on ": "",

    "is pointing out that ": "",
    "pointing out that ": "",

    "is highlighting that ": "",
    "highlighting that ": "",

    "is mentioning that ": "",
    "mentioning that ": "",

    "is noting that ": "",
    "noting that ": "",

    "is stating that ": "",
    "stating that ": "",

    "is implying that ": "",
    "implying that ": "",

    # weak cognition verbs
    "feels that ": "",
    "thinks that ": "",
    "believes that ": "",
    "realizes that ": "",
    "notices that ": "",
    "understands that ": "",
    "recognizes that ": "",

    # meme-style weak emotional phrases
    "is making fun of ": "mocks ",
    "makes fun of ": "mocks ",

    "is joking about ": "jokes about ",
    "jokes about ": "jokes about ",

    "is mocking ": "mocks ",
    "mocking ": "mocks ",

    "is criticizing ": "hates ",
    "criticizing ": "hates ",

    "is annoyed by ": "hates ",
    "annoyed by ": "hates ",

    "is annoyed at ": "hates ",
    "annoyed at ": "hates ",

    "is angry about ": "hates ",
    "angry about ": "hates ",

    "is frustrated with ": "hates ",
    "frustrated with ": "hates ",

    "is sad about ": "sad about ",
    "sad about ": "sad about ",

    "is worried about ": "fears ",
    "worried about ": "fears ",

    "is concerned about ": "fears ",
    "concerned about ": "fears ",

    "is happy about ": "loves ",
    "happy about ": "loves ",

    "is excited about ": "loves ",
    "excited about ": "loves ",

    # low-information wrappers
    "this meme shows ": "",
    "the meme shows ": "",
    "this meme means ": "",
    "the meme means ": "",
    "this shows ": "",
    "this means ": "",
    "this represents ": "",

    # hedging
    "kind of ": "",
    "sort of ": "",
    "somewhat ": "",
    "apparently ": "",
    "basically ": "",
    "simply ": "",
    "just ": "",
}


def sanitize_caption(text: str) -> str:
    if not text:
        return text

    text = text.lower()

    # apply all replacements globally
    for old, new in SANITIZE_REPLACEMENTS.items():
        text = text.replace(old, new)

    # collapse repeated whitespace
    text = re.sub(r"\s+", " ", text)

    # remove repeated punctuation
    text = re.sub(r"([.!?]){2,}", r"\1", text)

    # remove leading punctuation leftovers
    text = re.sub(r"^[,.\-:; ]+", "", text)

    # remove trailing punctuation leftovers
    text = re.sub(r"[,.\-:; ]+$", "", text)

    return text.strip()