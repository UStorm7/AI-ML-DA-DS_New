import re

STOPWORDS = frozenset("""
a an and are as at be been but by can could did do does for from had has have how i if in into is it
its may might more most no nor not of on or our s such than that the their them then there these they
this those to was we were what when where which while who will with would you
""".split())
# "no" / "not" are stopwords for retrieval but matter a lot for verification.
NEGATIONS = frozenset("no not none never neither nor without lack lacks absence absent fail fails "
                      "failed unable cannot reduced decrease decreased decreases lower inhibit inhibits "
                      "inhibited suppress suppresses prevent prevents".split())
_TOKEN = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*")


def _stem(tok: str) -> str:
    # Tiny suffix stripper: enough to merge plural / verb forms for BM25
    # without pulling in NLTK.
    for suf in ("ations", "ation", "ings", "ing", "ies", "es", "ed", "s"):
        if tok.endswith(suf) and len(tok) - len(suf) >= 3:
            return tok[: -len(suf)] + ("y" if suf == "ies" else "")
    return tok


def tokenize(text: str, stem: bool = True, keep_stopwords: bool = False) -> list[str]:
    toks = _TOKEN.findall(text.lower())
    if not keep_stopwords:
        toks = [t for t in toks if t not in STOPWORDS]
    return [_stem(t) for t in toks] if stem else toks


NUMBER = re.compile(r"\d+(?:\.\d+)?")
