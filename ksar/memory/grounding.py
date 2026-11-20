"""
Grounding: is a proposed memory fact really supported by what the owner said?

Three steps, cheapest first, all deterministic until the last one:
  1. the quoted evidence must exist in the owner's own words (the journal is the source of truth)
  2. concept match: words are compared by meaning groups ('like', 'love', 'prefer' are one concept; 'delete', 'erase',
     'remove' another), numbers and names must match exactly, and negation must agree ('never delete' != 'may delete')
  3. only for borderline facts, ONE short yes/no question to the background model about that exact quote
Anything still unclear is kept PENDING (re-checked later against newer conversation) instead of being guessed.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .store import strip_tags

_GROUPS = [
    "like love enjoy prefer prefers fond favorite favourite fan адор люб нрав предпочит",
    "dislike hate avoid despise detest ненавиж избег",
    "never forbid prohibit ban banned forbidden prohibited нельзя запрещ никогда",
    "always must need needs require required should обязат всегда нужно должен",
    "allow allowed permit permitted permission may разреш можно",
    "delete deleted remove removed erase wipe trash destroy удал сотр стер",
    "modify change edit overwrite alter rewrite replace измен перепис редакт",
    "ask asking confirm confirmation approval спрос подтвер",
    "short brief concise direct terse shortly кратк коротк",
    "long detailed verbose thorough подробн длинн",
    "want wish plan goal aim intend hope dream хоч план цел мечт",
    "study studies university college degree school master bachelor учит универс вуз",
    "code coding program programming script software код програм",
    "answer answers reply replies respond response ответ отвеч",
    "project app application product проект приложен",
    "buy purchase order pay купи заказ оплат",
    "cost price budget euro dollar eur usd money бюджет цен стоим",
    "language english german russian spanish french язык англи немец русск",
    "work job career employment profession работ карьер профес",
    "live lives living stay reside жив прожив",
    "use uses using utilize run runs running использ запуск",
]
_CONCEPT: dict = {}
for i, g in enumerate(_GROUPS):
    for w in g.split():
        _CONCEPT[w[:5]] = i

# Only words that flip the MEANING of a claim. ('without asking' is a manner, not a negation.)
NEGATORS = {"no", "not", "never", "dont", "don't", "doesn't", "doesnt", "cannot", "can't", "cant", "won't", "wont",
            "нельзя", "не", "никогда", "нет"}
GENERIC = set("""user owner prefers prefer wants want has have is are the a an of to in on for with and or that this
it as at by be been will would should could can may might i me my you your we our they them their his her its
currently now temporarily probably possibly likely""".split())
_WORD = re.compile(r"[^\W_]+", re.U)


def words(text: str) -> list:
    return [w for w in _WORD.findall(str(text).lower())]


def concepts(text: str, drop_generic: bool = True) -> list:
    out = []
    for w in words(strip_tags(text)):
        if drop_generic and (w in GENERIC or w in NEGATORS):
            continue
        if len(w) < 2:
            continue
        key = w[:5]
        out.append(("c", _CONCEPT[key]) if key in _CONCEPT else ("w", key if not w.isdigit() else w))
    return out


def negative(text: str) -> bool:
    ws = words(text)
    n = sum(1 for w in ws if w in NEGATORS)
    return n % 2 == 1


def normalize_loose(text: str) -> str:
    return " ".join(words(text))


def evidence_in(evidence: str, source: str, need: float = 0.85) -> bool:
    """The quoted words must appear in the source (substring, or nearly all word stems in order-free overlap)."""
    e, s = normalize_loose(evidence), normalize_loose(source)
    if len(e) < 3:
        return False
    if e in s:
        return True
    et, st = {w[:5] for w in e.split()}, {w[:5] for w in s.split()}
    return bool(et) and len(et & st) / len(et) >= need


@dataclass
class Support:
    label: str                 # yes | maybe | no
    score: float = 0.0
    reason: str = ""


def _support_one(fact: str, source: str) -> Support:
    value = fact.split(":", 1)[1] if ":" in fact else fact
    fc = concepts(value)
    if not fc:
        return Support("no", 0.0, "nothing to check")
    sc = set(concepts(source, drop_generic=False))
    covered = sum(1 for c in fc if c in sc)
    score = covered / len(fc)
    nums = [w for w in words(value) if w.isdigit() and len(w) >= 2]
    if any(n not in words(source) for n in nums):
        return Support("no", score * 0.3, "a number in the fact is not in what the owner said")
    if negative(value) != negative(source) and (score >= 0.4):
        return Support("no", score, "the fact and the owner's words disagree (negation)")
    if score >= 0.7:
        return Support("yes", score, f"{covered}/{len(fc)} concepts found")
    if score >= 0.4:
        return Support("maybe", score, f"only {covered}/{len(fc)} concepts found")
    return Support("no", score, f"only {covered}/{len(fc)} concepts found")


_RANK = {"yes": 2, "maybe": 1, "no": 0}


def support(fact: str, source: str) -> Support:
    """How well `source` (the owner's words) supports `fact`, by meaning groups, numbers and negation.
    A long source is judged sentence by sentence (negation in one sentence must not reject a fact from another)."""
    parts = [p for p in re.split(r"[.!?;\n]+", source) if len(p.strip()) > 2]
    if len(parts) <= 1:
        return _support_one(fact, source)
    best = None
    for part in parts + [source]:
        r = _support_one(fact, part)
        if best is None or (_RANK[r.label], r.score) > (_RANK[best.label], best.score):
            best = r
    return best


def ground(fact: str, evidence: str, source: str) -> Support:
    """Full check of one fact. `source` = everything the owner wrote in the processed turns."""
    if evidence:
        if not evidence_in(evidence, source):
            return Support("no", 0.0, "the quoted evidence is not in the owner's messages")
        best = support(fact, evidence)
        if best.label != "yes":
            alt = support(fact, source)
            if alt.score > best.score:
                best = alt
        return best
    return support(fact, source)


def verify_prompt(fact: str, evidence: str) -> str:
    return ("Decide if the OWNER'S WORDS support the CLAIM. Answer with JSON only: "
            '{"answer": "yes" | "no" | "unsure"}. "yes" = the words state it or clearly imply it; '
            '"no" = they do not or contradict it; "unsure" = cannot tell. Do not guess.\n\n'
            f"OWNER'S WORDS: {evidence[:600]}\nCLAIM: {fact[:300]}")


def parse_verify(raw: str) -> str:
    from ..llm import extract_json
    d = extract_json(raw)
    a = str(d.get("answer", "")).lower() if isinstance(d, dict) else ""
    return a if a in ("yes", "no", "unsure") else "unsure"
