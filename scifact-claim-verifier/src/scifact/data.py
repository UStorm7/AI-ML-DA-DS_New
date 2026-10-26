"""SciFact (Wadden et al., EMNLP 2020): 1.4k expert-written scientific claims
verified against a corpus of 5,183 research-paper abstracts."""
import json
import tarfile
from dataclasses import dataclass, field
from pathlib import Path
from urllib.request import urlretrieve

URL = "https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz"
DATA_DIR = Path(__file__).resolve().parents[2] / "data"
LABELS = ["SUPPORT", "CONTRADICT", "NEI"]


@dataclass
class Doc:
    doc_id: int
    title: str
    sentences: list[str]

    @property
    def text(self) -> str:
        return self.title + " " + " ".join(self.sentences)


@dataclass
class Claim:
    id: int
    text: str
    # doc_id -> {"label": SUPPORT|CONTRADICT, "sentences": [rationale sentence indices]}
    evidence: dict[int, dict] = field(default_factory=dict)
    cited_doc_ids: list[int] = field(default_factory=list)

    @property
    def label(self) -> str:
        """Claim-level label. (No SciFact claim has both SUPPORT and CONTRADICT evidence.)"""
        if not self.evidence:
            return "NEI"
        return next(iter(self.evidence.values()))["label"]


def download(data_dir: Path = DATA_DIR) -> Path:
    root = data_dir / "data"
    if not (root / "corpus.jsonl").exists():
        data_dir.mkdir(parents=True, exist_ok=True)
        archive = data_dir / "scifact.tar.gz"
        urlretrieve(URL, archive)
        with tarfile.open(archive) as tar:
            tar.extractall(data_dir, filter="data")
        archive.unlink()
    return root


def _read_jsonl(path: Path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def load_corpus(data_dir: Path = DATA_DIR) -> dict[int, Doc]:
    root = download(data_dir)
    return {d["doc_id"]: Doc(d["doc_id"], d["title"], d["abstract"])
            for d in _read_jsonl(root / "corpus.jsonl")}


def load_claims(split: str, data_dir: Path = DATA_DIR) -> list[Claim]:
    """split: 'train' (809 claims) or 'dev' (300). The official test labels are
    hidden, so 'dev' is used as the held-out evaluation set."""
    root = download(data_dir)
    claims = []
    for c in _read_jsonl(root / f"claims_{split}.jsonl"):
        evidence = {
            int(doc_id): {"label": sets[0]["label"],
                          "sentences": sorted({i for s in sets for i in s["sentences"]})}
            for doc_id, sets in c.get("evidence", {}).items()
        }
        claims.append(Claim(c["id"], c["claim"], evidence, c.get("cited_doc_ids", [])))
    return claims
