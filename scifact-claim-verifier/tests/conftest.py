import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scifact.data import Claim, Doc  # noqa: E402


@pytest.fixture
def corpus():
    return {
        1: Doc(1, "Aspirin and heart attack", ["Aspirin reduces the risk of heart attack.",
                                                "The trial enrolled 5000 adults."]),
        2: Doc(2, "Vitamin C and colds", ["Vitamin C does not prevent the common cold.",
                                           "Effects on duration were small."]),
        3: Doc(3, "Exercise and sleep", ["Regular exercise improves sleep quality in older adults."]),
    }


@pytest.fixture
def claims():
    return [
        Claim(10, "Aspirin lowers heart attack risk.", {1: {"label": "SUPPORT", "sentences": [0]}}, [1]),
        Claim(11, "Vitamin C prevents colds.", {2: {"label": "CONTRADICT", "sentences": [0]}}, [2]),
        Claim(12, "Exercise causes insomnia in children.", {}, [3]),
    ]
