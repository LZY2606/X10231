import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from adcp_evidence.fingerprint import content_id
from adcp_evidence.model import load_case_pack
from adcp_evidence.service import CASE_DIR, materials_to_sources


@pytest.fixture(scope="session")
def case_dir():
    return CASE_DIR


@pytest.fixture(scope="session")
def materials(case_dir):
    return load_case_pack(case_dir)


@pytest.fixture(scope="session")
def baseline_sources(materials):
    return materials_to_sources(materials)
