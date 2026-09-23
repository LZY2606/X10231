import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from adcp.casepack import replay  # noqa: E402
from adcp.storage import Store  # noqa: E402


@pytest.fixture()
def tmp_path_db(tmp_path):
    return str(tmp_path / "test.db")


@pytest.fixture(scope="module")
def replayed():
    """A store with the full case-pack narrative replayed once."""
    path = tempfile.mktemp(suffix=".db")
    store = Store(path)
    versions = replay(store)
    yield store, versions
    store.close()
    for suffix in ("", "-wal", "-shm"):
        if os.path.exists(path + suffix):
            os.remove(path + suffix)
