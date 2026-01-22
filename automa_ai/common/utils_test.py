import pytest
from unittest import mock
from types import SimpleNamespace

# Import your target functions / classes
from automa_ai.common.utils import load_memory_store_plugins
from automa_ai.memory.manager import MemoryStoreRegistry


# Dummy memory store for testing
class DummyMemoryStore:
    pass


def test_load_memory_store_plugins_registers_store(monkeypatch):
    # 1️⃣ Create a fake entry point
    fake_ep = SimpleNamespace(
        name="dummy_store",
        load=lambda: DummyMemoryStore  # load() should return the class
    )

    # 2️⃣ Patch importlib.metadata.entry_points to return our fake entry point
    with mock.patch("importlib.metadata.entry_points") as mock_eps:
        mock_eps.return_value = { "automa_ai.memory_stores": [fake_ep] }

        # Clear registry first
        MemoryStoreRegistry._stores.clear()

        # 3️⃣ Call the function
        load_memory_store_plugins()

        # 4️⃣ Assert it was registered
        registered_cls = MemoryStoreRegistry.get("dummy_store")
        assert registered_cls is DummyMemoryStore