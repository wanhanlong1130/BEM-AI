import pytest

from automa_ai.skills.config import SkillsConfig, SkillRegistryEntry


def test_from_dict_none_returns_default():
    config = SkillsConfig.from_dict(None)

    assert config.enabled is False
    assert config.allowed_roots is None
    assert config.registry == {}


def test_from_dict_string_entry():
    config = SkillsConfig.from_dict(
        {
            "enabled": True,
            "registry": {"write_sql": "/tmp/write_sql.md"},
        }
    )

    assert config.enabled is True
    assert isinstance(config.registry["write_sql"], SkillRegistryEntry)
    assert config.registry["write_sql"].path == "/tmp/write_sql.md"


def test_from_dict_invalid_entry_type_raises():
    with pytest.raises(TypeError):
        SkillsConfig.from_dict(
            {
                "enabled": True,
                "registry": {"bad": 123},
            }
        )
