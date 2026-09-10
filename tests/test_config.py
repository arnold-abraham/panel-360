import pytest

from panel360.config import Panel360Config, ensure_output_dir, load_config, require_dir

INI_CONTENT = """
[PATHS]
binary_folder = data/binary
actual_next24_binary_folder = data/actual
test_binary_folder = data/test
output_folder = out

[FORECAST]
input_days = 7
"""


def write_ini(tmp_path, content=INI_CONTENT):
    path = tmp_path / "config.ini"
    path.write_text(content)
    return str(path)


def test_load_config_reads_values(tmp_path):
    config = load_config(write_ini(tmp_path))

    assert config == Panel360Config(
        binary_folder="data/binary",
        actual_next24_binary_folder="data/actual",
        test_binary_folder="data/test",
        output_folder="out",
        input_days=7,
    )


def test_load_config_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(str(tmp_path / "does_not_exist.ini"))


def test_load_config_env_override(tmp_path, monkeypatch):
    monkeypatch.setenv("PANEL360_OUTPUT_FOLDER", "/override/output")
    monkeypatch.setenv("PANEL360_INPUT_DAYS", "3")

    config = load_config(write_ini(tmp_path))

    assert config.output_folder == "/override/output"
    assert config.input_days == 3
    # Fields not overridden still come from the file.
    assert config.binary_folder == "data/binary"


def test_load_config_influx_disabled_by_default(tmp_path):
    config = load_config(write_ini(tmp_path))

    assert config.influx_enabled is False


def test_load_config_influx_enabled_when_all_env_vars_set(tmp_path, monkeypatch):
    monkeypatch.setenv("PANEL360_INFLUX_URL", "http://localhost:8086")
    monkeypatch.setenv("PANEL360_INFLUX_TOKEN", "secret-token")
    monkeypatch.setenv("PANEL360_INFLUX_ORG", "panel360")
    monkeypatch.setenv("PANEL360_INFLUX_BUCKET", "panel-status")

    config = load_config(write_ini(tmp_path))

    assert config.influx_enabled is True
    assert config.influx_url == "http://localhost:8086"
    assert config.influx_bucket == "panel-status"


def test_load_config_influx_disabled_when_partially_set(tmp_path, monkeypatch):
    monkeypatch.setenv("PANEL360_INFLUX_URL", "http://localhost:8086")
    monkeypatch.setenv("PANEL360_INFLUX_TOKEN", "secret-token")

    config = load_config(write_ini(tmp_path))

    assert config.influx_enabled is False


def test_require_dir_raises_for_missing_path(tmp_path):
    with pytest.raises(FileNotFoundError):
        require_dir(str(tmp_path / "missing"), "some folder")


def test_require_dir_returns_path_for_existing_dir(tmp_path):
    assert require_dir(str(tmp_path), "some folder") == str(tmp_path)


def test_ensure_output_dir_creates_missing_directory(tmp_path):
    target = tmp_path / "nested" / "output"
    result = ensure_output_dir(str(target))
    assert result == str(target)
    assert target.is_dir()
