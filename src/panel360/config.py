"""Configuration loading for the Panel360 pipelines.

Values come from an INI file (default: ``config/config.ini``) and can be
overridden per-field with environment variables, which is what makes the
same config usable unchanged both locally and inside Docker (bind-mounted
data directories, etc.).
"""

import configparser
import os
from dataclasses import dataclass

DEFAULT_CONFIG_PATH = "config/config.ini"

_ENV_OVERRIDES = {
    "binary_folder": "PANEL360_BINARY_FOLDER",
    "actual_next24_binary_folder": "PANEL360_ACTUAL_NEXT24_BINARY_FOLDER",
    "test_binary_folder": "PANEL360_TEST_BINARY_FOLDER",
    "output_folder": "PANEL360_OUTPUT_FOLDER",
    "input_days": "PANEL360_INPUT_DAYS",
}

# Optional InfluxDB settings: env-var only, no config.ini equivalent — unset means disabled.
_INFLUX_ENV_VARS = {
    "influx_url": "PANEL360_INFLUX_URL",
    "influx_token": "PANEL360_INFLUX_TOKEN",
    "influx_org": "PANEL360_INFLUX_ORG",
    "influx_bucket": "PANEL360_INFLUX_BUCKET",
}


@dataclass
class Panel360Config:
    binary_folder: str
    actual_next24_binary_folder: str
    test_binary_folder: str
    output_folder: str
    input_days: int
    influx_url: str = None
    influx_token: str = None
    influx_org: str = None
    influx_bucket: str = None

    @property
    def influx_enabled(self):
        return all((self.influx_url, self.influx_token, self.influx_org, self.influx_bucket))


def load_config(config_path=None):
    """Load config from ``config_path`` (or ``$PANEL360_CONFIG_PATH``/default), applying env overrides."""
    config_path = config_path or os.environ.get("PANEL360_CONFIG_PATH", DEFAULT_CONFIG_PATH)

    parser = configparser.ConfigParser()
    if not parser.read(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    values = {
        "binary_folder": parser["PATHS"]["binary_folder"],
        "actual_next24_binary_folder": parser["PATHS"]["actual_next24_binary_folder"],
        "test_binary_folder": parser["PATHS"]["test_binary_folder"],
        "output_folder": parser["PATHS"]["output_folder"],
        "input_days": parser["FORECAST"]["input_days"],
    }

    for field, env_var in _ENV_OVERRIDES.items():
        if env_var in os.environ:
            values[field] = os.environ[env_var]

    values["input_days"] = int(values["input_days"])

    for field, env_var in _INFLUX_ENV_VARS.items():
        if env_var in os.environ:
            values[field] = os.environ[env_var]

    return Panel360Config(**values)


def require_dir(path, description):
    """Raise a clear ``FileNotFoundError`` if ``path`` is not an existing directory."""
    if not os.path.isdir(path):
        raise FileNotFoundError(f"{description} not found: {path!r}. Check your config.ini or PANEL360_* env vars.")
    return path


def ensure_output_dir(path):
    """Create ``path`` (and parents) if it doesn't already exist, then return it."""
    os.makedirs(path, exist_ok=True)
    return path
