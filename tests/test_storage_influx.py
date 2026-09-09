from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

pytest.importorskip("influxdb_client")

from panel360.config import Panel360Config
from panel360.storage.influx import MEASUREMENT, write_anomaly_results


def make_config():
    return Panel360Config(
        binary_folder="data/binary",
        actual_next24_binary_folder="data/actual",
        test_binary_folder="data/test",
        output_folder="out",
        input_days=7,
        influx_url="http://localhost:8086",
        influx_token="secret-token",
        influx_org="panel360",
        influx_bucket="panel-status",
    )


def make_results_df():
    return pd.DataFrame(
        {
            "alarm": ["Green", "Red"],
            "max_temperature": [42.5, 61.2],
            "forecasted": ["no", "yes"],
        },
        index=pd.to_datetime(["2024-01-01 00:00", "2024-01-01 01:00"]),
    )


@patch("panel360.storage.influx.InfluxDBClient")
def test_write_anomaly_results_writes_one_point_per_row(mock_client_cls):
    mock_client = MagicMock()
    mock_client_cls.return_value.__enter__.return_value = mock_client
    config = make_config()

    write_anomaly_results(config, make_results_df())

    mock_client_cls.assert_called_once_with(url=config.influx_url, token=config.influx_token, org=config.influx_org)
    write_api = mock_client.write_api.return_value
    write_api.write.assert_called_once()
    _, kwargs = write_api.write.call_args
    assert kwargs["bucket"] == config.influx_bucket
    assert kwargs["org"] == config.influx_org
    points = kwargs["record"]
    assert len(points) == 2
    assert all(point._name == MEASUREMENT for point in points)
    assert {point.__dict__["_tags"]["forecasted"] for point in points} == {"no", "yes"}
