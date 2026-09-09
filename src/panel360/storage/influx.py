"""Write anomaly forecast results to InfluxDB.

Optional: only used when ``Panel360Config.influx_enabled`` is true. Requires
the ``influx`` extra (``pip install -e ".[influx]"``).
"""

import logging

from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS

logger = logging.getLogger(__name__)

MEASUREMENT = "panel_status"


def _build_points(results_df):
    points = []
    for timestamp, row in results_df.iterrows():
        point = (
            Point(MEASUREMENT)
            .tag("forecasted", row["forecasted"])
            .field("alarm", str(row["alarm"]))
            .field("max_temperature", float(row["max_temperature"]))
            .time(timestamp)
        )
        points.append(point)
    return points


def write_anomaly_results(config, results_df):
    """Write one point per row of ``results_df`` (indexed by timestamp) to InfluxDB."""
    points = _build_points(results_df)

    with InfluxDBClient(url=config.influx_url, token=config.influx_token, org=config.influx_org) as client:
        write_api = client.write_api(write_options=SYNCHRONOUS)
        write_api.write(bucket=config.influx_bucket, org=config.influx_org, record=points)

    logger.info("Wrote %d points to InfluxDB bucket %r", len(points), config.influx_bucket)
