"""Command-line interface for shipment performance summarization.

This script can be used directly to generate a daily shipment performance
summary from a shipments CSV file:

    python main.py path/to/shipments.csv path/to/output_summary.csv

Column names and rolling window size can be customized via optional
arguments; run with ``-h`` for details.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from shipment_summary import summarize_shipments


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute a daily per-carrier shipment performance summary, "
            "including 7-day rolling averages."
        )
    )

    parser.add_argument(
        "input_csv",
        type=Path,
        help="Path to the input shipments CSV file.",
    )
    parser.add_argument(
        "output_csv",
        type=Path,
        help="Path where the output summary CSV will be written.",
    )

    parser.add_argument(
        "--carrier-col",
        default="carrier",
        help=(
            "Name of the carrier column in the input CSV. Default: 'carrier'. "
            "Use this if your data uses a different column name (e.g. 'carrier_name')."
        ),
    )
    parser.add_argument(
        "--ship-date-col",
        default="ship_date",
        help=(
            "Name of the ship date column in the input CSV. Default: 'ship_date'. "
            "The column should be a date or datetime string."
        ),
    )
    parser.add_argument(
        "--delivery-date-col",
        default="delivery_date",
        help=(
            "Name of the delivery date column in the input CSV. Default: 'delivery_date'. "
            "Used to calculate delivery time."
        ),
    )
    parser.add_argument(
        "--promised-date-col",
        default="promised_delivery_date",
        help=(
            "Name of the promised delivery date column in the input CSV. "
            "Default: 'promised_delivery_date'. Used to calculate on-time delivery."
        ),
    )
    parser.add_argument(
        "--window-days",
        type=int,
        default=7,
        help=(
            "Rolling window size (in calendar days) for moving averages. "
            "Default: 7."
        ),
    )

    return parser.parse_args()


def main() -> None:
    # Basic logging configuration; in a production deployment this can be
    # adjusted or integrated with a centralized logging system.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    args = _parse_args()
    logger = logging.getLogger("shipment_summary_cli")

    try:
        summarize_shipments(
            input_path=args.input_csv,
            output_path=args.output_csv,
            carrier_col=args.carrier_col,
            ship_date_col=args.ship_date_col,
            delivery_date_col=args.delivery_date_col,
            promised_delivery_date_col=args.promised_date_col,
            window_days=args.window_days,
        )
    except Exception as exc:  # pragma: no cover - defensive catch
        logger.exception("Failed to compute shipment summary: %s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
