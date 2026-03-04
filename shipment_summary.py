"""Shipment performance summarization using pandas and NumPy.

This module provides a single public function, :func:`summarize_shipments`,
which reads a shipment-level CSV file and produces a clean daily summary per
carrier, including:

- Daily shipment count.
- Daily average delivery time (in days).
- Daily on-time delivery rate.
- 7-day rolling average delivery time (in days), weighted by shipment volume.
- 7-day rolling on-time delivery rate, weighted by shipment volume.

The summary is written to an output CSV and also returned as a pandas
DataFrame so it can be integrated into other services (e.g. an API endpoint).

The implementation is vectorized using pandas/NumPy to avoid slow Python
loops, and it fills in missing calendar dates per carrier so that there are
no gaps in the daily output.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _ensure_required_columns(df: pd.DataFrame, required: list[str]) -> None:
    """Validate that all required columns are present.

    Parameters
    ----------
    df:
        The input DataFrame.
    required:
        List of column names that must be present.

    Raises
    ------
    ValueError
        If any of the required columns are missing.
    """

    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(
            f"Input data is missing required column(s): {', '.join(missing)}. "
            "Please verify the input CSV schema."
        )


def _build_complete_date_range(
    daily: pd.DataFrame,
    carrier_col: str,
    date_col: str,
) -> pd.DataFrame:
    """Expand per-carrier time series so there are no missing dates.

    For each carrier, this function creates a continuous daily date index from
    the minimum to maximum observed date and reindexes the aggregated metrics
    to that index. Days with no shipments will have zero counts and NaN metrics
    until rolling windows are applied.

    Parameters
    ----------
    daily:
        DataFrame with at least ``carrier_col`` and ``date_col`` plus metric
        columns.
    carrier_col:
        Name of the carrier column.
    date_col:
        Name of the date column (daily grain).

    Returns
    -------
    pd.DataFrame
        DataFrame with a continuous daily series per carrier.
    """

    result_frames: list[pd.DataFrame] = []

    for carrier, sub in daily.groupby(carrier_col):
        sub = sub.sort_values(date_col)
        if sub.empty:
            continue

        first_date = sub[date_col].min()
        last_date = sub[date_col].max()
        full_index = pd.date_range(first_date, last_date, freq="D", name=date_col)

        sub = sub.set_index(date_col).reindex(full_index)
        sub[carrier_col] = carrier

        result_frames.append(sub.reset_index())

    if not result_frames:
        # No data available; return an empty DataFrame with same columns
        return daily.loc[[]].copy()

    complete = pd.concat(result_frames, ignore_index=True)
    return complete


def summarize_shipments(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    *,
    carrier_col: str = "carrier",
    ship_date_col: str = "ship_date",
    delivery_date_col: str = "delivery_date",
    promised_delivery_date_col: str = "promised_delivery_date",
    window_days: int = 7,
) -> pd.DataFrame:
    """Read shipment-level data and compute a daily performance summary.

    Parameters
    ----------
    input_path:
        Path to the input ``shipments.csv`` file. The CSV is expected to
        contain at least these columns (names configurable via parameters):

        - ``carrier_col``: Carrier identifier (e.g. carrier name or code).
        - ``ship_date_col``: Timestamp or date when the shipment was sent.
        - ``delivery_date_col``: Timestamp or date when the shipment was
          actually delivered. Used to compute delivery time.
        - ``promised_delivery_date_col`` (optional but recommended):
          Promised or expected delivery date/time for the shipment. Used to
          compute on-time delivery.

    output_path:
        Where to write the summarized CSV. Parent directories will be created
        if needed.
    carrier_col:
        Name of the carrier identifier column in ``input_path``.
    ship_date_col:
        Name of the shipment date column.
    delivery_date_col:
        Name of the actual delivery date column.
    promised_delivery_date_col:
        Name of the promised delivery date column. If this column is missing
        from the CSV, on-time metrics will be set to NaN.
    window_days:
        Size of the rolling window (in calendar days) for the moving averages.
        Default is 7.

    Returns
    -------
    pd.DataFrame
        The summarized daily performance DataFrame that was written to the
        output CSV. Columns include:

        - ``carrier`` (or configured name)
        - ``date`` (shipment day)
        - ``shipments_count``
        - ``avg_delivery_time_days``
        - ``on_time_rate``
        - ``<window_days>d_avg_delivery_time_days``
        - ``<window_days>d_on_time_rate``

    Raises
    ------
    FileNotFoundError
        If the input CSV does not exist.
    ValueError
        If required columns are missing or if, after cleaning, there is no
        usable data.
    RuntimeError
        If reading or writing the CSV fails for unexpected reasons.
    """

    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    # Read CSV with date parsing; handle common read-time errors explicitly.
    parse_dates = [col for col in {ship_date_col, delivery_date_col, promised_delivery_date_col} if col is not None]

    try:
        df = pd.read_csv(input_path, parse_dates=parse_dates)
    except Exception as exc:  # pragma: no cover - defensive catch
        raise RuntimeError(f"Failed to read input CSV '{input_path}': {exc}") from exc

    # Validate always-required columns.
    _ensure_required_columns(df, [carrier_col, ship_date_col, delivery_date_col])

    # Promised date is optional; warn if missing.
    has_promised_date = promised_delivery_date_col in df.columns
    if not has_promised_date:
        logger.warning(
            "Promised delivery date column '%s' is missing; on-time metrics will be NaN.",
            promised_delivery_date_col,
        )

    # Basic cleaning: drop rows with missing critical identifiers.
    initial_rows = len(df)
    df = df.dropna(subset=[carrier_col, ship_date_col])

    if df.empty:
        raise ValueError(
            "No valid shipment rows after dropping records with missing carrier/ship_date."
        )

    cleaned_rows = len(df)
    if cleaned_rows < initial_rows:
        logger.info("Dropped %d rows with missing carrier or ship_date", initial_rows - cleaned_rows)

    # Normalize and clean up types.
    # - Carrier as stripped string / category for memory efficiency.
    df[carrier_col] = (
        df[carrier_col]
        .astype("string")
        .str.strip()
    )
    df = df[df[carrier_col].notna() & (df[carrier_col] != "")]
    df[carrier_col] = df[carrier_col].astype("category")

    # Ensure dates are parsed as datetimes; coerce bad values to NaT instead of raising.
    df[ship_date_col] = pd.to_datetime(df[ship_date_col], errors="coerce")
    df[delivery_date_col] = pd.to_datetime(df[delivery_date_col], errors="coerce")
    if has_promised_date:
        df[promised_delivery_date_col] = pd.to_datetime(
            df[promised_delivery_date_col], errors="coerce"
        )

    # Drop rows where ship_date is invalid after coercion.
    before_valid_ship = len(df)
    df = df.dropna(subset=[ship_date_col])
    if df.empty:
        raise ValueError("All ship_date values are invalid or missing after parsing.")

    if len(df) < before_valid_ship:
        logger.info(
            "Dropped %d rows with invalid ship_date values after parsing.",
            before_valid_ship - len(df),
        )

    # Ship day: date-level grain (normalized to midnight without time component).
    df["ship_day"] = df[ship_date_col].dt.normalize()

    # ------------------------------------------------------------------
    # Compute per-shipment metrics in a vectorized way.
    # ------------------------------------------------------------------

    # Delivery time in days (can be fractional). Only for rows with a valid delivery_date.
    has_delivery = df[delivery_date_col].notna()
    df.loc[has_delivery, "delivery_time_days"] = (
        (df.loc[has_delivery, delivery_date_col] - df.loc[has_delivery, ship_date_col])
        .dt.total_seconds()
        / 86400.0
    )

    # Guard against negative delivery times (e.g. corrupted data where delivery < ship).
    negative_delivery_mask = df["delivery_time_days"] < 0
    if negative_delivery_mask.any():
        logger.warning(
            "Found %d rows with negative delivery times; setting them to NaN.",
            int(negative_delivery_mask.sum()),
        )
        df.loc[negative_delivery_mask, "delivery_time_days"] = np.nan

    # On-time flag: True if delivered on/before promised date.
    if has_promised_date:
        eligible_mask = df[delivery_date_col].notna() & df[promised_delivery_date_col].notna()
        df.loc[eligible_mask, "on_time_flag"] = (
            df.loc[eligible_mask, delivery_date_col] <= df.loc[eligible_mask, promised_delivery_date_col]
        )
        # Non-eligible rows remain NaN in on_time_flag.
    else:
        df["on_time_flag"] = np.nan

    # ------------------------------------------------------------------
    # Aggregate to daily per-carrier level.
    # ------------------------------------------------------------------

    group_cols = [carrier_col, "ship_day"]

    # Aggregations are designed to support volume-weighted rolling metrics.
    daily = (
        df.groupby(group_cols, dropna=False)
        .agg(
            shipments_count=("ship_day", "size"),
            # For average delivery time we use sum and count so later we can compute
            # weighted rolling averages.
            delivery_time_days_sum=("delivery_time_days", "sum"),
            delivery_time_days_count=("delivery_time_days", "count"),
            # On-time metrics: count True values and eligible shipments.
            on_time_shipments=(
                "on_time_flag",
                # sum of True (1) ignoring NaN; if all NaN => 0
                lambda s: s.dropna().sum() if len(s) else 0,
            ),
            on_time_eligible=("on_time_flag", "count"),
        )
        .reset_index()
    )

    if daily.empty:
        raise ValueError("No aggregated daily shipment data could be computed from input.")

    # ------------------------------------------------------------------
    # Ensure continuous date index per carrier (fill missing calendar dates).
    # ------------------------------------------------------------------

    daily = _build_complete_date_range(daily, carrier_col=carrier_col, date_col="ship_day")

    # For days with no shipments, fill counts/sums with zero so rolling windows
    # can still be computed correctly. Keep them as numeric.
    for col in [
        "shipments_count",
        "delivery_time_days_sum",
        "delivery_time_days_count",
        "on_time_shipments",
        "on_time_eligible",
    ]:
        if col in daily.columns:
            daily[col] = daily[col].fillna(0)

    # Cast integer-like columns to integer dtype where possible.
    int_cols = ["shipments_count", "delivery_time_days_count", "on_time_shipments", "on_time_eligible"]
    for col in int_cols:
        if col in daily.columns:
            # Use pandas nullable integer type to safely handle any remaining NaNs.
            daily[col] = daily[col].astype("Int64")

    # ------------------------------------------------------------------
    # Per-day metrics (un-windowed).
    # ------------------------------------------------------------------

    # Average delivery time (days) per day, based only on shipments that have
    # a valid delivery_time_days.
    with np.errstate(divide="ignore", invalid="ignore"):
        daily["avg_delivery_time_days"] = daily["delivery_time_days_sum"] / daily[
            "delivery_time_days_count"
        ].replace({0: np.nan})

        # On-time rate per day: on_time_shipments / on_time_eligible.
        daily["on_time_rate"] = daily["on_time_shipments"] / daily["on_time_eligible"].replace({0: np.nan})

    # ------------------------------------------------------------------
    # Rolling (window_days) metrics per carrier.
    # ------------------------------------------------------------------

    if window_days <= 0:
        raise ValueError("window_days must be a positive integer.")

    daily = daily.sort_values([carrier_col, "ship_day"])  # Ensure correct temporal order.

    # Helper to compute group-wise rolling sums.
    def _rolling_sum_by_carrier(series: pd.Series) -> pd.Series:
        return (
            series.groupby(daily[carrier_col])
            .transform(lambda s: s.rolling(window_days, min_periods=1).sum())
        )

    daily["delivery_time_days_sum_rolling"] = _rolling_sum_by_carrier(daily["delivery_time_days_sum"].astype(float))
    daily["delivery_time_days_count_rolling"] = _rolling_sum_by_carrier(
        daily["delivery_time_days_count"].astype(float)
    )
    daily["on_time_shipments_rolling"] = _rolling_sum_by_carrier(daily["on_time_shipments"].astype(float))
    daily["on_time_eligible_rolling"] = _rolling_sum_by_carrier(daily["on_time_eligible"].astype(float))

    # Rolling averages, volume-weighted by shipment counts in the window.
    window_suffix = f"{window_days}d"
    with np.errstate(divide="ignore", invalid="ignore"):
        daily[f"{window_suffix}_avg_delivery_time_days"] = (
            daily["delivery_time_days_sum_rolling"]
            / daily["delivery_time_days_count_rolling"].replace({0: np.nan})
        )
        daily[f"{window_suffix}_on_time_rate"] = (
            daily["on_time_shipments_rolling"]
            / daily["on_time_eligible_rolling"].replace({0: np.nan})
        )

    # ------------------------------------------------------------------
    # Final cleanup & output.
    # ------------------------------------------------------------------

    # Rename ship_day to a generic date column for clarity in the output.
    daily = daily.rename(columns={"ship_day": "date"})

    # Drop intermediate columns that are not needed in the final summary CSV.
    daily = daily[
        [
            carrier_col,
            "date",
            "shipments_count",
            "avg_delivery_time_days",
            "on_time_rate",
            f"{window_suffix}_avg_delivery_time_days",
            f"{window_suffix}_on_time_rate",
        ]
    ].copy()

    # Sort consistently for easier downstream consumption.
    daily = daily.sort_values([carrier_col, "date"]).reset_index(drop=True)

    # Optional: round floating-point metrics for readability.
    float_cols = [
        "avg_delivery_time_days",
        "on_time_rate",
        f"{window_suffix}_avg_delivery_time_days",
        f"{window_suffix}_on_time_rate",
    ]
    for col in float_cols:
        if col in daily.columns:
            daily[col] = daily[col].astype(float).round(4)

    # Ensure parent directory exists before writing.
    if output_path.parent and not output_path.parent.exists():
        output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        daily.to_csv(output_path, index=False)
    except Exception as exc:  # pragma: no cover - defensive catch
        raise RuntimeError(f"Failed to write output CSV '{output_path}': {exc}") from exc

    logger.info("Wrote shipment summary to %s (rows=%d)", output_path, len(daily))

    return daily


__all__ = ["summarize_shipments"]
