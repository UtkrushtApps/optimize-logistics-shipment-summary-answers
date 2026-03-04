# Solution Steps

1. Create a new Python module file named `shipment_summary.py` that will contain the core summarization logic using pandas and NumPy.

2. Import the required standard-library modules (`logging`, `Path` from `pathlib`, and typing utilities) and third-party libraries (`pandas` and `numpy`).

3. Configure a module-level logger using `logging.getLogger(__name__)` so that the logic can emit useful runtime information and warnings.

4. Implement a small helper function `_ensure_required_columns(df, required)` that checks if all required column names exist in the DataFrame and raises a `ValueError` with a clear message if any are missing.

5. Implement another helper function `_build_complete_date_range(daily, carrier_col, date_col)` that, for each carrier, builds a continuous daily date index from the minimum to maximum observed date and reindexes the aggregated metrics to this index, returning the concatenated result for all carriers.

6. Define the main function `summarize_shipments(input_path, output_path, *, carrier_col='carrier', ship_date_col='ship_date', delivery_date_col='delivery_date', promised_delivery_date_col='promised_delivery_date', window_days=7)` with a detailed docstring describing expected input columns, behavior, return value, and possible exceptions.

7. Inside `summarize_shipments`, convert `input_path` and `output_path` to `Path` objects and verify that the input file exists; raise `FileNotFoundError` if it does not.

8. Read the CSV into a pandas DataFrame using `pd.read_csv`, specifying `parse_dates` for the ship, delivery, and promised date columns (when present), and wrap this in a `try/except` that raises a `RuntimeError` on unexpected read failures.

9. Validate that the always-required columns (carrier, ship date, delivery date) are present using `_ensure_required_columns`; check whether the promised date column exists and log a warning if it does not (on-time metrics will then be NaN).

10. Drop rows with missing `carrier_col` or `ship_date_col` as they are critical identifiers; log how many rows were removed and raise a `ValueError` if no rows remain.

11. Normalize carrier values by stripping whitespace, dropping empty strings, and casting the carrier column to `category` type for memory efficiency.

12. Coerce the ship, delivery, and promised date columns to pandas datetime using `pd.to_datetime(..., errors='coerce')`, then drop rows with invalid ship dates (`NaT`) and log how many were removed; raise a `ValueError` if this leaves no data.

13. Create a new column `ship_day` using `df[ship_date_col].dt.normalize()` to represent the shipment day at daily grain (midnight, no time component).

14. Compute a per-shipment `delivery_time_days` column only where `delivery_date_col` is non-null: subtract `ship_date_col` from `delivery_date_col`, convert the timedelta to seconds, divide by 86400 to get days, and store as a float.

15. Detect any negative delivery times (where delivery occurs before ship date); log a warning, and set those values to `NaN` so they do not corrupt averages.

16. If the promised delivery column exists, build an `on_time_flag` boolean column where both delivery and promised dates are present (`notna`) and the actual delivery is less than or equal to the promised date; leave rows without both timestamps as `NaN` in `on_time_flag`. If the promised date column is missing, create `on_time_flag` filled with `NaN`.

17. Group the cleaned DataFrame by `[carrier_col, 'ship_day']` and aggregate to daily metrics: `shipments_count` as the group size, `delivery_time_days_sum` as the sum of `delivery_time_days`, `delivery_time_days_count` as the count of non-null `delivery_time_days`, `on_time_shipments` as the sum of `on_time_flag` (treating True as 1, ignoring NaN), and `on_time_eligible` as the count of non-null `on_time_flag`. Reset the index to return a flat DataFrame.

18. If the aggregated `daily` DataFrame is empty, raise a `ValueError` indicating that no daily shipment data could be computed.

19. Call `_build_complete_date_range(daily, carrier_col, 'ship_day')` to ensure that for each carrier you have a continuous series of dates from the first to the last shipment day, even for days with zero shipments.

20. For the count and sum columns (`shipments_count`, `delivery_time_days_sum`, `delivery_time_days_count`, `on_time_shipments`, `on_time_eligible`), fill missing values with zero so that subsequent rolling-window calculations behave correctly on days with no shipments.

21. Cast the integer-like columns (`shipments_count`, `delivery_time_days_count`, `on_time_shipments`, `on_time_eligible`) to pandas' nullable integer type (`Int64`) for correctness and memory efficiency.

22. Compute per-day metrics (not yet rolling): `avg_delivery_time_days` as `delivery_time_days_sum / delivery_time_days_count` (treating zero counts as NaN), and `on_time_rate` as `on_time_shipments / on_time_eligible` (again treating zero eligible as NaN); use NumPy error-state context to suppress divide-by-zero warnings.

23. Validate that `window_days` is a positive integer; raise a `ValueError` if it is not.

24. Sort the `daily` DataFrame by `[carrier_col, 'ship_day']` to guarantee proper temporal order for rolling operations.

25. Implement a helper inside `summarize_shipments` that, for a given numeric Series, groups by `carrier_col` and applies a `.rolling(window_days, min_periods=1).sum()` transform, returning the aligned rolling sums for each carrier.

26. Use this helper to compute rolling sums for `delivery_time_days_sum`, `delivery_time_days_count`, `on_time_shipments`, and `on_time_eligible`, storing them in new columns like `delivery_time_days_sum_rolling` and `on_time_eligible_rolling`.

27. Compute rolling, volume-weighted metrics using the rolling sums: `f"{window_days}d_avg_delivery_time_days"` as `delivery_time_days_sum_rolling / delivery_time_days_count_rolling` and `f"{window_days}d_on_time_rate"` as `on_time_shipments_rolling / on_time_eligible_rolling`, replacing zero denominators with NaN to avoid invalid values.

28. Rename the `ship_day` column to a generic `date` column for clarity in the output, and then subset the DataFrame to only the final output columns: carrier, date, shipments_count, avg_delivery_time_days, on_time_rate, and the two rolling metrics.

29. Sort the resulting summary by `[carrier_col, 'date']`, reset the index, and round floating-point metric columns (average delivery time and on-time rate, both daily and rolling) to a few decimal places (e.g. 4) for readability.

30. Ensure that the output directory exists (create parent directories using `mkdir(parents=True, exist_ok=True)` if necessary), then write the final summary DataFrame to `output_path` via `to_csv(index=False)` inside a `try/except` that raises `RuntimeError` on unexpected write failures.

31. Return the final summary DataFrame from `summarize_shipments` so that it can be used programmatically in other parts of the system (such as an API endpoint).

32. Create a separate `main.py` file to act as a command-line interface that imports `summarize_shipments` from `shipment_summary`.

33. In `main.py`, use `argparse` to define positional arguments for `input_csv` and `output_csv`, and optional arguments for `--carrier-col`, `--ship-date-col`, `--delivery-date-col`, `--promised-date-col`, and `--window-days` with appropriate defaults and help text.

34. Configure basic logging in `main()` using `logging.basicConfig` for consistent, timestamped console output.

35. In `main()`, parse the command-line arguments, then call `summarize_shipments` with the parsed arguments (mapping CLI flags to function parameters) inside a `try/except` block that logs any exception and exits with a non-zero status code on failure.

36. Add the standard `if __name__ == '__main__': main()` guard at the bottom of `main.py` so the script can be run directly from the command line.

