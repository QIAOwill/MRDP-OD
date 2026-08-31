"""Generate deterministic synthetic TSV files matching the MRDP-OD input schema."""
from __future__ import annotations

from itertools import permutations
from pathlib import Path
import argparse
import math

import numpy as np
import pandas as pd


REGIONS = ("Beijing_Group", "Chang_Delta", "Zhu_Delta")
CITY_STATIC_COLUMNS = (
    "city_id city_name gdp gdp_per_capita income_per_capita fixed_asset_investment "
    "population population_density age_0_14_ratio age_15_64_ratio age_65_plus_ratio "
    "primary_industry_ratio secondary_industry_ratio tertiary_industry_ratio area_km2 "
    "builtup_area_km2 road_density road_area_per_capita poi_food_ratio poi_company_ratio "
    "poi_shopping_ratio poi_transport_ratio poi_finance_ratio poi_hotel_ratio "
    "poi_edu_culture_ratio poi_tourism_ratio poi_auto_ratio poi_business_residential_ratio "
    "poi_life_service_ratio poi_entertainment_ratio poi_medical_ratio poi_sports_ratio "
    "poi_density reachable_city_200km reachable_city_500km reachable_city_1000km "
    "dialect_diversity_index dialect_count avg_wage_latest house_price_latest "
    "scenic_spot_count a_level_scenic_count is_provincial_capital is_municipality "
    "urban_agglomeration"
).split()


def _write(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, sep="\t", index=False, encoding="utf-8")


def generate_region(root: Path, region: str, region_index: int, days: int) -> None:
    rng = np.random.default_rng(20260701 + region_index)
    dates = pd.date_range("2021-01-01", periods=days, freq="D")
    city_ids = [1000 * (region_index + 1) + i for i in range(1, 5)]
    city_names = [f"Synthetic-{region_index + 1}-{i}" for i in range(1, 5)]
    city_lookup = dict(zip(city_ids, city_names))
    folder = root / region

    static_rows = []
    for i, (city_id, name) in enumerate(city_lookup.items()):
        values = {
            "city_id": city_id, "city_name": name, "gdp": 5000 + 900 * i,
            "gdp_per_capita": 8.0 + i, "income_per_capita": 4.0 + 0.4 * i,
            "fixed_asset_investment": 1800 + 250 * i, "population": 400 + 70 * i,
            "population_density": 600 + 90 * i, "age_0_14_ratio": 0.16,
            "age_15_64_ratio": 0.69, "age_65_plus_ratio": 0.15,
            "primary_industry_ratio": 0.05, "secondary_industry_ratio": 0.39,
            "tertiary_industry_ratio": 0.56, "area_km2": 8000 + 500 * i,
            "builtup_area_km2": 500 + 60 * i, "road_density": 1.8 + 0.1 * i,
            "road_area_per_capita": 18 + i, "poi_density": 2.0 + 0.2 * i,
            "reachable_city_200km": 2, "reachable_city_500km": 6,
            "reachable_city_1000km": 15, "dialect_diversity_index": 0.2 + 0.03 * i,
            "dialect_count": 2 + i, "avg_wage_latest": 8.0 + 0.5 * i,
            "house_price_latest": 1.5 + 0.2 * i, "scenic_spot_count": 20 + i,
            "a_level_scenic_count": 5 + i, "is_provincial_capital": int(i == 0),
            "is_municipality": 0, "urban_agglomeration": region,
        }
        for column in CITY_STATIC_COLUMNS:
            if column.startswith("poi_") and column not in values:
                values[column] = round(0.03 + 0.005 * ((i + len(column)) % 8), 4)
        static_rows.append(values)
    _write(pd.DataFrame(static_rows)[CITY_STATIC_COLUMNS], folder / f"{region}_city_static.txt")

    dynamic_rows = []
    for day_index, date in enumerate(dates):
        for i, (city_id, name) in enumerate(city_lookup.items()):
            seasonal = math.sin(2 * math.pi * day_index / 365.25)
            dynamic_rows.append({
                "date": date, "city_id": city_id, "city_name": name,
                "is_weekend": int(date.dayofweek >= 5), "is_holiday": 0,
                "holiday_type": "none", "month": date.month, "day_of_week": date.dayofweek,
                "temp_mean": round(15 + 10 * seasonal + i, 3),
                "wind_speed": round(2.0 + 0.2 * i + rng.uniform(-0.2, 0.2), 3),
                "snow_flag": int(date.month in (1, 2) and region_index == 0),
                "dewpoint_temperature": round(8 + 7 * seasonal + 0.5 * i, 3),
            })
    _write(pd.DataFrame(dynamic_rows), folder / f"{region}_city_dynamic.txt")

    pair_rows, weather_rows, od_rows = [], [], []
    for origin_id, destination_id in permutations(city_ids, 2):
        oi, di = city_ids.index(origin_id), city_ids.index(destination_id)
        distance = 80 + 70 * abs(oi - di) + 10 * region_index
        pair_rows.append({
            "origin_id": origin_id, "origin_city": city_lookup[origin_id],
            "destination_id": destination_id, "destination_city": city_lookup[destination_id],
            "distance_line": distance, "distance_road": round(distance * 1.18, 2),
            "distance_railway": round(distance * 1.08, 2), "is_adjacent": int(abs(oi-di) == 1),
            "same_province": int((oi // 2) == (di // 2)), "gdp_gap": abs(oi-di) * 900,
            "income_gap": abs(oi-di) * 0.4, "population_gap": abs(oi-di) * 70,
            "industry_structure_similarity": round(0.92 - 0.04 * abs(oi-di), 3),
            "poi_structure_similarity": round(0.90 - 0.03 * abs(oi-di), 3),
            "hsr_direct_flag": 1, "hsr_train_count": 8 - abs(oi-di),
            "hsr_min_travel_time": round(distance / 220, 3),
            "hsr_avg_travel_time": round(distance / 180, 3),
            "hsr_service_intensity": round((8 - abs(oi-di)) / 8, 3),
        })
        for day_index, date in enumerate(dates):
            weekly = 1 + 0.14 * math.sin(2 * math.pi * day_index / 7)
            trend = 1 + 0.0015 * day_index
            base = 90 + 18 * oi + 12 * di + 10 * region_index
            flow = max(0.0, base * weekly * trend + rng.normal(0, 3))
            od_rows.append({
                "date": date, "origin_id": origin_id, "origin_city": city_lookup[origin_id],
                "destination_id": destination_id, "destination_city": city_lookup[destination_id],
                "od_flow": round(flow, 3),
            })
            weather_rows.append({
                "date": date, "origin_id": origin_id, "origin_city": city_lookup[origin_id],
                "destination_id": destination_id, "destination_city": city_lookup[destination_id],
                "temp_diff": float(oi - di), "wind_max": round(2.5 + 0.2 * max(oi, di), 3),
                "snow_any": int(date.month in (1, 2) and region_index == 0),
            })
    _write(pd.DataFrame(pair_rows), folder / f"{region}_city_pair_static.txt")
    _write(pd.DataFrame(od_rows), folder / f"{region}_OD.txt")
    _write(pd.DataFrame(weather_rows), folder / f"{region}_pair_weather.txt")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "sample_data")
    parser.add_argument("--days", type=int, default=180)
    args = parser.parse_args()
    if args.days < 90:
        raise ValueError("--days must be at least 90 for the default temporal split and windows")
    for index, region in enumerate(REGIONS):
        generate_region(args.output, region, index, args.days)
    print(f"Synthetic sample data written to: {args.output.resolve()}")


if __name__ == "__main__":
    main()
