# Compute the indicators
#

import re
import pandas as pd
import numpy as np
import geopandas as gpd

from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from itertools import chain
from scipy.stats import entropy

from main import re_column, merge

# ENVIRONMENT VARIABLE
MM = MinMaxScaler()
ZSCORE = lambda x: (x - x.min()) / (x.max() - x.min())
POI_COL = ['Resta_1k', 'Busin_1k', 'Shopp_1k', 'Enter_1k', 'Gover_1k', 'Hotel_1k',
           'Lands_1k', 'Educa_1k', 'Passi_1k', 'Carse_1k', 'Sport_1k', 'Livin_1k',
           'Hospi_1k', 'Finan_1k', 'Trans_1k', 'Addre_1k', 'Publi_1k', 'Carma_1k',
           'Road_1k', 'Carsa_1k', 'Motor_1k', 'Indoo_1k', 'Event_1k']


def standardize_by_column(df):
    for c in df.columns:
        if c != "OBJECTID":
            vec = df[c].values.reshape(len(df), 1)
            df[c] = MM.fit_transform(vec).reshape(len(df), )

    return df


# TODO: add `pop density`, `house age` and `diversity` into the model
def extract_date(text):
    ext = re.findall(r"\d+", text)
    return int(ext.pop()) if ext else None


def fill_negative_population(df):
    mask = df["population"] < 0
    df.loc[mask, "population"] = 0

    for index in df[mask].index:
        hh = df.loc[index, "Household_"]
        v = df[df["Household_"] == hh].population.values
        pp = v[v > 0].mean()
        df.loc[index, "population"] = pp

    return df


def balance_index(df, s, w):
    """Intake the topic data df and the normalization vector `s`, weighted by vector `w`"""
    # it's calculated by rows:
    res = []
    for index, row in df.iterrows():
        v = row.values
        sum_v = sum([item * (item - 1) for item in v])
        v_sum = v.sum() * (v.sum() - 1)
        res += [w.values[index] * sum_v / v_sum / s.values[index]]

    return pd.Series(res)


if __name__ == "__main__":
    from main import URBAN, get_resident_data

    # Commented out at 25/09/2023 by Yi
    res_file = Path("resident_point_by_level")
    res = get_resident_data(res_file)
    # res = res[res["District"].isin(URBAN)]
    # res.index = range(len(res))
    # res = fill_negative_population(res)

    pop = gpd.read_file('202309-revision/new-population.shp', encoding="gb2312")
    res['population'] = pop['pop']
    res = res[res["District"].isin(URBAN)]
    # standardize household price / house age
    scaled_price = res["Price"].transform(ZSCORE)

    base_year = 2020  # ... the crawling date is 2020
    const_year = res["Construt_Y"].apply(extract_date).fillna(base_year)
    house_age = base_year - const_year
    house_age = house_age.transform(ZSCORE)

    # density
    num_building = res["Building_a"].apply(extract_date).fillna(1)
    density = res["population"] / num_building
    density = density.transform(ZSCORE)

    # load POI matched data:
    km1 = re_column(pd.read_excel("comm_poi_1km.xlsx"), tag="1k")
    km2 = re_column(pd.read_excel("comm_poi_2km.xlsx"), tag="2k")

    # diversity (based on the POI data), starting from 15
    div1 = km1.iloc[:, 14:].apply(entropy, axis=1).transform(ZSCORE).to_frame()
    div1["OBJECTID"] = km1["OBJECTID"]
    div2 = km2.iloc[:, 14:].apply(entropy, axis=1).transform(ZSCORE).to_frame()
    div2["OBJECTID"] = km2["OBJECTID"]
    div = div1.merge(div2, on="OBJECTID", how="left")
    div.columns = ["div_1k", "OBJECTID", "div_2k"]

    # accessibility index
    access = pd.read_excel('202309-revision/accessibility.xlsx')
    no1 = 0.2 * (np.log((scaled_price + 1).values) + np.log((house_age + 1).values) +
                 np.log((density + 1).values) + np.log((div["div_1k"] + 1).values) +
                 np.log(access['access_1k'] + 1))
    no2 = 0.2 * (np.log((scaled_price + 1).values) + np.log((house_age + 1).values) +
                 np.log((density + 1).values) + np.log((div["div_2k"] + 1).values) +
                 np.log(access['access_2k'] + 1))

    normalized = no1.to_frame().reset_index(drop=True)
    normalized["nor_2k"] = no2
    normalized.columns = ['nor_1k', 'nor_2k']
    normalized["OBJECTID"] = div["OBJECTID"]

    # Commented out at 26/09/2023
    # # load busstop data (need re-column):
    # bs1 = re_column(pd.read_excel("comm_with_busstop-1km.xlsx"), tag="1k")
    # bs2 = re_column(pd.read_excel("comm_with_busstop-2km.xlsx"), tag="2k")
    #
    # # load busline data:
    # bl1 = re_column(pd.read_excel("comm_with_busline-1km.xlsx"), tag="1k")
    # bl2 = re_column(pd.read_excel("comm_with_busline-2km.xlsx"), tag="2k")
    #
    # # sum up into a bus-oriented data:
    # bus1 = pd.DataFrame(bs1["busst_1k"].values + bl1["busli_1k"].values,
    #                     columns=["access_1k"])
    # bus1["OBJECTID"] = bs1["OBJECTID"]
    # bus2 = pd.DataFrame(bs2["busst_2k"].values + bl2["busli_2k"].values,
    #                     columns=["access_2k"])
    # bus2["OBJECTID"] = bs2["OBJECTID"]
    #
    # # load road data:
    # ro1 = re_column(pd.read_excel("comm_with_road-1km.xlsx"), tag="1k")
    # ro2 = re_column(pd.read_excel("comm_with_road-2km.xlsx"), tag="2k")

    # # agg all the metrics
    # agg = km1.merge(km2, on="OBJECTID", how="left") \
    #     .merge(ro1, on="OBJECTID", how="left") \
    #     .merge(ro2, on="OBJECTID", how="left") \
    #     .merge(normalized, on="OBJECTID", how="left")
    #
    # agg = agg.merge(bus1, on="OBJECTID", how="left") \
    #     .merge(bus2, on="OBJECTID", how="left")

    # New merging at 26/09/2023
    agg = km1.merge(km2, on="OBJECTID", how="left") \
             .merge(normalized, on="OBJECTID", how="left")

    agg = agg.fillna(0)

    keys = dict(
        edu=["Educa"],  # edu
        med=["Hospi"],  # med
        tra=["Trans", "Road"],  # tra
        rel=["Lands", "Sport", "Indoo", "Event", "Enter"],  # rel
        liv=["Resta", "Busin", "Shopp", "Gover", "Hotel",
             "Passi", "Carse", "Livin", "Finan", "Publi", "Carma",
             "Carsa", "Motor"]  # liv
    )

    # additional keys
    func_1k = lambda x: [f"{i}_1k" for i in x]
    func_2k = lambda x: [f"{i}_2k" for i in x]
    func_sd = lambda x: MM.fit_transform(x).sum(axis=1)

    # merge with community data
    resId = res[["OBJECTID"]]
    flatten_keys = list(chain.from_iterable(list(keys.values())))
    resId = merge(resId, agg, keys=func_1k(flatten_keys))
    resId = merge(resId, agg, keys=func_2k(flatten_keys))
    resId = resId.merge(normalized, on="OBJECTID", how="left")

    resId = resId.sort_values("OBJECTID")
    res = res.sort_values("OBJECTID")

    # aggregate indicators
    for level in ['1k', '2k']:
        # calculate the metric
        final = resId[["OBJECTID"]].reset_index(drop=True)

        for metric, ks in keys.items():
            kys = [c for c in resId if c.endswith(level)]
            ks = [f"{i}_{level}" for i in ks]
            if len(ks) > 1:
                summed = func_sd(resId[ks].values)
            else:
                summed = func_sd(resId[ks].values.reshape(len(resId), 1))

            resId["agg"] = summed / len(kys)
            # calculate the final indicators
            # TODO: new formula
            w = resId.loc[:, func_1k(flatten_keys)].sum(axis=1)
            weight = resId[ks].sum(axis=1) / w
            final[metric] = balance_index(resId[ks], resId[f"nor_{level}"], weight)

        # reshape back to GeoDataFrame
        reshape_keys = ["Name", "District", "population"]
        final = gpd.GeoDataFrame(final, geometry=res["geometry"].values, crs=res.crs)
        final[reshape_keys] = res[reshape_keys].reset_index(drop=True)

        out_dir = Path("202309-revision")
        out_dir.mkdir(parents=True, exist_ok=True)

        final.to_file(out_dir / f"livability_{level}.shp", encoding="gb2312")
