# !/usr/bin/python
# -*-coding:utf-8 -*-
# Main script for Shanghai's community livability index
#
import json
import pandas as pd
import geopandas as gpd

from tqdm import tqdm
from timeit import default_timer as timer
from math import radians, sin, cos, asin, sqrt
from collections import Counter, defaultdict

from multiprocessing import Pool

from pathlib import Path
from geopy.distance import distance

# Env. variable
POI_TYPE = {'事件活动': 'Event',
            '交通设施服务': 'Transport',
            '住宿服务': 'Hotel',
            '体育休闲服务': 'Sports',
            '公共设施': 'Public',
            '公司企业': 'Enterprise',
            '医疗保健服务': 'Hospital',
            '商务住宅': 'Business',
            '地名地址信息': 'Address',
            '室内设施': 'Indoor',
            '摩托车服务': 'Motor',
            '政府机构及社会团体': 'Government',
            '汽车服务': 'Car service',
            '汽车维修': 'Car main',
            '汽车销售': 'Car sale',
            '生活服务': 'Living',
            '科教文化服务': 'Education',
            '购物服务': 'Shopping',
            '通行设施': 'Passing',
            '道路附属设施': 'Road',
            '金融保险服务': 'Finance',
            '风景名胜': 'Landscape',
            '餐饮服务': 'Restaurant'}

HOUSE_PRICE = {
    1: [0, 10000],
    2: [10000, 20000],
    3: [20000, 30000],
    4: [30000, 40000],
    5: [40000, 50000],
    6: [50000, 60000],
    7: [60000, 70000],
    8: [70000, 80000],
    9: [80000, 999999]
}

# urban area of Shanghai
URBAN = ["黄埔", "徐汇", "长宁", "静安", "普陀", "虹口", "杨浦", "浦东"]
URBAN_long = ["黄浦区", "徐汇区", "长宁区", "静安区", "普陀区", "虹口区",
          "杨浦区", "浦东新区"]
URBAN2 = ["黄浦区", "徐汇区", "长宁区", "静安区", "普陀区", "虹口区",
          "杨浦区", "浦东新区", "嘉定区", "闵行区", "宝山区", "金山区", "青浦区"]
URBAN2_ENG = ["HP", "XH", "CN", "JA", "PT", "HK", "YP", "PD", "JD", "MH", "BS", "JS", "QP"]
MAPPING = {i: j for i, j in zip(URBAN2, URBAN2_ENG)}

# road map
# road_file = Path("上海路网") / "shanghai2路网.idx_54769_26641_16.shp"
# road = gpd.read_file(road_file)

# process POI data
# encode POI data into `gbk`
# root_path = Path("raw_data")
# geo = gpd.read_file(root_path / "上海poi.shp", encoding="gbk")
# geo.to_file('上海poi_gbk.shp', encoding="gbk")

# encode POI .shp file with POI types
poi_type_file = Path("raw_data/sh_poi/sh_poi_all.csv")
poi_type = pd.read_csv(poi_type_file, encoding="gbk")
level1_type = [each.split(";")[0] for each in poi_type["type"].values]  # ... level2 types have 235 items
poi_type_mapping = {i.strip(): POI_TYPE[t] for i, t in zip(poi_type["id"].values, level1_type)}

#give snapshot of each POI category with several records
categories = dict(
    edu=["Education"],
    med=["Hospital"],
    tra=["Transport"],
    rel=["Landscape", "Sports", "Indoor", "Event"],
    liv=["Restaurant", 'Business', 'Shopping', 'Living', 'Enterprise',
         'Car service', 'Government', 'Hotel', 'Passing', 'Finance',
         'Car main', 'Public', 'Car sale', 'Motor'])
category_reverse = defaultdict()
for k, items in categories.items():
    for i in items:
        category_reverse[i] = k

poi_type["level1"] = [POI_TYPE[t] for t in level1_type]
tar_keys = ["id", "name", "address", "level1", "cost", "lng", "lat"]

output = pd.DataFrame()
for key, items in categories.items():
    sub = poi_type[poi_type["level1"].isin(items)]
    if key == "liv":
        row = pd.DataFrame()
        for l, t in sub.groupby("level1"):
            row = pd.concat([row, t[tar_keys].iloc[:1, :]], axis=0)
    else:
        row = sub[tar_keys].iloc[:5, :]

    output = pd.concat([output, row], axis=0)

# read .shp file and append the POI type data onto it
# poi_file = Path("shanghai_poi_encoded") / "sh_poi_gbk.shp
# poi = gpd.read_file(poi_file)
# type_data = [poi_type_mapping.get(i.strip(), "NA") for i in poi["id"].values]
# poi["poi_type"] = type_data
# poi.to_file(poi_file, encoding="gbk")

# process resident point data
# ... 9 levels for resident point data (see HOUSE_PRICE)
# ... calculate the distance circle with a radiance being 2km
# then, for each community, we search for POIs within the range
# it should be stored in a spreadsheet:
# comm1, ... basic info ..., POI num, POI num by type (23 columns)
#

# distance algorithms: https://blog.csdn.net/weixin_42261305/article/details/105650489
#


def accelerator(func, iterable, pool_size, runtime=False, chuck=4):
    """
    Accelerate the process of func using the multi-processing method, if pool_size is 1, execute `for-loop`.

    :param func: the target function
    :param iterable: iterable object, such as list, numpy.ndarray, ...
    :param pool_size: the size of multi-process pool
    :param runtime: show the runtime of processing
    :param chuck: chuck size in pool.map()
    """
    t0 = timer()
    if pool_size > 1:
        with Pool(pool_size) as pool:
            result = pool.map(func, iterable, chuck)

    else:
        result = [func(each) for each in tqdm(iterable)]

    if runtime:
        print("Runtime: {} seconds".format(timer() - t0))

    return result


def measure(coord1, coord2, unit="km"):
    """ Calculate the distance between two coords
    :param coord1: a Point object
    :param coord2: another Point object
    :param unit: unit for distance output (m, km, mile)
    """
    d = distance([coord1.y, coord1.x], [coord2.y, coord2.x])
    return getattr(d, unit)


def quick_measure(c1, c2, unit="km"):
    lon1, lat1 = c1.y, c1.x
    lon2, lat2 = c2.y, c2.x
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    d_lon = lon2 - lon1
    d_lat = lat2 - lat1
    a = sin(d_lat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(d_lon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371  # 地球半径，千米
    return round(c * r, 3) if unit == "km" else round(c * r * 1000, 1)


def func(pack, unit="km"):
    return quick_measure(*pack, unit=unit)


def map_on_point(c, coords: list, psize=4, chuck=10):
    """Map distances between a given coord and a list of coords"""

    def gen(c, coords):
        for each in coords:
            yield c, each

    g = gen(c, coords)
    return accelerator(func, g, psize, True, chuck)


def geo_generator(path):
    for file in path.glob("*.shp"):
        geo = gpd.read_file(file)
        geo["geometry"] = geo["geometry"].to_crs("WGS84")
        yield geo


# The data template for POI matching
def get_poi_row():
    return {'cid': '',
            '徐汇区': 0,
            '浦东新区': 0,
            '黄浦区': 0,
            '长宁区': 0,
            '静安区': 0,
            '杨浦区': 0,
            '嘉定区': 0,
            '普陀区': 0,
            '闵行区': 0,
            '宝山区': 0,
            '虹口区': 0,
            '金山区': 0,
            '青浦区': 0,
            'Restaurant': 0,
            'Business': 0,
            'Shopping': 0,
            'Enterprise': 0,
            'Government': 0,
            'Hotel': 0,
            'Landscape': 0,
            'Education': 0,
            'Passing': 0,
            'Car service': 0,
            'Sports': 0,
            'Living': 0,
            'Hospital': 0,
            'Finance': 0,
            'Transport': 0,
            'Address': 0,
            'Public': 0,
            'Car main': 0,
            'Road': 0,
            'Car sale': 0,
            'Motor': 0,
            'Indoor': 0,
            'Event': 0}


def re_column(df, tag):
    cols = list(df.columns)
    df = df[cols[1:]]

    replace = [f"{MAPPING.get(each, each).replace(' ', '')[:5]}_{tag}" for each in cols[2:]]
    df.columns = ["OBJECTID"] + replace
    return pd.DataFrame(df)


def merge(src, tar, keys=[]):
    t = tar.copy(True)
    t.index = tar["OBJECTID"]

    columns = list(t.columns)
    columns.remove("OBJECTID")

    keys = columns if not keys else set(columns) & set(keys)
    for key in keys:
        res = []
        for i in src["OBJECTID"].values:
            if i in tar["OBJECTID"].values:
                item = float(t.loc[i, key])
                res += [item]
            else:
                res += [None]

        src.loc[:, key] = res

    return src


def type_counter(src, matched, get_func, keys, buffer, **kwargs):
    rows = []
    for index in range(len(src)):
        row = get_func()
        c1 = src["geometry"].values[index]
        row["cid"] = str(src["OBJECTID"].values[index])

        matched = matched.geometry.apply(lambda x: x.distance(c1))
        matched = matched[matched <= buffer]

        for key in keys:
            if key in matched.columns:
                counted = Counter(matched[key].values)
                row.update(counted)

        # write
        rows += [row]

    return rows


def distance_counter(src, circled, func, buffer, count_label=None, **kwargs):
    if count_label is None:
        count_label = "count"

    items = []
    for index in range(len(src)):
        item = func()
        c1 = src["geometry"].values[index]
        item["cid"] = str(src["OBJECTID"].values[index])

        matched = circled.geometry.apply(lambda x: x.distance(c1))
        matched = matched[matched <= buffer]

        item.update({count_label: len(matched)})
        items += [item]

    return items


def neighbour_compute(src,
                      tar,
                      outfile,
                      get_func,
                      counter,
                      keys: list = [],
                      count_label=None,
                      step=10,
                      buffer=2000):
    # get_func, the function to get data template
    # keys, the key columns for Counter() to do calculation

    # the unit for buffer is `meter`
    # write into a text file
    write_file = Path(outfile)
    if write_file.exists():
        write_file.unlink()

    rows = []
    with write_file.open("a") as doc:
        # processing
        for i in tqdm(range(0, len(src), step)):
            comm = src.iloc[i: (i + step), :]
            comm["OBJECTID"] = comm["OBJECTID"].astype(object)
            x = comm.buffer(buffer).unary_union

            t0 = timer()
            neighbours = tar["geometry"].intersection(x)
            print(f"time: {timer() - t0}")

            circled = tar[~neighbours.is_empty]
            row = counter(comm, circled, get_func, buffer, count_label, keys=keys)

            for r in row:
                doc.write(json.dumps(r) + "\n")

            rows += row

    # output
    dt = pd.DataFrame(rows)
    dt.to_excel(write_file.with_suffix(".xlsx"), encoding="gbk")
    return dt


def get_resident_data(path):
    for i in range(1, 10, 1):
        file = path / f"Level{i}.shp"
        df = gpd.read_file(file)
        df["Level"] = i
        if i == 1:
            res = df.copy(True)
        else:
            res = gpd.GeoDataFrame(pd.concat([res, df], axis=0),
                                   crs=df.crs)
    res.index = range(len(res))
    return res


if __name__ == "__main__":
    ########################################
    # load community data
    # use leveled data
    comm_dir = Path("resident_point_by_level")
    res = get_resident_data(comm_dir)

    # load POI matched data:
    km1 = re_column(pd.read_excel("comm_poi_1km.xlsx"), tag="1k")
    km2 = re_column(pd.read_excel("comm_poi_2km.xlsx"), tag="2k")

    merged = merge(res, km1).dropna()
    merged = merge(res, km2).dropna()

    # load POI data
    # poi_file = Path("shanghai_poi_encoded") / "sh_poi_circled.shp"
    # poi = gpd.read_file(poi_file)

    # # calculate with POI data
    # rows = neighbour_compute(res,
    #                          poi,
    #                          "comm_with_poi.text",
    #                          get_poi_row,
    #                          type_counter,
    #                          buffer=2000,
    #                          count_label="busline")

    ########################################
    # load busline data
    routine_dir = Path("shanghai_bus")
    routine = gpd.read_file(routine_dir / "上海公交线路.shp")
    routine = routine.to_crs("EPSG:3857")  # only under EPSG:3857, the unit is meter

    # calculate nearest bus lines
    near_busline = lambda: dict(cid="", busline=0)
    rows = neighbour_compute(res,
                             routine,
                             "comm_with_busline-1km.text",
                             near_busline,
                             distance_counter,
                             buffer=1000,
                             count_label="busline")

    ########################################
    # load bus data
    routine_dir = Path("shanghai_bus")
    station = gpd.read_file(routine_dir / "上海公交站点.shp")
    station = station.to_crs("EPSG:3857")  # only under EPSG:3857, the unit is meter

    near_busstop = lambda: dict(cid="", busstop=0)
    rows = neighbour_compute(res,
                             station,
                             "comm_with_busstop-1km.text",
                             near_busstop,
                             distance_counter,
                             buffer=1000,
                             count_label="busstop")

    ########################################
    # load road data
    road_dir = Path("shanghai_road")
    road = gpd.read_file(road_dir / "shanghai_road.shp")
    road = road.to_crs("EPSG:3857")  # only under EPSG:3857, the unit is meter
