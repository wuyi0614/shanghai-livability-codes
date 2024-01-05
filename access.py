# Created at 23/09/2023 by Yi WU
#
# The script for the calculation of accessibility using the cumulative opportunity method (CUM) with POIs data.
# It is for the first-round of revision.
#
# Reference:
# - Barboza, M. H., Carneiro, M. S., Falavigna, C., Luz, G., & Orrico, R. (2021). Balancing time: Using a new accessibility measure in Rio de Janeiro. Journal of Transport Geography, 90, 102924.
# - Guan, J., Zhang, K., Shen, Q., & He, Y. (2020). Dynamic modal accessibility gap: measurement and application using travel routes data. Transportation Research Part D: Transport and Environment, 81, 102272.

import pandas as pd

from pathlib import Path


# redefine the following functions for revision
def accessibility(raw: pd.DataFrame, coords: list, radius=None, key='access'):
    """Accessibility Score Calculation
    A location-based cumulative opportunity method based accessibility when using travel mode m for location i
    The equation is written as:
        A_i^m = sum_j{delta_j^m * O_j}
    where O_j is the number of POIs at the destination j,
          delta is the threshold function that has a value of `1` if j is located within the threshold travel time r
          from location i by travel mode m, and a value of `0` otherwise.

    Reference: Guan et al. (2020), https://doi.org/10.1016/j.trd.2020.102272
    :param raw:    the raw dataframe of RBCs' locations, IDs, and metadata
    :param coords: the target list of coords, specifically bus and metro
    :param radius: the radius of POIs in distance, default 1 and 2 if None
    :param key:    key for output, as the column name
    """
    out = []
    for _, row in raw.iterrows():
        row = row.to_dict()
        c = Coord(*(row['Lat_Lon_fr'].split(',')))
        ds = map_on_point(c, coords, 4, 10)
        if radius:  # given the right radius
            radius = float(radius)
            row[f'{key}_{radius}k'] = len(list(filter(lambda x: x <= radius, ds)))
        else:
            row[f'{key}_1k'] = len(list(filter(lambda x: x <= 1, ds)))
            row[f'{key}_2k'] = len(list(filter(lambda x: x <= 2, ds)))
        out += [row]

    return pd.DataFrame(out)


class Coord(object):
    """A simplified coordinate object"""

    def __init__(self, lng, lat):
        self.y = float(lng)
        self.x = float(lat)


if __name__ == '__main__':
    from main import *

    # redo transport-related data
    transit = pd.read_excel('202309-revision/shanghai-transport.xlsx')
    # metro-related points
    mask = (transit['id'].str.startswith('BX')) & (transit['adname'].isin(URBAN2))
    metro = transit[mask]
    metro['loc'] = metro[['lng', 'lat']].apply(lambda x: Coord(*x), axis=1)
    print(f'Got {len(metro)} metro points')

    # bus-related points
    mask = (transit['id'].str.startswith('BV')) & (transit['adname'].isin(URBAN2))
    bus = transit[mask]
    bus['loc'] = bus[['lng', 'lat']].apply(lambda x: Coord(*x), axis=1)
    print(f'Got {len(bus)} bus points')

    # resident data
    res = pd.read_excel('202309-revision/resident.xlsx')
    res['loc'] = res['Lat_Lon_fr'].apply(lambda x: Coord(*(x.split(','))))
    # map coords in bulk on one coord from residents' locations
    ds = map_on_point(res['loc'].values[0], metro['loc'].values)

    # overall test and temporarily save the data
    out = accessibility(res, metro['loc'].values, key='metro')
    out.to_excel('202309-revision/metro-access.xlsx', index=False)
    busout = accessibility(res, bus['loc'].values, key='bus')
    busout.to_excel('202309-revision/bus-access.xlsx', index=False)
    # produce the final score
    oj = len(metro) + len(bus)
    acs = out[['OBJECTID']].copy(True)
    acs['access_1k'] = (out['metro_1k'] + busout['bus_1k']) / oj
    acs['access_2k'] = (out['metro_2k'] + busout['bus_2k']) / oj
    acs.to_excel('202309-revision/accessibility.xlsx', index=False)
