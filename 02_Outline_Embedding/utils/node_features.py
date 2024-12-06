import numpy as np
import math

import shapely
from shapely.geometry import Polygon, LineString

import torch
from torch_geometric.utils import to_undirected, remove_self_loops
from torch_geometric.data import Data


def node_graph(apa_coor, apa_geo):
    num_op = len(apa_coor)
    apa_coor = apa_coor[0:-1]
    # apa_coor.pop(num_op-1)

    node_lst = []
    num_p = len(apa_coor)
    for j in range(num_p):
        p = apa_coor[j]
        if j == 0:
            sindex = -1
            oindex = j
            eindex = 1
        elif j == (len(apa_coor)-1):
            sindex = j-1
            oindex = j
            eindex = 0
        else:
            sindex = j-1
            oindex = j
            eindex = j+1

        sp = apa_coor[sindex]
        s = np.array(sp)

        op = apa_coor[oindex]
        o = np.array(op)
        ox = op[0]
        oy = op[1]

        ep = apa_coor[eindex]
        e = np.array(ep)

        Area = apa_geo.area
        local_polygon = Polygon((sp, op, ep))
        larea = (local_polygon.area) / Area

        se = LineString((sp, ep))
        llength = se.length / math.sqrt(Area)

        osv = s - o
        oev = e - o

        langle = angle_between(osv, oev)
        if langle < 0:
            langle = langle + (2*math.pi)


        oop = (0, 0)
        oo = np.array(oop)
        regional_polygon = Polygon((sp, oop, ep))
        regional_polygon_area = regional_polygon.area
        rarea = regional_polygon_area / Area

        regional_polygon_perimeter = regional_polygon.length / 2
        rperimeter = regional_polygon_perimeter / math.sqrt(Area)

        rradius = (regional_polygon_area / regional_polygon_perimeter) / math.sqrt(Area)

        oosv = s - oo
        ooev = e - oo
        rangle = angle_between(oosv, ooev)
        if rangle < 0:
            rangle = rangle + (2*math.pi)

        #ox, oy,
        nl = [larea, llength, langle, rarea, rperimeter, rradius, rangle]

        node_lst.append(nl)


    ms_lst = []
    me_lst = []
    for k in range(num_p):
        if k == (num_p - 1):
            ms = k
            me = 0
        else:
            ms = k
            me = k+1
        ms_lst.append(ms)
        me_lst.append(me)
    mse = [ms_lst, me_lst]

    datasets = []
    for i in range(2):
        node_f = torch.FloatTensor(node_lst)
        x = node_f

        edge_index = torch.tensor(mse, dtype=torch.long)
        edge_index, _ = remove_self_loops(edge_index)
        edge_index = to_undirected(edge_index=edge_index)

        data = Data(x=x, edge_index=edge_index)
        datasets.append(data)
    return datasets




def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'
        The sign of the angle is dependent on the order of v1 and v2
        so acos(norm(dot(v1, v2))) does not work and atan2 has to be used, see:
        https://stackoverflow.com/questions/21483999/using-atan2-to-find-angle-between-two-vectors
    """
    arg1 = np.cross(v1, v2)
    arg2 = np.dot(v1, v2)
    angle = np.arctan2(arg1, arg2)
    return angle

    