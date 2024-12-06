import numpy as np
import math

import shapely
from shapely.geometry import LineString

import torch
from torch_geometric.utils import to_undirected, remove_self_loops
from torch_geometric.data import Data

def segments(polyline):
    return list(map(LineString, zip(polyline.coords[:-1], polyline.coords[1:])))


def edge_graph(apa_line, apa_wall):
    wall_seg = segments(apa_line)
    num_seg = len(wall_seg)


    edge_lst = []
    for l in range(num_seg):
        seg = wall_seg[l]
        seg_length = seg.length
        seg_pro = apa_wall[l]
        south_cos = wall_segment_cosine("south", seg)
        east_cos = wall_segment_cosine("east", seg)
        north_cos = wall_segment_cosine("north", seg)
        west_cos = wall_segment_cosine("west", seg)

        if south_cos < 0:
            south_cos = 0
        if east_cos < 0:
            east_cos = 0
        if north_cos < 0:
            north_cos = 0
        if west_cos < 0:
            west_cos = 0

        if seg_pro == "I":
            south_cos = 0
            east_cos = 0
            north_cos = 0
            west_cos = 0
            
        if seg_pro == "O":
            seg_boo = 1
        else:
            seg_boo = 0

            

        edge = [seg_boo, seg_length, south_cos, north_cos, west_cos, east_cos]
        edge_lst.append(edge)


    ms_lst = []
    me_lst = []
    for k in range(num_seg):
        if k == (num_seg - 1):
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
        node_features = torch.FloatTensor(edge_lst)
        x = node_features

        edge_index = torch.tensor(mse, dtype=torch.long)
        edge_index, _ = remove_self_loops(edge_index)
        edge_index = to_undirected(edge_index=edge_index)

        data = Data(x=x, edge_index=edge_index)
        datasets.append(data)
    return datasets



def wall_segment_cosine(direction, apa_line_seg):
    seg_s = list(apa_line_seg.coords)[0]
    seg_e = list(apa_line_seg.coords)[1]
    
    normal_x = seg_e[0] - seg_s[0]
    normal_y = seg_e[1] - seg_s[1]
    
    normal_s = (-normal_y, normal_x)
    normal_e = (normal_y, -normal_x)
    
    o = np.array([-normal_y, normal_x])
    w = np.array([normal_y, -normal_x])
    
    if direction == "south":
        d = np.array([-normal_y, normal_x-1])
    if direction == "east":
        d = np.array([-normal_y+1, normal_x])
    if direction == "north":
        d = np.array([-normal_y, normal_x+1])
    if direction == "west":
        d = np.array([-normal_y-1, normal_x])
        
    od = d - o
    ow = w - o
    
    cosine = np.dot(od, ow) / (np.linalg.norm(od) * np.linalg.norm(ow))
    return cosine


    