import numpy as np
from collections import defaultdict
import shapely
import shapely.wkt
from shapely.geometry import LineString, MultiLineString, Polygon, Point, MultiPoint
from shapely.prepared import prep


def list_duplicates(seq):
    tally = defaultdict(list)
    for i,item in enumerate(seq):
        tally[item].append(i)
    return ((key,locs) for key,locs in tally.items() if len(locs)>1)


# clean wall and wkt
# wall = ((IOOI))
# wkt = 'POLYGON ((x0 y0, x1 y1, x2 y2, x3 y3, x0 y0))'
def read_wall_wkt(wall, wkt):
    # wall to list
    # print('wall:', wall)
    wall_l = wall.split("), (")[0]
    wall_l = wall_l.split("((")[1]
    wall_l = wall_l.split("))")[0]
    wall_c = [*wall_l]

    #clean wkt
    wkt_l = wkt.split("((")[1]
    wkt_l = wkt_l.split("))")[0]
    wkt_l = wkt_l.split("), (")

    if len(wkt_l) == 1:
        wkt_c = wkt
    else:
        wkt_c = "POLYGON ((" + wkt_l[0] + "))"

    wkt_c = wkt_c.split("((")[1]
    wkt_c = wkt_c.split("))")[0]  
    wkt_c = wkt_c.split(", ")

    # remove duplicate point
    num_p = len(wkt_c) - 1
    remove_index = []
    for dup in sorted(list_duplicates(wkt_c)):
        dup_index = dup[1]
        if 0 in dup_index and num_p in dup_index and len(dup_index) == 2:
            pass
            
        elif 0 in dup_index and num_p in dup_index and len(dup_index) > 2:
            dup_index_num = len(dup_index)-1
            for j in range(1, dup_index_num):
                ri = dup_index[j]
                remove_index.append(ri)
                
        else:
            dup_index_num = len(dup_index)-1
            for j in range(dup_index_num):
                ri = dup_index[j]
                remove_index.append(ri)

    re_num = len(remove_index)
    rest_num = len(wkt_c) - re_num - 1
    wall_num = len(wall_c)
    wkt_wall = len(wkt_c) - 1

    wall_f = []
    wkt_f = []
    for p in range(len(wkt_c)):
        if p not in remove_index:
            wkt_u = wkt_c[p]
            wkt_f.append(wkt_u)

            if wall_num == wkt_wall:
                if p < (len(wkt_c)-1):
                    wall_u = wall_c[p]
                    wall_f.append(wall_u)
                

    if wall_num == rest_num:
        wall_ff = wall_c
    else:
        wall_ff = wall_f

    wkt_f = ", ".join(wkt_f)
    wkt_f = "POLYGON ((" + wkt_f + "))"
    
    # print('wall_ff:', wall_ff)
    
    # print("wkt_c:", len(wkt_c))
    # print("remove_index:", remove_index)
    # print("wall_num:", wall_num)
    # print("rest_num:", rest_num)
    
    
    
    return wall_ff, wkt_f



def clean_geometry(wall, wkt):
    # load geometry
    # print('wall:', wall)
    geo = shapely.wkt.loads(wkt)
    
    # move to (0,0)
    geo_centroid = geo.centroid
    translation_vector = (-geo_centroid.x, -geo_centroid.y)
    moved_coords = [(x + translation_vector[0], y + translation_vector[1]) for x, y in geo.exterior.coords]
    moved_geo = shapely.wkt.loads('POLYGON ((' + ', '.join([f'{x} {y}' for x, y in moved_coords]) + '))')

    # if counterclockwise
    if moved_geo.exterior.is_ccw:
        geo_ccw = moved_geo
        wall_ccw = wall
    else:
        geo_ccw = shapely.geometry.polygon.orient(moved_geo, 1)
        
        walltypes = len(list(set(wall)))
        if walltypes == 1:
            wall_ccw = wall
        else:
            wall_ccw = wall[::-1]

    # print('wall_ccw:', wall_ccw)
    # ccw_geo 
    coor_ccw = geo_ccw.exterior.coords
    coor_ccw = list(coor_ccw)
    coor_ccw = coor_ccw[:-1]
        
    coor_ccw_num = len(coor_ccw)
    coor_ccw_xpy_lst = []
    for i in range(coor_ccw_num):
        coor_ccw_x = coor_ccw[i][0]
        coor_ccw_y = coor_ccw[i][1]
        coor_ccw_xpy = coor_ccw_x + coor_ccw_y
        coor_ccw_xpy_lst.append(coor_ccw_xpy)
        
    coor_ccw_xpy_min_index = np.array(coor_ccw_xpy_lst).argmin()
    coor_ccw_sort_index = []
    for i in range(len(coor_ccw_xpy_lst)):
        index_max = len(coor_ccw_xpy_lst) - 1 - coor_ccw_xpy_min_index
        if i <= index_max:
            sort_index = coor_ccw_xpy_min_index + i
        else:
            sort_index =  i - len(coor_ccw_xpy_lst) + coor_ccw_xpy_min_index
        coor_ccw_sort_index.append(sort_index)
        
    
    # print(coor_ccw_sort_index)
    # print(len(coor_ccw_sort_index))
    # print(wall_ccw)
    # print(len(wall_ccw))


    
    coor_sort_lst = []
    wall_sort_lst = []
    for i in range(len(coor_ccw_sort_index)):
        sort_index = coor_ccw_sort_index[i]
        sort_coor = coor_ccw[sort_index]
        sort_wall = wall_ccw[sort_index]
        coor_sort_lst.append(sort_coor)
        wall_sort_lst.append(sort_wall)
        
    geo_s = Polygon(coor_sort_lst)
    wall_s = wall_sort_lst
    return wall_s, geo_s

    
def segments(polyline):
    return list(map(LineString, zip(polyline.coords[:-1], polyline.coords[1:])))




def points4cv(x, y, xmin_abs, ymin_abs, scale):
    points = []
    for j in range(len(x)):
        xp =x[j]
        yp =y[j]

        xp = (xp + xmin_abs +1) * scale
        yp = (yp + ymin_abs +1) * scale
        p = [int(xp), int(yp)]
        points.append(p)

    p_4_cv = np.array(points)
    return p_4_cv




def gridpoints(apa_geo, size):
    latmin, lonmin, latmax, lonmax = apa_geo.bounds
    prep_moved_apa_geo = prep(apa_geo)
    
    # construct a rectangular mesh
    gp = []
    for lat in np.arange(latmin, latmax, size):
        for lon in np.arange(lonmin, lonmax, size):
            gp.append(Point((round(lat,5), round(lon,5))))
    gps = prep_moved_apa_geo.contains(gp)   
    gpf = [i for indx,i in enumerate(gp) if gps[indx] == True]
    grid_points = MultiPoint(gpf)
    return grid_points



def exterior_wall(apa_line, apa_wall):
    apa_wall_O = [i for indx,i in enumerate(segments(apa_line)) if apa_wall[indx] == "O"]
    apa_wall_O = MultiLineString(apa_wall_O)
    return apa_wall_O


def geo_coor(apa_geo):
    apa_coor = apa_geo.exterior.coords
    apa_coor = list(apa_coor)
    return apa_coor




