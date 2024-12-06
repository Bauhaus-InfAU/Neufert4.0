import pandas as pd
import numpy as np
import math
from statistics import mean, stdev
from collections import defaultdict

import shapely
import shapely.wkt
from shapely.geometry import Point, MultiPoint, LineString, MultiLineString, Polygon, LinearRing
from shapely.ops import voronoi_diagram, substring, unary_union, nearest_points
from shapely import affinity
from shapely.prepared import prep

import cv2 as cv


def segments(polyline):
    return list(map(LineString, zip(polyline.coords[:-1], polyline.coords[1:])))


def scale_move_x(x, xmin_abs, scale):
    xn = (x / scale) - 1 - xmin_abs
    return xn

def scale_move_y(y, ymin_abs, scale):
    yn = (y / scale) - 1 - ymin_abs
    return yn

def scale_area(a, scale):
    a = a / (scale**2)
    return a

def scale_perimeter(p, scale):
    p = p / scale
    return p


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


# Dir_S_longestedge, Dir_N_longestedge, Dir_W_longestedge, Dir_E_longestedge, Dir_S_max, Dir_N_max, Dir_W_max, Dir_E_max, Facade_length, Facade_ratio
def wall_direction_ratio(apa_line, apa_wall):
    apa_wall_O = [i for indx,i in enumerate(segments(apa_line)) if apa_wall[indx] == "O"]
    apa_wall_O = MultiLineString(apa_wall_O)

    wall_O_length = []
    wall_O_south = []
    wall_O_east = []
    wall_O_north = []
    wall_O_west = []
    apa_wall_O_num = len(apa_wall_O.geoms)

    if apa_wall_O_num > 0:
        for i in range(apa_wall_O_num):
            wall_seg = apa_wall_O.geoms[i]
            wall_length = wall_seg.length
            south_cos = wall_segment_cosine("south", wall_seg)
            east_cos = wall_segment_cosine("east", wall_seg)
            north_cos = wall_segment_cosine("north", wall_seg)
            west_cos = wall_segment_cosine("west", wall_seg)

            if south_cos < 0:
                south_cos = 0
            if east_cos < 0:
                east_cos = 0
            if north_cos < 0:
                north_cos = 0
            if west_cos < 0:
                west_cos = 0

            wall_O_length.append(wall_length)
            wall_O_south.append(south_cos)
            wall_O_east.append(east_cos)
            wall_O_north.append(north_cos)
            wall_O_west.append(west_cos)


        max_length_index = np.array(wall_O_length).argmax()
        Dir_S_longestedge =  wall_O_south[max_length_index]   
        Dir_N_longestedge =  wall_O_north[max_length_index]   
        Dir_W_longestedge =  wall_O_west[max_length_index]  
        Dir_E_longestedge =  wall_O_east[max_length_index]

        Dir_S_max = max(wall_O_south)
        Dir_N_max = max(wall_O_north)
        Dir_W_max = max(wall_O_west)
        Dir_E_max = max(wall_O_east)

        Facade_length = apa_wall_O.length
        apa_line_length = apa_line.length
        Facade_ratio = Facade_length / apa_line_length
    else:
        Dir_S_longestedge = 0
        Dir_N_longestedge = 0
        Dir_W_longestedge = 0
        Dir_E_longestedge = 0
        Dir_S_max = 0
        Dir_N_max = 0
        Dir_W_max = 0
        Dir_E_max = 0
        Facade_length = 0
        Facade_ratio = 0

    return Dir_S_longestedge, Dir_N_longestedge, Dir_W_longestedge, Dir_E_longestedge, Dir_S_max, Dir_N_max, Dir_W_max, Dir_E_max, Facade_length, Facade_ratio


# apa_geo
def apartment_perimeter(apa_geo):
    perimeter =apa_geo.length
    return perimeter



def apartment_area(apa_geo):
    area =apa_geo.area
    return area


def boundingbox(apa_geo):
    boundingbox = apa_geo.bounds
    return boundingbox


# BBox_width_x, BBox_height_y, Aspect_ratio, Extent, ULC_x, ULC_y, LRC_x, LRC_y
def boundingbox_features(apa_geo):
    # [Aspect_ratio, Extent] ---> https://docs.opencv.org/3.4/d1/d32/tutorial_py_contour_properties.html

    bbox_xy = boundingbox(apa_geo)
    bbox_geo = Polygon([(bbox_xy[0], bbox_xy[1]), (bbox_xy[2], bbox_xy[1]), (bbox_xy[2], bbox_xy[3]), (bbox_xy[0], bbox_xy[3])])
    
    BBox_width_x = bbox_xy[2] - bbox_xy[0]
    BBox_height_y = bbox_xy[3] - bbox_xy[1]
    Aspect_ratio = BBox_width_x / BBox_height_y
    
    bbox_geo_area = bbox_geo.area
    Area = apartment_area(apa_geo)
    Extent = Area / bbox_geo_area
    
    ULC_x = bbox_xy[0]
    ULC_y = bbox_xy[3]
    LRC_x = bbox_xy[2]
    LRC_y = bbox_xy[1]
    
    return BBox_width_x, BBox_height_y, Aspect_ratio, Extent, ULC_x, ULC_y, LRC_x, LRC_y


# Max_diameter
def max_diameter(apa_geo):
    # [Max_diameter] ---> https://www.mvtec.com/doc/halcon/12/en/diameter_region.html
    apa_coor = list(apa_geo.exterior.coords)
    
    pp_dis_lst = []
    for i in apa_coor:
        for j in apa_coor:
            pp_dis = Point(i).distance(Point(j))
            pp_dis_lst.append(pp_dis)

    max_diameter = max(pp_dis_lst)
    return max_diameter



def fractality(apa_geo):
    # [Fractality] ---> https://onlinelibrary.wiley.com/doi/epdf/10.1111/j.1538-4632.2000.tb00419.x
    # Basaraner, M. and Cetinkaya, S. (2017) ‘Performance of shape indices and classification schemes for characterising perceptual shape complexity of building footprints in GIS’, International Journal of Geographical Information Science, 31(10), pp. 1952–1977. doi:10.1080/13658816.2017.1346257. 
    Area = apartment_area(apa_geo)
    Perimeter = apartment_perimeter(apa_geo)

    fractality = 1 - ((math.log(Area) / (2 * math.log(Perimeter))))
    return fractality



def circularity(apa_geo):
    # [Circularity] ---> https://www.mvtec.com/doc/halcon/12/en/circularity.html
    apa_coor = list(apa_geo.exterior.coords)
    op_dis_lst = []
    for i in apa_coor:
        op_dis = Point((0, 0)).distance(Point(i))
        op_dis_lst.append(op_dis)
        
    Max_radius = max(op_dis_lst)
    
    Area = apartment_area(apa_geo)
    
    circularity = Area / ((math.pi) * (Max_radius**2))
    return circularity



def outer_radius(p_4_cv, xmin_abs, ymin_abs, scale):
    # [Outer_radius] ---> https://docs.opencv.org/4.x/d3/dc0/group__imgproc__shape.html#ga8ce13c24081bbc7151e9326f412190f1
    (xmin,ymin),radius = cv.minEnclosingCircle(p_4_cv)
    mini_Enclosing_Cir_x = scale_move_x(xmin, xmin_abs, scale)
    mini_Enclosing_Cir_y = scale_move_y(ymin, ymin_abs, scale)

    mini_Enclosing_Cir_radius = scale_perimeter(radius, scale)
    outer_radius = mini_Enclosing_Cir_radius
    return outer_radius



def inner_radius(apa_geo, apa_line):
    # [Inner_radius] ---> https://www.sthu.org/blog/14-skeleton-offset-topology/index.html
    dis_p = []
    for i in np.arange(0, apa_line.length, 0.1):
        s = substring(apa_line, i, i+0.1)
        dis_p.append(s.boundary.geoms[0])
    mp = MultiPoint(dis_p)

    regions = voronoi_diagram(mp)

    vo_p = []
    for i in range(len(regions.geoms)):
        vo = regions.geoms[i]
        b = list(vo.exterior.coords)
        for j in range(len(b)):
            p = Point(b[j])
            vo_p.append(p)
    vo_p = MultiPoint(vo_p)
    vo_p = unary_union(vo_p)
    vo_p_b = []
    for i in range(len(vo_p.geoms)):
        t_c_p = vo_p.geoms[i]
        pc = apa_geo.contains(t_c_p)
        vo_p_b.append(pc)
    vo_filtered_p = [i for indx,i in enumerate(vo_p.geoms) if vo_p_b[indx] == True]

    vo_d = []
    for i in range(len(vo_filtered_p)):
        c = Point(vo_filtered_p[i])
        d_min = c.distance(apa_line)
        vo_d.append(d_min)

    vo_r_max = max(vo_d)
    vo_r_max_index = vo_d.index(vo_r_max)
    vo_c_max = vo_filtered_p[vo_r_max_index]
    vo_c_max = list(vo_c_max.coords)

    max_Inner_Circle_x = vo_c_max[0][0]
    max_Inner_Circle_y = vo_c_max[0][1]
    max_Inner_Circle_r = vo_r_max
    inner_radius = max_Inner_Circle_r
    return inner_radius



def roundness_features(apa_line):
    # [Dist_mean, Dist_sigma, Roundness] ---> https://www.mvtec.com/doc/halcon/12/en/roundness.html
    rou_p = []
    for i in np.arange(0, apa_line.length, 0.5):
        s = substring(apa_line, i, i+0.5)
        rou_p.append(s.boundary.geoms[0])
    rp = MultiPoint(rou_p)
    
    ro_dis = []
    for i in range(len(rp.geoms)):
        rpp = rp.geoms[i]
        ro = Point(rpp).distance(Point((0, 0)))
        ro_dis.append(ro)
        
    dist_mean = mean(ro_dis) 
#     dist_sigma = stdev(ro_dis) 
         
    dev_lst = []
    for i in ro_dis:
        dev = (i - dist_mean)**2
        dev_lst.append(dev)
    dist_sigma = mean(dev_lst)
    dist_sigma = math.sqrt(dist_sigma)
    roundness = 1 - (dist_sigma/dist_mean)
    
    return dist_mean, dist_sigma, roundness


def compactness(apa_geo):
    # [Compactness] ---> https://fisherzachary.github.io/public/r-output.html
    Area = apartment_area(apa_geo)
    Perimeter = apartment_perimeter(apa_geo)
    
    compactness = (4*(math.pi)) * (Area / (Perimeter**2))
    return compactness



def equivalent_diameter(apa_geo):
    # https://docs.opencv.org/4.x/d1/d32/tutorial_py_contour_properties.html
    Area = apartment_area(apa_geo)
    
    equivalent_diameter = math.sqrt((4 * Area) / math.pi)
    return equivalent_diameter




def shape_membership_index(apa_line):
    # [Shape_membership_index] ---> Basaraner, M. and Cetinkaya, S. (2017) ‘Performance of shape indices and classification schemes for characterising perceptual shape complexity of building footprints in GIS’, International Journal of Geographical Information Science, 31(10), pp. 1952–1977. doi:10.1080/13658816.2017.1346257. 

    line_smi = LineString([(0, 0), (30, 0)])

    numl = 30
    line_rot_degree = 360 / numl
    line_rot = []
    for an in range(numl):
        ang = an*line_rot_degree
        lr = affinity.rotate(line_smi, ang, (0, 0))
        line_rot.append(lr)
    line_rot = MultiLineString(line_rot)
    smip = shapely.intersection(apa_line, line_rot)


    simo_dis = []
    for i in range(len(smip.geoms)):
        sim_p = smip.geoms[i]
        simo = Point(sim_p).distance(Point((0, 0)))
        simo_dis.append(simo)
    sim_r_max = max(simo_dis)

    simo_maxd = []
    for j in simo_dis:
        rmax_d = j / sim_r_max
        simo_maxd.append(rmax_d)

    simo_maxd_mean = mean(simo_maxd) 

    simo_rad = []
    for j in range(len(simo_dis)):
        s = simo_dis[j]    

        if j == (len(simo_dis) - 1):
            nu = 0
        else:
            nu = j+1
        e = simo_dis[nu]

        if s <= e:
            a = np.array([1,s])
            b = np.array([0,s])
            c = np.array([1,e])
        else:
            a = np.array([1,e])
            b = np.array([0,e])
            c = np.array([1,s])

        ba = a - b
        bc = c - b

        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle_rad = np.arccos(cosine_angle)

        simo_rad.append(angle_rad)

    simo_rad_min = min(simo_rad)
    simo_rad_max = max(simo_rad)
    simo_cos = math.cos(simo_rad_max - simo_rad_min)
    shape_membership_index = simo_cos * simo_maxd_mean
    return shape_membership_index


def convexity(p_4_cv, apa_geo, xmin_abs, ymin_abs, scale):
    # [Convexity] ---> Basaraner, M. and Cetinkaya, S. (2017) ‘Performance of shape indices and classification schemes for characterising perceptual shape complexity of building footprints in GIS’, International Journal of Geographical Information Science, 31(10), pp. 1952–1977. doi:10.1080/13658816.2017.1346257. 

    hull = cv.convexHull(p_4_cv)
    hull_x = []
    hull_y = []
    for h in range(len(hull)):
        h_x = hull[h][0][0]
        h_x = scale_move_x(h_x, xmin_abs, scale)
        hull_x.append(h_x)

        h_y = hull[h][0][1]
        h_y = scale_move_y(h_y, ymin_abs, scale)
        hull_y.append(h_y)

    hull_xy = []
    for i in range(len(hull_x)):
        hx = hull_x[i]
        hy = hull_y[i]
        hull_xy.append((hx, hy))
    hull_geo = Polygon(hull_xy)
    Hull_area = hull_geo.area
    
    Area = apartment_area(apa_geo)
    convexity = Area / Hull_area
    return convexity, hull_geo



def rectangle_features(p_4_cv, apa_geo, xmin_abs, ymin_abs, scale):
    # [Rectangularity] ---> Basaraner, M. and Cetinkaya, S. (2017) ‘Performance of shape indices and classification schemes for characterising perceptual shape complexity of building footprints in GIS’, International Journal of Geographical Information Science, 31(10), pp. 1952–1977. doi:10.1080/13658816.2017.1346257. 
    rect = cv.minAreaRect(p_4_cv)
    miniRect_rotation_angle = rect[2]
    box = cv.boxPoints(rect)
    box = np.intp(box)

    miniRect_x = []
    miniRect_y = []
    for b in range(len(box)):

        b_x = box[b][0]
        b_x = scale_move_x(b_x, xmin_abs, scale)
        miniRect_x.append(b_x)

        b_y = box[b][1]
        b_y = scale_move_y(b_y, ymin_abs, scale)
        miniRect_y.append(b_y)

    miniRec_xy = []
    for i in range(len(miniRect_x)):
        minirecx = miniRect_x[i]
        minirecy = miniRect_y[i]
        miniRec_xy.append((minirecx, minirecy))
    miniRect_geo = Polygon(miniRec_xy)
    miniRect_area = miniRect_geo.area
    
    Area = apartment_area(apa_geo)
    rectangularity = Area / miniRect_area
    rect_phi = (miniRect_rotation_angle * math.pi) / 180

    miniRect_line = miniRect_geo.boundary
    miniRect_segments = segments(miniRect_line)

    seg_len = []
    for s in miniRect_segments:
        seg_len.append(s.length)
    rect_width = max(seg_len)
    rect_height = min(seg_len)
    return rectangularity, rect_phi, rect_width, rect_height


def squareness(apa_geo):
    # [Squareness] ---> Basaraner, M. and Cetinkaya, S. (2017) ‘Performance of shape indices and classification schemes for characterising perceptual shape complexity of building footprints in GIS’, International Journal of Geographical Information Science, 31(10), pp. 1952–1977. doi:10.1080/13658816.2017.1346257. 

    Area = apartment_area(apa_geo)
    Perimeter = apartment_perimeter(apa_geo)
    
    squareness = (4*(math.sqrt(Area))) / Perimeter
    return squareness



def moments(apa_geo):
    # https://leancrew.com/all-this/2018/01/python-module-for-section-properties/
    pts = list(apa_geo.exterior.coords)

    if pts[0] != pts[-1]:
        pts = pts + pts[:1]
    x = [ c[0] for c in pts ]
    y = [ c[1] for c in pts ]
    sxx = syy = sxy = 0
    a = apartment_area(apa_geo)
    cx = apa_geo.centroid.x
    cy = apa_geo.centroid.y
    for i in range(len(pts) - 1):
        sxx += (y[i]**2 + y[i]*y[i+1] + y[i+1]**2)*(x[i]*y[i+1] - x[i+1]*y[i])
        syy += (x[i]**2 + x[i]*x[i+1] + x[i+1]**2)*(x[i]*y[i+1] - x[i+1]*y[i])
        sxy += (x[i]*y[i+1] + 2*x[i]*y[i] + 2*x[i+1]*y[i+1] + x[i+1]*y[i])*(x[i]*y[i+1] - x[i+1]*y[i])
    return sxx/12 - a*cy**2, syy/12 - a*cx**2, sxy/24 - a*cx*cy


def moment_index(apa_geo, Convexity, Compactness):
	# https://www.researchgate.net/publication/228557311_A_COMBINED_AUTOMATED_GENERALIZATION_MODEL_BASED_ON_THE_RELATIVE_FORCES_BETWEEN_SPATIAL_OBJECTS
	Ixx, Iyy, Ixy = moments(apa_geo)
	ratio = max(Ixx, Iyy) / min(Ixx, Iyy)
	# Convexity, Hull_geo = convexity(p_4_cv)
	# Compactness = compactness(apa_geo)
	moment_index = (Convexity * Compactness) / ratio
	return moment_index



def ndetour_index(apa_geo, Hull_geo):
    # [nDetour_index] ---> Basaraner, M. and Cetinkaya, S. (2017) ‘Performance of shape indices and classification schemes for characterising perceptual shape complexity of building footprints in GIS’, International Journal of Geographical Information Science, 31(10), pp. 1952–1977. doi:10.1080/13658816.2017.1346257. 

    Hull_line = Hull_geo.boundary
    Hull_length = Hull_line.length
    Area = apartment_area(apa_geo)
    ndetour_index = (2 * math.sqrt(Area * math.pi)) / Hull_length
    return ndetour_index


def ncohesion_index(apa_geo, grid_points):
    # [nCohesion_index] ---> Basaraner, M. and Cetinkaya, S. (2017) ‘Performance of shape indices and classification schemes for characterising perceptual shape complexity of building footprints in GIS’, International Journal of Geographical Information Science, 31(10), pp. 1952–1977. doi:10.1080/13658816.2017.1346257. 

    grid_p = grid_points.geoms
    grid_n = len(grid_p)
    gg_dis_lst = []
    for i in grid_p:
        for j in grid_p:
            gg_dis = Point(i).distance(Point(j))
            gg_dis_lst.append(gg_dis)
            
    Area = apartment_area(apa_geo)
    ncohesion_index = (0.9054 * math.sqrt(Area / math.pi)) / (sum(gg_dis_lst) / (grid_n * (grid_n-1)))
    return ncohesion_index



def nproximity_nspin_index(apa_geo, grid_points):
    grid_p = grid_points.geoms
    
    go_dis_lst = []
    for i in grid_p:
        go_dis = Point(i).distance(Point(0,0))
        go_dis_lst.append(go_dis)

    go_dis_mean = mean(go_dis_lst)
    Area = apartment_area(apa_geo)
    nproximity_index = ((2 / 3) * math.sqrt(Area / math.pi)) / go_dis_mean
    
    nspin_index = (0.5 * (Area / math.pi)) / (go_dis_mean**2)
    
    return nproximity_index, nspin_index



def nexchange_index(apa_geo):
    Area = apartment_area(apa_geo)

    eac_r = math.sqrt(Area / math.pi) 
    eac = Point(0,0).buffer(eac_r)
    eac_inter = apa_geo.intersection(eac)

    if eac_inter.geom_type == "Polygon":
        eac_area = eac_inter.area
    else:
        eacga_lst = []
        for i in range(len(eac_inter.geoms)):
            eacg = eac_inter.geoms[i]
            eacga = eacg.area
            eacga_lst.append(eacga)
        eac_area = sum(eacga_lst)
    nexchange_index = eac_area / Area
    return nexchange_index



def nperimeter_index(apa_geo):
    Area = apartment_area(apa_geo)
    Perimeter = apartment_perimeter(apa_geo)

    nperimeter_index = (2 * math.sqrt(math.pi * Area)) / Perimeter
    return nperimeter_index



def ndepth_index(apa_geo, apa_line, grid_points):
    moved_apa_line = apa_line
    grid_p = grid_points.geoms
    
    nea_len_lst = []
    for i in grid_p:
        nea_line = LineString(nearest_points(moved_apa_line, i))
        nea_len = nea_line.length
        nea_len_lst.append(nea_len)
    nea_len_mean = mean(nea_len_lst)

    Area = apartment_area(apa_geo)
    ndepth_index = (3 * nea_len_mean) / math.sqrt(Area / math.pi)
    return ndepth_index



def ngirth_index(apa_geo, Inner_radius):
    Area = apartment_area(apa_geo)

    ngirth_index = Inner_radius / math.sqrt(Area / math.pi)
    return ngirth_index



def nrange_index(apa_geo, Outer_radius):
    Area = apartment_area(apa_geo)

    nrange_index =  math.sqrt(Area / math.pi) / Outer_radius
    return nrange_index



def ntraversal_index(apa_geo, apa_line):
    rou_p = []
    for i in np.arange(0, apa_line.length, 0.5):
        s = substring(apa_line, i, i+0.5)
        rou_p.append(s.boundary.geoms[0])
    rp = MultiPoint(rou_p)
    
    rp_n = len(rp.geoms)
    bb_dis_lst = []
    for i in rp.geoms:
        for j in rp.geoms:
            bb_dis = Point(i).distance(Point(j))
            bb_dis_lst.append(bb_dis)
    bb_dis_mean = sum(bb_dis_lst) / (rp_n * (rp_n-1))
    
    Area = apartment_area(apa_geo)
    
    ntraversal_index = (4 * (math.sqrt(Area / math.pi) / math.pi)) / bb_dis_mean
    return ntraversal_index

