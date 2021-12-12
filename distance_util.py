from math import nan
from math import sqrt
from numpy import ones,vstack
from numpy.linalg import lstsq

def line_equation(points):
    
    # format of points = [(1,5),(3,4)]

    x_coords, y_coords = zip(*points)
    A = vstack([x_coords,ones(len(x_coords))]).T
    m, c = lstsq(A, y_coords)[0]

    #print("Line Solution is y = {m}x + {c}".format(m=m,c=c))
    # m is slope, c is coeff.
    
    return m, c

def h_distance(slope, constant, box_center):

    if slope != 0:
        box_x, box_y = box_center
        point = (box_y - constant) / slope
        return(abs(point-box_x))

    else:
        return 1

def v_distance(slope, constant, box_center):

    if slope != 0:
        box_x, box_y = box_center
        point = slope*box_x + constant
        return(abs(point-box_y))

    else:
        box_x, box_y = box_center
        return(abs(constant-box_y))

def vector_distance(h_dist, v_dist):
    return sqrt((h_dist**2) + (v_dist**2))

def near_score(sea_cords, road_cords, box_center):

    sea_eqn = line_equation([(sea_cords[0], sea_cords[1]), (sea_cords[2], sea_cords[3])])
    road_eqn = line_equation([(road_cords[0], road_cords[1]), (road_cords[2], road_cords[3])])

    sea_dist = h_distance(sea_eqn[0], sea_eqn[1], box_center)
    road_dist = h_distance(road_eqn[0], road_eqn[1], box_center)

    return(sea_dist/road_dist)

if __name__ == '__main__':
    
    test_point = [(1,5),(3,4)]
    
    aa = line_equation(test_point)

    print(aa)
    print(h_distance(aa[0],aa[1],(10,10)))
    print(near_score( [0,10,3,20] , [10,0,7,5] , (5,5) ))