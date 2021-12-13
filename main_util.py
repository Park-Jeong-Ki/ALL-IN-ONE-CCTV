#-*- coding: utf-8 -*-
import numpy as np
import cv2
import time
from config import *
                     
        
def get_risks(lanes, bboxes):
    """Return risks of bboxes

    Args:
        lanes (numpy array): lanes interested
        bboxes (numpy array): Bboxes consist of x, y, w, h

    Return:
        list: risk (float) of bboxes
    """

    risks = []
    for bbox in bboxes:
        risks.append(get_risk(lanes, bbox))

    return risks


def update(lanes, bboxes, car_count, ids, id_info): 
    '''Update the number of cars in each road and their information
    
    Args:
        lanes (numpy array): lanes dictionary
        bboxes (numpy array): tracking bboxes
        car_count (numpy array): The number of cars in each road
        ids (list): tracking ids
        id_info (dict): 'road'(int) is the number the vehicle is in, 'time'(float) is how much time it is on parking line
                        'is_parking_violate'(int 0 or 1) is whether it violate parking lane
                        'is_count'(int 0 or 1) is whether it is counted

    Return:
        car_count (numpy array): the number of cars in each road updated
        id_info (dict): tracking id information
    '''

    if len(bboxes) != 0:
        idx = 0
        for i in range(MAX_ID_NUM):
            if i not in ids:
                id_info['road'][i] = 0
            else:
                if id_info['is_count'][i] == 0:
                    cnt_lane = lanes['cnt_lane'][0]
                    x, y, w, h = bboxes[idx]
                    for b, a in np.argwhere(cnt_lane):
                        if x < a < x+w and y < b < y+h:
                            car_count[id_info['road'][i]] += 1
                            id_info['is_count'][i] = 1
                            break
                idx += 1

        id_info['road'] = update_id_road(bboxes, lanes, ids, id_info['road'])
        id_info['time'] = update_id_time(bboxes, lanes, ids, id_info['time'])
        id_info['is_parking_violate'] = check_parking_violate(id_info['time'], id_info['is_parking_violate'])

    return car_count, id_info


def get_parking_violate_box(bboxes, ids, id_info):
    violate_ids = np.argwhere(id_info['is_parking_violate']).flatten()
    parking_bboxes = []
    for i, id in enumerate(ids):
        if id in violate_ids:
            parking_bboxes.append(bboxes[i])

    return parking_bboxes


def get_risk(lanes, bbox): 
    if not lanes: 
        return 0

    else:
        x, y, w, h = bbox
        risks = []
        base = np.zeros(shape=IMAGE_SIZE, dtype=np.uint8)
        cv2.line(base, (x, y+h), (x+w, y+h), 1, 1)

        for lane in lanes:
            intersect = lane * base
            lane_pts = np.argwhere(lane)
            intersect_pts = np.argwhere(intersect) 

            if not len(intersect_pts):
                risks.append(0)
                continue

            else:
                intersect_pt = intersect_pts[len(intersect_pts)//2] 
                y_coord, x_coord = intersect_pt
                y1 = y_coord + IMAGE_SIZE[0]//20
                x0_line = lane_pts[lane_pts[:, 0] == y_coord][:, 1]
                x1_line = lane_pts[lane_pts[:, 0] == y1][:, 1]

                if not x1_line.size:
                    return 0

                else:
                    x0 = x0_line[len(x0_line)//2]
                    x1 = x1_line[len(x1_line)//2]
                    if x1 == x0:
                        risks.append(0)
                        continue
                    else:
                        a = (y1 - y_coord) / (x1 - x0) 

            dist_from_intersect = x_coord - x if a < 0 else x + w - x_coord 
            risk = max(1 - abs(1 - 2*dist_from_intersect * (a + RISK_DETECT_RANGE) / (a * w)), 0) 
            risks.append(risk)

        return max(risks) 



def update_id_time(bboxes, lanes, ids, id_time):
    idx = 0
    for i in range(MAX_ID_NUM):
        if i not in ids:
            id_time[i] = 0
        else:
            if get_risk(lanes['parking_lane'], bboxes[idx]) == 0:
                id_time[i] = 0
            else:
                if id_time[i] == 0:
                    id_time[i] = time.time()
            idx += 1

    return id_time


def update_id_road(bboxes, lanes, ids, id_road):
    roads = lanes['road']
    for i, id in enumerate(ids):
        for j, road in enumerate(roads):
            x, y, w, h = bboxes[i]
            if road[y+2*h//3 ,x+w//2] != 0: 
                id_road[id] = j+1
                break

    return id_road

                
def check_parking_violate(id_time, id_is_parking_violate):
    for i, t in enumerate(id_time):
        if t == 0:
            id_is_parking_violate[i] = 0
            continue
        now = time.time()
        if now - t > PARKING_VIOLATE_THRESHOLD_SECOND:
            id_is_parking_violate[i] = 1

    return id_is_parking_violate