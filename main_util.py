#-*- coding: utf-8 -*-
import numpy as np
import cv2
import time
import config
                     
        
def get_risks(lanes, bboxes):
    risks = []
    for bbox in bboxes:
        risks.append(get_risk(lanes, bbox))
    return risks


def get_risk(lanes, bbox):
    if not lanes: 
        return 0

    else:
        x, y, w, h = bbox
        risks = []
        image_size = lanes[0].shape
        base = np.zeros(shape=image_size, dtype=np.uint8) 
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
                y1 = y_coord + image_size[0]//20
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
                        a = (y1 - y_coord) / (x1 - x0) # lane의 평균변화율

            dist_from_center = w*abs(a) / (2*abs(a)+2*config.RISK_DETECT_RANGE)
            # 두 바퀴의 중점에서 가까운 모서리까지의 거리
            dist_from_intersect = x_coord - x if a < 0 else x + w - x_coord
            
            risk = max(1 - abs(1 - dist_from_intersect / dist_from_center), 0)
            risks.append(risk)

        return max(risks)


def update_id_road(bboxes, lanes, ids, id_road):
    roads = lanes['road']
    for i, id in enumerate(ids):
        for j, road in enumerate(roads):
            x, y, w, h = bboxes[i]
            if road[y+2*h//3 ,x+w//2] != 0: 
                id_road[id] = j+1
                break

    return id_road


def update(bboxes, lanes, car_count, ids, id_road, id_time, id_is_parking_violate, id_is_count): 
    if len(bboxes) != 0:
        idx = 0
        for i, road_num in enumerate(id_road):
            if i not in ids:
                id_road[i] = 0
            else:
                if id_is_count[i] == 0:
                    cnt_lane = lanes['cnt_lane'][0]
                    x, y, w, h = bboxes[idx]
                    for b, a in np.argwhere(cnt_lane):
                        if x < a < x+w and y < b < y+h:
                            car_count[id_road[i]] += 1
                            id_is_count[i] = 1
                            break
                idx += 1
        id_road = update_id_road(bboxes, lanes, ids, id_road)
        id_time = update_id_time(bboxes, lanes, ids, id_time)
        id_is_parking_violate = check_parking_violate(id_time, id_is_parking_violate)

    return car_count, id_road, id_time, id_is_parking_violate, id_is_count


def update_id_time(bboxes, lanes, ids, id_time):
    parking_lane = lanes['parking_lane']
    idx = 0
    for i in range(len(id_time)):
        if i not in ids:
            id_time[i] = 0
        else:
            if get_risk(parking_lane, bboxes[idx]) == 0:
                id_time[i] = 0
            else:
                if id_time[i] == 0:
                    id_time[i] = time.time()
            idx += 1

    return id_time

                
def check_parking_violate(id_time, id_violate):
    for i, t in enumerate(id_time):
        if t == 0:
            id_violate[i] = 0
            continue
        now = time.time()
        if now - t > 30:
            id_violate[i] = 1

    return id_violate