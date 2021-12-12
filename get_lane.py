from json import load
import numpy as np
import cv2


def get_lane_map(image_shape, polygon_points):
    label_map = np.zeros(image_shape)
    cv2.fillPoly(label_map, [polygon_points], 1)
    return label_map


def get_lanes(path):
    road = {}
    lanes = {
        'solid_lane':[],
        'dashed_lane':[],
        'parking_lane':[],
        'stop_lane':[]
    }

    with open(path) as fp:
        json = load(fp)
    image_shape = json['imageHeight'], json['imageWidth']

    for obj in json['shapes']:
        pts = np.array(obj['points'], dtype=np.int32)
        if obj['label'].startswith('road'):
            road[obj['label'][-1]] = get_lane_map(image_shape, pts)
        else:
            lanes[obj['label']].append(get_lane_map(image_shape, pts))

    lanes['road'] = [road[k] for k in sorted(road)]

    return lanes