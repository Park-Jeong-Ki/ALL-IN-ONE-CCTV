import cv2
import numpy as np
import matplotlib.pyplot as plt

import image_util


#initialize color map
cmap = plt.get_cmap('tab20b')
colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

# 유효한 frame_index를 지정함 (return False 되는 frame_index는 처리안하고 SKIP함)
def on_frame_valid(frame_index):
    if frame_index % 5 == 0:
        return True
    else:
        return False


# 매 프레임마다 인식 결과를 보내고 결과 화면을 리턴함
def on_frame_result(frame_index, frame_image, result_obj_list):

    # 1. 결과 score 계산
    score = 0

    # 2. 결과 화면 구성
    for item in result_obj_list:
        track_id = item["track_id"]
        action = item["action"]
        box = item["box"]

        color = colors[track_id % len(colors)]
        color = [i * 255 for i in color]

        title = f"{track_id}|{action}|{score}"

        cv2.rectangle(frame_image, (box["x"], box["y"]), (box["x"] + box["w"], box["y"] + box["h"]), color, 2)
        cv2.rectangle(frame_image, (box["x"], box["y"]-30), (box["x"]+(len(title)*17), box["y"]), color, -1)
        
        cv2.putText(frame_image, title, (box["x"], box["y"]-10), 0, 0.75, (255,255,255), 2)

