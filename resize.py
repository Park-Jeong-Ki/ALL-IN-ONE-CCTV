from PIL import Image, ImageFont, ImageDraw
import cv2
import numpy as np
import os
import image_util
BASE_PATH = './video_sample/'
SAVE_PATH = 'resized/'

# ROI 설정
ROI_DEFAULT = (700, 400, 1920, 1080)
roi_rect = ROI_DEFAULT

# ROI 좌표를 CV2 에 맞게 좌표 설정
roi_box = {
    "x": roi_rect[0],
    "y": roi_rect[1],
    "w": roi_rect[2] - roi_rect[0],
    "h": roi_rect[3] - roi_rect[1],
}

# file_list: 이미지 파일 이름 저장된 리스트
file_list = os.listdir(BASE_PATH)

for file in file_list:
    
    # 이미지 읽고
    original_image = cv2.imread(os.path.join(BASE_PATH, file))
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    # ROI로 크롭 하고 다시 이미지 만들고
    frame_roi = image_util.crop_image(original_image, roi_box)
    result = np.asarray(frame_roi)
    result = cv2.cvtColor(frame_roi, cv2.COLOR_RGB2BGR)

    # 이미지 저장
    cv2.imwrite(os.path.join(SAVE_PATH,file[-8:]), result)
    print(os.path.join(SAVE_PATH,file[-8:]))

