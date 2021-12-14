import os
from config import *
import io
from PIL import Image
import base64
import numpy as np
import hashlib
import cv2


def resize(image, w, h):
    resized_image = cv2.resize(image, dsize=(w, h), interpolation=cv2.INTER_AREA)
    return resized_image

# RGBA(4채널) -> RGB(3채널)
def rgba_to_rgb(image_rgba):
    image_rgb = Image.new("RGB", image_rgba.size, (255, 255, 255))
    image_rgb.paste(image_rgba, mask=image_rgba.split()[3]) # 3 is the alpha channel
    return image_rgb

# PIL Image(RGB) -> Numpy Array(BGR)
def to_image_np(image):
    if type(image) == np.ndarray: return image

    return np.array(image)[...,::-1]

# Numpy Array(BGR) -> PIL Image(RGB)
def to_image(image_np):
    if type(image_np) != np.ndarray: return image_np

    return Image.fromarray(image_np[...,::-1])

#deprecated
def image_to_numpy_array(image):
    return to_image_np(image)

#deprecated
def numpy_array_to_image(image_np, convert_rgb = False):
    return to_image(image_np)

def encode_base64_np(image_np):
    retval, buffer = cv2.imencode('.jpg', image_np)
    jpg_as_text = base64.b64encode(buffer).decode()
    return jpg_as_text

def decode_base64_np(jpg_as_text):
    jpg_original = base64.b64decode(jpg_as_text)
    jpg_as_np = np.frombuffer(jpg_original, dtype=np.uint8)
    image_np = cv2.imdecode(jpg_as_np, flags=1)
    return image_np

def decode_base64(img_base64):
    imgdata = base64.b64decode(img_base64)
    image = Image.open(io.BytesIO(imgdata))
    return image

# PIL.Image, numpy image 모두 base64로 처리
def encode_base64(image):
    image = to_image(image)

    output_str = io.BytesIO()
    image.save(output_str, "JPEG")
    result = base64.b64encode(output_str.getvalue())
    output_str.close()

    return result.decode()



def encode_base64_from_file(file_path):
    # with open(img_fn, "rb") as f:
    #     data = f.read()
    #     return data.encode("base64")
    with open(file_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()

    return encoded_string


def clone_image(img):
    new_img = Image.new("RGB", img.size, (255,255,255))
    new_img.paste(img, img)

    return new_img


def image_to_bytes(image):
    imgByteArr = io.BytesIO()
    image.save(imgByteArr, format='JPEG')
    imgByteArr = imgByteArr.getvalue()
    return imgByteArr




def crop_image(image_np, box):
       
    x1 = box["x"]
    y1 = box["y"]
    x2 = box["x"] + box["w"]
    y2 = box["y"] + box["h"]

    if x1 < 0: x1=0
    if y1 < 0: y1=0

    return image_np[y1:y2, x1:x2]



# box 정보에 대해 margin 값만큼 확장함
def expand_box(box, margin, img_width, img_height):
    x = box["x"]
    y = box["y"]
    w = box["w"]
    h = box["h"]

    x = x - margin
    x = max(x, 0)
    y = y - margin
    y = max(y, 0)

    w = w + 2*margin
    if x + w > img_width:
        w = img_width - x
    
    h = h + 2*margin
    if y + h > img_height:
        h = img_height - y

    new_box = {
        "x": x, 
        "y": y,
        "w": w,
        "h": h
    }

    return new_box





def imwrite(file_path, img, params=None):
    try: 
        ext = os.path.splitext(file_path)[1]
        result, n = cv2.imencode(ext, img, params)
        if result: 
            with open(file_path, mode='w+b') as f:
                n.tofile(f)
                return True
        else: 
            return False 
    except Exception as e: 
        print(e)
        return False


def imread( file_path ) :
    stream = open( file_path.encode("utf-8") , "rb")
    bytes = bytearray(stream.read())
    np_array = np.asarray(bytes, dtype=np.uint8)
    return cv2.imdecode(np_array , cv2.IMREAD_UNCHANGED)

def read_hangul_path_file( file_path ) :
    return imread(file_path)


# 2개의 이미지 해쉬값을 비교해서 같은 이미지인지 체크함
def get_image_hash(img_path):
    image = Image.open(img_path)  

    m = hashlib.md5()
    with io.BytesIO() as memf:
        image.save(memf, 'PNG')
        data = memf.getvalue()
        m.update(data)

    return m.hexdigest()


def get_masked_image(image_np):
    ROI_LINE = (350, 740)
    y1, y2 = ROI_LINE

    height, width, _ = image_np.shape
    mask = np.zeros(image_np.shape, np.uint8)

    points = [(0, y1), (width, y2), (width, height), (0, width)]
    points = np.array(points, np.int32)
    points = points.reshape((-1, 1, 2))

    mask = cv2.polylines(mask, [points], True, (255, 255, 255), 2)
    mask2 = cv2.fillPoly(mask.copy(), [points], (255, 255, 255))
    mask3 = cv2.fillPoly(mask.copy(), [points], (0, 255, 0))

    show_image = cv2.addWeighted(src1=image_np, alpha=0.8, src2=mask3, beta=0.2, gamma=0)

    roi_image = cv2.bitwise_and(mask2, image_np)
    return roi_image


def draw_image(image, lanes, bboxes, parking_violate_bboxes=None, risks=None, car_count=None):
    
    if len(bboxes) != 0:
        if risks is not None:
            for risk, box in zip(risks, bboxes):
                x, y, w, h = box
                color = (255, 0, 0) if risk > VIOLATE_THRESHOLD else (0, 255, 0)
                cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)
                cv2.putText(image, str(int(risk*100)), (x, y), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5, (255, 255, 0), 2)
        else:
            for box in bboxes:
                x, y, w, h = box
                cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)


    if parking_violate_bboxes is not None and len(parking_violate_bboxes) != 0:
        for box in parking_violate_bboxes:
            x, y, w, h = box
            cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)

    if car_count is not None:
        road_str = ''.join([str(int(car_count[i])) + '/' for i in range(1, len(car_count))])[:-1]
        (x1, y1), (x2, y2) = lanes['cnt_lane'][1]
        cv2.line(image, (x1, y1), (x2, y2), (0, 0, 0), 3)
        cv2.putText(image, road_str, (50, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 3, (0, 0, 0), 2)

    image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
    cv2.imwrite('result.jpg', image)

    return image


if __name__ == "__main__":

    image_path = r"D:\DATA\@car\@test_kolas\@test\test1\N11\23527856_002.jpg"
    image_np = cv2.imread(image_path, cv2.IMREAD_COLOR)
    
    base64_image = encode_base64_np(image_np)
    print(base64_image)

    image_np2 = decode_base64_np(base64_image)
    print(type(image_np2))

    cv2.imshow("car", image_np2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    exit()
    BASE_FOLDER_PATH = r"D:\DATA\@car\car_color\train\0000 white"
    hash_set = set()
    for file in os.listdir(BASE_FOLDER_PATH):
        filepath = os.path.join(BASE_FOLDER_PATH, file)

        image_hash = get_image_hash(filepath)
        print(image_hash[:10], filepath)
        if image_hash in hash_set:
            print("         중복 발견!!", )
        else:
            hash_set.add(image_hash)

    print(len(hash_set))
    exit()
    b64_image = encode("../sample_image/1.jpg")
    # b64_image = convert_image_to_base64("../sample_image/6.jpg").decode()
    # b64_image = "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"
    # b64_image = encode_img(r"D:\DATA\captcha\nice_captcha_1000\1_20190605_095244.jpg").decode()
    print(b64_image)

    image = decode(b64_image)
    image.show()
    # b64_image = encode_img(r"D:\DATA\@car\carplate_data\aocr_train\generated\1\00ah3020.jpg")
    # b64_image = encode_img(r"D:\DATA\@car\carplate_data\aocr_train\1\0009_hyundai1_000333.jpg.jpg")
    # b64_image = convert_image_to_base64_json(r"D:\DATA\captcha\nice_captcha_1000\1_20190605_095239.jpg", None)
    # img = decode_img(b64_image)
    # img.show()
    # print(b64_image)
    # print(type(b64_image.decode()))
    # print(b64_image)
    # exit()

