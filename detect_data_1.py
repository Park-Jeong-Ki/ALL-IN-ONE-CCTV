import tensorflow as tf
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

from absl import app, flags
from absl.flags import FLAGS
import core.utils as utils
from core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
import cv2
import numpy as np
from core.config import cfg
import main_util as mu
from get_lane import get_lanes
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from config import *

# deep sort imports
from deep_sort import preprocessing, nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet


flags.DEFINE_string('framework', 'tf', '(tf, tflite, trt')
flags.DEFINE_string('weights', './checkpoints/yolov4-608',
                    'path to weights file')
flags.DEFINE_integer('size', 608, 'resize images to')
flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
flags.DEFINE_string('lane', 'lane.json', 'path of lane')             
flags.DEFINE_string('image', './resized', 'path of image')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_string('output', 'result.jpg', 'path to output image')
flags.DEFINE_float('iou', 0.45, 'iou threshold')
flags.DEFINE_float('score', 0.5, 'score threshold')
flags.DEFINE_string('video', './DEMO.avi', 'path to input video or set to 0 for webcam')


def get_tracking_list (tracker):
    track_id_list = []
    track_bboxes_list=[]

    for track in tracker.tracks:
        if not track.is_confirmed() or track.time_since_update > 1:
            continue 

        track_id = int(track.track_id)
        track_id_list.append(track_id)
        bbox = track.to_tlbr()
        crop_box = (int(bbox[0]), int(bbox[1]), (int(bbox[2])-int(bbox[0])), (int(bbox[3])-int(bbox[1]))) # x y w h
        track_bboxes_list.append(crop_box) 

    return track_id_list, track_bboxes_list


def draw_image(image, cnt_points, bboxes, risks, car_count, parking_bboxes,track_id_list,track_bboxes_array):

    if len(bboxes) != 0:
        for risk, box in zip(risks, bboxes):
        # YOLO BBOX 그림
            color = (255, 0, 0) if risk > 0.25 else (0, 255, 0)
            if int(box[2]) <= 220:
                cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]+ int(box[0])), int(box[3])+int(box[1])), color, 2)
            #cv2.putText(image, str(int(risk*100)), (x, y), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5, (255, 255, 0), 2)

    if len(parking_bboxes) != 0:
        for box in parking_bboxes:
            x, y, w, h = box
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

    road_str = ''.join([str(int(car_count[i])) + '/' for i in range(1, len(car_count))])[:-1]

    (x1, y1), (x2, y2) = cnt_points
    cv2.line(image, (x1, y1), (x2, y2), (0, 0, 0), 3)
    cv2.putText(image, road_str, (50, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 3, (0, 0, 0), 2)

    for track_id,box in zip(track_id_list,track_bboxes_array):
        if track_id != 15:
            cv2.putText(image, str(track_id), (int(box[0]), int(box[1])), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 255, 0), 2)
    image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)


    
    return image

def main(_argv):
    # 1-1. create instance of Deep SORT
    # Definition of the parameters
    max_cosine_distance = 0.4
    nn_budget = None
    nms_max_overlap = 1.0
    
    # initialize deep sort
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    # calculate cosine distance metric
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    # initialize tracker
    tracker = Tracker(metric)

    
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    input_size = FLAGS.size
    image_folder = FLAGS.image
    video_path = FLAGS.video
    image_list = sorted(os.listdir(image_folder))
    lanes = get_lanes(FLAGS.lane)
    

    print('frame_size={}'.format(FRAME_SIZE))
    fps = 15
    out = None
    codec = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(video_path, codec, fps, FRAME_SIZE)
    

    # 1-2. load tflite model if flag is set
    if FLAGS.framework == 'tf':
        saved_model_loaded = tf.saved_model.load(FLAGS.weights, tags=[tag_constants.SERVING])
        infer = saved_model_loaded.signatures['serving_default']

    car_count = np.zeros(shape=len(lanes['road'])+1)
    id_road = np.zeros(shape=1000, dtype=np.uint8)
    id_time = np.zeros(shape=1000)
    id_is_parking_violate = np.zeros(shape=1000, dtype=np.uint8)
    id_is_count = np.zeros(shape=1000, dtype=np.uint8)

    for image in image_list:
        if image == '.DS_Store':
            continue
        print(image)
        image_path = os.path.join(image_folder, image)
        original_image = cv2.imread(image_path)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

        # image_data = utils.image_preprocess(np.copy(original_image), [input_size, input_size])
        image_data = cv2.resize(original_image, (input_size, input_size))
        image_data = image_data / 255.
        image_data = image_data[np.newaxis, ...].astype(np.float32)

        # run detections on tflite if flag is set
        if FLAGS.framework == 'tf':
            # YOLO V4 inference
            batch_data = tf.constant(image_data)
            pred_bbox = infer(batch_data)
            for key, value in pred_bbox.items():
                boxes = value[:, :, 0:4]
                pred_conf = value[:, :, 4:]

        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=FLAGS.iou,
            score_threshold=FLAGS.score
        )

        # convert data to numpy arrays and slice out unused elements
        num_objects = valid_detections.numpy()[0]
        bboxes = boxes.numpy()[0]
        bboxes = bboxes[0:int(num_objects)]
        scores = scores.numpy()[0]
        scores = scores[0:int(num_objects)]
        classes = classes.numpy()[0]
        classes = classes[0:int(num_objects)]

        original_h, original_w, _ = original_image.shape
        bboxes = utils.format_boxes(bboxes, original_h, original_w)
        pred_bbox = [bboxes, scores, classes, num_objects]

        class_names = utils.read_class_names(cfg.YOLO.CLASSES)

        # custom allowed classes (uncomment line below to customize tracker for only people)
        allowed_classes = ['car', 'truck', 'bus', 'motorcycle']

        # loop through objects and use class index to get class name, allow only classes in allowed_classes list
        names = []
        deleted_indx = []
        for i in range(num_objects):
            class_indx = int(classes[i])
            class_name = class_names[class_indx]
            # print(class_name)
            # print(scores[i])
            
            bbox = bboxes[i]

            if class_name not in allowed_classes:
                deleted_indx.append(i)
            elif bbox[0] < 1 or bbox[1] < 1:
                # bbox의 x, y 좌표가 양쪽 끝에 걸린 경우 제외 (화면안으로 진입중인 차량 제외)
                deleted_indx.append(i)
            else:
                names.append(class_name)
        names = np.array(names)

        bboxes = np.delete(bboxes, deleted_indx, axis=0)
        scores = np.delete(scores, deleted_indx, axis=0)


        bboxes = np.array(bboxes, dtype=np.uint16)
        features = encoder(original_image, bboxes)
        detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(bboxes, scores, names, features)]

        # run non-maxima supression
        boxs = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        classes = np.array([d.class_name for d in detections])
        indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]           


        # Call the tracker
        tracker.predict()
        tracker.update(detections)
        track_id_list, track_bboxes_list = get_tracking_list(tracker)
        risks = mu.get_risks(lanes['solid_lane'], bboxes)

        car_count, id_road, id_time, id_is_parking_violate, id_is_count = \
            mu.update(track_bboxes_list, lanes, car_count, track_id_list, id_road, id_time, id_is_parking_violate, id_is_count)
        track_id_list, track_bboxes_array = np.array(track_id_list), np.array(track_bboxes_list)

        
        violate_ids = np.argwhere(id_is_parking_violate).flatten()
        parking_bboxes = []
        
        for i, id in enumerate(track_id_list):
            if id in violate_ids:
                parking_bboxes.append(track_bboxes_array[i])

        image = draw_image(original_image, lanes['cnt_lane'][1], bboxes, risks, car_count, parking_bboxes,track_id_list,track_bboxes_array)

        cv2.imshow("Output Video", image)
        cv2.imwrite('result.jpg', image)
        out.write(image)

        # q 누르면 종료
        if cv2.waitKey(1) == ord('q'):
            break

    out.release()
    cv2.destroyAllWindows()
    print('종료')

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
