import numpy as np
import cv2
import sys, os, time, argparse, logging
import simplejson
import argparse
import math

import utils.lib_images_io as myio
from utils.lib_plot import draw_action_result as drawActionResult
import utils.lib_commons as myfunc
import utils.lib_feature_proc as myproc
from utils.lib_classifier import ClassifierOnlineTest
from utils.lib_classifier import *  # Import sklearn related libraries
import tensorflow as tf
CURR_PATH = os.path.dirname(os.path.abspath(__file__)) + "/"
DRAW_FPS = True

# Enable GPU memory growth for TensorFlow 2.x
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU memory growth is enabled.")
    except RuntimeError as e:
        print(f"Error enabling GPU memory growth: {e}")

# INPUTS ==============================================================
def parse_input_method():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source", required=False, default="webcam", choices=["webcam", "folder", "txtscript"]
    )
    return parser.parse_args().source


arg_input = parse_input_method()
FROM_WEBCAM = arg_input == "webcam"
FROM_TXTSCRIPT = arg_input == "txtscript"
FROM_FOLDER = arg_input == "folder"

# PATHS and SETTINGS =================================
if FROM_WEBCAM:
    folder_suffix = "3"
    DO_INFER_ACTIONS = True
    SAVE_RESULTANT_SKELETON_TO_TXT_AND_IMAGE = False
    image_size = "432x368"
    OpenPose_MODEL = "mobilenet_thin"

elif FROM_FOLDER:
    folder_suffix = "4"
    DO_INFER_ACTIONS = True
    SAVE_RESULTANT_SKELETON_TO_TXT_AND_IMAGE = True

    def set_source_images_from_folder():
        return CURR_PATH + "../data_test/apple/", 1

    SRC_IMAGE_FOLDER, SKIP_NUM_IMAGES = set_source_images_from_folder()
    folder_suffix += SRC_IMAGE_FOLDER.split("/")[-2]  # plus folder name
    image_size = "240x208"
    OpenPose_MODEL = "cmu"

elif FROM_TXTSCRIPT:
    folder_suffix = "5"
    DO_INFER_ACTIONS = False
    SAVE_RESULTANT_SKELETON_TO_TXT_AND_IMAGE = True
    SRC_IMAGE_FOLDER = CURR_PATH + "../data/source_images3/"
    VALID_IMAGES_TXT = "valid_images.txt"
    image_size = "432x368"
    OpenPose_MODEL = "cmu"

if DO_INFER_ACTIONS:
    LOAD_MODEL_PATH = CURR_PATH + "../model/trained_classifier.pickle"
    action_labels = ["jump", "kick", "punch", "run", "sit", "squat", "stand", "walk", "wave"]

if SAVE_RESULTANT_SKELETON_TO_TXT_AND_IMAGE:
    SKELETON_FOLDER = CURR_PATH + "skeleton_data/"
    SAVE_DETECTED_SKELETON_TO = CURR_PATH + f"skeleton_data/skeletons{folder_suffix}/"
    SAVE_DETECTED_SKELETON_IMAGES_TO = CURR_PATH + f"skeleton_data/skeletons{folder_suffix}_images/"
    SAVE_IMAGES_INFO_TO = CURR_PATH + f"skeleton_data/images_info{folder_suffix}.txt"

    os.makedirs(SKELETON_FOLDER, exist_ok=True)
    os.makedirs(SAVE_DETECTED_SKELETON_TO, exist_ok=True)
    os.makedirs(SAVE_DETECTED_SKELETON_IMAGES_TO, exist_ok=True)

# OpenPose include files and configs ==============================================================

sys.path.append(CURR_PATH + "githubs/tf-pose-estimation")
from tf_pose.networks import get_graph_path, model_wh
from tf_pose.estimator import TfPoseEstimator
from tf_pose import common

logger = logging.getLogger("TfPoseEstimator")
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s"
)
ch.setFormatter(formatter)
logger.addHandler(ch)

# Openpose Human Pose Detection ========================================
class SkeletonDetector:
    def __init__(self, model=None, image_size=None):
        self.model = model or "mobilenet_thin"
        self.image_size = image_size or "432x368"
        self.resize_out_ratio = 4.0
        w, h = model_wh(self.image_size)
        self.e = TfPoseEstimator(
            get_graph_path(self.model),
            target_size=(w, h) if w > 0 and h > 0 else (432, 368),
        )
        self.fps_time = time.time()

    def detect(self, image):
        humans = self.e.inference(
            image, resize_to_default=(self.image_size != "0x0"), upsample_size=self.resize_out_ratio
        )
        return humans

    def draw(self, img_disp, humans):
        img_disp = TfPoseEstimator.draw_humans(img_disp, humans, imgcopy=False)
        if DRAW_FPS:
            cv2.putText(
                img_disp,
                f"fps = {1.0 / (time.time() - self.fps_time):.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )
        self.fps_time = time.time()

    def humans_to_skelsList(self, humans, scale_y=None):
        if scale_y is None:
            scale_y = 1.0
        skeletons = []
        for human in humans:
            skeleton = [0] * 36
            for i, body_part in human.body_parts.items():
                idx = body_part.part_idx
                skeleton[2 * idx] = body_part.x
                skeleton[2 * idx + 1] = body_part.y * scale_y
            skeletons.append(skeleton)
        return skeletons


def add_white_region_to_left_of_image(image_disp):
    r, c, d = image_disp.shape
    blank = 255 + np.zeros((r, int(c / 4), d), np.uint8)
    image_disp = np.hstack((blank, image_disp))
    return image_disp


def remove_skeletons_with_few_joints(skeletons):
    good_skeletons = []
    for skeleton in skeletons:
        px = skeleton[2 : 2 + 13 * 2 : 2]
        py = skeleton[3 : 2 + 13 * 2 : 2]
        num_valid_joints = len([x for x in px if x != 0])
        num_leg_joints = len([x for x in px[-6:] if x != 0])
        total_size = max(py) - min(py)
        if num_valid_joints >= 5 and total_size >= 0.1 and num_leg_joints >= 2:
            good_skeletons.append(skeleton)
    return good_skeletons


class MultiPersonClassifier:
    def __init__(self, LOAD_MODEL_PATH, action_labels):
        self.create_classifier = lambda human_id: ClassifierOnlineTest(
            LOAD_MODEL_PATH, action_types=action_labels, human_id=human_id
        )
        self.dict_id2clf = {}

    def classify(self, dict_id2skeleton):
        old_ids = set(self.dict_id2clf)
        cur_ids = set(dict_id2skeleton)
        humans_not_in_view = list(old_ids - cur_ids)
        for human in humans_not_in_view:
            del self.dict_id2clf[human]

        id2label = {}
        for id, skeleton in dict_id2skeleton.items():
            if id not in self.dict_id2clf:
                self.dict_id2clf[id] = self.create_classifier(id)

            classifier = self.dict_id2clf[id]
            id2label[id] = classifier.predict(skeleton)

        return id2label

    def get(self, id):
        if len(self.dict_id2clf) == 0:
            return None
        if id == "min":
            id = min(self.dict_id2clf.keys())
        return self.dict_id2clf[id]


if __name__ == "__main__":
    my_detector = SkeletonDetector(OpenPose_MODEL, image_size)

    if FROM_WEBCAM:
        images_loader = myio.ReadFromWebcam()

    elif FROM_FOLDER:
        images_loader = myio.ReadFromFolder(SRC_IMAGE_FOLDER, SKIP_NUM_IMAGES)

    elif FROM_TXTSCRIPT:
        #images_loader = myio.DataLoader_txtscript(SRC_IMAGE_FOLDER, VALID_IMAGES_TXT)
        #images_loader.save_images_info(path=SAVE_IMAGES_INFO_TO)
        pass

    if DO_INFER_ACTIONS:
        multipeople_classifier = MultiPersonClassifier(LOAD_MODEL_PATH, action_labels)
    multiperson_tracker = myfunc.Tracker()

    ith_img = 1
    while ith_img <= images_loader.num_images:
        img, img_action_type, img_info = images_loader.load_next_image()
        image_disp = img.copy()

        print("\n\n========================================")
        print(f"\nProcessing {ith_img}/{images_loader.num_images}th image\n")

        humans = my_detector.detect(img)
        skeletons, scale_y = my_detector.humans_to_skelsList(humans)
        skeletons = remove_skeletons_with_few_joints(skeletons)

        dict_id2skeleton = multiperson_tracker.track(skeletons)

        if len(dict_id2skeleton):
            if DO_INFER_ACTIONS:
                min_id = min(dict_id2skeleton.keys())
                dict_id2label = multipeople_classifier.classify(dict_id2skeleton)
                print("predicted label is :", dict_id2label[min_id])
            else:
                min_id = min(dict_id2skeleton.keys())
                dict_id2skeleton = {min_id: dict_id2skeleton[min_id]}
                dict_id2label = {min_id: img_action_type}
                print("Ground_truth label is :", dict_id2label[min_id])

        my_detector.draw(image_disp, humans)

        if len(dict_id2skeleton):
            for id, label in dict_id2label.items():
                skeleton = dict_id2skeleton[id]
                skeleton[1::2] = skeleton[1::2] / scale_y
                drawActionResult(image_disp, id, skeleton, label)

        image_disp = add_white_region_to_left_of_image(image_disp)

        if DO_INFER_ACTIONS and len(dict_id2skeleton):
            multipeople_classifier.get(id="min").draw_scores_onto_image(image_disp)

        if SAVE_RESULTANT_SKELETON_TO_TXT_AND_IMAGE:
            ids = sorted(dict_id2skeleton.keys())
            skel_to_save = [img_info + dict_id2skeleton[id].tolist() for id in ids]

            myio.save_skeletons(
                SAVE_DETECTED_SKELETON_TO + myfunc.int2str(ith_img, 5) + ".txt", skel_to_save
            )
            cv2.imwrite(
                SAVE_DETECTED_SKELETON_IMAGES_TO + myfunc.int2str(ith_img, 5) + ".png", image_disp
            )

            if FROM_TXTSCRIPT or FROM_WEBCAM:
                cv2.imwrite(
                    SAVE_DETECTED_SKELETON_IMAGES_TO
                    + myfunc.int2str(ith_img, 5)
                    + "_src.png",
                    img,
                )

        if 1:
            if ith_img == 1:
                window_name = "action_recognition"
                cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
            cv2.imshow(window_name, image_disp)
            q = cv2.waitKey(1)
            if q != -1 and chr(q) == "q":
                break

        print("\n")
        ith_img += 1
