#!/usr/bin/env python
# coding: utf-8

'''
Read training images based on `valid_images.txt` and then detect skeletons.
    
In each image, there should be only 1 person performing one type of action.
Each image is named as 00001.jpg, 00002.jpg, ...

An example of the content of valid_images.txt is shown below:
    
    jump_03-12-09-18-26-176
    58 680

    jump_03-13-11-27-50-720
    65 393

    kick_03-02-12-36-05-185
    54 62
    75 84

The two indices (such as `56 680` in the first `jump` example)
represents the starting index and ending index of a certain action.

Input:
    SRC_IMAGES_DESCRIPTION_TXT
    SRC_IMAGES_FOLDER
    
Output:
    DST_IMAGES_INFO_TXT
    DST_DETECTED_SKELETONS_FOLDER
    DST_VIZ_IMGS_FOLDER
'''

import cv2
import yaml
import time
from tqdm import tqdm

if True:  # Include project path
    import sys
    import os
    ROOT = os.path.dirname(os.path.abspath(__file__))+"/../"
    CURR_PATH = os.path.dirname(os.path.abspath(__file__))+"/"
    sys.path.append(ROOT)

    from utils.lib_openpose import SkeletonDetector
    from utils.lib_tracker import Tracker
    from utils.lib_skeletons_io import ReadValidImagesAndActionTypesByTxt
    import utils.lib_commons as lib_commons


def par(path):  # Pre-Append ROOT to the path if it's not absolute
    return ROOT + path if (path and path[0] != "/") else path

# -- Settings


cfg_all = lib_commons.read_yaml(ROOT + "config/config.yaml")
cfg = cfg_all["s1_get_skeletons_from_training_imgs.py"]

IMG_FILENAME_FORMAT = cfg_all["image_filename_format"]
SKELETON_FILENAME_FORMAT = cfg_all["skeleton_filename_format"]


# Input
if True:
    SRC_IMAGES_DESCRIPTION_TXT = par(cfg["input"]["images_description_txt"])
    SRC_IMAGES_FOLDER = par(cfg["input"]["images_folder"])

# Output
if True:
    # This txt will store image info, such as index, action label, filename, etc.
    # This file is saved but not used.
    DST_IMAGES_INFO_TXT = par(cfg["output"]["images_info_txt"])

    # Each txt will store the skeleton of each image
    DST_DETECTED_SKELETONS_FOLDER = par(
        cfg["output"]["detected_skeletons_folder"])

    # Each image is drawn with the detected skeleton
    DST_VIZ_IMGS_FOLDER = par(cfg["output"]["viz_imgs_folders"])

# Openpose
if True:
    OPENPOSE_MODEL = cfg["openpose"]["model"]
    OPENPOSE_IMG_SIZE = cfg["openpose"]["img_size"]

# -- Functions


class ImageDisplayer(object):
    ''' A simple wrapper of using cv2.imshow to display image '''

    def __init__(self):
        self._window_name = "cv2_display_window"
        cv2.namedWindow(self._window_name)

    def display(self, image, wait_key_ms=1):
        cv2.imshow(self._window_name, image)
        cv2.waitKey(wait_key_ms)

    def __del__(self):
        cv2.destroyWindow(self._window_name)


# -- Main
if __name__ == "__main__":

    # -- Detector
    skeleton_detector = SkeletonDetector(OPENPOSE_MODEL, OPENPOSE_IMG_SIZE)
    multiperson_tracker = Tracker()

    # -- Image reader and displayer
    images_loader = ReadValidImagesAndActionTypesByTxt(
        img_folder=SRC_IMAGES_FOLDER,
        valid_imgs_txt=SRC_IMAGES_DESCRIPTION_TXT,
        img_filename_format=IMG_FILENAME_FORMAT)
    # This file is not used.
    images_loader.save_images_info(filepath=DST_IMAGES_INFO_TXT)
    img_displayer = ImageDisplayer()

    # -- Init output path
    os.makedirs(os.path.dirname(DST_IMAGES_INFO_TXT), exist_ok=True)
    os.makedirs(DST_DETECTED_SKELETONS_FOLDER, exist_ok=True)
    os.makedirs(DST_VIZ_IMGS_FOLDER, exist_ok=True)

    # -- Read images and process
    num_total_images = images_loader.num_images
    print(f"Starting skeleton detection for {num_total_images} images...")
    print(f"This may take 1-4 hours depending on your GPU and number of images.")
    
    # Initialize timing for ETA calculation
    start_time = time.time()
    processing_times = []
    
    # Create progress bar
    progress_bar = tqdm(
        total=num_total_images,
        desc="Detecting skeletons",
        unit="img",
        ncols=100,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
    )
    
    for ith_img in range(num_total_images):
        img_start_time = time.time()

        # -- Read image
        img, str_action_label, img_info = images_loader.read_image()

        # -- Detect
        humans = skeleton_detector.detect(img)

        # -- Draw
        img_disp = img.copy()
        skeleton_detector.draw(img_disp, humans)
        img_displayer.display(img_disp, wait_key_ms=1)

        # -- Get skeleton data and save to file
        skeletons, scale_h = skeleton_detector.humans_to_skels_list(humans)
        dict_id2skeleton = multiperson_tracker.track(
            skeletons)  # dict: (int human id) -> (np.array() skeleton)
        skels_to_save = [img_info + skeleton.tolist()
                         for skeleton in dict_id2skeleton.values()]

        # -- Save result

        # Save skeleton data for training
        filename = SKELETON_FILENAME_FORMAT.format(ith_img)
        lib_commons.save_listlist(
            DST_DETECTED_SKELETONS_FOLDER + filename,
            skels_to_save)

        # Save the visualized image for debug
        filename = IMG_FILENAME_FORMAT.format(ith_img)
        cv2.imwrite(
            DST_VIZ_IMGS_FOLDER + filename,
            img_disp)

        # Calculate processing time and update progress
        img_processing_time = time.time() - img_start_time
        processing_times.append(img_processing_time)
        
        # Update progress bar with additional info
        avg_time_per_img = sum(processing_times) / len(processing_times)
        remaining_images = num_total_images - (ith_img + 1)
        eta_seconds = remaining_images * avg_time_per_img
        
        # Update progress bar description with skeleton count and ETA
        if eta_seconds > 3600:  # More than 1 hour
            eta_str = f"{eta_seconds/3600:.1f}h"
        elif eta_seconds > 60:  # More than 1 minute
            eta_str = f"{eta_seconds/60:.1f}m"
        else:
            eta_str = f"{eta_seconds:.0f}s"
            
        progress_bar.set_description(
            f"Detecting skeletons (found {len(skeletons)} people, ETA: {eta_str})"
        )
        progress_bar.update(1)
        
        # Print detailed progress every 10% or every 50 images (whichever is less frequent)
        report_interval = max(1, min(50, num_total_images // 10))
        if (ith_img + 1) % report_interval == 0 or ith_img == 0:
            elapsed_time = time.time() - start_time
            progress_percent = ((ith_img + 1) / num_total_images) * 100
            tqdm.write(f"Progress: {ith_img+1}/{num_total_images} ({progress_percent:.1f}%) - "
                      f"Avg time/img: {avg_time_per_img:.2f}s - "
                      f"Elapsed: {elapsed_time/60:.1f}m - "
                      f"ETA: {eta_str}")

    progress_bar.close()
    
    # Final timing report
    total_time = time.time() - start_time
    avg_time_final = total_time / num_total_images
    print(f"\n✅ Skeleton detection completed!")
    print(f"📊 Total time: {total_time/60:.1f} minutes")
    print(f"⚡ Average time per image: {avg_time_final:.2f} seconds")
    print(f"🎯 Processed {num_total_images} images")

    print("Program ends")
