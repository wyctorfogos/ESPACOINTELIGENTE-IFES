import os
import re
import sys
import cv2
import json
import time
import argparse
import numpy as np
from utils import load_options
from utils import to_labels_array, to_labels_dict
from video_loader import MultipleVideoLoader
from is_wire.core import Logger
from collections import defaultdict, OrderedDict

from is_msgs.image_pb2 import ObjectAnnotations
from is_msgs.image_pb2 import HumanKeypoints as HKP
from google.protobuf.json_format import ParseDict
from itertools import permutations

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from analysis import SkeletonsCoord

colors = list(permutations([0, 255, 85, 170], 3))
links = [(HKP.Value('HEAD'), HKP.Value('NECK')), (HKP.Value('NECK'), HKP.Value('CHEST')),
         (HKP.Value('CHEST'), HKP.Value('RIGHT_HIP')), (HKP.Value('CHEST'), HKP.Value('LEFT_HIP')),
         (HKP.Value('NECK'), HKP.Value('LEFT_SHOULDER')),
         (HKP.Value('LEFT_SHOULDER'), HKP.Value('LEFT_ELBOW')),
         (HKP.Value('LEFT_ELBOW'), HKP.Value('LEFT_WRIST')),
         (HKP.Value('NECK'), HKP.Value('LEFT_HIP')), (HKP.Value('LEFT_HIP'),
                                                      HKP.Value('LEFT_KNEE')),
         (HKP.Value('LEFT_KNEE'), HKP.Value('LEFT_ANKLE')),
         (HKP.Value('NECK'), HKP.Value('RIGHT_SHOULDER')),
         (HKP.Value('RIGHT_SHOULDER'), HKP.Value('RIGHT_ELBOW')),
         (HKP.Value('RIGHT_ELBOW'), HKP.Value('RIGHT_WRIST')),
         (HKP.Value('NECK'), HKP.Value('RIGHT_HIP')),
         (HKP.Value('RIGHT_HIP'), HKP.Value('RIGHT_KNEE')),
         (HKP.Value('RIGHT_KNEE'), HKP.Value('RIGHT_ANKLE')),
         (HKP.Value('NOSE'), HKP.Value('LEFT_EYE')), (HKP.Value('LEFT_EYE'),
                                                      HKP.Value('LEFT_EAR')),
         (HKP.Value('NOSE'), HKP.Value('RIGHT_EYE')),
         (HKP.Value('RIGHT_EYE'), HKP.Value('RIGHT_EAR'))]


def render_skeletons(images, annotations, it, links, colors):
    for cam_id, image in images.items():
        skeletons = ParseDict(annotations[cam_id][it], ObjectAnnotations())
        for ob in skeletons.objects:
            parts = {}
            for part in ob.keypoints:
                parts[part.id] = (int(part.position.x), int(part.position.y))
            for link_parts, color in zip(links, colors):
                begin, end = link_parts
                if begin in parts and end in parts:
                    cv2.line(image, parts[begin], parts[end], color=color, thickness=4)
            for _, center in parts.items():
                cv2.circle(image, center=center, radius=4, color=(255, 255, 255), thickness=-1)


def render_skeletons_3d(ax, skeletons, links, colors):
    skeletons_pb = ParseDict(skeletons, ObjectAnnotations())
    for skeleton in skeletons_pb.objects:
        parts = {}
        for part in skeleton.keypoints:
            parts[part.id] = (part.position.x, part.position.y, part.position.z)
        for link_parts, color in zip(links, colors):
            begin, end = link_parts
            if begin in parts and end in parts:
                x_pair = [parts[begin][0], parts[end][0]]
                y_pair = [parts[begin][1], parts[end][1]]
                z_pair = [parts[begin][2], parts[end][2]]
                ax.plot(
                    x_pair,
                    y_pair,
                    zs=z_pair,
                    linewidth=3,
                    color='#{:02X}{:02X}{:02X}'.format(*reversed(color)))


def skeletons_localization(skeletons, initial_foot):
    skeletons_pb = ParseDict(skeletons, ObjectAnnotations())
    for skeleton in skeletons_pb.objects:
        parts = {}
        for part in skeleton.keypoints:
            parts[part.id] = (part.position.x, part.position.y, part.position.z)
        
        position = (parts[12], parts[15]) # (pe_direito, pe_esquerdo)
        log.info("Foot position {}", position)
        break
    return position

def calc_length(initial, final):
    length = np.sqrt((final[1]-initial[1])**2 + (final[0]-initial[0])**2)
    return length

def calc_step_time(initial, final):
    duration = (final -initial) * (1/7.0)
    return duration

def calc_waistline(skeletons):
    right_hip = None
    left_hip = None
    skeletons_pb = ParseDict(skeletons, ObjectAnnotations())
    for skeleton in skeletons_pb.objects:
        parts = {}
        for part in skeleton.keypoints:
            parts[part.id] = (part.position.x, part.position.y, part.position.z)
            if part.id == 10:
                right_hip = parts[10]
            if part.id == 13:
                left_hip = parts[13]

        if right_hip and left_hip:
            mid_hip = ((right_hip[0] + left_hip[0]) / 2, (right_hip[1] + left_hip[1]) / 2, (right_hip[2] + left_hip[2]) / 2) 
            # log.info("Head: {}", parts[1])
            # log.info("Right hip: {}", right_hip)
            # log.info("Left hip: {}", left_hip)
            return mid_hip
        else:
            mid_hip = None
            return mid_hip
        break

def altura_da_pessoa(skeletons):
    skeletons_pb= ParseDict(skeletons,ObjectAnnotations())
    for skeletons in skeletons_pb.objects:
        parts = {}
        for part in skeletons.keypoints:
            parts[part.id] = (part.position.x, part.position.y, part.position.z)
        altura_da_pessoa=parts[1][2]
        break
    return altura_da_pessoa

def place_images(output_image, images, x_offset=0, y_offset=0):
    w, h = images[0].shape[1], images[0].shape[0]
    output_image[0 + y_offset:h + y_offset, 0 + x_offset:w + x_offset, :] = images[0]
    output_image[0 + y_offset:h + y_offset, w + x_offset:2 * w + x_offset, :] = images[1]
    output_image[h + y_offset:2 * h + y_offset, 0 + x_offset:w + x_offset, :] = images[2]
    output_image[h + y_offset:2 * h + y_offset, w + x_offset:2 * w + x_offset, :] = images[3]



log = Logger(name='WatchVideos')
with open('keymap.json', 'r') as f:
    keymap = json.load(f)
options = load_options(print_options=False)

if not os.path.exists(options.folder):
    log.critical("Folder '{}' doesn't exist", options.folder)

with open('gestures.json', 'r') as f:
    gestures = json.load(f)
    gestures = OrderedDict(sorted(gestures.items(), key=lambda kv: int(kv[0])))

parser = argparse.ArgumentParser(
    description='Utility to capture a sequence of images from multiples cameras')
parser.add_argument('--person', '-p', type=int, required=True, help='ID to identity person')
parser.add_argument('--gesture', '-g', type=int, required=True, help='ID to identity gesture')
args = parser.parse_args()

person_id = args.person
gesture_id = args.gesture
if str(gesture_id) not in gestures:
    log.critical("Invalid GESTURE_ID: {}. \nAvailable gestures: {}", gesture_id,
                 json.dumps(gestures, indent=2))

if person_id < 1 or person_id > 999:
    log.critical("Invalid PERSON_ID: {}. Must be between 1 and 999.", person_id)

log.info("PERSON_ID: {} GESTURE_ID: {}", person_id, gesture_id)

cameras = [int(cam_config.id) for cam_config in options.cameras]
video_files = {
    cam_id: os.path.join(options.folder, 'p{:03d}g{:02d}c{:02d}.mp4'.format(
        person_id, gesture_id, cam_id))
    for cam_id in cameras
}
json_files = {
    cam_id: os.path.join(options.folder, 'p{:03d}g{:02d}c{:02d}_2d.json'.format(
        person_id, gesture_id, cam_id))
    for cam_id in cameras
}
json_locaizations_file = os.path.join(options.folder, 'p{:03d}g{:02d}_3d.json'.format(
    person_id, gesture_id))

if not all(
        map(os.path.exists,
            list(video_files.values()) + list(json_files.values()) + [json_locaizations_file])):
    log.critical('Missing one of video or annotations files from PERSON_ID {} and GESTURE_ID {}',
                 person_id, gesture_id)

size = (2 * options.cameras[0].config.image.resolution.height,
        2 * options.cameras[0].config.image.resolution.width, 3)
full_image = np.zeros(size, dtype=np.uint8)

video_loader = MultipleVideoLoader(video_files)
# load annotations
annotations = {}
for cam_id, filename in json_files.items():
    with open(filename, 'r') as f:
        annotations[cam_id] = json.load(f)['annotations']
#load localizations
with open(json_locaizations_file, 'r') as f:
    localizations = json.load(f)['localizations']

plt.ioff()
fig = plt.figure(figsize=(5,5))
ax=Axes3D(fig)

step_length = []
step_time = []
double_support = []
right_femur, right_shin, left_femur, left_shin = [], [], [], []
update_image = True
it_frames = 0
altura_instantanea = []
quadril = []
while True:
    if video_loader.n_loaded_frames() < video_loader.n_frames():
        update_image = True
    n_loaded_frames = video_loader.load_next()

    if update_image:
        frames = video_loader[it_frames]
        if frames is not None:
            render_skeletons(frames, annotations, it_frames, links, colors)
            frames_list = [frames[cam] for cam in sorted(frames.keys())]
            place_images(full_image, frames_list)

        ax.clear()
        ax.view_init()
        ax.set_xlim(-2.0, 2.0)
        ax.set_xticks(np.arange(-2.0, 2.0, 0.5))
        ax.set_ylim(-3.0, 3.0)
        ax.set_yticks(np.arange(-3.0, 3.0, 0.5))
        ax.set_zlim(-0.25, 1.5)
        ax.set_zticks(np.arange(0, 1.75, 0.5))
        ax.set_xlabel('X', labelpad=20)
        ax.set_ylabel('Y', labelpad=10)
        ax.set_zlabel('Z', labelpad=5)
        render_skeletons_3d(ax, localizations[it_frames], links, colors)


       
        fig.canvas.draw()
        data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        view_3d = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        display_image = cv2.resize(full_image, dsize=(0, 0), fx=0.5, fy=0.5)
        hd, wd, _ = display_image.shape 
        hv, wv, _ = view_3d.shape

        display_image = np.hstack([display_image, 255*np.ones(shape=(hd, wv, 3), dtype=np.uint8)])
        display_image[int((hd - hv) / 2):int((hd + hv) / 2),wd:,:] = view_3d
        cv2.imshow('', display_image)

        update_image = False
        

    key = cv2.waitKey(1)
    if key == -1:
        continue

    if key == ord(keymap['next_frames']):
        it_frames += keymap['big_step']
        it_frames = it_frames if it_frames < n_loaded_frames else 0
        update_image = True

    if key == ord(keymap['next_frame']):
        it_frames += 1
        it_frames = it_frames if it_frames < n_loaded_frames else 0

        altura_instantanea.append(altura_da_pessoa(localizations[it_frames]))
        log.info(" Altura: {}", altura_instantanea)
        
        if SkeletonsCoord.avg_limb_length(localizations[it_frames], links):
            joint = SkeletonsCoord.avg_limb_length(localizations[it_frames], links)
            log.info("Joint: {}", joint)
        else:
            log.warn("Some joint was not found")

        update_image = True

    if key == ord(keymap['previous_frames']):
        it_frames -= keymap['big_step']
        it_frames = n_loaded_frames - 1 if it_frames < 0 else it_frames
        update_image = True

    if key == ord(keymap['previous_frame']):
        it_frames -= 1
        it_frames = n_loaded_frames - 1 if it_frames < 0 else it_frames
        update_image = True

    if key == ord(keymap["left_foot"]):
        initial_foot = "left"
    
    if key == ord(keymap["right_foot"]):
        initial_foot = "right"
    
    if key == ord(keymap["inicial_step"]):
        if initial_foot is None:
            log.warn("You must assign the initial foot")
        elif initial_foot == "left":
            initial_position = skeletons_localization(localizations[it_frames], initial_foot)[1]
            step_initial_frame = it_frames
            log.info("Initial position saved as left foot")
        else:
            initial_position = skeletons_localization(localizations[it_frames], initial_foot)[0]
            step_initial_frame = it_frames
            log.info("Initial position saved as right foot")
    
    if key == ord(keymap["final_step"]):
        if initial_position is None:
            log.warn('You must first assign the initial position')
        elif initial_foot == "left":
            final_position = skeletons_localization(localizations[it_frames], initial_foot)[0]
            step_length.append(calc_length(initial_position, final_position))
            step_final_frame = it_frames
            step_time.append(calc_step_time(step_initial_frame, step_final_frame))

            del initial_position
            del final_position
            del step_initial_frame
            del step_final_frame

            log.info("Step length: {}", step_length)
            log.info("Step duration: {}", step_time)
        else:
            final_position = skeletons_localization(localizations[it_frames], initial_foot)[1]
            step_length.append(calc_length(initial_position, final_position))
            step_final_frame = it_frames
            step_time.append(calc_step_time(step_initial_frame, step_final_frame))

            del initial_position
            del final_position
            del step_initial_frame
            del step_final_frame

            log.info("Step length: {}", step_length)
            log.info("Step duration: {}", step_time)
    
    if key == ord(keymap["double_support"]):
        if len(step_time) % 2 != 0:
            log.warn("You must assign two steps to calculate double support!")
        else:
            print(len(step_time))
            for i in range(0, len(step_time)-1, 2):
                double_support.append(0.2 * (step_time[i] + step_time[i+1]))
            log.info("Double Support: {}", double_support)


    if key == ord(keymap['exit']):
        average_height = sum(altura_instantanea) / len(altura_instantanea)
        log.info("Altura média: {:.2f} m ", average_height )
        average_hip=0

        if len(quadril) !=0:
            average_hip = sum(quadril) / len(quadril)
        else:
            pass

        # log.info("Altura média do quadril: {:.2f}", average_hip)
        sys.exit(0)

log.info('Exiting')