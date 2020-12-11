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

colors = list(permutations([0, 255, 85, 170], 3))
links = [(HKP.Value('HEAD'), HKP.Value('NECK')),
        (HKP.Value('NECK'), HKP.Value('CHEST')),
        (HKP.Value('CHEST'), HKP.Value('RIGHT_HIP')),
        (HKP.Value('CHEST'), HKP.Value('LEFT_HIP')),
        (HKP.Value('NECK'), HKP.Value('LEFT_SHOULDER')),
        (HKP.Value('LEFT_SHOULDER'), HKP.Value('LEFT_ELBOW')),
        (HKP.Value('LEFT_ELBOW'), HKP.Value('LEFT_WRIST')),
        (HKP.Value('NECK'), HKP.Value('LEFT_HIP')),
        (HKP.Value('LEFT_HIP'), HKP.Value('LEFT_KNEE')),
        (HKP.Value('LEFT_KNEE'), HKP.Value('LEFT_ANKLE')),
        (HKP.Value('NECK'), HKP.Value('RIGHT_SHOULDER')),
        (HKP.Value('RIGHT_SHOULDER'), HKP.Value('RIGHT_ELBOW')),
        (HKP.Value('RIGHT_ELBOW'), HKP.Value('RIGHT_WRIST')),
        (HKP.Value('NECK'), HKP.Value('RIGHT_HIP')),
        (HKP.Value('RIGHT_HIP'), HKP.Value('RIGHT_KNEE')),
        (HKP.Value('RIGHT_KNEE'), HKP.Value('RIGHT_ANKLE')),
        (HKP.Value('NOSE'), HKP.Value('LEFT_EYE')),
        (HKP.Value('LEFT_EYE'), HKP.Value('LEFT_EAR')),
        (HKP.Value('NOSE'), HKP.Value('RIGHT_EYE')),
        (HKP.Value('RIGHT_EYE'), HKP.Value('RIGHT_EAR'))]

def render_skeletons(images, annotations, it, colors, links):
    for cam_id, image in images.items():
        skeletons = ParseDict(annotations[cam_id][it], ObjectAnnotations())
        for ob in skeletons.objects:
            parts = {}
            for part in ob.keypoints:
                parts[part.id] = (int(part.position.x), int(part.position.y))
            for link_parts, color in zip(links, colors):
                begin, end = link_parts
                if begin in parts and end in parts:
                    cv2.line(
                        image,
                        parts[begin],
                        parts[end],
                        color=color,
                        thickness=4)
            for _, center in parts.items():
                cv2.circle(
                    image,
                    center=center,
                    radius=4,
                    color=(255, 255, 255),
                    thickness=-1)


def place_images(output_image, images, x_offset=0, y_offset=0):
    w, h = images[0].shape[1], images[0].shape[0]
    output_image[0 + y_offset:h + y_offset, 0 + x_offset:w +
                 x_offset, :] = images[0]
    output_image[0 + y_offset:h + y_offset, w + x_offset:2 * w +
                 x_offset, :] = images[1]
    output_image[h + y_offset:2 * h + y_offset, 0 + x_offset:w +
                 x_offset, :] = images[2]
    output_image[h + y_offset:2 * h + y_offset, w + x_offset:2 * w +
                 x_offset, :] = images[3]


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
    description='Utility to capture a sequence of images from multiples cameras'
)
parser.add_argument(
    '--person', '-p', type=int, required=True, help='ID to identity person')
parser.add_argument(
    '--gesture', '-g', type=int, required=True, help='ID to identity gesture')
args = parser.parse_args()

person_id = args.person
gesture_id = args.gesture
if str(gesture_id) not in gestures:
    log.critical("Invalid GESTURE_ID: {}. \nAvailable gestures: {}",
                 gesture_id, json.dumps(gestures, indent=2))

if person_id < 1 or person_id > 999:
    log.critical("Invalid PERSON_ID: {}. Must be between 1 and 999.",
                 person_id)

log.info("PERSON_ID: {} GESTURE_ID: {}", person_id, gesture_id)

cameras = [int(cam_config.id) for cam_config in options.cameras]
video_files = {
    cam_id: os.path.join(
        options.folder, 'p{:03d}g{:02d}c{:02d}.mp4'.format(
            person_id, gesture_id, cam_id))
    for cam_id in cameras
}
json_files = {
    cam_id: os.path.join(
        options.folder, 'p{:03d}g{:02d}c{:02d}_2d.json'.format(
            person_id, gesture_id, cam_id))
    for cam_id in cameras
}
if not all(
        map(os.path.exists,
            list(video_files.values()) + list(json_files.values()))):
    log.critical(
        'Missing one of video or annotations files from PERSON_ID {} and GESTURE_ID {}',
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

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out0 = cv2.VideoWriter('cam0.avi', fourcc, 7.0, (1288,728))
out1 = cv2.VideoWriter('cam1.avi', fourcc, 7.0, (1288,728))
out2 = cv2.VideoWriter('cam2.avi', fourcc, 7.0, (1288,728))
out3 = cv2.VideoWriter('cam3.avi', fourcc, 7.0, (1288,728))

update_image = True
it_frames = 0
while True:
    n_loaded_frames = video_loader.load_next()
    frames = video_loader[it_frames]
    if frames is None:
        break
    
    render_skeletons(frames, annotations, it_frames, colors, links)
    frames_list = [frames[cam] for cam in sorted(frames.keys())]
    it_frames += 1

    out0.write(frames_list[0])
    out1.write(frames_list[1])
    out2.write(frames_list[2])
    out3.write(frames_list[3])

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
out0.release()
out1.release()
out2.release()
out3.release()
log.info('Exiting')
