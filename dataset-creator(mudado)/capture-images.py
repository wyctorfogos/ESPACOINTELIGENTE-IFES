import re
import os
import sys
import json
import shutil
import argparse
from datetime import datetime as DT
from collections import defaultdict, OrderedDict
from subprocess import Popen, PIPE, STDOUT
import time
import cv2
import numpy as np
from utils import load_options
from is_msgs.image_pb2 import Image
from is_wire.core import Channel, Subscription, Message, Logger


def get_id(topic):
    match_id = re.compile(r'CameraGateway.(\d+).Frame')
    match = match_id.search(msg.topic)
    if not match:
        return None
    else:
        return int(match.group(1))


def place_images(output_image, images):
    w, h = images[0].shape[1], images[0].shape[0]
    output_image[0:h, 0:w, :] = images[0]
    output_image[0:h, w:2 * w, :] = images[1]
    output_image[h:2 * h, 0:w, :] = images[2]
    output_image[h:2 * h, w:2 * w, :] = images[3]


def draw_info_bar(image,
                  text,
                  x,
                  y,
                  background_color=(0, 0, 0),
                  text_color=(255, 255, 255),
                  draw_circle=False):
    fontFace = cv2.FONT_HERSHEY_DUPLEX
    fontScale = 1.0
    thickness = 1
    ((text_width, text_height), _) = cv2.getTextSize(
        text=text, fontFace=fontFace, fontScale=fontScale, thickness=thickness)

    cv2.rectangle(
        display_image,
        pt1=(0, y - text_height),
        pt2=(x + text_width, y),
        color=background_color,
        thickness=cv2.FILLED)
    if draw_circle:
        cv2.circle(
            display_image,
            center=(int(x / 2), int(y - text_height / 2)),
            radius=int(text_height / 3),
            color=(0, 0, 255),
            thickness=cv2.FILLED)

    cv2.putText(
        display_image,
        text=text,
        org=(x, y),
        fontFace=fontFace,
        fontScale=fontScale,
        color=text_color,
        thickness=thickness)


log = Logger(name='Capture')

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
    sys.exit(-1)

if person_id < 1 or person_id > 999:
    log.critical("Invalid PERSON_ID: {}. Must be between 1 and 999.", person_id)
    sys.exit(-1)

log.info("PERSON_ID: {} GESTURE_ID: {}", person_id, gesture_id)

options = load_options(print_options=False)

if not os.path.exists(options.folder):
    os.makedirs(options.folder)

sequence = 'p{:03d}g{:02d}'.format(person_id, gesture_id)
sequence_folder = os.path.join(options.folder, sequence)
if os.path.exists(sequence_folder):
    log.warn(
        'Path to PERSON_ID={} GESTURE_ID={} already exists.\nWould you like to proceed? All data will be deleted! [y/n]',
        person_id, gesture_id)
    key = input()
    if key == 'y':
        shutil.rmtree(sequence_folder)
    elif key == 'n':
        sys.exit(0)
    else:
        log.critical('Invalid command \'{}\', exiting.', key)
        sys.exit(-1)

os.makedirs(sequence_folder)

channel = Channel(options.broker_uri)
subscription = Subscription(channel)
for camera in options.cameras:
    subscription.subscribe('CameraGateway.{}.Frame'.format(camera.id))

size = (2 * options.cameras[0].config.image.resolution.height,
        2 * options.cameras[0].config.image.resolution.width, 3)
full_image = np.zeros(size, dtype=np.uint8)

contador=0

images_data = {}
current_timestamps = {}
timestamps = defaultdict(list)
images = {}
n_sample = 0
display_rate = 2
start_save = False
sequence_saved = False
info_bar_text = "PERSON_ID: {} GESTURE_ID: {} ({})".format(
    person_id, gesture_id, gestures[str(gesture_id)])

while True:
    msg = channel.consume()
    camera = get_id(msg.topic)

    if camera is None:
        continue
    pb_image = msg.unpack(Image)
    if pb_image is None:
        continue
    data = np.fromstring(pb_image.data, dtype=np.uint8)
    images_data[camera] = data
    current_timestamps[camera] = DT.utcfromtimestamp(
        msg.created_at).isoformat()

    if len(images_data) == len(options.cameras):
        # save images
        if start_save and not sequence_saved:
            for camera in options.cameras:
                filename = os.path.join(
                    sequence_folder, 'c{:02d}s{:08d}.jpeg'.format(
                        camera.id, n_sample))
                with open(filename, 'wb') as f:
                    f.write(images_data[camera.id])
                timestamps[camera.id].append(current_timestamps[camera.id])
            n_sample += 1
            log.info('Sample {} saved', n_sample)

        # display images
        if n_sample % display_rate == 0:
            images = [
                cv2.imdecode(data, cv2.IMREAD_COLOR)
                for _, data in images_data.items()
            ]
            place_images(full_image, images)
            display_image = cv2.resize(full_image, (0, 0), fx=0.5, fy=0.5)
            # put recording message
            draw_info_bar(
                display_image,
                info_bar_text,
                x=50,
                y=50,
                draw_circle=start_save and not sequence_saved)

            cv2.imshow('', display_image)
    
            key = cv2.waitKey(1)
            if key == ord('s'):# and contador==0:
                if not start_save:
                    start_save = True
                    contador=1
                
                elif not sequence_saved:
                    timestamps_filename = os.path.join(options.folder, '{}_timestamps.json'.format(sequence))
                    with open(timestamps_filename, 'w') as f:
                        json.dump(timestamps, f, indent=2, sort_keys=True)
                    sequence_saved = True
                    contador=0

            if key == ord('p'):
                start_save=False

            if key == ord('q'):
                if not start_save or sequence_saved:
                    break
        # clear images dict
        images_data = {}
        current_timestamps = {}

log.info("Exiting")
