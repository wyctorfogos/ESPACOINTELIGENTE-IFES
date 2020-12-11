import os
import sys
import cv2
import json
import time
import socket
import datetime
from collections import defaultdict
from enum import Enum
from is_wire.core import Channel, Subscription, Message, Logger
from is_msgs.image_pb2 import ObjectAnnotations
from utils import load_options, make_pb_image, FrameVideoFetcher
from google.protobuf.json_format import MessageToDict

MIN_REQUESTS = 5
MAX_REQUESTS = 10 
DEADLINE_SEC = 15.0


class State(Enum):
    MAKE_REQUESTS = 1
    RECV_REPLIES = 2
    CHECK_END_OF_VIDEO_AND_SAVE = 3
    CHECK_FOR_TIMEOUTED_REQUESTS = 4
    EXIT = 5


log = Logger(name='Request2dSkeletons')
options = load_options(print_options=False)

if not os.path.exists(options.folder):
    log.critical("Folder '{}' doesn't exist", options.folder)
    sys.exit(-1)

files = next(os.walk(options.folder))[2]  # only files from first folder level
video_files = list(filter(lambda x: x.endswith('.mp4'), files))

pending_videos = []
n_annotations = {}
for video_file in video_files:
    base_name = video_file.split('.')[0]
    annotation_file = '{}_2d.json'.format(base_name)
    annotation_path = os.path.join(options.folder, annotation_file)
    video_path = os.path.join(options.folder, video_file)
    cap = cv2.VideoCapture(video_path)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if os.path.exists(annotation_path):
        # check if all annotations were done
        with open(annotation_path, 'r') as f:
            annotations_data = json.load(f)
        n_annotations_on_file = len(annotations_data['annotations'])
        if n_annotations_on_file == n_frames:
            log.info(
                "Video '{}' already annotated at '{}' with {} annotations",
                video_file, annotations_data['created_at'],
                n_annotations_on_file)
            continue

    pending_videos.append(video_file)
    n_annotations[base_name] = n_frames
if len(pending_videos) == 0:
    log.info("Exiting...")
    sys.exit(-1)

channel = Channel(options.broker_uri)
subscription = Subscription(channel)

requests = {}
annotations_received = defaultdict(dict)
state = State.MAKE_REQUESTS
frame_fetcher = FrameVideoFetcher(
    video_files=pending_videos, base_folder=options.folder)

while True:
    if state == State.MAKE_REQUESTS:
        state = State.RECV_REPLIES
        if len(requests) < MIN_REQUESTS:
            while len(requests) <= MAX_REQUESTS:
                base_name, frame_id, frame = frame_fetcher.next()
                if frame is None:
                    if len(requests) == 0:
                        state = State.EXIT
                    break
                pb_image = make_pb_image(frame)
                msg = Message(content=pb_image, reply_to=subscription)
                msg.timeout = DEADLINE_SEC
                channel.publish(msg, topic='SkeletonsDetector.Detect')
                requests[msg.correlation_id] = {
                    'content': pb_image,
                    'base_name': base_name,
                    'frame_id': frame_id,
                    'requested_at': time.time()
                }
        continue

    elif state == State.RECV_REPLIES:

        try:
            msg = channel.consume(timeout=1.0)
            if msg.status.ok():
                annotations = msg.unpack(ObjectAnnotations)
                cid = msg.correlation_id
                if cid in requests:
                    base_name = requests[cid]['base_name']
                    frame_id = requests[cid]['frame_id']
                    annotations_received[base_name][frame_id] = MessageToDict(
                        annotations,
                        preserving_proto_field_name=True,
                        including_default_value_fields=True)
                    del requests[cid]

            state = State.CHECK_END_OF_VIDEO_AND_SAVE
        except socket.timeout:
            state = State.CHECK_FOR_TIMEOUTED_REQUESTS
        continue

    elif state == State.CHECK_END_OF_VIDEO_AND_SAVE:

        for base_name in list(annotations_received.keys()):
            annotations_dict = annotations_received[base_name]
            if len(annotations_dict) == n_annotations[base_name]:
                output_annotations = {
                    'annotations':
                    [x[1] for x in sorted(annotations_dict.items())],
                    'created_at':
                    datetime.datetime.now().isoformat()
                }
                filename = os.path.join(options.folder,
                                        '{}_2d.json'.format(base_name))
                with open(filename, 'w') as f:
                    json.dump(output_annotations, f, indent=2)
                del annotations_received[base_name]
                log.info('{} has been saved.', filename)

        state = State.CHECK_FOR_TIMEOUTED_REQUESTS
        continue

    elif state == State.CHECK_FOR_TIMEOUTED_REQUESTS:

        new_requests = {}
        for cid in list(requests.keys()):
            request = requests[cid]
            if (request['requested_at'] + DEADLINE_SEC) > time.time():
                continue
            msg = Message(content=request['content'], reply_to=subscription)
            msg.timeout = DEADLINE_SEC
            channel.publish(msg, topic='SkeletonsDetector.Detect')
            new_requests[msg.correlation_id] = {
                'content': request['content'],
                'base_name': request['base_name'],
                'frame_id': request['frame_id'],
                'requested_at': time.time()
            }
            del requests[cid]
            log.warn("Message '{}' timeouted. Sending another request.", cid)

        requests.update(new_requests)
        state = State.MAKE_REQUESTS
        continue

    elif state == State.EXIT:

        log.info("Exiting...")
        sys.exit(-1)

    else:

        state = State.MAKE_REQUESTS
        continue
