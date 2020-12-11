import os
import cv2
import numpy as np
from sys import exit
import json
import shutil
import time
from collections import OrderedDict
from is_wire.core import Logger
from video_loader import MultipleVideoLoader

log = Logger(name='DisplayGestures')

with open('gestures.json', 'r') as f:
    gestures = json.load(f)
    gestures = OrderedDict(sorted(gestures.items(), key=lambda kv: int(kv[0])))

vl = MultipleVideoLoader({0: 'gestures_.MOV'})

it = 0
labels = set([0, vl.n_frames() - 1])
n_labels = 2 + 2 * len(gestures)
while True:
    n_loaded = vl.load_next()

    image = np.copy(vl[it][0])
    if it in labels:
        cv2.circle(image, (20, 20), 5, (255, 0, 0), 2, -1)
    cv2.imshow('', image)

    key = cv2.waitKey(1)

    if key == -1:
        continue

    if key == ord('k'):
        it += 1
        it = it if it < n_loaded else 0
        log.info("{}/{}", it, n_loaded)
        print(len(labels), sorted(labels))

    if key == ord('j'):
        it -= 1
        it = n_loaded - 1 if it < 0 else it
        log.info("{}/{}", it, (n_loaded - 1))
        print(len(labels), sorted(labels))

    if key == ord('s'):
        if it == 0 or it == (vl.n_frames() - 1):
            continue
        if it in labels:
            labels.remove(it)
        elif len(labels) < n_labels:
            labels.add(it)
        else:
            log.info("All gestures labeled. Press 'q' to quit and save")

    if key == ord('q'):
        if len(labels) < n_labels:
            log.warn('Only {} of {} gestures were set. Continue spoting!',
                     max(0, (int(len(labels) / 2) - 1)), len(gestures))
            continue

        fps = vl.fps()[0]
        labels_list = sorted(labels)
        spots = []

        g_id = 0
        for begin, end, g_id in zip(labels_list[1:-1:2], labels_list[2:-1:2],
                              gestures.keys()):
            ss, t = begin / fps, (end - begin) / fps
            log.info('{} -> {} | {:.2f} {:.2f}', begin, end, ss, t)
            spots.append({"gesture": g_id, "ss": ss, "t": t})

        if os.path.exists("samples"):
            shutil.rmtree("samples")
        os.makedirs("samples")
        with open("samples/spots.json", 'w') as f:
            json.dump(spots, f, indent=2)

        exit(0)