import os
import re
import sys
import cv2
import json
import time
import argparse
import numpy as np
import statistics
import pika
#import mqtt
from utils import load_options
from utils import to_labels_array, to_labels_dict
from video_loader import MultipleVideoLoader
from is_wire.core import Logger
from collections import defaultdict, OrderedDict
from utils import get_np_image
#from PIL import ImageGrab
from is_msgs.image_pb2 import ObjectAnnotations
from is_msgs.image_pb2 import HumanKeypoints as HKP
from google.protobuf.json_format import ParseDict
from itertools import permutations
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from analysis import SkeletonsCoord

#import pyscreenshot as ImageGrab

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
        deteccoes = 0 # Detections in each frame
        skeletons = ParseDict(annotations[cam_id][it], ObjectAnnotations())
        for ob in skeletons.objects:
            parts = {}
            for part in ob.keypoints:
                deteccoes += 1
                juntas[cam_id] += 1
                parts[part.id] = (int(part.position.x), int(part.position.y))
            for link_parts, color in zip(links, colors):
                begin, end = link_parts
                if begin in parts and end in parts:
                    cv2.line(image, parts[begin], parts[end], color=color, thickness=4)
            for _, center in parts.items():
                cv2.circle(image, center=center, radius=4, color=(255, 255, 255), thickness=-1)

        if deteccoes < 10:
            juntas[cam_id] -= deteccoes    
            # perdidas[cam_id] = 0
        else:
            perdidas[cam_id] += 15 - deteccoes
    
    return juntas, perdidas
        


def render_skeletons_3d(ax, skeletons, links, colors, juntas_3d, perdidas_3d):
    deteccoes_3d = 0
    skeletons_pb = ParseDict(skeletons, ObjectAnnotations())
    for skeleton in skeletons_pb.objects:
        parts = {}
        for part in skeleton.keypoints:
            deteccoes_3d += 1
            juntas_3d += 1 
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
    
    if deteccoes_3d < 10:
        juntas_3d -= deteccoes_3d
    else:
        perdidas_3d += 15 - deteccoes_3d
    return juntas_3d, perdidas_3d

def receive_information():
    connection = pika.BlockingConnection(
    pika.ConnectionParameters(host='localhost'))
    channel = connection.channel()
    channel.queue_declare(queue='SkeletonsGrouper.Localize')
    channel.basic_consume(exchange='', routing_key='SkeletonsGrouper.Localize', body = json.load({'dict': skeletons}).encode('utf-8'))
    print(body)
    # client =mqtt.connect('ws://guest:guest@localhost:15675/ws')
    # client.subscribe("SkeletonsGrouper/0/Localization")
    # print("Aqui")

receive_information()