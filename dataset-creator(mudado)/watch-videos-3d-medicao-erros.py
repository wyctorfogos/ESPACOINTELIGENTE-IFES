import os
import re
import sys
import cv2
from cv2 import imshow, resize, destroyAllWindows, waitKey
import json
import time
import argparse
import numpy as np
from utils import load_options
from utils import to_labels_array, to_labels_dict
from video_loader import MultipleVideoLoader
from is_wire.core import Logger
from collections import defaultdict, OrderedDict
import matplotlib 
import statistics
from is_msgs.image_pb2 import ObjectAnnotations
from is_msgs.image_pb2 import HumanKeypoints as HKP
from google.protobuf.json_format import ParseDict
from itertools import permutations
from analysis import SkeletonsCoord

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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
                #perdas=(15-len(parts))/15.0
                #perdas=perdas*100.0
                #print("Perdas na detecção: " +   str(perdas) + "%")
                #print(skeleton.keypoints)
                ax.plot(
                    x_pair,
                    y_pair,
                    zs=z_pair,
                    linewidth=3,
                    color='#{:02X}{:02X}{:02X}'.format(*reversed(color)))
        break
                    
def left_leg(skeletons):
    skeletons_pb = ParseDict( skeletons, ObjectAnnotations())
    for skeletons in skeletons_pb.objects:
        left_hip = None
        left_knee = None
        left_ankle = None
        parts = {}
        for part in skeletons.keypoints:
            parts[part.id]=(part.position.x,part.position.y,part.position.z)
            if part.id == 13:
                left_hip = parts[13]
            if part.id == 14:
                left_knee=parts[14]
            if part.id == 15:
                left_ankle=parts[15]

        if left_hip and left_knee and left_ankle:
            a=np.sqrt((left_ankle[0]-left_knee[0])**2+(left_ankle[1]-left_knee[1])**2+(left_ankle[2]-left_knee[2])**2)
            b=np.sqrt((left_knee[0]-left_hip[0])**2 +(left_knee[1]-left_hip[1])**2 +(left_knee[2]-left_hip[2])**2)
            left_leg=a+b
            return left_leg
        else:
            left_leg = 0
            return left_leg
        break

def right_leg(skeletons):
    skeletons_pb = ParseDict( skeletons, ObjectAnnotations())
    right_hip=None  
    right_ankle=None 
    right_knee=None
    right_leg=None 
    mid_point_ankle=None 
    left_ankle=None 

    for skeletons in skeletons_pb.objects:
        parts = {}
        for part in skeletons.keypoints:
            parts[part.id]=(part.position.x,part.position.y,part.position.z)
            if part.id == 10:
                right_hip = parts[10]
            if part.id == 11:
                right_knee=parts[11]
            if part.id == 12:
                right_ankle=parts[12]
            
            if part.id == 15:
                left_ankle=parts[15]

        if right_ankle and right_knee and right_hip and left_ankle:
            a=np.sqrt((right_ankle[0]-right_knee[0])**2+(right_ankle[1]-right_knee[1])**2+(right_ankle[2]-right_knee[2])**2)
            b=np.sqrt((right_knee[0]-right_hip[0])**2 +(right_knee[1]-right_hip[1])**2 +(right_knee[2]-right_hip[2])**2)
            right_leg=a+b
            mid_point_ankle=(left_ankle[2]+right_ankle[2])/2
            return right_leg, mid_point_ankle
        else:
            mid_point_ankle=0
            right_leg=0
            return right_leg,mid_point_ankle
        break


def perdas_3d(ax, skeletons, links, colors):
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
                perdas=(15-len(parts))/15.0
                perdas=perdas*100.0
                #print("Perdas na detecção: " +   str(perdas) + "%")
                return perdas

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


def plota_grafico_perdas(x,y):
    fig2,AX=plt.subplots()
    AX.plot(x,y)
    AX.set(xlabel='Medição',ylabel='Perda percentual (%)',title='Perdas na reconstrução 3D em função da amostragem')
    AX.grid()
    plt.show()

def data_sheet_of_members(alt_torn,comprimento_medio_perna_esquerda,comprimento_medio_perna_direita,altura_media):
    file_results=open("Resultados_medições_dos_membros.txt","w")

    file_results.write("Distância do tornozelo ao chão: %5.3f m" % alt_torn)
    file_results.write("\n")
    file_results.write("Tamanho da perna esquerda: %5.3f m" % comprimento_medio_perna_esquerda)
    file_results.write("\n")
    file_results.write("Tamanho da perna direita: %5.3f m" % comprimento_medio_perna_direita)
    file_results.write("\n")

    file_results.write("Altura média: %5.3f m" % altura_media) #(sum(altura_pessoa)/len(altura_pessoa)))
    file_results.write("\n")
    file_results.close()

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
ax = Axes3D(fig)

update_image = True
it_frames = 0
y = []
x = []
i=0
picos_distancia=[]
perna_esquerda=[]
perna_direita=[]
dist_chao=[]
altura_pessoa=[]
tempo_inicial=time.time()
distance_feet = []
picos_maximos_distancia=[]
dist_do_chao=[]
perna_direita_aux=0
dist_do_chao_aux=0

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
        ax.view_init(azim=28, elev=32)
        ax.set_xlim(-1.5, 1.5)
        ax.set_xticks(np.arange(-1.5, 2.0, 0.5))
        ax.set_ylim(-1.5, 1.5)
        ax.set_yticks(np.arange(-1.5, 2.0, 0.5))
        ax.set_zlim(-0.25, 1.5)
        ax.set_zticks(np.arange(0, 1.75, 0.5))
        ax.set_xlabel('X', labelpad=20)
        ax.set_ylabel('Y', labelpad=10)
        ax.set_zlabel('Z', labelpad=5)
        render_skeletons_3d(ax, localizations[it_frames], links, colors)
        perdas_no_3d=perdas_3d(ax, localizations[it_frames], links, colors)
        
        if perdas_no_3d is None:
            perdas_no_3d=100

        y.append(perdas_no_3d)
        x.append(i) 

        perna_esquerda.append(left_leg(localizations[it_frames]))
        aux_right_leg,dist_do_chao=right_leg(localizations[it_frames])
        perna_direita.append(aux_right_leg)
        
        if SkeletonsCoord.joint_coord(localizations[it_frames], 12) and SkeletonsCoord.joint_coord(localizations[it_frames], 15):
            right_foot = SkeletonsCoord.joint_coord(localizations[it_frames], 12)
            left_foot = SkeletonsCoord.joint_coord(localizations[it_frames], 15)
            distance_feet.append(np.sqrt((right_foot[0]-left_foot[0])**2 + (right_foot[1]-left_foot[1])**2 + (right_foot[2]-left_foot[2])**2))
            #instante.append(time.time() -tempo_inicial)
            #tempo_inicial

        # print(distance_feet)

        print("Perna direita: %5.3f" % aux_right_leg)
        dist_chao.append(dist_do_chao)
        print("Distacia do chão: %5.3f m" % dist_do_chao)
        print("Perdas no 3D: %5.2f"  % perdas_no_3d + "%")
        average_height=altura_da_pessoa(localizations[it_frames])
        altura_pessoa.append(average_height)
        fig.canvas.draw()
        #cv2.putText(titulo, "Taxa de perda: %.2f " % perdas_no_3d + " %", (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

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
        i=i+1

    if key == ord(keymap['next_frame']):
        it_frames += 1
        it_frames = it_frames if it_frames < n_loaded_frames else 0
        update_image = True
        i=i+1

    if key == ord(keymap['previous_frames']):
        it_frames -= keymap['big_step']
        it_frames = n_loaded_frames - 1 if it_frames < 0 else it_frames
        update_image = True
        i=i+1

    if key == ord(keymap['previous_frame']):
        it_frames -= 1
        it_frames = n_loaded_frames - 1 if it_frames < 0 else it_frames
        update_image = True
        i=i+1

    #if key == ord(keymap['exit']):
    #   sys.exit(0)    
    if waitKey(0) & 0xFF == ord('q'):
       break   

for j in range(0,len(distance_feet)-2):
    if distance_feet[j+1] > distance_feet[j+2] and distance_feet[j+1] > distance_feet[j]:
        picos_distancia.append(distance_feet[j+1])

for r in range(0,len(picos_distancia)-2):
    if picos_distancia[r+1] > picos_distancia[r+2] and picos_distancia[r+1] > picos_distancia[r]:
        picos_maximos_distancia.append(picos_distancia[r+1])

print(picos_distancia)

#print(picos_maximos_distancia)
#print(len(picos_distancia))

soma_perdas=sum(y)
tempo_final=time.time()
tempo_duplo_suporte=(tempo_final-tempo_inicial)
alt_torn=0.135#sum(dist_chao)/len(dist_chao)
print("Tempo total: %5.4f" % tempo_duplo_suporte)
perda_media=soma_perdas/len(x)
print("Perda média: %5.2f" % perda_media + " %")
comprimento_medio_perna_esquerda=sum(perna_esquerda)/len(perna_esquerda) + alt_torn
print("Tamanho da perna esquerda: %5.3f m" % comprimento_medio_perna_esquerda)
comprimento_medio_perna_direita=sum(perna_direita)/len(perna_direita) + alt_torn
print("Tamanho da perna direita: %5.3f m" % comprimento_medio_perna_direita)
altura_media=statistics.mean(altura_pessoa)
log.info("Altura da pessoa: {0:5.3f}", altura_media)

data_sheet_of_members(alt_torn,comprimento_medio_perna_esquerda,comprimento_medio_perna_direita,altura_media)

plota_grafico_perdas(x,y) 
log.info('Exiting')