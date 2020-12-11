import os
import re
import sys
import cv2
import json
import time
import argparse
import numpy as np
import pika 
import statistics
from utils import load_options
from utils import to_labels_array, to_labels_dict
from video_loader import MultipleVideoLoader
from is_wire.core import Channel, Subscription, Message, Logger, ContentType
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


#log = Logger(name='Send_3D')
#options = load_options(print_options=False)

#channel = Channel(options.broker_uri)
#subscription = Subscription(channel)

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
    # msg = Message(reply_to=subscription, content_type=ContentType.JSON)
    # body = json.dumps({'dict': skeletons_pb}).encode('utf-8')
    # msg.body = body
    # msg.timeout = 5.0
    # channel.publish(msg, topic='SkeletonsGrouper.Localization')
    # requests[msg.correlation_id] = skeletons_pb
    # {
    #     'body': body,
    #     'person_id': person_id,
    #     'gesture_id': gesture_id,
    #     'pos': pos,
    #     'requested_at': time.time()
    # }
    for skeleton in skeletons_pb.objects:
        parts = {}
        for part in skeleton.keypoints:
            deteccoes_3d += 1
            juntas_3d += 1 
            parts[part.id] = (part.position.x, part.position.y, part.position.z)
    #     msg = Message(reply_to=subscription, content_type=ContentType.JSON)
    #     body = json.dumps({'dict': parts}).encode('utf-8')
    #     msg.body = body
    #     msg.timeout = time.time()
    #     channel.publish(msg, topic='SkeletonsGrouper.Localize')
    #    # msg = channel.consume(timeout=5.0)
    #    print(skeleton)
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

def send_information(skeletons):
    skeletons_pb = ParseDict(skeletons, ObjectAnnotations())
    #print(skeletons_pb) 
    connection = pika.BlockingConnection(
    pika.ConnectionParameters(host='localhost'))
    channel = connection.channel()
    channel.queue_declare(queue='hello')
    for skeleton in skeletons_pb.objects:
        parts = {}
        for part in skeleton.keypoints:
            parts[part.id] = (part.position.x, part.position.y, part.position.z)
            channel.basic_publish(exchange='', routing_key='hello', body = json.dumps({'dict': skeletons}).encode('utf-8'))
            #msg=channel.basic_consume(queue='hello', on_message_callback=callback, auto_ack=True)
            #print(msg)
    print("Sent message!")
    connection.close()

def callback(ch, method, properties, body):
    print(" Received ")

def left_leg(skeletons):
        left_ankle=None 
        left_hip=None 
        left_knee=None
        skeletons_pb = ParseDict( skeletons, ObjectAnnotations())
        
        for skeletons in skeletons_pb.objects:
            parts = {}
            for part in skeletons.keypoints:
                parts[part.id]=(part.position.x,part.position.y,part.position.z)
                if part.id == 15:
                    left_ankle = parts[15]
                if part.id == 13:
                    left_hip = parts[13]
                if part.id == 14:
                    left_knee=parts[14]

            if left_ankle and left_hip and left_knee:
                left_hip=parts[13]
                left_knee=parts[14]
                left_ankle=parts[15]
                a=np.sqrt((left_ankle[0]-left_knee[0])**2+(left_ankle[1]-left_knee[1])**2+(left_ankle[2]-left_knee[2])**2)
                b=np.sqrt((left_knee[0]-left_hip[0])**2 +(left_knee[1]-left_hip[1])**2 +(left_knee[2]-left_hip[2])**2)
                left_leg=a+b
                return left_leg
            else:
                left_leg=0 
                return left_leg
            break

def right_leg(skeletons):   
    skeletons_pb = ParseDict( skeletons, ObjectAnnotations())
    for skeletons in skeletons_pb.objects:
        right_hip=None 
        right_knee=None 
        right_ankle=None
        right_leg=None 
        left_ankle=None
        mid_point_ankle=0
        parts = {}
        for part in skeletons.keypoints:
            parts[part.id]=(part.position.x,part.position.y,part.position.z)
            if part.id == 11:
                right_knee= parts[11]
            if part.id == 10:
                right_hip = parts[10]
            if part.id == 12:
                right_ankle=parts[12]
            if part.id == 15:
                left_ankle=parts[15]


        if right_ankle and right_hip and right_knee:
            a=np.sqrt((right_ankle[0]-right_knee[0])**2+(right_ankle[1]-right_knee[1])**2+(right_ankle[2]-right_knee[2])**2)
            b=np.sqrt((right_knee[0]-right_hip[0])**2 +(right_knee[1]-right_hip[1])**2 +(right_knee[2]-right_hip[2])**2)
            right_leg=a+b
            mid_point_ankle=(left_ankle[2]+right_ankle[2])/2

            return right_leg,mid_point_ankle
        
        else:
            mid_point_ankle=0
            right_leg=0 
            return right_leg, mid_point_ankle
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
    altura_da_pessoa=None 
    for skeletons in skeletons_pb.objects:
        parts = {}
        for part in skeletons.keypoints:
            parts[part.id] = (part.position.x, part.position.y, part.position.z)
        if altura_da_pessoa:
            altura_da_pessoa=parts[1][2]
        else:
            altura_da_pessoa=0
        break
    return altura_da_pessoa

def plota_grafico_perdas(x,y):
    fig,AX=plt.subplots()
    AX.plot(x,y)
    AX.set(xlabel='Medição',ylabel='Perda percentual (%)',title='Perdas na reconstrução 3D em função da amostragem')
    AX.grid()
    plt.show()

def plota_grafico_distance_feet(x,y):
    fig,AX=plt.subplots()
    AX.plot(x,y)
    AX.set(xlabel='Tempo (s) ',ylabel='Distância (m) ',title='Medida das distâncias (m) em função do tempo (s)')
    AX.grid()
    plt.show()

def plota_grafico(x,y, x_label='Tempo(s)', y_label='Comprimento(m)', titulo='Titulo' ):
    fig,AX=plt.subplots()
    AX.plot(x,y)
    AX.set(xlabel=x_label,ylabel=y_label,title=titulo)
    AX.grid()
    plt.show()

def place_images(output_image, images, x_offset=0, y_offset=0):
    w, h = images[0].shape[1], images[0].shape[0]
    output_image[0 + y_offset:h + y_offset, 0 + x_offset:w + x_offset, :] = images[0]
    output_image[0 + y_offset:h + y_offset, w + x_offset:2 * w + x_offset, :] = images[1]
    output_image[h + y_offset:2 * h + y_offset, 0 + x_offset:w + x_offset, :] = images[2]
    output_image[h + y_offset:2 * h + y_offset, w + x_offset:2 * w + x_offset, :] = images[3]

def file_maker(cam_id,juntas,perdidas,porcentagem,porcentagem_3d,perda_media,variancia,y,x,perna_esquerda,perna_direita,maior_passo_medido,tempo_total,velocidade_media, passos_por_min, contador,tempo_total_em_min):

    file_results=open("Resultados_reconstrucao_3D.txt","w")
    file_results.write("Resultados da reconstrução 3D \n")
    for cam_id in range(0, 4):
        porcentagem = (perdidas[cam_id]/juntas[cam_id]) * 100
        file_results.write("cam{}: Juntas detectadas: {} | Perdidas: {} |  {:.2f} %".format(cam_id, juntas[cam_id], perdidas[cam_id], porcentagem))
        file_results.write("\n")

    file_results.write("Juntas detectadas [Serviço 3d]: {} | Perdidas: {} |  {:.2f} %".format(juntas_3d, perdidas_3d, porcentagem_3d))
    file_results.write("\n")
    file_results.write("Média das medições das perdas no 3D: %5.2f" % perda_media + " %")
    file_results.write("\n")
    file_results.write("Variância das medições das perdas no 3D: %5.2f" % variancia + " %")
    file_results.write("\n")
    file_results.write("Desvio padrão das medições das perdas no 3D: %5.2f" % statistics.pstdev(y) + " %")
    file_results.write("\n")
    altura_media=sum(average_height)/len(x)
    file_results.write("Altura: %5.3f m" % altura_media)
    file_results.write("\n")
    perna_esquerda_media=sum(perna_esquerda)/len(perna_esquerda) +0.117
    file_results.write("Perna esquerda: %5.3f m" % perna_esquerda_media)
    file_results.write("\n")
    perna_direita_media=sum(perna_direita)/len(perna_direita)+0.117
    file_results.write("Perna direita: %5.3f m" % perna_direita_media)
    file_results.write("\n")
    file_results.write("Maior comprimento de passo: %.3f m" % maior_passo_medido)
    file_results.write("\n")
    file_results.write("Tempo total: %5.4f" % tempo_total)
    file_results.write("\n")
    file_results.write("Tempo de suporte duplo: %.3f s " % (0.2*tempo_total))
    file_results.write("\n")
    file_results.write("Velocidade média: %.3f m/s " % velocidade_media)
    file_results.write("\n")
    file_results.write("Número de passos: %d " % contador)
    file_results.write("\n")
    file_results.write("Passos por min: %.3f " % passos_por_min)
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
fig = plt.figure(figsize=(5, 5))
ax = Axes3D(fig)

update_image = True
output_file = 'p{:03d}g{:02d}_output.mp4'.format(person_id, gesture_id)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
vid = cv2.VideoWriter('record_screen.avi', fourcc, 60.0, (1288,728))
# video_writer = cv2.VideoWriter()

juntas = [0, 0, 0, 0]           # Lista de juntas detectadas em cada câmera
perdidas = [0, 0, 0, 0]         # Lista de juntas perdidas em cada câmera
juntas_3d = 0
perdidas_3d = 0
it_frames = 0
y = []
x = []
i=0
picos_distancia=[]
average_height=[]
tempo_inicial = time.time()
#tempo_inicial_1=tempo_inicial
tempo_inicial_vetor=[]
perna_esquerda=[]
perna_direita=[]
distance_feet = []
contador=0
#picos_maximos_distancia=[]
instante = []
instante_pico = []
tempo_passo = []
tempo_suporte_duplo = []
dist_do_chao=[]


for it_frames in range(video_loader.n_frames()):
    video_loader.load_next()

    frames = video_loader[it_frames]
    if frames is not None:
        juntas, perdidas = render_skeletons(frames, annotations, it_frames, links, colors)
        frames_list = [frames[cam] for cam in sorted(frames.keys())]
        place_images(full_image, frames_list)

    ax.clear()
    ax.view_init(azim=28, elev=32)
    ax.set_xlim(-1.5, 0)
    ax.set_xticks(np.arange(-1.5, 0.5, 0.5))
    ax.set_ylim(-3.0, 3.0)
    ax.set_yticks(np.arange(-5.0, 2.0, 0.5))
    ax.set_zlim(-0.25, 1.5)
    ax.set_zticks(np.arange(0, 1.75, 0.5))
    ax.set_xlabel('X', labelpad=20)
    ax.set_ylabel('Y', labelpad=10)
    ax.set_zlabel('Z', labelpad=5)
    juntas_3d, perdidas_3d = render_skeletons_3d(ax, localizations[it_frames], links, colors, juntas_3d, perdidas_3d)
    
    perdas_no_3d=perdas_3d(ax, localizations[it_frames], links, colors)
    i=i+1    
    if perdas_no_3d is None:
        perdas_no_3d=100

    y.append(perdas_no_3d)
    x.append(i) 

    average_height.append(altura_da_pessoa(localizations[it_frames]))

    perna_esquerda.append(left_leg(localizations[it_frames]))
    perna_direita_aux,dist_do_chao_aux=right_leg(localizations[it_frames])
    perna_direita.append(perna_direita_aux)
    dist_do_chao.append(dist_do_chao_aux)
    
    if SkeletonsCoord.joint_coord(localizations[it_frames], 12) and SkeletonsCoord.joint_coord(localizations[it_frames], 15):
        right_foot = SkeletonsCoord.joint_coord(localizations[it_frames], 12)
        left_foot = SkeletonsCoord.joint_coord(localizations[it_frames], 15)
        distance_feet.append(np.sqrt((right_foot[0]-left_foot[0])**2 + (right_foot[1]-left_foot[1])**2 + (right_foot[2]-left_foot[2])**2))
        instante.append(time.time() - tempo_inicial)
        # tempo_inicial
    
    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    view_3d = data.reshape(fig.canvas.get_width_height()[::-1] + (3, ))

    display_image = cv2.resize(full_image, dsize=(0, 0), fx=0.5, fy=0.5)
    hd, wd, _ = display_image.shape
    hv, wv, _ = view_3d.shape

    display_image = np.hstack([display_image, 255 * np.ones(shape=(hd, wv, 3), dtype=np.uint8)])
    display_image[int((hd - hv) / 2):int((hd + hv) / 2), wd:, :] = view_3d
        
    #print("Perdas no 3D: %5.2f"  % perdas_no_3d + "%")
    # if it_frames == 0:
    #     video_writer.open(
    #         filename=output_file,
    #         fourcc=0x00000021,
    #         fps=10.0,
    #         frameSize=(display_image.shape[1], display_image.shape[0]))

    # video_writer.write(display_image)
    cv2.imshow('', display_image)

    # image = np.array(ImageGrab.grab())
    # image = get_np_image(image)
    # image = np.array(cv2.grab())
    # image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    # cv2.imshow('', image)
    vid.write(display_image)
    # key = cv2.waitKey(1)

    send_information(localizations[it_frames])


    if cv2.waitKey(57) & 0xFF == ord('q'):
        break

for j in range(0,len(distance_feet)-2):
    if distance_feet[j] > distance_feet[0]:
        #tempo_inicial_vetor.append(time.time())
        #print(distance_feet[j])
        
        if distance_feet[j+1] > distance_feet[j+2] and distance_feet[j+1] > distance_feet[j]:
           contador=contador+1 #Conta quantos passos foram dados
           picos_distancia.append(distance_feet[j+1])
           instante_pico.append(instante[j+1])
        # else:
            # picos_distancia.append(None)
            # instante_pico.append(None)

#print(picos_distancia)
#print(instante_pico)

if np.mod(len(picos_distancia),2) == 0:
    for j in range(0, len(picos_distancia)-1):
        tempo_passo.append(instante_pico[j+1] - instante_pico[j])
    for j in range(0, len(picos_distancia)-2,2):
        tempo_suporte_duplo.append(instante_pico[j+2] - instante_pico[j])
else:
    for j in range(0,len(picos_distancia)-1):
        tempo_passo.append(instante_pico[j+1] - instante_pico[j])
    for j in range(0, len(picos_distancia)-2,2):
        tempo_suporte_duplo.append(instante_pico[j+2] - instante_pico[j])


#print(tempo_passo)
#print(tempo_suporte_duplo)
h = []
for j in range(1,len(tempo_passo)+1):
    h.append(j)
# print(tempo_inicial_vetor[0])
# print(picos_distancia[0])
# print(picos_distancia[1])
print("Meio comprimento de passo médio: %.3f m" % statistics.mean(picos_distancia))  

#for r in range(0,len(picos_distancia)-2):
#    if picos_distancia[r+1] > picos_distancia[r+2] and picos_distancia[r+1] > picos_distancia[r]:
#        picos_maximos_distancia.append(picos_distancia[r+1])
#        print([picos_maximos_distancia][r])

maior_passo_medido=picos_distancia[2]+picos_distancia[1]
print("Maior comprimento de passo: %.3f m" % maior_passo_medido)


for cam_id in range(0, 4):
    porcentagem = (perdidas[cam_id]/juntas[cam_id]) * 100
    log.info("cam{}: Juntas detectadas: {} | Perdidas: {} |  {:.2f} %".format(cam_id, juntas[cam_id], perdidas[cam_id], porcentagem))

porcentagem_3d = (perdidas_3d/juntas_3d) * 100
log.info("Juntas detectadas [Serviço 3d]: {} | Perdidas: {} |  {:.2f} %".format(juntas_3d, perdidas_3d, porcentagem_3d))

log.info('Exiting')
soma_perdas=sum(y)
tempo_total=time.time()-tempo_inicial
print("Tempo de suporte duplo: %.4f s" % (0.2*tempo_total))
velocidade_media=sum(picos_distancia)/tempo_total
print("Velocidade média: %.3f m/s " % velocidade_media)
perda_media=soma_perdas/len(x)
print("Perda média do 3D: %5.2f" % perda_media + " +- %5.3f" % statistics.pstdev(y) + " %" )
variancia=statistics.variance(y)
print("Variância de %5.3f " % variancia + " % no 3D")
print("Número de passos: %d " % contador)
tempo_total_em_min=tempo_total/60
passos_por_min=contador/tempo_total_em_min
print("Passos por min: %.3f " % passos_por_min)

plota_grafico_perdas(x,y) 
plota_grafico_distance_feet(instante,distance_feet)
plota_grafico(h,tempo_passo, 'Passo', 'Tempo de passo(s)', 'Tempo de Passos')
file_maker(cam_id,juntas,perdidas,porcentagem,porcentagem_3d,perda_media,variancia,y,x,perna_esquerda,perna_direita,maior_passo_medido,tempo_total,velocidade_media, passos_por_min, contador,tempo_total_em_min)

vid.release()
cv2.destroyAllWindows()