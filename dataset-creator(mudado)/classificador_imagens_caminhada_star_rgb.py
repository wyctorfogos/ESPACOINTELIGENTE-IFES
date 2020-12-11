import numpy as np
import cv2
import tensorflow as tf
import tensorflow.keras
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
import keras.layers
from keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import TensorBoard
from tensorflow.keras.models  import load_model
import pickle
import time
#from keras.backend.tensorflow_backend import set_session
from keras import regularizers
import sklearn
from sklearn.model_selection import train_test_split


CATEGORIAS=["Certo","Errado"]

IMG_SIZE=75

modelo_final = tf.keras.models.load_model("/home/julian/docker/ifes-2019-09-09/Modelos_para_treinamento/Modelo_classificador_imagens/Modelo_movimento_certo_e_errado_27072020")

cap = cv2.VideoCapture('/home/julian/docker/ifes-2019-09-09/Simulacao_de_movimento_deficiente/Simulacao_de_movimento_deficiente_6/p001g01c03.mp4')

if (cap.isOpened()== False): 
  print("Error opening video stream or file")

aux_diferenca_frame=[0]
aux_soma_frame=[0]
frame_anterior=[0]
k=0
tempo_inicial=time.time()
tempo_final=0
intervalo_de_tempo=0
i=0

while(cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #print(frame)
    if ret == True:
        if k!=0:    
            #frame=cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            frame=(frame/255.0)
            aux_diferenca_frame=(frame-frame_anterior)
            #print(aux_diferenca_frame)
            #cv2.imshow('Frame', aux_diferenca_frame)
            #cv.imshow('FG Mask', fgMask)
            if (k%2==0):
                aux_soma_frame=aux_soma_frame+aux_diferenca_frame
                    
            if (k%50==0): # Média de frames para contar 4 ciclos
                new_array = cv2.resize(aux_soma_frame, (IMG_SIZE, IMG_SIZE))
                new_array=(new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)/255)
                #print(new_array,(new_array).shape)
                    
                prediction = modelo_final.predict(new_array)#[prepare(filepath)])
                #prediction=(np.around(prediction).reshape([1,1]))
                print(prediction)
                prediction=(np.around(prediction).reshape([1,3]))
                print(prediction)
                movimento=CATEGORIAS[int(prediction[0][0])]
                text = "MOVIMENTO: {}".format(movimento)
                #cv2.putText(frame, text, (35, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (0, 255, 0),1, 5)
                cv2.putText(aux_soma_frame, text, (35, 50),cv2.FONT_HERSHEY_SIMPLEX, 1.5, (100,00,10),1, cv2.LINE_AA) 
                #cv2.imshow('', aux_soma_frame)
                cv2.imshow('Frame', aux_soma_frame)
                aux_soma_frame=0

            #aux_soma_frame=aux_soma_frame+aux_diferenca_frame

                    
            keyboard = cv2.waitKey(30)
           
        k=k+1
        #Atualiza o frame que será comparado na próxima iteração
        frame_anterior=frame
    # Press Q on keyboard to  exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
      break
# Break the loop

# When everything done, release the video capture object

cap.release()

# Closes all the frames

cv2.destroyAllWindows()
