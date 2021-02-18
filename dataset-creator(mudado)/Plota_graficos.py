import os
import re
import sys
import cv2
import json
import time
import argparse
import numpy as np
import math
import statistics
#import mplot3d from mpl_toolkits as axes3d
from mpl_toolkits import mplot3d
#matplotlib inline
import scipy
from scipy import interpolate
from sympy import S, symbols, printing
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
#from mpl_toolkits.mplot3d import Axes3D
from is_wire.core import Logger
from utils import load_options
from utils import to_labels_array, to_labels_dict
from video_loader import MultipleVideoLoader
from is_wire.core import Logger
from collections import defaultdict, OrderedDict
from utils import get_np_image
import csv
from is_msgs.image_pb2 import ObjectAnnotations

log = Logger(name='WatchVideos')

with open('keymap.json', 'r') as f:
    keymap = json.load(f)
options = load_options(print_options=False)

class Plota_graficos:
    def plota_grafico_perdas(y):
        fig,AX=plt.subplots()
        #y=y[1:]
        X=[]
        for i in range (0,len(y)):
            X.append(i)
        AX.plot(X,y)
        AX.set(xlabel='Medição',ylabel='Perda percentual (%)',title='Perdas na reconstrução 3D em função da amostragem')
        AX.grid()
        plt.savefig(options.folder+'/Perdas_na_reconstrucao3D.png')
        plt.show()
        
    def plota_grafico(y,title):
        fig,AX=plt.subplots()
        #y=y[1:]
        X=[]
        X=np.linspace(0, 100, num=len(y))
        
        AX.plot(X,y)
        AX.set(xlabel='N° de amostras',ylabel=title,title=title)
        AX.grid()
        plt.savefig(options.folder+'/'+title+'.png')
        plt.show()


    def plota_grafico_distance_feet(x,y):
        fig,AX=plt.subplots()
        y=y[2:]
        AX.plot(x,y)
        AX.set(xlabel='Tempo (s) ',ylabel='Distância (m) ',title='Medida das distâncias (m) em função do tempo (s)')
        AX.grid()
        plt.savefig(options.folder+'/Medidas_das_distancias_pelo_tempo.png')
        plt.show()

    def plota_grafico_tempo_de_passo(x,y,x_label='Tempo(s)', y_label='Comprimento(m)', titulo='Titulo'):
        fig,AX=plt.subplots()
        #y=y[1:]
        AX.plot(x,y)
        AX.set(xlabel=x_label,ylabel=y_label,title=titulo)
        AX.grid()
        plt.savefig(options.folder+'/Tempo_por_passo.png')
        plt.show()
    
    def plota_angulo_medido(y,titulo):
        x=len(y)
        #print(x)
        k=[]
        for i in range (0,x):
            k.append(i)

        fig,AX=plt.subplots()
        AX.plot(k,y)
        AX.set(xlabel='N° de amostras',ylabel='Ângulo (°)',title=titulo)
        AX.grid()
        plt.savefig(options.folder+'/'+titulo+'.png')
        plt.show()

    def plota_simetria(y,titulo):
        x=len(y)
        k=[]
        for i in range (0,x):
            k.append(i)

        fig,AX=plt.subplots()
        AX.plot(k,y)
        AX.set(xlabel='N° de amostras',ylabel='Simetria',title=titulo)
        AX.grid()
        plt.savefig(options.folder+'/'+titulo+'.png')
        plt.show()
    
    def plota_angulo_medido_normalizado(y,titulo):
    #     k_referencia=[]
        
    # ## Essas referências podem mudar conforme os ciclos em interesse para análise !!!##
    #     y_refencia=y[-1]

    #    # for i in range (0,len(y_refencia)+1):
    #    #k_referencia.append((100*i)/len(y_refencia))
       

    #     for i in range(0,quant_de_ciclos_desejado-1):
    #         a=np.array(y[i+1])
    #         ultimo_elemento=a[-1]
    #         B=np.array([ultimo_elemento])
    #         y[i]=np.append(np.array(y[i]),B)

    #     if quant_de_ciclos_desejado==1: #S for de 1 ciclo a análise final fica com menor quantidade de dados pra tirar a média
    #         y=y[:(quant_de_ciclos_desejado)]
        
    #     y=y[:(quant_de_ciclos_desejado-1)]
          
    #     ##Alinhando as curvas
    #     index_array_deslocado=0
    #     aux_array=[]
        
    #     aux_array=np.array(y[-1])
    #     indice_maior_valor=np.argmax(aux_array) #int(len(aux_array)*0.6) #np.argmax(aux_array) #

    #     for i in range(0,quant_de_ciclos_desejado-1):
    #         index_array_deslocado=np.argmax(y[i])
    #         while ((indice_maior_valor) != (index_array_deslocado+1)):
    #             index_array_deslocado=np.argmax(y[i])
    #             y[i]=np.roll(y[i],1)
    #             if (indice_maior_valor==0 and index_array_deslocado==0):
    #                 #print("break")
    #                 break
    #     aux=[]
    #     aux=np.mean(y,axis=0)
    #     k_referencia=np.linspace(0, 100, num=len(aux))
    #     #print(len(y[0]),len(k_referencia))
    #     ### Limpa aux_array! 
    #     aux_array=[]
    #     indice_maior_valor=int(len(y_refencia)*(pico_do_sinal/100.0))
    #     index_array_deslocado=np.argmax(aux)

    #     #### Alinhamento final!!!!!#####
    #     for i in range(0,len(y_refencia)): ## Array de referência !!!!!
    #         if (i ==indice_maior_valor):
    #             aux_array.append(1)
    #         else:
    #             aux_array.append(0)

    #     while ((indice_maior_valor) != (index_array_deslocado)):
    #             index_array_deslocado=np.argmax(aux)
    #             aux=np.roll(aux,1)
        
        k_referencia=np.linspace(0, 100, num=len(y))
        with open(options.folder+'/Parâmetros_de_todos_normalizado_'+titulo+'.csv', 'w') as myCsv:
            fieldnames=["Gait Cycle (%)","Angle (°)"]
            csvWriter = csv.DictWriter(myCsv,fieldnames=fieldnames)
            csvWriter.writeheader()
            for i in range(0,len(k_referencia)):
                csvWriter.writerow({'Gait Cycle (%)' : '%.5f' % k_referencia[i],'Angle (°)' : '%.5f' % y[i]})
        fig,AX=plt.subplots()
        AX.plot(k_referencia,y, label="Valor médio: {:3f} +- {:3f}".format(statistics.mean(y),statistics.pstdev(y)),color="gray", linewidth=5.0, linestyle="--")
        x=k_referencia
        Y=y

        ##interpolação da curva
        #s = interpolate.InterpolatedUnivariateSpline(x, Y)
        #xnew = np.arange(0,100)
        #ynew = s(xnew)
        #print(len(xnew),len(ynew))
        
        p = np.polyfit(x, Y, len(Y)-1)
        f = np.poly1d(p)
        xnew = k_referencia#np.arange(0,len(y_refencia)+1)
        ynew = f(xnew)
        # calculate new x's and y's
        ##x_new = np.linspace(x[0], x[-1], 50)
        ##y_new = f(x_new)

        x = symbols("x")
        poly = sum(S("{:6.2f}".format(v))*x**i for i, v in enumerate(p[::-1]))
        eq_latex = printing.latex(poly)
        plt.plot(xnew, ynew, label="${}$".format(eq_latex))
        desvio_padrao_curva_media=np.std(y)
        #AX.plot(x,Y, 'x', xnew, ynew, 'b')
        ##AX.errorbar(k_5,aux,desvio_padrao_curva_media)
        Sigma_new_vec = desvio_padrao_curva_media#ynew-aux
        lower_bound = y - Sigma_new_vec
        upper_bound = y + Sigma_new_vec
        #xnew = np.arange(0,100)
        plt.fill_between(xnew, lower_bound, upper_bound, color='green',alpha=.3)

        AX.set(xlabel='Gait Cycle %',ylabel='Angle (°)',title=titulo)
        AX.grid()
        plt.legend()
        plt.savefig(options.folder+'/'+titulo+'.png')
        plt.show()
        

    def trajetoria_vetor(vetor):
        X=[0]
        Y=[0]
        Z=[0]
        title='Trajetória vetor normal ao tórax'
        fig=plt.figure()
        ax=plt.axes(projection='3d')
        ax.set_xlim3d([-2.0, 2.0])
        ax.set_xlabel('X')

        ax.set_ylim3d([-2.0, 2.0])
        ax.set_ylabel('Y')

        ax.set_zlim3d([-2.0, 2.0])
        ax.set_zlabel('Z')

        ax.set_title('Trajetória vetor normal ao tórax')

        for i in range(0,len(vetor)):
            X.append(vetor[i][0])
            Y.append(vetor[i][1])
            Z.append(vetor[i][2])
        #print(vetor[0],vetor[1],vetor[2])
        ax.scatter(X,Y,Z,c='r',marker='o')
        #ax.plot3D(X,Y,Z)
        ax.grid()
        plt.savefig(options.folder+'/'+title+'.png')
        plt.show()

