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
import pika
import csv

k=0

with open('/home/julian/docker/ifes-2019-09-09/Parâmetros_1.csv', 'w') as csvfile:
    filewriter = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        filewriter.writerow(["Nº", "Altura (m)","Comprimento médio passo completo","Erro absoluto médio do comprimento de passo em metros","Desvio padrão comprimento passo medido em metros","Desvio padrão do erro de comprimento de passo em metros","Comprimento médio de meio passo em metros","Erro absoluto médio do meio comprimento de passo em metros","Desvio padrão do comprimento médio de meio passo em metros","Desvio padrão do erro de comprimento de meio passo em metros","Comprimento do Swing em metros","Erro absoluto médio do swing em metros","Desvio padrão do swing em metros","Desvio padrão do erro de swing em metros","Comprimento do Stance em metros","Erro absoluto médio do stance em metros","Desvio padrão do stance em metros","Desvio padrão do erro de stance em metros","Distância inicial do pé em metros","Erro absoluto médio da distância entre os pés em metros","Desvio padrão da distância inicial entre os pés","Desvio padrão do erro da distância inicial entre os pés em metros","Ângulo real de abertura das pernas em graus","Ângulo médio de abertura das pernas em graus","Erro absoluto médio do angulo entre as pernas em graus","Desvio padrão do ângulo médio dos passos em graus","Número de amostras do ângulo"])
        filewriter.writerow(["%i" % k,"%.4f" % statistics.mean(average_height),"%.4f" % statistics.mean(comprimento_passo_medido),"%.4f" % abs(erro_medio_comprimento_de_passo),"%.4f" % abs(statistics.pstdev(comprimento_passo_medido))," %.2f" % abs(statistics.pstdev(vetor_erro_comprimento_de_passo)),"%.4f" % statistics.mean(picos_distancia),"%.4f" % abs(erro_medio_meio_comprimento_de_passo),"%.4f"% abs(statistics.pstdev(picos_distancia)),"%.4f" % abs(statistics.pstdev(vetor_erro_comprimento_de_meio_passo))," %.4f" % statistics.mean(comprimento_swing),"%.4f" % abs(erro_swing),"%.4f" % abs(statistics.pstdev(comprimento_swing)),"%.4f" % abs(statistics.pstdev(vetor_erro_comprimento_swing)),"%.4f" % statistics.mean(comprimento_stance),"%.4f" % abs(erro_stance),"%.4f " % abs(statistics.pstdev(comprimento_stance)),"%.4f" % abs(statistics.pstdev(vetor_erro_comprimento_stance)),"%.4f" % distance_feet[0],"%.4f" % abs(erro_dist_inicial),"%.4f" % abs(statistics.pstdev(vetor_erro_distancia_dos_pes_inicial)),"%.4f" % math.degrees(math.acos(aux2)), "%.4f" % statistics.mean(angulo_caminhada), "%5.4f" % abs(erro_medio_angulo),"%5.4f" % abs(statistics.pstdev(angulo_caminhada)),"%i" % len(angulo_caminhada)])
        k=k+1