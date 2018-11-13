import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd
import networkx as nx
import community
import os
import scipy.misc
from scipy.sparse import csr_matrix
import scipy.sparse.csgraph as sc
import copy
import itertools
from music21 import *

env = environment.UserSettings()
env['musicxmlPath']
#env['musicxmlPath'] = r'C:\Program Files (x86)\Finale NotePad 2012\Finale NotePad.exe' #Path a la aplicacion Finale
#env['musicxmlPath'] = r'C:\Program Files (x86)\MuseScore 2\bin\MuseScore.exe'          #Path a la aplicacion MuseScore 
env['musicxmlPath']='/usr/bin/mscore'

#Cancion
song = converter.parse('./queen_up.xml')
#song=converter.parse('./vanillaice_iceicebaby.xml')

#Primer instrumento:
voz=song.getElementsByClass(stream.Part)[0].getElementsByClass(stream.Measure) #todas los compases de la parte voz

#Creamos el grafo que sera dirigido:
G = nx.DiGraph()

#Nodos
notas=[] #todas las notas incluyendo silencios en la voz analizada
nodos=[] #solamente los nodos que van a figurar en la red (no tiene elementos repetidos)
colores_nodos=[] #color de cada uno de los nodos en la lista nodos
colores_octava=['cyan','orange','blue','yellow','red','dodgerblue','green'] #colores por octava
colores_rest=['purple'] #colores de los silencios

#Nodos
for i,el in enumerate(voz.flat):
    if isinstance(el,note.Note):
        nota_name=str(el.name)+str(el.octave)+'/'+str(el.quarterLength)
        notas.append(nota_name)
        if G.has_node(nota_name)==False:
            G.add_node(nota_name)
            nodos.append(nota_name)
            colores_nodos.append(colores_octava[el.octave])
            
            
    elif isinstance(el,note.Rest):
        nota_name=str(el.name)+'/'+str(el.quarterLength)
        notas.append(nota_name)
        if G.has_node(nota_name)==False:
            G.add_node(nota_name)
            nodos.append(nota_name)
            colores_nodos.append(colores_rest[0])
            
#Enlaces pesados
enlaces=[]
for i in range(0,len(notas)-1):
    enlace=[notas[i],notas[i+1]]
    if G.has_edge(notas[i],notas[i+1])==False:
        G.add_edge(notas[i],notas[i+1],weight=1)
        enlaces.append(enlace)
    else:
        G[notas[i]][notas[i+1]]['weight']+=1
          
#Dibujamos la red
width=8
height=8
fig=plt.figure(figsize=(width, height))
fig.patch.set_facecolor('white')
pos= nx.spring_layout(G)
nx.draw_networkx_nodes(G,pos,node_color=colores_nodos,node_size=250,alpha=0.8)
nx.draw_networkx_edges(G,pos)
nx.draw_networkx_labels(G,pos,font_size=8)
plt.title('Queen-Under Pressure-Voz',fontsize=20)
plt.axis('off')
plt.show()


