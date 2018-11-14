import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd
import networkx as nx
import music21 as msc

#---------------------------------------------------------------------------------------------------------
#            FUNCIONES PARA ANALISIS DE MUSICA:
#---------------------------------------------------------------------------------------------------------

def f_xml2graph(cancion, nombre_parte=None):
        # Toma como input una canción (y el nombre de la parte o voz) y devuelve un grafo G
        
        # Cancion
        song = msc.converter.parse(cancion) # Lee la cancion, queda un elemento stream.Score
        
        Lp = len(song.parts) # Cantidad de partes (voces)
        lista_partes = list(np.zeros(Lp)) # Crea una lista donde se van a guardas los nombres de las partes
        
        for i,elem in enumerate(song.parts):
                lista_partes[i] = elem.partName # Guarda los nombres de las partes en la lista
        
        nombre_parte = nombre_parte or lista_partes[0] #Si no tuvo nombre_parte como input,toma la primera voz
        
        if not nombre_parte in lista_partes: #Si el nombre de la parte no esta en la lista,toma la primera voz
                part = song.parts[0]
                print(nombre_parte+' no está entre las partes: '+str(lista_partes)+'. Parte seleccionada: '+str(lista_partes[0]))
                # Ademas devuelve el "error" de que el nombre no esta entre las partes, y te dice que parte usa
        else:
                j = lista_partes.index(nombre_parte)
                part = song.parts[j]
                print('Parte seleccionada: '+str(lista_partes[j]))
                # Si el nombre si esta entre las partes, lo selecciona y tambien te dice que parte usa
        
        # Primer instrumento
        voz = part.getElementsByClass(msc.stream.Measure) # todos los compases de la parte voz seleccionada
        notas = [x for x in voz.flat if type(x)==msc.note.Note or type(x)==msc.note.Rest] # todas las notas incluyendo silencios en la voz analizada con offset 'absoluto' (flat)
        #notas = [x for x in voz.flat.notesAndRests] # esto también incluye acordes
        L = len(notas) # longitud de la voz en cantidad de figuras
        
        # Creamos el grafo que sera dirigido
        G = nx.DiGraph()
        
        # Nodos
        # Recorremos todas las notas de la voz, incluyendo silencios
        j=1 # contador de silencios, para asociarles una frecuencia cualquiera distinta a cada uno (para la posicion del grafico)
        for i,el in enumerate(notas):
                if isinstance(el,msc.note.Note):
                        nota_name = str(el.nameWithOctave)+'/'+str(el.quarterLength)
                        if not G.has_node(nota_name): # Si el grafo no tiene el nodo, lo agregamos con los atributos que se quieran
                                G.add_node(nota_name)
                                G.node[nota_name]['freq'] = el.pitch.frequency
                                G.node[nota_name]['octava'] = el.octave
                                G.node[nota_name]['duracion'] = el.quarterLength
                        notas[i] = nota_name
                
                elif isinstance(el,msc.note.Rest):
                        nota_name = str(el.name)+'/'+str(el.quarterLength)
                        if not G.has_node(nota_name):
                                G.add_node(nota_name)
                                G.node[nota_name]['freq'] = 8*j
                                G.node[nota_name]['octava'] = 1 # A los silencios se les asocia una octava cualquiera, tambien para el grafico
                                G.node[nota_name]['duracion'] = el.quarterLength
                                j+=1
                        notas[i] = nota_name
        
        # Enlaces pesados
        for i in range(L-1):  # recorremos desde la primera hasta la anteultima nota, uniendo sucesivas
                if G.has_edge(notas[i],notas[i+1]):
                        G[notas[i]][notas[i+1]]['weight']+=1 # si el enlace ya existe, se le agrega +1 al peso
                else:
                        G.add_edge(notas[i],notas[i+1],weight=1) # si el enlace no existe, se crea con peso 1
        
        return(G)

#-----------------------------------------------------------------------------------

def graficar(G):
        # Toma como input un grafo G y lo grafica
        # Para que funcione, el grafo debe tener como atributos freq, duracion y octava
        
        M = G.number_of_edges()
        N = G.number_of_nodes()
        nodos = G.nodes()
        freq_min = min(np.array(list(nx.get_node_attributes(G,'freq').values())))
        pos = dict()
        
        for nodo in nodos:
                f = G.node[nodo]['freq']
                d = G.node[nodo]['duracion']
                theta = 2*np.pi * np.log2(f/freq_min)
                x = np.cos(theta)*f/freq_min*(1+d/4)
                y = np.sin(theta)*f/freq_min*(1+d/4)
                pos[nodo] = np.array([x,y])
        
        octavas = np.array(list(nx.get_node_attributes(G,'octava').values()))
        oct_min = min(octavas)
        oct_max = max(octavas)
        colores_oct_nro = (octavas-oct_min)/(oct_max-oct_min)
        colores_oct = [ cm.summer(x) for x in colores_oct_nro ]
        
        nx.draw_networkx_nodes(G,pos,node_list=nodos,node_color=colores_oct,node_size=800,alpha=1)
        nx.draw_networkx_labels(G,pos)
        edges = nx.draw_networkx_edges(G,pos,width=3)
        weights = list(nx.get_edge_attributes(G,'weight').values())
        weight_max = max(weights)
        for i in range(M):
                edges[i].set_alpha(weights[i]/weight_max)       # set alpha value for each edge
        plt.axis('off')
        plt.show()

#-----------------------------------------------------------------------------------
def ql_2_fig(ql):
    #Toma como input un valor de quarterlength y devuelve un str
    #que es la figura que le corresponde.
    #Nota: falta completar con otros valores seguramente.
    figura='ninguna'
    if float(ql)==4:
        figura='redonda'
    elif float(ql)==3:
        figura='blanca puntillo'
    elif float(ql)==2:
        figura='blanca'
    elif float(ql)==1.5:
        figura='negra puntillo'
    elif float(ql)==1:
        figura='negra'
    elif float(ql)==0.5:
        figura='corchea'
    elif float(ql)==0.25:
        figura='semicorchea'
    elif float(ql)==float(1/3):
        figura='tresillo de corchea'
    return figura

#-----------------------------------------------------------------------------------
def f_motifs_rhytmic(cancion,length,nombre_parte=None):
        #Toma como input una canción (y el nombre de la parte o voz) y devuelve los motifs
        #ritmicos de tamano length y la frecuencia de aparicion de cada uno.
        
        #Cancion
        song = msc.converter.parse(cancion) # Lee la cancion, queda un elemento stream.Score
        
        Lp = len(song.parts) #Cantidad de partes (voces)
        lista_partes = list(np.zeros(Lp)) #Crea una lista donde se van a guardas los nombres de las partes
        
        for i,elem in enumerate(song.parts):
                lista_partes[i] = elem.partName #Guarda los nombres de las partes en la lista
                
        nombre_parte=nombre_parte or lista_partes[0]
        
        if not nombre_parte in lista_partes: #Si el nombre de la parte no esta en la lista, toma la primera voz
                part = song.parts[0]
                print(nombre_parte+' no está entre las partes: '+str(lista_partes)+'. Parte seleccionada: '+str(lista_partes[0]))
                #Ademas devuelve el "error" de que el nombre no esta entre las partes, y te dice que parte usa
        else:
                j = lista_partes.index(nombre_parte)
                part = song.parts[j]
                print('Parte seleccionada: '+str(lista_partes[j]))
                #Si el nombre sí esta entre las partes, lo selecciona y tambien te dice que parte usa
                
        #Primer instrumento
        voz = part.getElementsByClass(msc.stream.Measure)#todos los compases de la parte voz seleccionada
        motifs=[]
        rhytms=[]
        frecuencias=[]

        #Para eso vamos a recorrer la partitura y guardando solo el dato de la duracion rítmica en la lista llamada rhytms:

        for i,el in enumerate(voz.flat):
            if isinstance(el,msc.note.Note):
                rhytms.append(float(el.quarterLength))
            elif isinstance(el,msc.note.Rest):
                rhytms.append(float(el.quarterLength))

        #Una vez creada la lista rhytm empiezo a recorrerla tomando grupos de notas de tamano segun lo indique en length:
        motif_index=0
        for r in range(0,len(rhytms)-length+1):
            motif=[]
            for l in range(0,length):
                #motif.append(rhytms[r+l])
                motif.append(ql_2_fig(rhytms[r+l]))#aca se le puede descomentar para que guarde los motifs con el nombre de la figura usando la funcion ql_2_fig.
            if (motif in motifs)==False:
                motif_index+=1
                motifs.append(motif)
                frecuencias.append(1)
            else:
                frecuencias[motif_index-1]+=1
        motifs_rhytmic=motifs

        return (motifs_rhytmic,frecuencias)
