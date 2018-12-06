import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd
import networkx as nx
import music21 as msc
import math
import copy
from matplotlib.font_manager import FontProperties
import random

#---------------------------------------------------------------------------------------------------------
#            FUNCIONES PARA ANALISIS DE MUSICA:
#---------------------------------------------------------------------------------------------------------
#Lista de funciones:
# f_xml2graph (cancion, nombre_parte=0,modelo='melodia')
# graficar (G, color_map='rainbow',layout='espiral', labels=False)
# ql_2_fig (ql)
# f_motifs_rhytmic (cancion,length,nombre_parte=0)
# f_motifs_tonal (cancion,length,nombre_parte=0)
# f_grado_dist_M (G)
# f_grado_dist_R (G)
# f_tabla (G,nombre)
# f_xml2graph_armonia (cancion, index)
# f_armon (cancion, indexes)
# f_graficar_armonias_undirected(G, color_map='rainbow',layout='espiral')
# f_grafo_armonias_directed(Armonias)
# f_dist_escalas (cancion, nombre_parte=0)
# f_full_graph(path)
# f_hierarchy(G)
# f_transitivity_motifs(G)
# f_rewiring_directed(G)
# f_voices(path, modelo='melodia')
# f_merge(dict1,dict2,modelo='directed')
# f_graficar_armonias_directed(G, layout='random',labels=False)
# f_simultaneidad(cancion, [indexi,indexj])
# f_voice2nameabrev(instrumento_name)
# f_conect(G,H,cancion,indexes):
# f_get_layers_position
# f_graficar_2dy3d
# random_walk_1_M(G,k)
# f_compose(G,H)
#-----------------------------------------------------------------------------------

def f_xml2graph(cancion, nombre_parte=0,modelo='melodia'): 
    # Toma como input una canción y devuelve un grafo o una lista de grafos si se repite el nombre
    # cancion puede ser la ubicacion del archivo (str) o el Score de music21
    # Opcional: nombre_parte puede ser el nombre (str) o su indice
    # Opcional: modelo puede ser melodia o ritmo
    
    # Cancion
    if type(cancion)==msc.stream.Score:
        song = cancion # Si la cancion ya es un stream.Score, se queda con eso
    else:
        song = msc.converter.parse(cancion) # Sino lee la partitura, queda un elemento stream.Score

    # Lista de nombres de las partes
    Lp = len(song.parts) # Cantidad de partes (voces)
    lista_partes = list(np.zeros(Lp))
    for i,elem in enumerate(song.parts):
        lista_partes[i] = elem.partName # Guarda los nombres de las partes en la lista

    # Seleccion de la parte a usar
    # Si el input es el indice (int) intenta encontrarlo entre las partes; si no lo encuentra, selecciona la primera voz
    if type(nombre_parte)==int:
        try:
            part = song.parts[nombre_parte]
            #print('Partes: '+str(lista_partes)+'. Parte seleccionada: '+str(lista_partes[nombre_parte]))
        except IndexError:
            part = song.parts[0]
            #print(nombre_parte+' no es un índice aceptable. Partes: '+str(lista_partes)+'. Parte seleccionada: '+str(lista_partes[0]))
    # Si el input es nombre (str) y no está entre las partes, selecciona la primera voz
    elif not nombre_parte in lista_partes: 
        part = song.parts[0]
        #print(nombre_parte+' no está entre las partes: '+str(lista_partes)+'. Parte seleccionada: '+str(lista_partes[0]))
    else:
        indexes = [index for index, name in enumerate(lista_partes) if name == nombre_parte]
        if len(indexes)==1:
            part = song.parts[indexes[0]]
        else:
            part = []
            for j in indexes:
                part.append(song.parts[j])
        #print('Partes: '+str(lista_partes)+'. Parte(s) seleccionada(s): '+str([lista_partes[i] for i in indexes]))
    # En cualquier caso, devuelve una lista de partes y cuál selecciona (y aclaraciones neccesarias de cada caso)

    # Crea la(s) voz(ces) analizada(s) (todos los compases) y se queda con
    # todas las notas incluyendo silencios con offset 'absoluto' (flat)
    # IMPORTANTE: Si la voz contiene sonidos simultáneos, sólo se queda con el más agudo
    if type(part) == list:
        voz = []
        for parte in part:
            voz.append(parte.getElementsByClass(msc.stream.Measure))
        lista_notas = []
        for voice in voz:
            notes = [x for x in voice.flat if type(x)==msc.note.Note or type(x)==msc.note.Rest]
            tiempos = [x.offset for x in voice.flat if type(x)==msc.note.Note or type(x)==msc.note.Rest]
            indices = []

            for i in range(len(notes)-1):
                if tiempos[i+1] == tiempos[i]:
                    if type(notes[i+1])==msc.note.Rest:
                        indices.append(i+1)
                    elif type(notes[i])==msc.note.Rest:
                        indices.append(i)
                    elif (notes[i+1].pitch.frequency > notes[i].pitch.frequency):
                        indices.append(i)
                    else:
                        indices.append(i+1)

            indices = [x for x in indices[::-1]]

            for index in indices:
                del notes[index]
            lista_notas.append(notes)

    else:
        voz = part.getElementsByClass(msc.stream.Measure)
        notas = [x for x in voz.flat if type(x)==msc.note.Note or type(x)==msc.note.Rest]
        tiempos = [x.offset for x in voz.flat if type(x)==msc.note.Note or type(x)==msc.note.Rest]
        indices = []

        for i in range(len(notas)-1):
            if tiempos[i+1] == tiempos[i]:
                if type(notas[i+1])==msc.note.Rest:
                    indices.append(i+1)
                elif type(notas[i])==msc.note.Rest:
                    indices.append(i)
                elif (notas[i+1].pitch.frequency > notas[i].pitch.frequency):
                    indices.append(i)
                else:
                    indices.append(i+1)

        indices = [x for x in indices[::-1]]

        for index in indices:
            del notas[index]
    # Crea una lista de notas y silencios (notas) o una lista de listas (por cada voz) (lista_notas)

    # Crea el grafo dirigido, o lista de grafos dirigidos Gs si hay mas de una voz
    if type(part) == list:
        Gs = [] # Va a ser una lista de grafos, uno por cada voz analizada
        for notas in lista_notas:
            if len(notas)==0:
                continue
            G = nx.DiGraph()

            if modelo == 'melodia':
                oct_min = min(elem.octave for elem in notas if type(elem)==msc.note.Note) # para asignarle una octava a los silencios

                # Nodos
                # Recorremos todas las notas de la voz, incluyendo silencios
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
                        nota_name = str('rest')
                        if not G.has_node(nota_name):
                            G.add_node(nota_name)
                            G.node[nota_name]['freq'] = 2**(1/(2*np.pi))*20
                            #G.node[nota_name]['octava'] = oct_min-1 # A los silencios se les asocia una octava menos que las notas, para el grafico
                            G.node[nota_name]['octava'] = 0 # Aca cambie esto para que le asignemos octava 0 y asi no cambia de voz a voz.
                            G.node[nota_name]['duracion'] = 1
                        notas[i] = nota_name

            elif modelo == 'ritmo':
                # Si solo interesa el ritmo, agregamos los sonidos en la octava 1 y las figuras en la 2, para el grafico
                # Además, agregamos una frec inventada en funcion de la duracion, para poder usar la funcion graficar
                d_max = max( [el.quarterLength for el in notas] )
                d_min = min( [el.quarterLength for el in notas] )
                for i,el in enumerate(notas):
                    if isinstance(el,msc.note.Note):
                        nota_name = str(el.quarterLength)
                        if not G.has_node(nota_name):
                            G.add_node(nota_name)
                            d = el.quarterLength
                            G.node[nota_name]['freq'] = 2**(d/(2*np.pi))*20
                            G.node[nota_name]['octava'] = 1
                            G.node[nota_name]['duracion'] = d
                        notas[i] = nota_name

                    elif isinstance(el,msc.note.Rest):
                        nota_name = str(el.name)+'/'+str(el.quarterLength)
                        if not G.has_node(nota_name):
                            G.add_node(nota_name)
                            d = el.quarterLength
                            G.node[nota_name]['freq'] = 2**(d/(2*np.pi))*20
                            G.node[nota_name]['octava'] = 0 # Aca cambie esto para que le asignemos octava 0 y asi no cambia de voz a voz.
                            G.node[nota_name]['duracion'] = d
                        notas[i] = nota_name

            # Enlaces pesados
            L = len(notas)
            for i in range(L-1):  # recorremos desde la primera hasta la anteultima nota, uniendo sucesivas
                if G.has_edge(notas[i],notas[i+1]):
                    G[notas[i]][notas[i+1]]['weight']+=1 # si el enlace ya existe, se le agrega +1 al peso
                else:
                    G.add_edge(notas[i],notas[i+1],weight=1) # si el enlace no existe, se crea con peso 1
            # Finalmente, agrego atributo posicion (se debe hacer al final, porque necesita todos los nodos):
            #Comente esta parte para que la posicion la asigne unicamente cuando terminamos de mergear y vamos a graficar
            '''
            nodos = G.nodes()
            freq_min = min(np.array(list(nx.get_node_attributes(G,'freq').values())))
            for nodo in nodos:
                f = G.node[nodo]['freq']
                d = G.node[nodo]['duracion']
                theta = 2*np.pi * np.log2(f/freq_min)
                x = np.cos(theta)*f/freq_min*(1+d/4)
                y = np.sin(theta)*f/freq_min*(1+d/4)
                G.node[nodo]['x'] = x
                G.node[nodo]['y'] = y
            '''
            Gs.append(G)
        if len(Gs)==1:
            Gs = Gs[0]
    elif len([x for x in notas if type(x)==msc.note.Note])>0:
        G = nx.DiGraph()

        if modelo == 'melodia':
            oct_min = min(elem.octave for elem in notas if type(elem)==msc.note.Note) # para asignarle una octava a los silencios

            # Nodos
            # Recorremos todas las notas de la voz, incluyendo silencios
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
                    nota_name = str('rest')
                    if not G.has_node(nota_name):
                        G.add_node(nota_name)
                        G.node[nota_name]['freq'] = 2**(1/(2*np.pi))*20
                        #G.node[nota_name]['octava'] = oct_min-1 # A los silencios se les asocia una octava menos que las notas, para el grafico
                        G.node[nota_name]['octava'] = 0 # Aca cambie esto para que le asignemos octava 0 y asi no cambia de voz a voz.
                        G.node[nota_name]['duracion'] = 1
                    notas[i] = nota_name

        elif modelo == 'ritmo':
            # Si solo interesa el ritmo, agregamos los sonidos en la octava 1 y las figuras en la 2, para el grafico
            # Además, agregamos una frec inventada en funcion de la duracion, para poder usar la funcion graficar
            d_max = max( [el.quarterLength for el in notas] )
            d_min = min( [el.quarterLength for el in notas] )
            for i,el in enumerate(notas):
                if isinstance(el,msc.note.Note):
                    nota_name = str(el.quarterLength)
                    if not G.has_node(nota_name):
                        G.add_node(nota_name)
                        d = el.quarterLength
                        G.node[nota_name]['freq'] = 2**(d/(2*np.pi))*20
                        G.node[nota_name]['octava'] = 1
                        G.node[nota_name]['duracion'] = d
                    notas[i] = nota_name

                elif isinstance(el,msc.note.Rest):
                    nota_name = str(el.name)+'/'+str(el.quarterLength)
                    if not G.has_node(nota_name):
                        G.add_node(nota_name)
                        d = el.quarterLength
                        G.node[nota_name]['freq'] = 2**(d/(2*np.pi))*20
                        G.node[nota_name]['octava'] = 0
                        G.node[nota_name]['duracion'] = d
                    notas[i] = nota_name

        # Enlaces pesados
        L = len(notas)
        for i in range(L-1):  # recorremos desde la primera hasta la anteultima nota, uniendo sucesivas
            if G.has_edge(notas[i],notas[i+1]):
                G[notas[i]][notas[i+1]]['weight']+=1 # si el enlace ya existe, se le agrega +1 al peso
            else:
                G.add_edge(notas[i],notas[i+1],weight=1) # si el enlace no existe, se crea con peso 1
        # Finalmente, agrego atributo posicion (se debe hacer al final, porque necesita todos los nodos):
        '''
        nodos = G.nodes()
        freq_min = min(np.array(list(nx.get_node_attributes(G,'freq').values())))
        for nodo in nodos:
            f = G.node[nodo]['freq']
            d = G.node[nodo]['duracion']
            theta = 2*np.pi * np.log2(f/freq_min)
            x = np.cos(theta)*f/freq_min*(1+d/4)
            y = np.sin(theta)*f/freq_min*(1+d/4)
            G.node[nodo]['x'] = x
            G.node[nodo]['y'] = y
        '''
        Gs = G
    else:
        return None
    return(Gs)
#-----------------------------------------------------------------------------------

def graficar(G, color_map='rainbow',layout='espiral', labels=False):
	# Toma como input un grafo G y lo grafica
	# Para que funcione, los nodos deben tener como atributos freq, duracion y octava
	
	M = G.number_of_edges()
	N = G.number_of_nodes()
	nodos = G.nodes()
	freq_min = min(np.array(list(nx.get_node_attributes(G,'freq').values())))
	pos = dict()
	
	for nodo in nodos:
		f = G.node[nodo]['freq']
		d = G.node[nodo]['duracion']
		theta = 2*np.pi * np.log2(f/freq_min)
		if layout=='espiral':
			x = np.cos(theta)*f/freq_min*(1+d/4)
			y = np.sin(theta)*f/freq_min*(1+d/4)
			pos[nodo] = np.array([x,y])
		elif layout=='circular':
			nro_oct = G.node[nodo]['octava']
			x = np.cos(theta)*nro_oct*(1+d/12)
			y = np.sin(theta)*nro_oct*(1+d/12)
			pos[nodo] = np.array([x,y])
	
	#octavas = np.array(list(nx.get_node_attributes(G,'octava').values())) #chicos comente esta linea porque no me estaba mapeando bien los colores.
	octavas = np.array([G.node[nodo]['octava'] for i,nodo in enumerate(nodos)]) #agregue esta linea en reemplazo de la anterior.
	oct_min = min(octavas)
	oct_max = max(octavas)
	colores_oct_nro = (octavas - oct_min)/(oct_max - oct_min)
	m = cm.ScalarMappable(norm=None, cmap=color_map)
	colores_oct = m.to_rgba(colores_oct_nro)
	
	#Grafico
	#grados = np.array([d for d in dict(nx.degree(G)).values()])#comente esta linea y agregue la de arriba porque estaba mapeando mal los grados.
	grados = np.array([G.degree(d) for d in nodos])
	deg_max = max(grados)
	n_size = (grados/deg_max)
	nx.draw_networkx_nodes(G,pos,node_list=nodos,node_color=colores_oct,node_size=500*n_size)
	
	if labels==True:
		nx.draw_networkx_labels(G,pos)
	
	#Enlaces
	edges = G.edges()
	weights = np.array(list(nx.get_edge_attributes(G,'weight').values()))
	weight_max = max(weights)
	alphas = (weights/weight_max)**(1./2.)
	for e,edge in enumerate(edges):
		nx.draw_networkx_edges(G, pos, edgelist=[edge], width=3,alpha=alphas[e])
		
	plt.axis('off')
	#plt.show()
#-----------------------------------------------------------------------------------

def ql_2_fig(ql):
    #Toma como input un valor de quarterlength y devuelve un str
    #que es la figura que le corresponde.
    #Nota: falta completar con otros valores seguramente.
    #Si ese valor no lo encuentra devuelve 'no está en la lista' y lo agregamos
    #a mano en la lista.

    figura='no está en la lista'
    
    quarter_lengths=[1/8,0.5/3,0.25,1.0/3,0.5,2.0/3,0.75,1.0,1.25,1.5,1.75,2.0,2.25,2.5,2.75,3.0,3.25,3.5,3.75,4.0]
    figuras=['fusa','tresillo de semicorchea','semicorchea','tresillo de corchea','corchea','negra en tresillo','corchea puntillo','negra','negra semicorchea','negra puntillo','negra doble puntillo','blanca','blanca semicorchea','blanca corchea','blanca corchea puntillo','blanca puntillo','blanca puntillo semicorchea','blanca puntillo corchea','blanca puntillo corchea puntillo','redonda']
    figuras_abrev=['f','ts','s','tc','c','tn','cp','n','ns','np','npp','b','bs','bc','bcp','bp','bps','bpc','bpcp','r']
    index=indice(quarter_lengths,ql)
    if type(index) is not str:
        figura=figuras_abrev[index]
    else:
        figura=index
    
    return figura

def indice(a_list, value):
    try:
        return a_list.index(value)
    except ValueError:
        return 'no está en la lista'
#-----------------------------------------------------------------------------------

def f_motifs_rhytmic(cancion,length,nombre_parte=0):
	#Toma como input una canción (y el nombre de la parte o voz) y devuelve los motifs
	#ritmicos de tamano length y la frecuencia de aparicion de cada uno.
	#Realiza histograma, utilizando un cierto motif_umbral(empezamos a considerarlo motif
	#a partir de una cierta frecuencia en adelante)
	
	#Cancion
	if type(cancion)==msc.stream.Score:
		song = cancion # Si la cancion ya es un stream.Score, se queda con eso
	else:
		song = msc.converter.parse(cancion) # Sino, lee la partitura, queda un elemento stream.Score
	
	Lp = len(song.parts) #Cantidad de partes (voces)
	lista_partes = list(np.zeros(Lp)) #Crea una lista donde se van a guardas los nombres de las partes
	
	for i,elem in enumerate(song.parts):
		lista_partes[i] = elem.partName #Guarda los nombres de las partes en la lista
	
	# Seleccion de la parte a usar
	# Si el input es el indice (int) intenta encontrarlo entre las partes; si no lo encuentra, selecciona la primera voz
	if type(nombre_parte)==int:
		try:
			part = song.parts[nombre_parte]
			print('Partes: '+str(lista_partes)+'. Parte seleccionada: '+str(lista_partes[nombre_parte]))
		except IndexError:
			part = song.parts[0]
			print(nombre_parte+' no es un índice aceptable. Partes: '+str(lista_partes)+'. Parte seleccionada: '+str(lista_partes[0]))
	# Si el input es nombre (str) y no está entre las partes, selecciona la primera voz
	elif not nombre_parte in lista_partes: 
		part = song.parts[0]
		print(nombre_parte+' no está entre las partes: '+str(lista_partes)+'. Parte seleccionada: '+str(lista_partes[0]))
	else:
		j = lista_partes.index(nombre_parte)
		part = song.parts[j]
		print('Partes: '+str(lista_partes)+'. Parte seleccionada: '+str(lista_partes[j]))
	# En cualquier caso, devuelve una lista de partes y cuál selecciona (y aclaraciones neccesarias de cada caso)

	#Primer instrumento
	voz = part.getElementsByClass(msc.stream.Measure)#todos los compases de la parte voz seleccionada
	motifs=[]
	frecuencias=[]

	#Para eso vamos a recorrer la partitura y guardando solo el dato de la duracion rítmica en la lista llamada rhytms.
	#Esto lo voy a hacer para cada compas y luego vacio la lista rhytms_compas
	rhytms_compas=[]
	for c,compas in enumerate(voz):
		for i,el in enumerate(compas):
			if isinstance(el,msc.note.Note):
				rhytms_compas.append(float(el.quarterLength))
			elif isinstance(el,msc.note.Rest):
				rhytms_compas.append('rest/'+ql_2_fig(float(el.quarterLength)))

		#Una vez creada la lista rhytm_compas empiezo a recorrerla tomando grupos de notas de tamano segun lo indique en length:
		for r in range(0,len(rhytms_compas)-length+1):
			motif=[]
			for l in range(0,length):
				#motif.append(rhytms[r+l])
				if  type(rhytms_compas[r+l]) is not str:
                                    #motif.append(rhytms_compas[r+l])
				    motif.append(ql_2_fig(rhytms_compas[r+l]))#aca se le puede descomentar para que guarde los motifs con el nombre de la figura usando la funcion ql_2_fig.  
				else:
				    motif.append(rhytms_compas[r+l])
			#una vez tengo armado un motif me fijo si esta en la lista motifs
			if (motif in motifs)==False:
				motifs.append(motif)
				frecuencias.append(1)
			else:
				indice_motif=motifs.index(motif)
				frecuencias[indice_motif]+=1
		#vacio la lista
		rhytms_compas=[]
	motifs_rhytmic=motifs

	#Grafico
	#plt.figure
	#yTick_position=[]
	#yTick_name=[]
	#contador=-1
	#contador_tick=-0.5
	#motif_umbral=0
	#for m,motif in enumerate(motifs_rhytmic):
	#	if frecuencias[m]>motif_umbral:
	#		contador+=1
	#		contador_tick+=1
	#		plt.barh(contador,frecuencias[m],color='red')
	#		yTick_position.append(contador_tick)
	#		yTick_name.append(motif)
	#plt.yticks(yTick_position,yTick_name, rotation=0,fontsize=10)
	#plt.title('Rhytmics '+str(length)+'-Motifs',fontsize=20)
	#plt.show() 
	
	return (motifs_rhytmic,frecuencias)
#-----------------------------------------------------------------------------------

def f_motifs_tonal(cancion,length,nombre_parte=0):
	#Toma como input una canción (y el nombre de la parte o voz) y devuelve los motifs
	#tonales de tamano length y la frecuencia de aparicion de cada uno.
	#Realiza histograma, utilizando un cierto motif_umbral(empezamos a considerarlo motif
	#a partir de una cierta frecuencia en adelante)
	
	#Cancion
	if type(cancion)==msc.stream.Score:
		song = cancion # Si la cancion ya es un stream.Score, se queda con eso
	else:
		song = msc.converter.parse(cancion) # Sino, lee la partitura, queda un elemento stream.Score
	
	Lp = len(song.parts) #Cantidad de partes (voces)
	lista_partes = list(np.zeros(Lp)) #Crea una lista donde se van a guardas los nombres de las partes
	
	for i,elem in enumerate(song.parts):
		lista_partes[i] = elem.partName #Guarda los nombres de las partes en la lista
			
	# Seleccion de la parte a usar
	# Si el input es el indice (int) intenta encontrarlo entre las partes; si no lo encuentra, selecciona la primera voz
	if type(nombre_parte)==int:
		try:
			part = song.parts[nombre_parte]
			print('Partes: '+str(lista_partes)+'. Parte seleccionada: '+str(lista_partes[nombre_parte]))
		except IndexError:
			part = song.parts[0]
			print(nombre_parte+' no es un índice aceptable. Partes: '+str(lista_partes)+'. Parte seleccionada: '+str(lista_partes[0]))
	# Si el input es nombre (str) y no está entre las partes, selecciona la primera voz
	elif not nombre_parte in lista_partes: 
		part = song.parts[0]
		print(nombre_parte+' no está entre las partes: '+str(lista_partes)+'. Parte seleccionada: '+str(lista_partes[0]))
	else:
		j = lista_partes.index(nombre_parte)
		part = song.parts[j]
		print('Partes: '+str(lista_partes)+'. Parte seleccionada: '+str(lista_partes[j]))
	# En cualquier caso, devuelve una lista de partes y cuál selecciona (y aclaraciones neccesarias de cada caso)
			
	#Primer instrumento
	voz = part.getElementsByClass(msc.stream.Measure)#todos los compases de la parte voz seleccionada
	motifs=[]
	tones_compas=[]
	frecuencias=[]

	#Para eso vamos a recorrer la partitura y guardando solo el dato de la duracion rítmica en la lista llamada rhytms.
	#Esto lo voy a hacer para cada compas y luego vacio la lista tones_compas
	tones_compas=[]
	for c,compas in enumerate(voz):
		for i,el in enumerate(compas):
			if isinstance(el,msc.note.Note):
				tone_name=str(el.name)+str(el.octave)
				tones_compas.append(tone_name)
			elif isinstance(el,msc.note.Rest):
				tones_compas.append('rest')

		#Una vez creada la lista rhytm_compas empiezo a recorrerla tomando grupos de notas de tamano segun lo indique en length:
		for r in range(0,len(tones_compas)-length+1):
			motif=[]
			for l in range(0,length):
				#motif.append(rhytms[r+l])
				if  type(tones_compas[r+l]) is not str:
					motif.append(ql_2_fig(tones_compas[r+l]))#aca se le puede descomentar para que guarde los motifs con el nombre de la figura usando la funcion ql_2_fig.
				else:
					motif.append(tones_compas[r+l])
			#una vez tengo armado un motif me fijo si esta en la lista motifs
			if (motif in motifs)==False:
				motifs.append(motif)
				frecuencias.append(1)
			else:
				indice_motif=motifs.index(motif)
				frecuencias[indice_motif]+=1
		#vacio la lista
		tones_compas=[]
	motifs_tonal=motifs

	#Grafico
	#plt.figure
	yTick_position=[]
	yTick_name=[]
	contador=-1
	contador_tick=-0.5
	motif_umbral=0
	for m,motif in enumerate(motifs_tonal):
		if frecuencias[m]>motif_umbral:
			contador+=1
			contador_tick+=1
			plt.barh(contador,frecuencias[m],color='blue')
			yTick_position.append(contador_tick)
			yTick_name.append(motif)
	plt.yticks(yTick_position,yTick_name, rotation=0,fontsize=10)
	plt.title('Tonals '+str(length)+'-Motifs',fontsize=20)
	#plt.show()       

	return (motifs_tonal,frecuencias)
#-----------------------------------------------------------------------------------

def f_grado_dist(G,modelo): #el grafo puede ser dirigido o no dirigido
    
    H=G.copy()
    nodos=H.nodes() 
    N=len(nodos)
    if modelo == 'undirected':
        
        #calculo los grados que salen y entran de cada nodo
        kgrados = [H.degree(nodo) for nodo in nodos]
    
        # Contamos la cantidad de nodos que tienen un cierto k_grado, usando la funcion np.unique()
        # Guardamos el resultado en la variable histograma
        histograma = np.unique(kgrados,return_counts=True)
        k = histograma[0] # grados
        pk = histograma[1]/float(N) # pk = Nk/N, donde N es el numero total de nodos (cuentas normalizadas)
        maxgrado = max(k) #maximo grado

        logbin = np.logspace(0,np.log10(maxgrado),num=20,endpoint=True,base=10) # bineado en base 10
        histograma_logbin = np.histogram(kgrados,bins=logbin,density=False)

        # Normalizamos por el ancho de los bines y creamos el vector bin_centros
        bin_centros = []
        pk_logbin = []

        for i in range(len(logbin)-1):
            bin_centros.append((logbin[i+1]+logbin[i])/2)
            bin_ancho = logbin[i+1]-logbin[i]
            pk_logbin.append(histograma_logbin[0][i]/(bin_ancho*N)) #normalizamos por el ancho del bin y por el numero total de nodos

        fig=plt.figure(figsize=(8,8))
        plt.suptitle('Bin log - Escala log',fontsize=25)
        plt.plot(bin_centros,pk_logbin,'bo')
        plt.xlabel('$log(k)$',fontsize=20)
        plt.xscale('log')
        plt.ylabel('$log(p_{k})$',fontsize=20)
        plt.yscale('log')
        plt.title('Bin log - Escala log',fontsize=20)
    
    elif modelo == 'directed':
        #calculo los grados que salen y entran de cada nodo
        kgrados_out = [H.out_degree(nodo) for nodo in nodos]
        kgrados_in = [H.in_degree(nodo) for nodo in nodos]

        # Contamos la cantidad de nodos que tienen un cierto k_grado, usando la funcion np.unique()
        # Guardamos el resultado en la variable histograma
        histograma_out = np.unique(kgrados_out,return_counts=True)
        k_out = histograma_out[0] # grados
        pk_out = histograma_out[1]/float(N) # pk = Nk/N, donde N es el numero total de nodos (cuentas normalizadas)
        maxgrado_out = max(k_out) #maximo grado

        logbin_out = np.logspace(0,np.log10(maxgrado_out),num=20,endpoint=True,base=10) # bineado en base 10
        histograma_logbin_out = np.histogram(kgrados_out,bins=logbin_out,density=False)

        # Normalizamos por el ancho de los bines y creamos el vector bin_centros
        bin_centros_out = []
        pk_logbin_out = []
        for i in range(len(logbin_out)-1):
            bin_centros_out.append((logbin_out[i+1]+logbin_out[i])/2)
            bin_ancho = logbin_out[i+1]-logbin_out[i]
            pk_logbin_out.append(histograma_logbin_out[0][i]/(bin_ancho*N)) #normalizamos por el ancho del bin y por el numero total de nodos
        #idem in
        histograma_in = np.unique(kgrados_in,return_counts=True)
        k_in = histograma_in[0] # grados
        pk_in = histograma_in[1]/float(N) # pk = Nk/N, donde N es el numero total de nodos (cuentas normalizadas)
        maxgrado_in = max(k_in) #maximo grado

        logbin_in = np.logspace(0,np.log10(maxgrado_in),num=20,endpoint=True,base=10) # bineado en base 10
        histograma_logbin_in = np.histogram(kgrados_in,bins=logbin_in,density=False)

        # Normalizamos por el ancho de los bines y creamos el vector bin_centros
        bin_centros_in = []
        pk_logbin_in = []
        for i in range(len(logbin_in)-1):
            bin_centros_in.append((logbin_in[i+1]+logbin_in[i])/2)
            bin_ancho = logbin_in[i+1]-logbin_in[i]
            pk_logbin_in.append(histograma_logbin_in[0][i]/(bin_ancho*N)) #normalizamos por el ancho del bin y por el numero total de nodos
        # Escala logaritmica en ambos ejes
        fig=plt.figure(figsize=(16,8))
        plt.suptitle('Bin log - Escala log',fontsize=25)

        plt.subplot(1, 2, 1)
        plt.plot(bin_centros_out,pk_logbin_out,'bo')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('$log (k)$',fontsize=20)
        plt.ylabel('$log (p_{k})$',fontsize=20)
        plt.title('Enlaces salientes',fontsize=20)

        plt.subplot(1, 2, 2)
        plt.plot(bin_centros_in,pk_logbin_in,'bo')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('$log (k)$', fontsize=20)
        plt.ylabel('$log (p_{k})$', fontsize=20)
        plt.title('Enlaces entrantes',fontsize=20)

    #plt.show()
   
    #return(fig)
#-----------------------------------------------------------------------------------

def f_tabla(G,nombre):
    
    H=G.copy()
    nodos=H.nodes() 
    N=len(nodos)
    # El grado medio se puede calcular mediante k_mean=2m/n, donde m es la cantidad de enlaces total y n la cant de nodos
    nodes = H.number_of_nodes()
    enlaces = H.number_of_edges()
    
    if nx.is_directed(H)==False:
        K_mean = round(2*float(enlaces)/float(nodes))
    else:
        K_mean = 'NaN'
    
    if nx.is_directed(H)==True:
        #calculo los grados que salen y entran de cada nodo
        kgrados_out = [H.out_degree(nodo) for nodo in nodos]
        kgrados_in = [H.in_degree(nodo) for nodo in nodos]
        #los hacemos para el out
        histograma_out = np.unique(kgrados_out,return_counts=True) #pesamos los grados con los valores de su histograma
        k_out = histograma_out[0] # grados sin repetir
        pk_out= histograma_out[1] #frecuencia con la que aparecen sin normalizar
        n_out = len(k_out)
        
        K_mean_out= sum(kgrados_out)/float(N)
        K_min_out=np.unique(kgrados_out)[0]    
        K_max_out= np.unique(kgrados_out)[n_out-1]

        #idem para el in
        histograma_in = np.unique(kgrados_in,return_counts=True)
        k_in = histograma_in[0] # grados sin repetir
        pk_in= histograma_in[1] #frecuencia con la que aparecen sin normalizar
        n_in = len(k_in)

        #Lo hago de otra forma para verificar
        K_mean_in= np.dot(k_in,pk_in)/float(N)
        K_min_in=np.unique(kgrados_in)[0]    
        K_max_in= np.unique(kgrados_in)[n_in-1]
        
    else:
        K_mean_out= 'NaN'
        K_min_out= 'NaN'
        K_max_out= 'NaN'
        K_mean_in= 'NaN'
        K_max_in= 'NaN'
        K_min_in= 'NaN'
        
    # Densidad de la red uso density(G) (d = numero enlaces/enlaces maximos posibles)
    d = nx.density(H)

    if type(H) =='networkx.classes.digraph.DiGraph' or 'networkx.classes.graph.Graph':
        # Coef de clustering medio:
        # c_1 = #triangulos con vertice en 1 / triangulos posibles con vertice en 1
        # C_mean es el promedio de los c_i sobre todos los nodos de la red
        C_mean = nx.average_clustering(H)

        # Clausura transitiva de la red o Global Clustering o Transitividad:
        # C_g = 3*nx.triangles(G1) / sumatoria sobre (todos los posibles triangulos)
        C_gclust = nx.transitivity(H)
    else:
        C_mean= 'NaN'
        C_gclust= 'NaN'
        
    # Para calcular el diametro (la maxima longitud) primero hay que encontrar el mayor subgrafo conexo
    if nx.is_directed(G) == False:
        giant_graph = max(nx.connected_component_subgraphs(H),key=len)
        diam = nx.diameter(giant_graph)
    else:
        diam='NaN'
    
    dist=nx.average_shortest_path_length(H)
    dist=round(dist,2)
            

    # Creamos la tabla con las caracteristicas de las redes
    haytabla = pd.DataFrame({"Red":[nombre],
                        "Nodos":[nodes],
                        "Enlaces":[enlaces],
                        "<K>":[K_mean],
                        "<K_{in}>":[K_mean_in],
                        "<K_{out}>":[K_mean_out],
                        "K_{in} max":[K_max_in],
                        "K_{in} min":[K_min_in],
                        "K_{out} max":[K_max_out],
                        "K_{out} min":[K_min_out],
                        "Densidad":[d],
                        "<C_local>":[C_mean],
                        "C_global":[C_gclust],
                        "Diametro":[diam],
                        "<Min Dist>":[dist],
                       })
    
    
    return(haytabla)
#-----------------------------------------------------------------------------------

def f_xml2graph_armonia(cancion, index):
        #Toma como input una canción y el indice de la voz, y encuentra todas las armonias.
        #Obtiene un vector de armonias(2 o mas notas simultaneas) y el momento en el cual ocurrieron.
        #Se le puede pedir que armonias graficar, con mayor-tamano_armonia y menor_tamano_armonia

        #¿Que grafica?
        #1)Construye grafo no dirigido.(ver graficar_armonias_undirected)
        #2)Construye grafo dirigido donde los nodos son acordes.(ver graficar_armonias_directed)
        #3)Realiza histograma de las armonias

        #Notas: -si dos acordes estan ligados,los cuenta dos veces y no una vez sola.
        #       -si no encuentra armonias porque es una voz melodica pura devuelve un string :'No se encontraron armonias en esta voz'
        
        #Cancion
        song = msc.converter.parse(cancion)

      
        #Instrumento
        part=song.parts[index]
        print('Instrumento Seleccionado:'+str(part.partName))
        voz = part.getElementsByClass(msc.stream.Measure)#todos los compases dela parte voz seleccionada
        notas=[]#lista que va a contener cada uno de las notas. Si dos o mas notas so n simultaneas comparten el mismo offset
        tiempos=[]#lista que va a contener a los tiempos de cada una de las notas en la lista notas medidos desde el principio segun la cantidad offset
        frecuencias=[]#lista que va a contener la frecuencia de cada nota en la lista notas para despues ordenar en cada armonia de la mas grave a la mas aguda
        octavas=[]#lista que va a contener las octavas de cada nota

        #-------------------------------------------------------------------------------------------------------------------------------
        #1)Recorremos cada compas y nos guardamos todas las notas con sus respectivos tiempos.
        #-------------------------------------------------------------------------------------------------------------------------------
        for c,compas in enumerate(voz):
                #print('compas'+str(c)) #imprimo que compas es
                for i,el in enumerate(compas.flat):
                        isChord=str(type(el))=='<class \'music21.chord.Chord\'>' #me fijo si es un elemento del tipo acorde chord.Chord
                        if isinstance(el,msc.note.Note):#si es una nota
                                nota_name=str(el.nameWithOctave)
                                notas.append(nota_name)
                                tiempo_nota=float(compas.offset+el.offset)
                                tiempos.append(tiempo_nota)
                                frecuencias.append(el.pitch.frequency)
                                octavas.append(el.octave)
                        elif isinstance(el,msc.chord.Chord) & isChord==True:#si es un acorde pero no del tipo chorChord
                                for nc,noteChord in enumerate(el):
                                        nota_name=str(noteChord.nameWithOctave)
                                        notas.append(nota_name)
                                        tiempo_nota=float(compas.offset)+float(el.offset)
                                        tiempos.append(tiempo_nota)
                                        frecuencias.append(noteChord.pitch.frequency)
                                        octavas.append(noteChord.octave)

        #Incializamos el grafo en nx y en ig: los nodos seran notas. 
        G=nx.MultiGraph()
        #I=ig.Graph()
        Inodos=[]
        numero_nodo=-1
        
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        #2)Recorremos el vector de tiempos y nos fijamos todas las notas que caen en un mismo tiempo y las guardamos en la lista armonia y esta la guardamos en la lista armonias:
        #-----------------------------------------------------------------------------------------------------------------------------------------------------------------------
        armonias=[] #lista de armonia
        tiempos_unique=list(np.unique(tiempos))#vector de tiempos unicos donde ninguno se repite:
        armonia=[notas[0]] #al principio armonia tiene la primer nota de la cancion
        tiempo_actual=tiempos[0]
        tonos=[frecuencias[0]]#contendra las frecuencias de las notas en armonia, para ordenarlas de la mas grave a la mas aguda antes de guardar armonia
        octava=[octavas[0]]
        tiempos_armonias=[]
        menor_tamano_armonia=2 #indicamos cual es el tamano minimo de armonias a graficar
        mayor_tamano_armonia=5 #indicamos cual es el tamano maximo de armonias a graficar
        
        for t in range(1,len(tiempos)):
                tiempo_new=tiempos[t]
                if(tiempo_new-tiempo_actual)==0:#significa que seguimos en el mismo tiempo actual.
                        armonia.append(notas[t])
                        tonos.append(frecuencias[t])
                        octava.append(octavas[t])
                        tiempo_actual=tiempos[t]#acualizamos tiempos_old.
                else: #significa que cambiamos de tiempo.
                        tiempo_actual=tiempos[t]#actualizamos tiempos_old.
                        #reordenamos las notas en armonia de menor tono a mayor tono:
                        tonos_ordenados=np.sort(tonos)
                        octava_ordenada=[]
                        armonia_ordenada=[]
                        for f,tono in enumerate(tonos_ordenados):
                                indice=tonos.index(tono)
                                armonia_ordenada.append(armonia[indice])
                                octava_ordenada.append(octava[indice])
                        if (len(armonia)>=2 and len(set(armonia))==len(armonia)):#consideramos armonia si son 2 o mas sonidos que suenan simultaneos y 3 sonidos diferentes como minimo.
                                tiempos_armonias.append(tiempos[t-1])#es el tiempo en el que ocurrio esa armonia
                                armonias.append(armonia_ordenada)#guardamos la armonia ya ordenada 
                                #Agrego nodos al grafo:
                                for n,nota in enumerate(armonia_ordenada):
                                        if ((nota in Inodos)==False and len(armonia)>menor_tamano_armonia-1):#solo agrega el nodo si este no aparecio nunca y nodos en armnonias de mas de 2 notas.
                                                numero_nodo=numero_nodo+1
                                                #Grafo nx
                                                G.add_node(nota)
                                                G.node[nota]['freq']= tonos_ordenados[n]
                                                G.node[nota]['octava']= octava_ordenada[n]
                                                G.node[nota]['duracion']=4.0
                                                #Grafo ig
                                                #I.add_vertex(numero_nodo)
                                                #I.vs[numero_nodo]["name"]=nota
                                                Inodos.append(nota)
                                                #I.vs[numero_nodo]["freq"]= tonos_ordenados[n]
                                                #I.vs[numero_nodo]["octava"]= octava_ordenada[n]
                                                #I.vs[numero_nodo]["duracion"]=4.0
                                                
                        armonia=[notas[t]]
                        tonos=[frecuencias[t]]
                        octava=[octavas[t]]
                        

        #Agregamos los enlaces al grafo G e I:
        numero_enlace=-1
        color_dict= {"2":"purple","3": "blue", "4":"red","5":"green"}
        armonias_unicas=[list(i) for i in set(tuple(i) for i in armonias)]
        armonias_hist = [(x, armonias.count(x)) for x in armonias_unicas]
        
        colores_edges=[]
        pesos_edges=[]
        
        for a,armonia in enumerate(armonias_hist):
                tamano_armonia=len(armonia[0])
                if (tamano_armonia>menor_tamano_armonia-1 and tamano_armonia<mayor_tamano_armonia+1): #solo agregamos armonias de 2-3-4-5
                        for n in range (0,len(armonia[0])):
                                for m in range(n+1,len(armonia[0])):
                                        numero_enlace=numero_enlace+1
                                        edge_name=(armonia[0][n],armonia[0][m])
                                        #Grafo nx
                                        G.add_edge(armonia[0][n],armonia[0][m],key=str(numero_enlace),weight=armonia[1])
                                        G.add_edge(armonia[0][n],armonia[0][m],key=str(numero_enlace),tamano=tamano_armonia)
                                        G.add_edge(armonia[0][n],armonia[0][m],key=str(numero_enlace),color=color_dict[str(tamano_armonia)])
                                        #Grafo ig
                                        #I.add_edge(armonia[0][n],armonia[0][m])
                                        #I.es[numero_enlace]['tamano']=tamano_armonia
                                        #I.es[numero_enlace]['color']=color_dict[str(tamano_armonia)]
                                        #I.es[numero_enlace]['weigth']=armonia[1]
                                        pesos_edges.append(armonia[1])
                                        colores_edges.append(color_dict[str(tamano_armonia)])
                                        

        #1)Grafico con Igraph (no dirigido)
        f_graficar_armonias_undirected(G, color_map='rainbow',layout='espiral')
        #f_graficar_armonias_undirected_igraph(I, color_map='rainbow',layout='espiral')

        #2)Grafico con Igraph (dirigido):
        f_graficar_armonias_directed(armonias)
        #f_graficar_armonias_directed_igraph(armonias)

        #3)Histograma de armonias
        plt.figure
        yTick_position=[]
        yTick_name=[]
        contador=-1
        contador_tick=-0.5
        dtype = [('name', 'S28'), ('count', int)]
        armonias_tamanos=[len(armonias_unicas[i]) for i in range(0,len(armonias_unicas))]
        if len(armonias_tamanos)>0:
                max_tamano=np.max(armonias_tamanos)
                min_tamano=np.min(armonias_tamanos)
   
        for t,tamano in enumerate(np.arange(mayor_tamano_armonia,menor_tamano_armonia-1,-1)):
                armonias_T=[] #lista con pares
                for a, armonia in enumerate(armonias_hist):
                        if len(armonias_hist[a][0])==tamano:
                                armonias_T.append((str(armonias_hist[a][0]),int(armonias_hist[a][1])))       
                armonias_T=np.array(armonias_T,dtype=dtype)
                armonias_T=np.sort(armonias_T,order='count') #lo ordeno segun la propiedad count
                armonias_T=list(armonias_T)
                for j,d in enumerate(armonias_T):
                        contador=contador+1
                        contador_tick=contador_tick+1
                        armonia=str(armonias_T[j][0],'utf-8')
                        count_value=int(armonias_T[j][1])
                        plt.barh(contador,count_value,color=color_dict[str(tamano)],edgecolor='black')
                        yTick_position.append(contador_tick)
                        yTick_name.append(armonias_T[j][0])
        plt.yticks(yTick_position,yTick_name, rotation=0,fontsize=8)
        plt.title('2-3-4-5 Armonias',fontsize=20)
        plt.show()
                               
        #print(armonias,tiempos_armonias)
        if G.number_of_nodes() !=0:
                print('Armonias encontradas y su tiempo de aparicion')
                return(armonias,tiempos_armonias)
        else:
                return('No se encontraron armonias en esta voz')
#---------------------------------------------------------------------------
def f_armon(cancion, indexes):
        #Toma como input una canción y la lista indexes con los indices de las voces a analizar.
        #Encuentra todas las armonias que ocurrieron entre esos instrumentos seleccionados.
        #Obtiene un vector de armonias(2 o mas notas simultaneas) y el momento en el cual ocurrieron.

        #¿Que grafica?
        #1)Construye grafo el cual es no dirigido.(ver graficar_armonias_undirected)
        #2)Construye grafo dirigido donde los nodos son acordes.(ver graficar_armonias_directed)
        #3)Realiza histograma de las armonias
        
        
        #Notas: -si dos acordes estan ligados,los cuenta dos veces y no una vez sola.
        #       -pueden aparecer autoloops cuando dos voces toquen la misma nota de forma simultanea, pero esas armonias no las tenemos en cuenta.
        #       -si no encuentra armonias devuelve un string :'No se encontraron armonias entre estas voces'
        #       -elgrafico en igraph grafica armonias(cliques) de 2,3,4 y 5 los cuales los elegimos para que tengan notas diferentes.

       
        #Cancion
        song = msc.converter.parse(cancion) # Lee la cancion, queda un elemento stream.Score
        
        #Instrumentos
        instrumentos=[song.parts[indexes[i]].partName for i in range(0,len(indexes))]
        #print('Instrumentos Seleccionados:'+str(instrumentos))
        partituras=[song.parts[indexes[i]] for i in range(0,len(indexes))]
        compases=[partitura.getElementsByClass(msc.stream.Measure) for p,partitura in enumerate(partituras)]#todos los compases de las voces seleccionadas
        
        #Armonias
        Armonias_song=[]
        Tiempos_armonias_song=[]
        menor_tamano_armonia=2 #indicamos cual es el tamano minimo de armonias a graficar
        mayor_tamano_armonia=5 #indicamos cual es el tamano maximo de armonias a graficar

        #Incializamos el grafo en nx y en ig: los nodos seran notas. 
        G=nx.MultiGraph()
        #I=ig.Graph()
        Inodos=[]
        numero_nodo=-1

        #-------------------------------------------------------------------------------------------------------------------------------------
        #Estrategia: Recorremos cada compas, y recorremos por instrumento asi es mas ordenado, buscamos armonias dentro de cada compas y luego
        #una vez que las encontramos las guardamos y seguimos al siguiente compas.
        #-------------------------------------------------------------------------------------------------------------------------------------
        
        for numero_compas in range(0,len(compases[0])):
                Notas_compas=[]   #lista de listas de notas por cada voz. Cada lista contiene las notas que esa voz fue tocando en en el compas actual
                Tiempos_compas=[] #lista de listas de tiempos por cada voz. Cada lista contiene los tiempos en que sonaron esas notas en el compas actual.
                Frecuencias_compas=[]
                Octavas_compas=[]
                Armonias=[]
                Tiempos_armonias=[]
                #-----------------------------------------------------------------------------------------------------------------------------------------------------
                #1) Obtenemos todas las notas y los tiempos de ocurrencia de cada una de ellas en el compas actual y las guardamos en Notas_compas y en Tiempos_compas:
                #-----------------------------------------------------------------------------------------------------------------------------------------------------
                for inst in range(0,len(instrumentos)):
                        compas=compases[inst][numero_compas]
                        notas=[]
                        tiempos=[]
                        frecuencias=[]
                        octavas=[]
                        for i,el in enumerate(compas.flat):
                                isChord=str(type(el))=='<class \'music21.chord.Chord\'>' #me fijo si es un elemento del tipo acorde chord.Chord
                                if isinstance(el,msc.note.Note):#si es una nota
                                        nota_name=str(el.nameWithOctave)
                                        notas.append(nota_name)
                                        Notas_compas.append(nota_name)
                                        
                                        tiempo_nota=float(compas.offset+el.offset)
                                        tiempos.append(tiempo_nota)
                                        Tiempos_compas.append(tiempo_nota)
                                        
                                        frecuencias.append(el.pitch.frequency)
                                        Frecuencias_compas.append(el.pitch.frequency)
                                        
                                        octavas.append(el.octave)
                                        Octavas_compas.append(el.octave)
                                        
                                if isinstance(el,msc.chord.Chord) & isChord==True:#si es un acorde
                                        for nc,noteChord in enumerate(el):
                                                nota_name=str(noteChord.nameWithOctave)
                                                notas.append(nota_name)
                                                Notas_compas.append(nota_name)
                                                
                                                tiempo_nota=float(compas.offset)+float(el.offset)
                                                tiempos.append(tiempo_nota)
                                                Tiempos_compas.append(tiempo_nota)
                                                
                                                frecuencias.append(noteChord.pitch.frequency)
                                                Frecuencias_compas.append(noteChord.pitch.frequency)
                                                
                                                octavas.append(noteChord.octave)
                                                Octavas_compas.append(noteChord.octave)
                                                        
                #Reordenamos las notas en armonia de menor tiempo a mayor tiempo:
                                                
                Tiempos_compas_ordenada=np.sort(Tiempos_compas)
                Notas_compas_ordenada=[]
                Frecuencias_compas_ordenada=[]
                Octavas_compas_ordenada=[]
                
                Tiempos_compas_ordenada_unicos=list(np.unique(Tiempos_compas_ordenada))
                for z,tiempo_ordenado in enumerate(Tiempos_compas_ordenada_unicos):
                        #indice=Tiempos_compas.index(tiempo_ordenado)
                        indices=[i for i, e in enumerate(Tiempos_compas) if e == tiempo_ordenado]
                        for i,indice in enumerate(indices):
                                Notas_compas_ordenada.append(Notas_compas[indice])
                                Frecuencias_compas_ordenada.append(Frecuencias_compas[indice])
                                Octavas_compas_ordenada.append(Octavas_compas[indice])
                                
                #Reescribo las variables,con el ordenamiento segun el vector de tiempos.
                Tiempos_compas=Tiempos_compas_ordenada
                Notas_compas=Notas_compas_ordenada
                Frecuencias_compas=Frecuencias_compas_ordenada
                Octavas_compas=Octavas_compas_ordenada

                #-----------------------------------------------------------------------------------------------------------------------------      
                #2) Ahora buscamos las armonias que existan en el compas actual:
                #Antes de cambiar de compas buscamos armonias en Notas_compas y Tiempos_compas antes de vaciarlos y las guardamos en Armonias.
                #-----------------------------------------------------------------------------------------------------------------------------
        
                #Recorremos el vector de tiempos y nos fijamos todas las notas que caen en un mismo tiempo y las guardamos en la lista armonia y esta la guardamos en la lista armonias:

                if len(Notas_compas)!=0: #para que saltee aquellos compases donde todas las voces estan en silencio.
                        armonia=[Notas_compas[0]] #al principio armonia tiene la primer nota del compas
                        tiempo_actual=Tiempos_compas[0]
                        tonos=[Frecuencias_compas[0]]#contendra las frecuencias de las notas en armonia, para ordenarlas de la mas grave a la mas aguda antes de guardar armonia
                        octava=[Octavas_compas[0]]
                        
                        for t in range(1,len(Tiempos_compas)):
                                tiempo_new=Tiempos_compas[t]
                                if(tiempo_new-tiempo_actual)==0:#significa que seguimos en el mismo tiempo actual.
                                        armonia.append(Notas_compas[t])
                                        tonos.append(Frecuencias_compas[t])
                                        octava.append(Octavas_compas[t])
                                        tiempo_actual=Tiempos_compas[t]#actualizamos tiempos_old.
                                else: #significa que cambiamos de tiempo.
                                        tiempo_actual=Tiempos_compas[t]#actualizamos tiempos_old.
                                        #reordenamos las notas en armonia de menor tono a mayor tono:
                                        tonos_ordenados=np.sort(tonos)
                                        octava_ordenada=[]
                                        armonia_ordenada=[]
                                        for f,tono in enumerate(tonos_ordenados):
                                                indice=tonos.index(tono)
                                                armonia_ordenada.append(armonia[indice])
                                                octava_ordenada.append(octava[indice])
                                        if (len(armonia)>=2 and len(set(armonia))==len(armonia)):#consideramos armonia si son 2 o mas sonidos que suenan simultaneos y sin repetir.
                                                Tiempos_armonias.append(Tiempos_compas[t-1])#es el tiempo en el que ocurrio esa armonia
                                                Tiempos_armonias_song.append(Tiempos_compas[t-1])
                                                Armonias.append(armonia_ordenada)#guardamos la armonia ya ordenada solo si la armonia tiene 2 o mas notas.
                                                Armonias_song.append(armonia_ordenada)
                                                #Agrego nodos al grafo:
                                                for n,nota in enumerate(armonia_ordenada):
                                                        if ((nota in Inodos)==False and len(armonia)>menor_tamano_armonia-1):#solo agrega el nodo si este no aparecio nunca y nodos en armnonias de mas de 2 notas.
                                                                numero_nodo=numero_nodo+1
                                                                #Grafo nx
                                                                G.add_node(nota)
                                                                G.node[nota]['freq']= tonos_ordenados[n]
                                                                G.node[nota]['octava'] = octava_ordenada[n]
                                                                G.node[nota]['duracion']=4.0
                                                                #Grafo ig
                                                                #I.add_vertex(numero_nodo)
                                                                #I.vs[numero_nodo]["name"]=nota
                                                                Inodos.append(nota)
                                                                #I.vs[numero_nodo]["freq"]= tonos_ordenados[n]
                                                                #I.vs[numero_nodo]["octava"]= octava_ordenada[n]
                                                                #I.vs[numero_nodo]["duracion"]=4.0
                                                                 
                                        armonia=[Notas_compas[t]]
                                        tonos=[Frecuencias_compas[t]]
                                        octava=[Octavas_compas[t]]
     
        
        #Agregamos los enlaces al grafo G e I:
        numero_enlace=-1
        color_dict= {"2":"purple","3": "blue", "4":"red","5":"green"}
        colores_edges=[]
        pesos_edges=[]
        Armonias_song_unicas=[list(i) for i in set(tuple(i) for i in Armonias_song)]
        Armonias_hist = [(x, Armonias_song.count(x)) for x in Armonias_song_unicas]
        Armonias_tamanos=[len(Armonias_song_unicas[i]) for i in range(0,len(Armonias_song_unicas))]
        if len(Armonias_tamanos)>0:
                max_tamano=np.max(Armonias_tamanos)
                min_tamano=np.min(Armonias_tamanos)
        
        for a,armonia in enumerate(Armonias_hist):
                tamano_armonia=len(armonia[0])
                if (tamano_armonia>menor_tamano_armonia-1 and tamano_armonia<mayor_tamano_armonia+1): #solo agregamos armonias de 2-3-4-5
                        for n in range (0,len(armonia[0])):
                                for m in range(n+1,len(armonia[0])):
                                        numero_enlace=numero_enlace+1
                                        edge_name=(armonia[0][n],armonia[0][m])
                                        #Grafo nx
                                        G.add_edge(armonia[0][n],armonia[0][m],key=str(numero_enlace),weight=armonia[1])
                                        G.add_edge(armonia[0][n],armonia[0][m],key=str(numero_enlace),tamano=tamano_armonia)
                                        G.add_edge(armonia[0][n],armonia[0][m],key=str(numero_enlace),color=color_dict[str(tamano_armonia)])
                                        #Grafo ig
                                        #I.add_edge(armonia[0][n],armonia[0][m])
                                        #I.es[numero_enlace]['tamano']=tamano_armonia
                                        #I.es[numero_enlace]['color']=color_dict[str(tamano_armonia)]
                                        #I.es[numero_enlace]['weight']=armonia[1]
                                        pesos_edges.append(armonia[1])
                                        colores_edges.append(color_dict[str(tamano_armonia)])

        #1)Grafico con Igraph (no dirigido):
        #f_graficar_armonias_undirected(G, color_map='rainbow',layout='espiral');
        #f_graficar_armonias_undirected_igraph(I, color_map='rainbow',layout='espiral')

        #2)Grafico con Igraph (dirigido):
        D=f_grafo_armonias_directed(Armonias_song);
        #f_graficar_armonias_directed_igraph(Armonias_song)

        #3)Histograma de armonias:
        #plt.figure
        #yTick_position=[]
        #yTick_name=[]
        #contador=-1
        #contador_tick=-0.5
        #dtype = [('name', 'S28'), ('count', int)]
   
        #for t,tamano in enumerate(np.arange(mayor_tamano_armonia,menor_tamano_armonia-1,-1)):
        #        armonias_T=[] #lista con pares
        #        for a, armonia in enumerate(Armonias_hist):
        #                if len(Armonias_hist[a][0])==tamano:
        #                        armonias_T.append((str(Armonias_hist[a][0]),int(Armonias_hist[a][1])))        
        #        armonias_T=np.array(armonias_T,dtype=dtype)
        #        armonias_T=np.sort(armonias_T,order='count') #lo ordeno segun la propiedad count
        #        armonias_T=list(armonias_T)
        #        for j,d in enumerate(armonias_T):
        #                contador=contador+1
        #                contador_tick=contador_tick+1
        #                armonia=str(armonias_T[j][0],'utf-8')
        #                count_value=int(armonias_T[j][1])
        #                plt.barh(contador,count_value,color=color_dict[str(tamano)],edgecolor='black')
        #                yTick_position.append(contador_tick)
        #                yTick_name.append(armonias_T[j][0])
        #plt.yticks(yTick_position,yTick_name, rotation=0,fontsize=8)
        #plt.title('2-3-4-5 Armonias',fontsize=20)
        #plt.show()

        #print(Armonias_song,Tiempos_armonias_song)
        if G.number_of_nodes() !=0:
                #print('Armonias encontradas y su tiempo de aparicion')
                return(Armonias_song,Tiempos_armonias_song,D,G)
        else:
                #print('No se encontraron armonias entre estas voces')
                return(Armonias_song,Tiempos_armonias_song,D,G)
##---------------------------------------------------------------------------
def f_graficar_armonias_undirected(G, color_map='rainbow',layout='espiral',labels='false'):
        #Grafica el grafo no dirigido G. Graficamos en colores si son enlaces por armonias de 2-3-4-5 o mas notas.
        #Los enlaces estan pesados por la aparicion de esa armonia.El grosor del enlace es por su peso.
        #El tamano de los nodos se calculo segun el strength (es un grado pesado por el peso de los enlaces)
        M = G.number_of_edges()
        N = G.number_of_nodes()
        nodos = G.nodes()
        freq_min = min(np.array(list(nx.get_node_attributes(G,'freq').values())))
        pos = dict()

        if len(nodos)>0:
            for nodo in nodos:
                    f = G.node[nodo]['freq']
                    d = G.node[nodo]['duracion']
                    theta = 2*np.pi * np.log2(f/freq_min)
                    if layout=='espiral':
                            x = np.cos(theta)*f/freq_min*(1+d/4)
                            y = np.sin(theta)*f/freq_min*(1+d/4)
                            pos[nodo] = np.array([x,y])
                    elif layout=='circular':
                            nro_oct = G.node[nodo]['octava']
                            x = np.cos(theta)*nro_oct*(1+d/12)
                            y = np.sin(theta)*nro_oct*(1+d/12)
                            pos[nodo] = np.array([x,y])

            octavas=np.array([G.node[nodo]['octava'] for i,nodo in enumerate(nodos)]) #agregue esta linea en reemplazo de la anterior.
            oct_min = min(octavas)
            oct_max = max(octavas)
            colores_oct_nro = (octavas - oct_min)/(oct_max - oct_min)
            m = cm.ScalarMappable(norm=None, cmap=color_map)
            colores_oct = m.to_rgba(colores_oct_nro)

            #Grafico
            #fig=plt.figure(figsize=(16,16))
            grados = dict(nx.degree(G))
            nx.draw_networkx_nodes(G,pos,node_list=nodos,node_color=colores_oct,node_size=[50*v for v in grados.values()],alpha=1)
            if labels==True:
                nx.draw_networkx_labels(G,pos)

            #Enlaces
            #edges = nx.draw_networkx_edges(G,pos,width=3)
            edges=G.edges()
            weights = list(nx.get_edge_attributes(G,'weight').values())
            weight_max = max(weights)
            alphas = [(weights[i]/weight_max)**(1./2.) for i in range(0,M)]
            color_edges=list(nx.get_edge_attributes(G,'color').values())
            width_edges=list(nx.get_edge_attributes(G,'weight').values())

            #for i in range(M): #comente esta porque no me funcaba el edges[i]
                    #edges[i].set_alpha((weights[i]/weight_max)**(1./2.)) # valores de alpha para cada enlace

            for e,edge in enumerate(edges):#reemplace con esto que me parece que hace lo mismo
                    edges=nx.draw_networkx_edges(G, pos, edgelist=[edge], edge_color=color_edges[e],width=width_edges[e],alpha=alphas[e]) 
                        
            plt.axis('off')
            #plt.show()
        
        return()
##---------------------------------------------------------------------------
def f_grafo_armonias_directed(Armonias):
        #Recibe un vector de Armonias y realiza un grafo dirigido donde
        #Los nodos en vez de ser las notas son las armonias y se enlazan cuando ocurre una
        #despues de la otra.

        menor_tamano_armonia=2
        mayor_tamano_armonia=5
        color_dict= {"2":"purple","3": "blue", "4":"red","5":"green"}
        
        G=nx.MultiDiGraph()
        numero_nodo=-1
        Gnodos=[]
        color_nodos=[]
        
        if len(Armonias)>0: #si no encontro armonias no hace nada  
                #Creo nodos:
                for a in range(0,len(Armonias)):
                        tamano_armonia=len(Armonias[a])
                        if (tamano_armonia>menor_tamano_armonia-1 and tamano_armonia<mayor_tamano_armonia+1):
                                if (str(Armonias[a]) in Gnodos)==False:
                                        numero_nodo=numero_nodo+1
                                        G.add_node(str(Armonias[a]))
                                        Gnodos.append(str(Armonias[a]))
                                        color_nodos.append(color_dict[str(tamano_armonia)])
        
                #Creo enlaces si encontro armonias del tamano buscado
                if len(Gnodos)>0:
                        G.add_edge(str(Gnodos[0]),str(Gnodos[1]))
                        for a in range(1,len(Gnodos)-1):
                                G.add_edge(str(Gnodos[a]),str(Gnodos[a+1]))
                        
                        #fig=plt.figure(figsize=(16,16))
                        #pos=nx.random_layout(G)
                        #nx.draw_networkx_nodes(G,pos,node_list=Gnodos,node_color=color_nodos,node_size=2000,alpha=1)
                        #nx.draw_networkx_labels(G,pos,font_size=5,font_color='k')
                        #edges=G.edges()
                        #nx.draw_networkx_edges(G,pos,edge_list=edges,edge_color='black',width=1,alpha=1,arrowsize=50)
                        #plt.axis('off')
                        #plt.show()
                
                else:
                        print('No se encontraron armonias del tamano buscado entre estas voces')
        return(G)
#-----------------------------------------------------------------------------
def f_dist_escalas(cancion, nombre_parte=0):
    # Toma como input una canción y devuelve un grafo o una lista de grafos si se repite el nombre
    # cancion puede ser la ubicacion del archivo (str) o el Score de music21
    # Opcional: nombre_parte puede ser el nombre (str) o su indice
    
    # Cancion
    if type(cancion)==msc.stream.Score:
        song = cancion # Si la cancion ya es un stream.Score, se queda con eso
    else:
        song = msc.converter.parse(cancion) # Sino, lee la partitura, queda un elemento stream.Score
    key = song.analyze('key')
    tonica = msc.note.Note(key.tonic)

    # Lista de nombres de las partes
    Lp = len(song.parts) # Cantidad de partes (voces)
    lista_partes = list(np.zeros(Lp)) # Crea una lista donde se van a guardar los nombres de las partes
    for i,elem in enumerate(song.parts):
        lista_partes[i] = elem.partName # Guarda los nombres de las partes en la lista

    # Seleccion de la parte a usar
    # Si el input es el indice (int) intenta encontrarlo entre las partes; si no lo encuentra, selecciona la primera voz
    if type(nombre_parte)==int:
        try:
            part = song.parts[nombre_parte]
            #print('Partes: '+str(lista_partes)+'. Parte seleccionada: '+str(lista_partes[nombre_parte]))
        except IndexError:
            part = song.parts[0]
            #print(nombre_parte+' no es un índice aceptable. Partes: '+str(lista_partes)+'. Parte seleccionada: '+str(lista_partes[0]))
    # Si el input es nombre (str) y no está entre las partes, selecciona la primera voz
    elif not nombre_parte in lista_partes: 
        part = song.parts[0]
        #print(nombre_parte+' no está entre las partes: '+str(lista_partes)+'. Parte seleccionada: '+str(lista_partes[0]))
    else:
        indexes = [index for index, name in enumerate(lista_partes) if name == nombre_parte]
        if len(indexes)==1:
            part = song.parts[indexes[0]]
        else:
            part = []
            for j in indexes:
                part.append(song.parts[j])
        #print('Partes: '+str(lista_partes)+'. Parte(s) seleccionada(s): '+str([lista_partes[i] for i in indexes]))
    # En cualquier caso, devuelve una lista de partes y cuál selecciona (y aclaraciones neccesarias de cada caso)

    # Crea la(s) voz(ces) analizada(s) (todos los compases) y se queda con
    # todas las notas sin silencios con offset 'absoluto' (flat)
    # IMPORTANTE: Si la voz contiene sonidos simultáneos, sólo se queda con el más agudo
    if type(part) == list:
        voz = []
        for parte in part:
            voz.append(parte.getElementsByClass(msc.stream.Measure))
        lista_dist = []
        lista_dur = []
        for voice in voz:
            notes = [x for x in voice.flat if type(x)==msc.note.Note]
            tiempos = [x.offset for x in voice.flat if type(x)==msc.note.Note]
            indices = []

            for i in range(len(notes)-1):
                if tiempos[i+1] == tiempos[i]:
                    if (notes[i+1].pitch.frequency > notes[i].pitch.frequency):
                        indices.append(i)
                    else:
                        indices.append(i+1)

            indices = [x for x in indices[::-1]]

            for index in indices:
                del notes[index]
            intervalos = [msc.interval.Interval(tonica, nota) for nota in notes]
            distancias = [i.semitones for i in intervalos]
            duraciones = [nota.quarterLength for nota in notes]
            lista_dist.append(distancias)
            lista_dur.append(duraciones)
    else:
        voz = part.getElementsByClass(msc.stream.Measure)
        notas = [x for x in voz.flat if type(x)==msc.note.Note]
        tiempos = [x.offset for x in voz.flat if type(x)==msc.note.Note]
        indices = []

        for i in range(len(notas)-1):
            if tiempos[i+1] == tiempos[i]:
                if (notas[i+1].pitch.frequency > notas[i].pitch.frequency):
                    indices.append(i)
                else:
                    indices.append(i+1)

        indices = [x for x in indices[::-1]]

        for index in indices:
            del notas[index]
        intervalos = [msc.interval.Interval(tonica, nota) for nota in notas]
        distancias = [i.semitones for i in intervalos]
        duraciones = [nota.quarterLength for nota in notas]
    # Crea una lista de distancias en st (distancias) o una lista de listas (por cada voz) (lista_dist)

    # Creamos el grafo dirigido G o lista de grafos dirigidos Gs si hay mas de una voz
    if type(part) == list:
        Gs = [] # Va a ser una lista de grafos, uno por cada voz analizada
        for j,dist in enumerate(lista_dist):
            if len(dist)==0:
                continue
            G = nx.DiGraph()
            oct_min = int(min((np.floor(np.array(dist)/12))))
            # Nodos
            # Recorremos todas las notas de la voz
            for i,d in enumerate(dist):
                nota_name = str(d)+'/'+str(lista_dur[j][i])
                if not G.has_node(nota_name): # Si el grafo no tiene el nodo, lo agregamos con los atributos que se quieran
                    G.add_node(nota_name)
                    G.node[nota_name]['freq'] = 440*2**(d/12) # Pongo como frecuencia falsa de tonica: 440
                    G.node[nota_name]['octava'] = int(np.floor(d/12)) - oct_min+1
                    G.node[nota_name]['duracion'] = lista_dur[j][i]
                dist[i] = nota_name

            # Enlaces pesados
            L = len(dist)
            for i in range(L-1):  # recorremos desde la primera hasta la anteultima nota, uniendo sucesivas
                if G.has_edge(dist[i],dist[i+1]):
                    G[dist[i]][dist[i+1]]['weight']+=1 # si el enlace ya existe, se le agrega +1 al peso
                else:
                    G.add_edge(dist[i],dist[i+1],weight=1) # si el enlace no existe, se crea con peso 1
            Gs.append(G)
    elif len(distancias)>0:
        G = nx.DiGraph()

        oct_min = int(min((np.floor(np.array(distancias)/12))))

        # Nodos
        # Recorremos todas las notas de la voz
        for i,d in enumerate(distancias):
            nota_name = str(d)+'/'+str(duraciones[i])
            if not G.has_node(nota_name): # Si el grafo no tiene el nodo, lo agregamos con los atributos que se quieran
                G.add_node(nota_name)
                G.node[nota_name]['freq'] = 440*2**(d/12)
                G.node[nota_name]['octava'] = int(np.floor(d/12)) - oct_min+1
                G.node[nota_name]['duracion'] = duraciones[i]
            distancias[i] = nota_name
        
        # Enlaces pesados
        L = len(distancias)
        for i in range(L-1):  # recorremos desde la primera hasta la anteultima nota, uniendo sucesivas
            if G.has_edge(distancias[i],distancias[i+1]):
                G[distancias[i]][distancias[i+1]]['weight']+=1 # si el enlace ya existe, se le agrega +1 al peso
            else:
                G.add_edge(distancias[i],distancias[i+1],weight=1) # si el enlace no existe, se crea con peso 1
        Gs = G
    else:
        return None
    return(Gs)

#---------------------------------------------------------------------------
def f_full_graph(path,voz_principal=False): #hay que pasarle una direccion para un archivo y arma un grafo de ese xml con todas las voces

    #Creo los grafos que van a tener todas las voces de un artista y todos sus temas
    ls_m=[]
    ls_r=[]
    ls_d=[]
    
    song = msc.converter.parse(path)
    L=len(song.parts) #recorro todas las voces
    voces=song.parts #quiero que grafique todas las voces
    indexes=list(range(len(voces)))
    #para las armonias si le paso la lista con todas las voces me devuelve ya todo combinado
    armon,tiempos,D,U=f_armon (path, indexes); #analizamos las armonias
    
    if voz_principal==False:
        for i in range(L):
            #Uno los grafos en uno para cada voz para melodia
            m=f_xml2graph(path, nombre_parte=i, modelo='melodia');
            dist = f_dist_escalas(path, nombre_parte=i) 
            #Uno los grafos en uno para cada voz para ritmo
            r=f_xml2graph(path, nombre_parte=i, modelo='ritmo');
            if str(r.__class__) != "<class 'NoneType'>":
                ls_r.append(r)
            if str(dist.__class__) != "<class 'NoneType'>":
                ls_d.append(dist)
            if str(m.__class__) != "<class 'NoneType'>": 
                ls_m.append(m)
        M=nx.compose_all(ls_m)
        A=nx.compose_all(ls_d) # A de Absoluto, porque es el de las distancias a la tonica
        R=nx.compose_all(ls_r)
    elif voz_principal==True:
        M = f_xml2graph(path)
        A = f_dist_escalas(path) 
        R = f_xml2graph(path, modelo='ritmo')
    else:
        M = f_xml2graph(path,nombre_parte=voz_principal)
        A = f_dist_escalas(path,nombre_parte=voz_principal) 
        R = f_xml2graph(path,nombre_parte=voz_principal, modelo='ritmo')
        
    return(M,A,R,D,U)

#---------------------------------------------------------------------------
def f_hierarchy(G): #grafica Ck vs K en log, y a partir de eso uno ve si es una red jerárquica
    
    H=G.copy()
    #H = H.to_undirected()
    nodos=H.nodes() 
    grados=[]
    Cs=[]
    Cs_prom=[]
    Cs_error=[]
    cs_err=[]

    
    for i,nodo in enumerate(nodos):
        grados.append(H.degree(nodo))
        Cs.append(nx.clustering(H, nodo)) #sin pesos

    #Busco los Cs que tengan el mismo grado y los promedio    
    for i,k in enumerate(grados):
        ls=[]
        for j, c in enumerate(Cs):
            if k == grados[j]:
                ls.append(Cs[j])
        Cs_prom.append(np.mean(ls))
        Cs_error.append(np.std(ls))
        
        cs_err.append((1/np.log(10))*(1/Cs_prom[i])*Cs_error[i])
    
    
    plt.figure(figsize=(9,9))
    plt.plot(grados,Cs_prom,'bo')
    #plt.errorbar(grados, Cs_prom, xerr=None, yerr=cs_err, fmt=' ',ecolor='green',elinewidth=5,capsize=5,markersize=3)
    plt.xscale('log')
    plt.yscale('log')
    
    plt.xlabel('$k$')
    plt.ylabel('$C(k)$')
    plt.title('Bin lineal - Escala log')
    plt.show()
#------------------------------------------------------------------------------
def f_transitivity_motifs(G):
        #Toma un grafico dirigido y busca todos los 3-cliques.
        #Luego a cada clique le calcula su estado, que es un dibujo de como estan conectados:
        #Ejemplo: -Si el clique es: A<-->B-->--C-->A el estado de ese clique seria: ['','-->','<--','-->','','-->']
        #         -A cada clique le asignamos un id_clique que son 3 numeros de menor a mayor de acuerdo a la suma de grados in y out en cada nodo
        #         -El clique ['','-->','<--','-->','','-->'] tendrá id_clique=[-1,0,1]
        #Devuelve lista de 3-cliques y el estado de cada uno.
        #Realiza histograma agrupando 3-cliques por id_clique.
        
        J=G.to_undirected()
        enlaces=G.edges()
        cliques=list(nx.enumerate_all_cliques(J))
        k=3 #Tipo de k-cliques que queremos percolar.
        cliques_k=[] #nos quedamos con los k_cliques.
        for c,clique in enumerate(cliques):
            if len(clique)==k:
                cliques_k.append(clique)

        estados_clique=[]
        ids_clique=[]
        for c in range(0,len(cliques_k)):
                clique_actual=cliques_k[c]
                #nodos
                A=clique_actual[0]
                B=clique_actual[1]
                C=clique_actual[2]
        
                #Vector de estado clique:
                enlaces_clique=[(A,C),(A,B),(B,A),(B,C),(C,B),(C,A)]
                estado_clique=['','','','','','']
                id_clique=np.zeros(k,dtype=int)#tres numeros de menor a mayor, con la suma de los grados de cada nodo, que puede ser:-2(salen 2) ,0(llega uno sale uno) o 2(llegan dos)
                
                for n in range(0,len(clique_actual)):
                        for m in range(0,len(clique_actual)):
                                if (n!=m) and (clique_actual[n],clique_actual[m]) in enlaces:
                                        id_clique[n]=id_clique[n]-1
                                        id_clique[m]=id_clique[m]+1
                                        index=enlaces_clique.index((clique_actual[n],clique_actual[m]))
                                        if index %2 ==0:
                                                estado_clique[index]='<-'
                                                
                                        else:
                                                estado_clique[index]='->'
                id_clique=np.sort(id_clique)
                ids_clique.append(list(id_clique))                
                estados_clique.append(estado_clique)
        #Histograma
        transitivity_motifs=[list(i) for i in set(tuple(i) for i in ids_clique)]
        hist_transitivity_motifs = [(x, ids_clique.count(x)) for x in transitivity_motifs]
        plt.figure
        yTick_position=[]
        yTick_name=[]
        contador=-1
        contador_tick=-0.5
   
        for m, motif in enumerate(transitivity_motifs):
                contador=contador+1
                contador_tick=contador_tick+1
                count_value=int(hist_transitivity_motifs[m][1])
                plt.barh(contador,count_value,color='blue',edgecolor='black')
                yTick_position.append(contador_tick)
                yTick_name.append(str(motif))
        plt.yticks(yTick_position,yTick_name, rotation=0,fontsize=8)
        plt.title('Transitivity '+str(k)+'-motifs',fontsize=20)
        plt.show()

        return(cliques_k,estados_clique)
#---------------------------------------------------------------
def f_rewiring_directed(G):
    #Funcion de grafo G que toma un grafo dirigido y realiza un recableado
    #manteniendo el grado-in y el grado-out de cada nodo.
    #Estrategia que vimos en clase de Cherno de redes random:
    #1) Realizamos una copia del grafo G sin enlaces.
    #2) Vamos tomando pares de nodos al azar y creamos enlaces manteniendo el grado-in y el grado-out de cada nodo hasta agotar el cupo.
    #3) Buscamos los enlaces multiples que seran los problematicos(ej: que hayan dos enlaces del nodo A hacia el B) y recableamos hasta
    #   que el numero de enlaces multiples sea 0.
    #Devuelve grafo D del tipo MultiDiGraph pero que entre dos nodos solo pueden existir dos enlaces uno de A a B y otro de B hacia A, no
    #pueden aparecer multienlaces.
    
    nodos=list(G.nodes)
    enlaces=list(G.edges)
    grados_dict = dict(G.degree(nbunch=nodos))
    grados_in_dict=dict(G.in_degree(nbunch=nodos))
    grados_out_dict=dict(G.out_degree(nbunch=nodos))
    #k_nodo= list(grados_dict.values()) # lista de grados de cada nodo en nodos(ordenados)
    k_nodo=[G.degree(nodo) for n,nodo in enumerate(nodos)]
    #kin_nodo= list(grados_in_dict.values())
    kin_nodo=[G.in_degree(nodo) for n,nodo in enumerate(nodos)]
    #kout_nodo= list(grados_out_dict.values())
    kout_nodo=[G.out_degree(nodo) for n,nodo in enumerate(nodos)]
    
    #print('grado original G')
    k_nodo_antes=k_nodo
    kin_nodo_antes=kin_nodo
    kout_nodo_antes=kout_nodo
    #print('k_nodo_antes')
    #print(k_nodo_antes)
    #print('kin_nodo_antes')
    #print(kin_nodo_antes)
    #print('kout_nodo_antes')
    #print(kout_nodo_antes)
    
    #0)Cuento enlaces en el grafo original:
    #Multiples y Simples:
    enlaces_autoloops=[]
    enlaces_multiples=[]
    enlaces_simples=[]
    
    for i in range(0,len(nodos)):
        for j in range(0,len(nodos)):
                if(G.number_of_edges(nodos[i],nodos[j]))>1:
                        for k in range(0,G.number_of_edges(nodos[i],nodos[j])-1):#en este for agregamos al vector enlaces_multiples tantos enlaces como multiplicidd tenga el mismo menos 1(porque si hay 3 enlaces entre dos nodos solo hay que sacar 2 de ellos )                    
                                enlaces_multiples.append([nodos[i],nodos[j]]) #agrego multiplicidad -1 de enlaces muliples
                        enlaces_simples.append([nodos[i],nodos[j]]) #agrego uno simple
                elif (G.number_of_edges(nodos[i],nodos[j]))==1:
                        if i!=j:
                                enlaces_simples.append([nodos[i],nodos[j]])
                        elif i==j:
                                enlaces_autoloops.append([nodos[i],nodos[j]])
                                
    #print('enlaces totales antes: {}'.format(len(enlaces)))
    #print('enlaces autoloops antes: {}'.format(len(enlaces_autoloops)))                                          
    #print('enlaces multiples antes: {}'.format(len(enlaces_multiples)))
    #print('enlaces simples antes: {}'.format(len(enlaces_simples)))
    
    #1) Creo un multigraph D que acepte multiedges
    D = nx.MultiDiGraph()

    #Agrego nodos:
    D.add_nodes_from(nodos)
    
    #Inicializo kin_control, kout_control, nodosout_new y nodosin_new:
    kin_control=np.array(kin_nodo_antes) #cuando creo un enlace entre nodoout_i y nodoin_j se le restara un 1 a los lugares i y j de kout_control y kin_control respectiv.
    kout_control=np.array(kout_nodo_antes)
    nodosout_new=nodos
    nodosin_new=nodos

    #2)Agregamos enlaces de forma aleatoria al grafo D, manteniendo controlado que el cupo de cada nodo no puede exceder su grado.
    while(len(nodosin_new)>0 and len(nodosout_new)>0): #sigo mientras haya elementos en ambas listas.
        #Elijo uno random de pairs
        pair= (np.random.choice(nodosout_new),np.random.choice(nodosin_new)) #se pueden crear autoloops si salen dos mismos numeros.  
        #Actualizamos variable de control: k_control
        kout_control[nodos.index(pair[0])]=kout_control[nodos.index(pair[0])]-1 #actualizamos el kout
        kin_control[nodos.index(pair[1])]=kin_control[nodos.index(pair[1])]-1   #actualizamos el kin
        #creamos el enlace
        D.add_edge(pair[0], pair[1])
        #Actualizamos variable de control: nodos_new
        if kout_control[nodos.index(pair[0])]==0: #solo actualizo kout_control cuando alguno de los valores llega a cero
                index_nonzerok=[i for i in range(0,len(kout_control)) if kout_control[i]>0] #buscamos los lugares de k_control donde hayan elementos dinstintos a cero
                index_equalzero=[i for i in range(0,len(kout_control)) if kout_control[i]==0]#buscamos los lugares de k_control donde hayan elementos igual a cero
                nodosout_new=[nodos[index_nonzerok[i]] for i in range(0,len(index_nonzerok))] #actualizamos la lista de nodos asi no volvemos a tomar nodos que ya recableamos por completo o sea aquellos que alcanzaron k_control[i]=0   
        if kin_control[nodos.index(pair[1])]==0:
                index_nonzerok=[i for i in range(0,len(kin_control)) if kin_control[i]>0] #buscamos los lugares de k_control donde hayan elementos dinstintos a cero
                index_equalzero=[i for i in range(0,len(kin_control)) if kin_control[i]==0]#buscamos los lugares de k_control donde hayan elementos igual a cero
                nodosin_new=[nodos[index_nonzerok[i]] for i in range(0,len(index_nonzerok))]#actualizamos la lista de nodos asi no volvemos a tomar nodos que ya recableamos por completo o sea aquellos que alcanzaron k_control[i]=0
    #print('grafico D')
    enlaces=list(D.edges())
    #Enlaces problemáticos:
    #Selfloops:
    #print('autoloops intermedio: {}'.format(len(list(D.selfloop_edges()))))
    autoloops=list(D.selfloop_edges())
    #print(autoloops)
    
    #Multiples y Simples:
    enlaces_simples_autoloops=[]
    enlaces_multiples=[]
    enlaces_simples=[]
    for i in range(0,len(nodos)):
            for j in range(0,len(nodos)):
                    if(D.number_of_edges(nodos[i],nodos[j]))>1:
                            for k in range(0,D.number_of_edges(nodos[i],nodos[j])-1):#en este for agregamos al vector enlaces_multiples tantos enlaces como multiplicidd tenga el mismo menos 1(porque si hay 3 enlaces entre dos nodos solo hay que sacar 2 de ellos )
                                    enlaces_multiples.append([nodos[i],nodos[j]])
                            if i!=j:
                                    enlaces_simples.append([nodos[i],nodos[j]])
                            elif i==j:
                                    enlaces_simples_autoloops.append([nodos[i],nodos[j]])
                    elif (D.number_of_edges(nodos[i],nodos[j]))==1:
                            if i!=j:
                                    enlaces_simples.append([nodos[i],nodos[j]])
                            elif i==j:
                                    enlaces_simples_autoloops.append([nodos[i],nodos[j]])
                                
    #print('enlaces totales intermedio: {}'.format(len(enlaces)))
    #print('enlaces autoloops intermedio: {}'.format(len(enlaces_simples_autoloops)))                                          
    #print('enlaces multiples intermedio: {}'.format(len(enlaces_multiples)))
    #print('enlaces simples intermedio: {}'.format(len(enlaces_simples)))
    
    
    #Comparamos grados en esta etapa intermedia si queremos:
    grados_dict = dict(D.degree(nbunch=nodos))
    grados_in_dict=dict(D.in_degree(nbunch=nodos))
    grados_out_dict=dict(D.out_degree(nbunch=nodos))
    #k_nodo_despues= list(grados_dict.values())# lista de grados de cada nodo en nodos(ordenados)
    k_nodo_intermedio=[D.degree(nodo) for n,nodo in enumerate(nodos)]
    #kin_nodo_despues= list(grados_in_dict.values())
    kin_nodo_intermedio=[D.in_degree(nodo) for n,nodo in enumerate(nodos)]
    #kout_nodo_despues= list(grados_out_dict.values())
    kout_nodo_intermedio=[D.out_degree(nodo) for n,nodo in enumerate(nodos)]
    #print(k_nodo_intermedio)
    #print(kin_nodo_intermedio)
    #print(kout_nodo_intermedio)
    
    #Hasta acá el programa lo que hizo fue reconectar las puntas conservando el constraint de los grado de los nodos.
    #El problema de esto es que aparecieron multienlaces entre nodos distintos o en el mismo nodo(o sea por ej dos enlaces (1-->2) o 2 enlaces(1-->1).
    #Estos enlaces los vamos a llamar enlaces problemáticos.

    #Por ultimo hay que eliminar estos enlaces que son problemáticos:
    #numero_autoloops=len(autoloops)
    numero_enlaces_multiples=len(enlaces_multiples)
    
    #4) Eliminamos los enlaces multiples:
    numero_enlaces_multiples=len(enlaces_multiples)
    #print('Recableando multiples...')
    while(numero_enlaces_multiples >0):
        for em in enlaces_multiples:
            idx = np.random.choice(len(enlaces_simples),1)[0] #elijo un enlace dentro de los simples o sea no problematicos
            enlace_elegido=enlaces_simples[idx]
            loscuatronodos=[em[0],em[1],enlace_elegido[0],enlace_elegido[1]]
            A = nx.to_pandas_adjacency(D)
            a1=A[em[0]][enlace_elegido[0]]
            a2=A[em[0]][enlace_elegido[1]]
            a3=A[em[1]][enlace_elegido[0]]
            a4=A[em[1]][enlace_elegido[1]]
            adjacencynumber=a1+a2+a3+a4
            #A continuación solo recableamos si los 4 nodos son distintos sino no, porque puedo vovler a crear un autoloop y  ademas...
            #solo recableamos si son enlaces no adyacentes sino no, esto evita que se vuelvan a formar mutienlaces.
            controlnumber=adjacencynumber + len(np.unique(loscuatronodos))
            #tiene ya esos enlaces simples el grafo D?
            yaestan=0
            if D.has_edge(em[0],enlace_elegido[1]):
                yaestan=yaestan+1
            elif D.has_edge(enlace_elegido[0],em[1]):
                yaestan=yaestan+1
                                
            if (controlnumber>=3 and yaestan==0):
                #Hago el swap:
                #Creo dos nuevos en el orden correcto un out lo tengo que conectar con un in 0:out 1:in
                D.add_edge(em[0],enlace_elegido[1])
                D.add_edge(enlace_elegido[0],em[1])
                #Elimino dos
                D.remove_edge(em[0],em[1])
                D.remove_edge(enlace_elegido[0],enlace_elegido[1])
                #Tengo que actualizar enlaces simples:
                enlaces_simples.remove([enlace_elegido[0],enlace_elegido[1]])
                enlaces_simples.append([em[0],enlace_elegido[1]])
                enlaces_simples.append([enlace_elegido[0],em[1]])
                #Tengo que actualizar enlaces_multiples
                enlaces_multiples.remove([em[0],em[1]])
                numero_enlaces_multiples=len(enlaces_multiples)

    #5)Nos fijamos los autoloops:
    autoloops=list(D.nodes_with_selfloops())
    #print('autoloops final: {}'.format(len(list(D.nodes_with_selfloops()))))
    
    #Por ultimo me fijo los multiples al final:(deberia ser cero)
    
    enlaces_simples_autoloops=[]
    enlaces_multiples=[]
    enlaces_simples=[]
    for i in range(0,len(nodos)):
            for j in range(0,len(nodos)):
                    if(D.number_of_edges(nodos[i],nodos[j]))>1:
                            for k in range(0,D.number_of_edges(nodos[i],nodos[j])-1):#en este for agregamos al vector enlaces_multiples tantos enlaces como multiplicidd tenga el mismo menos 1(porque si hay 3 enlaces entre dos nodos solo hay que sacar 2 de ellos )
                                    enlaces_multiples.append([nodos[i],nodos[j]])
                            if i!=j:
                                    enlaces_simples.append([nodos[i],nodos[j]])
                            elif i==j:
                                    enlaces_simples_autoloops.append([nodos[i],nodos[j]])
                    elif (D.number_of_edges(nodos[i],nodos[j]))==1:
                            if i!=j:
                                    enlaces_simples.append([nodos[i],nodos[j]])
                            elif i==j:
                                    enlaces_simples_autoloops.append([nodos[i],nodos[j]])
                                    
    #print('enlaces totales final: {}'.format(len(enlaces)))
    #print('enlaces autoloops final: {}'.format(len(enlaces_simples_autoloops)))                                          
    #print('enlaces multiples final: {}'.format(len(enlaces_multiples)))
    #print('enlaces simples final: {}'.format(len(enlaces_simples)))
 

    #6) Chequeo final para ver que se mantuvo el grado k de los nodos:
    grados_dict = dict(D.degree())
    k_nodo_despues=[D.degree(nodo) for n,nodo in enumerate(nodos)]
    #print(k_nodo_despues)
    diferencia=np.array(k_nodo_despues)-np.array(k_nodo_antes)
    if (len(np.where(diferencia!=0)[0])==0):
        #print('Rewiring exitoso')
    
        return(D)
#-------------------------------------------------------------------------
#toma un xml y separa los grafos de cada voz, devuelve un dict con con cada grafo como .value y la voz como .key
def f_voices(path, modelo='melodia'): #'melodia' 'ritmo' 'armoniaD' y 'armoniaU'
    
    #creo la lista de listas
    g=dict()

    if modelo == 'melodia':
        song = msc.converter.parse(path)
        for i,elem in enumerate(song.parts):#recorro todas las voces
            m=f_xml2graph(path, nombre_parte=i, modelo=modelo); #obtengo el grafo
            g.update({elem.partName:m})  

        
    if modelo == 'ritmo':
        song = msc.converter.parse(path)
        for i,elem in enumerate(song.parts):#recorro todas las voces
            m=f_xml2graph(path, nombre_parte=i, modelo=modelo); #obtengo el grafo
            g.update({elem.partName:m})  

    if modelo == 'armoniaD':
        song = msc.converter.parse(path)
        for i,elem in enumerate(song.parts):#recorro todas las voces
            chords,tiempos,D,U=f_armon (path, voces) #obtengo el grafo
            g.update({elem.partName:m})  

    if modelo == 'armoniaU':
        song = msc.converter.parse(path)
        for i,elem in enumerate(song.parts):#recorro todas las voces
            chords,tiempos,D,U=f_armon (path, voces) #obtengo el grafo
            g.update({elem.partName:m})  
            
    return(g)
#-------------------------------------------------------------------------
#Toma dos dict y mergea los dos grafos si ambas voces son iguales. Devuelve un dict de los grafos correspondientes 
# con la etiqueta de la voz que pertenecian. 

def f_merge(dict1,dict2,modelo='directed'): #puede ser directed o undirected

    #creo la lista que va a contener los grafos mergeados
    g=dict()
    
    #defino las longitudes
    L1=len(ls1)
    L2=len(ls2)
    
    #defino los instrumentos de las listas
    inst1 = [key for key in ls1.keys()]
    inst2 = [key for key in ls2.keys()]
    dif1 = list(set(inst1)-set(inst2))
    dif2 = list(set(inst2)-set(inst1))
    dif=dif1+dif2
    inter=list(set(inst1) & set(inst2))
    
    #me armo un dict con los grafos y voces que no van a aparecer
    h=dict()
    for i, inst in enumerate(dif):
        if inst in inst1:
            g= ls1['inst']
        else:
            g= ls2['inst']
            
        h.update({inst, g})

    if modelo == 'directed': 
        instrumentos=[]
        for key1, value1 in ls1.iteritems():
            for key2, value2 in ls2.iteritems():
                #Creo los grafos que van a tener todas las canciones mergeadas pero en distintas voces
                M=nx.DiGraph()
                if key1 == key2:
                    #Uno los grafos en uno para cada voz para melodia
                    M=nx.compose(value1,value2)
                    instrumentos.append(key2)
                    g.update({key1:M})  
                            
        
    if modelo == 'undirected': #el modelo me cambia en el tipo de grafo que me tengo que crear
        instrumentos=[]
        for key1, value1 in ls1.iteritems():
            for key2, value2 in ls2.iteritems():
                #aca es no directed
                R=nx.Graph()
                if key1 == key2:
                    #Uno los grafos en uno para cada voz para melodia
                    R=nx.compose(value1,value2)
                    instrumentos.append(key2)
                    g.update({key1:M})
    g.update(h) #le agrego los que quedaron fuera

        
    return(g)
#-------------------------------------------------------------------------
def f_graficar_armonias_directed(G, layout='random',labels=False): #puede ser 'circular' tambien, no se recomienda
        #Recibe un vector de Armonias y realiza un grafico dirigido donde
        #Los nodos en vez de ser las notas son las armonias y se enlazan cuando ocurre una
        #despues de la otra.

        menor_tamano_armonia=2
        mayor_tamano_armonia=5
        color_dict= {"2":"yellow","3": "orange", "4":"red","5":"purple"}

        numero_nodo=-1
        color_nodos=[]
        
        M = G.number_of_edges()
        N = G.number_of_nodes()
        nodos = G.nodes()
        
        
        pos=eval('nx.'+layout+'_layout')(G)

        if labels==True:
            nx.draw_networkx_labels(G,pos,font_size=5,font_color='k')
            
        #asigno los colores a los nodos   
        for i, nodo in enumerate(nodos):
            nod=nodo.split(",")
            tamano_armonia=len(nod)
            if (tamano_armonia>menor_tamano_armonia-1 and tamano_armonia<mayor_tamano_armonia+1):
                color_nodos.append(color_dict[str(tamano_armonia)])    
            
        #Creo enlaces si encontro armonias del tamano buscado
        if len(nodos)>0:
            grados = dict(nx.degree(G))
            nx.draw_networkx_nodes(G,pos,node_list=nodos,node_color=color_nodos,node_size=[50*v for v in grados.values()],alpha=1)
            edges=G.edges()
            nx.draw_networkx_edges(G,pos,edge_list=edges,edge_color='black',width=3,alpha=0.6)
            plt.axis('off')

        return
#---------------------------------------------------------------------------
def f_simultaneidad(cancion, indexes):
        #Toma una cancion y una lista con dos indices [i,j] uno para cada voz.
        #Devuelve una lista de enlaces entre notas cuando estas sonaron en simultaneo [(C4,G4),w,(C4,E3,w)...]
        #La primer nota del enlace corresponde a la voz i y la segunda nota a la voz j. w es la frecuencia de aparicion del enlace.
        #Nota: el enlace solo ocurre cuando la simultaneidad es entre notas de distintas voces.
        #      Si un C4 y un G4 suenan en una misma voz y son simultaneas ese enlace no es creado. Deberia aprecer en el grafo de armonia.
        #Cancion
        song = msc.converter.parse(cancion) # Lee la cancion, queda un elemento stream.Score
        
        #Instrumentos
        instrumentos=[song.parts[indexes[i]].partName for i in range(0,len(indexes))]
        print('Instrumento Seleccionados:'+str(instrumentos))
        partituras=[song.parts[indexes[i]] for i in range(0,len(indexes))]
        compases=[partitura.getElementsByClass(msc.stream.Measure) for p,partitura in enumerate(partituras)]#todos los compases de las voces seleccionadas
        
        #Enlaces
        Enlaces=[]

        #Recorro los compases y en cada compas extraigo las notas de las dos voces. Luego me fijo donde fueron simultaneas y creo los enlaces.
        
        for numero_compas in range(0,len(compases[0])):
                Notas_compas=[]   #lista de listas de notas por cada voz. Cada lista contiene las notas que esa voz fue tocando en en el compas actual
                Tiempos_compas=[] #lista de listas de tiempos por cada voz. Cada lista contiene los tiempos en que sonaron esas notas en el compas actual.
  
                for inst in range(0,len(instrumentos)):
                        compas=compases[inst][numero_compas]
                        notas=[]
                        tiempos=[]
                        frecuencias=[]
                        octavas=[]
                        for i,el in enumerate(compas.flat):
                                isChord=str(type(el))=='<class \'music21.chord.Chord\'>' #me fijo si es un elemento del tipo acorde chord.Chord
                                if isinstance(el,msc.note.Note):#si es una nota
                                        nota_name=str(el.nameWithOctave)+'/'+str(el.quarterLength)
                                        notas.append(nota_name)
                                        
                                        tiempo_nota=float(compas.offset+el.offset)
                                        tiempos.append(tiempo_nota)
                                        
                                        frecuencias.append(el.pitch.frequency)
                                        
                                        octavas.append(el.octave)
                                        
                                if isinstance(el,msc.chord.Chord) & isChord==True:#si es un acorde
                                        for nc,noteChord in enumerate(el):
                                                nota_name=str(noteChord.nameWithOctave)+'/'+str(noteChord.quarterLength)
                                                notas.append(nota_name)
                                                
                                                tiempo_nota=float(compas.offset)+float(el.offset)
                                                tiempos.append(tiempo_nota)
                                                
                                                frecuencias.append(noteChord.pitch.frequency)
                                                
                                                octavas.append(noteChord.octave)

                        Notas_compas.append(notas)
                        Tiempos_compas.append(tiempos)
                #print('Notas: Instrumento1/Instrumento2')
                #print(Notas_compas)
                #print(Tiempos_compas)
                #Ahora me fijo cuales notas son simultaneas y creo el enlace:
                if (len(Tiempos_compas[0])>0 and len(Tiempos_compas[1])>0):
                        for i,itiempos in enumerate(Tiempos_compas[0]):
                                for j,jtiempos in enumerate(Tiempos_compas[1]):
                                        if itiempos==jtiempos:
                                                Enlaces.append([Notas_compas[0][i],Notas_compas[1][j]])
        Enlaces_unicos=[list(i) for i in set(tuple(i) for i in Enlaces)]
        Enlaces_hist = [(x, Enlaces.count(x)) for x in Enlaces_unicos]
        Enlaces_simult=Enlaces_hist
        return(Enlaces_simult)
#---------------------------------------------------------------------------
def f_voicename2abrev(inst):
        #diccionario de voces:
        voice_2_abrev=dict()
        #mantengamos actualizada la siguiente lista segun las canciones que vayamos usando asegurandonos que mapee bien todas las voces en su abreviatura.
        voice_2_abrev={'Voice-Voz-Synth. Voice':'Vz','Soprano':'SVz','Alto':'AVz','Tenor':'TVz','Piano-piano':'Pi','Teclado-Keyboard-Organ':'Tcl','Guitare classique-Guitarra Clasica-Classic Guitar':'clGt.','Guitar-Electric Guitar-Elektrische gitaar-Clean Guitar-Clean Guitar 1-Clean Guitar 2-Overdrive Guitar':'eGt','Bajo-Bass-Bass Guitar-Bass 1-Bass 2-Bass 3-Elektrische bas,bass':'Bj','Contrabajo-Double Bass':'CBj','Violin-Violino':'Vl','Viola':'Vla','Violoncello-Cello-Chelo':'Vch','Flute-Flauta-Flûte-Flet':'Fl','Trumpet-Trompeta-Trumpet in Bb':'Tp','Trombone-Bass Trombone':'Tb','Tuba-C Tuba':'Tba','Clarinet-B♭-Clarinet-Clarinet in Bb-Bass Clarinet':'Cl','Fagot-Bassoon':'Fg','Saxophone-Alto Saxophone-Tenor Saxophone-Baritone Saxophone-Baritone Sax':'Sx'}
        keys=list(voice_2_abrev)
        abrev='empty'
        for k in keys:
                if inst in k:
                        abrev=voice_2_abrev[k]
        return(abrev)
#---------------------------------------------------------------------------
def f_conect(G,H,cancion,indexes):
    #Cancion
    song = msc.converter.parse(cancion) # Lee la cancion, queda un elemento stream.Score
    #Instrumentos
    instrumentos=[song.parts[indexes[i]].partName for i in range(0,len(indexes))]
    #Instrumentos que pueden haber mas de uno:
    instrum_mult=['Pi']

    #Grafos in
    Grafos=[G,H]

    #Grafo out
    F=nx.DiGraph()

    #Abrev
    Abrev=[]
    for g,grafo in enumerate(Grafos):
            nodos=grafo.nodes()
            mapping=dict()
            abrev=f_voicename2abrev(instrumentos[g])
            if abrev in instrum_mult: #si se trata de piano o algun otro instrumento que se nos ocurra que queramos mantener separados podemos decirle que etiquete con un numero ademas.
                abrev=abrev+str(g)
            Abrev.append(abrev)
            #Ahora reetiqueamos los nodos
            for n,nodo in enumerate(nodos):
                    mapping[nodo]=nodo+'/'+abrev
            Grafos[g]=nx.relabel_nodes(Grafos[g],mapping)#reetiueta los grafos de entrada
            F.add_nodes_from(Grafos[g].nodes())#Agrego nodos
            F.add_edges_from(Grafos[g].edges())#Agrego intraenlaces
    
    #Por ultimo llamamos a f_simultaneidad para que cree los interenlaces entre los grafos:
    Enlaces=f_simultaneidad(cancion,indexes)
    #Agregamos interenlaces los cuales son bidirigidos:
    for e,enlace in enumerate(Enlaces):
            F.add_edge(enlace[0][0]+'/'+Abrev[0],enlace[0][1]+'/'+Abrev[1],weight=enlace[1]/2)
            F.add_edge(enlace[0][1]+'/'+Abrev[1],enlace[0][0]+'/'+Abrev[0],weight=enlace[1]/2)
 
    return(F)
#---------------------------------------------------------------------------
def get_layers_2d_position(G,base_pos,Nn,radio=50, proj_angle=45,number_of_layers=5):
    pos = base_pos
    Nold=0
    centro_espiral=np.array([-radio*(math.cos(60*np.pi/180)),radio*(math.sin(60*np.pi/180))])
    for layer in range(0,number_of_layers):
        N =  Nn[layer]
        #calculo un nuevo centro:
        xcentro=radio*(math.cos(60*np.pi/180))
        if layer%2==0:
            signo=-1
        else:
            signo=1
        ycentro=radio*(math.sin(60*np.pi/180))*signo
        centro_espiral=centro_espiral+np.array([xcentro,ycentro])
        for j in range(N):
            numero_nodo=(Nold+j)
            pos[numero_nodo][0] *= math.cos(proj_angle)
            pos[numero_nodo][1] *= math.sin(proj_angle)
            pos[numero_nodo] = np.array([pos[numero_nodo][0]+centro_espiral[0], pos[numero_nodo][1]+centro_espiral[1]],dtype=np.float32)
        Nold=Nold+N
    return pos
#---------------------------------------------------------------------------
def get_layers_3d_position(G,base_pos,Nn,layer_vertical_shift=2,
                 layer_horizontal_shift=0.0, proj_angle=45,number_of_layers=5):
    pos = base_pos
    Nold=0
    for layer in range(0,number_of_layers):
        N =  Nn[layer]
        for j in range(N):
            numero_nodo=(Nold+j)
            pos[numero_nodo][0] *= math.cos(proj_angle)
            pos[numero_nodo][1] *= math.sin(proj_angle)
            pos[numero_nodo] = np.array([pos[numero_nodo][0]+layer*layer_horizontal_shift, pos[numero_nodo][1]+layer*layer_vertical_shift],dtype=np.float32)
        Nold=Nold+N
    return pos     
#---------------------------------------------------------------------------
def f_graficar_2dy3d(cancion,indexes):
    
    #Cancion
    song_name=cancion
    song=msc.converter.parse(song_name)
        
    #Instrumentos
    instrumentos=[song.parts[indexes[i]].partName for i in range(0,len(indexes))]
    print('Instrumento Seleccionados:'+str(instrumentos))
    partituras=[song.parts[indexes[i]] for i in range(0,len(indexes))]

    #Inicializamos variables
    #Creamos el grafico 3D full:
    FullGraph=nx.DiGraph()
    Grafos=[]
    Nodos=dict()
    Numero_nodo=-1 #va a llevar la cuenta de nodos totales
    Numero_enlace=-1 #va a llevar la cuenta de enlaces totales
    Pos=[]           #lista con los vectores posicion de cada 
    Enlaces=dict()
    lista_enlaces=[]
    pos_all=dict()

    #Creamos los grafos que seran dirigido:


    for inst in range(len(instrumentos)):
        #Inicializamos el grafo:
        G=nx.DiGraph()

        #Numero de Instrumento:
        voz=song.getElementsByClass(msc.stream.Part)[indexes[inst]].getElementsByClass(msc.stream.Measure) #todas los compases de la parte voz
        notas=[] #todas las notas incluyendo silencios en la voz analizada
        nodos=[] #solamente los nodos que van a figurar en la red (no tiene elementos repetidos)
        numero_nodo=-1
        pos=[]
        Nold=len(Nodos)#el numero de nodos totales hasta la última que se analizó
        #Nodos
        for i,el in enumerate(voz.flat):
            if isinstance(el,msc.note.Note):
                nota_name=str(el.name)+str(el.octave)+'/'+str(el.quarterLength)
                isornot=nota_name in nodos #chequeo si nota_name esta en nodos
                notas.append(nota_name)
                if isornot==False:
                    nodos.append(nota_name)#si no lo estaba lo agregamos
                    numero_nodo=numero_nodo+1
                    Numero_nodo=Numero_nodo+1
                    G.add_node(numero_nodo)
                    FullGraph.add_node(Numero_nodo)
                    G.node[numero_nodo]['name']=str(el.nameWithOctave)+'/'+str(el.quarterLength)
                    FullGraph.node[Numero_nodo]['name']=str(el.nameWithOctave)+'/'+str(el.quarterLength)
                    G.node[numero_nodo]['freq'] = el.pitch.frequency
                    FullGraph.node[Numero_nodo]['freq']=el.pitch.frequency
                    G.node[numero_nodo]['octava'] = el.octave
                    FullGraph.node[Numero_nodo]['octava']=el.octave
                    G.node[numero_nodo]['duracion'] = el.quarterLength
                    FullGraph.node[Numero_nodo]['duracion']=el.quarterLength
                    Nodos[Numero_nodo]={'id':Numero_nodo,'name':nota_name,'freq':el.pitch.frequency,'octava':el.octave,'duracion':el.quarterLength}
            
            elif isinstance(el,msc.note.Rest):
                nota_name=str(el.name)+'/'+str(1)+'/'+str('inst:')+str(inst) #si es un silencio no vamos a diferenciar sus duraciones
                isornot=nota_name in nodos
                notas.append(nota_name)
                if isornot==False:
                    nodos.append(nota_name)
                    numero_nodo=numero_nodo+1
                    Numero_nodo=Numero_nodo+1
                    G.add_node(numero_nodo)
                    FullGraph.add_node(Numero_nodo)
                    G.node[numero_nodo]['name']=str(el.name)
                    FullGraph.node[Numero_nodo]['name']=str(el)+'/'+str(el.quarterLength)
                    G.node[numero_nodo]['freq'] = 2**(1/(2*np.pi))*20
                    FullGraph.node[Numero_nodo]['freq']=2**(1/(2*np.pi))*20
                    G.node[numero_nodo]['octava'] = 0
                    FullGraph.node[Numero_nodo]['octava']=0
                    G.node[numero_nodo]['duracion'] = 1
                    FullGraph.node[Numero_nodo]['duracion']=1
                    Nodos[Numero_nodo]={'id':Numero_nodo,'name':nota_name,'freq':2**(1/(2*np.pi))*20,'octava':0,'duracion':1}#le asignamos octava 0 a todos los silencios
                
        #Posiciones de nodos:
        pos=dict()
        freq_min = min(np.array(list(nx.get_node_attributes(G,'freq').values())))
        layout='espiral'
        for n,nodo in enumerate(nodos):
            f = G.node[n]['freq']
            d = G.node[n]['duracion']
            nro_oct = G.node[n]['octava']
            theta = 2*np.pi * np.log2(f/freq_min)
            if layout=='espiral':
                x = np.cos(theta)*f/freq_min*(1+d/4)
                y = np.sin(theta)*f/freq_min*(1+d/4)
                pos[n]=np.array([x,y])
                Nodos[Nold+n]['pos']=[x,y]
                pos_all[Nold+n]=[x,y]
            elif layout=='circular':
                nro_oct = G.node[n]['octava']
                x = np.cos(theta)*nro_oct*(1+d/12)
                y = np.sin(theta)*nro_oct*(1+d/12)
                pos[n]=np.array([x,y])
                Nodos[Nold+n]['pos']=[x,y]
                pos_all[Nold+n]=[x,y]
            
        #Intra-Enlaces(dentro de cada capa):
        for i in range(0,len(notas)-1):
            idx1=Nold+nodos.index(notas[i])
            idx2=Nold+nodos.index(notas[i+1])
            isornot=[idx1,idx2] in lista_enlaces
            if isornot==False:
                Numero_enlace=Numero_enlace+1
                Enlaces[str([idx1,idx2])]={'name':[idx1,idx2],'weight':1}
                lista_enlaces.append([idx1,idx2])
            else:
                Enlaces[str([idx1,idx2])]['weight']+=1
            
        #Terminamos agregando el grafo G y el vector posicion a las listas Grafos y Pos
        Grafos.append(G)
        Pos.append(pos)

    #Colores nodos:
    octavas = np.array([Nodos[n]['octava'] for n in range(0,len(Nodos))])
    oct_min = min(octavas)
    oct_max = max(octavas)
    colores_oct_nro = (octavas - oct_min)/(oct_max - oct_min)
    color_map='rainbow'
    m = cm.ScalarMappable(norm=None, cmap=color_map)
    colores_oct = m.to_rgba(colores_oct_nro) #colores por octava
    for n in range(0,len(Nodos)):
        Nodos[n]['color']=colores_oct[n]

    #Numero de nodos:
    Nodos_number=[]
    Nodos_number_acum=[0]
    suma=-1
    for g,grafos in enumerate(Grafos):
        suma=suma+Grafos[g].number_of_nodes()
        Nodos_number.append(Grafos[g].number_of_nodes())
        Nodos_number_acum.append(suma)

    Enlaces_intra_idx=[]
    #Inter-Enlaces (entre capas)
    for ig in range(0,len(Grafos)-1): 
        for jg in range(ig+1,len(Grafos)):
            nodos_ig_dict=Grafos[ig].nodes('name')
            nodos_jg_dict=Grafos[jg].nodes('name')
            nodos_ig=[nodos_ig_dict[nodo] for n,nodo in enumerate(list(Grafos[ig].nodes()))]
            nodos_jg=[nodos_jg_dict[nodo] for n,nodo in enumerate(list(Grafos[jg].nodes()))]
            print('Creando inter-enlaces entre las voces:')
            Enlaces_intra_names=f_simultaneidad(song_name,[indexes[ig],indexes[jg]])
            #Nota:puede pasar que una nota que este en los enlaces no aparezca dentro de los nodos de melodia, porque melodia solo se quedo con la mas aguda...
            for e,enlace_intra in enumerate(Enlaces_intra_names):
                notaig=enlace_intra[0][0]
                notajg=enlace_intra[0][1]
                weight=enlace_intra[1]
                #Ahora lo que hay que hacer es agarrar el nombre de las notas y mapearlas a la variable numero de nodo.
                if (notaig in nodos_ig) and (notajg in nodos_jg): #nos fijamos si esta en la lista de nodos que melodia encontro
                    Enlaces_intra_idx.append([Nodos_number_acum[ig]+nodos_ig.index(notaig),Nodos_number_acum[jg]+nodos_jg.index(notajg),weight/2])
                    Enlaces_intra_idx.append([Nodos_number_acum[jg]+nodos_jg.index(notajg),Nodos_number_acum[ig]+nodos_ig.index(notaig),weight/2])
                
    #Agregamos los Inter-enlaces:
    for e,enlace_intra in enumerate(Enlaces_intra_idx):
        FullGraph.add_edge(enlace_intra[0],enlace_intra[1])
        FullGraph[enlace_intra[0]][enlace_intra[1]]['weight']=enlace_intra[2]

    #Agregamos los Intra-enlaces al grafo mg
    #Intra-Enlaces (dentro de cada capa)
    for e,enlace in enumerate(Enlaces):
        FullGraph.add_edge(Enlaces[enlace]['name'][0],Enlaces[enlace]['name'][1])
        FullGraph[Enlaces[enlace]['name'][0]][Enlaces[enlace]['name'][1]]['weight']=Enlaces[str([idx1,idx2])]['weight']

    #Pesos de los enlaces:
    weights = [FullGraph[u][v]['weight']/20 for u,v in FullGraph.edges()]

    ##Grafico 2-d
    R=100 #radio entorno de las layers
    Delta=25
    nl=len(indexes)#numero de layers
    plt.figure(figsize=(15,15))
    plt.title('2d Layout')
    base_pos2d=copy.deepcopy(pos_all)
    pos2d=get_layers_2d_position(FullGraph,base_pos=base_pos2d,Nn=Nodos_number,radio=R,number_of_layers=nl)
    nx.draw(FullGraph,pos=pos2d,node_color=colores_oct,node_size=60,width=weights)
    #Nombres de los instrumentos
    text_yposition=0
    font = FontProperties()
    font.set_style('normal')
    font.set_weight('bold')
    centro_espiral=np.array([-R*(math.cos(60*np.pi/180)),R*(math.sin(60*np.pi/180))])
    for i,inst in enumerate(instrumentos):
        xcentro=R*(math.cos(60*np.pi/180))
        if i%2==0:
            signo=-1
        else:
            signo=1
        ycentro=R*(math.sin(60*np.pi/180))*signo
        centro_espiral=centro_espiral+np.array([xcentro,ycentro])
        centro_texto=[centro_espiral[0],-R/2-Delta]
        plt.text(centro_texto[0],centro_texto[1],"{}".format(inst),fontproperties=font)


    ##Grafico 3-d
    fig2=plt.figure(figsize=(15,15))
    plt.title('3d Layout')
    base_pos3d=copy.deepcopy(pos_all)
    pos3d=get_layers_3d_position(FullGraph,base_pos=base_pos3d,Nn=Nodos_number,layer_vertical_shift=50,layer_horizontal_shift=0.0,proj_angle=50,number_of_layers=nl)
    nx.draw(FullGraph,pos=pos3d,node_color=colores_oct,node_size=60,width=weights)
    #Nombres de los instrumentos
    text_yposition=0
    font = FontProperties()
    font.set_style('normal')
    font.set_weight('bold')
    for inst in instrumentos:
        text_xposition=0.85
        text_yposition=text_yposition+(1/float(len(instrumentos)+1))
        plt.text(text_xposition,text_yposition,"{}".format(inst),fontproperties=font,transform=plt.gca().transAxes)
    plt.show()
#---------------------------------------------------------------------------------
#toma un grafo y la cantidad de los nuevos nodos. Devuelve una lista random con las notas
def random_walk_1_M(G,k):
    ls = []
    nodos=list(G.nodes())
    #Me armo un dict, con el key del nombre del nodo y una lista
    weights={} 
    
    M=nx.to_numpy_matrix(G) #obtenemos la matriz de adyacencia no esparsa
    L=M.shape[1]
    #normalizamos
    for i, nodo in enumerate(nodos):
        if (np.sum(M[i]) > 0):
            M[i] = M[i]/np.sum(M[i])
        #los pesos de salto son los valores de las filas normalizadas.
        #paso M[i] a una lista, le hago .tolist()[0] asi no me grafica extra corchetes
        weights.update({str(nodo): M[i].tolist()[0]})  

    #Hago la caminata
    #Me paro en algun nodo inicial random
    nodo_i=random.choice(nodos)
    for i in range(k):
        if i == 0: #para el primer paso
            ls.append(nodo_i)
            #Para ese nodo inicial veo que elementro de matriz le corresponde
            #se podria hacer dos random walks, una con los pesos acumulados (cum_weights) y otras sin.
            #la lista tiene que ser los nodos 
            nodo_ran=random.choices(nodos, weights[nodo_i],k=1)
            #weights[str(nodo_i)] me da lista de primeros vecinos
            ls.append(nodo_ran[0]) #agrego el que haya salido. Le pongo [0] xq random.choices me devuelve una lista
        else:
            #veo los vecinos del nuevo nodo
            nodo_ran=random.choices(nodos, weights[nodo_ran[0]],k=1)
            ls.append(nodo_ran[0]) #agrego el que haya salido
        
    return(ls)
#---------------------------------------------------------------------------------
def f_list2seq(lista,nombre):
    # Toma una lista de notas (generadas por la caminata al azar) y genera un stream
    # Guarda la partitura xml y el audio midi con el nombre asignado
    lista = [x.split("/") for x in lista]
    notas = lista.copy()
    L = len(notas)
    for i in range(L):
        if len(lista[i])==2:
            notas[i] = msc.note.Note(lista[i][0],quarterLength=float(lista[i][1]))
        elif len(lista[i])==1:
            notas[i] = msc.note.Rest(quarterLength=1.0)
    cancion = msc.stream.Stream()
    for i in range(L):
        cancion.append(notas[i])
    cancion.write("MusicXML", nombre+".xml")
    cancion.write("Midi", nombre+".mid")
#------------------------------------------------------------------------------------
def f_compose(G,H):
    #recibe un grafo G y otro H y devuelve un grafo compuesto de ambos.
    #El peso resultante es la suma de los pesos de los enlaces en comun si los hubiese.
    #Chequeamos los tipos de grafos:
    
    if type(G)!=type(H):
        out=0
    elif type(G)==type(H):
        out=1
        F=nx.DiGraph()
        #Nodos:
        nodosG=G.nodes()
        nodosH=H.nodes()
        nodosG_H=list(set(nodosG).difference(set(nodosH)))
        nodosH_G=list(set(nodosH).difference(set(nodosG)))
        nodosintersection=list(set(nodosG).intersection(set(nodosH)))
        for nodo in nodosG_H:
            F.add_node(nodo)
            F.node[nodo]['freq']= G.node[nodo]['freq']
            F.node[nodo]['duracion']= G.node[nodo]['duracion']
            F.node[nodo]['octava']=G.node[nodo]['octava']
        for nodo in nodosH_G:
            F.add_node(nodo)
            F.node[nodo]['freq']= H.node[nodo]['freq']
            F.node[nodo]['duracion']= H.node[nodo]['duracion']
            F.node[nodo]['octava']=H.node[nodo]['octava']
        for nodo in nodosintersection:
            F.add_node(nodo)
            F.node[nodo]['freq']= G.node[nodo]['freq']
            F.node[nodo]['duracion']= G.node[nodo]['duracion']
            F.node[nodo]['octava']=G.node[nodo]['octava']
        
        #Enlaces
        enlacesG=G.edges()
        enlacesH=H.edges()
        enlacesG_H=list(set(enlacesG).difference(set(enlacesH)))
        enlacesH_G=list(set(enlacesH).difference(set(enlacesG)))
        enlacesintersection=list(set(enlacesG).intersection(set(enlacesH)))
        #Enlaces G
        for enlace in enlacesG_H:
            enlaceweight_G=G.edges[enlace[0],enlace[1]]['weight']
            F.add_edge(enlace[0],enlace[1])
            F.edges[enlace[0],enlace[1]]['weight']=enlaceweight_G
        #Enlaces H
        for enlace in enlacesH_G:
            enlaceweight_H=H.edges[enlace[0],enlace[1]]['weight']
            F.add_edge(enlace[0],enlace[1])
            F.edges[enlace[0],enlace[1]]['weight']=enlaceweight_H
        #Enlaces interseccion
        for enlace in enlacesintersection:
            enlaceweight_G=G.edges[enlace[0],enlace[1]]['weight']
            enlaceweight_H=H.edges[enlace[0],enlace[1]]['weight']
            F.add_edge(enlace[0],enlace[1])
            F.edges[enlace[0],enlace[1]]['weight']=enlaceweight_G+enlaceweight_H
        
    if out==0:
        return('los tipos de grafo no son compatibles')
    elif out==1:
        return(F)
