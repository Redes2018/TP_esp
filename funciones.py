import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd
import networkx as nx
import music21 as msc

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
# graficar_armonias_undirected(G, color_map='rainbow',layout='espiral')
# graficar_armonias_directed(Armonias)
# f_dist_escalas (cancion, nombre_parte=0)
# f_full_graph(path)
# f_hierarchy(G)
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
                            G.node[nota_name]['octava'] = oct_min-1 # A los silencios se les asocia una octava menos que las notas, para el grafico
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
                            G.node[nota_name]['octava'] = 0.1
                            G.node[nota_name]['duracion'] = d
                        notas[i] = nota_name

            # Enlaces pesados
            L = len(notas)
            for i in range(L-1):  # recorremos desde la primera hasta la anteultima nota, uniendo sucesivas
                if G.has_edge(notas[i],notas[i+1]):
                    G[notas[i]][notas[i+1]]['weight']+=1 # si el enlace ya existe, se le agrega +1 al peso
                else:
                    G.add_edge(notas[i],notas[i+1],weight=1) # si el enlace no existe, se crea con peso 1
            Gs.append(G)
    else:
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
                        G.node[nota_name]['octava'] = oct_min-1 # A los silencios se les asocia una octava menos que las notas, para el grafico
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
                        G.node[nota_name]['octava'] = 0.1
                        G.node[nota_name]['duracion'] = d
                    notas[i] = nota_name

        # Enlaces pesados
        L = len(notas)
        for i in range(L-1):  # recorremos desde la primera hasta la anteultima nota, uniendo sucesivas
            if G.has_edge(notas[i],notas[i+1]):
                G[notas[i]][notas[i+1]]['weight']+=1 # si el enlace ya existe, se le agrega +1 al peso
            else:
                G.add_edge(notas[i],notas[i+1],weight=1) # si el enlace no existe, se crea con peso 1
        Gs = G
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
	grados = dict(nx.degree(G))
	nx.draw_networkx_nodes(G,pos,node_list=grados.keys(),node_color=colores_oct,node_size=[50*v for v in grados.values()])
	
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
    
    quarter_lengths=[0.5/3,0.25,1/3,0.5,0.75,1.0,1.25,1.5,1.75,2.0,2.25,2.5,2.75,3.0,3.25,3.5,3.75,4.0]
    figuras=['tresillo de semicorchea','semicorchea','tresillo de corchea','corchea','corchea puntillo','negra','negra semicorchea','negra puntillo','negra doble puntillo','blanca','blanca semicorchea','blanca corchea','blanca corchea puntillo','blanca puntillo','blanca puntillo semicorchea','blanca puntillo corchea','blanca puntillo corchea puntillo','redonda']
    figuras_abrev=['ts','s','tc','c','cp','n','ns','np','npp','b','bs','bc','bcp','bp','bps','bpc','bpcp','r']
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
				rhytms_compas.append('rest/'+ql_2_fig(el.quarterLength))

		#Una vez creada la lista rhytm_compas empiezo a recorrerla tomando grupos de notas de tamano segun lo indique en length:
		for r in range(0,len(rhytms_compas)-length+1):
			motif=[]
			for l in range(0,length):
				#motif.append(rhytms[r+l])
				if  type(rhytms_compas[r+l]) is not str:
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
	yTick_position=[]
	yTick_name=[]
	contador=-1
	contador_tick=-0.5
	motif_umbral=0
	for m,motif in enumerate(motifs_rhytmic):
		if frecuencias[m]>motif_umbral:
			contador+=1
			contador_tick+=1
			plt.barh(contador,frecuencias[m],color='red')
			yTick_position.append(contador_tick)
			yTick_name.append(motif)
	plt.yticks(yTick_position,yTick_name, rotation=0,fontsize=10)
	plt.title('Rhytmics '+str(length)+'-Motifs',fontsize=20)
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

def f_grado_dist_M(G):
    
    H=G.copy()
    nodos=H.nodes() 
    N=len(nodos)
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

def f_grado_dist_R(G):
    
    H=G.copy()
    nodos=H.nodes() 
    N=len(nodos)
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
    
    plt.plot(bin_centros,pk_logbin,'bo')
    plt.xlabel('$log(k)$',fontsize=20)
    plt.xscale('log')
    plt.ylabel('$log(p_{k})$',fontsize=20)
    plt.yscale('log')
    plt.title('Bin log - Escala log',fontsize=20)
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

    # Coef de clustering medio:
    # c_1 = #triangulos con vertice en 1 / triangulos posibles con vertice en 1
    # C_mean es el promedio de los c_i sobre todos los nodos de la red
    C_mean = nx.average_clustering(H)

    # Clausura transitiva de la red o Global Clustering o Transitividad:
    # C_g = 3*nx.triangles(G1) / sumatoria sobre (todos los posibles triangulos)
    C_gclust = nx.transitivity(H)

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
        #1)Construye grafo con Igraph el cual es no dirigido.(ver graficar_armonias_undirected)
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
        G=nx.Graph()
        I=ig.Graph()
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
        menor_tamano_armonia=3 #indicamos cual es el tamano minimo de armonias a graficar
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
                                        G.add_node(nota)
                                        G.node[nota]['freq'] = tonos_ordenados[n]
                                        G.node[nota]['octava'] = octava_ordenada[n]
                                        G.node[nota]['duracion']=4.0 #por ahora no distinguimos duracion en las armonias
                                        if ((nota in Inodos)==False and len(armonia)>menor_tamano_armonia-1):#solo agrega el nodo si este no aparecio nunca y nodos en armnonias de mas de 2 notas.
                                                numero_nodo=numero_nodo+1
                                                I.add_vertex(numero_nodo)
                                                I.vs[numero_nodo]["name"]=nota
                                                Inodos.append(nota)
                                                I.vs[numero_nodo]["freq"]= tonos_ordenados[n]
                                                I.vs[numero_nodo]["octava"]= octava_ordenada[n]
                                                I.vs[numero_nodo]["duracion"]=4.0
                        armonia=[notas[t]]
                        tonos=[frecuencias[t]]
                        octava=[octavas[t]]
                        
        #Agregamos los enlaces al grafo:
        
        for a,armonia in enumerate(armonias):
                for n in range (0,len(armonia)):
                        for m in range(n+1,len(armonia)):
                                if (G.has_edge(armonia[n],armonia[m])):
                                        G[armonia[n]][armonia[m]]['weight']+=1        
                                else:
                                        G.add_edge(armonia[n],armonia[m],weight=1)

        #Agregamos los enlaces al grafo I:
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
                                        I.add_edge(armonia[0][n],armonia[0][m])
                                        I.es[numero_enlace]['tamano']=tamano_armonia
                                        I.es[numero_enlace]['color']=color_dict[str(tamano_armonia)]
                                        I.es[numero_enlace]['weigth']=armonia[1]
                                        pesos_edges.append(armonia[1])
                                        colores_edges.append(color_dict[str(tamano_armonia)])

        #1)Grafico con Igraph (no dirigido)
        graficar_armonias_undirected(I, color_map='rainbow',layout='espiral')

        #2)Grafico con Igraph (dirigido):
        graficar_armonias_directed(armonias)

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
        #1)Construye grafo con Igraph el cual es no dirigido.(ver graficar_armonias_undirected)
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
        print('Instrumento Seleccionados:'+str(instrumentos))
        partituras=[song.parts[indexes[i]] for i in range(0,len(indexes))]
        compases=[partitura.getElementsByClass(msc.stream.Measure) for p,partitura in enumerate(partituras)]#todos los compases de las voces seleccionadas
        
        #Armonias
        Armonias_song=[]
        Tiempos_armonias_song=[]
        menor_tamano_armonia=3 #indicamos cual es el tamano minimo de armonias a graficar
        mayor_tamano_armonia=5 #indicamos cual es el tamano maximo de armonias a graficar

        #Incializamos el grafo en nx y en ig: los nodos seran notas. 
        G=nx.Graph()
        I=ig.Graph()
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
                                                        G.add_node(nota)
                                                        G.node[nota]['freq'] = tonos_ordenados[n]
                                                        G.node[nota]['octava'] = octava_ordenada[n]
                                                        G.node[nota]['duracion']=4.0 #por ahora no distinguimos duracion en las armonias
                                                        if ((nota in Inodos)==False and len(armonia)>menor_tamano_armonia-1):#solo agrega el nodo si este no aparecio nunca y nodos en armnonias de mas de 2 notas.
                                                                numero_nodo=numero_nodo+1
                                                                I.add_vertex(numero_nodo)
                                                                I.vs[numero_nodo]["name"]=nota
                                                                Inodos.append(nota)
                                                                I.vs[numero_nodo]["freq"]= tonos_ordenados[n]
                                                                I.vs[numero_nodo]["octava"]= octava_ordenada[n]
                                                                I.vs[numero_nodo]["duracion"]=4.0
                                        armonia=[Notas_compas[t]]
                                        tonos=[Frecuencias_compas[t]]
                                        octava=[Octavas_compas[t]]
                                
                        #Agregamos los enlaces al grafo G:
                        #Los enlaces se haran si dos notas pertenecen a una misma armonia. Los enlaces son no dirigidos.
        
                        for a,armonia in enumerate(Armonias):
                                for n in range (0,len(armonia)):
                                        for m in range(n+1,len(armonia)):
                                                if (G.has_edge(armonia[n],armonia[m])):
                                                        G[armonia[n]][armonia[m]]['weight']+=1        
                                                else:
                                                        G.add_edge(armonia[n],armonia[m],weight=1)
        #Agregamos los enlaces al grafo I:
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
                                        I.add_edge(armonia[0][n],armonia[0][m])
                                        I.es[numero_enlace]['tamano']=tamano_armonia
                                        I.es[numero_enlace]['color']=color_dict[str(tamano_armonia)]
                                        I.es[numero_enlace]['weigth']=armonia[1]
                                        pesos_edges.append(armonia[1])
                                        colores_edges.append(color_dict[str(tamano_armonia)])

        #1)Grafico con Igraph (no dirigido):
        graficar_armonias_undirected(I, color_map='rainbow',layout='espiral')

        #2)Grafico con Igraph (dirigido):
        graficar_armonias_directed(Armonias_song)

        #3)Histograma de armonias:
        plt.figure
        yTick_position=[]
        yTick_name=[]
        contador=-1
        contador_tick=-0.5
        dtype = [('name', 'S28'), ('count', int)]
   
        for t,tamano in enumerate(np.arange(mayor_tamano_armonia,menor_tamano_armonia-1,-1)):
                armonias_T=[] #lista con pares
                for a, armonia in enumerate(Armonias_hist):
                        if len(Armonias_hist[a][0])==tamano:
                                armonias_T.append((str(Armonias_hist[a][0]),int(Armonias_hist[a][1])))        
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

        #print(Armonias_song,Tiempos_armonias_song)
        if G.number_of_nodes() !=0:
                print('Armonias encontradas y su tiempo de aparicion')
                return(Armonias_song,Tiempos_armonias_song)
        else:
                return('No se encontraron armonias entre estas voces')
##---------------------------------------------------------------------------
def graficar_armonias_undirected(G, color_map='rainbow',layout='espiral'):
        #Grafica el grafo no dirigido G. Graficamos en colores si son enlaces por armonias de 2-3-4-5 o mas notas.
        #Los enlaces estan pesados por la aparicion de esa armonia.El grosor del enlace es por su peso.
        #El tamano de los nodos se calculo segun el strength (es un grado pesado por el peso de los enlaces)
        
        M=G.ecount()
        N=G.vcount()
        nodos=list(G.vs)
        if len(nodos)>0:
                nodos_name=[nodos[n]['name'] for n in range(0,len(nodos))]
                edges=list(G.es)
                freq_min=min(np.array(list(G.vs["freq"])))

                pos=dict()
                posiciones=[]

                for n,nodo in enumerate(nodos):
                        f=nodos[n]['freq']
                        d=nodos[n]['duracion']
                        theta=2*np.pi*np.log2(f/freq_min)
                        if layout=='espiral':
                                x = np.cos(theta)*f/freq_min*(1+d/4)
                                y = np.sin(theta)*f/freq_min*(1+d/4)
                                pos[n] = np.array([x,y])
                                posiciones.append(np.array([x,y]))
                        elif layout=='circular':
                                nro_oct = nodos[n]['octava']
                                x = np.cos(theta)*nro_oct*(1+d/12)
                                y = np.sin(theta)*nro_oct*(1+d/12)
                                pos[n] = np.array([x,y])
                                posiciones.append(np.array([x,y]))

                #Colores nodos
                octavas=np.array([nodos[n]['octava'] for n,nodo in enumerate(nodos)])
                oct_min = min(octavas)
                oct_max = max(octavas)
                colores_oct_nro = (octavas - oct_min)/(oct_max - oct_min)
                m = cm.ScalarMappable(norm=None, cmap=color_map)
                colores_oct = m.to_rgba(colores_oct_nro)
                color_nodos=[[colores_oct[i][0],colores_oct[i][1],colores_oct[i][2]] for i in range(0,len(colores_oct))]

                #Tamano de nodos(pesados)
                strength_nodos=G.strength()
                tamano_labels=strength_nodos+10*np.ones(len(strength_nodos))
                tamano_nodos=strength_nodos+50*np.ones(len(strength_nodos))
        
                #Enlaces
                color_edges=[]
                pesos_edges=[edges[e]['weigth'] for e in range(0,len(edges))]
                color_edges=[edges[e]['color'] for e in range(0,len(edges))]
        
                #Grafico
                layout = posiciones
                visual_style = {}
                visual_style["vertex_size"] = tamano_nodos #o 100
                visual_style["vertex_label"] = nodos_name
                visual_style["vertex_color"] = 'black'
                visual_style["vertex_frame_color"] = color_nodos
                visual_style["vertex_frame_width"] = 3
                visual_style["vertex_label_color"] = color_nodos
                visual_style["vertex_label_size"] = tamano_labels #o 50
                visual_style["edge_width"] = pesos_edges
                visual_style["edge_color"]= color_edges
                visual_style["layout"] = layout
                visual_style["bbox"] = (3000, 3000)
                visual_style["margin"] = 300
                ig.plot(G, **visual_style)
                ig.plot(G, "armonias_undirected.png", **visual_style)
        else:
                print('No se encontraron armonias del tamano buscado entre estas voces')
        
        return()
##---------------------------------------------------------------------------
def graficar_armonias_directed(Armonias):
        #Recibe un vector de Armonias y realiza un grafico dirigido donde
        #Los nodos en vez de ser las notas son las armonias y se enlazan cuando ocurre una
        #despues de la otra.

        menor_tamano_armonia=3
        mayor_tamano_armonia=5
        color_dict= {"2":"purple","3": "blue", "4":"red","5":"green"}
        
        J=ig.Graph(directed=True)
        numero_nodo=-1
        Jnodos=[]
        color_nodos=[]
        
        if len(Armonias)>0: #si no encontro armonias no hace nada  
                #Creo nodos:
                for a in range(0,len(Armonias)):
                        tamano_armonia=len(Armonias[a])
                        if (tamano_armonia>menor_tamano_armonia-1 and tamano_armonia<mayor_tamano_armonia+1):
                                if (str(Armonias[a]) in Jnodos)==False:
                                        numero_nodo=numero_nodo+1
                                        J.add_vertex(numero_nodo)
                                        J.vs[numero_nodo]["name"]=str(Armonias[a])
                                        Jnodos.append(str(Armonias[a]))
                                        color_nodos.append(color_dict[str(tamano_armonia)])
        
                #Creo enlaces si encontro armonias del tamano buscado
                if len(Jnodos)>0:
                        J.add_edge(str(Jnodos[0]),str(Jnodos[1]))
                        for a in range(1,len(Jnodos)-1):
                                J.add_edge(str(Jnodos[a]),str(Jnodos[a+1]))
                        
                        #Propiedades
                        layout = J.layout("random")
                        visual_style = {}
                        visual_style["vertex_size"] = 120
                        visual_style["vertex_label"] = Jnodos
                        visual_style["vertex_color"] = 'black'
                        visual_style["vertex_label_color"] = color_nodos
                        visual_style["vertex_frame_color"]= color_nodos
                        visual_style["vertex_label_size"] = 10
                        visual_style["edge_width"] = 4
                        visual_style["layout"] = layout
                        visual_style["bbox"] = (3000, 3000)
                        visual_style["margin"] = 300
                        ig.plot(J, **visual_style)
                        ig.plot(J, "armonias_directed.png", **visual_style)
                else:
                        print('No se encontraron armonias del tamano buscado entre estas voces')
        return()
#-----------------------------------------------------------------------------
def f_nx2Igraph(G,directed=False,color_map='rainbow',layout='espiral'):
        #Toma un grafo de nx y construye un grafo en igraph y lo grafica.

        #Creamos el grafo.
        if directed==False:
                I=ig.Graph()
        else:
                I=ig.Graph(directed=True)
                
        #Nodos
        nodos_name=list(G.nodes())
        enlaces=list(G.edges())
        for n,nodo in enumerate(nodos_name):
                I.add_vertex(n)
                I.vs[n]["name"]=nodo
                I.vs[n]["freq"]= G.node[nodo]['freq']
                I.vs[n]["octava"]= G.node[nodo]['octava']
                I.vs[n]["duracion"]=G.node[nodo]['duracion']
        nodos=list(I.vs)

        #Posiciones
        pos=dict()
        posiciones=[]
        freq_min=min(np.array(list(I.vs["freq"])))

        for n,nodo in enumerate(nodos):
                f=nodos[n]['freq']
                d=nodos[n]['duracion']
                theta=2*np.pi*np.log2(f/freq_min)
                if layout=='espiral':
                        x = np.cos(theta)*f/freq_min*(1+d/4)
                        y = np.sin(theta)*f/freq_min*(1+d/4)
                        pos[n] = np.array([x,y])
                        posiciones.append(np.array([x,y]))
                elif layout=='circular':
                        nro_oct = nodos[n]['octava']
                        x = np.cos(theta)*nro_oct*(1+d/12)
                        y = np.sin(theta)*nro_oct*(1+d/12)
                        pos[n] = np.array([x,y])
                        posiciones.append(np.array([x,y]))

        #Colores nodos
        octavas=np.array([nodos[n]['octava'] for n,nodo in enumerate(nodos)])
        oct_min = min(octavas)
        oct_max = max(octavas)
        colores_oct_nro = (octavas - oct_min)/(oct_max - oct_min)
        m = cm.ScalarMappable(norm=None, cmap=color_map)
        colores_oct = m.to_rgba(colores_oct_nro)
        color_nodos=[[colores_oct[i][0],colores_oct[i][1],colores_oct[i][2]] for i in range(0,len(colores_oct))]

        #Tamano de nodos(pesados)
        strength_nodos=I.strength()
        tamano_labels=strength_nodos+10*np.ones(len(strength_nodos))
        tamano_nodos=strength_nodos+50*np.ones(len(strength_nodos))

        #Enlaces
        pesos_edges=[]
        alphas=[]
        for e,enlace in enumerate(enlaces):
                I.add_edge(enlace[0],enlace[1])
                I.es[e]['weigth']=G[enlace[0]][enlace[1]]['weight']
                pesos_edges.append(G[enlace[0]][enlace[1]]['weight'])

        pesos_max=np.max(np.array(pesos_edges))
        alphas = [(pesos_edges[i]/pesos_max)**(1./2.) for i in range(0,len(pesos_edges))]
        
        #Propiedades
        layout = posiciones
        visual_style = {}
        visual_style["vertex_size"] = 40
        visual_style["vertex_label"] = nodos_name
        visual_style["vertex_color"] = 'black'
        visual_style["vertex_label_color"] = color_nodos
        visual_style["vertex_frame_color"]= color_nodos
        visual_style["vertex_label_size"] = 20
        visual_style["edge_width"] = pesos_edges
        visual_style["edge_opacity"] = alphas
        visual_style["layout"] = layout
        visual_style["bbox"] = (3000, 3000)
        visual_style["margin"] = 300
        ig.plot(I, **visual_style)
        #ig.plot(I, "new_graph.png", **visual_style)

        return()
#---------------------------------------------------------------------------

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
            print('Partes: '+str(lista_partes)+'. Parte seleccionada: '+str(lista_partes[nombre_parte]))
        except IndexError:
            part = song.parts[0]
            print(nombre_parte+' no es un índice aceptable. Partes: '+str(lista_partes)+'. Parte seleccionada: '+str(lista_partes[0]))
    # Si el input es nombre (str) y no está entre las partes, selecciona la primera voz
    elif not nombre_parte in lista_partes: 
        part = song.parts[0]
        print(nombre_parte+' no está entre las partes: '+str(lista_partes)+'. Parte seleccionada: '+str(lista_partes[0]))
    else:
        indexes = [index for index, name in enumerate(lista_partes) if name == nombre_parte]
        if len(indexes)==1:
            part = song.parts[indexes[0]]
        else:
            part = []
            for j in indexes:
                part.append(song.parts[j])
        print('Partes: '+str(lista_partes)+'. Parte(s) seleccionada(s): '+str([lista_partes[i] for i in indexes]))
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
    else:
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
    return(Gs)

#---------------------------------------------------------------------------
def f_full_graph(path): #hay que pasarle una direccion para un archivo y arma un grafo de ese xml con todas las voces

    #Creo los grafos que van a tener todas las voces de un artista y todos sus temas
    M=nx.DiGraph()
    R=nx.Graph()
    #H1=nx.DiGraph()
    #H2=nx.DiGraph()

    song = msc.converter.parse(path)
    L=len(song.parts) #recorro todas las voces
    for i in range(L):
        #Uno los grafos en uno para cada voz para melodia
        m=f_xml2graph(path, nombre_parte=i, modelo='melodia');
        M=nx.compose(M,m)
        #Uno los grafos en uno para cada voz para ritmo
        r=f_xml2graph(path, nombre_parte=i, modelo='ritmo');
        R=nx.compose(R,r)
        #Para la armonia mediante f_xml2graph_armonia (cancion, index)
        #h1=f_xml2graph_armonia(paths[j], i)
        #H1=nx.compose(H1,h1)
        #Para la armonia mediante f_armon (cancion, indexes)
        #h1=f_xml2graph_armonia(paths[j], i)
        #H1=nx.compose(H1,h1)
        
    return(M,R)

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
def transitivity_motifs(G):
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
