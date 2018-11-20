import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd
import networkx as nx
import music21 as msc

#---------------------------------------------------------------------------------------------------------
#            FUNCIONES PARA ANALISIS DE MUSICA:
#---------------------------------------------------------------------------------------------------------

def f_xml2graph(cancion, nombre_parte=None,modelo='melodia'): 
    # Toma como input una canción (y el nombre de la parte o voz) y
    # devuelve un grafo G o una lista de grafos Gs si mas de una parte tiene el mismo nombre
    # cancion puede ser la ubicacion del archivo o directamente el Score de music21
    
    # Cancion
    if type(cancion)==msc.stream.Score:
        song = cancion
    else:
        song = msc.converter.parse(cancion) # Lee la partitura, queda un elemento stream.Score

    # Lista de nombres de las partes
    Lp = len(song.parts) # Cantidad de partes (voces)
    lista_partes = list(np.zeros(Lp)) # Crea una lista donde se van a guardar los nombres de las partes
    for i,elem in enumerate(song.parts):
        lista_partes[i] = elem.partName # Guarda los nombres de las partes en la lista

    # Seleccion de la parte a usar
    nombre_parte = nombre_parte or lista_partes[0] # Si no tuvo nombre_parte como input,toma la primera voz

    # Si el nombre de la parte no esta en la lista, toma la primera voz
    if not nombre_parte in lista_partes: 
        part = song.parts[0]
        print(nombre_parte+' no está entre las partes: '+str(lista_partes)+'. Parte seleccionada: '+str(lista_partes[0]))
        # Ademas devuelve el "error" de que el nombre no esta entre las partes, y dice que parte usa
    else:
        indexes = [index for index, name in enumerate(lista_partes) if name == nombre_parte]
        if len(indexes)==1:
            part = song.parts[indexes[0]]
        else:
            part = []
            for j in indexes:
                part.append(song.parts[j])
        print('Partes: '+str(lista_partes)+'. Parte(s) seleccionada(s): '+str([lista_partes[i] for i in indexes]))
        # Si el nombre si esta entre las partes, lo selecciona y tambien dice que parte usa

    # Crea la(s) voz(ces) analizada(s) (todos los compases) y se queda con
    # todas las notas incluyendo silencios con offset 'absoluto' (flat)
    if type(part) == list:
        voz = []
        for parte in part:
            voz.append(parte.getElementsByClass(msc.stream.Measure))
        lista_notas = []
        for voice in voz:
            notes = [x for x in voice.flat if type(x)==msc.note.Note or type(x)==msc.note.Rest]
            lista_notas.append(notes)
    else:
        voz = part.getElementsByClass(msc.stream.Measure)
        notas = [x for x in voz.flat if type(x)==msc.note.Note or type(x)==msc.note.Rest]

    # Creamos el grafo dirigido G o lista de grafos dirigidos Gs si hay mas de una voz
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
	fig=plt.figure(figsize=(16,16))
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
def f_motifs_rhytmic(cancion,length,nombre_parte=None):
	#Toma como input una canción (y el nombre de la parte o voz) y devuelve los motifs
	#ritmicos de tamano length y la frecuencia de aparicion de cada uno.
	#Realiza histograma, utilizando un cierto motif_umbral(empezamos a considerarlo motif
	#a partir de una cierta frecuencia en adelante)
	
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
	plt.figure
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
	plt.show() 
	
	return (motifs_rhytmic,frecuencias)
#---------------------------------------------------------------------------------
def f_motifs_tonal(cancion,length,nombre_parte=None):
	#Toma como input una canción (y el nombre de la parte o voz) y devuelve los motifs
	#tonales de tamano length y la frecuencia de aparicion de cada uno.
	#Realiza histograma, utilizando un cierto motif_umbral(empezamos a considerarlo motif
	#a partir de una cierta frecuencia en adelante)
	
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
	plt.figure
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
	plt.show()       

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

    plt.show()
   
    return(fig)

#-----------------------------------------------------------------------------------
def f_grado_dist_H(G):
    
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
    plt.show()

    return(fig)
#-----------------------------------------------------------------------------------

def f_tabla(G,nombre):
    
    H=G.copy()
    nodos=H.nodes() 
    N=len(nodos)
    
    #calculo los grados que salen y entran de cada nodo
    kgrados_out = [H.out_degree(nodo) for nodo in nodos]
    kgrados_in = [H.in_degree(nodo) for nodo in nodos]

    # El grado medio se puede calcular mediante k_mean=2m/n, donde m es la cantidad de enlaces total y n la cant de nodos
    nodes = H.number_of_nodes()
    enlaces = H.number_of_edges()
    
    if nx.is_directed(H)==False:
        K_mean = round(2*float(enlaces)/float(nodes))
    else:
        K_mean = 'NaN'
    
    if nx.is_directed(H)==True:
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
	#Construye grafo no dirigido, dos notas resultan enlazadas si pertenecen a una armonia, es decir,
	#si hubo algun momento en el que ocurrieron simultaneamente. Además esos enlaces estan pesados.
	#Nota: si dos acordes estan ligados,los cuenta dos veces y no una vez sola.
	
	#Cancion
	song = msc.converter.parse(cancion) # Lee la cancion, queda un elemento stream.Score
	'''
	#--------------------------------------------------------------------------------------------------------
	#Comente esta parte porque cuando entraba dos voces tenian el mismo nombre y no podia elegir la segunda.
	#Ver si se puede arreglar eso porque esta muy piola esto de que te muestre las voces.
	#--------------------------------------------------------------------------------------------------------
	
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
	'''        
	#Instrumento
	part=song.parts[index]
	voz = part.getElementsByClass(msc.stream.Measure)#todos los compases dela parte voz seleccionada
	notas=[]#lista que va a contener cada uno de las notas. Si dos o mas notas so n simultaneas comparten el mismo offset
	tiempos=[]#lista que va a contener a los tiempos de cada una de las notas en la lista notas medidos desde el principio segun la cantidad offset
	frecuencias=[]#lista que va a contener la frecuencia de cada nota en la lista notas para despues ordenar en cada armonia de la mas grave a la mas aguda
	octavas=[]#lista que va a contener las octavas de cada nota

	for c,compas in enumerate(voz):
		#print('compas'+str(c)) #imprimo que compas es
		for i,el in enumerate(compas.flat):
			if isinstance(el,msc.note.Note):#si es una nota
				nota_name=str(el.nameWithOctave)
				notas.append(nota_name)
				tiempo_nota=float(compas.offset+el.offset)
				tiempos.append(tiempo_nota)
				frecuencias.append(el.pitch.frequency)
				octavas.append(el.octave)
			if isinstance(el,msc.chord.Chord):#si es un acorde
				for nc,noteChord in enumerate(el):
					nota_name=str(noteChord.nameWithOctave)
					notas.append(nota_name)
					tiempo_nota=float(compas.offset)+float(el.offset)
					tiempos.append(tiempo_nota)
					frecuencias.append(noteChord.pitch.frequency)
					octavas.append(noteChord.octave)
									
	#Listo: tenemos tres listas: notas,tiempos,frecuencias.
	#Incializamos el grafo: los nodos seran notas. 
	G=nx.Graph()
	
	#Recorremos el vector de tiempos y nos fijamos todas las notas que caen en un mismo tiempo y las guardamos en la lista armonia y esta la guardamos en la lista armonias:
	armonias=[] #lista de armonia
	tiempos_unique=list(np.unique(tiempos))#vector de tiempos unicos donde ninguno se repite:
	armonia=[notas[0]] #al principio armonia tiene la primer nota de la cancion
	tiempo_actual=tiempos[0]
	tonos=[frecuencias[0]]#contendra las frecuencias de las notas en armonia, para ordenarlas de la mas grave a la mas aguda antes de guardar armonia
	octava=[octavas[0]]
	tiempos_armonias=[]
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
			if len(armonia)>=2:#consideramos armonia si son 2 o mas sonidos que suenan simultaneos.
				tiempos_armonias.append(tiempos[t])#es el tiempo en el que ocurrio esa armonia
				armonias.append(armonia_ordenada)#guardamos la armonia ya ordenada solo si la armonia tiene 2 o mas notas.
				#Agrego nodos al grafo:
				for n,nota in enumerate(armonia_ordenada):
					G.add_node(nota)
					G.node[nota]['freq'] = tonos_ordenados[n]
					G.node[nota]['octava'] = octava_ordenada[n]
					G.node[nota]['duracion']=4.0 #por ahora no distinguimos duracion en las armonias
			armonia=[notas[t]]
			tonos=[frecuencias[t]]
			octava=[octavas[t]]
					
	#Agregamos los enlaces al grafo:
	#Los enlaces se haran si dos notas pertenecen a una misma armonia. Los enlaces son no dirigidos.
	
	for a,armonia in enumerate(armonias):
		for n in range (0,len(armonia)):
			if G.has_edge(armonia[-1+n],armonia[n]):
				G[armonia[-1+n]][armonia[n]]['weight']+=1        
			else:
				G.add_edge(armonia[-1+n],armonia[n],weight=1)
				
	return(G)
