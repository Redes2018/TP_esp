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
	# Toma como input una canción (y el nombre de la parte o voz) y devuelve un grafo G
	
	# Cancion
	song = msc.converter.parse(cancion) # Lee la cancion, queda un elemento stream.Score
	
	Lp = len(song.parts) # Cantidad de partes (voces)
	lista_partes = list(np.zeros(Lp)) # Crea una lista donde se van a guardas los nombres de las partes
	
	for i,elem in enumerate(song.parts):
		lista_partes[i] = elem.partName # Guarda los nombres de las partes en la lista
	
	nombre_parte = nombre_parte or lista_partes[0] # Si no tuvo nombre_parte como input,toma la primera voz
	
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
		for i,el in enumerate(notas):
			if isinstance(el,msc.note.Note):
				nota_name = str(el.quarterLength)
				if not G.has_node(nota_name):
					G.add_node(nota_name)
					d = el.quarterLength
					G.node[nota_name]['freq'] = 2**(d/(2*np.pi))*20
					G.node[nota_name]['octava'] = 2
					G.node[nota_name]['duracion'] = d
				notas[i] = nota_name

			elif isinstance(el,msc.note.Rest):
				nota_name = str(el.name)+'/'+str(el.quarterLength)
				if not G.has_node(nota_name):
					G.add_node(nota_name)
					d = el.quarterLength
					G.node[nota_name]['freq'] = 2**(d/(2*np.pi))*20
					G.node[nota_name]['octava'] = 1
					G.node[nota_name]['duracion'] = d
				notas[i] = nota_name
	
	# Enlaces pesados
	for i in range(L-1):  # recorremos desde la primera hasta la anteultima nota, uniendo sucesivas
		if G.has_edge(notas[i],notas[i+1]):
			G[notas[i]][notas[i+1]]['weight']+=1 # si el enlace ya existe, se le agrega +1 al peso
		else:
			G.add_edge(notas[i],notas[i+1],weight=1) # si el enlace no existe, se crea con peso 1
	
	return(G)

#-----------------------------------------------------------------------------------

def graficar(G, color_map='rainbow',layout='espiral'):
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
	
	octavas = np.array(list(nx.get_node_attributes(G,'octava').values()))
	oct_min = min(octavas)
	oct_max = max(octavas)
	colores_oct_nro = (octavas - oct_min)/(oct_max - oct_min)
	m = cm.ScalarMappable(norm=None, cmap=color_map)
	colores_oct = m.to_rgba(colores_oct_nro)
	
	nx.draw_networkx_nodes(G,pos,node_list=nodos,node_color=colores_oct,node_size=800,alpha=1)
	nx.draw_networkx_labels(G,pos)
	edges = nx.draw_networkx_edges(G,pos,width=3)
	weights = list(nx.get_edge_attributes(G,'weight').values())
	weight_max = max(weights)
	for i in range(M):
			edges[i].set_alpha((weights[i]/weight_max)**(1./2.)) # valores de alpha para cada enlace
	plt.axis('off')
	plt.show()

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
    plt.figure(figsize=(16,8))
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
   
    return(k_in, k_out, pk_in, pk_out, N)

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

    
    plt.figure(figsize=(8,8))
    
    plt.plot(bin_centros,pk_logbin,'bo')
    plt.xlabel('$log(k)$',fontsize=20)
    plt.xscale('log')
    plt.ylabel('$log(p_{k})$',fontsize=20)
    plt.yscale('log')
    plt.title('Bin log - Escala log',fontsize=20)
    plt.show()

    return(k, pk, N)
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
