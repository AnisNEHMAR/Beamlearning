#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

rnd = np.random.default_rng()

class Antenne:
    """
    Creation d'une antenne lineaire uniforme
    L: longueur d'antenne
    M: nombre de microphones
    d: distance entre chaque deux microphones 
    2D_coordinates: coordonnées des microphones dans un repére 2D 
    indices: indice(int) des microphones de -(M-1)/2 jusqu'a (M-1)/2
    """
    def __init__(self,L:float,M:int):
        self.M = M
        self.L = L
        self.coordonnees_2D, self.d, self.indices = self.creer_antenne_uni_lin(L,M)


    def creer_antenne_uni_lin(self,L:float,M:int): 

        # M doit etre un nombre impair pour que l'origine soit représenté par le microphone qui se trouve au centre d'antenne
        msg = "Nombre de microphones doit etre impair"
        assert M%2, msg
        
        d = L/(M-1)
        indices = np.linspace(-(M//2),M//2,M,dtype=int)
        coordonnees_2D = np.zeros((2,M))
        coordonnees_2D[0,:]= np.linspace(-L/2,L/2,M)
        return coordonnees_2D,d,indices

def retarder_signal(X,t0,fs):
    """
    introduire un retard -t0- au signal temporelle - X -
    fs : fréquence échantillonnage du signal temporelle t  
    """ 

    Xf = np.fft.rfft(X)
    N_dft = X.shape[0]
    df = fs/N_dft
    f = np.linspace(0, fs-df, N_dft)[:N_dft//2+1]
    
    Xf_retarde = Xf*np.exp(-1j*2*np.pi*f*t0)
    
    return np.fft.irfft(Xf_retarde)

    
def signal_genere(A, T, fs, f0): # fonction qui gener un sinus multiplié par la fenetre de hanning
        """
        creer un signal sinusoidale modulé par fenetre de hann 
        A: amplitude 
        T : duree (s)
        f0: fréquence central
        fs: fréquence échantillonnage
    """
        N = int(T*fs)

        t = np.linspace(0, T-1/fs, N)
        signal = A*np.sin(2*np.pi*f0*t) * np.hanning(N)
        
        return signal

    
def creer_signals_antenne(antenne_Obj, p_source, t_initial, T, theta0_deg,
                        fs, c0=340, SNR_dB=None):

        # on assume que notre onde sonore est plane à son arrivée à notre antenne
        # donc notre source sonore doit etre sufisament lointaine

    
    rnd = np.random.default_rng()
      
    # direction d'arrivée de signal sonore (propagation des ondes planes)
    theta0 = theta0_deg*np.pi/180
        
    #  vecteur des temps d'arrivée pour chaque capteur (microphone)
    temps_darrivee = -antenne_Obj.indices*antenne_Obj.d*np.cos(theta0)/c0
    
        
    N_initial= int(t_initial*fs)
    N_final = N_initial + p_source.shape[0]
       
    # Nombre d'echantillons (la durée)
    N = int(T*fs)
        
    # creer les signaux de sortie de chauqe micro  
    if SNR_dB is None:
        # pas de bruit
        p_array = np.zeros((antenne_Obj.M, N))
        
    
    else:
        # si SNR_dB est donné, ajouter un bruit aléatoire aux signaux du tableau au SNR désiré
        signal_var = np.var(p_source)
        bruit_var = signal_var/(10**(SNR_dB/10))
        p_array = rnd.normal(0., np.sqrt(bruit_var), (antenne_Obj.M, N))
        
    # pour chaque microphone d'antenne...
    for m in range(antenne_Obj.M):
        # ... déphasage de signal
        
        p_array[m, N_initial:N_final] += p_source
            
        
        p_array[m, :] = retarder_signal(p_array[m, :], temps_darrivee[m], fs)
        
    return p_array

def beamforming(Antenne_Obj, p_array, theta, weights, fs,c0=340):
    """
    compenser le déphasage de signal de sortie de chaque microphone par les retards calculé selon un angle de polarisation,
    faire la somme de ses signaux,
    si on est dans la bonne direction, la compensation des déphasage de différentes signaux est constructive,
    et l'energie sera maximale 
   """
    
    N_theta = theta.shape[0]
    
    M, N_time = p_array.shape
    
    
    y_beamformer = np.zeros((N_theta, N_time))

    
    for theta_i in range(N_theta):
        
        # calculer les délais(retard pour chaque signal (sortie de microphone))
        time_delays = -Antenne_Obj.indices*Antenne_Obj.d*np.cos(theta[theta_i])/c0
    
        # et pour chaque microphone..
        for m in range(M):
            # signal de sortie des antennes
            y_beamformer[theta_i, :] += weights[m]*retarder_signal(p_array[m, :], -time_delays[m], fs)
    
    
    y_beamformer *= 1./np.sum(weights**2)

    #calcul d'enregie
    y_beamf_polar = np.sum(y_beamformer**2,axis=1)
    y_beamf_polar = 10*np.log10(y_beamf_polar)

    #angle ou l'energie est max
    dB_max = y_beamf_polar.max()
    angle = np.where(y_beamf_polar == dB_max)
    angle = angle[0][0]
    
    return y_beamformer,y_beamf_polar,angle
