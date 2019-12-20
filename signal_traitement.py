# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 13:06:15 2019

@author: Thibaut Jolive PSI
"""
import numpy as np
import matplotlib.pyplot as plt
import random as rd

"""
Module permettant de faire de l'analyse de signaux de manière très simple
Attention Programmation objet donc utilisation de méthodes et non de fonctions!!!!!
"""
################
#Lecture des fichiers généré par les oscilloscopes Keysight InfiniiVision 2000 X-Series Oscilloscopes du lycée
################
def recherche(a,L):
	b=L[0]
	j=0
	for i in range(len(L)):
		if L[i]>b:
			b=L[i]
			j=i
	return j
def lecture_oscillo(name):
	"""
	Renvoi la tension en colonne 1 et 2 et le temps des fichiers générés
	par les oscilloscopes Keysight InfiniiVision 2000 X-Series Oscilloscopes
	"""
	file = open(name)
	contenu = file.readlines()
	t=[]
	u1=[]
	u2=[]
	for j in range(len(contenu)):
		l=contenu[j]
		contenu[j]=l.split(',')
	i=3
	
	while len(contenu[i])==3 and i<len(contenu)-1:
		l=contenu[i]
		l[2]=l[2].replace('\n','')
		l[2]=l[2].replace('E','e')
		l[1]=l[1].replace('E','e')
		l[0]=l[0].replace('E','e')
		if l[1]== '':
			l[1]=l[1].replace('','0')
		if l[2]== '':
			l[2]=l[2].replace('','0')
		t.append(float(l[0]))
		u1.append(float(l[1]))
		u2.append(float(l[2]))
		i+=1
	return u1,u2,t

class generation:
	def __init__(self, duree, Te, t=None):
		"""
		Permet de générer un signal de tout type
		duree : float : duree du signal
		Te : periode echantillonage
		"""
		if t != None:
			self.temps=np.array(t)
			self.duree=t[-1]-t[0]
			self.Te=t[1]-t[0]
			self.signal=0*self.temps
		else:
			self.temps=np.arange(0,duree,Te)
			self.duree=duree
			self.Te=Te
			self.signal=0*self.temps
		return None
	
	def impulsion(self,A=1):
		"""
		Génère une impulsion d'amplitude A
		A : float : amplitude
		"""
		s=0*self.temps
		s[0]=A
		self.signal=s
		return None
	
	def sinus(self,A=1,f=1,phi=0):
		"""
		Génère un signal sinusoïdale de fréquence f d'amplitude A et de phase phi
		A : float : amplitude
		f : float : fréquence
		phi : float : phase à l'origine pi/2 pour générer un cosinus
		"""
		t=self.temps
		self.signal=A*np.sin(2*np.pi*f*t+phi)
		return None
	def creneau(self,A=1,f=1,c=0.5):
		"""
		Genere un signal creneau d'amplitude A, de frequence f et de rapport cyclique c
		A : float : amplitude
		f : float : fréquence
		c : float : rapport cyclique
		"""
		t=self.temps
		x=[]
		for n in t:
			if f*n-int(f*n) < c:
				x.append(A)
			else:
				x.append(-A)
		self.signal=np.array(x)
		return None
	
	def triangle(self,f=1,A=1):
		"""
		Génère un signal triange de fréquence f et d'amplitude A
		A : float : amplitude
		f : float : fréquence
		"""
		t=self.temps
		s= [A*np.arcsin(np.sin(2*np.pi*z*f)) for z in t]
		self.signal=np.array(s)
		return None
	
	def bruit(self,A=1):
		"""
		Renvoi un bruit d'amplitude A
		A : float : amplitude
		"""
		t=self.temps
		x=[rd.uniform(-A,A) for i in t]
		self.signal=np.array(x)
		return None
	
	def renvoi(self):
		"""
		Renvoi le temps ainsi que la liste des valeurs du signal qui lui est associée
		"""
		return self.temps,self.signal

class traitement:
	
	def __init__(self, temps, signal1, signal2=None, number=0):
		self.signal1=np.array(signal1)
		if signal2 != None:
			self.signal2=np.array(signal2)
			self.control = 1
		else:
			self.signal2=None
			self.control = 0
		self.duree=temps[-1]-temps[0]
		self.Te=temps[1]-temps[0]
		self.n=len(temps)
		self.temps=np.array(temps)
		self.number=number
		return None
################
#Recupération des signaux
################
	def s_t(self,t,i=1):
		if i == 1:
			s=self.signal1
		else :
			s=self.signal2
		t=recherche(t,self.temps)
		return s[t]
	def recupere(self,i=1):
		"""
		Renvoi des listes correspondants aux valeurs du signal 1 ou 2 en fonction de celui voulu
		ainsi que la liste des temps
		i : 1 ou 2 : permet de choisir si on travail avec le signal 1 ou le signal 2
		"""
		if i==1:
			return (self.signal1,self.t)
		else:
			return (self.signal2,self.t)
	
	def show(self,i=1):
		"""
		Renvoi des listes correspondants aux valeurs du signal 1 ou 2 en fonction de celui voulu
		i : 1 ou 2 : permet de choisir si on travail avec le signal 1 ou le signal 2
		"""
		if i==1:
			return self.signal1
		else:
			return self.signal2
################
#Opérations simple et calculs sur un signal
################
	def supp_moye(self,i=1):
		"""
		Supprime la valeur moyenne du signal en fonction de celui que l'on souhaite
		i : 1 ou 2 : permet de choisir si on travail avec le signal 1 ou le signal 2
		"""
		if i ==1:
			sign=self.signal1
		else :
			sign=self.signal2
		moy=np.mean(sign)
		for i in range(len(sign)):
			sign[i]=sign[i]-moy
		return None
	
	def moye(self,i=1):
		"""
        Donne la valeur moyenne
		i : 1 ou 2 : permet de choisir si on travail avec le signal 1 ou le signal 2
		"""
		if i ==1:
			sign=self.signal1
		else :
			sign=self.signal2
		moy=np.mean(sign)
		return moy
    
	def amplitude(self,i=1):
		"""
		Donne l'amplitude de la liste, dans le cas d'un signal cette mesure n'est valable
		uniquement si il est purement harmonique
		"""
		if i ==1:
			L=self.signal1
		else :
			L=self.signal2
		mini=min(L)
		maxi=max(L)
		return (maxi-mini)/2

	def frequence(self,i=1):
		"""
		Donne la fréquence de la liste, dans le cas d'un signal cette mesure n'est valable
		uniquement si il est purement harmonique
		"""
		if i ==1:
			L=self.signal1
		else :
			L=self.signal2
		nb_zero=0
		t=self.temps
		indice=[]
		for p in range(len(L)-1):
			if L[p]*L[p+1]<0:
				nb_zero +=1
				indice.append(p)
		nb_zero=nb_zero//2
		return round(nb_zero/(t[indice[-1]]-t[indice[0]]),2)

	def dephasage(self):
		"""
		Mesure le déphasage entre deux liste, Dans le cadre d'un signal, mesure le
		déphasage entre X et Y par rapport à t
		renvoi le déphasage en radian
		"""
		X=self.signal1
		Y=self.signal2
		t=self.temps
		i=0
		while i<len(X)-2 and X[i]*X[i+1]>0:
			i=i+1
		j=0
		while j<len(X)-2 and Y[j]*Y[j+1]>0:
			j+=1
		delta_t=t[j]-t[i]
		phi = round(2*delta_t*np.pi,3)
		if phi>0:
			phi='+'+str(phi)
		elif phi==0:
			phi=''
		return phi
################
#Analyse spectrale
################
	def FFT(self,titre="Spectre frequence signal n°0",name="spectre_fft.png",f1=None,f2=None):
		"""
		Trace le spectre en amplitude du signal s(t) par l'algorithme de FFT et l'enregistre \n
		name : nom du fichier enregistré \n
		f1,f2 : permet de centrer sur la fréquence f1 ou f2 \n
		"""
		tp=self.temps
		u1p=self.signal1
		u2p=self.signal2
		number=self.number
		# Number of sample points
		N = len(tp)
		# sample spacing
		T = abs(tp[2]-tp[1])
		#x = np.linspace(0.0, N*T, N)
		#y = A1*np.cos(f2 * 2.0*np.pi*x)
		yf = np.fft.fft(u1p)
		xf = np.linspace(0.0, 1.0/(2.0*T), N//2)
		plt.figure()
		plt.title(titre)
		plt.xlabel('f')
		plt.ylabel('A')
		plt.xscale('log')
		if self.control==1:
			yf2=np.fft.fft(u2p)
			plt.plot(xf, 2.0/N * np.abs(yf[0:N//2]),xf, 2.0/N * np.abs(yf2[0:N//2]))
			if f1 != None or f2 != None:
				plt.xlim(min(f1/100,f2/100),max(f1*100,f2*100))
			plt.legend(['$s_{1}(t)$','$s_{2}(t)$'])
		else :
			plt.plot(xf, 2.0/N * np.abs(yf[0:N//2]))
			if f1 != None:
				plt.xlim(f1/100,f1*100)
			plt.legend(['$s(t)$'])
		plt.grid()
		plt.savefig(name)
		plt.show()
		
		return None
################
#Tracer des signaux
################	
	def trace(self,titre="Signal n°0",name="signal.png",t="$t$",z="$s(t)$",xlim=None,ylim=None):
		"""
		Trace le signal s(t) et l'enregistre
		name : nom du fichier enregistré
		t : nom porté par l'axe des abscisses
		z : nom porté par l'axe des ordonnées
		xlim : valeur limite en abscisses permet de tracer le signal sur l'intervalle [-xlim,xlim]
		ylim : valeur limite en ordonnées permet de visualiser les valeur du signal entre [-ylim,ylim]
		"""
		x=self.temps
		y1=self.signal1
		y2=self.signal2
		number=self.number
		plt.figure()
		plt.title(titre)
		plt.xlabel(t)
		plt.ylabel(z)
		if self.control==1:
			if ylim!= None:
				plt.ylim(-ylim,ylim)
			if xlim!= None:
				plt.xlim(-xlim,xlim)
			plt.plot(x,y1,x,y2)
			plt.legend(['$s_{1}(t)$','$s_{2}(t)$'])
		else:
			if ylim!= None:
				plt.ylim(-ylim,ylim)
			if xlim!= None:
				plt.xlim(-xlim,xlim)
			plt.plot(x,y1)
			plt.legend(['$s(t)$'])
		plt.grid()
		plt.savefig(name)
		plt.show()
		return None
#######
#Opérations sur des signaux
#######
	def somme(self):
		"""
		Somme les signaux 1 et 2
		renvoi la somme des signaux ainsi que l'échelle de temps
		"""
		s1=self.signal1
		s2=self.signal2
		return (s1+s2,self.temps)
	def multiplication(self,z=0):
		"""
		Multipilie le signal 1 et 2
		renvoi la multiplication des signaux ainsi que l'échelle de temps
		"""
		s1=self.signal1
		s2=self.signal2
		return (s1*s2+z,self.temps)
	def soustraction(self):
		"""
		Soustrait le signal 2 au signal 1
		renvoi la soustraction des signaux ainsi que l'échelle de temps
		"""
		s1=self.signal1
		s2=self.signal2
		return (s1-s2,self.temps)

#######
#Filtrage
#######		
	def filtre_passe_haut_1(self,G=1,fc=1,i=1):
	"""
	Réalise un filtrage linéaire passe-haut d'ordre 1 semblbable à un filtre analogique
	G : int : Gain du filtre
	fc : float ou int : fréquence de coupure du filtre
	i : int 1 ou 2 : permet de choisir de travailler avec le signal 1 ou le signal 2
	Attention le signal de base est détruit après opération de filtrage
	"""
	if i==1:
		e=self.signal1
	else:
		e=self.signal2
	te=self.Te
	tau=1/(2*fc*np.pi)
	s=[0]
	for n in range(len(e)-1):
		s.append((1-te/tau)*s[n]+e[n]-e[n+1])
	if i ==1:
		self.signal1=np.array(s)
	else:
		self.signal2=np.array(s)
	return None

	def passe_bas1(self,G=1,fc=1,i=1):
		"""
		Réalise un filtrage linéaire passe-bas d'ordre 1 semblbable à un filtre analogique
		G : int : Gain du filtre
		fc : float ou int : fréquence de coupure du filtre
		i : int 1 ou 2 : permet de choisir de travailler avec le signal 1 ou le signal 2
		Attention le signal de base est détruit après opération de filtrage
		"""
		if i==1:
			e=self.signal1
		else:
			e=self.signal2
		te=self.Te
		tau=1/(2*fc*np.pi)
		s=[0]
		for n in range(len(e) -1):
			s.append(s[n]+te/tau*(G*e[n]-s[n]))
		if i ==1:
			self.signal1=np.array(s)
		else:
			self.signal2=np.array(s)
		return None

	def integrateur(self,G=1,i=1):
		"""
		Réalise un filtre intégrateur
		G : int : Gain du filtre
		i : int 1 ou 2 : permet de choisir de travailler avec le signal 1 ou le signal 2
		Attention le signal de base est détruit après opération de filtrage
		"""
		if i ==1:
			e=self.signal1
		else:
			e=self.signal2
		s=[]
		so=0
		for i in e:
			so+=i
			s.append(so)
		if i ==1:
			self.signal1=G*np.array(s)
		else:
			self.signal2=G*np.array(s)
		return None
	
	def derivateur(self,G=1,i=1):
		"""
		Réalise un filtre dérivateur
		G : int : Gain du filtre
		i : int 1 ou 2 : permet de choisir de travailler avec le signal 1 ou le signal 2
		Attention le signal de base est détruit après opération de filtrage
		"""
		if i ==1:
			e=self.signal1
		else:
			e=self.signal2
		s=[e[0]]
		n=self.n
		for i in range(n):
			s.append(e[i]-e[i-1])
		if i ==1:
			self.signal1=G*np.array(s)
		else:
			self.signal2=G*np.array(s)
		return None
	def amplificateur(self,G=1,i=1):
		"""
		Réalise un filtre amplificateur
		G : int : Gain du filtre
		i : int 1 ou 2 : permet de choisir de travailler avec le signal 1 ou le signal 2
		Attention le signal de base est détruit après opération de filtrage
		"""
		if i==1:
			self.signal1=G*self.signal1
		else:
			self.signal2=G*self.signal2
		return None
	def decalage(self,A=0,i=1):
		"""
		Ajoute A en composante continue
		A : int : Composante continue
		i : int 1 ou 2 : permet de choisir de travailler avec le signal 1 ou le signal 2
		Attention le signal de base est détruit après cette opération
		"""
		if i==1:
			self.signal1=A+self.signal1
		else:
			self.signal2=A+self.signal2
		return None
