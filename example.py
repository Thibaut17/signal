# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 13:29:18 2019

@author: Thibaut Jolive PSI
"""
from signal_traitement import traitement as trait, lecture_oscillo
name="example.csv"

u1,u2,t = lecture_oscillo(name)

s=trait(t,u1,u2)
s.trace(xlim=0.005)
s.FFT(f1=s.frequence(1),f2=s.frequence(i=2))
s.passe_bas1()
s.trace(xlim=0.005)
s.FFT(f1=s.frequence(1),f2=s.frequence(i=2))

del u1,u2,t,name
