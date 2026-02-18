#Script "data treatment for piezo data"
#FA
#December 2025

from tkinter import *
from tkinter import ttk
from tkinter import filedialog
import pandas as pd
import numpy as np
import os
import csv

#pour tous les fichiers du dossier

#choisir le dossier contenant les fichiers .csv
def Choosefolder():
    directory = filedialog.askdirectory()
    directoryname = os.path.basename(str(directory))
    add_item(headerlist, directoryname)
    for root,dirs,files in os.walk(directory):
        for file in files:
            if file.endswith(".csv"):
                Importfile(file)
                Extractdata(directory,file)
                Periodtime(time)
                ExtractP(pressure)
                ExtractV(voltage)
                time.clear()
                pressure.clear()
                voltage.clear()
                perioditem.clear()
                period.clear()
    while len(headerlist) < len(Rlist):
        a=""
        add_item(headerlist, a)
    Canevas.create_rectangle(L/10,H/5,5*L/10,2*H/5,fill="green")
    Canevas.create_text(x+50,y+60,text="imported",fill="green")
    Canevas.create_text(x+50,y+80,text="calculated",fill="green")

#sortir les valeurs pour tous les fichiers
def Importfile(file):
    filename = os.path.basename(str(file))
    filename = filename.replace("' mode='r' encoding='UTF-8'>", "")
    frequency = filename.split('_')[1]
    frequency = frequency.replace("Hz","")
    resistance1 = filename.split('_')[2]
    resistance = resistance1.split('.')[0]
    resistance = resistance.replace("Ohms","")
    add_item(Rlist, resistance)
    add_item(freqlist, frequency)
    tp = 1/float(frequency)
    perioditem.append(tp)

#extraire données sous forme {[t], [P], [V]}
def Extractdata(directory,file):
    data=pd.read_csv(os.path.join(directory,file))
    Tiempo=data['time'].tolist()
    Presion=data['pressure'].tolist()
    Voltaje=data['voltage'].tolist()
    for i in Tiempo:
        time.append(i)
    for j in Presion:
        pressure.append(j)
    for k in Voltaje:
        voltage.append(k)

#découper une période
def Periodtime(list):
    tp = perioditem[0]
    for t in list:
        if t <= tp:
            period.append(t)

#Extraire la valeur d'amplitude moyenne pour P
def ExtractP(list):
    periodslice=[]
    ampli=[]
    n0=len(period)
    n=len(list)
    r=int(n/n0)
    for i in range(r):
        if i==0:
            ni=[]
        else:
            if i==1:
                ni=list[:n0]
            else:
                t1=i*n0
                t0=(i-1)*n0
                ni=list[t0:t1]
            periodslice.append(ni)
        
    for j in periodslice:
        A=max(j)-min(j)
        ampli.append(A)
        
    ampliP=sum(ampli)/len(ampli)
    add_item(Plist, ampliP)
    perioslice=[]
    ampli=[]

#Extraire la valeur d'amplitude moyenne pour V
def ExtractV(list):
    periodslice=[]
    ampli=[]
    n0=len(period)
    n=len(list)
    r=int(n/n0)
    for i in range(r):
        if i==0:
            ni=[]
        else:
            if i==1:
                ni=list[:n0]
            else:
                t1=i*n0
                t0=(i-1)*n0
                ni=list[t0:t1]
            periodslice.append(ni)
        
    for j in periodslice:
        A=max(j)-min(j)
        ampli.append(A)
        
    ampliV=sum(ampli)/len(ampli)
    add_item(Vlist, ampliV)
    perioslice=[]
    ampli=[]

#ajouter un couple {R,freq} à la liste
def add_item(list,item):
    list.append(item)
    return list

#exporter les données au format .csv
def Savelist():
    Datalists={'Sample': headerlist, 'Resistance (Ohms)': Rlist, 'Frequency (Hz)': freqlist, 'Amplitude P (bar)': Plist, 'Amplitude V (V)': Vlist}
    df = pd.DataFrame(Datalists)
    
    print(df)
    listname=filedialog.asksaveasfilename()
    with open(listname, 'w', newline='') as csvfile:
        df.to_csv(csvfile, index = False)
    Canevas.create_rectangle(L/10,H/5,9*L/10,2*H/5,fill="green")
    Canevas.create_text(x+50,y+120,text="saved",fill="green")

#nettoyer toutes les données pour recommencer l'opération
def Clear():
    time.clear()
    pressure.clear()
    voltage.clear()
    perioditem.clear()
    period.clear()
    
    headerlist.clear()
    Rlist.clear()
    freqlist.clear()
    Plist.clear()
    Vlist.clear()

    Canevas.create_rectangle(L/10,H/5,9*L/10,2*H/5,fill="white",outline="black")
    Canevas.create_rectangle(x+24,y+50,x+90,y+150,fill="white",outline="white")
    
#création du fichier (R, freq)
headerlist=[]
Rlist=[]
freqlist=[]
Plist=[]
Vlist=[]

#création des listes
time=[]
pressure=[]
voltage=[]
perioditem=[]
period=[]

#création de la fenêtre principale
gifdict={}

Mafenetre = Tk()
Mafenetre.title("Data treatment from piezo files")

#création du widget Canvas
L = 400
H = 150
l=150
x=100
y=20
Canevas=Canvas(Mafenetre, width = L, height = H, bg="white")
Canevas.pack(padx =5, pady =5)

Canevas.create_rectangle(L/10,H/5,9*L/10,2*H/5,fill="white",outline="black")

Canevas.create_text(x,y,text="Data treatment")
Canevas.create_text(x,y+60,text="{R, freq}:")
Canevas.create_text(x,y+80,text="{P, V}:")


#création du widget Bouton 'importer'
Button(Mafenetre, text ='Import .csv files and calculate amplitudes',font=(10), command = Choosefolder).pack(side=LEFT,padx=5,pady=5)

#création du widget Bouton 'exporter'
Button(Mafenetre, text ='Export',font=(10), command = Savelist).pack(side=LEFT,padx=5,pady=5)

#création du widget Bouton 'nettoyer'
Button(Mafenetre, text ='Clear',font=(10), command = Clear).pack(side=LEFT,padx=5,pady=5)

#création du widget Bouton 'quitter'
Button(Mafenetre, text ='Exit',font=(10), command = Mafenetre.destroy).pack(side=RIGHT,padx=5,pady=5)

Mafenetre.mainloop()
