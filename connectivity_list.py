#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A Set of functions useful when generating connectivity list section of Gaussian input file
Currently list_to_line asscribes only bond order = 1.0 (which is sufficient for Amber FF)

Created on Wed Oct 14 11:20:42 2020

@author: borowski
"""
import numpy as np

# cov_radii - a dictionary with element symbols as keys and cavalent radii [A] as values
# (up to 86 - Rn) (source: https://en.wikipedia.org/wiki/Covalent_radius)
cov_radii = {\
'H':0.31,'He':0.28,\
'Li':1.28,'Be':0.96,'B':0.84,'C':0.76,'N':0.71,'O':0.66,'F':0.57,'Ne':0.58,\
'Na':1.66,'Mg':1.41,'Al':1.21,'Si':1.11,'P':1.07,'S':1.05,'Cl':1.02,'Ar':1.06,\
'K':2.03,'Ca':1.76,'Sc':1.70,'Ti':1.60,'V':1.53,'Cr':1.39,'Mn':1.61,'Fe':1.52,'Co':1.50,'Ni':1.24,'Cu':1.32,'Zn':1.22,'Ga':1.22,'Ge':1.20,'As':1.19,'Se':1.20,'Br':1.20,'Kr':1.16,\
'Rb':2.20,'Sr':1.95,'Y':1.90,'Zr':1.75,'Nb':1.64,'Mo':1.54,'Tc':1.47,'Ru':1.46,'Rh':1.42,'Pd':1.39,'Ag':1.45,'Cd':1.44,'In':1.42,'Sn':1.39,'Sb':1.39,'Te':1.38,'I':1.39,'Xe':1.40,\
'Cs':2.44,'Ba':2.15,'Hf':1.87,'Ta':1.70,'W':1.62,'Re':1.51,'Os':1.44,'Ir':1.41,'Pt':1.36,'Au':1.36,'Hg':1.32,'Tl':1.45,'Pb':1.46,'Bi':1.48,'Po':1.40,'At':1.50,'Rn':1.50,\
'La':2.07,'Ce':2.04,'Pr':2.03,'Nd':2.01,'Pm':1.99,'Sm':1.98,'Eu':1.98,'Gd':1.96,'Tb':1.94,'Dy':1.92,'Ho':1.92,'Er':1.89,'Tm':1.90,'Yb':1.87,'Lu':1.75} 

#     
def coord_diffs(x1,y1,z1,x2,y2,z2):
    """
    calculates absolute value differences in coordinates

    Parameters
    ----------
    x1 : float
        x coordinate of point 1
    y1 : float
        y coordinate of point 1
    z1 : float
        z coordinate of point 1
    x2 : float
        x coordinate of point 2
    y2 : float
        y coordinate of point 2
    z2 : float
        z coordinate of point 2

    Returns
    -------
    abs(delta_x), abs(delta_y), abs(delta_z)

    """
    dx = np.abs(x1-x2)
    dy = np.abs(y1-y2)
    dz = np.abs(z1-z2)
    return dx, dy, dz


def diff_dist(dx,dy,dz):
    """
    calculates distance from coordinate differences

    Parameters
    ----------
    dx : float
        (x1-x2)
    dy : float
        (y1-y2)
    dz : float
        (z1-z2)

    Returns
    -------
    TYPE : float
        distance between two points

    """
    return np.sqrt(dx**2 + dy**2 + dz**2)


def lista_to_line(ind, lista):
    """
    takes a list lista of indices (zero-based) of atoms bonded to a given atom of
    (zero-based) index ind and returns a string formated as a line of connectivity
    section of Gaussian input file. 
    All bond orders are set as 1.0

    Parameters
    ----------
    ind : (zero-based) index of a given atom
    lista : list of indices of atoms bonded to a given atom

    Returns
    -------
    connect_line - string variable 

    """
    sep = ' '
    b_ord = '1.0'
    connect_line = sep + str(ind + 1) + sep
    for i in range(len(lista)):
        connect_line = connect_line + str(lista[i] + 1) + sep+ b_ord + sep
    connect_line = connect_line + '\n'
    return connect_line


