#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This is a collection of functions and classes to aid
ONIOM input generation from parsed (Amber) prmtop 
and prmcrd (rst7) files.

Created on Mon Apr 19 08:35:15 2021

@author: borowski
"""
import string
import numpy as np
import scipy.spatial

# CONSTANTS
letters = string.ascii_uppercase
digits = string.digits


class atom:
    def __init__(self, coords, index, element):
        self.coords = coords
        self.index = index
        self.element = element
        self.oniom_layer = 'L'
        self.tree_chain_classification = ''
        self.connect_list = []
        self.residue_label = ''
        self.residue_number = 0
        self.in_protein = False
        self.in_mainchain = None
        self.in_sidechain = None
        
    def get_coords(self):
        return self.coords
    def get_index(self):
        return self.index
    def get_element(self):
        return self.element
    def get_oniom_layer(self):
        return self.oniom_layer
    def get_tree_chain_classification(self):
        return self.tree_chain_classification
    def get_connect_list(self):
        return self.connect_list
    def get_residue_label(self):
        return self.residue_label
    def get_residue_number(self):
        return self.residue_number
    def get_in_protein(self):
        return self.in_protein
    def get_in_mainchain(self):
        return self.in_mainchain
    def get_in_sidechain(self):
        return self.in_sidechain    
    
    def set_oniom_layer(self,layer):
        self.oniom_layer = layer
    def set_tree_chain_classification(self,tree_chain_classification):
        self.tree_chain_classification = tree_chain_classification
    def set_connect_list(self,connect_list):
        self.connect_list = connect_list
    def set_residue_label(self,residue_label):
        self.residue_label = residue_label        
    def set_residue_number(self,residue_number):
        self.residue_number = residue_number
    def set_in_protein(self,in_protein):
        self.in_protein = in_protein
    def set_in_mainchain(self,in_mainchain):
        self.in_mainchain = in_mainchain
    def set_in_sidechain(self,in_sidechain):
        self.in_sidechain = in_sidechain

    
class link_atom(atom):
    def __init__(self, coords, index, element, \
                 at_type='', at_charge=0.0, bonded_to=0):
        atom.__init__(self, coords, index, element)
        self.at_type = ''
        self.at_charge = 0.0
        self.bonded_to = None
        
    def get_at_type(self):
        return self.at_type
    def get_at_charge(self):
        return self.at_charge
    def get_bonded_to(self):
        return self.bonded_to
    
    def set_at_type(self,at_type):
        self.at_type = at_type
    def set_at_charge(self,at_charge):
        self.at_charge = at_charge
    def set_bonded_to(self,bonded_to):
        self.bonded_to = bonded_to
        
        
class residue:
    def __init__(self, label, index):
        self.label = label
        self.index = index
        self.in_protein = False
        self.atoms = []
        self.main_chain_atoms = []

    def get_label(self):
        return self.label        
    def get_index(self):
        return self.index        
    def get_in_protein(self):
        return self.in_protein
    def get_atoms(self):
        return self.atoms
    def get_main_chain_atoms(self):
        return self.main_chain_atoms
    def next_atom(self):
        for atom in self.atoms:
            yield atom

    def set_in_protein(self,in_protein):
        self.in_protein = in_protein
    def add_atom(self,atom):
        self.atoms.append(atom)
        if atom.get_tree_chain_classification() == 'M':
            self.main_chain_atoms.append(atom)

def NC_in_main_chain(residue):
    """ checks is the residue contains N(M) and C(M) atoms
    if yes, returns True, otherwise returns False"""
    temp = []
    for atom in residue.get_main_chain_atoms():
        temp.append(atom.get_element())
    main_chain_elements = set(temp)
    if 'N' in main_chain_elements and 'C' in main_chain_elements:
        return True
    else:
        return False

def is_peptide_bond(residue_1,residue_2):
    """looks for O-C(M)-N(M) fragment between the two residues
    returns True if found, False otherwise"""
    for atom1 in residue_1.get_main_chain_atoms():
        if atom1.get_element() == 'C':
           for atom2 in residue_2.get_main_chain_atoms():
               if atom2.get_element() == 'N':
                   for atom3 in residue_1.get_atoms():
                       if atom3.get_element() == 'O' and atom3.get_tree_chain_classification() == 'E'\
                       and atom3.get_index() in atom1.get_connect_list() :
                                 return True
    return False
    

 
class chain:
    def __init__(self, label, index):
        self.label = label
        self.index = index       
        self.is_peptide = False
        self.residues = []

    def get_label(self):
        return self.label        
    def get_index(self):
        return self.index 
    def get_is_peptide(self):
        return self.is_peptide
    def get_residues(self):
        return self.residues
    def get_last_residue(self):
        return self.residues[-1]
    def next_residue(self):
        for residue in self.residues:
            yield residue    
                       

    def add_residue(self,residue):
        """if chain is empty, the residue to be added needs to contain N(M) and C(M) atoms
        if chain already includes residue(s), the residue to be added needs the above and 
        to make a paptide bond with the last residue in the chain"""
        if NC_in_main_chain(residue):
            if len(self.residues) == 0:        
                self.residues.append(residue)                   
            elif is_peptide_bond(self.get_last_residue(),residue):
                self.residues.append(residue)
                self.is_peptide = True
                
                           
def generate_label():
    prefix = ''
    i = 0
    while True:
        j = int(i/26)
        k = i%26
        if k == 0 and j > 0:
            prefix = prefix + letters[j-1]
        label = prefix + letters[k]
        i += 1
        yield label
        
        
def residue_within_r_from_atom(residue, atom, r):
    """checks if any atom of residue lies within r (d(at@res --- atom) <= r)
    of atom
    if yes, returns True, if no, returns False"""
    ref_coords = np.array(atom.get_coords(),dtype=float)
    for atom_from_resid in residue.get_atoms():
        coords = np.array(atom_from_resid.get_coords(),dtype=float)
        dist = scipy.spatial.distance.cdist(ref_coords,coords)
        if dist <= r:
            return True
    return False

def residues_within_r_from_atom(residue_list, atom, r):
    """checks if any atom of residue from a residue_list 
    lies within r (d(at@res --- atom) <= r) of atom
    if yes, returns a list with True value at corresponding index
    """
    result = []
    ref_coords = np.array(atom.get_coords(),dtype=float)
    for residue in residue_list:
        coords = []
        for atom_from_resid in residue.get_atoms():
            coords.append(atom_from_resid.get_coords())
        coords_arr = np.array(coords,dtype=float)
        dist = scipy.spatial.distance.cdist(ref_coords,coords_arr)
        if np.amin(dist) <= r:
            result.append(True)
        else:
            result.append(False)
    return result

def residues_within_r_from_atom_list(residue_list, atom_list, r):
    """ 
    checks if any atom of residue from a residue_list 
    lies within r (d(at@res --- atom) <= r) of any atom from atom_list;
    returns a boolean list of length len(residue_list)
    """
    res_list_length = len(residue_list)
    result = [False for i in range(res_list_length)]
    for atom_ref in atom_list:
        index_table = []
        temp_residue_list = []
        for i in range(res_list_length):
            if result[i] == False:
                temp_residue_list.append(residue_list[i])
                index_table.append(i)
        temp_result = residues_within_r_from_atom(temp_residue_list, atom_ref, r)
        for i in range(len(temp_residue_list)):
            result[index_table[i]] = (result[index_table[i]] or temp_result[i])
    return result
                

