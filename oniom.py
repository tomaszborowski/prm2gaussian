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
import re

# CONSTANTS
letters = string.ascii_uppercase
digits = string.digits


class atom:
    def __init__(self, coords, index, element):
        self.coords = coords
        self.index = index
        self.new_index = None
        self.element = element
        self.at_type = None
        self.new_at_type = None
        self.at_charge = 0.0
        self.oniom_layer = 'L'
        self.tree_chain_classification = ''
        self.connect_list = []
        self.in_mainchain = False
        self.frozen = 0     # 0 - not frozen, -1 - frozen
        
    def get_coords(self):
        return self.coords
    def get_index(self):
        return self.index
    def get_new_index(self):
        return self.new_index
    def get_element(self):
        return self.element
    def get_type(self):
        return self.at_type
    def get_new_type(self):
        return self.new_at_type
    def get_at_charge(self):
        return self.at_charge
    def get_oniom_layer(self):
        return self.oniom_layer
    def get_tree_chain_classification(self):
        return self.tree_chain_classification
    def get_connect_list(self):
        return self.connect_list
    def get_in_mainchain(self):
        return self.in_mainchain
    def get_frozen(self):
        return self.frozen
    
 
    def set_type(self,at_type):
        self.at_type = at_type
    def set_new_type(self,new_at_type):
        self.new_at_type = new_at_type
    def set_at_charge(self,at_charge):
        self.at_charge = at_charge    
    def set_oniom_layer(self,layer):
        self.oniom_layer = layer
    def set_tree_chain_classification(self,tree_chain_classification):
        self.tree_chain_classification = tree_chain_classification
    def set_connect_list(self,connect_list):
        self.connect_list = connect_list
    def set_in_mainchain(self,in_mainchain):
        self.in_mainchain = in_mainchain
    def set_frozen(self,frozen):
        self.frozen = frozen
    def set_new_index(self,new_index):
        self.new_index = new_index
        
        
class link_atom(atom):
    def __init__(self, coords, index, element, \
                 at_type='', at_charge=0.0, bonded_to=None):
        atom.__init__(self, coords, index, element)
        self.at_type = ''
        self.bonded_to = None
        self.H_coords = []
        self.mm_element = ''
        
    def get_type(self):
        return self.at_type
    def get_bonded_to(self):
        return self.bonded_to    
    def get_MM_coords(self):
         return self.coords 
    def get_coords(self):
        return self.H_coords    
    def get_mm_element(self):
        return self.mm_element
    
    def set_type(self,at_type):
        self.at_type = at_type
    def set_bonded_to(self,bonded_to):
        self.bonded_to = bonded_to
    def set_H_coords(self,H_coords):
        self.H_coords = H_coords
    def set_mm_element(self,mm_element):
        self.mm_element = mm_element

        
def atom_to_link_atom(at_ins, amb_type, chrg = 0.000001):
    """
    Based on atom object instance at_ins 
    generate and return a hydrogen link_atom_object
    ----
    at_ins : instance of atom object
    amb_type : string, amber atom type for H-link atom
    chrg : float, atomic charge to be ascribed to H-link atom
    """
    new_link_atom = link_atom( at_ins.get_coords(), at_ins.get_index(), 'H' )
    new_link_atom.set_connect_list( at_ins.get_connect_list() )
    new_link_atom.set_oniom_layer('H')
    new_link_atom.set_type(amb_type)
    new_link_atom.set_at_charge(chrg)
    new_link_atom.set_mm_element( at_ins.get_element() )
    return new_link_atom    
    
class residue:
    def __init__(self, label, index):
        self.label = label
        self.index = index
        self.new_index = None
        self.in_protein = False
        self.atoms = []
        self.main_chain_atoms = []
        self.trim = False

    def get_label(self):
        return self.label        
    def get_index(self):
        return self.index    
    def get_new_index(self):
        return self.new_index    
    def get_in_protein(self):
        return self.in_protein
    def get_atoms(self):
        return self.atoms
    def get_main_chain_atoms(self):
        return self.main_chain_atoms
    def next_atom(self):
        for atom in self.atoms:
            yield atom
    def get_trim(self):
        return self.trim
    
    def set_in_protein(self,in_protein):
        self.in_protein = in_protein
    def add_atom(self,atom):
        self.atoms.append(atom)
        if atom.get_tree_chain_classification() == 'M':
            self.main_chain_atoms.append(atom)
    def add_main_chain_atom(self,atom):
            self.main_chain_atoms.append(atom)
    def set_trim(self,trim):
        self.trim = trim
    def set_new_index(self,new_index):
        self.new_index = new_index    

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

def N_CO_in_residue(residue):
    """ checks is the residue contains N and CO as a first and two last atoms
    if yes, returns True, otherwise returns False"""
    temp = []
    for atom in residue.get_atoms():
        temp.append(atom.get_element())
    if len(temp) > 3:
        if temp[0] == 'N' and temp[-2] == 'C' and temp[-1] == 'O':
            return True
        elif temp[0] == 'N' and temp[-3] == 'C' and temp[-2] == 'O':
            return True
        else:
            return False
    else:
        return False
    
def main_side_chain(residue):
    """ determined which atoms are in the mainchain in a residue
    looks for N-C-C-O fragment and then H atoms bound to N or C
    and O atom bound to the second C 
    when found sets in_mainchain atribute of these atoms to True
    and adds them into main_chain_atoms list of this residue """
    for at1 in residue.get_atoms():
        if at1.get_element() == 'N':
            at1_connect = at1.get_connect_list()
            for at2 in residue.get_atoms():
                if at2.get_element() == 'C' and at2.get_index() in at1_connect:
                    at2_connect = at2.get_connect_list()
                    for at3 in residue.get_atoms():
                        if at3.get_element() == 'C' and at3.get_index() in at2_connect:
                            at3_connect = at3.get_connect_list()
                            for at4 in residue.get_atoms():
                                if at4.get_element() == 'O' and at4.get_index() in at3_connect:
                                    for at in [at1, at2, at3, at4]:
                                        at.set_in_mainchain(True)
                                        if at not in residue.get_main_chain_atoms():
                                            residue.add_main_chain_atom(at)
                                    for at5 in residue.get_atoms():
                                        if (at5.get_element() == 'H' and (at5.get_index() in at1_connect))\
                                        or (at5.get_element() == 'H' and (at5.get_index() in at2_connect))\
                                        or (at5.get_element() == 'O' and (at5.get_index() in at3_connect) and at5 != at4):
                                            at5.set_in_mainchain(True)
                                            residue.add_main_chain_atom(at5)

                                    

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

    
def is_peptide_bond2(residue_1,residue_2):
    """looks for O-C-N fragment between the two residues
    returns True if found, False otherwise
    assumes N is the first atom in a residue and CO are the last two in a residue
    in a peptide bond"""
    res1_atoms = residue_1.get_atoms()
    res2_atoms = residue_2.get_atoms()
    if len(res1_atoms) > 1 and len(res2_atoms) > 0:
        if res1_atoms[-2].get_element() == 'C' and res1_atoms[-1].get_element() == 'O'\
        and res2_atoms[0].get_element() == 'N'\
        and res1_atoms[-1].get_index() in res1_atoms[-2].get_connect_list()\
        and res2_atoms[0].get_index() in res1_atoms[-2].get_connect_list():
            return True
        else:
            return False
    else:
        return False    


 
class peptide:
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
        if len(self.residues)>0:
            return self.residues[-1]
        else:
            return None
    def next_residue(self):
        for residue in self.residues:
            yield residue                         
    def add_residue(self,residue):
        self.residues.append(residue)        

                           
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
    if not atom:
        result = [True for i in range(len(residue_list))]
        return result
    result = []
    temp = []
    temp.append(atom.get_coords())
    ref_coords = np.array(temp,dtype=float)
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
    if len(atom_list) == 0:
        result = [True for i in range(res_list_length)]
        return result
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
                


# functions to read input file containing info on: QM:MM setup, 
# trimming the model, freezing coordinates

def read_single_number(file, flag_line):
    """reads a single number from file and returns it as a numerical value 
    file : file object
    flag_line : string marker preceeding the value to be read"""
    file.seek(0)
    while True:
        a = file.readline()
        if not a:
            break
        match_flag=re.search(flag_line,a)
        if match_flag:
            if len(a.split())>1:
                return eval(a.split()[1])


def read_rsi_index(file, flag_line, end_line):
    """reads residue sidechain and index lines 
    contained between lines starting with flag_line
    and end_line from file 
    ---
    file : file object
    flag_line : string
    end_line : strong
    ---
    Returns lists:
    residue_index, sidechain_index and index_index
    """
    residue_index = []
    sidechain_index = []
    index_index = []
    # w pliku file wyszukaj linii zawierajacej flag_line
    file.seek(0)
    while True:
        a = file.readline()
        if not a:
            break
        match_flag=re.search(flag_line,a)
        if match_flag:            
            # wczytuj kolejne linie i sprawdzaj czy zawieraja "sidechain",
            # "residue" lub "index"
            while True:
                a = file.readline()
                if not a:
                    break
                elif re.search(end_line,a):
                    break
                match_residue=re.search("residue",a)
                match_sidechain=re.search("sidechain",a)
                match_index=re.search("index",a)
                if match_residue:                            
                    temp = a.split()
                    temp.remove('residue')
                    for item in temp:
                        residue_index.append(eval(item))
                elif match_sidechain:
                    temp = a.split()
                    temp.remove('sidechain')
                    for item in temp:
                        sidechain_index.append(eval(item))
                elif match_index:
                    temp = a.split()
                    temp.remove('index')
                    for item in temp:
                        index_index.append(eval(item))
            for lista in [residue_index, sidechain_index, index_index]:
                temp = set(lista)
                temp_2 = list(temp)
                lista = temp_2.sort()
    return residue_index, sidechain_index, index_index

def input_read_qm_part(file, residues, atoms):
    """
    Reads qm_part section from the prm2Gaussian input file
    Sets the oniom_layer attribute of atoms in the QM part to 'H' 
    Returns a list qm_part with atom objects making the QM part
    (the list is sorted wrt atom index)
    ----
    file : file object
    residues : a list with residue obects for the system
    atoms : a list with atom obects for the system
    """   
    assert (type(residues) == list), "residues must be a list of residue objects"
    assert (type(atoms) == list), "atoms must be a list of atom objects"
    # inicjalizacja pustej listy (na obiekty typu atom), ktora bedzie zwracana     
    qm_part = []
    residue_index,sidechain_index,index_index = read_rsi_index(file, "%qm_part", "%end_qm_part")
    if (len(residue_index) + len(sidechain_index) + len(index_index))>0:
        print('\nThe QM part of the system will consist of: ')
        print('residues: ',residue_index)
        print('and sidechains: ',sidechain_index)
        print('and atoms: ',index_index,'\n')
        for i in residue_index:
            if residues[i]:
                for at in residues[i].get_atoms():
                    qm_part.append(at)
            else:
                print('\n Residue list does not have residue with index: ',i, '\n')
        for i in sidechain_index:
            if residues[i]:
                for at in residues[i].get_atoms():
                    if not at.get_in_mainchain():
                        qm_part.append(at)
            else:
                    print('\n Residue list does not have residue with index: ',i, '\n')
        for i in index_index:
            if atoms[i]:
                qm_part.append(atoms[i])
            else:
                print('\n Atom list does not have atom with index: ',i,'\n')
        temp = set(qm_part)
        qm_part = list(temp)
        qm_part.sort(key=lambda x: x.get_index(), reverse=True)
        for at in qm_part:
            at.set_oniom_layer('H')
    return qm_part 

def input_read_link_atoms(file, atoms):
    """
    Reads link_atoms section from the prm2Gaussian input file
    
    Returns a list link_atoms with link_atom objects 
    (the list is sorted wrt atom index)
    ----
    file : file object
    atoms : a list with atom obects for the system
    """   
    assert (type(atoms) == list), "atoms must be a list of atom objects"
    # inicjalizacja pustej listy (na obiekty typu atom), ktora bedzie zwracana     
    MM_link_atoms = []
    HLA_types = []
    link_atoms = []
    index_index = []
    # w pliku file wyszukaj linii zawierajacej "%link_atoms"
    flag_line = "%link_atoms"
    end_line = "%end_link_atoms"
    file.seek(0)
    while True:
        a = file.readline()
        if not a:
            break
        match_flag=re.search(flag_line,a)
        if match_flag:            
            # wczytuj kolejne linie i sprawdzaj czy zawieraja "index"
            while True:
                a = file.readline()
                if not a:
                    break
                elif re.search(end_line,a):
                    break
                match_index=re.search("index",a)
                match_type=re.search("type",a)
                if match_index:
                    temp = a.split()
                    temp.remove('index')
                    for item in temp:
                        index_index.append(eval(item))
                elif match_type:
                    temp = a.split()
                    temp.remove('type')
                    for item in temp:
                        HLA_types.append(item)
            if len(index_index) == len(HLA_types) and len(index_index)>0 :
                index_index, HLA_types = (list(t) for t in zip(*sorted(zip(index_index, HLA_types))))
                print('\nThe MM atoms replaced in the QM part by H-link atom are ')
                print('atoms with index: ', index_index)
                print('with H-link atom Amber atom types: ', HLA_types,'\n')
                for i in index_index:
                    if atoms[i]:
                        MM_link_atoms.append(atoms[i])
                    else:
                        print('Atom list does not have atom with index: ',i,'\n')
                for at,tp in zip(MM_link_atoms, HLA_types):
                    link_atoms.append(atom_to_link_atom(at,tp))
    return link_atoms 

def input_read_trim(file, residues, atoms):
    """
    Reads trim data from the prm2Gaussian input file
    Sets the trim atribute of residues to be removed to True 
    Returns a list trimmed with residue objects with trim = True
    (the list is sorted wrt residue index)
    ----
    file : file object
    residues : a list with residue obects for the system
    atoms : a list with atom objects for the system
    """   
    assert (type(residues) == list), "residues must be a list of residue objects"
    assert (type(atoms) == list), "atoms must be a list of atom objects"
    # inicjalizacja pustej listy (na obiekty typu atom), ktora bedzie zwracana     
    trimmed = []
    trim_reference = []
    r_trim = read_single_number(file,"%r_trim")
    residue_index,sidechain_index,index_index = read_rsi_index(file, "%trim_ref", "%end_trim_ref")
    if (len(residue_index) + len(sidechain_index) + len(index_index))>0:
        print('\nThe reference with respect to which TRIMMING will be done consist of: ')
        print('residues: ',residue_index)
        print('and sidechains: ',sidechain_index)
        print('and atoms: ',index_index)
        print('\nAll protein atoms will be retained PLUS atoms of other residues with \
at least one atom within: ',r_trim,' A from the above reference \n')
        for i in residue_index:
            if residues[i]:
                for at in residues[i].get_atoms():
                    trim_reference.append(at)
            else:
                print('\n Residue list does not have residue with index: ',i, '\n')
        for i in sidechain_index:
            if residues[i]:
                for at in residues[i].get_atoms():
                    if not at.get_in_mainchain():
                        trim_reference.append(at)
            else:
                print('\n Residue list does not have residue with index: ',i, '\n')
        for i in index_index:
            if atoms[i]:
                trim_reference.append(atoms[i])
            else:
                print('\n Atom list does not have atom with index: ',i, '\n')
        temp = set(trim_reference)
        trim_reference = list(temp)
        trim_reference.sort(key=lambda x: x.get_index(), reverse=True)
            
        within_resids_bool = residues_within_r_from_atom_list(residues, trim_reference, r_trim)
        for w,resid in zip(within_resids_bool,residues):
            if w or resid.get_in_protein():
                resid.set_trim(False)
            else:
                resid.set_trim(True)
                trimmed.append(resid)
    return trimmed 

def input_read_freeze(file, residues, atoms):
    """
    Reads freeze data from the prm2Gaussian input file
    Sets the frozen atribute of atoms to be fixed to -1 
    Returns a list frozen with atom objects with frozen = -1
    (the list is sorted wrt atom index)
    ----
    file : file object
    residues : a list with residue obects for the system
    atoms : a list with atom objects for the system
    """   
    assert (type(residues) == list), "residues must be a list of residue objects"
    assert (type(atoms) == list), "atoms must be a list of atom objects"
    # inicjalizacja pustej listy (na obiekty typu atom), ktora bedzie zwracana     
    frozen = []
    freeze_reference = []
    r_free = read_single_number(file,"%r_free")
    residue_index,sidechain_index,index_index = read_rsi_index(file, "%freeze_ref", "%end_freeze_ref")
    if (len(residue_index) + len(sidechain_index) + len(index_index))>0:
        print('\nThe reference with respect to which FREEZING will be done consist of: ')
        print('residues: ',residue_index)
        print('and sidechains: ',sidechain_index)
        print('and atoms: ',index_index)
        print('\nAll residues with at least one atom within: ',r_free,' A from the above reference will not be fixed \n')
        for i in residue_index:
            if residues[i]:
                for at in residues[i].get_atoms():
                    freeze_reference.append(at)
            else:
                print('\n Residue list does not have residue with index: ',i, '\n')
        for i in sidechain_index:
            if residues[i]:
                for at in residues[i].get_atoms():
                    if not at.get_in_mainchain():
                        freeze_reference.append(at)
            else:
                print('\n Residue list does not have residue with index: ',i, '\n')
        for i in index_index:
            if atoms[i]:
                freeze_reference.append(atoms[i])
            else:
                print('\n Atom list does not have atom with index: ',i, '\n')
        temp = set(freeze_reference)
        freeze_reference = list(temp)
        freeze_reference.sort(key=lambda x: x.get_index(), reverse=True)
            
        within_resids_bool = residues_within_r_from_atom_list(residues, freeze_reference, r_free)
        for w,resid in zip(within_resids_bool,residues):
            if not w:
                frozen.append(resid)
                for at in resid.get_atoms():
                    at.set_frozen(-1)
    return frozen 