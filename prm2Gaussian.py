#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This is the main code for the script that is supposed
to read Amber prmtop and prmcrd(rst7) files and write
Gaussian16 Amber=softonly or ONIOM(QM,Amber=softonly) input file

The script expects 3 file names as command line arguments:
            #1 prmtop file
            #2 prmcrd(rst7) file
            #3 output G16 input file
optional    #4 input file for this script    

Warnings:
1. bond order is uniformly 1.0 for all bonded atom pairs in the connectivity section    

2. 1-4 interactions are all the same (as given by the NonBon line); at the moment
the script does not use the information stored in the scee_scale_factor and scnb_scale_factor
(might be useful to explore this for systems where not all 1-4 interactions are treated 
in the same way, but currently in G09 and G16 Revision C.01 the flags nbdir/nbterm are
not accepted in input)

3. for MM-only G16 input and real/low oniom point the total spin is calculated 
as either 1 or 2, depending on the total charge and atomic composition

4. dihedral term accepts periodicity from 1 to 4 (as in standard Amber and GAFF FF), 
higher values are ignored (warning is printed)

5. if atomic number read from prmtop is smaller than 1, it is corrected based on
the atom mass 

6. adds zero force constants for angles in triangular water molecules if explicit 
HW-HW bond data is present    

7. bond length scaling factor used to calculate the positions of H-link atoms is set
at 0.723886 (prm2Gaussian_functions : adjust_HLA_coords), 
atom type for H-link atoms is set to be 'HC' (before mapping to the G16 types) 
(this file, "check if all QM-MM bonds are capped with link atoms" section)

REQUIRED packages: numpy, pandas, scipy, re, sys, math, datetime, string, fortranformat
    
Created on Fri Oct  9 10:27:30 2020
Last update on 2/09/2021
branch: master

@author: borowski, wojdyla
Report bugs to: tomasz.borowski@ikifp.edu.pl or zuzanna.wojdyla@ikifp.edu.pl
"""
import sys
import numpy as np
import pandas as pd
import datetime
import string

from prm2Gaussian_functions import rad_to_deg, coord_to_atom_index, remove_redundant_data
from prm2Gaussian_functions import remove_eq_imp, remove_eq_dih, is_3rd_atom_central
from prm2Gaussian_functions import adjust_HLA_coords, write_xyz_file, res2atom_lists
from prm2Gaussian_functions import not_trimmed_res2atom_lists, write_xyz_file_MM_LA
from prm2Gaussian_functions import write_pdb_file

from read_prmtop import prmtop_read_pointers, prmtop_read_text_section
from read_prmtop import prmtop_read_numeric_section, crd_read_coordinates
from read_prmtop import atm_mass, atm_number, at_num_symbol
from read_prmtop import LEGIT_TEXT_FLAGS, LEGIT_NUM_FLAGS

from connectivity_list import gen_connectivity_line

from oniom import atom, residue, peptide
from oniom import generate_label, N_CO_in_residue, is_peptide_bond2
from oniom import main_side_chain
from oniom import read_single_number, input_read_qm_part, input_read_link_atoms
from oniom import input_read_trim, input_read_freeze, atom_to_link_atom

# CONSTANTS
letters = string.ascii_uppercase
digits = string.digits

HW_HW_bond = ('HW', 'HW')

# Important variables (switches):
VERBOSE = False
HW_HW_present = False
TRIM_MODEL = False
ONIOM = False
FREEZE = False
read_prm2Gaussian_inp = False

        
### ---------------------------------------------------------------------- ###
### Seting the file names                                                  ###
# prmtop_file = sys.argv[1]
# prmcrd_file = sys.argv[2]
# g16_inp_file = sys.argv[3]
# if len(sys.argv)>4:
#     prm2Gaussian_inp_file = sys.argv[4]
#     read_prm2Gaussian_inp = True

### ---------------------------------------------------------------------- ###
### test cases

prmtop_file = './pliki_do_testow/H6H/h6h-oxo+succinate+water_hyo.prmtop'
prmcrd_file = './pliki_do_testow/H6H/h6h-oxo+succinate+water_hyo.prmcrd'
g16_inp_file = './pliki_do_testow/H6H/h6h-oxo+succinate+water_hyo_2_09.mm.com'
prm2Gaussian_inp_file = './pliki_do_testow/H6H/prm2gaussian.oniom.inp_mm'
read_prm2Gaussian_inp = True

# prmtop_file = './pliki_do_testow/qm_ectc_core/5onn_1686_nga_c2_n2_2_76.prmtop'
# prmcrd_file = './pliki_do_testow/qm_ectc_core/5onn_1686_nga_c2_n2_2_76.rst7'
# g16_inp_file = './pliki_do_testow/qm_ectc_core/5onn_1686_nga_c2_n2_2_76.g16.com'
# prm2Gaussian_inp_file = './pliki_do_testow/qm_ectc_core/ectc_prm2g.oniom.inp'
# read_prm2Gaussian_inp = True

# prmtop_file = './pliki_do_testow/H6H/h6h-oxo+succinate+water_hyo.prmtop'
# prmcrd_file = './pliki_do_testow/H6H/h6h-oxo+succinate+water_hyo.prmcrd'
# g16_inp_file = './pliki_do_testow/H6H/h6h-oxo+succinate+water_hyo_17_07_b.com'
# prm2Gaussian_inp_file = './pliki_do_testow/H6H/prm2gaussian.oniom.inp'
# read_prm2Gaussian_inp = True
        
# prmtop_file = './pliki_do_testow/dihydroclavaminate/dihydroclavaminate.prmtop'
# prmcrd_file = './pliki_do_testow/dihydroclavaminate/dihydroclavaminate.prmcrd'
# g16_inp_file = './pliki_do_testow/dihydroclavaminate/dihydroclavaminate_21_04.com'

# prmtop_file = './pliki_do_testow/H6H/h6h-oxo+succinate+water_hyo.nowat.prmtop'
# prmcrd_file = './pliki_do_testow/H6H/h6h-oxo+succinate+water_hyo.nowat.prmcrd'
# g16_inp_file = './pliki_do_testow/H6H/h6h-oxo+succinate+water_hyo.nowat_24_03.com'

# prmtop_file = './pliki_do_testow/cellulose/NAG.prmtop'
# prmcrd_file = './pliki_do_testow/cellulose/first_frame.rst7'
# g16_inp_file = './pliki_do_testow/cellulose/first_frame.com'
    
# prmtop_file = './pliki_do_testow/2_atom_systems/coulomb/AB_coulomb.prmtop'
# prmcrd_file = './pliki_do_testow/2_atom_systems/coulomb/AB_coulomb.prmcrd'
# g16_inp_file = './pliki_do_testow/2_atom_systems/coulomb/AB_coulomb.com'

# prmtop_file = './pliki_do_testow/4_atom_systems/ABCD_2.prmtop'
# prmcrd_file = './pliki_do_testow/4_atom_systems/ABCD_2.prmcrd'
# g16_inp_file = './pliki_do_testow/4_atom_systems/ABCD_2.com'

# prmtop_file = './pliki_do_testow/n_butan/n_butan_1.prmtop'
# prmcrd_file = './pliki_do_testow/n_butan/n_butan_1.prmcrd'
# g16_inp_file = './pliki_do_testow/n_butan/n_butan_1.com'

# prmtop_file = './pliki_do_testow/cl_prop/cl_prop.prmtop'
# prmcrd_file = './pliki_do_testow/cl_prop/cl_prop.prmcrd'
# g16_inp_file = './pliki_do_testow/cl_prop/cl_prop.com'

# prmtop_file = './pliki_do_testow/c2h4/c2h4.prmtop'
# prmcrd_file = './pliki_do_testow/c2h4/c2h4.prmcrd'
# g16_inp_file = './pliki_do_testow/c2h4/c2h4.com'

# prmtop_file = './pliki_do_testow/clcobr/clcobr.prmtop'
# prmcrd_file = './pliki_do_testow/clcobr/clcobr.prmcrd'
# g16_inp_file = './pliki_do_testow/clcobr/clcobr.com'

### ---------------------------------------------------------------------- ###
### Reading from prmtop file                                               ###

# open the prmtop file and read-in pointers from the prmtop file:
if VERBOSE:
    print("Program started at: ", datetime.datetime.now(), "\n")

prmtop = open(prmtop_file, 'r')
pointers = prmtop_read_pointers(prmtop)

# calculate and populate exp_length dictionary based on the pointers
NATOM = pointers['NATOM']
NTYPES = pointers['NTYPES']
NRES = pointers['NRES']   
NUMBND = pointers['NUMBND']
NUMANG = pointers['NUMANG']
NPTRA = pointers['NPTRA']
NATYP = pointers['NATYP']
NBONH = pointers['NBONH']
NBONA = pointers['NBONA']
NTHETH = pointers['NTHETH']
NTHETA = pointers['NTHETA']
NPHIH = pointers['NPHIH']
NPHIA = pointers['NPHIA']
NNB = pointers['NNB']
NPHB = pointers['NPHB']
try:
    NSPM = prmtop_read_numeric_section(prmtop, FLAG='SOLVENT_POINTERS',exp_length=3)[1]
except AssertionError:
    NSPM = None 
#
exp_length = {'ATOM_NAME' : NATOM, 'CHARGE' : NATOM, 'ATOMIC_NUMBER' : NATOM,\
              'MASS' : NATOM, 'ATOM_TYPE_INDEX' : NATOM, 'NUMBER_EXCLUDED_ATOMS' : NATOM,\
              'NONBONDED_PARM_INDEX' : NTYPES**2, 'RESIDUE_LABEL' : NRES,\
              'RESIDUE_POINTER' : NRES, 'BOND_FORCE_CONSTANT' : NUMBND,\
              'BOND_EQUIL_VALUE' : NUMBND, 'ANGLE_FORCE_CONSTANT' : NUMANG,\
              'ANGLE_EQUIL_VALUE' : NUMANG, 'DIHEDRAL_FORCE_CONSTANT' : NPTRA,\
              'DIHEDRAL_PERIODICITY' : NPTRA, 'DIHEDRAL_PHASE' : NPTRA,\
              'SCEE_SCALE_FACTOR' : NPTRA, 'SCNB_SCALE_FACTOR' : NPTRA,\
              'SOLTY' : NATYP, 'LENNARD_JONES_ACOEF' : int((NTYPES*(NTYPES+1))/2),\
              'LENNARD_JONES_BCOEF' : int((NTYPES*(NTYPES+1))/2), 'BONDS_INC_HYDROGEN' : 3*NBONH,\
              'BONDS_WITHOUT_HYDROGEN' : 3*NBONA, 'ANGLES_INC_HYDROGEN' : 4*NTHETH,\
              'ANGLES_WITHOUT_HYDROGEN' : 4*NTHETA, 'DIHEDRALS_INC_HYDROGEN' : 5*NPHIH,\
              'DIHEDRALS_WITHOUT_HYDROGEN' : 5*NPHIA, 'EXCLUDED_ATOMS_LIST' : NNB,\
              'HBOND_ACOEF' : NPHB, 'HBOND_BCOEF' : NPHB, 'HBCUT' : NPHB, 'AMBER_ATOM_TYPE' : NATOM,\
              'TREE_CHAIN_CLASSIFICATION' : NATOM, 'JOIN_ARRAY' : NATOM, 'IROTAT' : NATOM,\
              'SOLVENT_POINTERS' : 3, 'ATOMS_PER_MOLECULE' : NSPM, 'BOX_DIMENSIONS' : 4,\
              'CAP_INFO' : 1, 'CAP_INFO2' : 4, 'RADIUS_SET' : 1, 'RADII' : NATOM,\
              'IPOL' : 1, 'POLARIZABILITY' : NATOM, 'SCREEN' : NATOM}


# will read to two dictionaries that contain data read from prmtop file,
# keys are flags in lower case

# read text sections of prmtop:    
prmtop_text_sections = {}
for flag in LEGIT_TEXT_FLAGS:
     prmtop_text_sections[flag.lower()] = prmtop_read_text_section(prmtop, flag, exp_length[flag])


# modify the content of LEGIT_NUM_FLAGS based on the pointers values, i.e. 
# remove those flags (pertaining to numeric sections) that are not present 
# in the read-in prmtop file
if pointers['IFBOX'] == 0:
    for item in ['SOLVENT_POINTERS', 'ATOMS_PER_MOLECULE', 'BOX_DIMENSIONS']:
        LEGIT_NUM_FLAGS.remove(item)

if pointers['IFCAP'] == 0:
    for item in ['CAP_INFO', 'CAP_INFO2']:
        LEGIT_NUM_FLAGS.remove(item)        

if prmtop_read_numeric_section(prmtop, 'IPOL', 1)[0] == 0:
    LEGIT_NUM_FLAGS.remove('POLARIZABILITY')

# read numeric sections from prmtop:
prmtop_num_sections = {}
for flag in LEGIT_NUM_FLAGS:
     prmtop_num_sections[flag.lower()] = prmtop_read_numeric_section(prmtop, flag, exp_length[flag])

# after reading close the prmtop file: 
prmtop.close()
if VERBOSE:
    print("prmtop file has been read ", datetime.datetime.now(), "\n")

### ---------------------------------------------------------------------- ###
### Reading from prmcrd / rst7 file                                        ###

# read number of atoms and coordinates from the prmcrd (rst7) file:
prmcrd = open(prmcrd_file, 'r')
natom_crd, coordinates = crd_read_coordinates(prmcrd)
prmcrd.close()

if VERBOSE:
    print("prmcrd/rst7 file has been read ", datetime.datetime.now(), "\n")


# check if the number of atoms matches that read from prmtop file:
if natom_crd != NATOM:
    print("ATTANTION, the number of atoms read from prmtop and prmcrd(rst7) files DO NOT MATCH !!!\n")
    print("from prmtop: ", NATOM, " \n")
    print("from prmcrd: ", natom_crd, " \n")
    

### ---------------------------------------------------------------------- ###
### Processing the prmtop data                                             ###

# converting atomic charges to atomic units:
for i in range(len(prmtop_num_sections['charge'])):
    prmtop_num_sections['charge'][i] /= 18.2223

# converting angle equilibrium values from radians to degrees:
for i in range(len(prmtop_num_sections['angle_equil_value'])):
    prmtop_num_sections['angle_equil_value'][i] = \
    rad_to_deg(prmtop_num_sections['angle_equil_value'][i])
    
# dihedral phase is read in [rad], convert to degrees:
for i in range(len(prmtop_num_sections['dihedral_phase'])):
    prmtop_num_sections['dihedral_phase'][i] = \
    rad_to_deg(prmtop_num_sections['dihedral_phase'][i])

# check if entries in atomic_number are OK, if not, infer from mass
for i in range(NATOM):
    if prmtop_num_sections['atomic_number'][i] < 1:
        print('Warning, atom #: ', i, ' has atomic number: ', prmtop_num_sections['atomic_number'][i])
        print('I try to get the correct value based on mass')
        mass_x = prmtop_num_sections['mass'][i]
        min_mass_diff = 300.0
        min_mass_diff_element = None
        for element, mass in zip(atm_mass.keys(), atm_mass.values()):
            mass_diff = abs(mass - mass_x)
            if mass_diff < min_mass_diff:
                min_mass_diff = mass_diff
                min_mass_diff_element = element
        prmtop_num_sections['atomic_number'][i] = atm_number[min_mass_diff_element]
        print('Assigned atomic number is: ' + str(atm_number[min_mass_diff_element]) +\
              ' ( ' + min_mass_diff_element + ' )')

# check if atom types are all unique and start with a letter when CAPITALIZED 
# G16 does not accepts types starting with numbers and does not distinguish small
# and capital letters, using ' ' or " " does not help (for G09 the latter worked)            

# from prmtop_text_sections['amber_atom_type'] select a unique set of atom types
unique_types = list( set(prmtop_text_sections['amber_atom_type']) )
unique_types.sort()
num_unq_types =  len(unique_types)

# capitalize all letters:
unique_types_CAP = []
for atom_type in unique_types:
    unique_types_CAP.append(atom_type.upper())

# for type starting with a digit add 'A' at the front: 
unique_types_temp = []
for atom_type in unique_types_CAP:
    if atom_type[0] in digits:
        new_atom_type = 'A' + atom_type
        unique_types_temp.append(new_atom_type)
    else:
        unique_types_temp.append(atom_type)

# test for repeating type symbols and append letters from letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
unique_types_NEW = unique_types_temp.copy()       
for i in range(num_unq_types):
    typ_i = unique_types_temp[i]
    for j in range(i+1, num_unq_types):
        typ_j = unique_types_temp[j]
        if typ_i == typ_j:
            for letter in letters:
                try_new_typ_j = typ_j + letter
                if try_new_typ_j not in set(unique_types_NEW):
                    unique_types_NEW[j] = try_new_typ_j 
                    break

# test if 1:1 mapping between unique_types and unique_types_NEW:
if len(set(unique_types_NEW)) != num_unq_types:
    print('ATTENTION, the mapping between original atom types (unique_types) and new\
          atom types - to be used in Gaussian (unique_types_NEW) is not 1:1 !!!')

# create dictionary translating original types to the new ones:
unq_to_NEW_types = dict( zip(unique_types, unique_types_NEW) )          

# print the mapping between the original and new atom types:
print('Applied mapping from original atom types to a set suitable for G16: \n')
for key in unq_to_NEW_types.keys():
    print(str(key) + ' ---> ' + str(unq_to_NEW_types[key]))

if VERBOSE:
    print("unique_types created ", datetime.datetime.now(), "\n")            
    print("initial processing done ", datetime.datetime.now(), "\n")

# connectivity list section
# connect_list - a dic with (int) keys - atom numbers (indexed from 0)
# and values - lists of atoms connected to the given atom (indexed from 0)
connect_list = {}
# read data from prmtop_num_sections['bonds_inc_hydrogen']:
for i in range(NBONH):
    atom_1 = coord_to_atom_index( prmtop_num_sections['bonds_inc_hydrogen'][3*i] ) - 1
    atom_2 = coord_to_atom_index( prmtop_num_sections['bonds_inc_hydrogen'][3*i+1] ) - 1 
    if atom_1 in connect_list.keys():
        connect_list[atom_1].append(atom_2)
    else:
        connect_list[atom_1] = [atom_2]
    if atom_2 in connect_list.keys():
        connect_list[atom_2].append(atom_1)
    else:
        connect_list[atom_2] = [atom_1]
# read data from prmtop_num_sections['bonds_without_hydrogen']:        
for i in range(NBONA):
    atom_1 = coord_to_atom_index( prmtop_num_sections['bonds_without_hydrogen'][3*i] ) - 1
    atom_2 = coord_to_atom_index( prmtop_num_sections['bonds_without_hydrogen'][3*i+1] ) - 1
    if atom_1 in connect_list.keys():
        connect_list[atom_1].append(atom_2)
    else:
        connect_list[atom_1] = [atom_2]
    if atom_2 in connect_list.keys():
        connect_list[atom_2].append(atom_1)
    else:
        connect_list[atom_2] = [atom_1]
        
# add empty lists for atoms not making any bonds
for i in range(NATOM):
    if i not in connect_list.keys():
        connect_list[i] = []
        
if VERBOSE:
    print("connect_list created ", datetime.datetime.now(), "\n")
        
# sort lists of connected atoms:
for i in range(NATOM):
    connect_list[i].sort()
if VERBOSE:
    print("connect_list sorted ", datetime.datetime.now(), "\n")

# read bonds information into a Pandas DataFrame with name bonds
# first_instance - a flag marking if this is the first occurance of a given type (now it's redundant)
columns = ['ind_to_tabels', 'Atom_types',\
           'force_constant', 'equil_value', 'first_instance']    
# gather the required info in an auxiliary list bonds_data:
bonds_data = []

for i in range(NUMBND):
    force_constant = prmtop_num_sections['bond_force_constant'][i]
    equil_value = prmtop_num_sections['bond_equil_value'][i]
    first_instance = True
    temp = [None,None,None]
    temps_i = []
    for bond_type in ['bonds_inc_hydrogen', 'bonds_without_hydrogen']:
        for j in range(int(round(len(prmtop_num_sections[bond_type])/3))):
                temp = prmtop_num_sections[bond_type][3*j : 3*j+3]
                if temp[-1] == i+1:
                    temps_i.append(temp)
    raws = []
    for temp in temps_i:
        atom_numbers = [int(round(n/3)) for n in temp[0:2]]
        atom_types = tuple( [prmtop_text_sections['amber_atom_type'][k] for k in atom_numbers] )
        raw = [i+1, atom_types, force_constant, equil_value, first_instance]
        raws.append(raw)
    unique_raws_tup = set(tuple(row) for row in raws)
    unique_raws = [list(raw) for raw in unique_raws_tup]
    bonds_data = bonds_data + unique_raws

# remove redundant data, e.g. one of the pair: (A-B), (B-A)
bonds_data_non_redund = remove_redundant_data(bonds_data,1)

if VERBOSE:
    print("auxiliary list bonds_data_non_redundant done ", datetime.datetime.now(), "\n")

# sorting bonds_data wrt (atom_types):
bonds_data_non_redund.sort(key = lambda x: x[1])    

if VERBOSE:
    print("auxiliary list bonds_data_non_redund sorted ", datetime.datetime.now(), "\n")
        
# create DataFrame from the list    
bonds = pd.DataFrame(bonds_data_non_redund, columns=columns)   

if VERBOSE:
    print("bonds DataFrame created ", datetime.datetime.now(), "\n")

# check if HW-HW bond is present:
for bond in bonds.Atom_types:
    if bond == HW_HW_bond:
       HW_HW_present = True
       break

# read angles information into a Pandas DataFrame with name angles
columns = ['ind_to_tabels', 'Atom_types',\
           'force_constant', 'equil_value', 'first_instance']    
# first make auxiliary list angles_data:
angles_data = []

for i in range(NUMANG):
    force_constant = prmtop_num_sections['angle_force_constant'][i]
    equil_value = prmtop_num_sections['angle_equil_value'][i]
    first_instance = True
    temp = [None,None,None,None]
    temps_i = []
    for ang_type in ['angles_inc_hydrogen', 'angles_without_hydrogen']:
        for j in range(int(round(len(prmtop_num_sections[ang_type])/4))):
            temp = prmtop_num_sections[ang_type][4*j : 4*j+4]
            if temp[-1] == i+1:
                temps_i.append(temp)
    raws = []
    for temp in temps_i:  
        atom_numbers = [int(round(n/3)) for n in temp[0:3]]
        atom_types = tuple( [prmtop_text_sections['amber_atom_type'][k] for k in atom_numbers] )
        raw = [i+1, atom_types, force_constant, equil_value, first_instance]
        raws.append(raw)
    unique_raws_tup = set(tuple(row) for row in raws)
    unique_raws = [list(raw) for raw in unique_raws_tup]        
    angles_data = angles_data + unique_raws

# remove redundant data, e.g. one of the pair: (A-B-C), (C-B-A)
angles_data_non_redund = remove_redundant_data(angles_data,1)

# add zero force constants for angles in triangular water molecules 
# if explicit HW-HW bond data is present in the bonds_data_non_redundant
if HW_HW_present:
    angles_data_non_redund.append([0, ('OW', 'HW', 'HW'), 0.0, 0.0, True])
    angles_data_non_redund.append([0, ('HW', 'OW', 'HW'), 0.0, 0.0, True])
    print('\n Adding dummy angle force constants for triangulat WAT, since HW-HW bond present \n')

if VERBOSE:
    print("auxiliary list angles_data_non_redund done ", datetime.datetime.now(), "\n")

# sorting angles_data wrt (atom_types):
angles_data_non_redund.sort(key = lambda x: x[1])
        
# create DataFrame from the list     
angles = pd.DataFrame(angles_data_non_redund, columns=columns)
if VERBOSE:
    print("angles DataFrame created ", datetime.datetime.now(), "\n")

# read dihedral angles information into a list dihedral_data
# (for debugging also into a Pandas DataFrame with the name angles dihedrals)
columns = ['ind_to_tabels', 'if_nonbonded', 'is_improper', 'Atom_types',\
           'force_constant', 'periodicity', 'phase', 'scee_scale_factor', 'scnb_scale_factor',
           'first_instance']    
# first make auxiliary list dihedral_data:
dihedral_data = []
for i in range(NPTRA):
    force_constant = prmtop_num_sections['dihedral_force_constant'][i]
    periodicity = prmtop_num_sections['dihedral_periodicity'][i]
    phase = prmtop_num_sections['dihedral_phase'][i]
    scee_scale_factor = prmtop_num_sections['scee_scale_factor'][i]
    scnb_scale_factor = prmtop_num_sections['scnb_scale_factor'][i]
    first_instance = True
    temp = [None,None,None,None,None]
    temps_i = []
    for dih_type in ['dihedrals_inc_hydrogen', 'dihedrals_without_hydrogen']:
        for j in range(int(round(len(prmtop_num_sections[dih_type])/5))):
            temp = prmtop_num_sections[dih_type][5*j : 5*j+5]
            if temp[-1] == i+1:
                temps_i.append(temp)
    raws = []
    for temp in temps_i:
        if temp[2] < 0:
            if_nonbonded = False
        else:
            if_nonbonded = True
        if temp[3] < 0:
            is_improper = True
            atom_numbers = [int(np.abs(n)/3) for n in temp[0:4]]
#           here check if the 3rd atom of the improper is the central one
#           Gaussian requires the central atom is the 3rd in the improper quartet:
            if not is_3rd_atom_central(atom_numbers,connect_list):
                atom_numbers = list(reversed(atom_numbers))
                if not is_3rd_atom_central(atom_numbers,connect_list):
                    print("Houston, I have a problem with the improper angle:", atom_numbers)
        else:
            is_improper = False
            atom_numbers = [int(np.abs(n)/3) for n in temp[0:4]]
            
        atom_types = tuple( [prmtop_text_sections['amber_atom_type'][k] for k in atom_numbers] )  
        raw = [i+1, if_nonbonded, is_improper, atom_types,\
            force_constant, periodicity, phase, scee_scale_factor, scnb_scale_factor, first_instance]
        raws.append(raw)
        
    unique_raws_tup = set(tuple(row) for row in raws)
    unique_raws = [list(raw) for raw in unique_raws_tup]
    dihedral_data = dihedral_data + unique_raws

# split dihedral_data into proper_dihedral_data and improper_dihedral_data
# to avoid removing impropers for the same quartet of atoms as for proper dih or vice versa    

proper_dihedral_data = []
improper_dihedral_data = []
for entry in dihedral_data:
    if entry[2]:
        improper_dihedral_data.append(entry)
    else:
        proper_dihedral_data.append(entry)    

# now remove redundant entries, e.g. one of: (A-B-C-D), (D-C-B-A)
dihedral_data_non_redund = remove_eq_dih(remove_redundant_data(proper_dihedral_data,3))
improper_data_non_redund = remove_eq_imp(remove_redundant_data(improper_dihedral_data,3))

if VERBOSE:        
    print("auxiliary lists dihedral_data_non_redund and improper_data_non_redundant\
          done ", datetime.datetime.now(), "\n")

# sorting dihedral_data wrt (atom_types):
dihedral_data_non_redund.sort(key = lambda x: x[3])
improper_data_non_redund.sort(key = lambda x: x[3])    

if VERBOSE:
    print("auxiliary list dihedral_data_non_redund sorted ", datetime.datetime.now(), "\n")
        
# create DataFrames from the lists (for debuging, as these DataFrames are not used) 
# dihedrals = pd.DataFrame(dihedral_data_non_redund, columns=columns)    
# impropers = pd.DataFrame(improper_data_non_redund, columns=columns)
# if VERBOSE:            
#     print("dihedrals and impropers DataFrames created ", datetime.datetime.now(), "\n")


# extracting atomic vdW parameters:
    
# for each atom type:   
    # find atom index for atom of this type and a pointer to lennard_jones_a(b) coef for 
    # the pair: this type, this type
    # read A and B and convert into epsilon and r_m
#
# epsilon and r_min are dictionaries with amber atom types as keys and
# epsilon and r_min values, respectively
epsilon = {}
r_min = {}
for type in unique_types:
    # i - zero-based index of atom in an amber_atom_type type
    i = prmtop_text_sections['amber_atom_type'].index(type)
    # j = atom_type_index(i):
    j = prmtop_num_sections['atom_type_index'][i]
    # index - index into lennard_jones_a(b)coef for A_ii and B_ii values:
    index = prmtop_num_sections['nonbonded_parm_index'][(NTYPES*(j-1)+j)-1]
    # read A and B values:
    A = prmtop_num_sections['lennard_jones_acoef'][index-1]
    B = prmtop_num_sections['lennard_jones_bcoef'][index-1]
    # convert (A, B) into (epsilon, r_min) values:
    if A > 0.0 and B > 0.0:
        eps = B**2/(4*A)
        r_m = 0.5 * ((2*A/B)**(1/6))
    else:
        eps=0.0
        r_m=0.0
    # add the values to the dictionaries:
    epsilon[type] = eps
    r_min[type] = r_m

if VERBOSE:    
    print("epsilon and r_min created ", datetime.datetime.now(), "\n")



### ------------------------------------------------------------------------- ###
### preparing obj data for model manipulation if prm2Gaussian_inp_file exists ###
    
atoms = []      # a list of atom objects
residues = []   # a list of residue objects
chains = []     # a list of peptide objects

if read_prm2Gaussian_inp:
    for i in range(NATOM):
        element = at_num_symbol[prmtop_num_sections['atomic_number'][i]]
        tree_chain_classification = prmtop_text_sections['tree_chain_classification'][i]
        at_type = prmtop_text_sections['amber_atom_type'][i]
        new_at_type = unq_to_NEW_types[ at_type ]
        at_chg = prmtop_num_sections['charge'][i]
        at_name = prmtop_text_sections['atom_name'][i]
    
        new_atom = atom(coordinates[i], i, element)
        new_atom.set_tree_chain_classification(tree_chain_classification)
        new_atom.set_connect_list(connect_list[i])
        new_atom.set_type(at_type)
        new_atom.set_new_type(new_at_type)
        new_atom.set_at_charge(at_chg)
        new_atom.set_name(at_name)
        
        atoms.append(new_atom)


    atom_ranges = prmtop_num_sections['residue_pointer']
    atom_ranges.append(NATOM+1)
    for i in range(NRES):
        label = prmtop_text_sections['residue_label'][i]
        new_residue = residue(label, i)   
    
        first_at_index = atom_ranges[i] - 1 
        last_at_index_plus_1 = atom_ranges[i+1] - 1
        for j in range(first_at_index,last_at_index_plus_1,1):
            new_residue.add_atom(atoms[j])
        
        residues.append(new_residue)

# process residues to set in_mainchain atribute of atoms (protein main chain)
# and populate main_chain_atoms atribute of residues    
    for res in residues:
        main_side_chain(res)
    
    
    gen_label = generate_label().__next__
    
    chain_indx = 0
    new_chain = peptide(gen_label(), chain_indx)
    
    for res in residues:
        if N_CO_in_residue(res):
            res.set_in_protein(True)
            chain_last_resid = new_chain.get_last_residue()
            if chain_last_resid:
                if is_peptide_bond2(chain_last_resid,res):
                    new_chain.add_residue(res)
                else:
                    chains.append(new_chain)
                    chain_indx += 1
                    new_chain = peptide(gen_label(), chain_indx)
                    new_chain.add_residue(res)
            else:
                new_chain.add_residue(res)
        elif new_chain.get_last_residue():
            chains.append(new_chain)
            chain_indx += 1
            new_chain = peptide(gen_label(), chain_indx)       
            
### ---------------------------------------------------------------------- ###
### Reading prm2Gaussian input file, if it exists                          ###

if read_prm2Gaussian_inp:
    prm2g = open(prm2Gaussian_inp_file, 'r')
    
    qm_part = input_read_qm_part(prm2g, residues, atoms)
    link_atoms = input_read_link_atoms(prm2g, atoms)
    trimmed = input_read_trim(prm2g, residues, atoms)
    frozen = input_read_freeze(prm2g, residues, atoms)
    
    qm_chg = read_single_number(prm2g, "%qm_charge")
    qm_mul = read_single_number(prm2g, "%qm_multip")
    
    prm2g.close()


    if VERBOSE:
        print("prm2Gaussian input file has been read ", datetime.datetime.now(), "\n")

# set LAH atribute of LAH atoms
    for lk_atom in link_atoms:
        atoms[lk_atom.get_index()].set_LAH(True)        

# set some key switch variables based on results of reading prm2Gaussian file
    if len(qm_part) > 0:
        ONIOM = True
    if len(trimmed) > 0:
        TRIM_MODEL = True
    if len(frozen) > 0:
        FREEZE = True
        
### ---------------------------------------------------------------------- ###
### Compute new atom indexes if TRIM_MODEL = True                          ###
    if TRIM_MODEL:
        old_new_at_ix = {}
        i = 0
        j = 0
        for res in residues:
            if not res.get_trim():
                res.set_new_index(i)
                i += 1
                for at in res.get_atoms():
                    at.set_new_index(j)
                    key = at.get_index()
                    old_new_at_ix[key] = j
                    j += 1


### ---------------------------------------------------------------------- ###
### H-link atom manipulations (HLA position, bonded_to, set new_at_type,   ###
### set new_index                
    if ONIOM:
        lk_at_indexes = []
    
        for at in link_atoms:
            lk_at_indexes.append( at.get_index() )
            con_list = at.get_connect_list()
            tp = at.get_type()
            if tp in unq_to_NEW_types.keys(): # set new_at_type
                at.set_new_type( unq_to_NEW_types[tp] ) 
            else:
                at.set_new_type(tp)
            at.set_new_index( atoms[ at.get_index() ].get_new_index() ) # set new_index
            for con_ix in con_list: # find H-layer (QM) atom bonded to a given link atom
                con_at = atoms[con_ix]
                if con_at.get_oniom_layer() == 'H':
                    qm_1 = con_at
                    at.set_bonded_to(con_ix)
                    adjust_HLA_coords(at, con_at)
                    break

        
### ---------------------------------------------------------------------- ###
### check if all QM-MM bonds are capped with link atoms                    ###
    qm_indexes = []
    for at in qm_part:
        qm_indexes.append( at.get_index() )
    
    link_atoms_updated = False    
    for at in qm_part:
        qm_at_connect = at.get_connect_list()
        for item in qm_at_connect:
            if (item not in qm_indexes) and (item not in lk_at_indexes.copy()):
                print("\nFound a QM-MM bond not capped with H-link atom")
                print("between atoms with (0-based) index of: ", at.get_index(), item)
                print("Adding a standard H-link atom with HC type\n")
                new_lk_atom = atom_to_link_atom(atoms[item], 'HC', 0.000001)
                new_lk_atom.set_bonded_to(at.get_index())
                if 'HC' in unq_to_NEW_types.keys():
                    new_lk_atom.set_new_type( unq_to_NEW_types['HC'] )
                else:
                    new_lk_atom.set_new_type( 'HC' )
                adjust_HLA_coords(new_lk_atom, at)
                link_atoms.append(new_lk_atom)
                lk_at_indexes.append(item)
                link_atoms_updated = True
    
    # sort link_atoms if this list was expanded:
    if link_atoms_updated:
        link_atoms.sort(key=lambda x: x.get_index(), reverse=True)
        lk_at_indexes.sort()            

### ---------------------------------------------------------------------- ###
### check if type and bonded parameters for HLA are present                ###
    redundant_bonds = []
    redundant_angles = []
    redundant_dihedrals = []

    for item in bonds_data:
        redundant_bonds.append(item[1])
    for item in angles_data:
        redundant_angles.append(item[1])
    for item in dihedral_data:
        redundant_dihedrals.append(item[3])

    for at in link_atoms:
        lk_at_ix = at.get_index()
        lk_at_nix = at.get_new_index()
        con_list = at.get_connect_list()
        for con_ix in con_list: # find H-layer (QM) atom bonded to a given link atom
            con_at = atoms[con_ix]
            if con_at.get_oniom_layer() == 'H':
                qm_1 = con_at
                qm_1_type = qm_1.get_type()
                qm_1_ix = qm_1.get_index()
                qm_1_nix = qm_1.get_new_index()
        at_type = at.get_type()
        if at_type not in unique_types: # check if given types (for H-link atoms) are present
            print('\nH-link atom type ', at_type, 'not present in this prmtop file')
            print('You will need to add its non-bonded and bonded parameters into G16 input file by hand')
            print('atom index and new index: ',lk_at_ix, lk_at_nix, '\n')
        bond_tup = (at_type, qm_1_type) # checking bond parameters
        bond_tup_r = (qm_1_type, at_type)
        if (bond_tup not in redundant_bonds) and (bond_tup_r not in redundant_bonds):
            print('For link atom of index, new index: ', lk_at_ix, lk_at_nix, 'bond parameters are missing: '\
                  , str((bond_tup)), lk_at_ix, lk_at_nix,'-', qm_1_ix, qm_1_nix,'\n')
        for item in qm_1.get_connect_list():
            if item != lk_at_ix:
                qm_2= atoms[item]
                qm_2_type = qm_2.get_type()
                qm_2_ix = qm_2.get_index()
                qm_2_nix = qm_2.get_new_index()
                ang_tup = (at_type, qm_1_type, qm_2_type)
                ang_tup_r = (qm_2_type, qm_1_type, at_type)
                if (ang_tup not in redundant_angles) and (ang_tup_r not in redundant_angles):
                    print('For link atom of index, new index: ', lk_at_ix, lk_at_nix, 'angle parameters are missing: '\
                          , str((ang_tup)), lk_at_ix, lk_at_nix, '-', qm_1_ix, qm_1_nix, '-', qm_2_ix, qm_2_nix,'\n')
                for item2 in qm_2.get_connect_list():
                    if item2 != qm_1_ix:
                        qm_3 = atoms[item2]
                        qm_3_type = qm_3.get_type()
                        qm_3_ix = qm_3.get_index()
                        qm_3_nix = qm_3.get_new_index()
                        dih_tup = (qm_3_type, qm_2_type, qm_1_type, at_type)
                        dih_tup_r = (at_type, qm_1_type, qm_2_type, qm_3_type)
                        if (dih_tup not in redundant_dihedrals) and (dih_tup_r not in redundant_dihedrals):
                            print('For link atom of index, new index: ', lk_at_ix, lk_at_nix, 'dihedral angle parameters are missing: '\
                          , str((dih_tup)), lk_at_ix, lk_at_nix, '-', qm_1_ix, qm_1_nix, '-', qm_2_ix, qm_2_nix, '-', qm_3_ix, qm_3_nix,'\n')


# set chain attribute for all residues belonging to peptide chains:
for chain in chains:
    chain_label = chain.get_label()
    for resid in chain.get_residues():
        resid.set_chain(chain_label)
            
# set chain attribute to all other residues that are not trimmed:
residue_list_no_trim = []        
for resid in residues:
    if not resid.get_trim():
        if resid.get_chain() == '':
            resid.set_chain( gen_label() )
            residue_list_no_trim.append(resid) 

### ---------------------------------------------------------------------- ###
### write out xyz files with qm_system, qm_part, model, HLA, MM_LA,  ###
### frozen, trimmed, qm_mm_free, qm_mm_frozen                              ###
if ONIOM:
    qm_system = qm_part + link_atoms
    qm_system.sort(key=lambda x: x.get_index(), reverse=True)
    write_xyz_file(qm_system, 'QM_SYSTEM.xyz')
    write_xyz_file(qm_part, 'QM_PART.xyz')

    if len(link_atoms) > 0:
        write_xyz_file_MM_LA(link_atoms, 'MM_LA.xyz')

if TRIM_MODEL:
    trimmed_atoms = res2atom_lists(trimmed)
    write_xyz_file(trimmed_atoms, 'TRIMMED.xyz')

if FREEZE:
    frozen_atoms = res2atom_lists(frozen)
    write_xyz_file(frozen_atoms, 'FROZEN.xyz')

if (ONIOM or TRIM_MODEL or FREEZE):
    model = not_trimmed_res2atom_lists(residues)
    write_xyz_file(model, 'MODEL.xyz')
    write_pdb_file(residues, 'MODEL.pdb')
else:
    xyz_file = open('MODEL.xyz', 'w')
    xyz_file.write(str(NATOM)+'\n')
    xyz_file.write('\n')
    for i in range(NATOM):    
        line = at_num_symbol[prmtop_num_sections['atomic_number'][i]] + '     ' +\
            '{:06.6f}'.format(coordinates[i][0]) + '     ' +\
            '{:06.6f}'.format(coordinates[i][1]) + '     ' +\
            '{:06.6f}'.format(coordinates[i][2]) + '\n'
        xyz_file.write(line)
    xyz_file.close()

### some analysis of the QM:MM system ###
if ONIOM:
    qm_part_charge = 0.0
    for at in qm_part:
        qm_part_charge += at.get_at_charge()
    print('\nQM part of the system (without H-link atoms) has a charge: ', "{:.2e}".format(qm_part_charge))

if read_prm2Gaussian_inp:
    if len(atoms) != NATOM:
        print('WARNING: length of atom list =', str(len(atoms)),' does not agree with NATOM = ',str(NATOM))
    
    if len(residues) != NRES:
        print('WARNING: length of residue list =', str(len(residues)),' does not agree with NRES = ',str(NRES))

    

### ---------------------------------------------------------------------- ###
### Writing the output - G16 MM or ONIOM input file                        ###

# open the file for writting
g16file = open(g16_inp_file, 'w')
empty_l = '\n'

# header section    
l1_g_input = '# Amber=(SoftOnly,Print) Geom=Connectivity \n'
l1_g_oniom_input = '# ONIOM(UB3LYP/def2SVP EmpiricalDispersion=GD3BJ:Amber=SoftFirst) Geom=Connectivity\n\
5d scf=(xqc,maxcycle=350) nosymm \n'

l3_g_input = 'comment line \n'

# calculate the total charge and round it to nearest integer value:
tot_q = np.sum(prmtop_num_sections['charge'])
tot_q_int = int(round(tot_q))
print('Total charge of the system is: ', "{:.2e}".format(tot_q), ' , which is rounded to: ', tot_q_int)
if VERBOSE:
    print(datetime.datetime.now(), "\n")

# for MM-only or real/low (ONIOM) model: assuming the total spin is singlet or dublet 
# (does not have any meaning for pure FF calculations):
nuclear_charge = 0    
if TRIM_MODEL:
    for res in residues:
        if not res.get_trim():
            for at in res.get_atoms():
                nuclear_charge += atm_number[at.get_element()]
    n_electrons = nuclear_charge - tot_q_int
else:    
    n_electrons = np.sum(prmtop_num_sections['atomic_number']) - tot_q_int

if n_electrons % 2:
    mm_mul = 2
else:
    mm_mul = 1
    
l5_g_input = str(tot_q_int) + '    ' + str(mm_mul)  + '\n'
if ONIOM:
    l5_g_oniom_input = str(tot_q_int) + '  ' + str(mm_mul) + '    ' + str(qm_chg) + '  ' + str(qm_mul) +\
    '    ' + str(qm_chg) + '  ' + str(qm_mul)  + '\n'  

# header section - writting to file
if ONIOM:
    g16file.write(l1_g_oniom_input)
else:
    g16file.write(l1_g_input)
    
g16file.write(empty_l)
g16file.write(l3_g_input)
g16file.write(empty_l)

if ONIOM:
    g16file.write(l5_g_oniom_input)
else:
    g16file.write(l5_g_input)


# coordinates section
# iterate over atoms and generate appopriate line; write the line into the file    
if (not ONIOM) and (not TRIM_MODEL) and (not FREEZE):
    for i in range(NATOM):    
        line = at_num_symbol[prmtop_num_sections['atomic_number'][i]] + '-' +\
            unq_to_NEW_types[ prmtop_text_sections['amber_atom_type'][i] ]  + '-' +\
            str(round(prmtop_num_sections['charge'][i], 6)) + '\t\t' +\
            '{:06.6f}'.format(coordinates[i][0]) + '     ' +\
            '{:06.6f}'.format(coordinates[i][1]) + '     ' +\
            '{:06.6f}'.format(coordinates[i][2]) + '\n'
        g16file.write(line)
else:
    for res in residues:
        if not res.get_trim():
            for at in res.get_atoms():
                el = at.get_element()
                tp = at.get_new_type()
                chg = at.get_at_charge()
                fzn = at.get_frozen()
                cord = at.get_coords()
                line = el + '-' + tp + '-' + str(round(chg, 6)) + '\t' + str(fzn) + '\t\t' +\
                '{:06.6f}'.format(cord[0]) + '     ' +\
                '{:06.6f}'.format(cord[1]) + '     ' +\
                '{:06.6f}'.format(cord[2])
                if not ONIOM:
                    line = line + '\n'
                elif ONIOM:
                    lr = at.get_oniom_layer()
                    line = line + '\t' + lr
                    if at.get_index() in lk_at_indexes:
                        lk_at = link_atoms[ lk_at_indexes.index( at.get_index() ) ]
                        el = lk_at.get_element()
                        tp = lk_at.get_new_type()
                        chg = lk_at.get_at_charge()
                        bto = lk_at.get_bonded_to() + 1 # shift from 0- to 1- based indexing
                        extra = el + '-' + tp + '-' + '{:02.6f}'.format(chg) + '\t' + str(bto)
                        line = line + '  ' + extra
                    line = line + '\n'
                g16file.write(line)

    
# write separator empty line:
g16file.write(empty_l)
if VERBOSE:
    print("coordinate section written to file ", datetime.datetime.now(), "\n")
    

# write connectivity list into the gaussian input,
# a given bond between two atoms is specified only once:
if not TRIM_MODEL:
    for i in range(NATOM):
        con_4_g16 = connect_list[i]
        for atom_idx in con_4_g16.copy():
            if atom_idx < i:
                con_4_g16.remove(atom_idx)
        g16file.write(gen_connectivity_line(i, con_4_g16))
else:
    for res in residues:
        if not res.get_trim():
            for at in res.get_atoms():
                at_ix = at.get_index()
                con_4_g16 = at.get_connect_list()
                for atom_idx in con_4_g16.copy():
                    if atom_idx < at_ix:
                        con_4_g16.remove(atom_idx)
                new_con_4_g16 = []
                for atom_idx in con_4_g16:
                    try:
                        new_con_4_g16.append( old_new_at_ix[atom_idx] )
                    except KeyError:
                        print("\n Warning: bond between these atoms is not preserved: ", at_ix, atom_idx)
                g16file.write(gen_connectivity_line(at.get_new_index(), new_con_4_g16))
                

# write separator empty line:
g16file.write(empty_l)
if VERBOSE:
    print("connect_list written to file ", datetime.datetime.now(), "\n")

# FF specification section

# vdW parameters
sep = '     '
for type in unique_types:
    new_type = unq_to_NEW_types[type]
    vdw_line = 'VDW' + sep + new_type + sep + '{:06.6f}'.format(r_min[type]) + sep + '{:06.6f}'.format(epsilon[type]) + '\n'
    g16file.write(vdw_line)
        
if VERBOSE:
    print("vdW section written to file ", datetime.datetime.now(), "\n")

# force constant is not changed, as amber and Gaussian use the same format ( K*(r - req)**2 ) 
for index, raw in bonds.loc[bonds['first_instance']].iterrows():
    at_1 = unq_to_NEW_types[ raw['Atom_types'][0] ]
    at_2 = unq_to_NEW_types[ raw['Atom_types'][1] ]
    bond_line = 'HrmStr1' + sep + at_1 + sep + at_2+ sep +\
        '{:05.3f}'.format(round(raw['force_constant'], 3)) + sep +\
        '{:05.3f}'.format(round(raw['equil_value'], 3)) + '\n'    
    g16file.write(bond_line)
if VERBOSE:
    print("bond parameters written to file ", datetime.datetime.now(), "\n")

# angle parameters
# force constant is not changed, as amber and Gaussian use the same format ( K*(theta - thetaeq)**2 )    
for index, raw in angles.loc[angles['first_instance']].iterrows():
    at_1 = unq_to_NEW_types[ raw['Atom_types'][0] ]
    at_2 = unq_to_NEW_types[ raw['Atom_types'][1] ]
    at_3 = unq_to_NEW_types[ raw['Atom_types'][2] ]
    angle_line = 'HrmBnd1' + sep + at_1 + sep + at_2 + sep + at_3 + sep +\
        '{:05.3f}'.format(round(raw['force_constant'], 3)) + sep +\
        '{:05.3f}'.format(round(raw['equil_value'], 3)) + '\n'
    g16file.write(angle_line)
if VERBOSE:
    print("angle parameters written to file ", datetime.datetime.now(), "\n")

# proper dihedral parameters
# force constant and phase_offset are not changed, as Amber and Gaussian use the same format
sep = '   ' # 3 spaces separator for lengthy dihedral and improper lines    
fstr = '{:08.6f}'
proper_lines = []
dihedral_data_non_redund_length = len(dihedral_data_non_redund)
i = 0
while i < dihedral_data_non_redund_length:
    raw = dihedral_data_non_redund[i]           
    if raw[2]:
        i += 1
    else:
        at_1 = unq_to_NEW_types[ raw[3][0] ]
        at_2 = unq_to_NEW_types[ raw[3][1] ]
        at_3 = unq_to_NEW_types[ raw[3][2] ]   
        at_4 = unq_to_NEW_types[ raw[3][3] ]
        dih_line = 'AmbTrs' + sep + at_1 + sep + at_2 + sep +\
          at_3 + sep + at_4 + sep
        PO = [0, 0, 0, 0]
        Mag = [0.0, 0.0, 0.0, 0.0]
        di = 0
        for j in range(4):
            if (i+j) < dihedral_data_non_redund_length:
                raw_2 = dihedral_data_non_redund[i+j]
                if raw[3] == raw_2[3]:
                    di += 1
                    periodicity = round(raw_2[5]) # periodicity can be 1, 2, 3 or 4
                    # check if periodicity is one of: 1, 2, 3 or 4
                    if periodicity in (1, 2, 3, 4):
                        po = int(round((raw_2[6]))) # Gaussian expects PO as integer !!!
                        mag = round((raw_2[4]), 6) # round to 6 digits for better precision
                        PO[periodicity-1] = po
                        Mag[periodicity-1] = mag
                    elif periodicity > 4:
                        w_line = raw[3][0] + sep + raw[3][1] + sep +\
                        raw[3][2] + sep + raw[3][3]
                        print('Warning: dihedral potential for: ', w_line, ' has periodicity > 4')
                        print('which will be ignored \n')
        i = i + di
        for k in range(4):
            dih_line = dih_line + str(PO[k]) + sep
        for l in range(4):       
            dih_line = dih_line + fstr.format(Mag[l]) + sep
        dih_line = dih_line + '1.0 \n' # NPaths set to 1 (Gaussian wants it as float)
        proper_lines.append(dih_line) 
    
for line in proper_lines:
    g16file.write(line)
if VERBOSE:
    print("proper dihedral parameters written to file ", datetime.datetime.now(), "\n")

# improper parameters
# force constant and phase_offset unchanged
for raw in improper_data_non_redund:
    if raw[2]:    
        at_1 = unq_to_NEW_types[ raw[3][0] ]
        at_2 = unq_to_NEW_types[ raw[3][1] ]
        at_3 = unq_to_NEW_types[ raw[3][2] ]   
        at_4 = unq_to_NEW_types[ raw[3][3] ]
        improper_line = 'ImpTrs' + sep + at_1 + sep + at_2 + sep +\
            at_3 + sep + at_4 + sep +\
            fstr.format(round(raw[4], 3)) + sep +\
            str(round((raw[6]), 1)) + sep +\
            str(round(raw[5], 1)) + '\n'
        g16file.write(improper_line)
if VERBOSE:
    print("improper dihedral parameters written to file ", datetime.datetime.now(), "\n")

# general settings
# Non-bonded interaction master function (standard amber FF)
non_bon_line ='NonBon 3 1 0 0 0.0 0.0 0.5 0.0 0.0 -1.2 \n'
g16file.write(non_bon_line)    

# flags for units used to specify FF    
units_lines = 'StrUnit 0 \n' + 'BndUnit 0 \n' + 'TorUnit 0 \n' + 'OOPUnit 0 \n'
g16file.write(units_lines)

# write final empty line:
g16file.write(empty_l)

# close the output (gaussian input) file
g16file.close()
if VERBOSE:
    print("Happy landing at: ", datetime.datetime.now(), "\n")
