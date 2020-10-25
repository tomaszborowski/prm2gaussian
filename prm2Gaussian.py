#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This is the main code for the script that is supposed
to read Amber prmtop and prmcrd(rst7) files and write
Gaussian16 Amber=softonly input file

The script expects 3 file names as command line arguments:
    #1 prmtop file
    #2 prmcrd(rst7) file
    #3 output G16 input file

Warnings:
1. bond order is uniformly 1.0 for all bonded atom pairs in the connectivity section    

2. 1-4 interactions are all the same (as given by the NonBon line); at the moment
the script does not use the information stored in the scee_scale_factor and scnb_scale_factor
(might be useful to explore this for systems where not all 1-4 interactions are treated 
in the same way, but currently in G09 and G16 Revision C.01 the flags nbdir/nbterm are
not accepted in input)

3. total spin is calculated as either 1 or 2, depending on the total charge and atomic
composition

4. dihedral term accepts periodicity from 1 to 4 (as in standard Amber and GAFF FF), 
higher values are ignored (warning is printed)

5. if atomic number read from prmtop is smaller than 1, it is corrected based on
the mass 
    
Created on Fri Oct  9 10:27:30 2020

@author: borowski, wojdyla
"""
import sys
import numpy as np
import pandas as pd

from read_prmtop import prmtop_read_pointers, prmtop_read_text_section
from read_prmtop import prmtop_read_numeric_section, crd_read_coordinates
from read_prmtop import atm_mass, atm_number, at_num_symbol
from read_prmtop import LEGIT_TEXT_FLAGS, LEGIT_NUM_FLAGS

from connectivity_list import cov_radii, coord_diffs, diff_dist, lista_to_line

### ---------------------------------------------------------------------- ###
### Seting the file names                                                  ###
prmtop_file = sys.argv[1]
prmcrd_file = sys.argv[2]
g16_inp_file = sys.argv[3]

# # dla testow:
# prmtop_file = './uklady_testowe/gly.prmtop'
# prmcrd_file = './uklady_testowe/gly.prmcrd'
# g16_inp_file = './uklady_testowe/gly.com'

# # dla testow:
# prmtop_file = './uklady_testowe/truncated_site3.prmtop'
# prmcrd_file = './uklady_testowe/truncated_site3.prmcrd'
# g16_inp_file = './uklady_testowe/truncated_site3.com'

### ---------------------------------------------------------------------- ###
### Reading from prmtop file                                               ###

# open the prmtop file and read-in pointers from the prmtop file:
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

### ---------------------------------------------------------------------- ###
### Reading from prmcrd / rst7 file                                        ###

# read coordinates from the prmcrd (rst7) file:
prmcrd = open(prmcrd_file, 'r')
natom_crd, coordinates = crd_read_coordinates(prmcrd)
prmcrd.close()

# check if the number of atoms matches that read from prmtop file:
if natom_crd != NATOM:
    print("ATTANTION, the number of atoms read from prmtop and prmcrd(rst7) files DO NOT MATCH !!!\n")
    print("from prmtop: ", NATOM, " \n")
    print("from prmcrd: ", natom_crd, " \n")
    

### ---------------------------------------------------------------------- ###
### Processing the data                                                    ###

# converting atomic charges to atomic units:
for i in range(len(prmtop_num_sections['charge'])):
    prmtop_num_sections['charge'][i] /= 18.2223

# angle force constants are read in from prmtop file in [kcal mol-1 rad-2]
# which is OK for G16 HrmBnd1
# angle eq values are read-in in [rad], whereas in G16 they must be given in deg
def rad_to_deg(angle):
    """
    converts angle from radians to degrees
    returns values in the 0 ... 360 range
    """
    return np.mod((180.0 * angle)/np.pi,360.0)

# converting angle equilibrium values from radians to degrees:
for i in range(len(prmtop_num_sections['angle_equil_value'])):
    prmtop_num_sections['angle_equil_value'][i] = \
    rad_to_deg(prmtop_num_sections['angle_equil_value'][i])
    
# dihedral phase is read-in in [rad], convert to degrees:
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
            

# bonds wczytaj do Pandas DataFrame o nazwie bonds
# first_instance jest flaga oznaczajaca pierwsze wystapienie danego typu wyrazu (tylko numery
# atomow ominiete przy porownywaniu rekordow)
columns = ['Atom_numbers', 'ind_to_tabels', 'Atom_types',\
           'force_constant', 'equil_value', 'first_instance']    
# pozbieraj odpowiednie info w kolumnach pomocniczej listy bonds_data:
bonds_data = []
for bond_type in ['bonds_inc_hydrogen', 'bonds_without_hydrogen']:
    for i in range(int(round(len(prmtop_num_sections[bond_type])/3))):
        temp = prmtop_num_sections[bond_type][3*i : 3*i+3]
        atom_numbers = [int(round(n/3)) for n in temp[0:2]]
        ind_to_tabels = temp[-1]        
        atom_types = [prmtop_text_sections['amber_atom_type'][i] for i in atom_numbers]
        force_constant = prmtop_num_sections['bond_force_constant'][ind_to_tabels - 1]
        equil_value = prmtop_num_sections['bond_equil_value'][ind_to_tabels - 1]
        first_instance = True
        raw = [atom_numbers, ind_to_tabels, atom_types, force_constant, equil_value, first_instance]
        for rzad in bonds_data:
            if rzad[1:] == raw[1:] :
                first_instance = False
                raw[-1] = first_instance
                break
        bonds_data.append(raw)
        
# utwórz DataFrame z pomocniczej listy    
bonds = pd.DataFrame(bonds_data, columns=columns)   



# katy wczytaj do Pandas DataFrame o nazwie angles
# first_instance jest flaga oznaczajaca pierwsze wystapienie danego typu wyrazu (tylko numery
# atomow ominiete przy porownywaniu rekordow)
columns = ['Atom_numbers', 'ind_to_tabels', 'Atom_types',\
           'force_constant', 'equil_value', 'first_instance']    
# pozbieraj odpowiednie info w kolumnach pomocniczej listy angles_data:
angles_data = []
for angle_type in ['angles_inc_hydrogen', 'angles_without_hydrogen']:
    for i in range(int(round(len(prmtop_num_sections[angle_type])/4))):
        temp = prmtop_num_sections[angle_type][4*i : 4*i+4]
        atom_numbers = [int(round(n/3)) for n in temp[0:3]]
        ind_to_tabels = temp[-1]        
        atom_types = [prmtop_text_sections['amber_atom_type'][i] for i in atom_numbers]
        force_constant = prmtop_num_sections['angle_force_constant'][ind_to_tabels - 1]
        equil_value = prmtop_num_sections['angle_equil_value'][ind_to_tabels - 1]
        first_instance = True
        raw = [atom_numbers, ind_to_tabels, atom_types, force_constant, equil_value, first_instance]
        for rzad in angles_data:
            if rzad[1:] == raw[1:] :
                first_instance = False
                raw[-1] = first_instance
                break
        angles_data.append(raw)
        
# utwórz DataFrame z pomocniczej listy    
angles = pd.DataFrame(angles_data, columns=columns)



# dihedrale wczytaj do Pandas DataFrame o nazwie dihedrals
# first_instance jest flaga oznaczajaca pierwsze wystapienie danego typu wyrazu (tylko numery
# atomow ominiete przy porownywaniu rekordow)
columns = ['Atom_numbers', 'ind_to_tabels', 'if_nonbonded', 'is_improper', 'Atom_types',\
           'force_constant', 'periodicity', 'phase', 'scee_scale_factor', 'scnb_scale_factor',
           'first_instance']    
# pozbieraj odpowiednie info w kolumnach pomocniczej listy dihedral_data:
dihedral_data = []
for dih_type in ['dihedrals_inc_hydrogen', 'dihedrals_without_hydrogen']:
    for i in range(int(len(prmtop_num_sections[dih_type])/5)):
        temp = prmtop_num_sections[dih_type][5*i : 5*i+5]
        atom_numbers = [int(np.abs(n)/3) for n in temp[0:4]]
        ind_to_tabels = temp[-1]
        if temp[2] < 0:
            if_nonbonded = False
        else:
            if_nonbonded = True
        if temp[3] < 0:
            is_improper = True
        else:
            is_improper = False
        atom_types = [prmtop_text_sections['amber_atom_type'][i] for i in atom_numbers]
        force_constant = prmtop_num_sections['dihedral_force_constant'][ind_to_tabels - 1]
        periodicity = prmtop_num_sections['dihedral_periodicity'][ind_to_tabels - 1]
        phase = prmtop_num_sections['dihedral_phase'][ind_to_tabels - 1]
        scee_scale_factor = prmtop_num_sections['scee_scale_factor'][ind_to_tabels - 1]
        scnb_scale_factor = prmtop_num_sections['scnb_scale_factor'][ind_to_tabels - 1]
        first_instance = True
        raw = [atom_numbers, ind_to_tabels, if_nonbonded, is_improper, atom_types,\
               force_constant, periodicity, phase, scee_scale_factor, scnb_scale_factor, first_instance]
        for rzad in dihedral_data:
            if rzad[1:] == raw[1:] :
                first_instance = False
                raw[-1] = first_instance
                break
        dihedral_data.append(raw)
        
# utwórz DataFrame z pomocniczej listy    
dihedrals = pd.DataFrame(dihedral_data, columns=columns)    
            

# extracting atomic vdW parameters:

# z prmtop_text_sections['amber_atom_type'] wybrać zestaw unikalnych typów atomów
unique_types = set(prmtop_text_sections['amber_atom_type'])    
# dla kazdego z tych unikalnych typow:   
    # wyznacz indeks atomu tego typu i wylicz pointer do lennard_jones_a(b)coef dla pary ten typ - ten typ
    # odczytaj A i B i przelicz na epsilon i r_m
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
    # convert in epsilon and r_min values:
    if A > 0.0 and B > 0.0:
        eps = B**2/(4*A)
        r_m = 0.5 * ((2*A/B)**(1/6))
    else:
        eps=0.0
        r_m=0.0
    # add the values to the dictionaries:
    epsilon[type] = eps
    r_min[type] = r_m
# 
    

### ---------------------------------------------------------------------- ###
### Writing the output - G16 input file                                    ###

# open the file for writting
g16file = open(g16_inp_file, 'w')
empty_l = '\n'


# header section    
l1_g_input = '# Amber=(SoftOnly,Print) Geom=Connectivity \n'
l3_g_input = 'comment line \n'

# calculate the total charge and round it to nearest integer value:
tot_q = np.sum(prmtop_num_sections['charge'])
tot_q_int = int(round(tot_q))
print('Total charge of the system is: ', tot_q, ' , which is rounded to: ', tot_q_int)

# assuming the total spin is singlet or dublet (does not have any meaning for pure FF calculations)
n_electrons = np.sum(prmtop_num_sections['atomic_number']) - tot_q_int
if n_electrons % 2:
    S = 2
else:
    S = 1
    
l5_g_input = str(tot_q_int) + '    ' + str(S)  + '\n'

# header section - writting to file
g16file.write(l1_g_input)
g16file.write(empty_l)
g16file.write(l3_g_input)
g16file.write(empty_l)
g16file.write(l5_g_input)


# coordinates section
# iterate over atoms and generate appopriate line; write the line into the file    
for i in range(NATOM):    
    line = at_num_symbol[prmtop_num_sections['atomic_number'][i]] + '-' +\
        prmtop_text_sections['amber_atom_type'][i] + '-' +\
        str(round(prmtop_num_sections['charge'][i], 6)) + '\t\t' +\
        '{:06.6f}'.format(coordinates[i][0]) + '     ' +\
        '{:06.6f}'.format(coordinates[i][1]) + '     ' +\
        '{:06.6f}'.format(coordinates[i][2]) + '\n'
    g16file.write(line)
    
# write final empty line:
g16file.write(empty_l)

    
# connectivity list section

# loop over all atom pairs and determine if they are bonded or not:
bond_radii_scale_factor = 1.3 # 130% of (R_i + R_j) as a threshold
for i in range(NATOM):
    partners = []
    for j in range(i+1,NATOM):
        R = cov_radii[at_num_symbol[prmtop_num_sections['atomic_number'][i]]] +\
            cov_radii[at_num_symbol[prmtop_num_sections['atomic_number'][j]]]
        Rs = bond_radii_scale_factor * R
        D = coord_diffs(coordinates[i][0], coordinates[i][1], coordinates[i][2],\
                        coordinates[j][0], coordinates[j][1], coordinates[j][2])
        dist = 100.0
        if max(D) <= Rs:
            dist = diff_dist(D[0], D[1], D[2])
        if dist <= Rs:
            partners.append(j)
    g16file.write(lista_to_line(i,partners))

# write final empty line:
g16file.write(empty_l)


# FF specification section

# vdW parameters
sep = '     '
for type in unique_types:
    vdw_line = 'VDW' + sep + type + sep + '{:06.6f}'.format(r_min[type]) + sep + '{:06.6f}'.format(epsilon[type]) + '\n'
    g16file.write(vdw_line)


# force constant is not changed, as amber and Gaussian use the same format ( K*(r - req)**2 ) 
for index, raw in bonds.loc[bonds['first_instance']].iterrows():
    bond_line = 'HrmStr1' + sep + raw['Atom_types'][0] + sep + raw['Atom_types'][1] + sep +\
        '{:05.3f}'.format(round(raw['force_constant'], 3)) + sep +\
        '{:05.3f}'.format(round(raw['equil_value'], 3)) + '\n'
    g16file.write(bond_line)


# angle parameters
# force constant is not changed, as amber and Gaussian use the same format ( K*(theta - thetaeq)**2 )    
for index, raw in angles.loc[angles['first_instance']].iterrows():
    angle_line = 'HrmBnd1' + sep + raw['Atom_types'][0] + sep + raw['Atom_types'][1] + sep +\
        raw['Atom_types'][2] + sep +\
        '{:05.3f}'.format(round(raw['force_constant'], 3)) + sep +\
        '{:05.3f}'.format(round(raw['equil_value'], 3)) + '\n'
    g16file.write(angle_line)


# proper dihedral parameters
# force constant and phase_offset are not changed, as Amber and Gaussian use the same format
sep = '   ' # 3 spaces separator for lengthy dihedral and improper lines    
fstr = '{:05.3f}'
proper_lines = []
proper_unique = dihedrals.loc[~dihedrals['is_improper'] & dihedrals['first_instance']]
for index, raw in proper_unique.iterrows():
    dih_line = 'AmbTrs' + sep + raw['Atom_types'][0] + sep + raw['Atom_types'][1] + sep +\
        raw['Atom_types'][2] + sep + raw['Atom_types'][3] + sep    
    PO = [0, 0, 0, 0]
    Mag = [0.0, 0.0, 0.0, 0.0]
    for index2, raw_2 in proper_unique.iterrows():
        if raw['Atom_types'] == raw_2['Atom_types']:
            i = round(raw_2['periodicity']) # i can be 1, 2, 3 or 4
            # check if periodicity is one of: 1, 2, 3 or 4
            if i in (1, 2, 3, 4):
                po = int(round((raw_2['phase']))) # Gaussian expects PO as integer !!!
                mag = round((raw_2['force_constant']), 3)
                PO[i-1] = po
                Mag[i-1] = mag
            elif i > 4:
                w_line = raw['Atom_types'][0] + sep + raw['Atom_types'][1] + sep +\
                    raw['Atom_types'][2] + sep + raw['Atom_types'][3]
                print('Warning: dihedral potential for: ', w_line, ' has periodicity > 4')
                print('which will be ignored \n')
    for i in range(4):
        dih_line = dih_line + str(PO[i]) + sep
    for i in range(4):       
        dih_line = dih_line + fstr.format(Mag[i]) + sep
    dih_line = dih_line + '1.0 \n' # NPaths set to 1 (Gaussian wants it as float)
    proper_lines.append(dih_line)
#    
proper_lines_unique = set(proper_lines)
for line in proper_lines_unique:
    g16file.write(line)


# improper parameters
# force constant and phase_offset unchanged 
for index, raw in dihedrals.loc[dihedrals['is_improper'] & dihedrals['first_instance']].iterrows():
    improper_line = 'ImpTrs' + sep + raw['Atom_types'][0] + sep + raw['Atom_types'][1] + sep +\
        raw['Atom_types'][2] + sep + raw['Atom_types'][3] + sep +\
        fstr.format(round(raw['force_constant'], 3)) + sep +\
        str(round((raw['phase']), 1)) + sep +\
        str(round(raw['periodicity'], 1)) + '\n'
    g16file.write(improper_line)


# general settings
# Non-bonded interaction master function (standard amber FF)
non_bon_line ='NonBon 3 1 0 0 0.0 0.0 0.5 0.0 0.0 -1.2 \n'
g16file.write(non_bon_line)    

# flags for units used to specify FF    
units_lines = 'StrUnit 0 \n' + 'BndUnit 0 \n' + 'TorUnit 0 \n' + 'OOPUnit 0 \n'
g16file.write(units_lines)

# write final empty line:
g16file.write(empty_l)

# close the file
g16file.close()
