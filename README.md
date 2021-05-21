# prm2gaussian
Command line script converting Amber prmtop and prmcrd files into Gaussian input file.

Amber parameters and atom types read from the prmtop file are converted to Gaussian
input format. Atom types are mapped into a set that is compliant with Gaussian input
format (capital letters, no numbers at the beginning). Atom connectivity is read from 
prmtop file and written to the Gaussian input, all bonds read are set to have order of 1. 
Total charge of the system is calculated from atomic partial charges read from the 
prmtop file. Total spin is either read from the input file (see below) or calculated 
to be singlet or dublet, depending on the atomic composition of the system and its 
total charge. If explicit HW-HW bond is detected, adds zero force constants for 
angles in triangular water molecules. If atomic number read from prmtop is smaller 
than 1, it is corrected based on the atom mass. Dihedral term accepts periodicity 
from 1 to 4 (as in standard Amber and GAFF FF), higher values are ignored 
(warning is printed). Bond length scaling factor used to calculate the positions 
of H-link atoms is set to 0.723886.
 

The required command line arguments are:
1) name of the input prmtop file
2) name of the input prmcrd file
3) name of the output Gaussian input file

The optional command line argument is:
4) input file for this script (its content is described below)


If envoked with only the three required command line arguments the resulting Gaussian input
file will correspond to Amber MM calculations for the whole system read from the prmtop/prmcrd
files. Note that due to a subtle difference in the way improper terms are calculated by amber (sander)
and Gaussian, the improper energies computed by the two programms will differ slightly. 

###############
## An example how to use the script to generate the Gaussian MM input for the whole system:
###############

prm2Gaussian.py h6h-oxo+succinate+water_hyo.prmtop h6h-oxo+succinate+water_hyo.prmcrd h6h-oxo+succinate+water_hyo.MM.com > prm2gaussian.MM.out

###############


The input file for this script allows one to:
a) specify the QM part for ONIOM(QM:Amber) QM/MM calculations,
b) specify the hydrogen link atoms and their atom types,
c) specify how large portion of solvent (molecules other than protein) will be retained in the model,
d) partition the model into an optimized and frozen parts,
e) specify charge and multiplicity for the QM part of the QM/MM model.


The input file can contain the following sections:

A)
%qm_part
...
%end_qm_part
to specify the QM-part of the ONIOM(QM:Amber) QM/MM system

B) 
%link_atoms
...
%end_link_atoms
to specify which atoms are to be replaced by hydrogen link atoms in QM computations as part of the ONIOM
calculations

C)
%trim_ref
...
%end_trim_ref
to specify atoms with respect to which a distance-based trimming of the model (non-protein residues) 
will be done

the trimming distance XX [in A] is specified in the line:
%r_trim  XX

D)
%freeze_ref
...
index
%end_freeze_ref
to specify atoms with respect to which a distance-based partitioning of the model into optimized and frozen
parts will be done

the optimized zone distance YY [in A] is specified in the line:
%r_free YY

E)
%qm_charge X
%qm_multip Y
to specify the total charge (X) and multiplicity (Y) of the QM subsystem of the ONIOM calculations.
The same multiplicity (Y) will be used for other ONIOM sub-calculations. 


The qm_part, trim_ref and freeze_ref sections can contain lines beginning with: "residue", "sidechain" and
"index" and followed by (0-based) indexes of residues, side chains or atoms, respectively. Side chain indexes
are the same as indexes of the parent residues. All indexes specified here pertain to the original system
read from the prmtop/prmcrd files. 

When trimming or freezing, the distance-based selection works at the level of whole residues, so even if only
one atom from a given residue is within the specified distance the whole residue will be selected.


It is advised to use some molecular graphics program (e.g. VMD) to visually inspect the system saved in the prmtop/prmcrd 
files and select the required residues, side chain or atoms (for example load into VMD the selekcja_oniom.vmd file 
attached as an example).


The link_atoms section contains two lines:
index - specifying indexes of atoms to be replaced by hydrogen link atoms in QM calculations
type - specifying amber atom type for the corresponding link atoms.

This(these) type(s) has to be chosen by the user, the type corresponds to oryginal Amber atom type 
(before any mapping applied by the script).

##################################
### Example input file content ###
##################################
%qm_part
sidechain 185 242 314 317 316 187 315
residue 318
index 
%end_qm_part


%link_atoms
index 3896 2984 3022
type HC HC HC
%end_link_atoms


%trim_ref
sidechain 185 242 314 317 316 187 315
residue 318
index
%end_trim_ref
%r_trim 20


%freeze_ref
sidechain 185 242 314 317 316 187 315
residue 318
index
%end_freeze_ref
%r_free 15


%qm_charge 0
%qm_multip 5
##################################
### End of example input file  ###
##################################



###############
## An example how to use the script to generate the Gaussian ONIOM(QM:Amber) input:
###############

prm2Gaussian.py h6h-oxo+succinate+water_hyo.prmtop h6h-oxo+succinate+water_hyo.prmcrd h6h-oxo+succinate+water_hyo.com prm2gaussian.oniom.inp > prm2gaussian.oniom.out

###############


The script generates a set of xyz files, which can be useful for checking if the model partitioning is as required. 
These files are:
QM_SYSTEM.xyz
QM_PART.xyz
MM_LA.xyz
TRIMMED.xyz
FROZEN.xyz
MODEL.xyz


REQUIRED packages: numpy, pandas, scipy, re, sys, datetime, string, fortranformat

There is ABSOLUTELY NO WARRANTY; you use this software at your own risk. 

If you find bugs, please report them to: tomasz.borowski@ikifp.edu.pl or zuzanna.wojdyla@ikifp.edu.pl
