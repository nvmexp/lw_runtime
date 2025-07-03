#/bin/bash

# CAUTION 
# run this from src/heur directory

for hf in $(ls *.cpp); do echo $hf; cp $hf $hf-bak; python ../../scripts/helpers/Tuners/tree_colwert.py ${hf%.cpp} > $hf.tmp; mv $hf.tmp $hf; done; ls -lah

