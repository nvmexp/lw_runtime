
# This script generates compare.lwbin and gpuburn.ptx
# gpuburn.ptx is then colwerted to gpuburn_ptx_string.h as a hexified string
# Last, symbol names are added to gpuburn_ptx_string.h by find_ptx_symbols.py
#
# sm_30 is used here for Kepler or newer

/usr/local/lwca/bin/lwcc -arch=sm_30 -ptx -keep compare.lw
bin2c compare.ptx --padd 0 --name gpuburn_ptx_string > gpuburn_ptx_string.h
python find_ptx_symbols.py compare.ptx gpuburn_ptx_string.h
