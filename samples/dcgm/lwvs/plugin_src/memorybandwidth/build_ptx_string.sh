
#This script generates bandwidth_calc.lwbin and bandwidth_calc.ptx
#Then, bandwidth_calc.ptx is colwerted to bandwidth_calc_ptx_string.h as a hexified string
#
#Finally, symbol names are added to bandwidth_calc_tx_string.h by find_ptx_symbols.py
#
# sm_30 is used here for Kepler or newer.

/usr/local/lwca/bin/lwcc -arch=sm_30 -ptx -keep bandwidth_calc.lw
bin2c bandwidth_calc.ptx --padd 0 --name bandwidth_calc_ptx_string > bandwidth_calc_ptx_string.h
python find_ptx_symbols.py
