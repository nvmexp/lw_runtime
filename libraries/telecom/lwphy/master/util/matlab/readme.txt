matlab scripts to generate hdf5 test-vector for lwPHY, which are stored in the folder GPU_test_input 

Run generate_all_TC to build all test-vectors.

Alternitvally individual test-vectors can be generated.

PUSCH:
Main_lls_sdk('uplink','pusch-TC231'); for TC 231-235 and 281-285

PUCCH:
Main_lls_sdk('uplink','pucch-TC1001'); for TC 1001-1005

PDSCH:
Main_lls_sdk('pdsch','pdsch-TC201'); for TC 201-205, 261-265, 301a, and 301b

DOWNLINK CONTROL
Main_lls_sdk('dlCtrl','DL_ctrl-TC2001'); for TC 2001-2004




