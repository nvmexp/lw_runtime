%%
 % Copyright (c) 2020, LWPU CORPORATION.  All rights reserved.
 %
 % LWPU CORPORATION and its licensors retain all intellectual property
 % and proprietary rights in and to this software, related documentation
 % and any modifications thereto.  Any use, reproduction, disclosure or
 % distribution of this software and related documentation without an express
 % license agreement from LWPU CORPORATION is strictly prohibited.
 %%

matlab scripts to generate test-vector for lwPHY. User can modify three things:

1.) bler lwrve paramaters (snr range, number of snr grid points, number of slots per snr value)
2.) Channel model (awgn, tdl, uniform reflectors)
3.) Pucsh test case (1-8, 231-235, 281-285 defined below)

-Modify bler and channel paramaters in configure_lls_sdk.m
-Generate test vectors using the command Main_lls_sdk('uplink', 'pusch-TC1')


Test casess 1-8 share the following paramaters:

Base station antennas:  4 x 2 (horizantal x polarization)
Time allocation:        the first 10 symbols.
Dmrs configuration:     type A mapping, type 1, maxLength = 1, additonalPosition = 0
fft precoder:           no
mcsTable:               1
numUsers:               1

testCase         nPrbs      nLayers     mcs     

    1       |     272    |    4     |   28    
    2       |     272    |    2     |   28    
    3       |     272    |    1     |   28    
    4       |     272    |    4     |   12   
    5       |     272    |    2     |   12    
    6       |     272    |    1     |   12    
    7       |     48     |    2     |   28    
    8       |     64     |    2     |   12   


Test casess 231-235 benchmark 104 PRBs, 1x4 MIMO under different code rates.
The following paramaters are shared by all cases:

Base station antennas:  2 x 2 (horizantal x polarization)
Time allocation:        the first 10 symbols.
Dmrs configuration:     type A mapping, type 1, maxLength = 1, additonalPosition = 0
fft precoder:           no
mcsTable:               2

testCase         nPrbs      nLayers     mcs     mcsTable

    231      |     104    |    1     |   27    |   2
    232      |     104    |    1     |   23    |   2
    233      |     104    |    1     |   14    |   2
    234      |     104    |    1     |   11    |   2
    235      |     104    |    1     |   5     |   2


Test casess 281-285 benchmark 272 PRBs, 8x16 MIMO under different code rates.
The following paramaters are shared by all cases:

Base station antennas:  8 x 2 (horizantal x polarization)
Time allocation:        the first 10 symbols.
Dmrs configuration:     type A mapping, type 1, maxLength = 2, additonalPosition = 1
fft precoder:           no
mcsTable:               2
numUsers:               2, with 4 layers per user


testCase         nPrbs      nLayers     mcs     mcsTable

    281      |     272    |    8     |   27    |   2
    282      |     272    |    8     |   23    |   2
    283      |     272    |    8     |   14    |   2
    284      |     272    |    8     |   11    |   2
    285      |     272    |    8     |   5     |   2



