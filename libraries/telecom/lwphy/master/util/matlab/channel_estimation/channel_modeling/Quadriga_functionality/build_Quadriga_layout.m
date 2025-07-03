 %%
 % Copyright (c) 2020, LWPU CORPORATION.  All rights reserved.
 %
 % LWPU CORPORATION and its licensors retain all intellectual property
 % and proprietary rights in and to this software, related documentation
 % and any modifications thereto.  Any use, reproduction, disclosure or
 % distribution of this software and related documentation without an express
 % license agreement from LWPU CORPORATION is strictly prohibited.
 %%




function QL = build_Quadriga_layout(PAR)



%build Quadriga layout object. (Models antenna arrays for UE and BS. Fixes the propogation evniorment.)



%outputs:

%QL --> Quadriga layout object



%%

%SIMULATION PARAMATERS



s = qd_simulation_parameters;

s.center_frequency = PAR.prop.carrier_frequency; %carrier frequency

s.sample_density = 2; %sample density (2 a good number)



%%

%LAYOUT PARAMATERS



QL = qd_layout(s); %build layout object



QL.tx_position = PAR.geo.tx_position; %location of BS antenna (x,y,z)



%build antennas for BS and UE, see class "qd_array" for details section

%2.2.2 in manual

QL.tx_array = PAR.ant.tx_array; %BS antenna array geometry

QL.rx_array = PAR.ant.rx_array; %UE antenna array



%fix the propogation elwiorment:

QL.set_scenario(PAR.prop.scenerio);











