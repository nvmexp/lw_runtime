 %%
 % Copyright (c) 2020, LWPU CORPORATION.  All rights reserved.
 %
 % LWPU CORPORATION and its licensors retain all intellectual property
 % and proprietary rights in and to this software, related documentation
 % and any modifications thereto.  Any use, reproduction, disclosure or
 % distribution of this software and related documentation without an express
 % license agreement from LWPU CORPORATION is strictly prohibited.
 %%


function CT = generate_rnd_channel_taps(QL)



%function generate random channel taps.



%inputs:

%QL --> Quadriga layout class



%outputs:

%CT.no_rxant --> number of UE antennas

%CT.no_txant --> number of Hub antennas

%CT.no_path --> number of propogation paths

%CT.coeff --> antenna coeffecients for the propogation paths. Dim: L_UE x L_HUB x no_paths x Nt

%CT.delay --> delay values for the propogations paths. Dim: L_UE x L_HUB x no_paths x Nt



%%

%START



p = QL.init_builder;

p.gen_ssf_parameters;

CT = p.get_channels;



end



