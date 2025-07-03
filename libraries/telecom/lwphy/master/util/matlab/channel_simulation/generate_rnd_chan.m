%%
 % Copyright (c) 2020, LWPU CORPORATION.  All rights reserved.
 %
 % LWPU CORPORATION and its licensors retain all intellectual property
 % and proprietary rights in and to this software, related documentation
 % and any modifications thereto.  Any use, reproduction, disclosure or
 % distribution of this software and related documentation without an express
 % license agreement from LWPU CORPORATION is strictly prohibited.
 %%

function sp = generate_rnd_chan(sp)

%function generates a random TF channel using the uniform reflector model

%outputs:
%sp.sim.H --> random TF channel. Dim: L_BS x L_UE x Nf x Nt

%%
%START

sp = generate_rnd_reflectors(sp);

sp.sim.H = reflectors_to_TF_chan(sp);










