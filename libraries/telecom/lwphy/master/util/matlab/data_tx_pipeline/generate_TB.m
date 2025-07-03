%%
 % Copyright (c) 2020, LWPU CORPORATION.  All rights reserved.
 %
 % LWPU CORPORATION and its licensors retain all intellectual property
 % and proprietary rights in and to this software, related documentation
 % and any modifications thereto.  Any use, reproduction, disclosure or
 % distribution of this software and related documentation without an express
 % license agreement from LWPU CORPORATION is strictly prohibited.
 %%


function bits = generate_TB(iue,PxschCfg,sp)

%function generates a random transport block

%%
%PARAMATERS

TBS = PxschCfg.coding.TBS;     % size of transport block
model = sp.sim.channel.model;  % dhannel model: 'uniform_reflectors', 'siso-awgn', or 'capture'

%%
%START

if strcmp(model,'capture')
    load(sp.sim.capture_file);

    bits = pusch_cfg{iue}.matlab_spec.true_TB_bits{1};
    bits = double(bits);
else
    %Generate one Transport Block of random bits
    bits = randi([0 1], [TBS 1]);    % Generate random bits
% ends

end

