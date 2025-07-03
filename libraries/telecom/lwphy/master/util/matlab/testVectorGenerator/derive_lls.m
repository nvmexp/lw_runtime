 %%
 % Copyright (c) 2020, LWPU CORPORATION.  All rights reserved.
 %
 % LWPU CORPORATION and its licensors retain all intellectual property
 % and proprietary rights in and to this software, related documentation
 % and any modifications thereto.  Any use, reproduction, disclosure or
 % distribution of this software and related documentation without an express
 % license agreement from LWPU CORPORATION is strictly prohibited.
 %%


function sp = derive_lls(sp)

%function takes high level lls paramaters and derives lower-level
%paramaters


%%
%CHANNEL

% build snr steps:
snrStart = sp.sim.channel.snrStart;                       % starting snr value
snrEnd = sp.sim.channel.snrEnd;                           % ending snr value
nSnrSteps = sp.sim.channel.nSnrSteps;                     % number of snr steps
sp.sim.channel.snr = linspace(snrStart,snrEnd,nSnrSteps); % snr steps (dB). Dim nSnrSteps x 1

% generate random TF channel:
model = sp.sim.channel.model;

switch(model)
    case 'uniform_reflectors'
        [H_data_cell,H_ctrl_cell] = generate_rnd_channel(sp);   
    case 'tdl'
        [H_data_cell,H_ctrl_cell] = generate_tdl_channel(sp);
end

if strcmp(model,'tdl') || strcmp(model,'uniform_reflectors')
    sp.sim.channel.H_data_cell = H_data_cell;
    sp.sim.channel.H_ctrl_cell = H_ctrl_cell;
end
            

end







