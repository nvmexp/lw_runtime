%%
 % Copyright (c) 2020, LWPU CORPORATION.  All rights reserved.
 %
 % LWPU CORPORATION and its licensors retain all intellectual property
 % and proprietary rights in and to this software, related documentation
 % and any modifications thereto.  Any use, reproduction, disclosure or
 % distribution of this software and related documentation without an express
 % license agreement from LWPU CORPORATION is strictly prohibited.
 %%

function [Y,sp] = apply_channel_main(i_snr,sp,nrSlot)

%function applies a wirless channel to the signals transmited by the UEs

%inputs:
%i_snr --> index of current snr

%ouputs:
%Y     --> BS recieved signal. Dim: Nf x Nt x L_BS

%%
%PARMATARS

model = sp.sim.channel.model;  %Channel model. Options: 'uniform_reflectors','siso-awgn','capture'
simType = sp.sim.opt.simType;  %Type of PHY simulation ('uplink', 'pdsch' ...)

%%
%START


if strcmp(model,'siso-awgn')
    Y = apply_trivial_channel(sp,nrSlot);
end

if strcmp(model,'uniform_reflectors') || strcmp(model,'tdl')
    if strcmp(simType,'uplink')
        Y = apply_UL_TF_channel(sp,nrSlot);
    else
        Y = apply_DL_TF_channel(sp,nrSlot);
    end
end

if strcmp(model,'capture')
    load(sp.sim.capture_file);
    a = rx_iq_data;
    Y = a(1:sp.gnb.nPrb*12,:,:); 
    %Y = apply_trivial_channel(sp,nrSlot);
end

[Y,sp] = add_awgn(Y,i_snr,sp);



