%%
 % Copyright (c) 2020, LWPU CORPORATION.  All rights reserved.
 %
 % LWPU CORPORATION and its licensors retain all intellectual property
 % and proprietary rights in and to this software, related documentation
 % and any modifications thereto.  Any use, reproduction, disclosure or
 % distribution of this software and related documentation without an express
 % license agreement from LWPU CORPORATION is strictly prohibited.
 %%

function [Y,sp] = add_awgn(Y,i_snr,sp)

%function adds white noise to the BS recieved signal.

%inputs/outputs:
%Y --> BS recieved signal. Dim: Nf x Nt x L_BS

%%
%PARAMATERS

model = sp.sim.channel.model;           % Channel model. Options: 'uniform_reflectors','siso-awgn','capture'
lwrrentSnr = sp.sim.channel.lwrrentSnr; % input  snr (dB)
N0 = 10^(-lwrrentSnr / 10);             % noise variance (linear)
snr = sp.sim.channel.snr;               % snr steps (dB). Dim nSnrSteps x 1

%%
%SEUTP

if strcmp(model,'capture')
    N0_capture = sp.sim.channel.N0_capture; % noise variance of capture (linear)

    if N0 <= N0_capture
        N0_add = 0;
        lwrrentSnr = -10*log10(N0_capture);
        snr(i_snr) = lwrrentSnr;
    else
        N0_add = N0 - N0_capture;
    end
    
else
    N0_add = N0;
end

%%
%START

Y = Y + sqrt(N0_add / 2) * (randn(size(Y)) + 1i*randn(size(Y)));

%%
%WRAP

sp.sim.channel.lwrrentSnr = lwrrentSnr;
sp.sim.channel.snr = snr;

end


