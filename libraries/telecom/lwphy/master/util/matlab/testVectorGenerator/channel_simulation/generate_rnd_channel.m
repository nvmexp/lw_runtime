 %%
 % Copyright (c) 2020, LWPU CORPORATION.  All rights reserved.
 %
 % LWPU CORPORATION and its licensors retain all intellectual property
 % and proprietary rights in and to this software, related documentation
 % and any modifications thereto.  Any use, reproduction, disclosure or
 % distribution of this software and related documentation without an express
 % license agreement from LWPU CORPORATION is strictly prohibited.
 %%


function H_cell = generate_rnd_channel(sp)

%function generates a random time-frequency channel

%outputs:
%H_cell --> cell containing TF channel of all users. Dim: numUes x 1

%each cell entry contains H, an array of dimension: L_BS x nl x Nf x Nt 

%%
%PARAMATERS

%numerology:
Nf = sp.gnb.numerology.Nf;     %total number of subcarriers 
Nt = sp.gnb.numerology.Nt;     %total number of OFDM symbols
df = sp.gnb.numerology.df;     %subcarrier spacing (Hz)
dt = sp.gnb.numerology.dt;     %OFDM symbols duration (s)
L_BS = sp.gnb.numerology.L_BS; %total number of bs antennas

%channel:
numReflectors = sp.sim.channel.numReflectors;  % Number of reflectors
delaySpread = sp.sim.channel.delaySpread;      % Delay spread (seconds)
DopplerSpread = sp.sim.channel.DopplerSpread;  % Doppler spread (Hz)

%pusch paramaters:
if strcmp(sp.sim.opt.simType,'pusch')
    numUes = sp.gnb.pusch.numUes;               % Total number of uplink users
    PxschCfg_cell = sp.gnb.pusch.PuschCfg_cell; % cell, contains uplink configurations of all users. 
elseif strcmp(sp.sim.opt.simType,'pdsch')
    numUes = sp.gnb.pdsch.numUes;               % Total number of uplink users
    PxschCfg_cell = sp.gnb.pdsch.PdschCfg_cell; % cell, contains uplink configurations of all users. 
end


%%
%SETUP

f = 0 : (Nf - 1);
f = f.' * df;

t = 0 : (Nt - 1);
t = t.' * dt;

%%
%START

H_cell = cell(numUes,1);

for iue = 1 : numUes
    
    %user configuration:
    PxschCfg = PxschCfg_cell{iue};
    nl = PxschCfg.mimo.nl;            %number of mimo layers
    
    %generte rnd channel for user:
    H = zeros(Nf,Nt,L_BS,nl);
   
    for jbs = 1 : L_BS
        for layer = 1 : nl
   
            %generate rnd taps:
            tau = delaySpread * rand(numReflectors,1);
            nu = DopplerSpread * (rand(numReflectors,1) - 0.5);
            a = sqrt(1 / (2 * numReflectors)) * (randn(numReflectors,1) + 1i*randn(numReflectors,1));
            
            %buid TF channel:
            for r = 1 : numReflectors
                H(:,:,jbs,layer) = H(:,:,jbs,layer) + ...
                    a(r) * exp(-2*pi*1i*tau(r)*f) * exp(2*pi*1i*nu(r)*t).';
            end
            
        end
    end
    
    %wrap:
    H_cell{iue} = permute(H,[3 4 1 2]);
    
end


end

            
            
            
            
                


















