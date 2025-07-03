%%
 % Copyright (c) 2020, LWPU CORPORATION.  All rights reserved.
 %
 % LWPU CORPORATION and its licensors retain all intellectual property
 % and proprietary rights in and to this software, related documentation
 % and any modifications thereto.  Any use, reproduction, disclosure or
 % distribution of this software and related documentation without an express
 % license agreement from LWPU CORPORATION is strictly prohibited.
 %%

function [H_data_cell, H_ctrl_cell] = generate_tdl_channel(sp)

% Function generates a tdl time-frequency channel. All users get the same tdl
% model, however each tap gets a random spatial signal. 

%Assumptions:
% cross-polarization antennas for users and gnb.
% channel constant across the slot

%outputs:
% H_data_cell --> cell containing TF channel of all data users. Dim: numUes_data x 1
% H_ctrl_cell --> cell containing TF channel of all control users. Dim: numUes_ctrl x 1

% each cell entry contains H, an array of dimension: L_BS x nl x Nf x Nt 

%%
%PARAMATERS

%numerology:
Nf = sp.gnb.numerology.Nf;     % total number of subcarriers 
Nt = sp.gnb.numerology.Nt;     % total number of OFDM symbols
df = sp.gnb.numerology.df;     % subcarrier spacing (Hz)
dt = sp.gnb.numerology.dt;     % OFDM symbols duration (s)
L_BS = sp.gnb.numerology.L_BS; % total number of bs antennas


%channel:
mode = sp.sim.channel.mode;                % choice of tdl model, options: (a,b,c,d,e,f)
antCorr_ue = sp.sim.channel.antCorr_ue;    % user antenna correlation. Options: low','med','high'
antCorr_gnb = sp.sim.channel.antCorr_gnb;  % gnb antenna correlation. Options: 'low','med','high'
norm_taps = sp.sim.channel.norm_taps;      % 0 or 1. Option to normalizes taps to unit energy
ds = sp.sim.channel.ds;                    % desired delay spread (ns).

%pusch paramaters:
if strcmp(sp.sim.opt.simType,'uplink')
    numUes_data = sp.gnb.pusch.numUes;            % Total number of pusch users
    numUes_ctrl = sp.gnb.pucch.numUes;            % Total number of pucch users
    PxschCfg_cell = sp.gnb.pusch.PuschCfg_cell;   % cell, contains uplink configurations of all users. 
elseif strcmp(sp.sim.opt.simType,'pdsch')
    numUes_data = sp.gnb.pdsch.numUes;             % Total number of pdsch users
    numUes_ctrl = 0;
    PxschCfg_cell = sp.gnb.pdsch.PdschCfg_cell;    % cell, contains uplink configurations of all users. 
end


%%
%SETUP

f = 0 : (Nf - 1);
f = f.' * df;

t = 0 : (Nt - 1);
t = t.' * dt;

%load tdl tap paramaters:
switch mode
    case('a')
        load('tdl_a.mat');
    case('b')
        load('tdl_b.mat');
    case('c')
        load('tdl_c.mat');
    case('d')
        load('tdl_d.mat');
    case('e')
        load('tdl_e.mat');
end

nTaps = length(T);
E = 10.^(T(:,3) / 10); % tap energy
tau = T(:,2) * ds * 10^(-9);     % tap delays

%normalize taps:
if norm_taps
    E = E / sum(E);
end


%%
%DATA

H_data_cell = cell(numUes_data,1);

for iue = 1 : numUes_data
    
    % extract user pusch cfg:
    PuschCfg = PxschCfg_cell{iue};
    L_UE = PuschCfg.mimo.nl;
    
    % build channel:
    H = zeros(L_BS,L_UE,Nf);
    
    for t = 1 : nTaps
        
        % generate tap frequency response:
        Ht_f = exp(-2*pi*1i*f*tau(t));
        
        % generate tap spatial response:
        Ht_c = generate_rnd_sr(L_BS,L_UE,antCorr_ue,antCorr_gnb);
        Ht_s = zeros(L_BS,L_UE);

        for i = 1 : ceil(L_BS/2)
            index_i = 2*(i-1) + 1 : 2*i;
            
            for j = 1 : ceil(L_UE/2)
                index_j = 2*(j-1) + 1 : 2*j;
                
                Px = generate_pol_mtx(xpr);
                Ht_s(index_i,index_j) = Ht_c(i,j) * Px;
            end
        end
        
        if L_UE == 1
            idx_ue = 1;
        else
            idx_ue = 1 : L_UE;
        end
        
        if L_BS == 1
            idx_bs = 1;
        else
            idx_bs = 1 : L_BS;
        end
        
        Ht_s = Ht_s(idx_bs,idx_ue);
         
        
        % combine tap spatial and freq channel:
        for k = 1 : Nf
            H(:,:,k) = H(:,:,k) + Ht_s*Ht_f(k);
        end
    end
        
        
    %constant across time:
    H = repmat(H,[1 1 1 14]);
    
    %normalize:
    E = abs(H).^2;
    H = H / sqrt(mean(E(:)));

    %wrap:
    H_data_cell{iue} = H;
        
    
end

%%
%CTRL

H_ctrl_cell = cell(numUes_ctrl,1);

for iue = 1 : numUes_ctrl
    
    % extract user pusch cfg:
    L_UE = 1;
    
    % build channel:
    H = zeros(L_BS,L_UE,Nf);
    
    for t = 1 : nTaps
        
        % generate tap frequency response:
        Ht_f = exp(-2*pi*1i*f*tau(t));
        
        % generate tap spatial response:
        Ht_c = generate_rnd_sr(L_BS,L_UE,antCorr_ue,antCorr_gnb);
        Ht_s = zeros(L_BS,L_UE);

        for i = 1 : ceil(L_BS/2)
            index_i = 2*(i-1) + 1 : 2*i;
            
            for j = 1 : ceil(L_UE/2)
                index_j = 2*(j-1) + 1 : 2*j;
                
                Px = generate_pol_mtx(xpr);
                Ht_s(index_i,index_j) = Ht_c(i,j) * Px;
            end
        end
        
        if L_UE == 1
            idx_ue = 1;
        else
            idx_ue = 1 : L_UE;
        end
        
        if L_BS == 1
            idx_bs = 1;
        else
            idx_bs = 1 : L_BS;
        end
        
        Ht_s = Ht_s(idx_bs,idx_ue);
         
        
        % combine tap spatial and freq channel:
        for k = 1 : Nf
            H(:,:,k) = H(:,:,k) + Ht_s*Ht_f(k);
        end
    end
        
        
    %constant across time:
    H = repmat(H,[1 1 1 14]);
    
    %normalize:
    E = abs(H).^2;
    H = H / sqrt(mean(E(:)));

    %wrap:
    H_ctrl_cell{iue} = H;  
end

    
end




            
            
            
            
                


















