%%
 % Copyright (c) 2020, LWPU CORPORATION.  All rights reserved.
 %
 % LWPU CORPORATION and its licensors retain all intellectual property
 % and proprietary rights in and to this software, related documentation
 % and any modifications thereto.  Any use, reproduction, disclosure or
 % distribution of this software and related documentation without an express
 % license agreement from LWPU CORPORATION is strictly prohibited.
 %%


function sp = configure_lls()

% =================================================================
% Defines (can't use #define so we pass constants in sp struct
% =================================================================
sp.def.SLOT = 14;
sp.def.PRB = 12;
sp.def.CBLARGE = 8448;
sp.def.TBCRCLEN = 24;    
sp.def.CBCRCLEN = 24;
sp.def.NRFFT = 4096;
sp.def.CRCPOLY24 = [1 1 0 0 0 0 1 1 0 0 1 0 0 1 1 0 0 1 1 1 1 1 0 1 1];

% =====================================
% Configuration parameters - SIMULATION
% =====================================
% Simulation type
sp.simType = 'pusch';      % type of PHY simulation
sp.resultsFileName = 'Results.mat';

% Simulation parameters
sp.sim.chEst = 'ideal';        % channel estimation ideal vs real
sp.sim.numUes = 8;       % number of UEs in simulation
sp.sim.numSlots = 1;      
sp.sim.slotNumber = 13; %slot number within the frame
sp.sim.snrdb = 10;
sp.sim.channel = 'uniform_reflectors';   % Interference free AWGN
                                % Possible channels:
                                % siso_awgn - interference free AWGN
                                % siso_rayleigh - interference free Rayleigh
                                % indep_rayleigh - independent Rayleigh f.
                                % uniform_reflectors
sp.sim.num_reflectors = 10;     % number of reflectors
sp.sim.delay_spread = 1*10^(-6);% delay spread (seconds)
sp.sim.Doppler_spread = 0;      % Doppler spread (Hz)
sp.sim.N0 = 10^(-3);            % noise variance (linear)

% ==============================
% Configuration parameters - gNB
% ==============================

% gNB parameters
% Numerology
sp.gnb.fc = 4;             % carrier frequency GHz
sp.gnb.mu = 1;             % 3GPP mu parameter
sp.gnb.nprb = 273;         % number of PRBs in carrier
sp.gnb.ntx_v = [1 1 1];      % number of tx antennas [N, M, p]
sp.gnb.nrx_v = [8 1 1];      % number of rx antennas [N, M, p]
sp.gnb.nl = 1;             % number of spatial layers per UE
sp.gnd.Nf = sp.gnb.nprb*12; %total number of subcarriers in the allocation
sp.gnd.Nt = 14; %total number of OFDM symbols in the allocation
sp.gnb.L_BS = sp.gnb.nrx_v(1)*sp.gnb.nrx_v(2)*sp.gnb.nrx_v(3); %total number of bs antennas
sp.gnb.L_UE = sp.gnb.nl*sp.sim.numUes; %total number of ue streams


% ================================
% Configuration parameters - PUSCH
% ================================
if strcmp(sp.simType,'pusch')
    % PUSCH configuration parameters
    % ==============================
    
    %%
    %DMRS
    
    %Choose all 3gpp dmrs configurations:
    sp.gnb.dmrs.Type = 1; %1 or 2. Type of dmrs to use.
    sp.gnb.dmrs.AdditionalPosition = 0; %number of additional dmrs to use across time
    sp.gnd.dmrs.maxLength = 2; %1 for single DMRS, 2 for double DMRS
    sp.gnd.dmrs.N_id = 40; %DMRS scrambling id
    sp.gnd.dmrs.n_scid = 0; %scrambling id configurations (0 or 1)
    
    %%
    % MCS
    % --
    sp.gnb.nrMcs = 100;     % use 100 for test mcs, otherwise modes in table XXX, TR XXX
    [sp.gnb.qam, sp.gnb.codeRate] = mcs_to_qam_rate(sp.gnb.nrMcs);
    sp.gnb.maxcbsize = 8448;      % LDPC base graph 1. Possible values: 8848 (graph 1), 3840 (graph 2)
end

% ==================
% Derived parameters 
% ==================
sp.gnb.ntx = sp.gnb.ntx_v(1)*sp.gnb.ntx_v(2)*sp.gnb.ntx_v(3);
sp.gnb.nrx = sp.gnb.nrx_v(1)*sp.gnb.nrx_v(2)*sp.gnb.nrx_v(3);
sp.gnb.numRe = sp.gnb.nl * sp.gnb.nprb * sp.def.PRB * sp.def.SLOT;
sp.gnb.Nf = sp.gnb.nprb * sp.def.PRB; %number of subcarriers in allocation
sp.gnb.Nt = sp.def.SLOT; %number of OFDM symbols in allocation
% sp = derive_dmrs_paramaters(sp);



% ==============
% variable sizes
% ==============
for ii=1:sp.sim.numUes
    sp.sizeof.txTbBits{ii} = [1, 0];       % Determined during initialization
    %    sp.sizeof.rxTbBits{ii} = [1, 0];       % Determined during initialization
    sp.sizeof.tbCrc{ii} = [1, sp.def.TBCRCLEN];
    sp.sizeof.cbCrc{ii} = [0, 0];          % Determined during initialization
    sp.sizeof.txCodedBits{ii} = [1, 0];    % Determined during initialization
    %    sp.sizeof.rxCodedBits{ii} = [1, 0];    % Determined during initialization
    sp.sizeof.txLayerSamples{ii} = [sp.gnb.nl, sp.gnb.nprb*sp.def.PRB, sp.def.SLOT];
    sp.sizeof.txSamplesFD{ii} = [sp.gnb.ntx, sp.def.NRFFT, sp.def.SLOT];
    sp.sizeof.rxLayerSamples{ii} = [sp.gnb.nl, sp.gnb.nprb*sp.def.PRB, sp.def.SLOT];
    sp.sizeof.txSamplesTD{ii} = [sp.gnb.ntx, sp.def.NRFFT, sp.def.SLOT];
end
sp.sizeof.rxSamplesFD = [sp.gnb.nrx, sp.def.NRFFT, sp.def.SLOT];
sp.sizeof.rxSamplesTD = [sp.gnb.nrx, sp.def.NRFFT, sp.def.SLOT];









