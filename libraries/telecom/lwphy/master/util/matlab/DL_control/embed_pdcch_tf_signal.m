%%
 % Copyright (c) 2020, LWPU CORPORATION.  All rights reserved.
 %
 % LWPU CORPORATION and its licensors retain all intellectual property
 % and proprietary rights in and to this software, related documentation
 % and any modifications thereto.  Any use, reproduction, disclosure or
 % distribution of this software and related documentation without an express
 % license agreement from LWPU CORPORATION is strictly prohibited.
 %%

function Xtf = embed_pdcch_tf_signal(Xtf,x,gnb,pdcch,h5File, tvName)

%function build dmrs signal and embeds it allong with encoded payload into
%the time-frequency domain

%inputs:
% Xtf     --> time-frequency signal. Dim: 3264 x 14
% x       --> qpsk payload
% pdcch   --> pdcch paramaters

%outputs:
% Xtf    --> time-frequency signal with embeded pdcch transmission. Dim: 3264 x 14

%%
%PARAMATERS

% gnb:
slotNumber = gnb.slotNumber;  % slot number

% pdcch:
startRb = pdcch.startRb;      % pdcch tx starting RB (0 indexing)
nRb = pdcch.nRb;              % number of pdcch tx RBs
startSym = pdcch.startSym;    % starting symbol pdcch tx (0 indexing)
nSym = pdcch.nSym;            % number of pdcch tx symbols (1-3)
dmrsId = pdcch.dmrsId;        % dmrs scrambling id
beta_qpsk = pdcch.beta_qpsk;    % power scaling of qpsk signal
beta_dmrs = pdcch.beta_dmrs;  % power scaling of dmrs signal


%%
%SETUP

pdcch_start_freq = 12*startRb; % starting subcarrier of pdcch tranmission (0 indexing)

%%
%STEP 1
% here we compute the pdcch dmrs

endRb = startRb + nRb;
r = zeros(3*nRb,nSym);

for i = 0 : (nSym - 1)
    
    % compute seed:
    t = startSym + i;
    c_init =  mod(2^17*(14*slotNumber+t+1)*(2*dmrsId+1)+2*dmrsId,2^31);
    
    % compute Gold sequence:
    c = build_Gold_sequence(c_init,6*endRb);
    
    % extract end of sequence
    c = c(end - 6*nRb + 1 : end);      
    
    % qpsk modulate:
    r(:,i+1) = qpsk_modulate(c,6*nRb);
    
    % scale power:
    r(:,i+1) = beta_dmrs * r(:,i+1);
 
end

%%
%STEP 2
% here we embed the pdcch dmrs

dmrs_idx_base = 1 : 4 : (1 + 4*(3*nRb-1));

for i = 0 : (nSym - 1)
    t = i + startSym;
    Xtf(dmrs_idx_base + pdcch_start_freq + 1,t+1) = r(:,i+1);
end

%%
%STEP 3
% here we embed the pdcch qpsk signal

% number of qpsks per ofdm symbol
nqpsk_per_sym = nRb*9;

% scale power of qpsks:
x = beta_qpsk * x;

% compute indicies:
qpsk_idx_base = 0 : (12*nRb - 1);
qpsk_idx_base(dmrs_idx_base + 1) = [];  

% embed:
for i = 0 : (nSym - 1)
    
    t = startSym + i;
    
    Xtf((qpsk_idx_base + pdcch_start_freq) + 1,t+1) = ...
        x((nqpsk_per_sym*i + 1): nqpsk_per_sym*(i+1));
    
end

%hdf5_write_lw(h5File, 'r', r);
%hdf5_write_lw(h5File, 'Xtf', Xtf);

% Write input / output values to test vector
PdcchParams.slot_number = gnb.slotNumber;
PdcchParams.start_rb = uint32(pdcch.startRb);
PdcchParams.n_rb = uint32(pdcch.nRb);
PdcchParams.start_sym = uint32(pdcch.startSym);
PdcchParams.n_sym = uint32(pdcch.nSym);
PdcchParams.dmrs_id = uint32(pdcch.dmrsId);
PdcchParams.beta_qam = single(pdcch.beta_qpsk);
PdcchParams.beta_dmrs = single(pdcch.beta_dmrs);

hdf5_write_lw(h5File, 'PdcchParams', PdcchParams);  
hdf5_write_lw(h5File, 'tf_signal',   complex(zeros(size(Xtf))));
hdf5_write_lw(h5File, 'ref_output',   complex(Xtf));
%hdf5_write_lw(h5File, 'qam_payload',   complex(x(1:72)));%FIXME
hdf5_write_lw(h5File, 'qam_payload',   complex(x));
% Write configuration to test vector
params.startRb = pdcch.startRb;
params.slot_number = gnb.slotNumber;
params.n_rb = pdcch.nRb;
params.start_sym = pdcch.startSym;
params.dmrs_id = pdcch.dmrsId;
params.beta_qam = pdcch.beta_qpsk;
params.beta_dmrs = pdcch.beta_dmrs;
params.n_f = size(Xtf,1);
params.n_t = size(Xtf,2);

n_f = size(Xtf,1);
n_t = size(Xtf,2);

hdf5_write_lw(h5File, 'params', params);
hdf5_write_lw(h5File, 'n_f', n_f);
hdf5_write_lw(h5File, 'n_t', n_t);

H5F.close(h5File);
fprintf(strcat('GPU HDF5 test file \"', tvName, '\" generated successfully.\n'));



    
    
    



