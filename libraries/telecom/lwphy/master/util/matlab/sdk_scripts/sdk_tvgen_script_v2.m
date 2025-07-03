
% Copyright (c) 2020, LWPU CORPORATION.  All rights reserved.
%
% LWPU CORPORATION and its licensors retain all intellectual property
% and proprietary rights in and to this software, related documentation
% and any modifications thereto.  Any use, reproduction, disclosure or
% distribution of this software and related documentation without an express
% license agreement from LWPU CORPORATION is strictly prohibited.



% lwPHY Test Vector Generation Script

clear all; close all;

%% CONFIGURATION

numUes = 8;                     % Number of UEs
numBsAnt = 16;                  % Number of gNB antennas
snrdB = 40;                     % SNR in dB 
rnti = [46 64 43 47 9100 9101 9104 9105];   % RNTI UEs 1-8
TBS = 108552;                   % Transport block size (bits)
nPrb = 272;                     % Number of PRB 
nSym = 6;                       % Number of data symbols in time domain
nSc = 12;                       % Number of sub-carriers per PRB
Nre = nSc*nSym*nPrb;            % Number of Resource Elements per layer 
nl = 1;                         % Number of layers per UE
bgn = 1;                        % Base graph number
ncellid = 20;                   % Cell ID
q=0;                            % Codeword index
n = Nre*6;                      % Number of LLRs

timeDataMask = [1 2 5 6 7 8];   % Position of data symbols within slot
timeDmrsMask = [3 4 9 10];      % Position of DMRS symbols within slot

%% PUSCH TRANSMIT CHAIN

% Initialize cell arrays;
tvDataIn = cell(1,numUes);
txGrid = cell(1,numUes);

%rng(10);                       % Random number generator seed can be fixed uncommenting this line 

% Loop over all UEs
for ueIdx = 1:numUes
    % Generate TB data
    tvDataIn{ueIdx} = round(rand(108552,1));
    
    % Aggregate CRC
    tvDataInCrc = nrCRCEncode(tvDataIn{ueIdx},'24A');
    
    % TB segmentation
    txCbs = nrCodeBlockSegmentLDPC(tvDataInCrc,1);
    
    % LDPC encoding
    K = size(txCbs,1);                        
    C = size(txCbs,1);                        
    txCodedCbs = nrLDPCEncode(txCbs,bgn); 
    
    % Rate matching
    txRateMatCbs = nrRateMatchLDPC(txCodedCbs,Nre*6, 0,'64QAM',1);

    % Scrambling
    q=0;
    n = Nre*6;
    scrseq = nrPDSCHPRBS(ncellid, rnti(ueIdx), q, n);
    
    %txScramRateMatCbs = txRateMatCbs;
    txScramRateMatCbs = xor(txRateMatCbs, scrseq);

    % QAM Modulation
    txSym{ueIdx} = nrSymbolModulate(logical(txScramRateMatCbs),'64QAM');

    % RE mapping
    txGrid{ueIdx} = zeros(nPrb*nSc,14);
    txGrid{ueIdx}(:,timeDataMask) = reshape(txSym{ueIdx},length(txSym{ueIdx})/nSym,nSym);
    
    % Load DMRS
    load('test_dmrs.mat');
    for ueIdx = 1:numUes
        txGrid{ueIdx}(:,timeDmrsMask) = sqrt(2) * dmrs{ueIdx};
    end

end

%% FREQUENCY DOMAIN CHANNEL MODEL

% Load channel model
load('test_channel_1.mat');

% Re-format transmit signal
X = zeros(nSc*nPrb,14,numUes);

for ueIdx = 1:numUes
    index = (ueIdx - 1)*nl + 1 :  ueIdx*nl;
    X(:,:,index) = txGrid{ueIdx};
end

X = permute(X,[3 1 2]);

% Apply channel
Y = zeros(numBsAnt,nSc*nPrb,14);

for f = 1 : nSc*nPrb
    for t = 1 : 14
        Y(:,f,t) = H(:,:,f,t)*X(:,f,t);
    end
end

% Add noise
N0 = 10^(-snrdB/10);
Y = Y + sqrt(N0 / 2) * (randn(size(Y)) + 1i*randn(size(Y)));

TF_recieved_signal = permute(Y,[2 3 1]);

%% WRITE RECEIVED SIGNAL TO .MAT FILE

save('sdk_test_input.mat','TF_recieved_signal');
fprintf('Test vector generated successfully.\n');
%%


