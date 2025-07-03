 %%
 % Copyright (c) 2020, LWPU CORPORATION.  All rights reserved.
 %
 % LWPU CORPORATION and its licensors retain all intellectual property
 % and proprietary rights in and to this software, related documentation
 % and any modifications thereto.  Any use, reproduction, disclosure or
 % distribution of this software and related documentation without an express
 % license agreement from LWPU CORPORATION is strictly prohibited.
 %%


function out = modulate_bits(in,modulation)

%function modulates bits into QAM symbols

%%
%QAM MAPPING

load(fullfile('./pusch_tx_pipeline/QAM/qam_mapping.mat'));

switch modulation
    
    case 'QPSK'
        bits_per_QAM = 2;
        QAM_mapping = QPSK_mapping;
        
    case '16QAM'
        bits_per_QAM = 4;
        QAM_mapping = QAM16_mapping;
        
    case '64QAM'
        bits_per_QAM = 6;
        QAM_mapping = QAM64_mapping;
        
    case '256QAM'
        bits_per_QAM = 8;
        QAM_mapping = QAM256_mapping;
end

%%
%START

num_qams = length(in) / bits_per_QAM;
out = zeros(num_qams,1);

for i = 1 : num_qams
    index = (i-1)*bits_per_QAM + 1 : i*bits_per_QAM;
    bits = (flip(in(index)))';
%     bits = flip(bits');
    qam_index = b2d(bits);
    out(i) = QAM_mapping(qam_index + 1);
end


    
    
    
    
    
    
    
    
    
    
