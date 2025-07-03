 %%
 % Copyright (c) 2020, LWPU CORPORATION.  All rights reserved.
 %
 % LWPU CORPORATION and its licensors retain all intellectual property
 % and proprietary rights in and to this software, related documentation
 % and any modifications thereto.  Any use, reproduction, disclosure or
 % distribution of this software and related documentation without an express
 % license agreement from LWPU CORPORATION is strictly prohibited.
 %%


function out = modulate_bits_wrapper(in,modulation)

%function modulates bits into QAM symbols

%%
%QAM MAPPING

load(fullfile('./pusch_tx_pipeline/QAM/qam_mapping.mat'));
in = int32(in);
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
z = 2.^(bits_per_QAM-1:-1:0);
z = flip(z);


nQAMs = length(in) / bits_per_QAM;

nQAMs = int32(nQAMs);
bits_per_QAM = int32(bits_per_QAM);
z = int32(z);

QAM_mapping_real = real(QAM_mapping);
QAM_mapping_imag = imag(QAM_mapping);

[out_real,out_imag] = modulate_mex(in,nQAMs,bits_per_QAM,...
    QAM_mapping_real,QAM_mapping_imag,z);

out = out_real + 1i*out_imag;





    
    
    
    
    
    
    
    
    
    
