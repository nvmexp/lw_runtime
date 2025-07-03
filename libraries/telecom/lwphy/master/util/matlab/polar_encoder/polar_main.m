%%
 % Copyright (c) 2020, LWPU CORPORATION.  All rights reserved.
 %
 % LWPU CORPORATION and its licensors retain all intellectual property
 % and proprietary rights in and to this software, related documentation
 % and any modifications thereto.  Any use, reproduction, disclosure or
 % distribution of this software and related documentation without an express
 % license agreement from LWPU CORPORATION is strictly prohibited.
 %%
function polar_main(varargin)


rng('default');

enTvGen = 1;
enDbgPrint = 0;

%%
%PARAMATERS

if nargin == 1
    tcName = varargin{1};
    [K,E] = load_polar_TC(tcName);
else
    % number of information bits (1-164):
    K = 69;

    % number of bits to transmits: (K <= E <= 8192)
    E = 216;
    
    tcName = 'polarEnc-TC0';
end



%%
%START

% generate random data:
c = round(rand(K,1));

% polar encoder:
[d,N] = polar_encode_alt(c,K,E,enDbgPrint);

% polar rate matching:
e = polar_rate_match_alt(d,N,K,E,enDbgPrint);


% % TV generation:
if enTvGen
    
    tvDirName = 'GPU_test_input'; [status,msg] = mkdir(tvDirName);
    
    encPrms.nInfoBits = K;
    encPrms.nCodedBits = N;
    encPrms.nTxBits = E;

    infoBytes = bits2bytes(c);
    codedBytes = bits2bytes(d);
    txBytes = bits2bytes(e);

    tvName = sprintf('TV_%s_infoBits_%d_codedBits_%d_txBits_%d.h5',tcName,encPrms.nInfoBits, encPrms.nCodedBits, encPrms.nTxBits);
    h5File  = H5F.create([tvDirName filesep tvName], 'H5F_ACC_TRUNC', 'H5P_DEFAULT', 'H5P_DEFAULT');
    %h5File  = H5F.create(tvName, 'H5F_ACC_TRUNC', 'H5P_DEFAULT', 'H5P_DEFAULT');
    hdf5_write_lw(h5File, 'encPrms', encPrms);
    hdf5_write_lw(h5File, 'InfoBits', uint8(infoBytes));
    hdf5_write_lw(h5File, 'CodedBits', uint8(codedBytes));
    hdf5_write_lw(h5File, 'TxBits', uint8(txBytes));
    H5F.close(h5File);
    
    fprintf(strcat('Generated GPU TV \"', tvName, '\" successfully.\n'));
end

end






















    
