%%
 % Copyright (c) 2020, LWPU CORPORATION.  All rights reserved.
 %
 % LWPU CORPORATION and its licensors retain all intellectual property
 % and proprietary rights in and to this software, related documentation
 % and any modifications thereto.  Any use, reproduction, disclosure or
 % distribution of this software and related documentation without an express
 % license agreement from LWPU CORPORATION is strictly prohibited.
 %%

function xTF = embed_qams(xTF,symbols,PuschCfg)

%function embeds a user's qam symbols into the TF grid

%inputs:
%xTF      --> empty TF grid. Dim: Nf x Nt x nl
%symbols  --> users qam symbols. Dim: N_data * nl

%outputs:
%xTF      --> empty TF grid with embeded qam signal. Dim: Nf x Nt x nl

%%
%PARAMATERS

%mimo paramaters:
nl = PuschCfg.mimo.nl;                    % number of layers transmited by user

%allocation paramaters:
nPrb = PuschCfg.alloc.nPrb;               % number of prbs in allocation
startPrb = PuschCfg.alloc.startPrb;       % starting prb of allocation
Nf_data = PuschCfg.alloc.Nf_data;         % number of data subcarriers in allocation
Nt_data = PuschCfg.alloc.Nt_data;         % number of data symbols in allocation
N_data = PuschCfg.alloc.N_data;           % number of data TF resources in allocation
symIdx_data = PuschCfg.alloc.symIdx_data; % indicies of data symbols. Dim: Nt_data x 1 


%%
%SHAPE SYMBOLS

%layer mapping:
symbols = reshape(symbols,nl,N_data);           % now dim: nl x N_data
symbols = symbols.';                            % now dim: N_data x nl

%frequency first mapping:
symbols = reshape(symbols,Nf_data,Nt_data,nl);  % now dim: Nf_data x Nt_data x nl

%%
%EMBED

freqIdx_data = 12 * (startPrb - 1) + 1 : 12 * (startPrb + nPrb - 1);

for layer = 1 : nl
    xTF(freqIdx_data, symIdx_data, layer) = symbols(:, :, layer);
end



end



















