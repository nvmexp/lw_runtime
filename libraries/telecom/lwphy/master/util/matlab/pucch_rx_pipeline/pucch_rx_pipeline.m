%%
 % Copyright (c) 2020, LWPU CORPORATION.  All rights reserved.
 %
 % LWPU CORPORATION and its licensors retain all intellectual property
 % and proprietary rights in and to this software, related documentation
 % and any modifications thereto.  Any use, reproduction, disclosure or
 % distribution of this software and related documentation without an express
 % license agreement from LWPU CORPORATION is strictly prohibited.
 %%

function nrSlot = pucch_rx_pipeline(Y, sp, nrSlot)

% function applies pucch receive pipeline to base station received signal

% inputs:
% Y       --> received signal. Dim: Nf x Nt x L_BS

% outputs:
% nrSlot  --> nrSlot with received data



%%
%START


 % colwert to gpu paramaters:
[nUe_pucch,Pucch_common,Pucch_ue_cell,Pucch_receiver] = extract_gpu_pucch_par(sp);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%GPU START%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

b_est = gpu_pucch_reciever(Y, nUe_pucch,Pucch_common,Pucch_ue_cell,Pucch_receiver,sp);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%GPU END%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% save estimate to structure:
nrSlot = wrap_pucch_estimtate(b_est,Pucch_ue_cell,nrSlot);


