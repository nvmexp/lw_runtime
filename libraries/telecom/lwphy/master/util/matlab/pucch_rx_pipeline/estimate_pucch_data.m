%%
 % Copyright (c) 2020, LWPU CORPORATION.  All rights reserved.
 %
 % LWPU CORPORATION and its licensors retain all intellectual property
 % and proprietary rights in and to this software, related documentation
 % and any modifications thereto.  Any use, reproduction, disclosure or
 % distribution of this software and related documentation without an express
 % license agreement from LWPU CORPORATION is strictly prohibited.
 %%

function data_est = estimate_pucch_data(H_est,Y_data_iue)

%function applies a matched filter to estimate pucch data

%inputs:
%H_est      --> estimate of pucch channel. Dim: 12 x nSym_data x L_BS
%Y_data_iue --> pucch data signal. Dim: 12 x nSym_data x L_BS

%outputs:
%data_est      --> hard estimate of pucch data (scaler).

%%
%START

%apply match filter:
m = conj(H_est) .* Y_data_iue;
m = sum(m(:));

%hard slice:
if real(m) <= 0
    data_est_real = -1 / sqrt(2);
else
    data_est_real = 1 / sqrt(2);
end

if imag(m) <= 0
    data_est_imag = -1 / sqrt(2);
else
    data_est_imag = 1 / sqrt(2);
end

data_est = data_est_real + 1i*data_est_imag;

end
