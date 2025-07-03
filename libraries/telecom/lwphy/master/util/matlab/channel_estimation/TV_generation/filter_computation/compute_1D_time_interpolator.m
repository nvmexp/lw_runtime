 %%
 % Copyright (c) 2020, LWPU CORPORATION.  All rights reserved.
 %
 % LWPU CORPORATION and its licensors retain all intellectual property
 % and proprietary rights in and to this software, related documentation
 % and any modifications thereto.  Any use, reproduction, disclosure or
 % distribution of this software and related documentation without an express
 % license agreement from LWPU CORPORATION is strictly prohibited.
 %%


function W_time = compute_1D_time_interpolator(PAR)



%function computes the 1D MMSE time interpolator



%outputs:

%W_time --> two options for the time interpolator. Dim: Nt x Nt_p x 2



%%

%PARAMETERS



%modulation parameters:

dt = PAR.mod.dt; %OFDM symbol duration (s)

Nt = PAR.mod.Nt; %number of symbols in subframe

df = PAR.mod.df; %subcarrier spacing (Hz)



%pilot parameters:

Nt_p = PAR.pilot.Nt_p; %number of time pilots

OCC_t = PAR.pilot.OCC_t; %Time OCC pilot. Dim: Nf_t x 1

pilot_grid_time = PAR.pilot.pilot_grid_time;



%MMSE parameters:

N0 = PAR.MMSE.N0; %noise variance (linear scale)

Doppler_spread = PAR.MMSE.Doppler_spread; %prior on channel Doppler spread (Hz)

delay_spread = PAR.MMSE.delay_spread; %prior on channel delay spread (s)



%%

%SETUP



data_grid_time = 1 : Nt;



PG = (1 / (df*6)) / delay_spread;



%%

%COVARIANCE



%covariance of pilot tones:

RYY = zeros(Nt_p);



for i = 1 : Nt_p

    for j = 1 : Nt_p

        RYY(i,j) = sinc(Doppler_spread*dt*(pilot_grid_time(i) - pilot_grid_time(j)));

    end

end



RYY = (OCC_t*OCC_t' + ones(Nt_p)) .* RYY + (N0 / PG) * eye(Nt_p);



%%

%CORRELATION



%correlation between pilots and total channel:

RXY = zeros(Nt,Nt_p);



for i = 1 : Nt

    for j = 1 : Nt_p

        RXY(i,j) = sinc(Doppler_spread*dt*(data_grid_time(i) - pilot_grid_time(j)));

    end

end



%%

%MMSE FILTER



W_time = RXY * pilw(RYY);



%%

%ADD OPTIONS



W_time = repmat(W_time,[1 1 2]);



W_time(:,:,2) = W_time(:,:,2) * diag(OCC_t);



x = 2;



















