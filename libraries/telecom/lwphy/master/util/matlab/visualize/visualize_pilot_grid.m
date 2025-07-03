 %%
 % Copyright (c) 2020, LWPU CORPORATION.  All rights reserved.
 %
 % LWPU CORPORATION and its licensors retain all intellectual property
 % and proprietary rights in and to this software, related documentation
 % and any modifications thereto.  Any use, reproduction, disclosure or
 % distribution of this software and related documentation without an express
 % license agreement from LWPU CORPORATION is strictly prohibited.
 %%


function visualize_pilot_grid(PAR)



%%

%PARAMATERS



%pilot paramaters:

pilot_grid_time = PAR.pilot.pilot_grid_time; %Time pilot grid

pilot_grid_freq = PAR.pilot.pilot_grid_freq; %Freq pilot grid



%modulation paramaters:

Nf = PAR.mod.Nf; %number of tones in subframe

Nt = PAR.mod.Nt; %number of symbols in subframe



%%

%START



P1 = ones(Nf,Nt);

P(pilot_grid_freq,pilot_grid_time) = 1;



[Xd,Yd] = meshgrid(1 : Nt,1 : Nf);

[Xg,Yg] = meshgrid(pilot_grid_time,pilot_grid_freq);





figure

plot(Xg(:),Yg(:),'b.');

xlabel('OFDM symbol');

ylabel('subcarrier');

title('location of pilot tones');

grid on



a = 2;