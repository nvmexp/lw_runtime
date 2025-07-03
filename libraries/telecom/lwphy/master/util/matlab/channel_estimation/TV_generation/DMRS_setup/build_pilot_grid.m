 %%
 % Copyright (c) 2020, LWPU CORPORATION.  All rights reserved.
 %
 % LWPU CORPORATION and its licensors retain all intellectual property
 % and proprietary rights in and to this software, related documentation
 % and any modifications thereto.  Any use, reproduction, disclosure or
 % distribution of this software and related documentation without an express
 % license agreement from LWPU CORPORATION is strictly prohibited.
 %%


function PAR = build_pilot_grid(PAR)



%build time and frequency pilot grids





%outputs:

%PAR.pilot.pilot_grid_time --> Time pilot grid.

%PAR.pilot.pilot_grid_freq --> Freq pilot grid.





%%

%PARAMETERS



%pilot parameters:

Nf_p = PAR.pilot.Nf_p; %number of frequency pilots



%modulation parameters:

num_PRB = PAR.cluster.num_PRB; %number of PRB per cluster



%%

%FREQ GRID



pilot_grid_freq = zeros(1,Nf_p);



count = 0;



for i = 1 : num_PRB

    for j = 1 : 2

        for k = 1 : 2

            count = count + 1;

            pilot_grid_freq(count) = (i - 1)*12 + 6*(j - 1) + (k - 1);

        end

    end

end



pilot_grid_freq = pilot_grid_freq + 1;



%%

%TIME GRID



pilot_grid_time = [3 4 11 12];



%%

%WRAP



PAR.pilot.pilot_grid_time = pilot_grid_time;

PAR.pilot.pilot_grid_freq = pilot_grid_freq;



%%

%VISUALIZE



if PAR.visualize == 1

    visualize_pilot_grid(PAR);

end



