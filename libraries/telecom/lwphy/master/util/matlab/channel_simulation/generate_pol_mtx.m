%%
 % Copyright (c) 2020, LWPU CORPORATION.  All rights reserved.
 %
 % LWPU CORPORATION and its licensors retain all intellectual property
 % and proprietary rights in and to this software, related documentation
 % and any modifications thereto.  Any use, reproduction, disclosure or
 % distribution of this software and related documentation without an express
 % license agreement from LWPU CORPORATION is strictly prohibited.
 %%

function Px = generate_pol_mtx(xpor)

%function generate a rnd corss polarization matrx

%inputs:
%xpor --> cross polarization power ratio (dB). See TR 38.901

%outputs:
%Px  --> cross polarization channel. Dim: 2 x 2

%%
%START

%colwert to linear:
K = sqrt(10^(-xpor/10));

%generate polarization mtrx:
P = exp(2*pi*1i*rand(2));
P(1,2) = K * P(1,2);
P(2,1) = K * P(2,1);

%colwert to cross polarization:
W = sqrt(1/2) * [1 1; 1 -1];
Px = W' * P * W;

end
