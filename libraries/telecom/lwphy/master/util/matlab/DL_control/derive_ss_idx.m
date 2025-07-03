%%
 % Copyright (c) 2020, LWPU CORPORATION.  All rights reserved.
 %
 % LWPU CORPORATION and its licensors retain all intellectual property
 % and proprietary rights in and to this software, related documentation
 % and any modifications thereto.  Any use, reproduction, disclosure or
 % distribution of this software and related documentation without an express
 % license agreement from LWPU CORPORATION is strictly prohibited.
 %%

function [dmrs_idx, qam_idx, pss_idx, sss_idx] = derive_ss_idx(N_id)

%function derivies and loads indicies for SS block

%inputs:
% N_id      --> physical cell id.

%ouputs:
% dmrs_idx  -->  indicies of dmrs payload. Dim: 144 x 1
% qam_idx   -->  indicies of qam payload.  Dim: 432 x 1
% pss_idx   -->  indicies for pss. Dim: 127 x 1 
% sss_idx   -->  indicies for sss. Dim: 127 x 1
% note: all indicies are 0 based

%%
%PSS/SSS

% these are always the same:
pss_idx = 56 : 182;
sss_idx = (56 : 182) + 240*2;


%%
%DMRS/QAMs

v = mod(N_id,4);

basic_dmrs_idx = [0 4 8] +  v;
basic_qam_idx = 0:11;
basic_qam_idx(basic_dmrs_idx+1) = [];

dmrs_idx = zeros(144,1);
qam_idx = zeros(432,1);

%compute indicies for 2nd SS block symbol:
for i = 1 : 20
    dmrs_idx(3*(i-1)+1:3*i) = basic_dmrs_idx + 12*(i-1) + 240;
    qam_idx(9*(i-1)+1:9*i) = basic_qam_idx + 12*(i-1) + 240;
end

% compute indicies for 3rd SS block symbol:
for i = 1 : 4
    dmrs_idx( (3*(i-1)+1:3*i) + 20*3) = basic_dmrs_idx + 12*(i-1) + 2*240;
    qam_idx( (9*(i-1)+1:9*i) + 20*9) = basic_qam_idx + 12*(i-1) + 2*240;
end

for i = 1 : 4
    dmrs_idx((3*(i-1)+1:3*i) + 24*3) = basic_dmrs_idx + 12*(i-1) + 2*240 + 192;
    qam_idx((9*(i-1)+1:9*i) + 24*9) = basic_qam_idx + 12*(i-1) + 2*240 + 192;
end

% compute indicies for 4th SS block symbol:
for i = 1 : 20
    dmrs_idx((3*(i-1)+1:3*i) + 28*3) = basic_dmrs_idx + 12*(i-1) + 3*240;
    qam_idx((9*(i-1)+1:9*i) + 28*9) = basic_qam_idx + 12*(i-1) + 3*240;
end

    
    
    
    
    
    
    
    
