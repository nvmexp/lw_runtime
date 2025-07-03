%%
 % Copyright (c) 2020, LWPU CORPORATION.  All rights reserved.
 %
 % LWPU CORPORATION and its licensors retain all intellectual property
 % and proprietary rights in and to this software, related documentation
 % and any modifications thereto.  Any use, reproduction, disclosure or
 % distribution of this software and related documentation without an express
 % license agreement from LWPU CORPORATION is strictly prohibited.
 %%

function Main_lls_sdk(varargin)

%clear
% 5G NR link level simulator (LLS) Main file. Supporting simulations for different PHY
% channels, according to configuration file "read_pars_lls()";
%
% Usage:    Main_lls - simulation type configured in configure_lls.m
%           Main_lls <simtype> - simulation types: 'uplink'
%           Main_lls <simtype> <testcase> - test cases: pusch-TC1,...,pusch-TC16

addpath(genpath('./'));


fprintf("\n==================================\nMain simulation file for 5G NR LLS\n==================================\n");


%%
%SETUP

rng('default');                   % option to use same rng seed

if nargin ~= 2
    fprintf('\nUsage: Main_lls_sdk <simType> <testCase>\n');
    fprintf("<simType>: 'uplink', 'pucch', 'pdsch', 'dlCtrl'\n");
    fprintf('<testCase>: refer to Release Notes for supported test cases \n');
    error('Error! Please specify two input arguments');
else
    simType = varargin{1};
    testCase = varargin{2};
end


if ~(strcmp(simType, 'dlCtrl') || strcmp(simType, 'polarEncode'))
    %Read configuration files:
    sp = configure_lls_sdk(simType, testCase);

    %derive paramaters:
    sp = derive_lls(sp);
end


%%
%SIMULATE

% Call Physical Channel simulations
if strcmp(simType, 'dlCtrl')
    if nargin == 2
        DL_ctrl_main(testCase)
    end
    
elseif strcmp(simType, 'polarEncode')
    polar_main(testCase);
    
elseif strcmp(sp.sim.opt.simType, 'uplink')
    fprintf("\n*** Begin UPLINK simulation ***\n\n");
    
    simulate_uplink_sdk(sp);               % call PUSCH simulation function
elseif strcmp(sp.sim.opt.simType, 'pdsch')
     fprintf("\n*** Begin PDSCH simulation ***\n\n");
     rng('default');
     simulate_pdsch_sdk(sp);               % call PUSCH simulation function
end

fprintf("\n***  End simulation  ***\n\n");





