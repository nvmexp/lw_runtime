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
if nargin == 1
    simType = varargin{1};
    testCase = '';
elseif nargin == 2
    simType = varargin{1};
    testCase = varargin{2};
else
    simType = '';
    testCase = '';
end

% Read configuration files:
sp = configure_lls_sdk(simType, testCase);

% derive paramaters:
sp = derive_lls(sp);


%%
%SIMULATE

% Call Physical Channel simulations
if strcmp(sp.sim.opt.simType, 'uplink')
    fprintf("\n*** Begin PUSCH simulation ***\n\n");
    
    simulate_uplink_sdk(sp);               % call PUSCH simulation function
end

fprintf("\n***  End simulation  ***\n\n");





