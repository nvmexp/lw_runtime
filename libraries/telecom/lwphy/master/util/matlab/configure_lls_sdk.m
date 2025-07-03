%%
 % Copyright (c) 2020, LWPU CORPORATION.  All rights reserved.
 %
 % LWPU CORPORATION and its licensors retain all intellectual property
 % and proprietary rights in and to this software, related documentation
 % and any modifications thereto.  Any use, reproduction, disclosure or
 % distribution of this software and related documentation without an express
 % license agreement from LWPU CORPORATION is strictly prohibited.
 %%


function sp = configure_lls_sdk(simType, testCase)

%script loads test case paramaters, configures snr for bler lwrve, and
%specifies channel model

%%
%=======================================
% Load test-case simulation paramaters
%=======================================

paramater_str = strcat(testCase,'.mat');
load(paramater_str);

%%
% ===================================
% Configuration parameters - CHANNEL
% ===================================

% Generate a single configuration
% sp.sim.channel.snrStart = 30;                 % starting snr value
% sp.sim.channel.snrEnd = 30;                  % ending snr value
% sp.sim.channel.nSnrSteps = 1;               % number of snr steps
% sp.sim.channel.nSlots = 1;                  % Number of simulated slots per snr step
% Generate multiple test vectors
sp.sim.channel.snrStart = 40;                  % starting snr value
sp.sim.channel.snrEnd = 40;                   % ending snr value
sp.sim.channel.nSnrSteps = 1;                % number of snr steps
sp.sim.channel.nSlots = 1;                    % Number of simulated slots per snr step

% Select a channel model. Options: 'uniform_reflectors','siso-awgn','capture'
% sp.sim.channel.model = 'uniform_reflectors';
% sp.sim.channel.model = 'siso-awgn';
sp.sim.channel.model = 'tdl';

%following paramaters used in uniform reflector channel model:
if strcmp(sp.sim.channel.model,'uniform_reflectors')
    sp.sim.channel.numReflectors = 10;             % Number of reflectors
    sp.sim.channel.delaySpread = 1*10^(-6);        % Delay spread (seconds)
end

%following paramaters used in tdl channel model:
if strcmp(sp.sim.channel.model,'tdl')
    sp.sim.channel.mode = 'c';          % choice of tdl mode, options: (a,b,c,d,e,f)
    sp.sim.channel.antCorr_ue = 'low';  % user antenna correlation. Options: low','med','high'
    sp.sim.channel.antCorr_gnb = 'low'; % gnb antenna correlation. Options: 'low','med','high'
    sp.sim.channel.norm_taps = 1;       % 0 or 1. Option to normalizes taps to unit energy
    sp.sim.channel.ds = 100;            % desired delay spread (ns).
end






