%%
 % Copyright (c) 2020, LWPU CORPORATION.  All rights reserved.
 %
 % LWPU CORPORATION and its licensors retain all intellectual property
 % and proprietary rights in and to this software, related documentation
 % and any modifications thereto.  Any use, reproduction, disclosure or
 % distribution of this software and related documentation without an express
 % license agreement from LWPU CORPORATION is strictly prohibited.
 %%


function DL_ctrl_main(varargin)

addpath(genpath('./'));

% function generate test vectors for downlink control channels. Can
% configure if pdcch and/or SS transmited. Can be run with test cases:
% DL_ctrl_main('DL_ctrl-TC301'),...,DL_ctrl_main('DL_ctrl-TC304')

%%
%SETUP

% set paramaters for which ctrl channel tranmitted:
if nargin ~=1
    error('Error! Must specify test case.')    
else
    test_case = varargin{1};
    [ss_block_flag,ss_slot_idx,pdcch1_flag,pdcch2_flag, pdcch1_PL, pdcch2_PL] = load_DL_ctrl_TC(test_case);
end

% set paramaters defining ctrl channel:
[gnb,pdcch_lwPHY_cell,pdcch_matlab_cell,ss_matlab_cell,ss_lwPHY] = cfg_DL_ctrl;

%%
%MATLAB
%here we use matlab to generate legal downlink payloads

%generate pdcch qams:
x_pdcch_cell = cell(2,1);

if pdcch1_flag
    pdcch_flag = 1;
    pdcch_PL = pdcch1_PL;
    [x_pdcch_cell{1}, h5File1, tvName1] = generate_pdcch_qpsk(pdcch_matlab_cell{1}, pdcch_flag, pdcch_PL, test_case);
end

if pdcch2_flag
    pdcch_flag = 2;
    pdcch_PL = pdcch2_PL;
    [x_pdcch_cell{2},h5File2, tvName2]  = generate_pdcch_qpsk(pdcch_matlab_cell{2}, pdcch_flag, pdcch_PL, test_case);
end

%generate pbch time-frequency signal:
Xtf_ss = zeros(240,4,4);

for i = 1 : 4
    ss = ss_matlab_cell{i};
    [Xtf_ss(:,:,i) ss_tv_data{i}] = generate_SS_block(ss,gnb);
end


%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%GPU START%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%initialize tf signal:
Xtf = zeros(273*12,14,4);

%build pdcch signal(s)
if pdcch1_flag
    x_pdcch = x_pdcch_cell{1};
    pdcch = pdcch_lwPHY_cell{1};
    h5File = h5File1;
    tvName = tvName1;
    Xtf = embed_pdcch_tf_signal(Xtf,x_pdcch,gnb,pdcch,h5File, tvName);
end


if pdcch2_flag
    x_pdcch = x_pdcch_cell{2};
    pdcch = pdcch_lwPHY_cell{2};
    h5File = h5File2;
    tvName = tvName2;
    Xtf = embed_pdcch_tf_signal(Xtf,x_pdcch,gnb,pdcch,h5File, tvName);
end


% embed ss signal:
if ss_block_flag
    if ss_slot_idx == 1
        Xtf = embed_ss_tf_signal(Xtf,Xtf_ss(:,:,1:2),ss_lwPHY);
        SSTxParams.blockIndex = uint32(0);
    else
        Xtf = embed_ss_tf_signal(Xtf,Xtf_ss(:,:,3:4),ss_lwPHY);
        SSTxParams.blockIndex = uint32(1);
    end
end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%GPU END%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   



%%
%SAVE HDF5

if ss_block_flag
iblock = 1;
    
% Save SS block test vector
SSTxParams.NID = uint32(gnb.N_id);
SSTxParams.nHF = uint32(ss.n_hf);
SSTxParams.Lmax = uint32(ss.L_max);
SSTxParams.f0 = uint32(ss_lwPHY.f0);
SSTxParams.t0 = uint32(ss_lwPHY.t0(iblock));
%SSTxParams.t0_1 = uint32(ss_lwPHY.t0(1));%[2,8];
%SSTxParams.t0_2 = uint32(ss_lwPHY.t0(2));%[2,8];
SSTxParams.ss_slot_idx = uint32(ss_slot_idx);

wrkspaceDir = pwd;
tvDirName = 'GPU_test_input'; 
if ~exist(tvDirName,'dir')
    [status,msg] = mkdir(tvDirName);
end
tvName = sprintf('TV_lwphy_%s_%s.h5',varargin{1},'SSBlock');
h5File  = H5F.create([tvDirName filesep tvName], 'H5F_ACC_TRUNC', 'H5P_DEFAULT', 'H5P_DEFAULT');

hdf5_write_lw(h5File, 'SSTxParams', SSTxParams);
hdf5_write_lw(h5File, 'x_crc', uint32(ss_tv_data{iblock}.x_crc));
hdf5_write_lw(h5File, 'x_encoded', ss_tv_data{iblock}.x_encoded);
hdf5_write_lw(h5File, 'x_rm', ss_tv_data{iblock}.x_rm);
hdf5_write_lw(h5File, 'x_scram', uint8(ss_tv_data{iblock}.x_scram));
hdf5_write_lw(h5File, 'x_qam', ss_tv_data{iblock}.x_qam);
hdf5_write_lw(h5File, 'X_tf_ss', ss_tv_data{iblock}.X_tf_ss);
%hdf5_write_lw(h5File, 'X_tf', Xtf(1:3264,:,1));
if iblock ==1
    Xtf(:,7:14,:) = 0;
elseif iblock == 2
    Xtf(:,1:7,:) = 0;
end
hdf5_write_lw(h5File, 'X_tf', Xtf);%(1:3264,:,:));
H5F.close(h5File);
fprintf(strcat('GPU HDF5 test file \"', tvName, '\" generated successfully.\n'));
end

 

% figure
% imagesc(abs(Xtf(:,:,1)));
% title('transmited slot (Note: matlab 1 indexing)');
% xlabel('OFDM symbol');
% ylabel('subcarrier');




