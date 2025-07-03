 %%
 % Copyright (c) 2020, LWPU CORPORATION.  All rights reserved.
 %
 % LWPU CORPORATION and its licensors retain all intellectual property
 % and proprietary rights in and to this software, related documentation
 % and any modifications thereto.  Any use, reproduction, disclosure or
 % distribution of this software and related documentation without an express
 % license agreement from LWPU CORPORATION is strictly prohibited.
 %%


function save_lwPHY_tv(j_slot,Y,PuschCfg,sp)

    % Generatee data and aux files in HDF5 format
            % saving data
            TF_received_signal = Y;
            %cd lwPHY_test_vector_generation
            %save('lwphy_test_input.mat','TF_received_signal'); 
            %generate_tv_hdf5_v4(sp.gnb.numerology.L_BS, sp.gnb.pusch.L_UE, sp.gnb.numerology.Nf, 10^(-sp.sim.channel.lwrrentSnr/10), sp.gnb.pusch.symIdx_data, sp.gnb.pusch.sd, sp.gnb.pusch.s);
            %generate_tv_hdf5_v4(sp.gnb.numerology.L_BS, sp.gnb.pusch.L_UE, sp.gnb.numerology.Nf, 10^(-35/10), sp.gnb.pusch.symIdx_data, sp.gnb.pusch.sd, sp.gnb.pusch.s);
                        
            % generate configuration files in HDF5 fomrat            
            % TB parameters
            % generate H5F object
            %parFileName = sprintf('lwphy_input_pars_tb.h5');
            %tvDirName = 'GPU_test_input';
            %h5File  = H5F.create([tvDirName filesep parFileName], 'H5F_ACC_TRUNC', 'H5P_DEFAULT', 'H5P_DEFAULT');            
            
            % Create parameter structure   
            % TB
            tb_pars.nRnti = uint32(sp.gnb.pusch.PuschCfg_cell{1}.n_rnti);
            % MIMO
            tb_pars.numLayers = uint32(sp.gnb.pusch.PuschCfg_cell{1}.mimo.nl);
            switch (tb_pars.numLayers)
                case 1
                    tb_pars.layerMap = uint32(1);
                case 2
                    tb_pars.layerMap = uint32(3);
                case 4
                    tb_pars.layerMap = uint32(15);
                otherwise
                    error("Error. Could not find a valid map");
            end
            % Resource allocation
            tb_pars.startPrb = uint32(sp.gnb.pusch.PuschCfg_cell{1}.alloc.startPrb - 1);
            tb_pars.numPRb = uint32(sp.gnb.pusch.PuschCfg_cell{1}.alloc.nPrb);
            tb_pars.startSym = uint32(sp.gnb.pusch.PuschCfg_cell{1}.alloc.startSym - 1);
            tb_pars.numSym = uint32(sp.gnb.pusch.PuschCfg_cell{1}.alloc.nSym);
            % Back-end parameters
            tb_pars.dataScramId = uint32(0);
            tb_pars.mcsTableIndex = uint32(sp.gnb.pusch.PuschCfg_cell{1}.coding.mcsTable);
            tb_pars.mcsIndex = uint32(sp.gnb.pusch.PuschCfg_cell{1}.coding.mcs);
            tb_pars.rv = uint32(0);
            % DMRS parameters
            tb_pars.dmrsType = uint32(sp.gnb.pusch.PuschCfg_cell{1}.dmrs.type);
            tb_pars.dmrsAddlPosition = uint32(sp.gnb.pusch.PuschCfg_cell{1}.dmrs.AdditionalPosition);
            tb_pars.dmrsMaxLength = uint32(sp.gnb.pusch.PuschCfg_cell{1}.dmrs.maxLength);
            tb_pars.dmrsScramId = uint32(sp.gnb.pusch.PuschCfg_cell{1}.dmrs.n_scid);
            tb_pars.dmrsEnergy = uint32(sp.gnb.pusch.PuschCfg_cell{1}.dmrs.energy);
            switch (sp.gnb.pusch.L_UE)
                case 1
                    dmrsCfg = 0;
                case 2
                    dmrsCfg = 1;
                case 4
                    dmrsCfg = 2;
                case 8
                    dmrsCfg = 3;
                otherwise
                    error("could not find a valid layer count %d to DMRS config mapping");
            end
            tb_pars.dmrsCfg = uint32(dmrsCfg);
                        
            %hdf5_write_lw2(h5File, 'tb_pars', tb_pars);



            %... 
            %H5F.close(h5File);
            %fprintf(strcat('GPU HDF5 test file \"', parFileName, '\" generated successfully.\n'));
            
            % gNB parameters
            % generate H5F object
            %parFileName = sprintf('lwphy_input_pars_gnb.h5');
            %tvDirName = 'GPU_test_input';
            %h5File  = H5F.create([tvDirName filesep parFileName], 'H5F_ACC_TRUNC', 'H5P_DEFAULT', 'H5P_DEFAULT');
            
            % Determine total number of PRBs
            totalPrb = tb_pars.numPRb; % FIXME only good for one TB
            totalNf = totalPrb * 12;
            
            % Create parameter structure
            gnb_pars.fc = uint32(sp.gnb.fc);
            gnb_pars.mu = uint32(sp.gnb.mu);
            gnb_pars.nRx = uint32(sp.gnb.nrx_v(1) * sp.gnb.nrx_v(2) * sp.gnb.nrx_v(3));  % CHECK            
            gnb_pars.nPrb = uint32(totalPrb);
            gnb_pars.cellId = uint32(sp.gnb.N_data_id);
            gnb_pars.slotNumber = uint32(sp.gnb.pusch.slotNumber);
            %h5_gnb_pars.dmrsScramId = 0;
            %h5_gnb_pars.dataScramId = 0;
            gnb_pars.Nf = uint32(totalNf);
            gnb_pars.Nt = uint32(sp.gnb.numerology.Nt);
            gnb_pars.df = uint32(sp.gnb.numerology.df);
            gnb_pars.dt = uint32(sp.gnb.numerology.dt);
            gnb_pars.numBsAnt = uint32(sp.gnb.nrx_v(1) * sp.gnb.nrx_v(2) * sp.gnb.nrx_v(3)); % CHECK
            gnb_pars.numBbuLayers = uint32(sp.gnb.pusch.L_UE);  % CHECK
            gnb_pars.numTb = uint32(sp.gnb.pusch.numUes);    % CHECK
            gnb_pars.ldpcnIterations = uint32(10);
            gnb_pars.ldpcEarlyTermination = uint32(0);
            gnb_pars.ldpcAlgoIndex = uint32(0);
            gnb_pars.ldpcFlags = uint32(0);
            gnb_pars.ldplwseHalf = uint32(0);
            
            %hdf5_write_lw2(h5File, 'gnb_pars', gnb_pars);
            
            %...
            %H5F.close(h5File);
            %fprintf(strcat('GPU HDF5 test file \"', parFileName, '\" generated successfully.\n'));
            
            %generate_tv_hdf5_v4(sp.gnb.numerology.L_BS, sp.gnb.pusch.L_UE, sp.gnb.numerology.Nf, 10^(-35/10), sp.gnb.pusch.symIdx_data, sp.gnb.pusch.sd, sp.gnb.pusch.s, gnb_pars, tb_pars);

            generate_tv_hdf5_v5(sp.sim.opt.testCase, sp.gnb.numerology.L_BS, sp.gnb.pusch.L_UE, 10^(-sp.sim.channel.lwrrentSnr/10), ...
                sp.sim.channel.lwrrentSnr, sp.gnb.pusch.symIdx_data, TF_received_signal, sp.gnb.pusch.s_grid, sp.gnb.pusch.sd, sp.gnb.pusch.s, ...
                gnb_pars, tb_pars, sp.gnb.pusch.reciever.ChEst, PuschCfg.coding.qam, j_slot, sp);
            
          
%             generate_tv_hdf5_v5(test_case_name, L_BS, L_UE, N0_data, snr_db, time_data_mask, TF_received_signal, s_grid, sd, s, gnb_pars, tb_pars, chEst_pars, qam, j_slot,sp)
            %cd .. 
            
            
            
%             % gNB parameters
%             % generate H5F object
%             parFileName = sprintf('lwphy_input_pars_gnb.h5');
%             tvDirName = 'GPU_test_input';
%             h5File  = H5F.create([tvDirName filesep parFileName], 'H5F_ACC_TRUNC', 'H5P_DEFAULT', 'H5P_DEFAULT');
%             
%             % Create parameter structure
%             gnb_pars.fc = uint32(sp.gnb.fc);
%             gnb_pars.mu = uint32(sp.gnb.mu);
%             gnb_pars.nRx = uint32(sp.gnb.nrx_v(1) * sp.gnb.nrx_v(2) * sp.gnb.nrx_v(3));  % CHECK
%             gnb_pars.nPrb = uint32(sp.gnb.nPrb);
%             gnb_pars.cellId = uint32(sp.gnb.N_id);
%             gnb_pars.slotNumber = uint32(sp.gnb.pusch.slotNumber);
%             %h5_gnb_pars.dmrsScramId = 0;
%             %h5_gnb_pars.dataScramId = 0;
%             gnb_pars.Nf = uint32(sp.gnb.numerology.Nf);
%             gnb_pars.Nt = uint32(sp.gnb.numerology.Nt);
%             gnb_pars.df = uint32(sp.gnb.numerology.df);
%             gnb_pars.dt = uint32(sp.gnb.numerology.dt);
%             gnb_pars.numBsAnt = uint32(sp.gnb.nrx_v(1) * sp.gnb.nrx_v(2) * sp.gnb.nrx_v(3)); % CHECK
%             gnb_pars.numBbuLayers = uint32(sp.gnb.pusch.L_UE);  % CHECK
%             gnb_pars.numTb = uint32(sp.gnb.pusch.numUes);    % CHECK
%             gnb_pars.ldpcnIterations = uint32(10);
%             gnb_pars.ldpcEarlyTermination = uint32(0);
%             gnb_pars.ldpcAlgoIndex = uint32(0);
%             gnb_pars.ldpcFlags = uint32(0);
%             gnb_pars.ldplwseHalf = uint32(0);
%             
%             hdf5_write_lw2(h5File, 'gnb_pars', gnb_pars);
%             
%             %...
%             H5F.close(h5File);
%             fprintf(strcat('GPU HDF5 test file \"', parFileName, '\" generated successfully.\n'));
%             
%             generate_tv_hdf5_v4(sp.gnb.numerology.L_BS, sp.gnb.pusch.L_UE, sp.gnb.numerology.Nf, 10^(-35/10), sp.gnb.pusch.symIdx_data, sp.gnb.pusch.sd, sp.gnb.pusch.s, gnb_pars, tb_pars);
%             
%             cd .. 
                       