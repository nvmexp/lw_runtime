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
               
            % Create parameter structure
            % TB
            for ii=1:sp.gnb.pusch.numUes
                tb_pars(ii).nRnti = uint32(sp.gnb.pusch.PuschCfg_cell{ii}.n_rnti);
                % MIMO
                tb_pars(ii).numLayers = uint32(sp.gnb.pusch.PuschCfg_cell{ii}.mimo.nl);
                if ii==1
                    switch (tb_pars(ii).numLayers)
                        case 1
                            tb_pars(ii).layerMap = uint32(1);
                        case 2
                            tb_pars(ii).layerMap = uint32(3);
                        case 4
                            if sp.gnb.pusch.numUes == 1
                                tb_pars(ii).layerMap = uint32(15);
                            elseif sp.gnb.pusch.numUes == 2
                                tb_pars(ii).layerMap = uint32(51);
                            end
                        otherwise
                            error("Error. Could not find a valid map");
                    end
                elseif ii==2
                    switch (tb_pars(ii).numLayers)
                        case 1
                            tb_pars(ii).layerMap = uint32(2);
                        case 2
                            tb_pars(ii).layerMap = uint32(12);
                        case 4
                            %tb_pars(ii).layerMap = uint32(240);
                            tb_pars(ii).layerMap = uint32(204);
                        otherwise
                            error("Error. Could not find a valid map");
                    end
                elseif ii==3
                    switch (tb_pars(ii).numLayers)
                        case 1
                            tb_pars(ii).layerMap = uint32(3);
                        otherwise
                            error("Error. Could not find a valid map");
                    end
                elseif ii==4
                    switch (tb_pars(ii).numLayers)
                        case 1
                            tb_pars(ii).layerMap = uint32(4);
                        otherwise
                            error("Error. Could not find a valid map");
                    end
                end
                % Resource allocation
                tb_pars(ii).startPrb = uint32(sp.gnb.pusch.PuschCfg_cell{ii}.alloc.startPrb - 1);
                tb_pars(ii).numPRb = uint32(sp.gnb.pusch.PuschCfg_cell{ii}.alloc.nPrb);
                tb_pars(ii).startSym = uint32(sp.gnb.pusch.PuschCfg_cell{ii}.alloc.startSym - 1);
                tb_pars(ii).numSym = uint32(sp.gnb.pusch.PuschCfg_cell{ii}.alloc.nSym);
                % Back-end parameters
                tb_pars(ii).dataScramId = uint32(0);
                tb_pars(ii).mcsTableIndex = uint32(sp.gnb.pusch.PuschCfg_cell{ii}.coding.mcsTable);
                tb_pars(ii).mcsIndex = uint32(sp.gnb.pusch.PuschCfg_cell{ii}.coding.mcs);
                tb_pars(ii).rv = uint32(0);
                % DMRS parameters
                tb_pars(ii).dmrsType = uint32(sp.gnb.pusch.PuschCfg_cell{ii}.dmrs.type);
                tb_pars(ii).dmrsAddlPosition = uint32(sp.gnb.pusch.PuschCfg_cell{ii}.dmrs.AdditionalPosition);
                tb_pars(ii).dmrsMaxLength = uint32(sp.gnb.pusch.PuschCfg_cell{ii}.dmrs.maxLength);
                tb_pars(ii).dmrsScramId = uint32(sp.gnb.pusch.PuschCfg_cell{ii}.dmrs.n_scid);
                tb_pars(ii).dmrsEnergy = uint32(sp.gnb.pusch.PuschCfg_cell{ii}.dmrs.energy);
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
                tb_pars(ii).dmrsCfg = uint32(dmrsCfg);
            end
      
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
            gnb_pars.ldplwseHalf = uint32(1);
            
            generate_tv_hdf5_v5(sp.sim.opt.testCase, sp.gnb.numerology.L_BS, sp.gnb.pusch.L_UE,...
                10^(-sp.sim.channel.lwrrentSnr/10), sp.sim.channel.lwrrentSnr, sp.gnb.pusch.symIdx_data,...
                TF_received_signal, sp.gnb.pusch.s_grid, sp.gnb.pusch.sd, sp.gnb.pusch.s, gnb_pars, tb_pars,...
                sp.gnb.pusch.reciever.ChEst, PuschCfg.coding.qam, j_slot, sp);
            %generate_tv_hdf5_v5(sp.sim.opt.testCase, sp.gnb.numerology.L_BS, sp.gnb.pusch.L_UE, 10^(-sp.sim.channel.lwrrentSnr/10), ...
            %    sp.sim.channel.lwrrentSnr, sp.gnb.pusch.symIdx_data, TF_received_signal, sp.gnb.pusch.s_grid, sp.gnb.pusch.sd, sp.gnb.pusch.s, ...
            %    gnb_pars, tb_pars, sp.gnb.pusch.reciever.ChEst, PuschCfg.coding.qam, j_slot, sp);
            
          

                       