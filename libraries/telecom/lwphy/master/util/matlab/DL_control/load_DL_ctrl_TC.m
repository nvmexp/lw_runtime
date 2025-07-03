%%
 % Copyright (c) 2020, LWPU CORPORATION.  All rights reserved.
 %
 % LWPU CORPORATION and its licensors retain all intellectual property
 % and proprietary rights in and to this software, related documentation
 % and any modifications thereto.  Any use, reproduction, disclosure or
 % distribution of this software and related documentation without an express
 % license agreement from LWPU CORPORATION is strictly prohibited.
 %%

function [ss_block_flag,ss_slot_idx,pdcch1_flag,pdcch2_flag, pdcch1_PL, pdcch2_PL] = load_DL_ctrl_TC(TC_str)

%inputs:
%TC_str --> string indicating test case

%outputs:
% ss_block_flag -->  0 or 1. Indicates if ss slot transmitted
% ss_slot_idx   -->   1 or 2. Indicates which ss slot transmitted
% pdcch1_flag   --> 0 or 1. Indicates if pdcch1 transmitted
% pdcch2_flag   --> 0 or 1. Indicates if pdcch2 transmitted

%%
%START

switch TC_str
    case 'DL_ctrl-TC2001'       % SSB only in the 1st half of the slot
        ss_block_flag = 1;
        ss_slot_idx = 1;
        pdcch1_flag = 0;
        pdcch2_flag = 0;
        pdcch1_PL = 0;
        pdcch2_PL = 0;
              
    case 'DL_ctrl-TC2002'       % DCI 1_1 with SSB (in the 1st slot): MCS 27 (45 bits + 24 bit CRC)
        
        ss_block_flag = 1;
        ss_slot_idx = 1;
        pdcch1_flag = 1;
        pdcch2_flag = 0; 
        pdcch1_PL = [1 0 0 0 1 1 1 0 0 1 0 1 1 0 0 1 0 1 1 0 1 1 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 1 0 1 0 0 0 0]';
        pdcch2_PL = 0;
              
     case 'DL_ctrl-TC2003'      % DCI 1_1 without SSB: MCS 27 (45 bits + 24 bit CRC)
        ss_block_flag = 0;
        ss_slot_idx = 1;
        pdcch1_flag = 1;
        pdcch2_flag = 0; 
        pdcch1_PL = [1 0 0 0 0 0 0 1 1 0 0 1 1 0 0 1 0 1 1 0 1 1 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 1 0 1 0 0 0 0]';
        pdcch2_PL = 0;
        
      case 'DL_ctrl-TC2004'      % DCI 0_0: MCS 27 (44 bits + 24 bit CRC)
        ss_block_flag = 0;
        ss_slot_idx = 1;
        pdcch1_flag = 0;
        pdcch2_flag = 1; 
        pdcch1_PL = 0;
        pdcch2_PL = [0 0 0 0 0 0 0 1 1 0 0 1 1 0 0 1 0 1 0 0 0 0 1 1 0 1 1 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0]';
                
%      case 'DL_ctrl-TC2005'      % DCI 1_1 with SSB (in the 1st slot): MCS 27 (45 bits + 24 bit CRC)+ DCI 0_0: MCS 27 (44 bits + 24 bit CRC)
%         ss_block_flag = 1;
%         ss_slot_idx = 1;
%         pdcch1_flag = 1;
%         pdcch2_flag = 1; 
%         pdcch1_PL = [1 0 0 0 1 1 1 0 0 1 0 1 1 0 0 1 0 1 1 0 1 1 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 1 0 1 0 0 0 0]';
%         pdcch2_PL = [0 0 0 0 0 0 0 1 1 0 0 1 1 0 0 1 0 1 0 0 0 0 1 1 0 1 1 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0]';   
%         
%      case 'DL_ctrl-TC2005'      % DCI 1_1 without SSB : MCS 27 (45 bits + 24 bit CRC)+ DCI 0_0: MCS 27 (44 bits + 24 bit CRC)
%         ss_block_flag = 0;
%         ss_slot_idx = 1;
%         pdcch1_flag = 1;
%         pdcch2_flag = 1; 
%         pdcch1_PL = [1 0 0 0 0 0 0 1 1 0 0 1 1 0 0 1 0 1 1 0 1 1 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 1 0 1 0 0 0 0]';
%         pdcch2_PL = [0 0 0 0 0 0 0 1 1 0 0 1 1 0 0 1 0 1 0 0 0 0 1 1 0 1 1 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0]';   
end        
     