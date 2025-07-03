/*
 * _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2021 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

/*
 * This file is used for OSFP cage bezel markings on CCI supported boards
 *
 * OSFPs supported by E4700 board: 6 (1-3, 5-7)
 * OSFPs supported for prospector: 8 (0-7)
 *
 */

#ifndef _CCI_OSFP_CAGE_BEZEL_MARKINGS_LR10_H_
#define _CCI_OSFP_CAGE_BEZEL_MARKINGS_LR10_H_

const char* cci_osfp_cage_bezel_markings_e4700_lr10[] =
{
   "INVALID", "1", "2", "3", "INVALID", "5", "6", "7"
};

const char* cci_osfp_cage_bezel_markings_prospector_lr10[6][8] =
{
   {"0:0", "0:1", "0:2", "0:3", "0:4", "0:5", "0:6", "0:7"},
   {"1:0", "1:1", "1:2", "1:3", "1:4", "1:5", "1:6", "1:7"},
   {"2:0", "2:1", "2:2", "2:3", "2:4", "2:5", "2:6", "2:7"},
   {"3:0", "3:1", "3:2", "3:3", "3:4", "3:5", "3:6", "3:7"},
   {"4:0", "4:1", "4:2", "4:3", "4:4", "4:5", "4:6", "4:7"},
   {"5:0", "5:1", "5:2", "5:3", "5:4", "5:5", "5:6", "5:7"},
};

#endif //_CCI_OSFP_CAGE_BEZEL_MARKINGS_LR10_H_
