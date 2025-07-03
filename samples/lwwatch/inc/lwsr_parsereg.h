
/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2015 by LWPU Corporation.  All rights reserved.  All information
 * contained herein is proprietary and confidential to LWPU Corporation.  Any
 * use, reproduction, or disclosure without the written permission of LWPU
 * Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */


/**************************************************************************
*                                                                         *
*    REGISTER FIELD HEADER FILE FOR THE LWSR ANALYZE WINDBG EXTENSION     *
*                                                                         *
***************************************************************************/


#ifndef LWSR_PARSEREG_H
#define LWSR_PARSEREG_H

#include "dpaux.h"

#define LW_ECHO dprintf

#define RM_LWSR_STATUS         LwS8
#define RM_AUX_STATUS          LwS32
#define RM_LWSR_OK             0
#define RM_LWSR_PORT_ERROR    -1
#define RM_LWSR_MUTEX_FAIL    -2

/*-----------------------
    SRC ID REG FIELDS
-----------------------*/
typedef struct 
{
    LwU32 lwsr_frame_stats;

    // fields
    LwU32 lwsr_tfc;
    LwU8 lwsr_tfrd;

}lwsr_inforeg_fields;


/*-----------------------
    CAP REG FIELDS
-----------------------*/
typedef struct
{
    LwU32 lwsr_cap0;
    LwU8 lwsr_cap1;
    LwU8 lwsr_cap2;
    LwU8 lwsr_cap3;
    LwU8 lwsr_bufcap;
    LwU16 lwsr_max_pixel_clk;
    LwU16 lwsr_srcl;
    LwU16 lwsr_srbl;
    LwU8 lwsr_cap4;
    LwU32 lwsr_src_id;

    // src id fields
    LwU16 lwsr_pdid;
    LwU8 lwsr_pvid;
    LwU8 lwsr_version;

    // cap0 fields
    LwU8 lwsr_nltc;
    LwU8 lwsr_ccdn;
    LwU8 lwsr_ncc;
    LwF32 lwsr_mlir;

    // cap1 fields
    LwU8 lwsr_srcc;
    LwU8 lwsr_ser;
    LwU8 lwsr_sxr;
    LwU8 lwsr_srcp;

    // cap2 fields
    LwU8 lwsr_sel;
    LwU8 lwsr_spd;
    LwU8 lwsr_std;
    LwU8 lwsr_s3d;
    LwU8 lwsr_scl;

    // cap3 fields
    LwU8 lwsr_sspc;
    LwU8 lwsr_srbr;
    LwU8 lwsr_srfc;
    LwU8 lwsr_srcf;
    LwU8 lwsr_srec;
    LwU8 lwsr_sred;
    LwU8 lwsr_srss;
    LwU8 lwsr_srgc;

    // buffer cap in MB
    LwF32 lwsr_srbs;

    // max pixel clock in MHz
    LwF32 lwsr_smpc;

    // cap4 fields
    LwU8 lwsr_srlw_support;
    LwU8 lwsr_srao_support;
    LwU8 lwsr_scsc_support; 

}lwsr_capreg_fields;



/*-----------------------
    CNTRL REG FIELDS
------------------------*/
typedef struct
{
    LwU8 lwsr_nlt_trans;
    LwU8 lwsr_src_control;
    LwU8 lwsr_misc_control1;
    LwU8 lwsr_misc_control2;
    LwU8 lwsr_intrpt_mask;
    LwU8 lwsr_intrpt_enable;
    LwU8 lwsr_resync1;
    LwU8 lwsr_resync2;

    // NLT fields
    LwU8 lwsr_nlts;

    // src control fields
    LwU8 lwsr_sec;
    LwU8 lwsr_senm;
    LwU8 lwsr_srrd_disable;
    LwU8 lwsr_srm;
    LwU8 lwsr_ser;
    LwU8 lwsr_seym;
    LwU8 lwsr_sexr;
    LwU8 lwsr_sexm;

    // misc cntrl1 fields
    LwU8 lwsr_gstl;
    LwU8 lwsr_gstm;
    LwU8 lwsr_sptl;
    LwU8 lwsr_sptm;
    LwU8 lwsr_sese;
    LwU8 lwsr_sesm;
    LwU8 lwsr_sxse;
    LwU8 lwsr_sxsm;

    // misc cntrl2 fields
    LwU8 lwsr_scnf;
    LwU8 lwsr_s3ds;
    LwU8 lwsr_sbse;
    LwU8 lwsr_sbfe;
    LwU8 lwsr_ssde;

    // intr mask fields
    LwU8 lwsr_isef;
    LwU8 lwsr_isem;
    LwU8 lwsr_isxd;
    LwU8 lwsr_isxm;
    LwU8 lwsr_isbo;
    LwU8 lwsr_isbm;
    LwU8 lwsr_isvb;
    LwU8 lwsr_isvm;

    // intr enable fields
    LwU8 lwsr_ieac;
    LwU8 lwsr_ieam;
    LwU8 lwsr_iscd;
    LwU8 lwsr_iscm;
    LwU8 lwsr_isnd;
    LwU8 lwsr_isnm;
    LwU8 lwsr_isxe;
    LwU8 lwsr_extension_isxm;

    // 345h 
    LwU8 lwsr_345h;
    LwU8 lwsr_ispe;
    LwU8 lwsr_ispm;

    // resync1 fields
    LwU8 lwsr_srrm;
    LwU8 lwsr_srcf;
    LwU8 lwsr_res2;
    LwU8 lwsr_srrd_delay;

    // resync2 fields
    LwU8 lwsr_srfe;
    LwU8 lwsr_res3;

}lwsr_controlreg_fields;


/*-----------------------
    STATUS REG FIELDS
-----------------------*/
typedef struct
{
    LwU8 lwsr_src_status1;
    LwU8 lwsr_src_status2;
    LwU8 lwsr_src_status3;
    LwU8 lwsr_interrupt_status;

    // src status1 fields
    LwU8 lwsr_srst;
    LwU8 lwsr_srsf;
    LwU8 lwsr_srbs;
    LwU8 lwsr_srbo;

    // src status2 fields
    LwU8 lwsr_sint;
    LwU8 lwsr_sinm;
    LwU8 lwsr_scom;
    LwU8 lwsr_srcv;
    LwU8 lwsr_srcr;
    LwU8 lwsr_srct;

    // src status3 fields
    LwU8 lwsr_srs4;
    LwU8 lwsr_sps4;
    LwU8 lwsr_spst;

    // interrupt status fields
    LwU8 lwsr_bit0;
    LwU8 lwsr_bit1;
    LwU8 lwsr_bit2;
    LwU8 lwsr_bit3;
    LwU8 lwsr_bit4;
    LwU8 lwsr_bit5;

}lwsr_statusreg_fields;



/*-----------------------
    TIMING REG FIELDS
-----------------------*/
typedef struct
{
    // sr mode timing regs
    LwU16 lwsr_selfrefresh_srpc;
    LwU8 lwsr_selfrefresh_35Bh;
    LwU8 lwsr_selfrefresh_365h;
    LwU8 lwsr_selfrefresh_366h;

    LwF32 lwsr_src_panel_srpc;

    LwU16 lwsr_selfrefresh_srha;
    LwU16 lwsr_selfrefresh_srhbl;
    LwU16 lwsr_selfrefresh_srhfp;
    LwU16 lwsr_selfrefresh_srhbp;
    LwU8 lwsr_selfrefresh_srhs;
    LwU8 lwsr_selfrefresh_srhb;
    LwU8 lwsr_selfrefresh_srhsp;
    LwU16 lwsr_src_panel_htotal;

    LwU16 lwsr_selfrefresh_srva;
    LwU16 lwsr_selfrefresh_srvbl;
    LwU16 lwsr_selfrefresh_srvfp;
    LwU16 lwsr_selfrefresh_srvbp;
    LwU8 lwsr_selfrefresh_srvs;
    LwU8 lwsr_selfrefresh_srvb;
    LwU8 lwsr_selfrefresh_srvsp;
    LwU16 lwsr_src_panel_vtotal;

    LwU8 lwsr_selfrefresh_srcs;
    LwU8 lwsr_selfrefresh_srfp;
    LwF32 lwsr_src_panel_refreshrate;

    // passthru mode timing regs
    LwU16 lwsr_passthrough_srpc;
    LwU8 lwsr_passthrough_373h;
    LwU8 lwsr_passthrough_37Dh;

    LwF32 lwsr_gpu_src_srpc;

    LwU16 lwsr_passthrough_ptha;
    LwU16 lwsr_passthrough_pthbl;
    LwU16 lwsr_passthrough_pthfp;
    LwU16 lwsr_passthrough_pthbp;
    LwU8 lwsr_passthrough_pths;
    LwU8 lwsr_passthrough_pthb;
    LwU8 lwsr_passthrough_pthsp;

    LwU16 lwsr_passthrough_ptva;
    LwU16 lwsr_passthrough_ptvbl;
    LwU16 lwsr_passthrough_ptvfp;
    LwU16 lwsr_passthrough_ptvbp;
    LwU8 lwsr_passthrough_ptvs;
    LwU8 lwsr_passthrough_ptvb;
    LwU8 lwsr_passthrough_ptvsp;

    // Blank timing limits
    LwU16 lwsr_vbmn;
    LwU16 lwsr_vbmx;
    LwU16 lwsr_hbmn;
    LwU16 lwsr_hbmx;

}lwsr_timingreg_fields;


/*-----------------------
    LINK REG FIELDS
-----------------------*/
typedef struct
{
// link interface gpu-src
    LwU8 lwsr_link_gpu_src;
    LwU8 lwsr_lgsl;
    LwU8 lwsr_lgsg;
    LwU8 lwsr_lgspf;

    // link interface src-panel1
    LwU8 lwsr_link_src_panel1;
    LwU8 lwsr_lspf;

    // link interface src-panel2
    LwU8 lwsr_link_src_panel2;
    LwU8 lwsr_lsvc;
    LwU8 lwsr_lshr;

    // link interface type
    LwU8 lwsr_link_type;
    LwU8 lwsr_ltyp_gpu_src;
    LwU8 lwsr_ltyp_src_panel;

    // link control
    LwU8 lwsr_link_control;
    LwU8 lwsr_lgss;
    LwU8 lwsr_lsps;

}lwsr_linkreg_fields;


/*-----------------------
     BL REG FIELDS
-----------------------*/
typedef struct
{
    // blcap1
    LwU8 lwsr_blcap1;
    LwU8 lwsr_bl_adjustment_capable;
    LwU8 lwsr_bl_pin_en_capable;
    LwU8 lwsr_bl_aux_en_capable;
    LwU8 lwsr_pslftst_pin_en_capable;
    LwU8 lwsr_pslftst_aux_en_capable;
    LwU8 lwsr_frc_en_capable;
    LwU8 lwsr_color_eng_capable;
    LwU8 lwsr_set_pwr_capable;

    // bl adjust cap
    LwU8 lwsr_blcap_adj;
    LwU8 lwsr_bl_bright_pwm_pin_capable;
    LwU8 lwsr_bl_bright_aux_set_capable;
    LwU8 lwsr_bl_bright_aux_byte_count;
    LwU8 lwsr_bl_aux_pwm_prod_capable;
    LwU8 lwsr_bl_pwm_freq_pin_pt_capable;
    LwU8 lwsr_bl_aux_freq_set_capable;
    LwU8 lwsr_bl_dynamic_capable;
    LwU8 lwsr_bl_vsync_update_capable;

    // blcap2
    LwU8 lwsr_blcap2;
    LwU8 lwsr_lcd_ovrdrv;
    LwU8 lwsr_bl_1reg_drv;
    LwU8 lwsr_bl_1str_drv;

    // blcap3
    LwU8 lwsr_blcap3;
    LwU8 lwsr_x_region_cap;
    LwU8 lwsr_y_region_cap;

    // disp panel feature cntrl
    LwU8 lwsr_disp_cntrl;
    LwU8 lwsr_bl_enable;
    LwU8 lwsr_blackvideo_enable;
    LwU8 lwsr_frc_enable;
    LwU8 lwsr_clreng_enable;
    LwU8 lwsr_vsync_bl_updt_en;

    // blmodeset
    LwU8 lwsr_bl_mode_set;
    LwU8 lwsr_bl_bright_cntrl_mode;
    LwU8 lwsr_bl_pwm_freq_pin_pt_enable;
    LwU8 lwsr_bl_aux_freq_set_enable;
    LwU8 lwsr_bl_dynamic_enable;
    LwU8 lwsr_bl_rg_bl_enable;
    LwU8 lwsr_bl_updt_britnes;

    // bl brightness 
    LwU16 lwsr_bl_brightness;
    LwU8 lwsr_bl_brightness_msb;
    LwU8 lwsr_bl_brightness_lsb;

    // pwm gen and bl cntrl stat
    LwU32 lwsr_bl_pwm;
    LwU8 lwsr_pwmgen_bit_count;
    LwU8 lwsr_pwmgen_bit_count_min;
    LwU8 lwsr_pwmgen_bit_count_max;
    LwU8 lwsr_bl_cntrl_status;

    // bl pwm frequency
    LwU8 lwsr_bl_freq_set;

    // bl brightness range
    LwU8 lwsr_bl_brighness_min;
    LwU8 lwsr_bl_brighness_max;

}lwsr_backlightreg_fields;



/*-----------------------
    DIAG REG FIELDS
-----------------------*/
typedef struct
{
    LwU32 lwsr_diagnostic_390h;
    LwU32 lwsr_dsfc;
    LwU8 lwsr_dsrd;

    LwU32 lwsr_xtcsl;
    LwU8 lwsr_drfc;

    LwU8 lwsr_diagnostic_397h;
    LwU8 lwsr_srts;

    LwU32 lwsr_diagnostic_398h;
    LwU32 lwsr_frtf;
    LwU8 lwsr_srrd_diag;

}lwsr_diagnosticreg_fields;


/*-----------------------
    MUTEX REG FIELD
-----------------------*/
typedef struct
{
    LwU32 lwsr_mutex_sprn;

}lwsr_mutex_sprn_field;


/*-----------------------
  MUTEX UNLOCK STRUCT
-----------------------*/
typedef struct
{
    LwU32 lwsr_mutex_frame_stats;
    LwU32 lwsr_mutex_src_id;

    // fields
    LwU32 lwsr_mutex_tfc;
    LwU8 lwsr_mutex_tfrd;
    LwU16 lwsr_mutex_pdid;
    LwU8 lwsr_mutex_pvid;
    LwU8 lwsr_mutex_version;

}lwsr_mutex_unlock_fields;


 /*-----------------------------
   LWSR Function declarations
 ------------------------------*/

// LWSR Mutex related Functions
RM_LWSR_STATUS lwsrMutexComputeM(LwU32, lwsr_mutex_unlock_fields *, lwsr_mutex_sprn_field *,LwU32);
RM_LWSR_STATUS lwsrCheckMutexAcquisition(LwU32 port);
LwS32 lwsrMutexSPRN(LwU32, lwsr_mutex_sprn_field *);
RM_LWSR_STATUS lwsrReadMutexReg(LwU32, lwsr_mutex_unlock_fields *, lwsr_mutex_sprn_field *);
void lwsrGenerateMessageKey(LwU8 *, lwsr_mutex_unlock_fields *, lwsr_mutex_sprn_field *);
LwU32  lwsrAuthUnlockCopyFunc(LwU8 *pBuff, LwU32 index, LwU32 size, void *pInfo);
void * lwsrAuthUnlockAllocFunc(LwU32 size);
void lwsrAuthUnlockFreeFunc(void *pAddress);

// LWSR Print Functions
void printHexString(LwU8 *, LwU32);
void printAlignedString(const char *, LwU32);

// LWSR commands entry point functions
void lwsrcap_entryfunction(LwU32 port);
void lwsrinfo_entryfunction(LwU32 port, LwU32 verbose_level);
void lwsrtiming_entryfunction(LwU32 port);
void lwsrmutex_entryfunction(LwU32 port, LwU32 sub_function_option, LwU32 dArgc);
RM_LWSR_STATUS lwsrsetrr_entryfunction(LwU32 port, LwU32 refresh_rate);

//  LWSR Register Parsing Functions
RM_LWSR_STATUS parseLWSRInfoRegs(LwU32, lwsr_inforeg_fields *);
RM_LWSR_STATUS parseLWSRCapabilityRegs(LwU32, lwsr_capreg_fields *);
RM_LWSR_STATUS parseLWSRControlRegs(LwU32, lwsr_controlreg_fields *);
RM_LWSR_STATUS parseLWSRStatusRegs(LwU32, lwsr_statusreg_fields *);
RM_LWSR_STATUS parseLWSRTimingRegs(LwU32, lwsr_timingreg_fields *);
RM_LWSR_STATUS parseLWSRLinkRegs(LwU32, lwsr_linkreg_fields *);
RM_LWSR_STATUS parseLWSRBacklightRegs(LwU32, lwsr_backlightreg_fields *);
RM_LWSR_STATUS parseLWSRDiagnosticRegs(LwU32, lwsr_diagnosticreg_fields *);



#endif
