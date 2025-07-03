
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
*      DISPLAY MESSAGES FOR THE LWSR ANALYZE WINDBG EXTENSION             *
*                                                                         *
***************************************************************************/
#include "lwsr_msg.h"

//  FUNCTION TO PRINT ANALYSIS LOG OF LWSR CAPABILITY REGISTERS

void echoLWSRCapabilityRegs(lwsr_capreg_fields *s_capreg)
{
    const char *str = NULL;

    LW_ECHO(" ********************************************************\n");
    LW_ECHO("    Based on LWSR Spec SP-04925-001_v1.0 | Feb 2015\n");
    LW_ECHO(" ********************************************************\n\n");

    LW_ECHO(" -----------------------------------------------------------------------------\n");
    LW_ECHO("                        SRC ID : DPCD addr 003A0h (0x%x)                      \n",s_capreg->lwsr_src_id);
    LW_ECHO(" -----------------------------------------------------------------------------\n");
    str = "Device ID";
    printAlignedString(str,47);
    LW_ECHO(": %04xh\n",s_capreg->lwsr_pdid);
    str = "Vendor ID";
    printAlignedString(str,47);
    LW_ECHO(": %04xh\n",s_capreg->lwsr_pvid);
    str = "Version";
    printAlignedString(str,47);
    LW_ECHO(": %04xh\n",s_capreg->lwsr_version);
    LW_ECHO("\n");

    LW_ECHO(" -----------------------------------------------------------------------------\n");
    LW_ECHO("               NLT CAPABILITIES : DPCD addr 00330h ~ 00333h (0x%x)            \n",s_capreg->lwsr_cap0);
    LW_ECHO(" -----------------------------------------------------------------------------\n");

    str = "NLT Capability";
    printAlignedString(str,47);
    if (s_capreg->lwsr_nltc == 0x01)
        LW_ECHO(": NLT Capable(%d)\n",s_capreg->lwsr_nltc);
    else
        LW_ECHO(": NLT Incapable(%d)\n",s_capreg->lwsr_nltc);

    str = "Config changes during NLT";
    printAlignedString(str,47);
    if (s_capreg->lwsr_ccdn == 0x02)
        LW_ECHO(": Allowed(%d)\n",s_capreg->lwsr_ccdn>>1);
    else
        LW_ECHO(": NOT Allowed(%d)\n",s_capreg->lwsr_ccdn>>1);

    str = "NLT capable config";
    printAlignedString(str,47);
    if (s_capreg->lwsr_ncc == 0x04)
        LW_ECHO(": Capable of NLT completion(%d)\n",s_capreg->lwsr_ncc>>2);
    else
        LW_ECHO(": NOT Capable of NLT completion(%d)\n",s_capreg->lwsr_ncc>>2);

    str = "Max LCD Image Retention Time";
    printAlignedString(str,47);
    LW_ECHO(": %.2f mS\n",s_capreg->lwsr_mlir/1000);
    LW_ECHO("\n");

    LW_ECHO(" -----------------------------------------------------------------------------\n");
    LW_ECHO("                    CAPABILITIES 1 : DPCD addr 00335h (0x%x)                  \n",s_capreg->lwsr_cap1);
    LW_ECHO(" -----------------------------------------------------------------------------\n");

    str = "Sparse refresh Capability";
    printAlignedString(str,47);
    if (s_capreg->lwsr_srcc == 0x01)
        LW_ECHO(": Supported(%d)\n",s_capreg->lwsr_srcc);
    else
        LW_ECHO(": NOT Supported(%d)\n",s_capreg->lwsr_srcc);

    str = "SR Entry Request (method)";
    printAlignedString(str,47);
    switch(s_capreg->lwsr_ser >> 1)
    {
        case 0: LW_ECHO(": Unsupported(%d)\n",s_capreg->lwsr_ser >> 1); break;
        case 1: LW_ECHO(": Inband signalling via infoframe(%d)\n",s_capreg->lwsr_ser >> 1); break;
        case 2: LW_ECHO(": Sideband signalling via Aux bus(%d)\n",s_capreg->lwsr_ser >> 1); break;
        case 3: LW_ECHO(": Both inband, sideband signalling support(%d)\n",s_capreg->lwsr_ser >> 1); break;
    }

    str = "SR Exit Request (method)";
    printAlignedString(str,47);
    switch(s_capreg->lwsr_sxr >> 3)
    {
        case 0: LW_ECHO(": Unsupported(%d)\n",s_capreg->lwsr_sxr >> 3); break;
        case 1: LW_ECHO(": Inband signalling via infoframe(%d)\n",s_capreg->lwsr_sxr >> 3); break;
        case 2: LW_ECHO(": Sideband signalling via Aux bus(%d)\n",s_capreg->lwsr_sxr >> 3); break;
        case 3: LW_ECHO(": Both inband, sideband signalling support(%d)\n",s_capreg->lwsr_sxr >> 3); break;
    }

    str = "Resync Capabilities";
    printAlignedString(str,47);
    switch(s_capreg->lwsr_srcp >> 5)
    {
        case 0: LW_ECHO(": Immediate resync supported(%d)\n",s_capreg->lwsr_srcp >> 5); break;
        case 1: LW_ECHO(": Sliding Sync alignment supported(%d)\n",s_capreg->lwsr_srcp >> 5); break;
        case 2: LW_ECHO(": Supports Frame Lock alignment supported(%d)\n",s_capreg->lwsr_srcp >> 5); break;
        case 4: LW_ECHO(": Supports Resync using Blank Stretching supported(%d)\n",s_capreg->lwsr_srcp >> 5); break;
        default:LW_ECHO(": Undefined(%d)\n",s_capreg->lwsr_srcp >> 5); break;
    }
    LW_ECHO("\n");

    LW_ECHO(" -----------------------------------------------------------------------------\n");
    LW_ECHO("                   CAPABILITIES 2 : DPCD addr 00336h (0x%x)                   \n",s_capreg->lwsr_cap2);
    LW_ECHO(" -----------------------------------------------------------------------------\n");

    str = "SR Entry latency";
    printAlignedString(str,47);
    LW_ECHO(": %d frames\n",s_capreg->lwsr_sel + 1);

    str = "SRC Panel Dither";
    printAlignedString(str,47);
    if (s_capreg->lwsr_spd == 0x10)
        LW_ECHO(": Supported(%d)\n",s_capreg->lwsr_spd >> 4);
    else
        LW_ECHO(": NOT supported(%d)\n",s_capreg->lwsr_spd >> 4);

    str = "SRC Play back";
    printAlignedString(str,47);
    if (s_capreg->lwsr_std == 0x20)
        LW_ECHO(": Supported(%d)\n",s_capreg->lwsr_std >> 5);
    else
        LW_ECHO(": NOT supported(%d)\n",s_capreg->lwsr_std >> 5);

    str = "SRC 3D Stereo";
    printAlignedString(str,47);
    if (s_capreg->lwsr_s3d == 0x40)
        LW_ECHO(": Supported(%d)\n",s_capreg->lwsr_s3d >> 6);
    else
        LW_ECHO(": NOT supported(%d)\n",s_capreg->lwsr_s3d >> 6);

    str = "SRC Compressed FB";
    printAlignedString(str,47);
    if (s_capreg->lwsr_scl == 0x80)
        LW_ECHO(": Supported(%d) [Refer SCOM to check if lwrrently active]\n",s_capreg->lwsr_scl >> 7);
    else
        LW_ECHO(": NOT supported(%d)\n",s_capreg->lwsr_scl >> 7);
    LW_ECHO("\n");


    LW_ECHO(" -----------------------------------------------------------------------------\n");
    LW_ECHO("                    CAPABILITIES 3 : DPCD addr 00337h (0x%x)                  \n",s_capreg->lwsr_cap3);
    LW_ECHO(" -----------------------------------------------------------------------------\n");

    str = "Separate PClks for passthru/SR modes";
    printAlignedString(str,47);
    if (s_capreg->lwsr_sspc == 0x01)
        LW_ECHO(": Supported(%d)\n",s_capreg->lwsr_sspc);
    else 
        LW_ECHO(": NOT supported(%d)\n",s_capreg->lwsr_sspc);

    str = "Burst Refresh mode";
    printAlignedString(str,47);
    if (s_capreg->lwsr_srbr == 0x02)
        LW_ECHO(": Supported(%d)\n",s_capreg->lwsr_srbr >> 1);
    else
        LW_ECHO(": NOT supported(%d)\n",s_capreg->lwsr_srbr >> 1);

    str = "SRC FRAME_LOCK#";
    printAlignedString(str,47);
    if (s_capreg->lwsr_srfc == 0x04)
        LW_ECHO(": Supported(%d)\n",s_capreg->lwsr_srfc >> 2);
    else
        LW_ECHO(": NOT supported(%d)\n",s_capreg->lwsr_srfc >> 2);

    str = "Configurable FRAME_LOCK";
    printAlignedString(str,47);
    if (s_capreg->lwsr_srcf == 0x08)
        LW_ECHO(": Supported(%d)\n",s_capreg->lwsr_srcf >> 3);
    else
        LW_ECHO(": NOT supported(%d)\n",s_capreg->lwsr_srcf >> 3);

    str = "SRC Interrupt on SR-Entry Caching";
    printAlignedString(str,47);
    if (s_capreg->lwsr_srec == 0x10)
        LW_ECHO(": Supported(%d)\n",s_capreg->lwsr_srec >> 4);
    else
        LW_ECHO(": NOT supported(%d)\n",s_capreg->lwsr_srec >> 4);

    str = "SRC Interrupt on SR-Entry Done";
    printAlignedString(str,47);
    if (s_capreg->lwsr_sred == 0x20)
        LW_ECHO(": Supported(%d)\n",s_capreg->lwsr_sred >> 5);
    else
        LW_ECHO(": NOT supported(%d)\n",s_capreg->lwsr_sred >> 5);

    str = "Spread Spectrum";
    printAlignedString(str,47);
    if (s_capreg->lwsr_srss == 0x40)
        LW_ECHO(": Supported(%d)\n",s_capreg->lwsr_srss >> 6);
    else
        LW_ECHO(": NOT supported(%d)\n",s_capreg->lwsr_srss >> 6);

    str = "GPU Power Control";
    printAlignedString(str,47);
    if (s_capreg->lwsr_srgc == 0x80)
        LW_ECHO(": Supported(%d)\n",s_capreg->lwsr_srgc >> 7);
    else
        LW_ECHO(": NOT supported(%d)\n",s_capreg->lwsr_srgc >> 7);

    LW_ECHO("\n");

    LW_ECHO(" -----------------------------------------------------------------------------\n");
    LW_ECHO("                CRASH SYNC CAPABILITIES : DPCD addr 0033Fh (0x%x)             \n",s_capreg->lwsr_cap4);
    LW_ECHO(" -----------------------------------------------------------------------------\n");

    str = "SR FB Retention (low-power standby mode)";
    printAlignedString(str,47);
    if (s_capreg->lwsr_srlw_support == 0x01)
        LW_ECHO(": Supported(%d)\n",s_capreg->lwsr_srlw_support);
    else
        LW_ECHO(": NOT Supported(%d)\n",s_capreg->lwsr_srlw_support);

    str = "SR Enter at Power On";
    printAlignedString(str,47);
    if (s_capreg->lwsr_srao_support == 0x02)
        LW_ECHO(": Supported(%d)\n",s_capreg->lwsr_srao_support >> 1);
    else
        LW_ECHO(": NOT Supported(%d)\n",s_capreg->lwsr_srao_support >> 1);

    str = "Crash Sync Capabilities";
    printAlignedString(str,47);
    switch(s_capreg->lwsr_scsc_support >> 2)
    {
        case 0: LW_ECHO(": Unknown(%d)\n",s_capreg->lwsr_scsc_support >> 2); break;
        case 1: LW_ECHO(": Supports Crash Sync type1(%d)\n",s_capreg->lwsr_scsc_support >> 2); break;
        case 2: LW_ECHO(": Supports Crash Sync type1 and type3(%d)\n",s_capreg->lwsr_scsc_support >> 2); break;
        case 3: LW_ECHO(": Supports combined Crash sync type1 and SR update as atomic operation(%d)\n",s_capreg->lwsr_scsc_support >> 2); break;
        case 4: 
        case 5:
        case 6: LW_ECHO(": Reserved(%d)\n",s_capreg->lwsr_scsc_support >> 2); break;
        case 7: LW_ECHO(": Supports combined Crash sync type1, type3 and SR update as atomic operation(%d)\n",s_capreg->lwsr_scsc_support >> 2); break;
    }
    LW_ECHO("     Type1  (Delayed)     => in Vblank & outside active/fetching region\n");
    LW_ECHO("     Type3 (Immediate)    => in Vblank as well as in active/fetching region\n");
    LW_ECHO("\n");

    LW_ECHO(" -----------------------------------------------------------------------------\n");
    LW_ECHO("                      BUFFER CAP : DPCD addr 00338h (0x%x)                    \n",s_capreg->lwsr_bufcap);
    LW_ECHO(" -----------------------------------------------------------------------------\n");
    str = "SR Buffer Size (panel local buffer)";
    printAlignedString(str,47);
    LW_ECHO(": %f MB (%d)\n",s_capreg->lwsr_srbs ,s_capreg->lwsr_bufcap);
    LW_ECHO("\n\n");

    LW_ECHO(" -----------------------------------------------------------------------------\n");
    LW_ECHO("               MAX PIXEL CLOCK : DPCD addr 00339h ~ 0033Ah (0x%x)             \n",s_capreg->lwsr_max_pixel_clk);
    LW_ECHO(" -----------------------------------------------------------------------------\n");
    str = "Max Pclk supported";
    printAlignedString(str,47);
    LW_ECHO(": %f MHz (%d)\n",s_capreg->lwsr_smpc ,s_capreg->lwsr_max_pixel_clk);
    LW_ECHO("\n\n");

    LW_ECHO(" -----------------------------------------------------------------------------\n");
    LW_ECHO("                CACHE LATENCY : DPCD addr 0033Bh ~ 0033Dh (0x%x)              \n",s_capreg->lwsr_srcl);
    LW_ECHO(" -----------------------------------------------------------------------------\n");
    str = "SR Request to Cache Latency (min)";
    printAlignedString(str,47);
    LW_ECHO(": %d lines\n",s_capreg->lwsr_srcl);
    LW_ECHO("\n\n");

    LW_ECHO(" -----------------------------------------------------------------------------\n");
    LW_ECHO("           BURST MODE CACHE LATENCY : DPCD addr 0033Dh ~ 0033Eh (0x%x)        \n",s_capreg->lwsr_srbl);
    LW_ECHO(" -----------------------------------------------------------------------------\n");
    str = "Burst SR Request to Cache Latency (min)";
    printAlignedString(str,47);
    if (s_capreg->lwsr_srbl == 0)
        LW_ECHO(": %d lines (%d)[SRCL used]\n",s_capreg->lwsr_srcl, s_capreg->lwsr_srbl);
    else
        LW_ECHO(": %d lines (%d)\n",s_capreg->lwsr_srbl, s_capreg->lwsr_srbl);
    LW_ECHO("\n -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-\n\n");
}


// ----------------------------------------------------------------------------------------------------------------------------


//  FUNCTION TO PRINT ANALYSIS LOG OF LWSR INFO REGISTERS

void echoLWSRInfoRegs(lwsr_inforeg_fields *s_inforeg)
{
    const char *str = NULL;

    LW_ECHO(" -----------------------------------------------------------------------------\n");
    LW_ECHO("                      FRAME STATS : DPCD addr 003A4h (0x%x)                   \n",s_inforeg->lwsr_frame_stats);
    LW_ECHO(" -----------------------------------------------------------------------------\n");

    str = "Total frame count";
    printAlignedString(str,38);
    if (s_inforeg->lwsr_tfc == 0)
        LW_ECHO(": %d frames [Machine Reset]\n",s_inforeg->lwsr_tfc);
    else
        LW_ECHO(": %d frames \n",s_inforeg->lwsr_tfc);

    if (s_inforeg->lwsr_tfrd == 1)
        LW_ECHO("Register rollover detected at 003A4h... (%d)\n",s_inforeg->lwsr_tfrd);
    /*else
        LW_ECHO("Register rollover NOT detected at 003A4h... (%d)\n",s_inforeg->lwsr_tfrd);*/
    LW_ECHO("\n\n -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-\n\n");
}


// ---------------------------------------------------------------------------------------------------------------------------------


//  FUNCTION TO PRINT ANALYSIS LOG OF LWSR CONTROL REGISTERS

void echoLWSRControlRegs(lwsr_controlreg_fields *s_cntrlreg)
{
    const char *str = NULL;

    LW_ECHO(" ********************************************************\n");
    LW_ECHO("    Based on LWSR Spec SP-04925-001_v1.0 | Feb 2015\n");
    LW_ECHO(" ********************************************************\n\n");

    LW_ECHO(" -----------------------------------------------------------------------------\n");
    LW_ECHO("                    NLT TRANSITION : DPCD addr 00334h (0x%x)                  \n",s_cntrlreg->lwsr_nlt_trans);
    LW_ECHO(" -----------------------------------------------------------------------------\n");
    str = "NLT Start";
    printAlignedString(str,38);
    if (s_cntrlreg->lwsr_nlts == 1)
        LW_ECHO(": (%d)Source requesting initiate NLT protocol. Sending idle pattern.\n",s_cntrlreg->lwsr_nlts);
    if (s_cntrlreg->lwsr_nlts == 0)
        LW_ECHO(": (%d)Sink successfully synchronized to link.\n",s_cntrlreg->lwsr_nlts);
    LW_ECHO("\n\n");

    LW_ECHO(" -----------------------------------------------------------------------------\n");
    LW_ECHO("                     STATE CONTROL : DPCD addr 00340h (0x%x)                  \n",s_cntrlreg->lwsr_src_control);
    LW_ECHO(" -----------------------------------------------------------------------------\n");

    str = "SR feature Enable Control";
    printAlignedString(str,38);
    if (s_cntrlreg->lwsr_sec == 0x01)
        LW_ECHO(": Enabled(%d)\n",s_cntrlreg->lwsr_sec);
    else
        LW_ECHO(": NOT enabled(%d)\n",s_cntrlreg->lwsr_sec);

    /*LW_ECHO("SR Enable Mask : %d ",s_cntrlreg->lwsr_senm >> 1);
    LW_ECHO("[Always reads as 0]\n");*/

    str = "Glitch-free SR Resync";
    printAlignedString(str,38);
    if (s_cntrlreg->lwsr_srrd_disable == 0x04)
        LW_ECHO(": Enabled(%d)\n",s_cntrlreg->lwsr_srrd_disable >> 2);
    else
        LW_ECHO(": NOT enabled(%d)\n",s_cntrlreg->lwsr_srrd_disable >> 2);

    /*LW_ECHO("SR Resync Mask : %d ",s_cntrlreg->lwsr_srm >> 3);
    LW_ECHO("[Always reads as 0]\n");*/

    str = "SR Entry Request";
    printAlignedString(str,38);
    if (s_cntrlreg->lwsr_sese == 0x10)
    {
        if (s_cntrlreg->lwsr_ser == 0x10)
            LW_ECHO(": Request in progress (%d)\n",s_cntrlreg->lwsr_ser >> 4);
        else    
            LW_ECHO(": Request complete OR no request (%d)\n",s_cntrlreg->lwsr_ser >> 4);
    }
    else
        LW_ECHO(": Bit field invalid as sideband control is disabled\n");

    /*LW_ECHO("SR Entry Mask : %d ",s_cntrlreg->lwsr_seym >> 5);
    LW_ECHO("[Always reads as 0]\n");*/

    str = "SR Exit Request";
    printAlignedString(str,38);
    if (s_cntrlreg->lwsr_sxse == 0x40)
    {
        if (s_cntrlreg->lwsr_sexr == 0x40)
            LW_ECHO(": Request in progress (%d)\n",s_cntrlreg->lwsr_sexr >> 6);
        else
            LW_ECHO(": Request complete OR no request (%d)\n",s_cntrlreg->lwsr_sexr >> 6);
    }
    else
        LW_ECHO(": Bit field invalid as sideband control is disabled\n");

    /*LW_ECHO("SR Exit Mask : %d ",s_cntrlreg->lwsr_sexm >> 7);
    LW_ECHO("[Always reads as 0]\n");*/
    LW_ECHO("\n");

    LW_ECHO(" -----------------------------------------------------------------------------\n");
    LW_ECHO("                    MISC CONTROL1 : DPCD addr 00341h (0x%x)                   \n",s_cntrlreg->lwsr_misc_control1);
    LW_ECHO(" -----------------------------------------------------------------------------\n");

    str = "GPU-SRC Timing Latch";
    printAlignedString(str,38);
    if (s_cntrlreg->lwsr_gstl == 0x01)
        LW_ECHO(": Changes applied immediately in current frame(%d)\n",s_cntrlreg->lwsr_gstl);
    else
        LW_ECHO(": Changes buffered & applied from next frame(%d)\n",s_cntrlreg->lwsr_gstl);

    /*LW_ECHO("GS Timing Latch Mask : %d ",s_cntrlreg->lwsr_gstm >> 1);
    LW_ECHO("[Always reads as 0]\n");*/

    str = "SRC-Panel Timing Latch";
    printAlignedString(str,38);
    if (s_cntrlreg->lwsr_sptl == 0x04)
        LW_ECHO(": Changes applied immediately in current frame(%d)\n",s_cntrlreg->lwsr_sptl >> 2);
    else
        LW_ECHO(": Changes buffered & applied from next frame(%d)\n",s_cntrlreg->lwsr_sptl >> 2);

    /*LW_ECHO("SP Timing Latch Mask : %d ",s_cntrlreg->lwsr_sptm >> 3);
    LW_ECHO("[Always reads as 0]\n");*/

    str = "SR Entry (method) Select";
    printAlignedString(str,38);
    if (s_cntrlreg->lwsr_sese == 0x10)
        LW_ECHO(": Sideband control enabled(%d)\n",s_cntrlreg->lwsr_sese >> 4);
    else
        LW_ECHO(": Inband control enabled(%d)\n",s_cntrlreg->lwsr_sese >> 4);

    /*LW_ECHO("Sideband Entry Mask : %d ",s_cntrlreg->lwsr_sesm >> 5);
    LW_ECHO("[Always reads as 0]\n");*/

    str = "SR Exit (method) Select";
    printAlignedString(str,38);
    if (s_cntrlreg->lwsr_sxse == 0x40)
        LW_ECHO(": Sideband control enabled(%d)\n",s_cntrlreg->lwsr_sxse >> 6);
    else
        LW_ECHO(": Inband control enabled(%d)\n",s_cntrlreg->lwsr_sxse >> 6);

    /*LW_ECHO("Sideband Exit Mask : %d ",s_cntrlreg->lwsr_sxsm >> 7);
    LW_ECHO("[Always reads as 0]\n");*/
    LW_ECHO("\n");

    LW_ECHO(" -----------------------------------------------------------------------------\n");
    LW_ECHO("                     MISC CONTROL2 : DPCD addr 00342h (0x%x)                  \n",s_cntrlreg->lwsr_misc_control2);
    LW_ECHO(" -----------------------------------------------------------------------------\n");

    str = "Capture Frames";
    printAlignedString(str,38);
    LW_ECHO(": %d frames\n",s_cntrlreg->lwsr_scnf+1);

    str = "3DStereo";
    printAlignedString(str,38);
    if (s_cntrlreg->lwsr_s3ds == 0x10)
        LW_ECHO(": Enabled(%d)\n",s_cntrlreg->lwsr_s3ds >> 4);
    else
        LW_ECHO(": NOT enabled(%d)\n",s_cntrlreg->lwsr_s3ds >> 4);

    str = "Buffered/Burst Refresh Mode";
    printAlignedString(str,38);
    if (s_cntrlreg->lwsr_sbse == 0x20)
        LW_ECHO(": Enabled(%d)\n",s_cntrlreg->lwsr_sbse >> 5);
    else
        LW_ECHO(": NOT enabled(%d)\n",s_cntrlreg->lwsr_sbse >> 5);

    str = "GPU or SRC FRAME_LOCK# select";
    printAlignedString(str,38);
    if (s_cntrlreg->lwsr_sbfe == 0x40)
        LW_ECHO(": SRC_FRAME_LOCK# (%d) -> src to synchronize frame on gpu vsync\n",s_cntrlreg->lwsr_sbfe >> 6);
    else
        LW_ECHO(": GPU_FRAME_LOCK# (%d) -> src to synchronize frame on gpu frame lock\n",s_cntrlreg->lwsr_sbfe >> 6);

    str = "SRC Side Dithering";
    printAlignedString(str,38);
    if (s_cntrlreg->lwsr_ssde == 0x80)
        LW_ECHO(": Enabled(%d)\n",s_cntrlreg->lwsr_ssde >> 7);
    else
        LW_ECHO(": NOT enabled(%d)\n",s_cntrlreg->lwsr_ssde >> 7);
    LW_ECHO("\n");

    LW_ECHO(" -----------------------------------------------------------------------------\n");
    LW_ECHO("                    INTERRUPT MASK : DPCD addr 00343h (0x%x)                  \n",s_cntrlreg->lwsr_intrpt_mask);
    LW_ECHO(" -----------------------------------------------------------------------------\n");

    str = "Interrupt on SR Entry Failure";
    printAlignedString(str,38);
    LW_ECHO(": (%d)",s_cntrlreg->lwsr_isef);
    if (s_cntrlreg->lwsr_isef == 0x01)
        LW_ECHO("IRQ enabled - SR Entry request failure\n");
    else
        LW_ECHO("IRQ disabled\n");

    /*LW_ECHO("SR Entry IRQ Mask :  %d ",s_cntrlreg->lwsr_isem >> 1);
    LW_ECHO("[Always reads as 0]\n");*/

    str = "Interrupt on SR Exit Resync Done";
    printAlignedString(str,38);
    LW_ECHO(": (%d)",s_cntrlreg->lwsr_isxd >> 2);
    if (s_cntrlreg->lwsr_isxd == 0x04)
        LW_ECHO("IRQ enabled - SR exit and resync complete\n");
    else
        LW_ECHO("IRQ disabled\n");

    /*LW_ECHO("SR Exit/Resync Done IRQ Mask :  %d ",s_cntrlreg->lwsr_isxm >> 3);
    LW_ECHO("[Always reads as 0]\n");*/

    str = "Interrupt on SR Buffer Overflow";
    printAlignedString(str,38);
    LW_ECHO(": (%d)",s_cntrlreg->lwsr_isbo >> 4);
    if (s_cntrlreg->lwsr_isbo == 0x10)
        LW_ECHO("IRQ enabled - SR buffer overflow !\n");
    else
        LW_ECHO("IRQ disabled\n");

    /*LW_ECHO("SR Buffer Overflow IRQ Mask :  %d ",s_cntrlreg->lwsr_isbm >> 5);
    LW_ECHO("[Always reads as 0]\n");*/

    str = "Interrupt on SR Vertical Blank";
    printAlignedString(str,38);
    LW_ECHO(": (%d)",s_cntrlreg->lwsr_isvb >> 6);
    if (s_cntrlreg->lwsr_isvb == 0x40)
        LW_ECHO("IRQ enabled - in SR vertical blank interval\n");
    else
        LW_ECHO("IRQ disabled\n");

    /*LW_ECHO("SR Vertical Blank IRQ Mask :  %d ",s_cntrlreg->lwsr_isvm >> 7);
    LW_ECHO("[Always reads as 0]\n");*/

    LW_ECHO("\n");

    LW_ECHO(" -----------------------------------------------------------------------------\n");
    LW_ECHO("                   INTERRUPT ENABLE : DPCD addr 00344h (0x%x)                 \n",s_cntrlreg->lwsr_intrpt_enable);
    LW_ECHO(" -----------------------------------------------------------------------------\n");

    str = "Interrupt Enable control";
    printAlignedString(str,38);
    LW_ECHO(": (%d)",s_cntrlreg->lwsr_ieac);
    if (s_cntrlreg->lwsr_ieac == 0x01)
        LW_ECHO("HPD/IRQ Interrupt line enabled\n");
    else
        LW_ECHO("HPD/IRQ Interrupt line disabled\n");

    /*LW_ECHO("Interrupt Enable Mask :  %d ",s_cntrlreg->lwsr_ieam >> 1);
    LW_ECHO("[Always reads as 0]\n");*/

    str = "Interrupt on SR Entry Caching";
    printAlignedString(str,38);
    LW_ECHO(": (%d)",s_cntrlreg->lwsr_iscd >> 2);
    if (s_cntrlreg->lwsr_iscd == 0x04)
        LW_ECHO("IRQ enabled - SR entry request and frame caching in progress\n");
    else
        LW_ECHO("IRQ disabled\n");

    /*LW_ECHO("SR Entry Caching IRQ Mask :  %d ",s_cntrlreg->lwsr_iscm >> 3);
    LW_ECHO("[Always reads as 0]\n");*/

    str = "Interrupt on SR Entry Done";
    printAlignedString(str,38);
    LW_ECHO(": (%d)",s_cntrlreg->lwsr_isnd >> 4);
    if (s_cntrlreg->lwsr_isnd == 0x10)
        LW_ECHO("IRQ enabled - SR entry and frame caching complete\n");
    else
        LW_ECHO("IRQ disabled\n");

    /*LW_ECHO("SR Entry Done IRQ Mask :  %d ",s_cntrlreg->lwsr_isnm >> 5);
    LW_ECHO("[Always reads as 0]\n");*/

    str = "Interrupt on Extension Event";
    printAlignedString(str,38);
    LW_ECHO(": (%d)",s_cntrlreg->lwsr_isxe >> 6);
    if (s_cntrlreg->lwsr_isxe == 0x40)
        LW_ECHO("IRQ enabled - Event signal from extension interface\n");
    else
        LW_ECHO("IRQ disabled\n");

    /*LW_ECHO("SR Vertical Blank IRQ Mask :  %d ",s_cntrlreg->lwsr_extension_isxm >> 7);
    LW_ECHO("[Always reads as 0]\n");*/
    LW_ECHO("\n");

    LW_ECHO(" -----------------------------------------------------------------------------\n");
    LW_ECHO("                             DPCD addr 00345h (0x%x)                          \n",s_cntrlreg->lwsr_345h);
    LW_ECHO(" -----------------------------------------------------------------------------\n");

    str = "Interrupt on Panel Event";
    printAlignedString(str,38);
    if(s_cntrlreg->lwsr_ispe)
        LW_ECHO("Enable assert on TCON-external IRQ event\n");
    else
        LW_ECHO("TCON-external IRQ disabled\n");

    /*LW_ECHO("Panel Event IRQ Mask :  %d ",s_cntrlreg->lwsr_ispm >> 1);
    LW_ECHO("[Always reads as 0]\n");*/
    LW_ECHO("\n");

    LW_ECHO(" -----------------------------------------------------------------------------\n");
    LW_ECHO("                   RESYNC CONTROL 1 : DPCD addr 00346h (0x%x)                 \n",s_cntrlreg->lwsr_resync1);
    LW_ECHO(" -----------------------------------------------------------------------------\n");

    str = "Resync Method";
    printAlignedString(str,38);
    switch(s_cntrlreg->lwsr_srrm)
    {
        case 0: LW_ECHO(": Immediate Resync(%d)\n",s_cntrlreg->lwsr_srrm); break;
        case 1: LW_ECHO(": Sliding Sync alignment(%d)\n",s_cntrlreg->lwsr_srrm); break;
        case 2: LW_ECHO(": FRAME_LOCK# alignment(%d)\n",s_cntrlreg->lwsr_srrm); break;
        case 3: LW_ECHO(": Blank Stretching alignment(%d)\n",s_cntrlreg->lwsr_srrm); break;
        case 4: LW_ECHO(": Double-buffering using extra SRC Frame Buffer(%d)\n",s_cntrlreg->lwsr_srrm); break;
        case 5:
        case 6:
        case 7: LW_ECHO(": RESERVED for future use\n"); break;
    }

    str = "FRAME_LOCK# generation";
    printAlignedString(str,38);
    if (s_cntrlreg->lwsr_srcf == 0x08)
        LW_ECHO(": Generated continuously(%d)\n",s_cntrlreg->lwsr_srcf >> 3);
    else
        LW_ECHO(": Single shot alignment. Generated once(%d)\n",s_cntrlreg->lwsr_srcf >> 3);

    str = "Resync Delay";
    printAlignedString(str,38);
    LW_ECHO(": %d frames\n",s_cntrlreg->lwsr_srrd_delay >> 6);

    LW_ECHO("\n");

    LW_ECHO(" -----------------------------------------------------------------------------\n");
    LW_ECHO("                   RESYNC CONTROL 2 : DPCD addr 00347h (0x%x)                 \n",s_cntrlreg->lwsr_resync2);
    LW_ECHO(" -----------------------------------------------------------------------------\n");

    str = "FRAME_LOCK# Edge Select";
    printAlignedString(str,38);
    switch(s_cntrlreg->lwsr_srfe)
    {
        case 0: LW_ECHO(": Leading edge of SRC VSync [default](%d)\n",s_cntrlreg->lwsr_srfe); break;
        case 1: LW_ECHO(": Trailing edge of SRC VSync [start of vertical back porch](%d)\n",s_cntrlreg->lwsr_srfe); break;
        case 2: LW_ECHO(": Leading edge of SRC VBlank [start of vertical front porch](%d)\n",s_cntrlreg->lwsr_srfe); break;
        case 3: LW_ECHO(": Trailing edge of SRC VBlank [end of vertical back porch](%d)\n",s_cntrlreg->lwsr_srfe); break;
    }
    LW_ECHO("\n");
}


// ---------------------------------------------------------------------------------------------------------------------------------


//  FUNCTION TO PRINT ANALYSIS LOG OF LWSR STATUS REGISTERS

void echoLWSRStatusRegs(lwsr_statusreg_fields *s_statusreg, lwsr_controlreg_fields *s_cntrlreg)
{
    const char *str = NULL;

    LW_ECHO(" -----------------------------------------------------------------------------\n");
    LW_ECHO("                     SRC STATUS 1 : DPCD addr 00348h (0x%x)                   \n",s_statusreg->lwsr_src_status1);
    LW_ECHO(" -----------------------------------------------------------------------------\n");

    str = "Self Refresh State";
    printAlignedString(str,38);
    switch(s_statusreg->lwsr_srst)
    {
        case 0: LW_ECHO(": SR Disabled/Offline (%d)\n",s_statusreg->lwsr_srst); break;
        case 1: LW_ECHO(": SR Idle/Pass-through (%d)\n",s_statusreg->lwsr_srst); break;
        case 2: LW_ECHO(": SR Entry Triggered (%d)\n",s_statusreg->lwsr_srst); break;
        case 3: LW_ECHO(": SR Entry Caching (%d)\n",s_statusreg->lwsr_srst); break;
        case 4: LW_ECHO(": SR Entry Ready (%d)\n",s_statusreg->lwsr_srst); break;
        case 5: LW_ECHO(": SR Active (%d)\n",s_statusreg->lwsr_srst); break;
        case 6: LW_ECHO(": SR Exit Triggered (%d)\n",s_statusreg->lwsr_srst); break;
        case 7: LW_ECHO(": SR Exit Resync (%d)\n",s_statusreg->lwsr_srst); break;
    }

    str = "SR State Failure Condition";
    printAlignedString(str,38);
    switch(s_statusreg->lwsr_srsf >> 3)
    {
        case 0: LW_ECHO(": No error (%d)\n",s_statusreg->lwsr_srsf >> 3); break;
        case 1: LW_ECHO(": SR Entry Failure (%d)\n",s_statusreg->lwsr_srsf >> 3); break;
        case 2: LW_ECHO(": SR Resync Failure (%d)\n",s_statusreg->lwsr_srsf >> 3); break;
        case 3: LW_ECHO(": Re-Enter requested during a Resync (%d)\n",s_statusreg->lwsr_srsf >> 3); break;
    }

    str = "Burst Refresh State";
    printAlignedString(str,38);
    switch(s_statusreg->lwsr_srbs >> 5)
    {
        case 1: LW_ECHO(": SR Burst Resync in progress, displaying existing cached frame(%d) [if SRST and SBSE are set]\n",s_statusreg->lwsr_srbs >> 5); break;
        case 0:
        case 2:
        case 3: LW_ECHO(": Reserved (%d)\n",s_statusreg->lwsr_srbs >> 5); break;
    }

    str = "Buffer overrun";
    printAlignedString(str,38);
    if (s_statusreg->lwsr_srbo == 0x80)
        LW_ECHO(": SRC FB overflow [with pixels from GPU] (%d)\n",s_statusreg->lwsr_srbo >> 7);
    else
        LW_ECHO(": No FB overflow (%d)\n",s_statusreg->lwsr_srbo >> 7);
    LW_ECHO("\n");

    LW_ECHO(" -----------------------------------------------------------------------------\n");
    LW_ECHO("                     SRC STATUS 2 : DPCD addr 00349h (0x%x)                   \n",s_statusreg->lwsr_src_status2);
    LW_ECHO(" -----------------------------------------------------------------------------\n");

    str = "Interrupt Status";
    printAlignedString(str,38);
    if (s_statusreg->lwsr_sint == 0x01)
        LW_ECHO(": Interrupt HPD/IRQ line asserted due to SRC generated event(%d)\n",s_statusreg->lwsr_sint);
    else
        LW_ECHO(": No interrupt condition(%d)\n",s_statusreg->lwsr_sint);

    /*LW_ECHO("Interrupt Mask : %d",s_statusreg->lwsr_sinm >> 1);
    LW_ECHO("[Always reads as 0]\n");*/

    str = "FB Compression";
    printAlignedString(str,38);
    if (s_statusreg->lwsr_scom == 0x04)
        LW_ECHO(": Active(%d)\n",s_statusreg->lwsr_scom >> 2);
    else
        LW_ECHO(": Inactive(%d)\n",s_statusreg->lwsr_scom >> 2);

    str = "During blank";
    printAlignedString(str,38);
    switch(s_statusreg->lwsr_srcv >> 3)
    {
        case 0: LW_ECHO(": SRC Raster within Active Display period(%d)\n",s_statusreg->lwsr_srcv >> 3); break;
        case 1: LW_ECHO(": SRC Raster scan-out is within Hblank(%d)\n",s_statusreg->lwsr_srcv >> 3); break;
        case 2: LW_ECHO(": SRC Raster scan-out is within Vblank(%d)\n",s_statusreg->lwsr_srcv >> 3); break;
        case 3: LW_ECHO(": Undefined\n"); break;
    }

    str = "Panel-driven Refresh Required";
    printAlignedString(str,38);
    switch(s_statusreg->lwsr_srcr >> 5)
    {
        case 0: LW_ECHO(": No action required (%d)\n",s_statusreg->lwsr_srcr >> 5); break;
        case 1: LW_ECHO(": GC driven refresh requested by panel(%d)\n",s_statusreg->lwsr_srcr >> 5); break;
    }

    str = "Panel-component Event";
    printAlignedString(str,38);
    switch(s_statusreg->lwsr_srct >> 6)
    {
        case 0: LW_ECHO(": No action required (%d)\n",s_statusreg->lwsr_srct >> 6); break;
        case 1: LW_ECHO(": GC action requested by panel for (external events) (%d)\n",s_statusreg->lwsr_srct >> 6); break;
    }

    LW_ECHO("\n");

    LW_ECHO(" -----------------------------------------------------------------------------\n");
    LW_ECHO("                       SRC STATUS 3 : DPCD addr 0034Ah (0x%x)                  \n",s_statusreg->lwsr_src_status3);
    LW_ECHO(" -----------------------------------------------------------------------------\n");

    str = "SRC FB Retention settings";
    printAlignedString(str,38);
    switch(s_statusreg->lwsr_srs4)
    {
        case 0: LW_ECHO(": No retention, FB powered down with SRC(%d)\n",s_statusreg->lwsr_srs4); break;
        case 1: LW_ECHO(": SRC retained while powering down(%d)\n",s_statusreg->lwsr_srs4); break;
        case 2: LW_ECHO(": SRC retained while powering down(%d)\n",s_statusreg->lwsr_srs4);
                str = "";
                printAlignedString(str,38);
                LW_ECHO("(%d)On power-on SRC : \n",s_statusreg->lwsr_srs4);
                str = "";
                printAlignedString(str,38);
                LW_ECHO("\t\ta)enables panel VDD\n");
                str = "";
                printAlignedString(str,38);
                LW_ECHO("\t\tb)restores SRC-panel timings\n");
                str = "";
                printAlignedString(str,38);
                LW_ECHO("\t\tc)enables scan-out\n");
                str = "";
                printAlignedString(str,38);
                LW_ECHO("\t\td)enables backlight\n"); break;
        case 3: LW_ECHO(": Reserved(%d)\n",s_statusreg->lwsr_srs4); break;
    }

    str = "SRC LCD Extended Ilwersion Pattern";
    printAlignedString(str,38);
    if (s_statusreg->lwsr_sps4 >> 6 == 0)
        LW_ECHO(": SRC supports only 2-phase ilwersion pattern(%d)\n",s_statusreg->lwsr_sps4 >> 5);
    else 
        LW_ECHO(": SRC is using 3-phase/4-phase ilwersion patterns(%d)\n",s_statusreg->lwsr_sps4 >> 5);

    str = "SRC LCD Ilwersion Pattern";
    printAlignedString(str,38);
    if (s_statusreg->lwsr_spst >> 7 == 0)
        LW_ECHO(": First pixel of frame has negative polarity(%d)\n",s_statusreg->lwsr_spst >> 7);
    else if (s_statusreg->lwsr_spst >> 7 == 1)
        LW_ECHO(": First pixel of frame has positive polarity(%d)\n",s_statusreg->lwsr_spst >> 7);

    LW_ECHO("\n\n");

    LW_ECHO(" -----------------------------------------------------------------------------\n");
    LW_ECHO("                  INTERRUPT STATUS : DPCD addr 0034bh (0x%x)                  \n",s_statusreg->lwsr_interrupt_status);
    LW_ECHO(" -----------------------------------------------------------------------------\n");

    str = "SRC Interrupt Source";
    printAlignedString(str,38);
    if (s_statusreg->lwsr_bit0 == 0x01)
    {    
        if (s_cntrlreg->lwsr_iscd >> 2 == 0x01)
            LW_ECHO(": IRQ Assert => SR ENTRY, FRAME CACHING IN PROGRESS !\n");
        if (s_cntrlreg->lwsr_isnd >> 4 == 0x01)
            LW_ECHO(": IRQ Assert => SR ENTRY, FRAME CACHING COMPLETE !\n");
        if (s_cntrlreg->lwsr_isxd >> 2 == 0x01)
            LW_ECHO(": IRQ Assert => SR EXIT AND RESYNC COMPLETE !\n");

        LW_ECHO(" For more information : Read 'Self Refresh State' (SRST) field,\n 'Burst Refresh State' (SRBS) field\n");
    }

    else if (s_statusreg->lwsr_bit1 == 0x02)
    {
        LW_ECHO(": IRQ Assert => SR ENTRY REQUEST CANNOT BE COMPLETED !\n");
        LW_ECHO(" For more information : Read 'SR State Failure Condition' (SRSF) field\n");
    }

    else if (s_statusreg->lwsr_bit2 == 0x04)
    {
        LW_ECHO(": IRQ Assert => SR BUFFER OVERFLOW WITH GPU PIXELS !\n");
        LW_ECHO(" For more information : Read 'Buffer Overrun' (SRBO) field\n");
    }

    else if (s_statusreg->lwsr_bit3 == 0x08)
    {
        LW_ECHO(": IRQ Assert => SRC AT END OF ACTIVE RASTER & START OF SR VERTICAL BLANK !\n");
        LW_ECHO(" For more information : Read 'During Blank' (SRCV) field\n");
    }

    else if (s_statusreg->lwsr_bit4 == 0x10)
        LW_ECHO(": IRQ Assert => EXTENSION EVENT !\n");

    else if (s_statusreg->lwsr_bit5 == 0x20)
    {
        LW_ECHO(": IRQ Assert => PANEL DRIVEN EVENT !\n");
        LW_ECHO(" For more information : Read SRCR and SRCT fields\n");
    }
    else
        LW_ECHO(": No IRQ asserted\n");
    LW_ECHO("\n\n -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-\n\n");
}


// ---------------------------------------------------------------------------------------------------------------------------------


//  FUNCTION TO PRINT ANALYSIS LOG OF LWSR TIMING REGISTERS

void echoLWSRTimingRegs(lwsr_timingreg_fields *s_timingreg)
{
    const char *str = NULL;

    LW_ECHO(" -----------------------------------------------------------------------------\n");
    LW_ECHO("              SELF REFRESH MODE TIMING : DPCD addr 00350h ~ 00367h            \n");
    LW_ECHO(" -----------------------------------------------------------------------------\n");

    str = "SR mode (SRC-Panel) Pixel Clock";
    printAlignedString(str,38);
    LW_ECHO(": %f MHz(%d)\n",s_timingreg->lwsr_src_panel_srpc,s_timingreg->lwsr_selfrefresh_srpc);

    str = "Horizontal Total";
    printAlignedString(str,38);
    LW_ECHO(": %d lines\n",s_timingreg->lwsr_src_panel_htotal);

    str = "Horizontal Active";
    printAlignedString(str,38);
    LW_ECHO(": %d pixels\n",s_timingreg->lwsr_selfrefresh_srha);

    str = "Horizontal Blanking";
    printAlignedString(str,38);
    LW_ECHO(": %d pixels\n",s_timingreg->lwsr_selfrefresh_srhbl);

    str = "Horizontal Front Porch";
    printAlignedString(str,38);
    LW_ECHO(": %d pixels\n",s_timingreg->lwsr_selfrefresh_srhfp);

    str = "Horizontal Back Porch";
    printAlignedString(str,38);
    LW_ECHO(": %d pixels\n",s_timingreg->lwsr_selfrefresh_srhbp);

    str = "Horizontal Sync";
    printAlignedString(str,38);
    LW_ECHO(": %d pixels\n",s_timingreg->lwsr_selfrefresh_srhs);

    str = "Horizontal Border";
    printAlignedString(str,38);
    LW_ECHO(": %d pixels\n",s_timingreg->lwsr_selfrefresh_srhb);

    str = "Horizontal Sync Polarity";
    printAlignedString(str,38);
    if ((s_timingreg->lwsr_selfrefresh_srhsp >> 7) == 0x01)
        LW_ECHO(": Positive(%d)\n",s_timingreg->lwsr_selfrefresh_srhsp >> 7);
    else
        LW_ECHO(": Negative(%d)\n",s_timingreg->lwsr_selfrefresh_srhsp >> 7);

    // -------------------------------------------------------------------------------------------

    str = "Vertical Total";
    printAlignedString(str,38);
    LW_ECHO(": %d lines\n",s_timingreg->lwsr_src_panel_vtotal);

    str = "Vertical Active";
    printAlignedString(str,38);
    LW_ECHO(": %d lines\n",s_timingreg->lwsr_selfrefresh_srva);

    str = "Vertical Blanking";
    printAlignedString(str,38);
    LW_ECHO(": %d lines\n",s_timingreg->lwsr_selfrefresh_srvbl);

    str = "Vertical Front Porch";
    printAlignedString(str,38);
    LW_ECHO(": %d lines\n",s_timingreg->lwsr_selfrefresh_srvfp);

    str = "Vertical Back Porch";
    printAlignedString(str,38);
    LW_ECHO(": %d lines\n",s_timingreg->lwsr_selfrefresh_srvbp);

    str = "Vertical Sync";
    printAlignedString(str,38);
    LW_ECHO(": %d lines\n",s_timingreg->lwsr_selfrefresh_srvs);

    str = "Vertical Border";
    printAlignedString(str,38);
    LW_ECHO(": %d lines\n",s_timingreg->lwsr_selfrefresh_srvb);

    str = "Vertical Sync Polarity";
    printAlignedString(str,38);
    if (s_timingreg->lwsr_selfrefresh_srvsp >> 7 == 0x01)
        LW_ECHO(": Positive(%d)\n",s_timingreg->lwsr_selfrefresh_srvsp >> 7);
    else
        LW_ECHO(": Negative(%d)\n",s_timingreg->lwsr_selfrefresh_srvsp >> 7);
    LW_ECHO("\n");

    str = "SRC-Panel Refresh Rate";
    printAlignedString(str,38);
    LW_ECHO(": %0.2f Hz\n",s_timingreg->lwsr_src_panel_refreshrate);

    LW_ECHO(" -----------------------------------------------------------------------------\n");
    LW_ECHO("             PASS THROUGH MODE TIMING : DPCD addr 00368h ~ 0037Fh             \n");
    LW_ECHO(" -----------------------------------------------------------------------------\n");

    str = "PT mode (GPU-SRC) Pixel Clock";
    printAlignedString(str,38);
    LW_ECHO(": %f MHz (%d)\n",s_timingreg->lwsr_gpu_src_srpc, s_timingreg->lwsr_passthrough_srpc);

    str = "Horizontal Active";
    printAlignedString(str,38);
    LW_ECHO(": %d pixels\n",s_timingreg->lwsr_passthrough_ptha);

    str = "Horizontal Blanking";
    printAlignedString(str,38);
    LW_ECHO(": %d pixels\n",s_timingreg->lwsr_passthrough_pthbl);

    str = "Horizontal Front Porch";
    printAlignedString(str,38);
    LW_ECHO(": %d pixels\n",s_timingreg->lwsr_passthrough_pthfp);

    str = "Horizontal Back Porch";
    printAlignedString(str,38);
    LW_ECHO(": %d pixels\n",s_timingreg->lwsr_passthrough_pthbp);

    str = "Horizontal Sync";
    printAlignedString(str,38);
    LW_ECHO(": %d pixels\n",s_timingreg->lwsr_passthrough_pths);

    str = "Horizontal Border";
    printAlignedString(str,38);
    LW_ECHO(": %d pixels\n",s_timingreg->lwsr_passthrough_pthb);

    str = "Horizontal Sync Polarity";
    printAlignedString(str,38);
    if (s_timingreg->lwsr_passthrough_pthsp >> 7 == 0x01)
        LW_ECHO(": Positive(%d)\n",s_timingreg->lwsr_passthrough_pthsp >> 7);
    else
        LW_ECHO(": Negative(%d)\n",s_timingreg->lwsr_passthrough_pthsp >> 7);

// -------------------------------------------------------------------------------------------------

    str = "Vertical Active";
    printAlignedString(str,38);
    LW_ECHO(": %d lines\n",s_timingreg->lwsr_passthrough_ptva);

    str = "Vertical Blanking";
    printAlignedString(str,38);
    LW_ECHO(": %d lines\n",s_timingreg->lwsr_passthrough_ptvbl);

    str = "Vertical Front Porch";
    printAlignedString(str,38);
    LW_ECHO(": %d lines\n",s_timingreg->lwsr_passthrough_ptvfp);

    str = "Vertical Back Porch";
    printAlignedString(str,38);
    LW_ECHO(": %d lines\n",s_timingreg->lwsr_passthrough_ptvbp);

    str = "Vertical Sync";
    printAlignedString(str,38);
    LW_ECHO(": %d lines\n",s_timingreg->lwsr_passthrough_ptvs);

    str = "Vertical Border";
    printAlignedString(str,38);
    LW_ECHO(": %d lines\n",s_timingreg->lwsr_passthrough_ptvb);

    str = "Vertical Sync Polarity";
    printAlignedString(str,38);
    if (s_timingreg->lwsr_passthrough_ptvsp >> 7 == 0x01)
        LW_ECHO(": Positive(%d)\n",s_timingreg->lwsr_passthrough_ptvsp >> 7);
    else
        LW_ECHO(": Negative(%d)\n",s_timingreg->lwsr_passthrough_ptvsp >> 7);
    LW_ECHO("\n");

    LW_ECHO(" -----------------------------------------------------------------------------\n");
    LW_ECHO("               BLANK TIMING LIMITS : DPCD addr 00388h ~ 0038Eh               \n");
    LW_ECHO(" -----------------------------------------------------------------------------\n");

    str = "Vertical Blanking Min";
    printAlignedString(str,38);
    LW_ECHO(": %d lines\n",s_timingreg->lwsr_vbmn);
    str = "Vertical Blanking Max";
    printAlignedString(str,38);
    LW_ECHO(": %d lines\n",s_timingreg->lwsr_vbmx);
    str = "Horizontal Blanking Min";
    printAlignedString(str,38);
    LW_ECHO(": %d pixels\n",s_timingreg->lwsr_hbmn);
    str = "Horizontal Blanking Max";
    printAlignedString(str,38);
    LW_ECHO(": %d pixels\n",s_timingreg->lwsr_hbmx);
    LW_ECHO("\n");

    LW_ECHO(" -----------------------------------------------------------------------------\n");
    LW_ECHO("                    LWSR CRASH SYNC : DPCD addr 00366h (0x%x)                 \n",s_timingreg->lwsr_selfrefresh_366h);
    LW_ECHO(" -----------------------------------------------------------------------------\n");

    str = "Crash Sync Request";
    printAlignedString(str,38);
    switch(s_timingreg->lwsr_selfrefresh_srcs)
    {
        case 0: LW_ECHO(": Normal [default timings](%d)\n",s_timingreg->lwsr_selfrefresh_srcs); break;
        case 1: LW_ECHO(": Type 1 [Reset when outside active/fetching region](%d)\n",s_timingreg->lwsr_selfrefresh_srcs); break;
        case 2: LW_ECHO(": Undefined(%d)\n",s_timingreg->lwsr_selfrefresh_srcs); break;
        case 3: LW_ECHO(": Type 3 [Reset Immediately](%d)\n",s_timingreg->lwsr_selfrefresh_srcs); break; 
        case 5: LW_ECHO(": Combined => Reset outside active/fetching & initiate an SR Update(%d)",s_timingreg->lwsr_selfrefresh_srcs);
        case 7: LW_ECHO(": Combined => Reset immediately & initiate an SR Update(%d)",s_timingreg->lwsr_selfrefresh_srcs);
    }

    str = "Field Polarity Request";
    printAlignedString(str,38);
    switch(s_timingreg->lwsr_selfrefresh_srfp >> 4)
    {
        case 0: LW_ECHO(": Normal ilwersion pattern sequencing(%d)\n",s_timingreg->lwsr_selfrefresh_srfp >> 4); break;
        case 1: LW_ECHO(": Repeat same pattern as last frame(%d)\n",s_timingreg->lwsr_selfrefresh_srfp >> 4); break;
        default:LW_ECHO(": Reserved for future use(%d)\n",s_timingreg->lwsr_selfrefresh_srfp >> 4); break;
    }
    LW_ECHO("\n\n -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-\n\n");
}


// ---------------------------------------------------------------------------------------------------------------------------------


//  FUNCTION TO PRINT ANALYSIS LOG OF LWSR LINK REGISTERS

void echoLWSRLinkRegs(lwsr_linkreg_fields *s_linkreg)
{
    const char *str = NULL;

    LW_ECHO(" -----------------------------------------------------------------------------\n");
    LW_ECHO("                    GPU-SRC LINK : DPCD addr 00380h (0x%x)                    \n",s_linkreg->lwsr_link_gpu_src);
    LW_ECHO(" -----------------------------------------------------------------------------\n");
    str = "GPU-SRC Link Lane Width";
    printAlignedString(str,38);
    switch(s_linkreg->lwsr_lgsl)
    {
        case 1: LW_ECHO(": 1 Lane eDP / Single Channel LVDS(%d)\n",s_linkreg->lwsr_lgsl); break;
        case 2: LW_ECHO(": 2 Lanes eDP / Dual Channel LVDS(%d)\n",s_linkreg->lwsr_lgsl); break;
        case 3: LW_ECHO(": 3 Lanes eDP [not applicable to LVDS](%d)\n",s_linkreg->lwsr_lgsl); break;
        case 4: LW_ECHO(": 4 Lanes eDP [not applicable to LVDS](%d)\n",s_linkreg->lwsr_lgsl); break;
        case 5: LW_ECHO(": 8 Lanes eDP [not applicable to LVDS](%d)\n",s_linkreg->lwsr_lgsl); break;
        default:LW_ECHO(": Undefined\n"); break;                        
    }

    str = "GPU-SRC Ganging";
    printAlignedString(str,38);
    if (s_linkreg->lwsr_lgsg >> 4 ==0x01)
        LW_ECHO(": Supported(%d) [multiple eDP/LVDS interfaces]\n",s_linkreg->lwsr_lgsg >> 4); 
    else
        LW_ECHO(": NOT Supported(%d)\n",s_linkreg->lwsr_lgsg >> 4); 

    str = "GPU-SRC Pixel Format";
    printAlignedString(str,38);
    switch(s_linkreg->lwsr_lgspf >> 5)
    {
        case 0: LW_ECHO(": 18bit R6:G6:B6 (%d)\n",s_linkreg->lwsr_lgspf >> 5); break;
        case 1: LW_ECHO(": 24bit R8:G8:B8 (%d)\n",s_linkreg->lwsr_lgspf >> 5); break;
        case 2: LW_ECHO(": 30bits R10:G10:B10 (%d)\n",s_linkreg->lwsr_lgspf >> 5); break;
        default:LW_ECHO(": Reserved\n"); break;                        
    }
    LW_ECHO("\n");

    LW_ECHO(" -----------------------------------------------------------------------------\n");
    LW_ECHO("              LINK INTERFACE SRC-PANEL 1 : DPCD addr 00381h (0x%x)             \n",s_linkreg->lwsr_link_src_panel1);
    LW_ECHO(" -----------------------------------------------------------------------------\n");

    str = "SRC-Panel Pixel Format";
    printAlignedString(str,38);
    switch(s_linkreg->lwsr_lspf)
    {
        case 0: LW_ECHO(": 18bit R6:G6:B6 (%d)\n",s_linkreg->lwsr_lspf); break;
        case 1: LW_ECHO(": 24bit R8:G8:B8 (%d)\n",s_linkreg->lwsr_lspf); break;
        case 2: LW_ECHO(": 30bits R10:G10:B10 (%d)\n",s_linkreg->lwsr_lspf); break;
        default:LW_ECHO(": Reserved\n"); break;                        
    }
    LW_ECHO("\n");

    LW_ECHO(" -----------------------------------------------------------------------------\n");
    LW_ECHO("              LINK INTERFACE SRC-PANEL 2 : DPCD addr 00382h (0x%x)            \n",s_linkreg->lwsr_link_src_panel2);
    LW_ECHO(" -----------------------------------------------------------------------------\n");
    str = "SRC-Panel Vertical Columns";
    printAlignedString(str,38);
    LW_ECHO(": %d columns\n",s_linkreg->lwsr_lsvc + 1);

    str = "SRC-Panel Horizontal Rows";
    printAlignedString(str,38);
    LW_ECHO(": %d rows\n",(s_linkreg->lwsr_lshr>>3) + 1);
    LW_ECHO("\n");

    LW_ECHO(" -----------------------------------------------------------------------------\n");
    LW_ECHO("                 LINK INTERFACE TYPE : DPCD addr 00383h (0x%x)                \n",s_linkreg->lwsr_link_type);
    LW_ECHO(" -----------------------------------------------------------------------------\n");

    str = "GPU-SRC Interface Type";
    printAlignedString(str,38);
    switch(s_linkreg->lwsr_ltyp_gpu_src)
    {
        case 0: LW_ECHO(": LVDS (%d)\n",s_linkreg->lwsr_ltyp_gpu_src); break;
        case 1: LW_ECHO(": DSI (%d)\n",s_linkreg->lwsr_ltyp_gpu_src); break;
        case 2: LW_ECHO(": eDP (%d)\n",s_linkreg->lwsr_ltyp_gpu_src); break;
        case 4: LW_ECHO(": DP [External](%d)\n",s_linkreg->lwsr_ltyp_gpu_src); break;
        case 5: LW_ECHO(": DVI/HDMI 1.x [External TMDS](%d)\n",s_linkreg->lwsr_ltyp_gpu_src); break;
        case 6: LW_ECHO(": HDMI 2.0 [External TMDS2](%d)\n",s_linkreg->lwsr_ltyp_gpu_src); break;
        default:LW_ECHO(": Reserved\n"); break;                        
    }

    str = "SRC-Panel Interface Type";
    printAlignedString(str,38);
    switch(s_linkreg->lwsr_ltyp_src_panel >> 4)
    {
        case 0: LW_ECHO(": LVDS (%d)\n",s_linkreg->lwsr_ltyp_src_panel >> 4); break;
        case 1: LW_ECHO(": DSI (%d)\n",s_linkreg->lwsr_ltyp_src_panel >> 4); break;
        case 2: LW_ECHO(": eDP (%d)\n",s_linkreg->lwsr_ltyp_src_panel >> 4); break;
        case 4: LW_ECHO(": DP [External](%d)\n",s_linkreg->lwsr_ltyp_src_panel >> 4); break;
        case 5: LW_ECHO(": DVI/HDMI 1.x [External TMDS](%d)\n",s_linkreg->lwsr_ltyp_src_panel >> 4); break;
        case 6: LW_ECHO(": HDMI 2.0 [External TMDS2](%d)\n",s_linkreg->lwsr_ltyp_src_panel >> 4); break;
        default:LW_ECHO(": Reserved\n"); break;                        
    }
    LW_ECHO("\n");

    LW_ECHO(" -----------------------------------------------------------------------------\n");
    LW_ECHO("                     LINK CONTROL : DPCD addr 00384h (0x%x)                   \n",s_linkreg->lwsr_link_control);
    LW_ECHO(" -----------------------------------------------------------------------------\n");

    str = "GPU-SRC Link Lane Set";
    printAlignedString(str,38);
    switch(s_linkreg->lwsr_lgss)
    {
        case 1: LW_ECHO(": 1 Lane eDP / Single Channel LVDS [Even only] (%d)\n",s_linkreg->lwsr_lgss); break;
        case 2: LW_ECHO(": 2 Lanes eDP / Dual Channel LVDS (%d)\n",s_linkreg->lwsr_lgss); break;
        case 3: LW_ECHO(": 3 Lanes eDP [not applicable to LVDS] (%d)\n",s_linkreg->lwsr_lgss); break;
        case 4: LW_ECHO(": 4 Lanes eDP [not applicable to LVDS] (%d)\n",s_linkreg->lwsr_lgss); break;
        case 5: LW_ECHO(": 8 Lanes eDP [not applicable to LVDS] (%d)\n",s_linkreg->lwsr_lgss); break;
        default:LW_ECHO(": Undefined\n"); break;    
    }

    str = "SRC-Panel Link Lane Set";
    printAlignedString(str,38);
    switch(s_linkreg->lwsr_lsps >> 4)
    {
        case 1: LW_ECHO(": 1 Lane eDP / Single Channel LVDS [Even only] (%d)\n",s_linkreg->lwsr_lsps >> 4); break;
        case 2: LW_ECHO(": 2 Lanes eDP / Dual Channel LVDS (%d)\n",s_linkreg->lwsr_lsps >> 4); break;
        case 3: LW_ECHO(": 3 Lanes eDP [not applicable to LVDS] (%d)\n",s_linkreg->lwsr_lsps >> 4); break;
        case 4: LW_ECHO(": 4 Lanes eDP [not applicable to LVDS] (%d)\n",s_linkreg->lwsr_lsps >> 4); break;
        case 5: LW_ECHO(": 8 Lanes eDP [not applicable to LVDS] (%d)\n",s_linkreg->lwsr_lsps >> 4); break;
        default:LW_ECHO(": Undefined\n"); break;                            
    }
    LW_ECHO("\n");
}


// ---------------------------------------------------------------------------------------------------------------------------------


//  FUNCTION TO PRINT ANALYSIS LOG OF LWSR BACKLIGHT REGISTERS

void echoLWSRBacklightRegs(lwsr_backlightreg_fields *s_blreg)
{
    const char *str = NULL;

    LW_ECHO(" -----------------------------------------------------------------------------\n");
    LW_ECHO("               BACKLIGHT CAPABILITY #1 : DPCD addr 00701h (0x%x)              \n",s_blreg->lwsr_blcap1);
    LW_ECHO(" -----------------------------------------------------------------------------\n");
    str = "BACKLIGHT_ADJUSTMENT_CAPABLE";
    printAlignedString(str,42);
    if (s_blreg->lwsr_bl_adjustment_capable == 0x01)
        LW_ECHO(": Capable(%d)\n",s_blreg->lwsr_bl_adjustment_capable);
    else
        LW_ECHO(": NOT Capable(%d)\n",s_blreg->lwsr_bl_adjustment_capable);

    str = "BACKLIGHT_PIN_ENABLE_CAPABLE";
    printAlignedString(str,42);
    if (s_blreg->lwsr_bl_pin_en_capable == 0x02)
        LW_ECHO(": Capable(%d)\n",s_blreg->lwsr_bl_pin_en_capable >> 1);
    else
        LW_ECHO(": NOT Capable(%d)\n",s_blreg->lwsr_bl_pin_en_capable >> 1);

    str = "BACKLIGHT_AUX_ENABLE_CAPABLE";
    printAlignedString(str,42);
    if (s_blreg->lwsr_bl_aux_en_capable == 0x04)
        LW_ECHO(": Capable(%d)\n",s_blreg->lwsr_bl_aux_en_capable >> 2);
    else
        LW_ECHO(": NOT Capable(%d)\n",s_blreg->lwsr_bl_aux_en_capable >> 2);

    str = "PANEL_SELF_TEST_PIN_ENABLE_CAPABLE";
    printAlignedString(str,42);
    if (s_blreg->lwsr_pslftst_pin_en_capable == 0x08)
        LW_ECHO(": Capable(%d)\n",s_blreg->lwsr_pslftst_pin_en_capable >> 3);
    else
        LW_ECHO(": NOT Capable(%d) \n",s_blreg->lwsr_pslftst_pin_en_capable >> 3);

    str = "PANEL_SELF_TEST_AUX_ENABLE_CAPABLE";
    printAlignedString(str,42);
    if (s_blreg->lwsr_pslftst_aux_en_capable == 0x10)
        LW_ECHO(": Capable(%d)\n ",s_blreg->lwsr_pslftst_aux_en_capable >> 4);
    else
        LW_ECHO(": NOT Capable(%d)\n",s_blreg->lwsr_pslftst_aux_en_capable >> 4);
    LW_ECHO("\n");

     str = "FRC_ENABLE_CAPABLE";
    printAlignedString(str,42);
    if (s_blreg->lwsr_frc_en_capable == 0x20)
        LW_ECHO(": Capable (%d)\n",s_blreg->lwsr_frc_en_capable >> 5);
    else
        LW_ECHO(": NOT capable (%d)\n",s_blreg->lwsr_frc_en_capable >> 5);

    str = "COLOR_ENGINE_CAPABLE";
    printAlignedString(str,42);
    if (s_blreg->lwsr_color_eng_capable == 0x40)
        LW_ECHO(": Capable (%d)\n",s_blreg->lwsr_color_eng_capable >> 6);
    else
        LW_ECHO(": NOT capable (%d)\n",s_blreg->lwsr_color_eng_capable >> 6);

    str = "SET_POWER_CAPABLE";
    printAlignedString(str,42);
    if (s_blreg->lwsr_set_pwr_capable == 0x80)
        LW_ECHO(": Supports all backlight settings update at VSync(%d)\n",s_blreg->lwsr_set_pwr_capable >> 7);
    else
        LW_ECHO(": Use new settings immediately(%d)",s_blreg->lwsr_set_pwr_capable >> 7);
    LW_ECHO("\n");

    LW_ECHO(" -----------------------------------------------------------------------------\n");
    LW_ECHO("               BACKLIGHT ADJUST CAPABILITY : DPCD addr 00702h (0x%x)              \n",s_blreg->lwsr_blcap_adj);
    LW_ECHO(" -----------------------------------------------------------------------------\n");
    str = "BACKLIGHT_BRIGHTNESS_PWM_PIN_CAPABLE";
    printAlignedString(str,42);
    if (s_blreg->lwsr_bl_bright_pwm_pin_capable == 0x00)
        LW_ECHO(": Supported/provided(%d)\n",s_blreg->lwsr_bl_bright_pwm_pin_capable);
    else
        LW_ECHO(": NOT supported/provided(%d)\n",s_blreg->lwsr_bl_bright_pwm_pin_capable);

    str = "BACKLIGHT_BRIGHTNESS_AUX_SET_CAPABLE";
    printAlignedString(str,42);
    if (s_blreg->lwsr_bl_bright_aux_set_capable == 0x02)
        LW_ECHO(": Capable (%d)\n",s_blreg->lwsr_bl_bright_aux_set_capable >> 1);
    else
        LW_ECHO(": NOT capable (%d)\n",s_blreg->lwsr_bl_bright_aux_set_capable >> 1);

    str = "BACKLIGHT_ BRIGHTNESS_AUX_BYTE_COUNT";
    printAlignedString(str,42);
    if (s_blreg->lwsr_bl_bright_aux_byte_count == 0x04)
        LW_ECHO(": Supported [16 bit value] (%d)\n",s_blreg->lwsr_bl_bright_aux_byte_count >> 2);
    else
        LW_ECHO(": Supported [8 bit value] (%d)\n",s_blreg->lwsr_bl_bright_aux_byte_count >> 2);

    str = "BACKLIGHT_AUX-PWM_PRODUCT_CAPABLE";
    printAlignedString(str,42);
    if (s_blreg->lwsr_bl_aux_pwm_prod_capable == 0x08)
        LW_ECHO(": Capable (%d)\n",s_blreg->lwsr_bl_aux_pwm_prod_capable >> 3);
    else
        LW_ECHO(": NOT capable (%d)\n",s_blreg->lwsr_bl_aux_pwm_prod_capable >> 3);

    str = "BACKLIGHT_FREQ_PWM_PIN_PASS-THRU_CAPABLE";
    printAlignedString(str,42);
    if (s_blreg->lwsr_bl_pwm_freq_pin_pt_capable == 0x10)
        LW_ECHO(": Capable (%d)\n",s_blreg->lwsr_bl_pwm_freq_pin_pt_capable >> 4);
    else
        LW_ECHO(": NOT capable (%d)\n",s_blreg->lwsr_bl_pwm_freq_pin_pt_capable >> 4);

    str = "BACKLIGHT_FREQ_AUX_SET_CAPABLE";
    printAlignedString(str,42);
    if (s_blreg->lwsr_bl_aux_freq_set_capable == 0x20)
        LW_ECHO(": Capable (%d)\n",s_blreg->lwsr_bl_aux_freq_set_capable >> 5);
    else
        LW_ECHO(": NOT capable (%d)\n",s_blreg->lwsr_bl_aux_freq_set_capable >> 5);

    str = "DYNAMIC_BACKLIGHT_CONTROL_CAPABLE";
    printAlignedString(str,42);
    if (s_blreg->lwsr_bl_dynamic_capable == 0x40)
        LW_ECHO(": Capable (%d)\n",s_blreg->lwsr_bl_dynamic_capable >> 6);
    else
        LW_ECHO(": NOT capable (%d)\n",s_blreg->lwsr_bl_dynamic_capable >> 6);

    str = "VSYNC_BACKLIGHT_UPDATE_CAPABLE";
    printAlignedString(str,42);
    if (s_blreg->lwsr_bl_vsync_update_capable == 0x80)
        LW_ECHO(": Supports all backlight settings update at VSync(%d)\n",s_blreg->lwsr_bl_vsync_update_capable >> 7);
    else
        LW_ECHO(": Use new settings immediately(%d)",s_blreg->lwsr_bl_vsync_update_capable >> 7);
    LW_ECHO("\n");

    LW_ECHO(" -----------------------------------------------------------------------------\n");
    LW_ECHO("               BACKLIGHT CAPABILITY #2 : DPCD addr 00703h (0x%x)              \n",s_blreg->lwsr_blcap2);
    LW_ECHO(" -----------------------------------------------------------------------------\n");

    str = "OVERDRIVE_ENGINE_ENABLED";
    printAlignedString(str,42);
    if (s_blreg->lwsr_lcd_ovrdrv == 0x01)
        LW_ECHO(": Sink has LCD overdrive functionality\n");
    else
        LW_ECHO(": No LCD overdrive functionality \n");

    str = "BACKLIGHT_SINGLE_REGION_DRIVE";
    printAlignedString(str,42);
    if (s_blreg->lwsr_bl_1reg_drv == 0x02)
        LW_ECHO(": Supports single bl o/p switched between regions\n");
    else
        LW_ECHO(": N/A\n");

    str = "BACKLIGHT_SINGLE_STRING_DRIVE";
    printAlignedString(str,42);
    if (s_blreg->lwsr_bl_1str_drv == 0x04)
        LW_ECHO(": Supports single bl string o/p\n");
    else
        LW_ECHO(": N/A\n");

    LW_ECHO("\n");

    LW_ECHO(" -----------------------------------------------------------------------------\n");
    LW_ECHO("               BACKLIGHT CAPABILITY #3 : DPCD addr 00704h (0x%x)              \n",s_blreg->lwsr_blcap3);
    LW_ECHO(" -----------------------------------------------------------------------------\n");

    str = "X_REGION_CAP";
    printAlignedString(str,42);
    LW_ECHO(": %d independant 1D regions (horizontal direction)",s_blreg->lwsr_x_region_cap + 1);
    if (s_blreg->lwsr_x_region_cap == 0)
        LW_ECHO("[No regional backlight support]\n");
    else
        LW_ECHO("\n");

    str = "Y_REGION_CAP";
    printAlignedString(str,42);
    LW_ECHO(": %d independant 1D regions (vertical direction)",(s_blreg->lwsr_y_region_cap >> 4) + 1);
    if (s_blreg->lwsr_y_region_cap == 0)
        LW_ECHO("[No regional backlight support]\n");
    else
        LW_ECHO("\n");

    LW_ECHO("\n");

    LW_ECHO(" -----------------------------------------------------------------------------\n");
    LW_ECHO("            DISPLAY PANEL FEATURE CONTROL : DPCD addr 00720h (0x%x)           \n",s_blreg->lwsr_disp_cntrl);
    LW_ECHO(" -----------------------------------------------------------------------------\n");
    str = "SRC controlled display backlight function";
    printAlignedString(str,42);
    if (s_blreg->lwsr_bl_enable == 0x01)
        LW_ECHO(": Enabled(%d)\n",s_blreg->lwsr_bl_enable);
    else
        LW_ECHO(": NOT enabled(%d)\n",s_blreg->lwsr_bl_enable);

    str = "BLACK_VIDEO";
    printAlignedString(str,42);
    if (s_blreg->lwsr_blackvideo_enable == 0x02)
        LW_ECHO(": Enabled(%d)\n",s_blreg->lwsr_blackvideo_enable >> 1);
    else
        LW_ECHO(": NOT enabled(%d)\n",s_blreg->lwsr_blackvideo_enable >> 1);

    str = "FRC";
    printAlignedString(str,42);
    if (s_blreg->lwsr_frc_enable == 0x04)
        LW_ECHO(": Enabled(%d)\n",s_blreg->lwsr_frc_enable >> 2);
    else
        LW_ECHO(": NOT enabled(%d)\n",s_blreg->lwsr_frc_enable >> 2);

    str = "COLOR_ENGINE";
    printAlignedString(str,42);
    if (s_blreg->lwsr_clreng_enable == 0x08)
        LW_ECHO(": Enabled(%d)\n",s_blreg->lwsr_clreng_enable >> 3);
    else
        LW_ECHO(": NOT enabled(%d)\n",s_blreg->lwsr_frc_enable >> 3);

    str = "VSYNC_BACKLIGHT_UPDATE";
    printAlignedString(str,42);
    if (s_blreg->lwsr_vsync_bl_updt_en == 0x80)
        LW_ECHO(": Enabled(%d)\n",s_blreg->lwsr_vsync_bl_updt_en >> 7);
    else
        LW_ECHO(": NOT enabled(%d)\n",s_blreg->lwsr_vsync_bl_updt_en >> 7);
    LW_ECHO("\n");

    LW_ECHO(" -----------------------------------------------------------------------------\n");
    LW_ECHO("                  BACKLIGHT MODE SET : DPCD addr 00721h (0x%x)                \n",s_blreg->lwsr_bl_mode_set);
    LW_ECHO(" -----------------------------------------------------------------------------\n");

    str = "BACKLIGHT_BRIGHTNESS_CONTROL_MODE";
    printAlignedString(str,42);
    switch(s_blreg->lwsr_bl_bright_cntrl_mode)
    {
        case 0: LW_ECHO(": Backlight controled by PWM pin on eDP or SMBus(%d)\n",s_blreg->lwsr_bl_bright_cntrl_mode); break;  
        case 1: LW_ECHO(": Backlight brightness pre-set panel default level(%d)\n",s_blreg->lwsr_bl_bright_cntrl_mode); break;
        case 2: LW_ECHO(": Backlight controlled by Backlight Setting registers(%d)\n",s_blreg->lwsr_bl_bright_cntrl_mode); break;
        case 3: LW_ECHO(": Backlight controlled by product of SMBus/PWM and Backlight Setting Registers(%d)\n",s_blreg->lwsr_bl_bright_cntrl_mode); break;
    }

    str = "BACKLIGHT_FREQ_PWM_PIN_PASS-THRU";
    printAlignedString(str,42);
    if (s_blreg->lwsr_bl_pwm_freq_pin_pt_capable == 0x10)
        LW_ECHO(": Enabled (%d)\n",s_blreg->lwsr_bl_pwm_freq_pin_pt_enable >> 2);
    else
        LW_ECHO(": NOT enabled (%d)\n",s_blreg->lwsr_bl_pwm_freq_pin_pt_enable >> 2);

    str = "BACKLIGHT_FREQ_AUX_SET";
    printAlignedString(str,42);
    if (s_blreg->lwsr_bl_aux_freq_set_enable == 0x08)
        LW_ECHO(": Enabled (%d)\n",s_blreg->lwsr_bl_aux_freq_set_enable >> 3);
    else
        LW_ECHO(": NOT enabled (%d)\n",s_blreg->lwsr_bl_aux_freq_set_enable >> 3);

    str = "DYNAMIC_BACKLIGHT";
    printAlignedString(str,42);
    if (s_blreg->lwsr_bl_dynamic_enable == 0x10)
        LW_ECHO(": Enabled (%d)\n",s_blreg->lwsr_bl_dynamic_enable >> 4);
    else
        LW_ECHO(": NOT enabled (%d)\n",s_blreg->lwsr_bl_dynamic_enable >> 4);

    str = "REGIONAL_BACKLIGHT";
    printAlignedString(str,42);
    LW_ECHO(": (%d)",s_blreg->lwsr_bl_rg_bl_enable >> 5);
    if (s_blreg->lwsr_bl_rg_bl_enable == 0x20)
    LW_ECHO(": Enabled (%d)\n",s_blreg->lwsr_bl_rg_bl_enable >> 5);
    else
        LW_ECHO(": NOT enabled (%d)\n",s_blreg->lwsr_bl_rg_bl_enable >> 5);

    str = "UPDATE_REGION_BRIGHTNESS";
    printAlignedString(str,42);
    if (s_blreg->lwsr_bl_updt_britnes == 0x40)
        LW_ECHO(": Bl brightness control regs changes allowed (%d)\n",s_blreg->lwsr_bl_updt_britnes >> 6);
    else
        LW_ECHO(": Regional bl update in progress (%d)\n",s_blreg->lwsr_bl_updt_britnes >> 6);
    LW_ECHO("\n");


    LW_ECHO(" -----------------------------------------------------------------------------\n");
    LW_ECHO("             BACKLIGHT BRIGHTNESS : DPCD addr 00722h ~ 00723h (0x%x)          \n",s_blreg->lwsr_bl_brightness);
    LW_ECHO(" -----------------------------------------------------------------------------\n");
    str = "BACKLIGHT_BRIGHTNESS";
    printAlignedString(str,42);
    LW_ECHO(": %d\n",s_blreg->lwsr_bl_brightness);

    str = "BACKLIGHT_BRIGHTNESS_MSB";
    printAlignedString(str,42);
    LW_ECHO(": %d\n",s_blreg->lwsr_bl_brightness_msb);

    str = "BACKLIGHT_BRIGHTNESS_LSB";
    printAlignedString(str,42);
    LW_ECHO(": %d\n",s_blreg->lwsr_bl_brightness_lsb);
    LW_ECHO("\n");


    LW_ECHO(" -----------------------------------------------------------------------------\n");
    LW_ECHO("  PWM GENERATION AND BL CONTROLLER STATUS: DPCD addr 00724h ~ 00727h (0x%x)  \n",s_blreg->lwsr_bl_pwm);
    LW_ECHO(" -----------------------------------------------------------------------------\n");
    if (s_blreg->lwsr_pwmgen_bit_count < s_blreg->lwsr_pwmgen_bit_count_min)
        s_blreg->lwsr_pwmgen_bit_count = s_blreg->lwsr_pwmgen_bit_count_min;
    if (s_blreg->lwsr_pwmgen_bit_count > s_blreg->lwsr_pwmgen_bit_count_max)
        s_blreg->lwsr_pwmgen_bit_count = s_blreg->lwsr_pwmgen_bit_count_max;

    str = "Effective PWMGEN_BIT_COUNT";
    printAlignedString(str,42);
    LW_ECHO(": %d\n",s_blreg->lwsr_pwmgen_bit_count);

    str = "PWMGEN_BIT_COUNT_MIN";
    printAlignedString(str,42);
    LW_ECHO(": %d\n",s_blreg->lwsr_pwmgen_bit_count_min);

    str = "PWMGEN_BIT_COUNT_MAX";
    printAlignedString(str,42);
    LW_ECHO(": %d\n",s_blreg->lwsr_pwmgen_bit_count_max);

    str = "BACKLIGHT_CONTROL_STATUS";
    printAlignedString(str,42);
    if (s_blreg->lwsr_bl_cntrl_status == 0)
        LW_ECHO(": Normal operation mode(%d)",s_blreg->lwsr_bl_cntrl_status);
    else
        LW_ECHO(": Backlight control mode(%d)",s_blreg->lwsr_bl_cntrl_status);
    LW_ECHO("\n");


    LW_ECHO(" -----------------------------------------------------------------------------\n");
    LW_ECHO("                  BACKLIGHT FREQUENCY: DPCD addr 00728h (0x%x)                \n",s_blreg->lwsr_bl_freq_set);
    LW_ECHO(" -----------------------------------------------------------------------------\n");
    str = "BACKLIGHT_FREQ_SET control value(F)";
    printAlignedString(str,42);
    LW_ECHO(": %d\n",s_blreg->lwsr_bl_freq_set);
    LW_ECHO("\n\n");


    LW_ECHO(" -----------------------------------------------------------------------------\n");
    LW_ECHO("             BACKLIGHT BRIGHTNESS RANGE: DPCD addr 00732h ~ 00733h             \n");
    LW_ECHO(" -----------------------------------------------------------------------------\n");
    str = "DBC_MINIMUM_BRIGHTNESS_CAP";
    printAlignedString(str,42);
    LW_ECHO(": %d percent of normal\n",s_blreg->lwsr_bl_brighness_min*5);

    str = "DBC_MAXIMUM_BRIGHTNESS_CAP";
    printAlignedString(str,42);
    LW_ECHO(": %d percent of normal\n",s_blreg->lwsr_bl_brighness_max*5);
    LW_ECHO("\n");

}


// ---------------------------------------------------------------------------------------------------------------------------------


//  FUNCTION TO PRINT ANALYSIS LOG OF LWSR DIAGNOSTIC REGISTERS

void echoLWSRDiagnosticRegs(lwsr_diagnosticreg_fields *s_diagreg)
{
    const char *str;

    LW_ECHO(" -----------------------------------------------------------------------------\n");  
    LW_ECHO("                DIAGNOSTIC REGISTERS : DPCD addr 00390h ~ 0039Bh              \n");
    LW_ECHO(" -----------------------------------------------------------------------------\n");

    str = "Last Self-Refresh Frame Count";
    printAlignedString(str,38);
    LW_ECHO(": %d frames [Reset on every SR Entry Request]\n",s_diagreg->lwsr_dsfc);

    if (s_diagreg->lwsr_dsrd == 0x01)
        LW_ECHO("Register rollover detected at 00390h... (%d)\n",s_diagreg->lwsr_dsrd);
    /*else
        LW_ECHO("Register rollover NOT detected at 00390h... (%d)\n",s_diagreg->lwsr_dsrd);*/

    str = "Current Scanline";
    printAlignedString(str,38);
    LW_ECHO(": Line %d  [from line 0 at VSync, counting through blanking & active region]\n",s_diagreg->lwsr_xtcsl);

    str = "Resync Frame Counter";
    printAlignedString(str,38); 
    LW_ECHO(": %d frames [in resync interval from trigger to commit]\n",s_diagreg->lwsr_drfc);

    str = "SRC Panel Side Timing Status";
    printAlignedString(str,38);
    switch(s_diagreg->lwsr_srts)
    {
        case 0: LW_ECHO(": Idle [pass through] mode, using GPU-SRC timings(%d)\n",s_diagreg->lwsr_srts); break;
        case 1: LW_ECHO(": Active Region(%d)\n",s_diagreg->lwsr_srts); break;
        case 2: LW_ECHO(": Horizontal Blank(%d)\n",s_diagreg->lwsr_srts); break;
        case 4: LW_ECHO(": Vertical Blank(%d)\n",s_diagreg->lwsr_srts); break;
        default: LW_ECHO(": Undefined\n"); break;
    }

    str = "Total Self-Refresh Frame Count";  
    printAlignedString(str,38);
    LW_ECHO(": %d frames [Cumulative Frame count since first entry to SR mode after power-on]\n",s_diagreg->lwsr_frtf);

    if (s_diagreg->lwsr_srrd_diag == 0x01)
        LW_ECHO("Register rollover detected at 00398h... (%d)\n",s_diagreg->lwsr_srrd_diag);
    /*else
        LW_ECHO("Register rollover NOT detected at 00398h... (%d)\n",s_diagreg->lwsr_srrd_diag);*/

    LW_ECHO("\n");
}

// ---------------------------------------------------------------------------------------------------------------------------------
