/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2003 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

//*****************************************************
//
// lwwatch WinDbg Extension
// retodd@lwpu.com - 2.1.2002
// dac.c
//
//*****************************************************

//
// includes
//
#include "lw_ref.h"
#include "lw40/dev_dac.h"
#include "lw40/dev_vga.h"
#include "lw40/dev_bus.h"
#include "lw40/dev_tvo.h"
#include "os.h"
#include "dac.h"
#include "chip.h"
#include "hal.h"


//-----------------------------------------------------
// quickFPRegPerHeadDump
// Used for dumpFPRegs to dump a FP register per number of heads.
//-----------------------------------------------------
VOID quickFPRegPerHeadDump(char *initStr, U032 Regs[MAX_CRTCS][REGS_MAX_INDEX],
                           U032 iterations, U032 index, U032 ByteFlag)
{
    U032 j = 0;

    dprintf("%s", initStr);
    if(ByteFlag & QFPDUMP_PRINT_BYTE)
    {
        for(j=0; j<iterations; j++)
            dprintf("0x%02x    ", Regs[j][index]);
    }
    else if(ByteFlag & QFPDUMP_DECIMAL_TIMINGS)
    {
        for(j=0; j<iterations; j++)
            dprintf("  %8d    ", Regs[j][index]);
    }
    else
    {
        for(j=0; j<iterations; j++)
            dprintf("0x%08x    ", Regs[j][index]);
    }
    dprintf("\n");
}

//-----------------------------------------------------
// dumpSLIVideoBridgeRegisters
//
//-----------------------------------------------------
VOID dumpSLIVideoBridgeRegisters()
{
    U032 mChipRegs[5][2];
    U032 dvoregs[9];
    U032 rasterCount[2];
    U008 vgaregs[4][2];
    U008 lock;
    U032 i;

    if(!IsLW40orBetter())
      return;

    lock = UnlockExtendedCRTCs(0);

    for (i=0; i<2; i++)
    {
        rasterCount[i]  = GPU_REG_RD32(0x600808 + i*0x2000);
        mChipRegs[0][i] = GPU_REG_RD32(0x680900 + i*0x2000);
        mChipRegs[1][i] = GPU_REG_RD32(0x680904 + i*0x2000);
        mChipRegs[2][i] = GPU_REG_RD32(0x680908 + i*0x2000);
        mChipRegs[3][i] = GPU_REG_RD32(0x680920 + i*0x2000);
        mChipRegs[4][i] = GPU_REG_RD32(0x680924 + i*0x2000);

        vgaregs[0][i]   = REG_RDCR(0x33, i*0x2000);
        vgaregs[1][i]   = REG_RDCR(0x59, i*0x2000);
        vgaregs[2][i]   = REG_RDCR(0x9f, i*0x2000);
        vgaregs[3][i]   = REG_RDCR(0x1b, i*0x2000);
        // disable overflow
        REG_WRCR(0x1b, (vgaregs[3][i]&((U008)0x7f)), i*0x2000);
    }

    dvoregs[0] = GPU_REG_RD32(0x1280);
    dvoregs[1] = GPU_REG_RD32(0x1230);
    dvoregs[2] = GPU_REG_RD32(0x1234);
    dvoregs[3] = GPU_REG_RD32(0x1238);
    dvoregs[4] = GPU_REG_RD32(0x1240);
    dvoregs[5] = GPU_REG_RD32(0x1244);
    dvoregs[6] = GPU_REG_RD32(0x1248);

    dprintf("GPU BAR 0: " PhysAddr_FMT "\n", lwBar0);
    dprintf("                                 **Head A**    **Head B**\n");
    dprintf("MCHIP_GENERAL_CONTROL:(68x900) = 0x%08x    0x%08x\n", mChipRegs[0][0], mChipRegs[0][1]);
    dprintf("MCHIP_VDISPLAY_FIELD: (68x904) = 0x%08x    0x%08x\n", mChipRegs[1][0], mChipRegs[1][1]);
    dprintf("MCHIP_MASTER_SYNC:    (68x908) = 0x%08x    0x%08x\n", mChipRegs[2][0], mChipRegs[2][1]);
    dprintf("MCHIP_TESTPOINT_DATA: (68x920) = 0x%08x    0x%08x\n", mChipRegs[3][0], mChipRegs[3][1]);
    dprintf("MCHIP_RMGT:           (68x924) = 0x%08x    0x%08x\n", mChipRegs[4][0], mChipRegs[4][1]);
    dprintf("LW_CIO_CRE_LCD__INDEX:( CR33 ) =    0x%02x          0x%02x\n", vgaregs[0][0], vgaregs[0][1]);
    dprintf("LW_CIO_CRE_CTRL:      ( CR59 ) =    0x%02x          0x%02x\n", vgaregs[1][0], vgaregs[1][1]);
    dprintf("LW_CIO_CRE_DVOB:      ( CR9F ) =    0x%02x          0x%02x\n", vgaregs[2][0], vgaregs[2][1]);
    dprintf("LW_CIO_CRE_FF_INDEX:  ( CR1B ) = 0x%02x  0x%02x    0x%02x  0x%02x\n", 
            vgaregs[3][0], REG_RDCR(0x1b, 0), vgaregs[3][1], REG_RDCR(0x1b, 0x2000));
    dprintf("RasterCount is:       (60x808) = %s    %s\n", 
             ((rasterCount[0] == GPU_REG_RD32(0x600808))?"  stuck   ":"  moving  "),
             ((rasterCount[1] == GPU_REG_RD32(0x602808))?"  stuck   ":"  moving  ") );


    dprintf("\n");

    dprintf("LW_PBUS_DVOIO_PADCTL: (001280) = 0x%08x\n", dvoregs[0]);
    dprintf("LW_PBUS_DVOAIO_A:     (001230) = 0x%08x\n", dvoregs[1]);
    dprintf("LW_PBUS_DVOAIO_B:     (001234) = 0x%08x\n", dvoregs[2]);
    dprintf("LW_PBUS_DVOAIO_C:     (001238) = 0x%08x\n", dvoregs[3]);
    dprintf("LW_PBUS_DVOBIO_A:     (001240) = 0x%08x\n", dvoregs[1]);
    dprintf("LW_PBUS_DVOBIO_B:     (001244) = 0x%08x\n", dvoregs[2]);
    dprintf("LW_PBUS_DVOBIO_C:     (001248) = 0x%08x\n", dvoregs[3]);

    RestoreExtendedCRTCs(lock, 0);

    dprintf("\n");
    dprintf("\n");
     
}



//-----------------------------------------------------
// dumpHWSEQRam
//
//-----------------------------------------------------
VOID dumpHWSEQRam()
{
    U008 ramIndex;
    U032 seqPointers;
    U008 A_FPON_Index, B_FPOFF_Index, C_SUS_Index, D_RES_Index;

    // TJCMOD - We also need to limit to mobile chips
    if(!IsLW17orBetter())
    {
        dprintf("lw: There is no hardware sequencer on this chip.\n");
        return;
    }

    dprintf("\n");
    dprintf("lw: Hardware Sequencer related registers:\n");
    dprintf("\n");
    dprintf("LW_PBUS_SEQ_PTR            :1304   = 0x%08x\n", GPU_REG_RD32(LW_PBUS_SEQ_PTR));
    dprintf("LW_PBUS_SEQ_STATUS         :1308   = 0x%08x\n", GPU_REG_RD32(LW_PBUS_SEQ_STATUS));
    dprintf("LW_PBUS_SEQ_BYP            :1310   = 0x%08x\n", GPU_REG_RD32(LW_PBUS_SEQ_BYP));
    dprintf("LW_PBUS_SEQ_BYP2           :1314   = 0x%08x\n", GPU_REG_RD32(LW_PBUS_SEQ_BYP2));

    dprintf("\n\n");
    dprintf("lw: Hardware Sequencer Ram contents:\n\n");

    for(ramIndex = 0; ramIndex < LW_PBUS_SEQ_RAM__SIZE_1; ramIndex++)
    {
        dprintf(" 0x%04x: ", (0x00001400 + (ramIndex) * 4));
        dprintf(" 0x%08x\n", GPU_REG_RD32(0x00001400 + (ramIndex) * 4));
    }
    dprintf("\n");

    seqPointers = GPU_REG_RD32(LW_PBUS_SEQ_PTR);

    A_FPON_Index =  (U008)((seqPointers)       & 0x0000003F); 
    B_FPOFF_Index = (U008)((seqPointers >> 8)  & 0x0000003F);  
    C_SUS_Index =   (U008)((seqPointers >> 16) & 0x0000003F); 
    D_RES_Index =   (U008)((seqPointers >> 24) & 0x0000003F);  
    
    A_FPON_Index =  (A_FPON_Index)  ? A_FPON_Index  : LW_PBUS_SEQ_PTR_A_FPON_DEFAULT;
    B_FPOFF_Index = (B_FPOFF_Index) ? B_FPOFF_Index : LW_PBUS_SEQ_PTR_B_FPOFF_DEFAULT;
    C_SUS_Index =   (C_SUS_Index)   ? C_SUS_Index   : LW_PBUS_SEQ_PTR_C_SUS_DEFAULT; 
    D_RES_Index =   (D_RES_Index)   ? D_RES_Index   : LW_PBUS_SEQ_PTR_D_RES_DEFAULT; 

    dprintf("\n");
    dprintf("lw: Hardware Sequencer Programs:\n");
    dprintf("\n");
    dprintf("lw:  A_FPON  program starts at: 0x%04X\n", 0x00001400+(A_FPON_Index));
    dprintf("lw:  B_FPOFF program starts at: 0x%04X\n", 0x00001400+(B_FPOFF_Index));
    dprintf("lw:  C_SUS   program starts at: 0x%04X\n", 0x00001400+(C_SUS_Index));
    dprintf("lw:  D_RES   program starts at: 0x%04X\n", 0x00001400+(D_RES_Index));

    dprintf("\n");
}

//-----------------------------------------------------
// dumpVGARegs
//
//-----------------------------------------------------
VOID dumpVGARegs()
{
    U008 oldIndex, oldLock, cr3d;
    U008 data08;
    U032 i, j, k, MaxCrtcReg = GetMaxCrtcReg() + 1;
    U032 SavedIsLW11 = IsLW11();
    U008 Crtc[MAX_CRTCS][256];
    U008 scratch[MAX_CRTCS][16];
    U008 stats[MAX_CRTCS][4];
    U008 sequencer[MAX_CRTCS][10];
    U008 graphics[MAX_CRTCS][16];
    U008 attribute[MAX_CRTCS][32];
    U008 attributeIndex[MAX_CRTCS];
    volatile U008 scratch8;
    U032 Head, prevHead = 0;
    U032 broadcastWasEnabled = 0;
    U032 NumCrtcs = GetNumCrtcs();
    U032 crtcOffset = 0x0;

    if (IsLW50orBetter())
    {
        dprintf("%s is not supported on LW50+. Please use your favorite 32 bit reg dumping extension to read VGA regs (LW_PDISP_VGA*)\n", __FUNCTION__);
        return;
    }

    // Disable Broadcast mode (Doesn't touch CR registers)
    broadcastWasEnabled = SetBroadcastBit(0);

    for(Head = 0; Head < NumCrtcs; Head++)
    {
        dprintf("lw: Reading Head %c regs                  \r", ((char) 'A'+Head));
        prevHead = EnableHead(Head);

        // Fixup the crtcOffset
        crtcOffset = 0x0;
        if(Head == 1)
            crtcOffset = 0x2000;

        dprintf("lw: Reading Misc Regs:                    \r");
        ///////////////////////////
        // Misc REGISTERS
        ///////////////////////////
        // Feature Control
        stats[Head][0] = GPU_REG_RD08(LW_PRMCIO_INP0__READ_MONO + crtcOffset);

        // Input Status 1
        stats[Head][1] = GPU_REG_RD08(LW_PRMCIO_INP0__WRITE_COLOR + crtcOffset);

        // Input Status 0
        stats[Head][2] = GPU_REG_RD08(LW_PRMVIO_MISC__WRITE);

        // Feature Control
        stats[Head][3] = GPU_REG_RD08(LW_PRMVIO_MISC__READ);

        dprintf("lw: Reading Sequencer Regs:               \r");
        ///////////////////////////
        // Sequencer REGISTERS
        ///////////////////////////
        data08 = GPU_REG_RD08(LW_PRMVIO_SRX);  // Save Index
        for (i = 0; i <= 0x04; i++)
        {
            GPU_REG_WR08(LW_PRMVIO_SRX, (U008) i);
            sequencer[Head][i] = GPU_REG_RD08(LW_PRMVIO_SR_RESET);
        }
        GPU_REG_WR08(LW_PRMVIO_SRX, data08);  // Restore Index

        dprintf("lw: Reading Graphics Regs:                \r");
        ///////////////////////////
        // Graphics REGISTERS
        ///////////////////////////
        data08 = GPU_REG_RD08(LW_PRMVIO_GRX);  // Save Index
        for (i = 0; i <= 0x08; i++)
        {
            GPU_REG_WR08(LW_PRMVIO_GRX, (U008) i);
            graphics[Head][i] = GPU_REG_RD08(LW_PRMVIO_GX_SR);
        }
        GPU_REG_WR08(LW_PRMVIO_GRX, data08);  // Restore Index

        dprintf("lw: Reading CRTC Regs:                    \r");
        ///////////////////////////
        // CRTC REGISTERS
        ///////////////////////////        
        // unlock extended registers
        oldIndex = GPU_REG_RD08(0x6013d4 + crtcOffset);
        oldLock = UnlockExtendedCRTCs(crtcOffset);

        // Read the actual registers, not the shadows!
        cr3d = REG_RDCR(0x3D, crtcOffset);
        REG_WRCR(0x3D, (U008) (cr3d | 0x1), crtcOffset);

        // print theCRTC registers for this head
        // There are more registers on LW17, but I thought this would
        // get us there mostly
        for (i = 0; i < MaxCrtcReg; i++)
        {
            // For LW11, use the prevHead to write CR44 to prevent
            // read problems with CR44.
            if((SavedIsLW11) && (i == 0x44))
            {
                if(prevHead == 0x1)
                    data08 = (U008) 0x3;
                else
                    data08 = (U008) 0x0;
            }
            else
                data08 = REG_RDCR((U008) i, crtcOffset);
            Crtc[Head][i] = data08;
        }

        if(IsLW17orBetter())
        {
            /////////////////////////////
            // Scratch Registers
            /////////////////////////////
            // Before we lock everything back up, we can should read all
            // the scratch registers to dump those as well
            for(i=0; i<16; i++)
            {
                REG_WRCR(0x57, (U008) i, crtcOffset);
                scratch[Head][i] = REG_RDCR( 0x58, crtcOffset);
            }

            // Program Cr57 back to the original index read
            REG_WRCR(0x57, (U008) Crtc[Head][0x57], crtcOffset);
        }

        REG_WRCR(0x3D, cr3d, crtcOffset);
        RestoreExtendedCRTCs(oldLock, crtcOffset);
        GPU_REG_WR08(0x6013d4 + crtcOffset, oldIndex);

        dprintf("lw: Reading Attribute Regs:               \r");
        ///////////////////////////
        // Attribute REGISTERS
        ///////////////////////////        
        // Read Index first
        scratch8 = GPU_REG_RD08(LW_PRMCIO_INP0__COLOR + crtcOffset);   // Reset ATC FlipFlop
        attributeIndex[Head] = GPU_REG_RD08(LW_PRMCIO_ARX + crtcOffset);

        // Read attribute registers
        for (i = 0; i <= 0x14; i++)
        {
            scratch8 = GPU_REG_RD08(LW_PRMCIO_INP0__COLOR + crtcOffset);   // Reset ATC FlipFlop
            GPU_REG_WR08(LW_PRMCIO_ARX + crtcOffset, (U008) i);
            attribute[Head][i] = GPU_REG_RD08(LW_PRMCIO_AR_PALETTE__READ + crtcOffset);
        }

        // Restore Index register
        scratch8 = GPU_REG_RD08(LW_PRMCIO_INP0__COLOR + crtcOffset);   // Reset ATC FlipFlop
        GPU_REG_WR08(LW_PRMCIO_ARX + crtcOffset, attributeIndex[Head]);
        scratch8 = GPU_REG_RD08(LW_PRMCIO_INP0__COLOR + crtcOffset);   // Reset ATC FlipFlop

        dprintf("lw: Restoring Head %c                     \r", ((char) 'A'+prevHead));
        EnableHead(prevHead);
    }

    dprintf("                                              \n");

    if (NumCrtcs > 1)
    {
        if(broadcastWasEnabled)
            dprintf("lw: Original Head was Broadcast\n");
        else if(prevHead)
            dprintf("lw: Original Head was B\n");
        else
            dprintf("lw: Original Head was A\n");
    }
    else
    {
        dprintf("lw: Original Head was A\n");
    }
    dprintf("\n");

    ///////////////////////////
    // Print Results
    ///////////////////////////        
    dprintf("          *** Head A ***");

    for(Head = 1; Head < NumCrtcs; Head++)
        dprintf("                  *** Head %c ***", ('A'+Head));
    dprintf("\n");

    dprintf("Misc    : %02X", stats[0][3]);
    for(Head = 1; Head < NumCrtcs; Head++)
        dprintf("                              %02X", stats[Head][3]);
    dprintf("\n");

    // SR registers
    for (i = 0; i <= 0x04; i++)
    {
        k=i;
        dprintf("\nSR00    :");
        for(Head = 0; Head < NumCrtcs; Head++)
        {
            i=k;
            for (j = 0; (j < 8) && (i <= 0x4); i++, j++)
                dprintf(" %02X", sequencer[Head][i]);
            dprintf("        ");
            for(; j<8; j++)
                dprintf("   "); 
        }
    }
    dprintf("\n");

    // GR registers
    for (i = 0; i <= 0x08; i++)
    {
        k=i;
        dprintf("\nGR00    :");
        for(Head = 0; Head < NumCrtcs; Head++)
        {
            i=k;
            for (j = 0; (j < 9) && (i <= 0x8); i++, j++)
                dprintf(" %02X", graphics[Head][i]);
            dprintf("     ");
            for(; j<9; j++)
                dprintf("   ");
        }
    }
    dprintf("\n\n");

    // CR registers
    for (i = 0; i < MaxCrtcReg; )
    {
        k = i;
        dprintf("\nCR%02X    :", i);

        for(Head = 0; Head < NumCrtcs; Head++)
        {
            i = k;
            for (j = 0; (j < 8) && (i < MaxCrtcReg); i++, j++)
                dprintf(" %02X", Crtc[Head][i]);

            dprintf("        ");
            for(; j<8; j++)
                dprintf("   ");

        }
    }                              
    dprintf("\n\n");

    if(IsLW17orBetter())
    {
        // Scratch registers
        for (i = 0; i < 16; )
        {
            k = i;
            dprintf("\nSC%02X    :", i);

            for(Head = 0; Head < NumCrtcs; Head++)
            {
                i = k;
                for (j = 0; (j < 8) && (i < 16); i++, j++)
                    dprintf(" %02X", scratch[Head][i]);

                dprintf("        ");
                for(; j<8; j++)
                    dprintf("   ");

            }
        }                              
        dprintf("\n\n");
    }

    // AR Index
    dprintf("\nARX     : %02X", attributeIndex[0]);
    for(Head = 1; Head < NumCrtcs; Head++)
    {
        dprintf("                              %02X", attributeIndex[Head]);
    }
    dprintf("\n");

    // AR registers
    for (i = 0; i <= 0x14; )
    {
        k = i;
        dprintf("\nAR%02X    :", i);
        for(Head = 0; Head < NumCrtcs; Head++)
        {
            i = k;
            for (j = 0; (j < 8) && (i <= 0x14); i++, j++)
                dprintf(" %02X", attribute[Head][i]);
            dprintf("        ");
            for(; j<8; j++)
                dprintf("   ");
        }
    }                              
    dprintf("\n\n");

    // Restore Broadcast state
    SetBroadcastBit(broadcastWasEnabled);
}

//-----------------------------------------------------
// VGATimings
//
//-----------------------------------------------------
VOID VGATimings(U032 HeadBitmask, U032 Timing, U032 value, U032 format)
{
    U008 oldIndex, oldLock;
    U008 data08;
    U032 i;
    U032 Crtc[MAX_CRTCS][256];
    U008 sequencer[MAX_CRTCS][10];
    U032 Head, prevHead = 0;
    U032 broadcastWasEnabled = 0;
    U032 NumCrtcs = GetNumCrtcs();
    U032 crtcOffset = 0x0;
    U032 VIOOffset = 0;
    U032 T[MAX_CRTCS][0xc];
    U032 mask;
    char NamedT[0xD][32] =
      {
        { "Horizontal Total          : " },
        { "Horizontal Display End    : " },
        { "Horizontal Blank Start    : " },
        { "Horizontal Blank End      : " },
        { "Horizontal Retrace Start  : " },
        { "Horizontal Retrace End    : " },
        { "Vertical Total            : " },
        { "Vertical Display End      : " },
        { "Vertical Blank Start      : " },
        { "Vertical Blank End        : " },
        { "Vertical Retrace Start    : " },
        { "Vertical Retrace End      : " },
        { "CRTC Timings for            " },
      };

    if (IsLW50orBetter())
    {
        dprintf("%s is not supported on LW50+. Please use your favorite 32 bit "
                "reg dumping extension to read VGA regs (LW_PDISP_VGA*) or "
                "\"!dchnmstate core\" to read the method state which contains "
                "both FE and BE timings\n", __FUNCTION__);
        return;
    }

    // Disable Broadcast mode (Doesn't touch CR registers)
    broadcastWasEnabled = SetBroadcastBit(0);

    for(Head = 0; Head < NumCrtcs; Head++)
    {
        if(!(HeadBitmask & BIT(Head)) )
          continue;

        dprintf("lw: Reading Head %c regs                  \r", ((char) 'A'+Head));
        prevHead = EnableHead(Head);

        // Fixup the crtcOffset
        crtcOffset = 0x0;
        if(Head == 1)
        {
            crtcOffset = 0x2000;
            if(IsLW40orBetter())
              VIOOffset = 0x2000;
            else
              VIOOffset = 0x0;
        }

        dprintf("lw: Reading Sequencer Regs:               \r");

        // We need SR1 to know if we're 8 bpp or 9 bpp
        data08 = GPU_REG_RD08(LW_PRMVIO_SRX + VIOOffset);  // Save Index
        GPU_REG_WR08(LW_PRMVIO_SRX + VIOOffset, (U008) 1);
        sequencer[Head][1] = GPU_REG_RD08(LW_PRMVIO_SR_RESET + VIOOffset);
        GPU_REG_WR08(LW_PRMVIO_SRX + VIOOffset, data08);  // Restore Index

        dprintf("lw: Reading CRTC Regs:                    \r");
        // unlock extended registers
        oldIndex = GPU_REG_RD08(0x6013d4 + crtcOffset);
        oldLock = UnlockExtendedCRTCs(crtcOffset);

        // Read the actual registers, not the shadows!
        Crtc[Head][0x3D] = (U032) REG_RDCR(0x3D, crtcOffset);
        REG_WRCR(0x3D, (U008) (Crtc[Head][0x3D] | 0x1), crtcOffset);

        // Here's the list of registers needed
        Crtc[Head][0x00] = (U032) REG_RDCR((U008) 0x00, crtcOffset);
        Crtc[Head][0x2D] = (U032) REG_RDCR((U008) 0x2D, crtcOffset);
        Crtc[Head][0x55] = (U032) REG_RDCR((U008) 0x55, crtcOffset);
        Crtc[Head][0x01] = (U032) REG_RDCR((U008) 0x01, crtcOffset);
        Crtc[Head][0x02] = (U032) REG_RDCR((U008) 0x02, crtcOffset);
        Crtc[Head][0x03] = (U032) REG_RDCR((U008) 0x03, crtcOffset);
        Crtc[Head][0x05] = (U032) REG_RDCR((U008) 0x05, crtcOffset);
        Crtc[Head][0x1A] = (U032) REG_RDCR((U008) 0x1A, crtcOffset);
        Crtc[Head][0x25] = (U032) REG_RDCR((U008) 0x25, crtcOffset);
        Crtc[Head][0x04] = (U032) REG_RDCR((U008) 0x04, crtcOffset);
        Crtc[Head][0x56] = (U032) REG_RDCR((U008) 0x56, crtcOffset);
        Crtc[Head][0x06] = (U032) REG_RDCR((U008) 0x06, crtcOffset);
        Crtc[Head][0x07] = (U032) REG_RDCR((U008) 0x07, crtcOffset);
        Crtc[Head][0x41] = (U032) REG_RDCR((U008) 0x41, crtcOffset);
        Crtc[Head][0x12] = (U032) REG_RDCR((U008) 0x12, crtcOffset);
        Crtc[Head][0x15] = (U032) REG_RDCR((U008) 0x15, crtcOffset);
        Crtc[Head][0x09] = (U032) REG_RDCR((U008) 0x09, crtcOffset);
        Crtc[Head][0x16] = (U032) REG_RDCR((U008) 0x16, crtcOffset);
        Crtc[Head][0x10] = (U032) REG_RDCR((U008) 0x10, crtcOffset);
        Crtc[Head][0x11] = (U032) REG_RDCR((U008) 0x11, crtcOffset);


        REG_WRCR(0x3D, (U008) Crtc[Head][0x3D], crtcOffset);
        RestoreExtendedCRTCs(oldLock, crtcOffset);
        GPU_REG_WR08(0x6013d4 + crtcOffset, oldIndex);

        dprintf("lw: Restoring Head %c                     \r", ((char) 'A'+prevHead));
        EnableHead(prevHead);
    }

    // Callwlate results
    for(Head = 0; Head < NumCrtcs; Head++)
    {
        if(!(HeadBitmask & BIT(Head)) )
          continue;

        T[Head][VGAT_HT] =     (Crtc[Head][0x00])                 |
                           ((!!(Crtc[Head][0x2D] & BIT(0))) << 8) |
                           ((!!(Crtc[Head][0x55] & BIT(0))) << 9);
        T[Head][VGAT_HT] += 5;  // Actual value in register is #chars - 5


        T[Head][VGAT_HDE] =    (Crtc[Head][0x01])                 |
                           ((!!(Crtc[Head][0x2D] & BIT(1))) << 8) |
                           ((!!(Crtc[Head][0x55] & BIT(2))) << 9);
        T[Head][VGAT_HDE] += 1; // Actual value in register is #chars - 1

        T[Head][VGAT_HBS] =    (Crtc[Head][0x02])                 |
                           ((!!(Crtc[Head][0x2D] & BIT(2))) << 8) |
                           ((!!(Crtc[Head][0x55] & BIT(4))) << 9);

        T[Head][VGAT_HBE] =    (Crtc[Head][0x03]  &  0x1F )       |
                           ((!!(Crtc[Head][0x05]  & BIT(7))) << 5);

        // We need to keep track of which bits are valid or not
        // to correct for the wraparound problems.
        mask = 0x3F;
        // Horizontal Blank End [6] = CR25[4]  // Only if CR1A[2] is = 0
        if(!(Crtc[Head][0x1A] & BIT(2)) )
        {
            T[Head][VGAT_HBE] |= ((!!(Crtc[Head][0x25] & BIT(4))) << 6);
            mask |= BIT(6);
        }

        // Horizontal Blank End [7]   = CR55[6]  // Only if CR56[7] is = 1
        if(Crtc[Head][0x56] & BIT(7))
        {
            T[Head][VGAT_HBE] |= ((!!(Crtc[Head][0x55] & BIT(6))) << 7);
            mask |= BIT(7);
        }

        // Now, since we're only save the low bits of HBE, we need to
        // reconstruct the high bits.  Since HBE >= HBS, we can use the
        // high bits from HBS to infer HBE.
        // From Jeff Irwin - through Dhawal.
        // if (HBS[7:0] > HBE[7:0])
        //     final_HBE = (HBS[9:8] + 0x100) | HBE[7:0];
        // else
        //     final_HBE = HBS[9:8] | HBE[7:0];
        if((T[Head][VGAT_HBS] & mask) > (T[Head][VGAT_HBE] & mask))
        {
            T[Head][VGAT_HBE] = ((T[Head][VGAT_HBS] & (0x3FF & ~mask))+(mask+1)) | (T[Head][VGAT_HBE] & mask);
        }
        else
        {
            T[Head][VGAT_HBE] = (T[Head][VGAT_HBS]&(0x3FF & ~mask)) | (T[Head][VGAT_HBE] & mask);
        }

        T[Head][VGAT_HRS] =    (Crtc[Head][0x04])                 |
                           ((!!(Crtc[Head][0x2D] & BIT(3))) << 8) |
                           ((!!(Crtc[Head][0x56] & BIT(0))) << 9);

        T[Head][VGAT_HRE] = (Crtc[Head][0x05] & 0x1F);

        // We need to keep track of which bits are valid or not
        // to correct for the wraparound problems.
        mask = 0x1F;

        // Horizontal Sync End [5:5] = CR56[2]     // Only if CR56[7] is = 1
        if(Crtc[Head][0x56] & BIT(7))
        {
            T[Head][VGAT_HRE] = ((!!(Crtc[Head][0x56] & BIT(2))) << 5);
            mask |= BIT(5);
        }

        // Now, since we're only save the low bits of HRE, we need to
        // reconstruct the high bits.  Since HRE >= HRS, we can use the
        // high bits from HRS to infer HRE.
        // From Jeff Irwin  - through Dhawal
        // if (HRS[5:0] > HRE[5:0])
        //    final_HRE = (HRS[9:6] + 0x40) | HRE[5:0]
        // else
        //    final_HRE = HRS[9:6] | HRE[5:0]
        if((T[Head][VGAT_HRS] & mask) > (T[Head][VGAT_HRE] & mask))
        {
            T[Head][VGAT_HRE] = ((T[Head][VGAT_HRS] & (0x3FF&~mask)) + (mask+1)) | (T[Head][VGAT_HRE] & mask);
        }
        else
        {
            T[Head][VGAT_HRE] = (T[Head][VGAT_HRS] & (0x3FF&~mask)) | (T[Head][VGAT_HRE] & mask);
        }

        // Adjustments to make things line up properly
        //
        T[Head][VGAT_HBS] += 1; // help match HBS to HDE (1-based start)
        T[Head][VGAT_HBE] += 1; // help match HBE to HT (1-based start)
        // From dev_vga.ref manual:
        // For Native modes, no front porch would be horizontal blank start+1.
        // Since we have to subtract 1 here and we need to add 1 for 1-based 
        // start, effectively, we don't need to change anything then.
        // T[Head][VGAT_HRS] += 0; 
        T[Head][VGAT_HRE] += 1; // (1-based start)


        // We need to multiply each result by the number of
        // pixels in a char.
        if(sequencer[Head][1] & BIT(0))
        {
            T[Head][VGAT_HT]  *= 8;
            T[Head][VGAT_HDE] *= 8;
            T[Head][VGAT_HBS] *= 8;
            T[Head][VGAT_HBE] *= 8;
            T[Head][VGAT_HRS] *= 8;
            T[Head][VGAT_HRE] *= 8;
        }
        else
        {
            T[Head][VGAT_HT]  *= 9;
            T[Head][VGAT_HDE] *= 9;
            T[Head][VGAT_HBS] *= 9;
            T[Head][VGAT_HBE] *= 9;
            T[Head][VGAT_HRS] *= 9;
            T[Head][VGAT_HRE] *= 9;
        }

        T[Head][VGAT_VT]  =    (Crtc[Head][0x06])                 |
                           ((!!(Crtc[Head][0x07] & BIT(0))) << 8) |
                           ((!!(Crtc[Head][0x07] & BIT(5))) << 9) |
                           ((!!(Crtc[Head][0x25] & BIT(0))) <<10) |
                           ((!!(Crtc[Head][0x41] & BIT(0))) <<11) |
                           ((!!(Crtc[Head][0x41] & BIT(1))) <<12);
        // The value programmed in this register = Total number of scan lines - 2.
        T[Head][VGAT_VT] += 2;


        T[Head][VGAT_VDE] =    (Crtc[Head][0x12])                 |
                           ((!!(Crtc[Head][0x07] & BIT(1))) << 8) |
                           ((!!(Crtc[Head][0x07] & BIT(6))) << 9) |
                           ((!!(Crtc[Head][0x25] & BIT(1))) <<10) |
                           ((!!(Crtc[Head][0x41] & BIT(2))) <<11) |
                           ((!!(Crtc[Head][0x41] & BIT(3))) <<12);
        // The value in this register = Total number of displayed scan lines - 1.
        T[Head][VGAT_VDE] += 1;

        T[Head][VGAT_VBS] =    (Crtc[Head][0x15])                 |
                           ((!!(Crtc[Head][0x07] & BIT(3))) << 8) |
                           ((!!(Crtc[Head][0x09] & BIT(5))) << 9) |
                           ((!!(Crtc[Head][0x25] & BIT(3))) <<10) |
                           ((!!(Crtc[Head][0x41] & BIT(6))) <<11) |
                           ((!!(Crtc[Head][0x41] & BIT(7))) <<12);


        T[Head][VGAT_VBE] = (Crtc[Head][0x16] & 0x7F);

        // We need to keep track of which bits are valid or not
        // to correct for the wraparound problems.
        mask = 0x7F;

        // Vertical Blank End [7:7] = CR56[4]      // Only if CR56[7] is set.
        if(Crtc[Head][0x56] & BIT(7))
        {
            T[Head][VGAT_VBE] |= ((!!(Crtc[Head][0x56] & BIT(4))) << 7);
            mask |= BIT(7);
        }

        // Now, since we're only save the low bits of VBE, we need to
        // reconstruct the high bits.  Since VBE >= VBS, we can use the
        // high bits from VBS to infer VBE.
        // Interpreted from Jeff Irwin's earlier suggestions
        // if (VBS[7:0] > VBE[5:0])
        //    final_VBE = (VBS[12:8] + 0x100) | VBE[7:0]
        // else
        //    final_VBE = VBS[12:8] | VBE[7:0]
        if((T[Head][VGAT_VBS] & mask) > (T[Head][VGAT_VBE] & mask))
        {
            T[Head][VGAT_VBE] = ((T[Head][VGAT_VBS] & (0x1FFF&~mask)) + (mask+1)) | (T[Head][VGAT_VBE] & mask);
        }
        else
        {
            T[Head][VGAT_VBE] = (T[Head][VGAT_VBS] & (0x1FFF&~mask)) | (T[Head][VGAT_VBE] & mask);
        }

        

        T[Head][VGAT_VRS] =    (Crtc[Head][0x10])                 |
                           ((!!(Crtc[Head][0x07] & BIT(2))) << 8) |
                           ((!!(Crtc[Head][0x07] & BIT(7))) << 9) |
                           ((!!(Crtc[Head][0x25] & BIT(2))) <<10) |
                           ((!!(Crtc[Head][0x41] & BIT(4))) <<11) |
                           ((!!(Crtc[Head][0x41] & BIT(5))) <<12);

        T[Head][VGAT_VRE] = (((U032) Crtc[Head][0x11]) &  0x0F );

        // Now, since we're only save the low bits of VRE, we need to
        // reconstruct the high bits.  Since VRE >= VRS, we can use the
        // high bits from HRS to infer VRE.
        // Interpreted from Jeff Irwin's earlier suggestions
        // if (VRS[3:0] > VRE[3:0])
        //    final_VRE = (VRS[12:4] + 0x10) | VRE[3:0]
        // else
        //    final_VRE = VRS[12:4] | VRE[3:0]
        if((T[Head][VGAT_VRS]&0xF) > (T[Head][VGAT_VRE]&0xF))
        {
            T[Head][VGAT_VRE] = ((T[Head][VGAT_VRS]&0x1FF0)+0x10) | (T[Head][VGAT_VRE]&0xF);
        }
        else
        {
            T[Head][VGAT_VRE] = (T[Head][VGAT_VRS]&0x1FF0) | (T[Head][VGAT_VRE]&0xF);
        }

        // Adjustments to make things line up properly
        //
        T[Head][VGAT_VBS] += 1; // Help match VBS to VDE (1-based start)
        T[Head][VGAT_VBE] += 1; // Help match VBE to VT (1-based start)
        T[Head][VGAT_VRS] += 1; // (1-based start)
        T[Head][VGAT_VRE] += 1; // (1-based start)
    }

    dprintf("                                              \n");

    // Are we dumping or writing??
    if(value == 0xFFFFFFFF)
    {
        // We're dumping
        if (NumCrtcs > 1)
        {
            if(broadcastWasEnabled)
                dprintf("lw: Original Head was Broadcast\n");
            else if(prevHead)
                dprintf("lw: Original Head was B\n");
            else
                dprintf("lw: Original Head was A\n");
        }
        else
        {
            dprintf("lw: Original Head was A\n");
        }

        dprintf("\n");

        ///////////////////////////
        // Print Results
        ///////////////////////////        
        dprintf("%s", NamedT[0xC]);
        for(Head = 0; Head < NumCrtcs; Head++)
        {
            if(!(HeadBitmask & BIT(Head)) )
              continue;

            dprintf("**Head %c**    ", 0x41+Head);
        }
        dprintf("\n");

        for(i=0; i<0xC; i++)
        {
            // Only print out the timings requested
            if( (Timing != VGAT_ALL) &&
                (Timing != i) )
              continue;

            dprintf("%s",NamedT[i]);
            for(Head = 0; Head < NumCrtcs; Head++)
            {
                if(!(HeadBitmask & BIT(Head)) )
                  continue;

                if(format & 0x1)
                {
                    dprintf("  %8d    ", T[Head][i]);
                }
                else
                {
                    dprintf("0x%08x    ", T[Head][i]);
                }
            }
            dprintf("\n");

            // Separate Horizontal and vertical
            if(i==5)
              dprintf("\n");
        }
    }
    else
    {
        U032 orgvalue = value;

        // ok we're programming the timings here.
        // Print out the head first
        dprintf("            %s", NamedT[0xC]);
        for(Head = 0; Head < NumCrtcs; Head++)
        {
            if(!(HeadBitmask & BIT(Head)) )
              continue;

            dprintf("**Head %c**    ", 0x41+Head);
        }
        dprintf("\n");
        // Then print out the old timing:
        dprintf("Old Timing: ");
        dprintf("%s",NamedT[Timing]);
        for(Head = 0; Head < NumCrtcs; Head++)
        {
            if(!(HeadBitmask & BIT(Head)) )
              continue;

            if(format & 0x1)
            {
                dprintf("  %8d    ", T[Head][Timing]);
            }
            else
            {
                dprintf("0x%08x    ", T[Head][Timing]);
            }
        }
        dprintf("\n");
        // Then print out the new timing:
        dprintf("New Timing: ");
        dprintf("%s",NamedT[Timing]);
        for(Head = 0; Head < NumCrtcs; Head++)
        {
            if(!(HeadBitmask & BIT(Head)) )
              continue;

            if(format & 0x1)
            {
                dprintf("  %8d    ", value);
            }
            else
            {
                dprintf("0x%08x    ", value);
            }
        }
        dprintf("\n");

        dprintf("Writing out value now....");


        for(Head = 0; Head < NumCrtcs; Head++)
        {
            U032 BlankWidthChars, SyncWidthChars, BlankHeight;
            U032 PixelsPerChar = ((sequencer[Head][1] & BIT(0))?8:9);

            if(!(HeadBitmask & BIT(Head)) )
              continue;

            // Pre-divide by 8 or 9 pixels per char for horizontal timings
            if( Timing <= VGAT_HRE )
              value = orgvalue/PixelsPerChar;

            if(Head == 1)
              crtcOffset = 0x2000;
            else
              crtcOffset = 0x0;

            oldIndex = GPU_REG_RD08(0x6013d4 + crtcOffset);
            oldLock = UnlockExtendedCRTCs(crtcOffset);

            // Unlock CR0-7
            REG_WRCR(0x11, ((U008)(Crtc[Head][0x11] & ~0x80)), crtcOffset);

            BlankWidthChars = ( (T[Head][VGAT_HBE] - T[Head][VGAT_HBS]) / PixelsPerChar );

            SyncWidthChars = ( (T[Head][VGAT_HRE] - T[Head][VGAT_HRS]) / PixelsPerChar );

            BlankHeight = T[Head][VGAT_VBE] - T[Head][VGAT_VBS];

            // Now let's callwlate the new Timing:
            switch(Timing)
            {
                case VGAT_HT:
                    // test
                    if(value<5)
                    {
                        dprintf("\nERROR: Horizontal Total must be %d or greater! Exiting...\n",
                               (PixelsPerChar*5) );
                        break;
                    }
                    // Actual value in register is #chars - 5
                    value-=5;

                    // Write out values:
                    Crtc[Head][0x00] = (U008)(value & 0xFF);
                    REG_WRCR(0x00, ((U008)Crtc[Head][0x00]), crtcOffset);

                    Crtc[Head][0x2D] &= ~BIT(0);
                    Crtc[Head][0x2D] |= (BIT(0) * !!(value & 0x100));
                    REG_WRCR(0x2D, ((U008)Crtc[Head][0x2D]), crtcOffset);

                    Crtc[Head][0x55] &= ~BIT(0);
                    Crtc[Head][0x55] |= (BIT(0) * !!(value & 0x200));
                    REG_WRCR(0x55, ((U008)Crtc[Head][0x55]), crtcOffset);
                    break;
                case VGAT_HDE:
                    // test
                    if(value<1)
                    {
                        dprintf("\nERROR: Horizontal Display End must be %d or greater! Exiting...\n",
                               PixelsPerChar );
                        break;
                    }
                    // Actual value in register is #chars - 1
                    value-=1;

                    // Write out values:
                    Crtc[Head][0x01] = (U008)(value & 0xFF);
                    REG_WRCR(0x01, ((U008)Crtc[Head][0x01]), crtcOffset);

                    Crtc[Head][0x2D] &= ~BIT(1);
                    Crtc[Head][0x2D] |= (BIT(1) * !!(value & 0x100));
                    REG_WRCR(0x2D, ((U008)Crtc[Head][0x2D]), crtcOffset);

                    Crtc[Head][0x55] &= ~BIT(2);
                    Crtc[Head][0x55] |= (BIT(2) * !!(value & 0x200));
                    REG_WRCR(0x55, ((U008)Crtc[Head][0x55]), crtcOffset);
                    break;
                case VGAT_HBS:
                    // test
                    if(value<1)
                    {
                        dprintf("\nERROR: Horizontal Blank Start must be %d or greater! Exiting...\n",
                               PixelsPerChar );
                        break;
                    }
                    // Actual value in register is #chars - 1
                    value-=1;

                    // Write out values:
                    Crtc[Head][0x02] = (U008)(value & 0xFF);
                    REG_WRCR(0x02, ((U008)Crtc[Head][0x02]), crtcOffset);

                    Crtc[Head][0x2D] &= ~BIT(2);
                    Crtc[Head][0x2D] |= (BIT(2) * !!(value & 0x100));
                    REG_WRCR(0x2D, ((U008)Crtc[Head][0x2D]), crtcOffset);

                    Crtc[Head][0x55] &= ~BIT(4);
                    Crtc[Head][0x55] |= (BIT(4) * !!(value & 0x200));
                    REG_WRCR(0x55, ((U008)Crtc[Head][0x55]), crtcOffset);
                    break;
                case VGAT_HBE:
                    // test
                    if( orgvalue < T[Head][VGAT_HBS] )
                    {
                        dprintf("\nERROR: The requested Horizontal Blank End (%d) must be greater than or equal\n", orgvalue);
                        dprintf("       to the current Horizontal Blank Start (%d)! Exiting...\n", T[Head][VGAT_HBS]);
                        break;
                    }
                    if( orgvalue > (T[Head][VGAT_HBS]+(255*PixelsPerChar)) )
                    {
                        dprintf("\nERROR: The requested Horizontal Blank End (%d) must be less than or equal\n", orgvalue);
                        dprintf("       to the current Horizontal Blank Start + %d (%d+%d=%d)! Exiting...\n",
                                        (255*PixelsPerChar), T[Head][VGAT_HBS], 
                                        (255*PixelsPerChar), 
                                        (T[Head][VGAT_HBS]+(255*PixelsPerChar)) );
                        break;
                    }
                    // Actual value in register is #chars - 1
                    value-=1;

                    // Write out values:
                    Crtc[Head][0x03] &= ~(0x1F);
                    Crtc[Head][0x03] |= (U008)(value & 0x1F);
                    REG_WRCR(0x03, ((U008)Crtc[Head][0x03]), crtcOffset);

                    Crtc[Head][0x05] &= ~BIT(7);
                    Crtc[Head][0x05] |= (BIT(7) * !!(value & 0x20));
                    REG_WRCR(0x05, ((U008)Crtc[Head][0x05]), crtcOffset);

                    // Horizontal Blank End [6] = CR25[4]  // Only if CR1A[2] is = 0
                    // Go ahead and write out the 7th bit
                    Crtc[Head][0x25] &= ~BIT(4);
                    Crtc[Head][0x25] |= (BIT(4) * !!(value & 0x40));
                    REG_WRCR(0x25, ((U008)Crtc[Head][0x25]), crtcOffset);

                    // Do we need a 7th bit?
                    if(value & 0x40)
                    {
                        // Yes...
                        // Clear Cr1A[2]
                        Crtc[Head][0x1A] &= ~BIT(2);
                        REG_WRCR(0x1A, ((U008)Crtc[Head][0x1A]), crtcOffset);
                    }
                    else
                    {
                        // No...
                        // Set Cr1A[2]
                        Crtc[Head][0x1A] |= BIT(2);
                        REG_WRCR(0x1A, ((U008)Crtc[Head][0x1A]), crtcOffset);
                    }

                    // Horizontal Blank End [7]   = CR55[6]  // Only if CR56[7] is = 1
                    // Go ahead and write out the 8th bit
                    Crtc[Head][0x55] &= ~BIT(6);
                    Crtc[Head][0x55] |= (BIT(6) * !!(value & 0x80));
                    REG_WRCR(0x55, ((U008)Crtc[Head][0x55]), crtcOffset);

                    // Do we need a 8th bit here or for VBE or a 6th bit for HRE??
                    if( ( value & 0x80 )          || 
                        ( SyncWidthChars > 0x1F ) ||
                        ( BlankHeight > 0x7F ) )
                    {
                        // Yes... but make sure that HRE's 6th bit is set correctly
                        // Horizontal Sync End [5:5] = CR56[2]     // Only if CR56[7] is = 1
                        Crtc[Head][0x56] &= ~BIT(2);
                        Crtc[Head][0x56] |= (BIT(2) * !!(T[Head][VGAT_HRE] & 0x20));
                        REG_WRCR(0x56, ((U008)Crtc[Head][0x56]), crtcOffset);

                        // Also make sure that VBE's 8th bit is set as well
                        // Vertical Blank End [7:7] = CR56[4]      // Only if CR56[7] is set.
                        Crtc[Head][0x56] &= ~BIT(4);
                        Crtc[Head][0x56] |= (BIT(4) * !!(T[Head][VGAT_VBE] & 0x80));
                        REG_WRCR(0x56, ((U008)Crtc[Head][0x56]), crtcOffset);


                        // Set Cr56[7]
                        Crtc[Head][0x56] |= BIT(7);
                        REG_WRCR(0x56, ((U008)Crtc[Head][0x56]), crtcOffset);
                    }
                    else
                    {
                        // No...
                        // Clear Cr56[7]
                        Crtc[Head][0x56] &= ~BIT(7);
                        REG_WRCR(0x56, ((U008)Crtc[Head][0x56]), crtcOffset);
                    }
                    break;
                case VGAT_HRS:
                    // Actual value in register is #chars + 1
                    value+=1;

                    // Write out values:
                    Crtc[Head][0x04] = (U008)(value & 0xFF);
                    REG_WRCR(0x04, ((U008)Crtc[Head][0x04]), crtcOffset);

                    Crtc[Head][0x2D] &= ~BIT(3);
                    Crtc[Head][0x2D] |= (BIT(3) * !!(value & 0x100));
                    REG_WRCR(0x2D, ((U008)Crtc[Head][0x2D]), crtcOffset);

                    Crtc[Head][0x56] &= ~BIT(0);
                    Crtc[Head][0x56] |= (BIT(0) * !!(value & 0x200));
                    REG_WRCR(0x56, ((U008)Crtc[Head][0x56]), crtcOffset);
                    break;
                    break;
                case VGAT_HRE:
                    // test
                    if( orgvalue < T[Head][VGAT_HRS] )
                    {
                        dprintf("\nERROR: The requested Horizontal Retrace End (%d) must be greater than or equal\n", orgvalue);
                        dprintf("       to the current Horizontal Retrace Start (%d)! Exiting...\n", T[Head][VGAT_HRS]);
                        break;
                    }
                    if( orgvalue > (T[Head][VGAT_HRS]+(63*PixelsPerChar)) )
                    {
                        dprintf("\nERROR: The requested Horizontal Retrace End (%d) must be less than or equal\n", orgvalue);
                        dprintf("       to the current Horizontal Retrace Start + %d (%d+%d=%d)! Exiting...\n",
                                        (63*PixelsPerChar), T[Head][VGAT_HRS], 
                                        (63*PixelsPerChar), 
                                        (T[Head][VGAT_HRS] + (63*PixelsPerChar)) );
                        break;
                    }
                    // Actual value in register is #chars - 1
                    value-=1;

                    // Write out values:
                    Crtc[Head][0x05] &= ~(0x1F);
                    Crtc[Head][0x05] |= (U008)(value & 0x1F);
                    REG_WRCR(0x05, ((U008)Crtc[Head][0x05]), crtcOffset);

                    // Horizontal Sync End [5:5] = CR56[2]     // Only if CR56[7] is = 1
                    // Go ahead and write out the 6th bit
                    Crtc[Head][0x56] &= ~BIT(2);
                    Crtc[Head][0x56] |= (BIT(2) * !!(value & 0x20));
                    REG_WRCR(0x56, ((U008)Crtc[Head][0x56]), crtcOffset);

                    // Do we need a 6th bit here or an 8th bit for HBE or VBE?
                    if( ( value & 0x20 )           ||
                        ( BlankWidthChars > 0x7F ) ||
                        ( BlankHeight > 0x7F ) )
                    {
                        // Yes... but make sure that HBE's 8th bit is set correctly
                        // Horizontal Blank End [7]   = CR55[6]  // Only if CR56[7] is = 1
                        Crtc[Head][0x55] &= ~BIT(6);
                        Crtc[Head][0x55] |= (BIT(6) * !!(T[Head][VGAT_HBE] & 0x80));
                        REG_WRCR(0x55, ((U008)Crtc[Head][0x55]), crtcOffset);

                        // Also make sure that VBE's 8th bit is set as well
                        // Vertical Blank End [7:7] = CR56[4]      // Only if CR56[7] is set.
                        Crtc[Head][0x56] &= ~BIT(4);
                        Crtc[Head][0x56] |= (BIT(4) * !!(T[Head][VGAT_VBE] & 0x80));
                        REG_WRCR(0x56, ((U008)Crtc[Head][0x56]), crtcOffset);


                        // Set Cr56[7]
                        Crtc[Head][0x56] |= BIT(7);
                        REG_WRCR(0x56, ((U008)Crtc[Head][0x56]), crtcOffset);
                    }
                    else
                    {
                        // No...
                        // Clear Cr56[7]
                        Crtc[Head][0x56] &= ~BIT(7);
                        REG_WRCR(0x56, ((U008)Crtc[Head][0x56]), crtcOffset);
                    }
                    break;
                case VGAT_VT:
                    // test
                    if(value<2)
                    {
                        dprintf("\nERROR: Vertical Total must be greater than 2! Exiting...\n");
                        break;
                    }
                    // Actual value in register is - 2
                    value-=2;

                    // Write out values:
                    Crtc[Head][0x06] = (U008)(value & 0xFF);
                    REG_WRCR(0x06, ((U008)Crtc[Head][0x06]), crtcOffset);

                    Crtc[Head][0x07] &= ~BIT(0);
                    Crtc[Head][0x07] |= (BIT(0) * !!(value & 0x100));
                    REG_WRCR(0x07, ((U008)Crtc[Head][0x07]), crtcOffset);

                    Crtc[Head][0x07] &= ~BIT(5);
                    Crtc[Head][0x07] |= (BIT(5) * !!(value & 0x200));
                    REG_WRCR(0x07, ((U008)Crtc[Head][0x07]), crtcOffset);

                    Crtc[Head][0x25] &= ~BIT(0);
                    Crtc[Head][0x25] |= (BIT(0) * !!(value & 0x400));
                    REG_WRCR(0x25, ((U008)Crtc[Head][0x25]), crtcOffset);

                    Crtc[Head][0x41] &= ~(0x3);
                    Crtc[Head][0x41] |= (0x3 & (value >> 11));
                    REG_WRCR(0x41, ((U008)Crtc[Head][0x41]), crtcOffset);
                    break;
                case VGAT_VDE:
                    // test
                    if(value<1)
                    {
                        dprintf("\nERROR: Vertical Display Enable must be greater than 1! Exiting...\n");
                        break;
                    }
                    // Actual value in register is - 1
                    value-=1;

                    // Write out values:
                    Crtc[Head][0x12] = (U008)(value & 0xFF);
                    REG_WRCR(0x12, ((U008)Crtc[Head][0x12]), crtcOffset);

                    Crtc[Head][0x07] &= ~BIT(1);
                    Crtc[Head][0x07] |= (BIT(1) * !!(value & 0x100));
                    REG_WRCR(0x07, ((U008)Crtc[Head][0x07]), crtcOffset);

                    Crtc[Head][0x07] &= ~BIT(6);
                    Crtc[Head][0x07] |= (BIT(6) * !!(value & 0x200));
                    REG_WRCR(0x07, ((U008)Crtc[Head][0x07]), crtcOffset);

                    Crtc[Head][0x25] &= ~BIT(1);
                    Crtc[Head][0x25] |= (BIT(1) * !!(value & 0x400));
                    REG_WRCR(0x25, ((U008)Crtc[Head][0x25]), crtcOffset);

                    Crtc[Head][0x41] &= ~(0xc);
                    Crtc[Head][0x41] |= (0xc & (value >> 9));
                    REG_WRCR(0x41, ((U008)Crtc[Head][0x41]), crtcOffset);
                    break;
                case VGAT_VBS:
                    // test
                    if(value<1)
                    {
                        dprintf("\nERROR: Vertical Blank Start must be greater than 1! Exiting...\n");
                        break;
                    }
                    // Actual value in register is - 1
                    value-=1;

                    // Write out values:
                    Crtc[Head][0x15] = (U008)(value & 0xFF);
                    REG_WRCR(0x15, ((U008)Crtc[Head][0x15]), crtcOffset);

                    Crtc[Head][0x07] &= ~BIT(3);
                    Crtc[Head][0x07] |= (BIT(3) * !!(value & 0x100));
                    REG_WRCR(0x07, ((U008)Crtc[Head][0x07]), crtcOffset);

                    Crtc[Head][0x09] &= ~BIT(5);
                    Crtc[Head][0x09] |= (BIT(5) * !!(value & 0x200));
                    REG_WRCR(0x09, ((U008)Crtc[Head][0x09]), crtcOffset);

                    Crtc[Head][0x25] &= ~BIT(3);
                    Crtc[Head][0x25] |= (BIT(3) * !!(value & 0x400));
                    REG_WRCR(0x25, ((U008)Crtc[Head][0x25]), crtcOffset);

                    Crtc[Head][0x41] &= ~(0xc0);
                    Crtc[Head][0x41] |= (0xc0 & (value >> 5));
                    REG_WRCR(0x41, ((U008)Crtc[Head][0x41]), crtcOffset);
                    break;
                case VGAT_VBE:
                    // test
                    if( value < T[Head][VGAT_VBS] )
                    {
                        dprintf("\nERROR: The requested Vertical Blank End (%d) must be greater than or equal\n", value);
                        dprintf("       the current Vertical Blank Start (%d)! Exiting...\n", T[Head][VGAT_VBS]);
                        break;
                    }
                    if( value > (255+T[Head][VGAT_VBS]) )
                    {
                        dprintf("\nERROR: The requested Vertical Blank End (%d) must be less than or equal\n", value);
                        dprintf("       to the current Vertical Blank Start + 255 lines (%d+255=%d)! Exiting...\n",
                                        T[Head][VGAT_VBS], (T[Head][VGAT_VBS]+255) );
                        break;
                    }
                    // Actual value in register is - 1
                    value-=1;
                    mask = value-T[Head][VGAT_VBS];

                    // Write out values:
                    Crtc[Head][0x16] &= ~(0x7F);
                    Crtc[Head][0x16] |= (U008)(value & 0x7F);
                    REG_WRCR(0x16, ((U008)Crtc[Head][0x16]), crtcOffset);

                    // Vertical Blank End [7:7] = CR56[4]      // Only if CR56[7] is set.
                    // Go ahead and write out the 8th bit
                    Crtc[Head][0x56] &= ~BIT(4);
                    Crtc[Head][0x56] |= (BIT(4) * !!(value & 0x80));
                    REG_WRCR(0x56, ((U008)Crtc[Head][0x56]), crtcOffset);

                    // Do we need a 8th bit here or for HBE or a sixth bit for HRE?
                    if( ( value & 0x80 )           ||
                        ( BlankWidthChars > 0x7F ) ||
                        ( SyncWidthChars > 0x1F ) )
                    {
                        // Yes... but make sure that HBE's 8th bit is set correctly
                        // Horizontal Blank End [7]   = CR55[6]  // Only if CR56[7] is = 1
                        Crtc[Head][0x55] &= ~BIT(6);
                        Crtc[Head][0x55] |= (BIT(6) * !!(T[Head][VGAT_HBE] & 0x80));
                        REG_WRCR(0x55, ((U008)Crtc[Head][0x55]), crtcOffset);

                        // Also make sure that HRE's 6th bit is set correctly
                        // Horizontal Sync End [5:5] = CR56[2]     // Only if CR56[7] is = 1
                        Crtc[Head][0x56] &= ~BIT(2);
                        Crtc[Head][0x56] |= (BIT(2) * !!(T[Head][VGAT_HRE] & 0x20));
                        REG_WRCR(0x56, ((U008)Crtc[Head][0x56]), crtcOffset);

                        // Set Cr56[7]
                        Crtc[Head][0x56] |= BIT(7);
                        REG_WRCR(0x56, ((U008)Crtc[Head][0x56]), crtcOffset);
                    }
                    else
                    {
                        // No...
                        // Clear Cr56[7]
                        Crtc[Head][0x56] &= ~BIT(7);
                        REG_WRCR(0x56, ((U008)Crtc[Head][0x56]), crtcOffset);
                    }
                    break;
                case VGAT_VRS:
                    // test
                    if(value<1)
                    {
                        dprintf("\nERROR: Vertical Retrace Start must be greater than 1! Exiting...\n");
                        break;
                    }
                    // Actual value in register is - 1
                    value-=1;

                    // Write out values:
                    Crtc[Head][0x10] = (U008)(value & 0xFF);
                    REG_WRCR(0x10, ((U008)Crtc[Head][0x10]), crtcOffset);

                    Crtc[Head][0x07] &= ~BIT(2);
                    Crtc[Head][0x07] |= (BIT(2) * !!(value & 0x100));
                    REG_WRCR(0x07, ((U008)Crtc[Head][0x07]), crtcOffset);

                    Crtc[Head][0x07] &= ~BIT(7);
                    Crtc[Head][0x07] |= (BIT(7) * !!(value & 0x200));
                    REG_WRCR(0x07, ((U008)Crtc[Head][0x07]), crtcOffset);

                    Crtc[Head][0x25] &= ~BIT(2);
                    Crtc[Head][0x25] |= (BIT(2) * !!(value & 0x400));
                    REG_WRCR(0x25, ((U008)Crtc[Head][0x25]), crtcOffset);

                    Crtc[Head][0x41] &= ~(0x30);
                    Crtc[Head][0x41] |= (0x30 & (value >> 7));
                    REG_WRCR(0x41, ((U008)Crtc[Head][0x41]), crtcOffset);
                    break;
                    break;
                case VGAT_VRE:
                    // test
                    if( value < T[Head][VGAT_VRS] )
                    {
                        dprintf("\nERROR: The requested Vertical Retrace End (%d) must be greater than or equal\n", value);
                        dprintf("       to the current Vertical Retrace Start (%d)! Exiting...\n", T[Head][VGAT_VRS]);
                        break;
                    }
                    if( value > (T[Head][VGAT_VRS]+15) )
                    {
                        dprintf("\nERROR: The requested Vertical Retrace End (%d) must be less than or equal\n", value);
                        dprintf("       to the current Vertical Retrace Start + 15 (%d+15=%d)! Exiting...\n",
                                        T[Head][VGAT_VRS], (T[Head][VGAT_VRS]+15) );
                        break;
                    }
                    // Actual value in register is - 1
                    value-=1;

                    // Write out values:
                    Crtc[Head][0x11] &= ~(0xF);
                    Crtc[Head][0x11] |= (U008)(value & 0xF);
                    REG_WRCR(0x11, ((U008)Crtc[Head][0x11]), crtcOffset);
                    break;
            }

            dprintf("...done!\n");

            // restore lock for CR0-7
            REG_WRCR(0x11, ((U008)(Crtc[Head][0x11])), crtcOffset);

            RestoreExtendedCRTCs(oldLock, crtcOffset);
            GPU_REG_WR08(0x6013d4 + crtcOffset, oldIndex);
        }
      
    }

    dprintf("\n\n");

    // Restore Broadcast state
    SetBroadcastBit(broadcastWasEnabled);
}

VOID dumpPalette(U032 flags)
{
    U032 i, j, k;
    U032 NumCrtcs = GetNumCrtcs();
    U032 rgb[MAX_CRTCS][256];
    U032 crtcOffset = 0x0;
    U032 genControl[MAX_CRTCS];

    for(i=0; i<NumCrtcs; i++)
    {
        dprintf("lw: Reading Head %c Palette                                  \r", ((char) 'A'+i));
        crtcOffset = i*0x2000;

        genControl[i] = GPU_REG_RD32(0x680600 + crtcOffset);

        if(flags & BIT(0))
        {
            GPU_REG_WR32(0x680600 + crtcOffset, (genControl[i] | BIT(21)));
            
            // Read in 10 bit lut here
            // Start at index 0 and auto increment
            GPU_REG_WR32(0x680620 + crtcOffset, 0);

            for(j=0; j<256; j++)
            {
                if((j%0x10)==0)
                {
                    dprintf("lw: Reading Head %c Palette Reg: 0x%02x       \r", ((char) 'A'+i), j);
                }
                rgb[i][j] = GPU_REG_RD32(0x00680624 + crtcOffset);
            }

            GPU_REG_WR32(0x680600 + crtcOffset, genControl[i]);
        }
        else
        {
            // Read in 8 bit VGA lut here.
            GPU_REG_WR08(0x6813c7 + crtcOffset, 0);

            for(j=0; j<256; j++)
            {
                if((j%0x10)==0)
                {
                    dprintf("lw: Reading Head %c Palette Reg: 0x%02x       \r", ((char) 'A'+i), j);
                }
                rgb[i][j] = (U032) GPU_REG_RD08(0x6813c9 + crtcOffset);
                rgb[i][j] <<= 8;
                rgb[i][j] |= (U032) GPU_REG_RD08(0x6813c9 + crtcOffset);
                rgb[i][j] <<= 8;
                rgb[i][j] |= (U032) GPU_REG_RD08(0x6813c9 + crtcOffset);
            }
        }
    }


    ///////////////////////////
    // Print Results
    ///////////////////////////        
    dprintf("        ");
    for(i = 0; i < NumCrtcs; i++)
    {
        dprintf("**Head %c**                              ", 0x41+i);
    }
    dprintf("\n");

    // print out which type of lut is used here.
    dprintf("Type:   ");
    for(i = 0; i < NumCrtcs; i++)
    {
        if(flags & BIT(0))
        {
            dprintf("10-bit b[29:20] g[19:10] r[9:0]         ");
        }
        else
        {
            dprintf("8-bit  b[23:16] g[15:7]  r[7:0]         ");
        }
    }
    dprintf("\n");
    

    // print out each lut entry
    dprintf("Entry   Palette Data\n");

    for(j=0; j<256; j+=4)
    {
        dprintf("    %02X  ",j);
        for(i = 0; i < NumCrtcs; i++)
        {
            for(k=0; k<4; k++)
            {
                dprintf("%08x ", rgb[i][j+k]);
            }
            dprintf("    ");
        }
        dprintf("\n");
    }
    dprintf("\n");

    // For VGA palettes, we should print out the pixel mask
    if(!(flags & BIT(0)) )
    {
        dprintf("68x3c6: ");
        for(i = 0; i < NumCrtcs; i++)
        {
            dprintf("Pixel Mask for Head %c: 0x%02X             ", 
                    ((char) 'A'+i), 
                    GPU_REG_RD08(0x006813C6 + crtcOffset));
        }
        dprintf("\n");
    }


}


//-----------------------------------------------------
// tvoRegRd
// 
//-----------------------------------------------------
U032 tvoRegRd(U032 reg)
{
    GPU_REG_WR32(LW_PTVO_ENCODER_INDEX, reg);
    return (U008) GPU_REG_RD32(LW_PTVO_ENCODER_DATA);
}

//-----------------------------------------------------
// tvoRegWr
//
//-----------------------------------------------------
VOID tvoRegWr(U032 reg, U032 val)
{
    GPU_REG_WR32(LW_PTVO_ENCODER_INDEX, reg);
    GPU_REG_WR32(LW_PTVO_ENCODER_DATA, val);
}

//-----------------------------------------------------
// dumpTvRegs
// - Only doing internal regs for now...
//-----------------------------------------------------
typedef struct PTVO_CVE_REG
{
    char *desc;
    U008 idx;
} PTVO_CVE_REG;

#define PTVO_CVE_ENTRY(x) {#x, x}

VOID dumpTvRegs(U032 crtcOffset)
{
    U008 oldIndex, oldLock, cr3d;
    U032 oldCveIndex;
    U032 i = 0;

    PTVO_CVE_REG ptvoCveRegList[] ={PTVO_CVE_ENTRY(LW_PTVO_INDIR_CHROMA_BYTE3),
                                    PTVO_CVE_ENTRY(LW_PTVO_INDIR_CHROMA_BYTE2),
                                    PTVO_CVE_ENTRY(LW_PTVO_INDIR_CHROMA_BYTE1),
                                    PTVO_CVE_ENTRY(LW_PTVO_INDIR_CHROMA_BYTE0),
                                    PTVO_CVE_ENTRY(LW_PTVO_INDIR_CHROMA_PHASE),
                                    PTVO_CVE_ENTRY(LW_PTVO_INDIR_MISC0),
                                    PTVO_CVE_ENTRY(LW_PTVO_INDIR_MISC1),
                                    PTVO_CVE_ENTRY(LW_PTVO_INDIR_MISC2),
                                    PTVO_CVE_ENTRY(LW_PTVO_INDIR_HSYNC_WIDTH),
                                    PTVO_CVE_ENTRY(LW_PTVO_INDIR_BURST_WIDTH),
                                    PTVO_CVE_ENTRY(LW_PTVO_INDIR_BACK_PORCH),
                                    PTVO_CVE_ENTRY(LW_PTVO_INDIR_CB_BURST_LEVEL),
                                    PTVO_CVE_ENTRY(LW_PTVO_INDIR_CR_BURST_LEVEL),
                                    PTVO_CVE_ENTRY(LW_PTVO_INDIR_SLAVE_MODE),
                                    PTVO_CVE_ENTRY(LW_PTVO_INDIR_BLACK_LEVEL_MSB),
                                    PTVO_CVE_ENTRY(LW_PTVO_INDIR_BLACK_LEVEL_LSB),
                                    PTVO_CVE_ENTRY(LW_PTVO_INDIR_BLANK_LEVEL_MSB),
                                    PTVO_CVE_ENTRY(LW_PTVO_INDIR_BLANK_LEVEL_LSB),
                                    PTVO_CVE_ENTRY(LW_PTVO_INDIR_NUM_LINES_MSB),
                                    PTVO_CVE_ENTRY(LW_PTVO_INDIR_NUM_LINES_LSB),
                                    PTVO_CVE_ENTRY(LW_PTVO_INDIR_WHITE_LEVEL_MSB),
                                    PTVO_CVE_ENTRY(LW_PTVO_INDIR_WHITE_LEVEL_LSB),
                                    PTVO_CVE_ENTRY(LW_PTVO_INDIR_CR_GAIN),
                                    PTVO_CVE_ENTRY(LW_PTVO_INDIR_CB_GAIN),
                                    PTVO_CVE_ENTRY(LW_PTVO_INDIR_TINT),
                                    PTVO_CVE_ENTRY(LW_PTVO_INDIR_BREEZE_WAY),
                                    PTVO_CVE_ENTRY(LW_PTVO_INDIR_FRONT_PORCH),
                                    PTVO_CVE_ENTRY(LW_PTVO_INDIR_ACTIVELINE_MSB),
                                    PTVO_CVE_ENTRY(LW_PTVO_INDIR_ACTIVELINE_LSB),
                                    PTVO_CVE_ENTRY(LW_PTVO_INDIR_FIRSTVIDEOLINE),
                                    PTVO_CVE_ENTRY(LW_PTVO_INDIR_MISC3),
                                    PTVO_CVE_ENTRY(LW_PTVO_INDIR_SYNC_LEVEL),
                                    PTVO_CVE_ENTRY(LW_PTVO_INDIR_VBI_BLANK_LEVEL_MSB),
                                    PTVO_CVE_ENTRY(LW_PTVO_INDIR_VBI_BLANK_LEVEL_LSB),
                                    PTVO_CVE_ENTRY(LW_PTVO_INDIR_SOFT_RESET),
                                    PTVO_CVE_ENTRY(LW_PTVO_INDIR_PRODUCT_VERSION),
                                    PTVO_CVE_ENTRY(LW_PTVO_INDIR_WSS_MISC0),
                                    PTVO_CVE_ENTRY(LW_PTVO_INDIR_WSS_CLOCK_MSB),
                                    PTVO_CVE_ENTRY(LW_PTVO_INDIR_WSS_CLOCK_LSB),
                                    PTVO_CVE_ENTRY(LW_PTVO_INDIR_WSS_DATAF1_USB),
                                    PTVO_CVE_ENTRY(LW_PTVO_INDIR_WSS_DATAF1_MSB),
                                    PTVO_CVE_ENTRY(LW_PTVO_INDIR_WSS_DATAF1_LSB),
                                    PTVO_CVE_ENTRY(LW_PTVO_INDIR_WSS_DATAF0_USB),
                                    PTVO_CVE_ENTRY(LW_PTVO_INDIR_WSS_DATAF0_MSB),
                                    PTVO_CVE_ENTRY(LW_PTVO_INDIR_WSS_DATAF0_LSB),
                                    PTVO_CVE_ENTRY(LW_PTVO_INDIR_WSS_LINEF1),
                                    PTVO_CVE_ENTRY(LW_PTVO_INDIR_WSS_LINEF0),
                                    PTVO_CVE_ENTRY(LW_PTVO_INDIR_WSS_LEVEL_MSB),
                                    PTVO_CVE_ENTRY(LW_PTVO_INDIR_WSS_LEVEL_LSB),
                                    PTVO_CVE_ENTRY(LW_PTVO_INDIR_NOTCH_MISC0),
                                    PTVO_CVE_ENTRY(LW_PTVO_INDIR_CHROMA_FREQ2_BYTE3),
                                    PTVO_CVE_ENTRY(LW_PTVO_INDIR_CHROMA_FREQ2_BYTE2),
                                    PTVO_CVE_ENTRY(LW_PTVO_INDIR_CHROMA_FREQ2_BYTE1),
                                    PTVO_CVE_ENTRY(LW_PTVO_INDIR_CHROMA_FREQ2_BYTE0),
                                    PTVO_CVE_ENTRY(LW_PTVO_INDIR_TT_LEVEL),
                                    PTVO_CVE_ENTRY(LW_PTVO_INDIR_TT_MISC0),
                                    PTVO_CVE_ENTRY(LW_PTVO_INDIR_TT_LINEF0START),
                                    PTVO_CVE_ENTRY(LW_PTVO_INDIR_TT_LINEF0END),
                                    PTVO_CVE_ENTRY(LW_PTVO_INDIR_TT_LINEF1START),
                                    PTVO_CVE_ENTRY(LW_PTVO_INDIR_TT_LINEF1END),
                                    PTVO_CVE_ENTRY(LW_PTVO_INDIR_TT_CLOCK),
                                    PTVO_CVE_ENTRY(LW_PTVO_INDIR_TT_MISC1),
                                    PTVO_CVE_ENTRY(LW_PTVO_INDIR_TT_LINEDISABLE_MSB),
                                    PTVO_CVE_ENTRY(LW_PTVO_INDIR_TT_LINEDISABLE_LSB),
                                    PTVO_CVE_ENTRY(LW_PTVO_INDIR_TT_STARTPOS),
                                    PTVO_CVE_ENTRY(LW_PTVO_INDIR_TT_DATALENGTH),
                                    PTVO_CVE_ENTRY(LW_PTVO_INDIR_TT_FCODE),
                                    PTVO_CVE_ENTRY(LW_PTVO_INDIR_CC_LEVEL),
                                    PTVO_CVE_ENTRY(LW_PTVO_INDIR_CC_CLOCK),
                                    PTVO_CVE_ENTRY(LW_PTVO_INDIR_CC_MISC0),
                                    PTVO_CVE_ENTRY(LW_PTVO_INDIR_CC_MISC1),
                                    PTVO_CVE_ENTRY(LW_PTVO_INDIR_CC_MISC2),
                                    PTVO_CVE_ENTRY(LW_PTVO_INDIR_CC_MISC3),
                                    PTVO_CVE_ENTRY(LW_PTVO_INDIR_CC_DATA0F0),
                                    PTVO_CVE_ENTRY(LW_PTVO_INDIR_CC_DATA1F0),
                                    PTVO_CVE_ENTRY(LW_PTVO_INDIR_CC_DATA0F1),
                                    PTVO_CVE_ENTRY(LW_PTVO_INDIR_CC_DATA1F1)};

    if (!IsLW17orBetter())
    {
        dprintf("lw: Not supported on this chip...\n");
        return;
    }

    dprintf("lw: TV State for crtcOffset 0x%04x:\n", crtcOffset);

    dprintf("LW_PRAMDAC_TVO_SETUP       :6808C0 = 0x%08x\n", GPU_REG_RD32(LW_PRAMDAC_TVO_SETUP + 0x0000));
    dprintf("LW_PRAMDAC_TVO_SETUP       :6828C0 = 0x%08x\n", GPU_REG_RD32(LW_PRAMDAC_TVO_SETUP + 0x2000));
    dprintf("LW_PRAMDAC_TEST_CONTROL    :680608 = 0x%08x\n", GPU_REG_RD32(LW_PRAMDAC_TEST_CONTROL + 0x0000));
    dprintf("LW_PRAMDAC_TEST_CONTROL    :682608 = 0x%08x\n", GPU_REG_RD32(LW_PRAMDAC_TEST_CONTROL + 0x2000));
    dprintf("LW_PRAMDAC_DACCLK          :68052c = 0x%08x\n", GPU_REG_RD32(LW_PRAMDAC_DACCLK + 0x0000));
    dprintf("LW_PRAMDAC_DACCLK          :68252c = 0x%08x\n", GPU_REG_RD32(LW_PRAMDAC_DACCLK + 0x2000));
    dprintf("LW_PRAMDAC_COMPOSITE       :680630 = 0x%08x\n", GPU_REG_RD32(LW_PRAMDAC_COMPOSITE + 0x0000));
    dprintf("LW_PRAMDAC_COMPOSITE       :682630 = 0x%08x\n", GPU_REG_RD32(LW_PRAMDAC_COMPOSITE + 0x2000));

    dprintf("LW_PTVO_CONTROL            :00d200 = 0x%08x\n", GPU_REG_RD32(LW_PTVO_CONTROL + crtcOffset));
    dprintf("LW_PTVO_CONTROL_OUT        :00d204 = 0x%08x\n", GPU_REG_RD32(LW_PTVO_CONTROL_OUT + crtcOffset));
    dprintf("LW_PTVO_OVERSCAN_COMP      :00d208 = 0x%08x\n", GPU_REG_RD32(LW_PTVO_OVERSCAN_COMP + crtcOffset));
    dprintf("LW_PTVO_OVERSCAN_COLOR     :00d20c = 0x%08x\n", GPU_REG_RD32(LW_PTVO_OVERSCAN_COLOR + crtcOffset));
    dprintf("LW_PTVO_HSCALE_STATUS      :00d210 = 0x%08x\n", GPU_REG_RD32(LW_PTVO_HSCALE_STATUS + crtcOffset));
    dprintf("LW_PTVO_VSCALE_STATUS      :00d214 = 0x%08x\n", GPU_REG_RD32(LW_PTVO_VSCALE_STATUS + crtcOffset));
    dprintf("LW_PTVO_HPOS               :00d218 = 0x%08x\n", GPU_REG_RD32(LW_PTVO_HPOS + crtcOffset));
    dprintf("LW_PTVO_VPOS               :00d21c = 0x%08x\n", GPU_REG_RD32(LW_PTVO_VPOS + crtcOffset));
    dprintf("LW_PTVO_ENCODER_STATUS1    :00d228 = 0x%08x\n", GPU_REG_RD32(LW_PTVO_ENCODER_STATUS1  + crtcOffset));
    dprintf("LW_PTVO_ENCODER_STATUS2    :00d22c = 0x%08x\n", GPU_REG_RD32(LW_PTVO_ENCODER_STATUS2 + crtcOffset));
    dprintf("LW_PTVO_ENCODER_STATUS3    :00d230 = 0x%08x\n", GPU_REG_RD32(LW_PTVO_ENCODER_STATUS3 + crtcOffset));
    // LW_PTVO_TELETEXT(i)
    dprintf("LW_PTVO_DEBUG              :00d244 = 0x%08x\n", GPU_REG_RD32(LW_PTVO_DEBUG + crtcOffset));
    dprintf("LW_PTVO_INPUT_COLOR        :00d248 = 0x%08x\n", GPU_REG_RD32(LW_PTVO_INPUT_COLOR + crtcOffset));
    dprintf("LW_PTVO_CRC_CONTROL        :00d24c = 0x%08x\n", GPU_REG_RD32(LW_PTVO_CRC_CONTROL + crtcOffset));
    dprintf("LW_PTVO_INPUT_VSYNC_CRC    :00d250 = 0x%08x\n", GPU_REG_RD32(LW_PTVO_INPUT_VSYNC_CRC + crtcOffset));
    dprintf("LW_PTVO_HSCALE_VSYNC_CRC   :00d254 = 0x%08x\n", GPU_REG_RD32(LW_PTVO_HSCALE_VSYNC_CRC + crtcOffset));
    dprintf("LW_PTVO_VSCALE_VSYNC_CRC   :00d258 = 0x%08x\n", GPU_REG_RD32(LW_PTVO_VSCALE_VSYNC_CRC + crtcOffset));
    dprintf("LW_PTVO_OUTPUT_VSYNC_CRC   :00d25c = 0x%08x\n", GPU_REG_RD32(LW_PTVO_OUTPUT_VSYNC_CRC + crtcOffset));
    dprintf("LW_PTVO_INPUT_TRIGGER_CRC  :00d260 = 0x%08x\n", GPU_REG_RD32(LW_PTVO_INPUT_TRIGGER_CRC + crtcOffset));
    dprintf("LW_PTVO_HSCALE_TRIGGER_CRC :00d264 = 0x%08x\n", GPU_REG_RD32(LW_PTVO_HSCALE_TRIGGER_CRC  + crtcOffset));
    dprintf("LW_PTVO_VSCALE_TRIGGER_CRC :00d268 = 0x%08x\n", GPU_REG_RD32(LW_PTVO_VSCALE_TRIGGER_CRC + crtcOffset));
    dprintf("LW_PTVO_OUTPUT_TRIGGER_CRC :00d26c = 0x%08x\n", GPU_REG_RD32(LW_PTVO_OUTPUT_TRIGGER_CRC  + crtcOffset));
    dprintf("LW_PTVO_INPUT_TRIGGER_PIXELS   :00d270 = 0x%08x\n", GPU_REG_RD32(LW_PTVO_INPUT_TRIGGER_PIXELS + crtcOffset));
    dprintf("LW_PTVO_HSCALE_TRIGGER_PIXELS  :00d274 = 0x%08x\n", GPU_REG_RD32(LW_PTVO_HSCALE_TRIGGER_PIXELS + crtcOffset));
    dprintf("LW_PTVO_VSCALE_TRIGGER_PIXELS  :00d278 = 0x%08x\n", GPU_REG_RD32(LW_PTVO_VSCALE_TRIGGER_PIXELS + crtcOffset));
    dprintf("LW_PTVO_OUTPUT_TRIGGER_PIXELS  :00d27c = 0x%08x\n", GPU_REG_RD32(LW_PTVO_OUTPUT_TRIGGER_PIXELS  + crtcOffset));
    dprintf("LW_PTVO_HPHASE             :00d300 = 0x%08x\n", GPU_REG_RD32(LW_PTVO_HPHASE + crtcOffset));
    dprintf("LW_PTVO_HRES               :00d304 = 0x%08x\n", GPU_REG_RD32(LW_PTVO_HRES + crtcOffset));
    //  LW_PTVO_HFILTER_Y_W0(i)
    // ...
    dprintf("LW_PTVO_HDEBUG             :00d410 = 0x%08x\n", GPU_REG_RD32(LW_PTVO_HDEBUG + crtcOffset));
    dprintf("LW_PTVO_HPHASE_OVERRIDE    :00d414 = 0x%08x\n", GPU_REG_RD32(LW_PTVO_HPHASE_OVERRIDE + crtcOffset));
    dprintf("LW_PTVO_HINCR_OVERRIDE     :00d418 = 0x%08x\n", GPU_REG_RD32(LW_PTVO_HINCR_OVERRIDE + crtcOffset));
    dprintf("LW_PTVO_HSRES_OVERRIDE     :00d41c = 0x%08x\n", GPU_REG_RD32(LW_PTVO_HSRES_OVERRIDE + crtcOffset));
    dprintf("LW_PTVO_VPHASE_FIELD1      :00d500 = 0x%08x\n", GPU_REG_RD32(LW_PTVO_VPHASE_FIELD1 + crtcOffset));
    dprintf("LW_PTVO_VPHASE_FIELD2      :00d504 = 0x%08x\n", GPU_REG_RD32(LW_PTVO_VPHASE_FIELD2 + crtcOffset));
    dprintf("LW_PTVO_VRES               :00d508 = 0x%08x\n", GPU_REG_RD32(LW_PTVO_VRES + crtcOffset));
    // LW_PTVO_VFILTER_YC_W0(i) 
    // ...
    dprintf("LW_PTVO_VDEBUG             :00d600 = 0x%08x\n", GPU_REG_RD32(LW_PTVO_VDEBUG + crtcOffset));
    dprintf("LW_PTVO_VPHASE1_OVERRIDE   :00d604 = 0x%08x\n", GPU_REG_RD32(LW_PTVO_VPHASE1_OVERRIDE + crtcOffset));
    dprintf("LW_PTVO_VPHASE2_OVERRIDE   :00d608 = 0x%08x\n", GPU_REG_RD32(LW_PTVO_VPHASE2_OVERRIDE + crtcOffset));
    dprintf("LW_PTVO_VINCR_OVERRIDE     :00d60c = 0x%08x\n", GPU_REG_RD32(LW_PTVO_VINCR_OVERRIDE + crtcOffset));
    dprintf("LW_PTVO_VSRES_OVERRIDE     :00d610 = 0x%08x\n", GPU_REG_RD32(LW_PTVO_VSRES_OVERRIDE + crtcOffset));
    dprintf("LW_PTVO_VFILTCTRL          :00d614 = 0x%08x\n", GPU_REG_RD32(LW_PTVO_VFILTCTRL + crtcOffset));
    dprintf("LW_PTVO_HFILTCTRL          :00d618 = 0x%08x\n", GPU_REG_RD32(LW_PTVO_HFILTCTRL + crtcOffset));
    dprintf("LW_PTVO_VSTATUS_IPHASE     :00d61c = 0x%08x\n", GPU_REG_RD32(LW_PTVO_VSTATUS_IPHASE + crtcOffset));

    oldCveIndex = GPU_REG_RD32(LW_PTVO_ENCODER_INDEX);
    dprintf("\n");
    dprintf("lw: Zoran Registers:\n");
    dprintf("%-34s %-8s %s\n", "Register Name", "Index", "Value");
    for (i = 0; i < sizeof(ptvoCveRegList) / sizeof(PTVO_CVE_REG); i++)
    {
        GPU_REG_WR32(LW_PTVO_ENCODER_INDEX, ptvoCveRegList[i].idx);
        dprintf("%-34s 0x%02x     0x%02x\n", ptvoCveRegList[i].desc, ptvoCveRegList[i].idx, 
            GPU_REG_RD_DRF(_PTVO, _ENCODER_DATA, _VALUE));
    }
    GPU_REG_WR32(LW_PTVO_ENCODER_INDEX, oldCveIndex);

    oldIndex = GPU_REG_RD08(0x6013d4 + crtcOffset);
    oldLock = UnlockExtendedCRTCs(crtcOffset);

    // Read the actual registers, not the shadows!
    cr3d = REG_RDCR(0x3D, crtcOffset);
    REG_WRCR(0x3D, (U008) (cr3d | 0x1), crtcOffset);

    dprintf("\n");
    dprintf("lw: Cr Registers:\n");
    dprintf("CR 0x44    = 0x%02x\n", REG_RDCR(0x44, crtcOffset));
    dprintf("CR 0x3b    = 0x%02x\n", REG_RDCR(0x3b, crtcOffset));
    dprintf("CR 0x17    = 0x%02x\n", REG_RDCR(0x17, crtcOffset));
    dprintf("CR 0x1a    = 0x%02x\n", REG_RDCR(0x1a, crtcOffset));
    dprintf("CR 0x28    = 0x%02x\n", REG_RDCR(0x28, crtcOffset));
    dprintf("CR 0x33    = 0x%02x\n", REG_RDCR(0x33, crtcOffset));
    dprintf("CR 0x53    = 0x%02x\n", REG_RDCR(0x53, crtcOffset));
    dprintf("CR 0x54    = 0x%02x\n", REG_RDCR(0x54, crtcOffset));
    dprintf("CR 0x59    = 0x%02x\n", REG_RDCR(0x59, crtcOffset));

    REG_WRCR(0x3D, cr3d, crtcOffset);
    RestoreExtendedCRTCs(oldLock, crtcOffset);
    GPU_REG_WR08(0x6013d4 + crtcOffset, oldIndex);
}

//-----------------------------------------------------
// UnlockExtendedCRTCs
// - Unlock the extened CRTC registers
//-----------------------------------------------------
U008 UnlockExtendedCRTCs(U032 crtcOffset)
{
    U008 LwrrentLockState;
    U032 indexFrom;
    U032 indexData;

    // set up the indices
    indexFrom = 0x6013d4 + crtcOffset;
    indexData = 0x6013d5 + crtcOffset;

    // unlock
    GPU_REG_WR08(indexFrom, 0x1f);
    LwrrentLockState = GPU_REG_RD08(indexData);
    GPU_REG_WR08(indexData, 0x57);

    return LwrrentLockState;
}

//-----------------------------------------------------
// RestoreExtendedCRTCs
// - Restore the lock value to the extened CRTC registers
//-----------------------------------------------------
VOID RestoreExtendedCRTCs(U008 crLock, U032 crtcOffset)
{
    U032 indexFrom;
    U032 indexData;

    // set up the indices
    indexFrom = 0x6013d4 + crtcOffset;
    indexData = 0x6013d5 + crtcOffset;

    if(crLock == 3)
    {
        crLock = 0x57;
    }
    else
    {
        crLock = 0x99;
    }

    GPU_REG_WR08(indexFrom, 0x1f);
    GPU_REG_WR08(indexData, crLock);
}

//-----------------------------------------------------
// SetBroadcastBit - No CR Registers changed here!
// - Reads the broadcast state
// - Sets the broadcast state to the value specified in state
// - returns the previous broadcast state
//-----------------------------------------------------
U032 SetBroadcastBit(U032 state)
{
    U032 OldPBusDebug1, NewPBusDebug1;
    OldPBusDebug1 = GPU_REG_RD32(LW_PBUS_DEBUG_1);
    // Clear the old broadcast bit
    NewPBusDebug1 = OldPBusDebug1 & ~(DRF_DEF(_PBUS, _DEBUG_1, _DISP_MIRROR, _ENABLE));
    // Or in the new broadcast state
    NewPBusDebug1 |= DRF_NUM(_PBUS, _DEBUG_1, _DISP_MIRROR, state);

    GPU_REG_WR32(LW_PBUS_DEBUG_1, NewPBusDebug1);

    return DRF_VAL(_PBUS, _DEBUG_1, _DISP_MIRROR, OldPBusDebug1);
}
