/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 1999-2002 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

/***************************************************************************\
*                                                                           *
* Module: i2c.C                                                             *
*   This file implements the user's I2C interface for LWWATCH.              *
*                                                                           *
*   Ryan V. Bissell <rbissell@lwpu.com>  20020915                         *
*                                                                           *
\***************************************************************************/

//
// includes
//
#include "lwwatch.h"
#include "os.h"
#include "lw40/dev_vga.h"
#include "i2c.h"
#include "chip.h"
#include "dac.h"
#include "kepler/gk110/dev_pmgr.h"


#define TERSE   0
#define NORMAL  1
#define PARSING 2

#if LWWATCHCFG_IS_PLATFORM(OSX) || LWWATCHCFG_IS_PLATFORM(UNIX) || defined(USERMODE)
#define I2C_DEBUGOUT(level, params) { if (level <= gVerbosity) dprintf params; }
#else
#define I2C_DEBUGOUT(level, params) { if (level <= gVerbosity) dprintf##params; }
#endif

U008 gPortID = 0;
U008 gHeadID = 0;
U008 gVerbosity = NORMAL;

U032 wMaxWaitStates=3000;  // max wait states for I2C bus syncronisation
ICBENTRY ICBEntry[MAX_ICB_ENTRIES];

U032 getI2CCRTCOffset(void);


void i2cWriteWrIndex
(
    U032 portID,
    U008 data
)
{
    U032 crtcOffset = getI2CCRTCOffset();

    switch( ICBEntry[portID].CR.Access )
    {
        case 0: //LW_ICB20_REC_ACCESS_INDEXED:
            REG_WRCR( ICBEntry[portID].CR.WriteIdx, data, crtcOffset );
            break;
        case 1: // LW_ICB20_REC_ACCESS_DIRECT_IO:
            {
                ULONG Size = 1;

                //osIoWriteByte( pDev, ((U016) ICB20_DIR_WRITEPORT(pDev, portID)), data );
                WriteIoSpace( (U016) (ICBEntry[portID].DI.WriteIO | (ICBEntry[portID].DI.HiIO << 8)),
                     data, &Size);  // Do a byte write
            }
            break;
        case 2: // LW_ICB20_REC_ACCESS_PCI_IO:
            /*
            {
                U032    uI2CPortW;

                dacGetPCIIOMappedI2C( pDev, &uI2CPortW,
                    (U008) ICB20_PCI_BUS(pDev, portID),
                    (U008) ICB20_PCI_DEVICE(pDev, portID),
                    (U008) ICB20_PCI_FUNCTION(pDev, portID),
                    (U008) ICB20_PCI_BAR(pDev, portID),
                    (U008) ICB20_PCI_OFFSET(pDev, portID) );

                switch( ICB20_PCI_TYPE(pDev, portID) )
                {
                    case LW_ICB20_REC_PCI_IO_TYPE_NFORCE_IGP_MCP:
                        uI2CPortW += 1;
                        break;
                    default:
                        DBG_PRINTF(( DBG_MODULE_GLOBAL, DEBUGLEVEL_ERRORS,
                          "LWRM: Unknown PCI IO DDC mapping in i2cWriteWrIndex\n" ));
                        uI2CPortW = 0;
                        break;
                }

                if( uI2CPortW )
                    osIoWriteByte( pDev, uI2CPortW, data );
            }
            */
            break;
        case 3: // LW_ICB20_REC_ACCESS_VIRTUAL:
            //DBG_PRINTF(( DBG_MODULE_GLOBAL, DEBUGLEVEL_ERRORS,
            //  "LWRM: i2cWriteWrIndex: LW_ICB20_REC_ACCESS_VIRTUAL not implemented!\n" ));
            break;
        case 4: // I2C_PORT_ACCESS_UNUSED:
            break;
        default:
            //DBG_PRINTF(( DBG_MODULE_GLOBAL, DEBUGLEVEL_ERRORS,
            //  "LWRM: i2cWriteWrIndex: unknown I2C_PORT_ACCESS method! (%x)\n",
            //  ICB20_ACCESS(pDev, portID) ));
            break;
    }
}


U008 i2cReadWrIndex
(
    U032 portID
)
{
    U008 data = 0xFF;
    U032 crtcOffset = getI2CCRTCOffset();

    switch( ICBEntry[portID].CR.Access )
    {
        case 0: //LW_ICB20_REC_ACCESS_INDEXED:
            // CRTC_RD( ICB20_IDX_WRITEPORT(pDev, portID), data, Head );
            data = REG_RDCR( ICBEntry[portID].CR.WriteIdx, crtcOffset );
            break;
        case 1: // LW_ICB20_REC_ACCESS_DIRECT_IO:
            {
                ULONG Size = 1;

                //data = osIoReadByte( pDev, ((U016) ICB20_DIR_WRITEPORT(pDev, portID)) );
                ReadIoSpace( (U016) (ICBEntry[portID].DI.WriteIO | (ICBEntry[portID].DI.HiIO << 8)),
                     (ULONG *) &data, &Size);  // Do a byte read
            }
            break;
        case 3: // LW_ICB20_REC_ACCESS_PCI_IO:
            /*
            {
                U032    uI2CPortW;

                dacGetPCIIOMappedI2C( pDev, &uI2CPortW,
                    (U008) ICB20_PCI_BUS(pDev, portID),
                    (U008) ICB20_PCI_DEVICE(pDev, portID),
                    (U008) ICB20_PCI_FUNCTION(pDev, portID),
                    (U008) ICB20_PCI_BAR(pDev, portID),
                    (U008) ICB20_PCI_OFFSET(pDev, portID) );

                switch( ICB20_PCI_TYPE(pDev, portID) )
                {
                    case LW_ICB20_REC_PCI_IO_TYPE_NFORCE_IGP_MCP:
                        uI2CPortW += 1;
                        break;
                    default:
                        DBG_PRINTF(( DBG_MODULE_GLOBAL, DEBUGLEVEL_ERRORS,
                          "LWRM: Unknown PCI IO DDC mapping in i2cReadWrIndex\n" ));
                        uI2CPortW = 0;
                        break;
                }

                if( uI2CPortW )
                    data = osIoReadByte( pDev, uI2CPortW );
            }
            */
            break;
        case 2: // LW_ICB20_REC_ACCESS_VIRTUAL:
            //DBG_PRINTF(( DBG_MODULE_GLOBAL, DEBUGLEVEL_ERRORS,
            //  "LWRM: i2cReadWrIndex: LW_ICB20_REC_ACCESS_VIRTUAL not implemented!\n" ));
            break;
        case 7: // I2C_PORT_ACCESS_UNUSED:
            break;
        default:
            //DBG_PRINTF(( DBG_MODULE_GLOBAL, DEBUGLEVEL_ERRORS,
            //  "LWRM: i2cReadWrIndex: unknown I2C_PORT_ACCESS method! (%x)\n",
            //  ICB20_ACCESS(pDev, portID) ));
            break;
    }

    return data;
}


U008 i2cReadStatusIndex
(
    U032 portID
)
{
    U008 data = 0xFF;
    U032 crtcOffset = getI2CCRTCOffset();

    switch( ICBEntry[portID].CR.Access )
    {
        case 0: //LW_ICB20_REC_ACCESS_INDEXED:
            //CRTC_RD( ICB20_IDX_READPORT(pDev, portID), data, Head );
            data = REG_RDCR( ICBEntry[portID].CR.ReadIdx, crtcOffset );
            break;
        case 1: // LW_ICB20_REC_ACCESS_DIRECT_IO:
            {
                ULONG Size = 1;

                // data = osIoReadByte( pDev, ((U016) ICB20_DIR_READPORT(pDev, portID)) );
                ReadIoSpace( (U016) (ICBEntry[portID].DI.ReadIO | (ICBEntry[portID].DI.HiIO << 8)),
                     (ULONG *) &data, &Size);  // Do a byte read
            }
            break;
        case 3: // LW_ICB20_REC_ACCESS_PCI_IO:
            /*
            {
                U032    uI2CPortR;

                dacGetPCIIOMappedI2C( pDev, &uI2CPortR,
                    (U008) ICB20_PCI_BUS(pDev, portID),
                    (U008) ICB20_PCI_DEVICE(pDev, portID),
                    (U008) ICB20_PCI_FUNCTION(pDev, portID),
                    (U008) ICB20_PCI_BAR(pDev, portID),
                    (U008) ICB20_PCI_OFFSET(pDev, portID) );

                switch( ICB20_PCI_TYPE(pDev, portID) )
                {
                    case LW_ICB20_REC_PCI_IO_TYPE_NFORCE_IGP_MCP:
                        break;
                    default:
                        DBG_PRINTF(( DBG_MODULE_GLOBAL, DEBUGLEVEL_ERRORS,
                          "LWRM: Unknown PCI IO DDC mapping in i2cReadStatusIndex\n" ));
                        uI2CPortR = 0;
                        break;
                }

                if( uI2CPortR )
                    data = osIoReadByte( pDev, uI2CPortR );
            }
            */
            break;
        case 2: // LW_ICB20_REC_ACCESS_VIRTUAL:
            //DBG_PRINTF(( DBG_MODULE_GLOBAL, DEBUGLEVEL_ERRORS,
            //  "LWRM: i2cReadStatusIndex: LW_ICB20_REC_ACCESS_VIRTUAL not implemented!\n" ));
            break;
        case 7: // I2C_PORT_ACCESS_UNUSED:
            break;
        default:
            //DBG_PRINTF(( DBG_MODULE_GLOBAL, DEBUGLEVEL_ERRORS,
            //  "LWRM: i2cReadStatusIndex: unknown I2C_PORT_ACCESS method! (%x)\n",
            //  ICB20_ACCESS(pDev, portID) ));
            break;
    }

    return data;
}

U008 i2cHardwareInit
(
    U032 portID
)
{
    // On dual-headed devices, enable I2C interface
    //if (IsMultiHead(pDev))
    //{
    //    AssocDDC(pDev, Head);
    //}

    //Refresh our view of how to update I2C, if necessary
    //For PCI IO mapped (from the DCB) update the access method to DIRECT_IO with current mappings
    /*
    if( DRF_VAL(_ICB20, _REC, _ACCESS,  (pDev->Dac.PCIIOBackupICB[portID])) == LW_ICB20_REC_ACCESS_PCI_IO )
    {
        U032    uI2CPortBase = 0;
        U032    uI2CPortR = 0;
        U032    uI2CPortW = 0;

        //Get the current I2C base port, based on the DCB PCI IO mapping info
        dacGetPCIIOMappedI2C( pDev, &uI2CPortBase,
            (U008) DRF_VAL(_ICB20, _REC, _PCI_IO_BUS,      pDev->Dac.PCIIOBackupICB[portID]),
            (U008) DRF_VAL(_ICB20, _REC, _PCI_IO_DEVICE,   pDev->Dac.PCIIOBackupICB[portID]),
            (U008) DRF_VAL(_ICB20, _REC, _PCI_IO_FUNCTION, pDev->Dac.PCIIOBackupICB[portID]),
            (U008) DRF_VAL(_ICB20, _REC, _PCI_IO_BAR,      pDev->Dac.PCIIOBackupICB[portID]),
            (U008) DRF_VAL(_ICB20, _REC, _PCI_IO_OFFSET,   pDev->Dac.PCIIOBackupICB[portID]) );

        switch( DRF_VAL(_ICB20, _REC, _PCI_IO_TYPE, pDev->Dac.PCIIOBackupICB[portID]) )
        {
            case LW_ICB20_REC_PCI_IO_TYPE_NFORCE_IGP_MCP:
                uI2CPortR = uI2CPortBase;
                uI2CPortW = uI2CPortBase + 1;
                break;
            default:
                DBG_PRINTF(( DBG_MODULE_GLOBAL, DEBUGLEVEL_ERRORS,
                  "LWRM: Unknown PCI IO DDC mapping in i2cHardwareInit\n" ));
                uI2CPortR = 0;
                uI2CPortW = 0;
                break;
        }

        //After colwerting the FromDCB PCI IO mapping to a direct IO mapping, stash it in the "active" I2CInfo
        if((uI2CPortBase!=0) && (uI2CPortR!=0) && (uI2CPortW!=0))
        {
            // Initialize the Port to 0 to start, then OR in the details
            pDev->Dac.VBiosICB[portID] = 0;
            pDev->Dac.VBiosICB[portID] |= DRF_DEF(_ICB20, _REC, _ACCESS, _DIRECT_IO);
            pDev->Dac.VBiosICB[portID] |= DRF_NUM(_ICB20, _REC, _DIR_IO_WRITE_PORT, (uI2CPortW & 0xFF));
            pDev->Dac.VBiosICB[portID] |= DRF_NUM(_ICB20, _REC, _DIR_IO_READ_PORT,  (uI2CPortR & 0xFF));
            pDev->Dac.VBiosICB[portID] |= DRF_NUM(_ICB20, _REC, _DIR_IO_HI_PORT,    ((uI2CPortW >> 8) & 0xFF));
        }
    }
    */

    //
    // If the initialization has already been done then just return
    //
    // if (iniFlag == TRUE)
    //    return TRUE;

    i2cWriteWrIndex(portID, 0x31);

    return TRUE;
}


VOID i2cWriteCtrl
(
    U032 portID,
    U008 reg,
    U008 bit
)
{
    U008 data;

    //
    // Get the current status and toggle
    //
    data = i2cReadWrIndex(portID);

    data &= 0xf0;
    data |= I2C_ENABLE;

    if (reg == SCL_REG)
    {
      if (bit)
        data |=  I2C_SRCK;
      else
        data &= ~I2C_SRCK;
    }
    else
    {
      if (bit)
        data |=  I2C_SRD;
      else
        data &= ~I2C_SRD;
    }

    i2cWriteWrIndex(portID, data);
    //FlushWB();
}

U008 i2cReadCtrl
(
    U032 portID,
    U008 reg
)
{
    U008 data;

    data = i2cReadStatusIndex(portID);

    if (reg == SCL_REG)
        return ( (data & I2C_SRCK_IN) != 0);
    else
        return ( (data & I2C_SRD_IN) != 0);
}

/**********************************************************************/

VOID ReadSDA(U032 portID, U008 *data)
{
    *data = i2cReadCtrl(portID, SDA_REG);
}

VOID ReadSCL(U032 portID, U008 *data)
{
    *data = i2cReadCtrl(portID, SCL_REG);
}

VOID SetSCLLine(U032 portID)
{
    i2cWriteCtrl(portID, SCL_REG, 1);
}

VOID ResetSCLLine(U032 portID)
{
    i2cWriteCtrl(portID, SCL_REG, 0);
}

VOID SetSDALine(U032 portID)
{
    i2cWriteCtrl(portID, SDA_REG, 1);
}

VOID ResetSDALine(U032 portID)
{
    i2cWriteCtrl(portID, SDA_REG, 0);
}

/*
 * waits for a specified line til it goes high
 * giving up after MAX_WAIT_STATES attempts
 * return:  0 OK
 *         -1 fail (time out)
 */
U008 WaitHighSDALine(U032 portID)
{
    U008    data_in;
    U032    retries = wMaxWaitStates;

    do
    {
        ReadSDA(portID, &data_in);      // wait for the line going high
        if (data_in)
            break;
        osPerfDelay(I2CDELAY);
    } while (--retries);        // count down is running

    if (!retries)
        return((U008)-1);
    return 0;
}

U008 WaitHighSCLLine(U032 portID)
{
    U008    data_in;
    U032    retries = wMaxWaitStates;

    do
    {
        osPerfDelay(1);              // 1.0 us delay   NEEDED??
        ReadSCL(portID, (U008 *)&data_in);   // wait for the line going high
        if (data_in)
            break;
    } while (--retries);            // count down is running

    if (!retries)
    {
        return((U008)-1);
    }

    return(0);
}


VOID i2cStart(U032 portID)
{
    SetSDALine(portID);
    osPerfDelay(I2CDELAY);
    SetSCLLine(portID);
    osPerfDelay(I2CDELAY);      // spec requires clock to be high min of 4us
    WaitHighSCLLine(portID);
    ResetSDALine(portID);
    osPerfDelay(I2CDELAY);
    ResetSCLLine(portID);
#if (0)
    I2C_DEBUGOUT(NORMAL, ("\nS"));
#endif
}


VOID i2cStop(U032 portID)
{

    osPerfDelay(I2CDELAY * 20);
    ResetSCLLine(portID);
    ResetSDALine(portID);
    osPerfDelay(I2CDELAY);
    SetSCLLine(portID);
    osPerfDelay(I2CDELAY);      // spec requires clock to be high min of 4us
    WaitHighSCLLine(portID);
    SetSDALine(portID);
    osPerfDelay(I2CDELAY);
#if (0)
    I2C_DEBUGOUT(NORMAL, ("P\n"));
#endif
}


/*
 * I2cAck() returns 1: fail
 *                  0: acknolege
 */

U008 i2cAck(U032 portID)
{
    U008 ack;

    ResetSCLLine(portID);
    osPerfDelay(I2CDELAY);
    SetSDALine(portID);
    osPerfDelay(I2CDELAY);
    SetSCLLine(portID);
    osPerfDelay(I2CDELAY);     // spec requires clock to be high min of 4us
    WaitHighSCLLine(portID);
    ReadSDA(portID, &ack);
    ResetSCLLine(portID);
#if (0)
    I2C_DEBUGOUT(NORMAL, ("%s", ack ? "N " : "A "));
#endif
    return (ack);
}


VOID i2cInit(U032 portID)
{
    SetSCLLine(portID);
    osPerfDelay(I2CDELAY);     // spec requires clock to be high min of 4us
    WaitHighSCLLine(portID);
    SetSDALine(portID);
}


U008 i2cSendByte(U032 portID, U008 byte)
{
    U008 i;
    U008 save = byte;

    for (i=0;i<8;i++)
    {
        ResetSCLLine(portID);
        osPerfDelay(I2CDELAY/2);
        if (byte & 0x80)
            SetSDALine(portID);
        else
            ResetSDALine(portID);
        osPerfDelay(I2CDELAY/2);
        SetSCLLine(portID);
        osPerfDelay(I2CDELAY);    // clock must be high at least 4us
        WaitHighSCLLine(portID);
#if (0)
        I2C_DEBUGOUT(NORMAL, ("%s", (byte & 0x80) ? "1" : "0"));
#endif
        byte <<= 1;
    }
#if (0)
    I2C_DEBUGOUT(NORMAL, ("(%02x)", save));
#endif

    i = i2cAck(portID);

    return (i);
}


U032 i2cReceiveByte(U032 portID, U008 *byte, U008 nack)
{
    U008 data=0;
    U008 i;
    U032 status;

    ResetSCLLine(portID);
    SetSDALine(portID);
    osPerfDelay(1);

    *byte = 0;
    for (i=0;i<8;i++)
    {
        ResetSCLLine(portID);
        ResetSCLLine(portID);  // 2nd needed?
        osPerfDelay(I2CDELAY_CLK_LOW);           // clock must be low at least 4.7 us
        SetSCLLine(portID);
        status = WaitHighSCLLine(portID) ? RM_ERROR : RM_OK;
        if (status != RM_OK)
            goto done;
        osPerfDelay(I2CDELAY_CLK_HIGH);          // clock must be high at least 4us

        ReadSDA(portID, &data);
#if (0)
        I2C_DEBUGOUT(NORMAL, ("%s", (data) ? "1" : "0"));
#endif
        *byte <<= 1;
        *byte  |= (data == 1);
    }
#if (0)
    I2C_DEBUGOUT(NORMAL, ("(%02x)", *byte));
#endif

    ResetSCLLine(portID);
    if (nack)
    {
        SetSDALine(portID);         // send Nack
    }
    else
        ResetSDALine(portID);       // send Ack
#if (0)
    I2C_DEBUGOUT(NORMAL, ("%s", nack ? "N " : "A "));
#endif
    osPerfDelay(I2CDELAY_CLK_LOW);           // clock must be low at least 4.7 us
    osPerfDelay(MACI2CDELAY);
    SetSCLLine(portID);
    status = WaitHighSCLLine(portID) ? RM_ERROR : RM_OK;
    osPerfDelay(I2CDELAY_CLK_HIGH);          // clock must be high at least 4us
    osPerfDelay(MACI2CDELAY);
    ResetSCLLine(portID);
    osPerfDelay(I2CDELAY_CLK_LOW);           // clock must be low at least 4.7 us
    osPerfDelay(MACI2CDELAY);

done:

    return status;
}


U032 i2cWrite(U032 portID, U008 ChipAdr, U016 AdrLen, U008 *Adr, U016 DataLen, U008 *Data)
{
    //
    // Enable writes to the I2C port
    //
    i2cHardwareInit(portID);

    i2cStart(portID);
    if ( i2cSendByte(portID, (U008)(ChipAdr<<1)) ) // send chip adr. with write bit
    {
        i2cStop(portID);                         // ack failed --> generate stop condition
        return 0xFFFFFFFF;
    }
    for ( ; AdrLen; AdrLen--)
    {
        if ( i2cSendByte(portID, *Adr++) )        // send sub-register byte(s)
        {
            i2cStop(portID);                    // ack failed --> generate stop condition
            return 0xFFFFFFFF;
        }
    }
    for ( ; DataLen; DataLen--)            // send data byte(s)
    {
        if ( i2cSendByte(portID, *Data++) )
        {
            i2cStop(portID);                     // ack failed --> generate stop condition
            return 0xFFFFFFFF;
        }
    }
    i2cStop(portID);
    return 0;
}


U032 i2cRead(U032 portID, U008 ChipAdr, U016 AdrLen, U008 *Adr, U016 DataLen, U008 *Data)
{
    U008 dat;
    U032 status = RM_ERROR;        // pessimist

    //DBG_PRINT_STRING_VALUE(DEBUGLEVEL_TRACEINFO, "LWRM: ChipAdr ", (U032)ChipAdr);
    //DBG_PRINT_STRING_VALUE(DEBUGLEVEL_TRACEINFO, "LWRM: AdrLen ", (U032)AdrLen);
    //DBG_PRINT_STRING_VALUE(DEBUGLEVEL_TRACEINFO, "LWRM: Adr ", (U032)*Adr);
    //DBG_PRINT_STRING_VALUE(DEBUGLEVEL_TRACEINFO, "LWRM: DataLen ", (U032)DataLen);

    //
    // Enable writes to the I2C port
    //
    i2cHardwareInit(portID);

    i2cStart(portID);
    i2cSendByte(portID, (U008)(ChipAdr<<1));        // send chip adr. with write bit

    for ( ; AdrLen; AdrLen--)               // send sub-register address byte(s)
    {
        if ( i2cSendByte(portID, *Adr++) )
        {
            goto done;
        }
    }

    osPerfDelay(I2CDELAY);    // give the device some time to parse the subaddress

    i2cStart(portID);                             // send again chip address for switching to read mode
    if ( i2cSendByte(portID, (U008)((ChipAdr<<1) | 1)) )  // send chip adr. with read bit
    {
        goto done;
    }

    for (status = RM_OK; DataLen && (status == RM_OK); DataLen--)
    {
        status = i2cReceiveByte(portID,
                                (U008 *)&dat,
                                (U008)((DataLen == 1) ? NACK : ACK));         // receive byte(s)
        *Data++ = dat;
    }

done:
    i2cStop(portID);
    //DBG_PRINT_STRING_VALUE(DEBUGLEVEL_TRACEINFO, "LWRM: Data ", (U032)*Data);

    return status;
}

U032 i2cRead_EDDC(U032 portID, U008 SegmentAddr, U008 ChipAdr, U008 SubByteAddr, U016 DataLen, U008 *Data)
{
    //extended DDC compatibility not confirmed on date modified. No monitors with edids greater than 256 are
    //easily obtainable or locatable.

    U008 dat;
    U032 status = 0xFFFFFFFF;        // pessimist

    //
    // Enable writes to the I2C port
    //
    i2cHardwareInit(portID);

    //if segment!=0, set the segment with this sequence first
    if(SegmentAddr)
    {
        //send start
        i2cStart(portID);

        //send segment register addr
        i2cSendByte(portID, 0x60);

        //send the segment number
        i2cSendByte(portID, SegmentAddr);
    }

    i2cStart(portID);
    i2cSendByte(portID, (U008)(ChipAdr<<1));        // send chip adr. with write bit
    i2cSendByte(portID, (U008)SubByteAddr);

    i2cStart(portID);                             // send again chip address for switching to read mode
    if ( i2cSendByte(portID, (U008)((ChipAdr<<1) | 1)) )  // send chip adr. with read bit
    {
        goto done;
    }

    for (status = RM_OK; DataLen && (status == RM_OK); DataLen--)
    {
        status = i2cReceiveByte(portID,
                                (U008 *)&dat,
                                (U008)((DataLen == 1) ? NACK : ACK));         // receive byte(s)
        *Data++ = dat;
    }

done:
    i2cStop(portID);
    //DBG_PRINT_STRING_VALUE(DEBUGLEVEL_TRACEINFO, "LWRM: Data ", (U032)*Data);

    return status;
}

U032 i2cSend(U032 portID, U008 ChipAdr, U016 AdrLen, U008 *Adr, U016 DataLen, U008 *Data, U032 NoStopFlag)
{

    if ( ChipAdr ) {
        //
        // Enable writes to the I2C port
        //
        i2cHardwareInit(portID);

        i2cStart(portID);
        if ( i2cSendByte(portID, (U008)(ChipAdr<<1)) ) // send chip adr. with write bit
        {
            i2cStop(portID);                         // ack failed --> generate stop condition
            return 0xFFFFFFFF;
        }
    }

    for ( ; AdrLen; AdrLen--)
    {
        if ( i2cSendByte(portID, *Adr++) )        // send sub-register byte(s)
        {
            i2cStop(portID);                    // ack failed --> generate stop condition
            return 0xFFFFFFFF;
        }
    }

    for ( ; DataLen; DataLen--)            // send data byte(s)
    {
        if ( i2cSendByte(portID, *Data++) )
        {
            i2cStop(portID);                     // ack failed --> generate stop condition
            return 0xFFFFFFFF;
        }
    }

    if ( NoStopFlag == 0 )
        i2cStop(portID);

    return 0;
}


U032 i2cWrite_ALT(U032 portID, U008 ChipAdr, U016 AdrLen, U008 *Adr, U016 DataLen, U008 *Data)
{
    //
    // Enable writes to the I2C port
    //
    i2cHardwareInit(portID);

    i2cStart(portID);
    if ( i2cSendByte(portID, (U008)(ChipAdr<<1)) ) // send chip adr. with write bit
    {
        i2cStop(portID);                         // ack failed --> generate stop condition
        return 0xFFFFFFFF;
    }
    for ( ; DataLen; DataLen--)            // send data byte(s)
    {
        if ( i2cSendByte(portID, *Data++) )
        {
            i2cStop(portID);                     // ack failed --> generate stop condition
            return 0xFFFFFFFF;
        }
    }
    i2cStop(portID);
    return 0;
}


U032 i2cRead_ALT(U032 portID, U008 ChipAdr, U016 AdrLen, U008 *Adr, U016 DataLen, U008 *Data)
{
    U008 dat;

    //DBG_PRINT_STRING_VALUE(DEBUGLEVEL_TRACEINFO, "LWRM: ChipAdr ", (U032)ChipAdr);
    //DBG_PRINT_STRING_VALUE(DEBUGLEVEL_TRACEINFO, "LWRM: AdrLen ", (U032)AdrLen);
    //DBG_PRINT_STRING_VALUE(DEBUGLEVEL_TRACEINFO, "LWRM: Adr ", (U032)*Adr);
    //DBG_PRINT_STRING_VALUE(DEBUGLEVEL_TRACEINFO, "LWRM: DataLen ", (U032)DataLen);

    //
    // Enable writes to the I2C port
    //
    i2cHardwareInit(portID);

    i2cStart(portID);
    i2cSendByte(portID, (U008)((ChipAdr<<1) | 1));        // send chip adr. with write bit
    for ( ; DataLen ; DataLen--)
    {
        i2cReceiveByte(portID, (U008 *)&dat, (U008)((DataLen == 1) ? NACK : ACK));         // receive byte(s)
        *Data++ = dat;
    }
    i2cStop(portID);

    //DBG_PRINT_STRING_VALUE(DEBUGLEVEL_TRACEINFO, "LWRM: Data ", (U032)*Data);

    return 0;
}


U032 i2cRead_ALT2(U032 portID, U008 ChipAdr, U016 AdrLen, U008 *Adr, U016 DataLen, U008 *Data)
{
    U008 dat;

    //DBG_PRINT_STRING_VALUE(DEBUGLEVEL_TRACEINFO, "LWRM: ChipAdr ", (U032)ChipAdr);
    //DBG_PRINT_STRING_VALUE(DEBUGLEVEL_TRACEINFO, "LWRM: AdrLen ", (U032)AdrLen);
    //DBG_PRINT_STRING_VALUE(DEBUGLEVEL_TRACEINFO, "LWRM: Adr ", (U032)*Adr);
    //DBG_PRINT_STRING_VALUE(DEBUGLEVEL_TRACEINFO, "LWRM: DataLen ", (U032)DataLen);

    //
    // Enable writes to the I2C port
    //
    i2cHardwareInit(portID);

    i2cStart(portID);
    i2cSendByte(portID, (U008)(ChipAdr<<1));        // send chip adr. with write bit

    for ( ; AdrLen; AdrLen--)               // send sub-register address byte(s)
    {
        if ( i2cSendByte(portID, *Adr++) )
        {
            i2cStop(portID);                      // ack failed --> generate stop condition
            return 0xFFFFFFFF;
        }
    }
    i2cStop(portID);

    i2cStart(portID);                             // send again chip address for switching to read mode
    if ( i2cSendByte(portID, (U008)(( ChipAdr<<1) | 1)) )  // send chip adr. with read bit
    {
        i2cStop(portID);                         // ack failed --> generate stop condition
        return 0xFFFFFFFF;
    }

    for ( ; DataLen ; DataLen--)
    {
        i2cReceiveByte(portID, (U008 *)&dat, (U008)((DataLen == 1) ? NACK : ACK));         // receive byte(s)
        *Data++ = dat;
    }

    i2cStop(portID);

    //DBG_PRINT_STRING_VALUE(DEBUGLEVEL_TRACEINFO, "LWRM: Data ", (U032)*Data);

    return 0;
}

static U008 i2cReadDevice(U008 addr, U008* read, U016 count)
{
  U032 i;
  U008 nack;

  if (!read) return 0;

  *read = 0;

  i2cInit(gPortID);
  i2cStart(gPortID);

  nack  =   i2cSendByte(gPortID, (U008)(addr | I2C_READCYCLE));

  if(!nack)
  {
    for (i=0; i<count; i++)
    {
      dprintf("%sR", i%8 ? " " : "\n");
      nack = !!i2cReceiveByte(gPortID, (U008*)(&read[i]), (U008)(i<(U032)(count-1) ? 0 : 1));
    }
  }

  i2cStop(gPortID);

  dprintf("\n");

  return !nack;
}

static U008 i2cReadRegisterEx(U008 addr, U008 sublen, U008* subaddr, U008* read, U016 count)
{
  U032 i;
  U008 nack;

  if (!read) return 0;

  i2cInit(gPortID);
  i2cStart(gPortID);
  nack  = i2cSendByte(gPortID, (U008)addr);

  for(i=0; !nack && i<sublen; i++)
      nack |= i2cSendByte(gPortID, (U008)subaddr[i]);

  if (!nack)
  {
    i2cStart(gPortID);
    nack  =   i2cSendByte(gPortID, (U008)(addr | I2C_READCYCLE));

    for (i=0; i<count; i++)
    {
      dprintf("%sR", i%8 ? "" : "\n");
      nack |= !!i2cReceiveByte(gPortID, (U008*)(&read[i]), (U008)(i<(U032)(count-1) ? 0 : 1));
    }
  }

  i2cStop(gPortID);

  dprintf("\n");

  return !nack;
}

static U008 i2cReadRegister(U008 addr, U008 subaddr, U008* read, U016 count)
{
  U032 i;
  U008 nack;

  if (!read) return 0;

  *read = 0;
  i2cInit(gPortID);
  i2cStart(gPortID);
  nack  = i2cSendByte(gPortID, (U008)addr);
  if (!nack)
      nack = i2cSendByte(gPortID, (U008)subaddr);

  if (!nack)
  {
    i2cStart(gPortID);
    nack  =   i2cSendByte(gPortID, (U008)(addr | I2C_READCYCLE));

    for (i=0; i<count; i++)
    {
      dprintf("%sR", i%8 ? "" : "\n");
      nack |= !!i2cReceiveByte(gPortID, (U008*)(&read[i]), (U008)(i<(U032)(count-1) ? 0 : 1));
    }
  }

  i2cStop(gPortID);

  dprintf("\n");

  return !nack;
}

static U008 i2cWriteDevice(U008 addr, U008* data, U008 count)
{
  U016 i;
  U008 nack;

  i2cInit(gPortID);
  i2cStart(gPortID);
  nack  = i2cSendByte(gPortID, (U008)addr);


  for (i=0; !nack && i<count; i++)
  {
    dprintf("%sW", i%8 ? "" : "\n");
    nack |= i2cSendByte(gPortID, (U008)data[i]);
  }

  i2cStop(gPortID);

  dprintf("\n");

  return !nack;
}


#if 0
static U008 i2cWriteRegister(U008 addr, U008 subaddr, U008 data)
{
  U008 nack;

  i2cInit(gPortID);
  i2cStart(gPortID);
  nack  = i2cSendByte(gPortID, (U008)addr);
  nack |= i2cSendByte(gPortID, (U008)subaddr);
  nack |= i2cSendByte(gPortID, (U008)data);

  i2cStop(gPortID);
  return !nack;
}
#endif


//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


U032 getI2CCRTCOffset()
{
    U032 offset = 0;
    U032 i;
    U032 EngineCtrl;

    for(i=0; i<GetNumCrtcs(); i++)
    {
        if(i==0)
          offset = 0;
        else if(i==1)
          offset = 0x2000;

        EngineCtrl = GPU_REG_RD32(LW_PCRTC_ENGINE_CTRL + offset);

        // Does this Head own I2C port?
        if(EngineCtrl & BIT(4))
          break;

        // Reset the offset back to 0
        // since if neither bit is set, I2C is owned by Head A.
        offset = 0;
    }

    return offset;
}

U008 hex2byte(char *input, U008* value, char* delims, char** stop)
{
    U008 i, temp;

    if (stop) *stop = (char*)NULL;

    skipDelims(&input, delims);

    *value = 0;
    for (i=2; i && *input && !isDelim(*input, delims); i--, input++)
    {
        *value *= 16;
        temp = TOUPPER(*input);

        if ((temp >= '0') && (temp <= '9'))
            *value += (U008)(temp - '0');
        else if ((temp >= 'A') && (temp <= 'F'))
            *value += (U008)(temp - 'A' +  10);
        else
            if (!isDelim(*input, delims))
            {
                if (stop) *stop = input;
                return 0;
            }
    }

    if (stop) *stop = input;

    if (*input && !isDelim(*input, delims))
        return 0;

    //do NOT skip past terminating delims!
    //reason:  some callers inspect the terminator to decide what comes next.
    //skipDelims(&input, delims);

    if (stop) *stop = input;
    return 1;
}


U008 dec2word(char *input, U016* value, char* delims, char** stop)
{
    U008 i, temp;

    if (stop) *stop = (char*)NULL;

    skipDelims(&input, delims);

    *value = 0;
    for (i=5; i && *input && !isDelim(*input, delims); i--, input++)
    {
        *value *= 10;
        temp = TOUPPER(*input);

        if ((temp >= '0') && (temp <= '9'))
            *value += (U008)(temp - '0');
        else
            if (!isDelim(*input, delims))
            {
                if (stop) *stop = input;
                return 0;
            }
    }

    if (stop) *stop = input;

    if (*input && !isDelim(*input, delims))
        return 0;

    //do NOT skip past terminating delims!
    //reason:  some callers inspect the terminator to decide what comes next.
    //skipDelims(&input, delims);

    if (stop) *stop = input;
    return 1;
}


void ChangePortID(char* input1023)
{
    U008 port;
    char* input;

    input = input1023;

    if (!isDelim(*input, GENERIC_DELIMS))
    {
        dprintf("*** invalid format for 'change <p>ort ID'.\n");
        dprintf("    format is: 'p <portID>' where <portID> is 0-F (hex).");
        return;
    }

    skipDelims(&input, GENERIC_DELIMS);

    port = (U008)(*input - '0');
    if (port > 0x0F)
    {
        dprintf("*** invalid format for 'change <p>ort ID'.\n");
        dprintf("    format is: 'p <portID>' where <portID> is 0-F (hex).");
        return;
    }

    dprintf("Current I2C port has been changed to %u.\n", port);
    gPortID = port;
}


void ChangeHeadID(char* input1023)
{
    U008 head;
    char* input;

    input = input1023;

    if (!isDelim(*input, GENERIC_DELIMS))
    {
        dprintf("*** invalid format for 'change <h>ead ID'.\n");
        dprintf("    format is: 'h <headID>' where <headID> is 0-F (hex).\n");
        return;
    }

    skipDelims(&input, GENERIC_DELIMS);

    head = (U008)(*input - '0');
    if (head > 0x0F)
    {
        dprintf("*** invalid format for 'change <p>ort ID'.\n");
        dprintf("    format is: 'p <portID>' where <portID> is 0-F (hex).\n");
        return;
    }

    dprintf("Current head has been changed to %u.\n", head);
    gHeadID = head;
}


void DumpI2C(U008* data, U008 numsubs, U008* subaddr, U016 count, U008 skip2)
{
    U032 i,j;
#if (0)
    I2C_DEBUGOUT(PARSING, ("PARSE: skip2='%u'\n", skip2));
    I2C_DEBUGOUT(PARSING, ("PARSE: numsubs='%u'\n", numsubs));
#endif
    dprintf("\n");
    dprintf("     ");
    for (i=0; i<16; i+=(1+!!skip2))
        dprintf("+%X%s", i+(!!skip2*(subaddr[0]%2)), i+(1+!!skip2)<16 ? ", " : "\n");
    dprintf("     ~~~");
    for (i=(60>>!!skip2); i; i--)
        dprintf("%s", i>1 ? "~" : "\n");

/*
    dprintf("     +0, +1, +2, +3, +4, +5, +6, +7, +8, +9, +A, +B, +C, +D, +E, +F\n");
    dprintf("     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n");
*/

    if (numsubs > 1)
    {
        //because it's //real// hard to show >2 dimensions with text output...
        dprintf("%s%02X  |", (i ? "\n" : ""), ((i+subaddr[0])/16)*16);

        for (j=(subaddr[0]%16)>>skip2; j; j--)
            dprintf("  %s", j==1 ? "**" : "  ");

        dprintf("%02X**\n\n\n", subaddr[1]);

        //a little relwrsion never hurt anyone
        DumpI2C(data, numsubs-1, &subaddr[1], count, skip2);
        return;
    }


    for (i=0; i<count; i++)
    {
        if (!i || !(((i*(1+!!skip2)+subaddr[0]) % 16>>!!skip2)))
            dprintf("%s%02X  |", (i ? "\n" : ""), ((i*(1+!!skip2)+subaddr[0])/16)*16);

        if (i == 0)
        {
            for (j=(subaddr[0]%16)>>skip2; j; j--)
                dprintf("    ");
        }


      //dprintf("%02X%s", data[i], ((i+subaddr[0]+1) % (16>>skip2)) ? ", " : "");
        dprintf("%02X, ", data[i]);
    }
}


U008 ReadRegisterI2C(U008 addr, U008 subaddr, U008* data64K, U008 count)
{
    dprintf("*** reading (decimal) %u bytes from device %X, starting from subaddress %X.\n", (U032)count, (U032)addr, (U032)subaddr);
    return i2cReadRegister(addr, subaddr, data64K, count);
}

U008 ReadRegisterExI2C(U008 addr, U008 sublen, U008* subaddr, U008* data64K, U016 count)
{
    dprintf("*** reading (decimal) %u bytes from device %X, starting from subaddress %X (depth>1).\n", (U032)count, (U032)addr, (U032)subaddr[0]);
    return i2cReadRegisterEx(addr, sublen, subaddr, data64K, count);
}

U008 ReadDeviceI2C(U008 addr, U008* data64K, U016 count)
{
    dprintf("*** reading (decimal) %u bytes from device %X.\n", count, addr);
    return i2cReadDevice(addr, data64K, count);
}


U008 WriteDeviceI2C(U008 addr, U008* data256, U008 count)
{
    dprintf("*** sending (decimal) %u bytes to device %X.\n", count, addr);
    return i2cWriteDevice(addr, data256, count);
}

void SyntaxError(char* input1024, char* stop, char* error)
{
    U032 i,j;
    LwU64 offset;

    dprintf("\n\n\n");
    dprintf("*** SYNTAX ERROR: %s\n", error);
    dprintf("    %s\n", input1024);

    offset = (stop - input1024);
    for (j=3; j; j--)
    {
        dprintf("    ");
        for (i=0; i<offset; i++)
            dprintf(" ");

        dprintf("^\n");
    }
    dprintf("\n");
}


void i2cPoll(char* input1023)
{
    char *input = input1023, *stop;

    U032 portID = gPortID;
    U032 i2cAddr = 0;   //I2c Device address for DDC capable HDMI Monitors

    //Backing up register values which will be used during this test
    U032 const i2cOverrideVal = GPU_REG_RD32(LW_PMGR_I2C_OVERRIDE(portID));
    U032 const i2cAddrVal = GPU_REG_RD32(LW_PMGR_I2C_ADDR(portID));
    U032 const i2cCntlVal = GPU_REG_RD32(LW_PMGR_I2C_CNTL(portID));
    U032 const i2cPollVal = GPU_REG_RD32(LW_PMGR_I2C_POLL(portID));
    U032 const i2cIntEnBit = GPU_REG_RD32(LW_PMGR_RM_INTR_EN_I2C) & BIT(portID);
    U032 const i2cIntMaskBit = GPU_REG_RD32(LW_PMGR_RM_INTR_MSK_I2C) & BIT(portID);
    U032 const i2cInterruptBit = GPU_REG_RD32(LW_PMGR_RM_INTR_I2C) & BIT(portID);

    U032 statusVal = 0;

    skipDelims(&input, GENERIC_DELIMS);

    if(FALSE == GetExpressionEx(input, (ULONG64 *)&i2cAddr, &stop))
    {
        SyntaxError(--input1023, stop, "bad format for I2C device (addr).");
        return;
    }

    dprintf("\nI2c Port ID ='0x%08X'\n", portID);
    dprintf("PARSE: i2cAddr='0x%08X'\n", i2cAddr);

    if(FLD_TEST_DRF(_PMGR, _I2C_OVERRIDE, _SCLPAD_IN, _ZERO, i2cOverrideVal )
        || FLD_TEST_DRF(_PMGR, _I2C_OVERRIDE, _SDAPAD_IN, _ZERO, i2cOverrideVal ))
    {
        dprintf("\nLW_PMGR_I2C_OVERRIDE_SCLPAD_IN or LW_PMGR_I2C_OVERRIDE_SDAPAD_IN"
                " bits are not pulled up, so this feature can't be verified.\n");
        return;
    }


    GPU_REG_WR32(LW_PMGR_I2C_OVERRIDE(portID),
            FLD_SET_DRF(_PMGR, _I2C_OVERRIDE, _SIGNALS, _DISABLE, i2cOverrideVal));

    GPU_REG_WR32(LW_PMGR_RM_INTR_EN_I2C,
            FLD_IDX_SET_DRF(_PMGR, _RM_INTR_EN_I2C, _I2C, portID, _DISABLED,
                                    GPU_REG_RD32(LW_PMGR_RM_INTR_EN_I2C)));

    GPU_REG_WR32(LW_PMGR_RM_INTR_MSK_I2C,
            FLD_IDX_SET_DRF(_PMGR, _RM_INTR_MSK_I2C, _I2C, portID, _ENABLED,
                                    GPU_REG_RD32(LW_PMGR_RM_INTR_MSK_I2C)));

    GPU_REG_WR32(LW_PMGR_I2C_ADDR(portID), i2cAddr);

    GPU_REG_WR32(LW_PMGR_I2C_CNTL(portID),
                    FLD_SET_DRF(_PMGR, _I2C_CNTL, _CMD, _RESET, i2cCntlVal));

    GPU_REG_WR32(LW_PMGR_I2C_POLL(portID),
            FLD_SET_DRF(_PMGR, _I2C_POLL, _PERIOD, _INIT, i2cPollVal)|
            FLD_SET_DRF(_PMGR, _I2C_POLL, _INITIAL_DEV_STATUS, _UNKNOWN, i2cPollVal)|
            FLD_SET_DRF(_PMGR, _I2C_POLL, _ENABLE, _YES, i2cPollVal));


    // Waiting for dev_status to be updated in given POLL_PERIOD
    osPerfDelay(LW_PMGR_I2C_POLL_PERIOD_INIT);

    GPU_REG_WR32(LW_PMGR_I2C_POLL(portID),
            FLD_SET_DRF(_PMGR, _I2C_POLL, _ENABLE, _NO, i2cPollVal));

    // Waiting for dev_status to be updated in given POLL_PERIOD
    osPerfDelay(LW_PMGR_I2C_POLL_PERIOD_INIT);

    if(DRF_IDX_DEF(_PMGR, _RM_INTR_I2C, _I2C, portID, _RESET) ==
                (GPU_REG_RD32(LW_PMGR_RM_INTR_I2C) & BIT(portID)))
    {
        dprintf("\nInterrupt asserted from h/w successfully\n");
    }
    else
    {
        dprintf("\nFailed: Interrupt did not arrive\n");
    }

    switch(statusVal = GPU_REG_IDX_RD_DRF(_PMGR, _I2C_CNTL, portID, _STATUS))
    {
        case LW_PMGR_I2C_CNTL_STATUS_OKAY:
            dprintf("\nACK arrived from h/w successfully\n");
            break;

        case LW_PMGR_I2C_CNTL_STATUS_NO_ACK:
            dprintf("\nNACK arrived from h/w\n");
            break;

        case LW_PMGR_I2C_CNTL_STATUS_TIMEOUT:
            dprintf("\nTimed out, no response from h/w\n");
            break;

        case LW_PMGR_I2C_CNTL_STATUS_BUS_BUSY:
            dprintf("\nI2c Bus busy\n");
            break;

        default:
            dprintf("\nControl Status bits (%x) are not valid...\n", statusVal);
    }


    //Restoring original register values back
    GPU_REG_WR32(LW_PMGR_I2C_OVERRIDE(portID), i2cOverrideVal);
    GPU_REG_WR32(LW_PMGR_I2C_ADDR(portID), i2cAddrVal);
    GPU_REG_WR32(LW_PMGR_I2C_CNTL(portID), i2cCntlVal);
    GPU_REG_WR32(LW_PMGR_I2C_POLL(portID), i2cPollVal);
    GPU_REG_WR32(LW_PMGR_RM_INTR_EN_I2C, ((~(BIT(portID))) | i2cIntEnBit)
                        & GPU_REG_RD32(LW_PMGR_RM_INTR_EN_I2C));
    GPU_REG_WR32(LW_PMGR_RM_INTR_MSK_I2C, ((~(BIT(portID))) | i2cIntMaskBit)
                        & GPU_REG_RD32(LW_PMGR_RM_INTR_MSK_I2C));
    GPU_REG_WR32(LW_PMGR_RM_INTR_I2C, ((~(BIT(portID))) | i2cInterruptBit)
                        & GPU_REG_RD32(LW_PMGR_RM_INTR_I2C));

 }


void ReadI2C(char* input1023)
{
    U008 ack;
    char* stop;
    char* input;
    U008 addr, subaddr[10];
    U016 count=1;
    U008 numsubs=0;
    U008 read64K[65536];
    U016 countmax=128;
    U008 skip2=0;

    input = input1023;

    if (!isDelim(*input, GENERIC_DELIMS))
    {
        SyntaxError(--input1023, input, "initial delims missing.");
        return;
    }

    skipDelims(&input, GENERIC_DELIMS);

    if (!hex2byte(input, &addr, ", \t\n", &stop))
    {
        SyntaxError(--input1023, stop, "bad format for I2C device (addr).");
        return;
    }
#if (0)
    I2C_DEBUGOUT(PARSING, ("PARSE: stop='%s'\n", stop));
    I2C_DEBUGOUT(PARSING, ("PARSE: addr=0x%02X\n", addr));
#endif
    //what delim stopped us?  If a ',' then there's a sub-address
    input = stop;
    while ((',' == *input) && (numsubs < 10))
    {
        input++;
        if (!hex2byte(input, &subaddr[numsubs], GENERIC_DELIMS "," , &stop))
        {
            SyntaxError(--input1023, stop, "bad format for I2C subaddress.");
            return;
        }
#if (0)
        I2C_DEBUGOUT(PARSING, ("PARSE: subaddress=0x%02X\n", subaddr[numsubs]));
#endif
        input = stop;
        numsubs++;
    }
#if (0)
    I2C_DEBUGOUT(PARSING, ("PARSE: subaddress depth=0x%02X\n", numsubs));
#endif
    if (numsubs > 9)
    {
        SyntaxError(--input1023, stop, "bad I2C read address depth (depth > 9).");
        return;
    }

    //there might be a count param
    input = stop;
    skipDelims(&input, GENERIC_DELIMS);
    if (*input)
    {
        if (!dec2word(input, &count, GENERIC_DELIMS, &stop))
        {
            SyntaxError(--input1023, stop, "bad format for read count (should be decimal.)");
            return;
        }
#if (0)
        I2C_DEBUGOUT(PARSING, ("PARSE: count=0x%02X\n", count));
#endif
    }
#if (0)
    else
        I2C_DEBUGOUT(PARSING, ("PARSE: <no count>.\n"));
#endif

    //there might be a '/f' for countmax override, or a '/2' for skip2
    input = stop;
    skipDelims(&input, GENERIC_DELIMS);
    while (*input == '/')
    {
        ++input;
        switch (TOUPPER(*input))
        {
            case 'F':
                ++input;
                countmax = 65535;
#if (0)
                I2C_DEBUGOUT(PARSING, ("PARSE: countmax override enabled.\n"));
#endif
                break;

            case '2':
                ++input;
                skip2 = 1;
#if (0)
                I2C_DEBUGOUT(PARSING, ("PARSE: skip-by-2 enabled.\n"));
#endif
                break;

            default:
                SyntaxError(--input1023, stop, "bad format; expected '/f' or '/2' or end of line.");
                return;
        }

        skipDelims(&input, GENERIC_DELIMS);
    }


    //that should be the end of input.
    skipDelims(&input, GENERIC_DELIMS);
    if (*input)
    {
        SyntaxError(--input1023, input, "data found past expected end-of-line.");
        return;
    }

    if (count > countmax)
    {
#if (0)
        I2C_DEBUGOUT(PARSING, ("PARSE: this guard was added because I2C is slooowwww via serial WinDBG.\n"));
#endif
        SyntaxError(--input1023, stop, "read count questionably high, precede count with '/f' to override max to 65535.");
        return;
    }


    if (numsubs)
        ack = ReadRegisterExI2C(addr, numsubs, subaddr, read64K, count);
    else
        ack = ReadDeviceI2C(addr, read64K, count);

    if (ack)
        DumpI2C(read64K, numsubs, subaddr, count, skip2);
#if (0)
    else
        I2C_DEBUGOUT(NORMAL, ("*** communications error (device not present?)\n"));
#endif
}



void WriteI2C(char* input1023)
{
#if (0)
    U008 ack;
#endif
    char* stop;
    char* input;
    U008 addr, count=0;
    U008 write256[256];

    input = input1023;

    if (!isDelim(*input, GENERIC_DELIMS))
    {
        SyntaxError(--input1023, input, "initial delims missing.");
        return;
    }

    skipDelims(&input, GENERIC_DELIMS);

    if (!hex2byte(input, &addr, ", \t", &stop))
    {
        SyntaxError(--input1023, stop, "bad format for I2C device (addr).");
        return;
    }
#if (0)
    I2C_DEBUGOUT(PARSING, ("PARSE: addr=0x%02X\n", addr));
#endif
    input = stop;
    skipDelims(&input, ", \t");
    while (*input)
    {
#if (0)
        I2C_DEBUGOUT(PARSING, ("PARSE: begin data: '%s'\n", input));
#endif
        if (!hex2byte(input, &write256[count], GENERIC_DELIMS, &stop))
        {
            SyntaxError(--input1023, stop, "bad format for I2C subaddress and/or data.");
            return;
        }
#if (0)
        I2C_DEBUGOUT(PARSING, ("PARSE: subaddr (or data)=0x%02X\n", write256[count]));
#endif
        count++;
        input = stop;
        skipDelims(&input, GENERIC_DELIMS);
    }

#if (0)
    ack = WriteDeviceI2C(addr, write256, count);
    if (!ack)
        I2C_DEBUGOUT(NORMAL, ("*** communications error (device not present?)\n"));
#endif
}


void i2cMenu()
{
    U008 done=0;
    U008 oldlock, oldindex;
    char input1024[1024];
    U032 i2cCRTCOffset = getI2CCRTCOffset();

    oldindex = GPU_REG_RD08(0x6013d4 + i2cCRTCOffset);
    oldlock = UnlockExtendedCRTCs(i2cCRTCOffset);

    dprintf("lw: Starting i2c Menu. (Type '?' for help)\n");

    while (!done)
    {
        dprintf("\n\n");
        memset(input1024, 0, sizeof(input1024));

        dprintf("current port: %u\n", gPortID);
        dprintf("current head: %u\n", gHeadID);
        if (osGetInputLine((U008 *)"i2c> ", (U008 *)input1024, sizeof(input1024)))
        {
            switch (TOUPPER(input1024[0]))
            {
                default:
                    dprintf("*** Unknown command!  Printing help...\n\n");
                    //intentional fall-through

                case '?':
                    dprintf("lw: i2c help file\n");
                    dprintf("USAGE: <COMMAND> [ARGS]\n\n");

                    dprintf("COMMAND  ARGUMENTS                        DESCRIPTION\n");
                    dprintf("~~~~~~~  ~~~~~~~~~                        ~~~~~~~~~~~\n\n");

                    dprintf("   ?                                      Prints this help file.\n");
                    dprintf("   +                                      Increases verbosity.\n");
                    dprintf("   -                                      Decreases verbosity.\n");
                    dprintf("   p     <portID>                         Changes the current I2C port ID, as listed in the DCB.\n");
                    dprintf("   h     <headID>                         Changes which head is used to control the I2C port.\n");
                    dprintf("   l     <addr>                           Poll on current I2C for connected displays.\n");
                    dprintf("   r     <addr>[, <subaddr>]* [<count>=1] Reads from an I2C device\n");
                    dprintf("   w     <addr>[, <subaddr>]* <data>*     Writes to an I2C device\n");
                    dprintf("   q                                      Quit the i2c interface, restoring previous configuration.\n");
                    break;

                case 'P':  //change current port
                    ChangePortID(&input1024[1]);
                    break;

                case 'H':  //change controlling head
                    ChangeHeadID(&input1024[1]);
                    break;

                case 'L':  //Poll on current I2c port
                    if (IsGK104orLater())
                    {
                        i2cPoll(&input1024[1]);
                    }
                    else
                    {
                        dprintf("This feature can only be verified on GK104 and later GPUs.");
                    }
                    break;

                case 'R':  //read from I2C device
                    ReadI2C(&input1024[1]);
                    break;

                case 'W':  //write to I2C device
                    WriteI2C(&input1024[1]);
                    break;

                case '+':  //increases verbosity
                    gVerbosity = (gVerbosity +1 ? gVerbosity+1 : gVerbosity);
                    dprintf("Verbosity level is now at %u.\n", gVerbosity);
                    break;

                case '-':  //decreases verbosity
                    gVerbosity = (gVerbosity ? gVerbosity-1 : gVerbosity);
                    dprintf("Verbosity level is now at %u.\n", gVerbosity);
                    break;

                case 'Q':  //quit
                    dprintf("Exiting user I2C interface.\n");
                    done = 1;
                    break;
            }
        }
    }

    RestoreExtendedCRTCs(oldlock, i2cCRTCOffset);
    GPU_REG_WR08(0x6013d4 + i2cCRTCOffset, oldindex);
}
