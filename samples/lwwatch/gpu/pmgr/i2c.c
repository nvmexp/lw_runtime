/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 1999-2016 by LWPU Corporation.  All rights reserved.  All
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
#include "i2c.h"
#include "chip.h"
#include "kepler/gk110/dev_pmgr.h"


#define TERSE   0
#define NORMAL  1
#define PARSING 2

#if LWWATCHCFG_IS_PLATFORM(UNIX) || LWWATCHCFG_FEATURE_ENABLED(MODS_UNIX) || defined(USERMODE)
#define I2C_DEBUGOUT(level, params) { if (level <= gVerbosity) dprintf params; }
#else
#define I2C_DEBUGOUT(level, params) { if (level <= gVerbosity) dprintf##params; }
#endif

LwU8 gPortID = 0;
LwU8 gHeadID = 0;
LwU8 gVerbosity = NORMAL;

LwU32 wMaxWaitStates=3000;  // max wait states for I2C bus syncronisation
ICBENTRY ICBEntry[MAX_ICB_ENTRIES];

LwU32 getI2CCRTCOffset(void);


void i2cWriteWrIndex
(
    LwU32 portID,
    LwU8 data
)
{
    LwU32 crtcOffset = getI2CCRTCOffset();

    switch( ICBEntry[portID].CR.Access )
    {
        case 0: //LW_ICB20_REC_ACCESS_INDEXED:
            REG_WRCR( ICBEntry[portID].CR.WriteIdx, data, crtcOffset );
            break;
        case 1: // LW_ICB20_REC_ACCESS_DIRECT_IO:
            dprintf("Unimplemented i2c direct io access\n");
            break;
        case 2: // LW_ICB20_REC_ACCESS_PCI_IO:
            /*
            {
                LwU32   uI2CPortW;

                dacGetPCIIOMappedI2C( pDev, &uI2CPortW,
                    (LwU8) ICB20_PCI_BUS(pDev, portID),
                    (LwU8) ICB20_PCI_DEVICE(pDev, portID),
                    (LwU8) ICB20_PCI_FUNCTION(pDev, portID),
                    (LwU8) ICB20_PCI_BAR(pDev, portID),
                    (LwU8) ICB20_PCI_OFFSET(pDev, portID) );

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


LwU8 i2cReadWrIndex
(
    LwU32 portID
)
{
    LwU8 data = 0xFF;
    LwU32 crtcOffset = getI2CCRTCOffset();

    switch( ICBEntry[portID].CR.Access )
    {
        case 0: //LW_ICB20_REC_ACCESS_INDEXED:
            // CRTC_RD( ICB20_IDX_WRITEPORT(pDev, portID), data, Head );
            data = REG_RDCR( ICBEntry[portID].CR.WriteIdx, crtcOffset );
            break;
        case 1: // LW_ICB20_REC_ACCESS_DIRECT_IO:
            dprintf("Unimplemented i2c direct io access\n");
            break;
        case 3: // LW_ICB20_REC_ACCESS_PCI_IO:
            /*
            {
                LwU32   uI2CPortW;

                dacGetPCIIOMappedI2C( pDev, &uI2CPortW,
                    (LwU8) ICB20_PCI_BUS(pDev, portID),
                    (LwU8) ICB20_PCI_DEVICE(pDev, portID),
                    (LwU8) ICB20_PCI_FUNCTION(pDev, portID),
                    (LwU8) ICB20_PCI_BAR(pDev, portID),
                    (LwU8) ICB20_PCI_OFFSET(pDev, portID) );

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


LwU8 i2cReadStatusIndex
(
    LwU32 portID
)
{
    LwU8 data = 0xFF;
    LwU32 crtcOffset = getI2CCRTCOffset();

    switch( ICBEntry[portID].CR.Access )
    {
        case 0: //LW_ICB20_REC_ACCESS_INDEXED:
            //CRTC_RD( ICB20_IDX_READPORT(pDev, portID), data, Head );
            data = REG_RDCR( ICBEntry[portID].CR.ReadIdx, crtcOffset );
            break;
        case 1: // LW_ICB20_REC_ACCESS_DIRECT_IO:
            dprintf("Unimplemented i2c direct io access\n");
            break;
        case 3: // LW_ICB20_REC_ACCESS_PCI_IO:
            /*
            {
                LwU32   uI2CPortR;

                dacGetPCIIOMappedI2C( pDev, &uI2CPortR,
                    (LwU8) ICB20_PCI_BUS(pDev, portID),
                    (LwU8) ICB20_PCI_DEVICE(pDev, portID),
                    (LwU8) ICB20_PCI_FUNCTION(pDev, portID),
                    (LwU8) ICB20_PCI_BAR(pDev, portID),
                    (LwU8) ICB20_PCI_OFFSET(pDev, portID) );

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

LwU8 i2cHardwareInit
(
    LwU32 portID
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
        LwU32   uI2CPortBase = 0;
        LwU32   uI2CPortR = 0;
        LwU32   uI2CPortW = 0;

        //Get the current I2C base port, based on the DCB PCI IO mapping info
        dacGetPCIIOMappedI2C( pDev, &uI2CPortBase,
            (LwU8) DRF_VAL(_ICB20, _REC, _PCI_IO_BUS,      pDev->Dac.PCIIOBackupICB[portID]),
            (LwU8) DRF_VAL(_ICB20, _REC, _PCI_IO_DEVICE,   pDev->Dac.PCIIOBackupICB[portID]),
            (LwU8) DRF_VAL(_ICB20, _REC, _PCI_IO_FUNCTION, pDev->Dac.PCIIOBackupICB[portID]),
            (LwU8) DRF_VAL(_ICB20, _REC, _PCI_IO_BAR,      pDev->Dac.PCIIOBackupICB[portID]),
            (LwU8) DRF_VAL(_ICB20, _REC, _PCI_IO_OFFSET,   pDev->Dac.PCIIOBackupICB[portID]) );

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


void i2cWriteCtrl
(
    LwU32 portID,
    LwU8 reg,
    LwU8 bit
)
{
    LwU8 data;

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

LwU8 i2cReadCtrl
(
    LwU32 portID,
    LwU8 reg
)
{
    LwU8 data;

    data = i2cReadStatusIndex(portID);

    if (reg == SCL_REG)
        return ( (data & I2C_SRCK_IN) != 0);
    else
        return ( (data & I2C_SRD_IN) != 0);
}

/**********************************************************************/

void ReadSDA(LwU32 portID, LwU8 *data)
{
    *data = i2cReadCtrl(portID, SDA_REG);
}

void ReadSCL(LwU32 portID, LwU8 *data)
{
    *data = i2cReadCtrl(portID, SCL_REG);
}

void SetSCLLine(LwU32 portID)
{
    i2cWriteCtrl(portID, SCL_REG, 1);
}

void ResetSCLLine(LwU32 portID)
{
    i2cWriteCtrl(portID, SCL_REG, 0);
}

void SetSDALine(LwU32 portID)
{
    i2cWriteCtrl(portID, SDA_REG, 1);
}

void ResetSDALine(LwU32 portID)
{
    i2cWriteCtrl(portID, SDA_REG, 0);
}

/*
 * waits for a specified line til it goes high
 * giving up after MAX_WAIT_STATES attempts
 * return:  0 OK
 *         -1 fail (time out)
 */
LwU8 WaitHighSDALine(LwU32 portID)
{
    LwU8    data_in;
    LwU32   retries = wMaxWaitStates;

    do
    {
        ReadSDA(portID, &data_in);      // wait for the line going high
        if (data_in)
            break;
        osPerfDelay(I2CDELAY);
    } while (--retries);        // count down is running

    if (!retries)
        return((LwU8)-1);
    return 0;
}

LwU8 WaitHighSCLLine(LwU32 portID)
{
    LwU8    data_in;
    LwU32   retries = wMaxWaitStates;

    do
    {
        osPerfDelay(1);              // 1.0 us delay   NEEDED??
        ReadSCL(portID, (LwU8 *)&data_in);   // wait for the line going high
        if (data_in)
            break;
    } while (--retries);            // count down is running

    if (!retries)
    {
        return((LwU8)-1);
    }

    return(0);
}


void i2cStart(LwU32 portID)
{
    SetSDALine(portID);
    osPerfDelay(I2CDELAY);
    SetSCLLine(portID);
    osPerfDelay(I2CDELAY);      // spec requires clock to be high min of 4us
    WaitHighSCLLine(portID);
    ResetSDALine(portID);
    osPerfDelay(I2CDELAY);
    ResetSCLLine(portID);
    I2C_DEBUGOUT(NORMAL, ("\nS"));
}


void i2cStop(LwU32 portID)
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
    I2C_DEBUGOUT(NORMAL, ("P\n"));
}


/*
 * I2cAck() returns 1: fail
 *                  0: acknolege
 */

LwU8 i2cAck(LwU32 portID)
{
    LwU8 ack;

    ResetSCLLine(portID);
    osPerfDelay(I2CDELAY);
    SetSDALine(portID);
    osPerfDelay(I2CDELAY);
    SetSCLLine(portID);
    osPerfDelay(I2CDELAY);     // spec requires clock to be high min of 4us
    WaitHighSCLLine(portID);
    ReadSDA(portID, &ack);
    ResetSCLLine(portID);
    I2C_DEBUGOUT(NORMAL, ("%s", ack ? "N " : "A "));
    return (ack);
}


void i2cInit(LwU32 portID)
{
    SetSCLLine(portID);
    osPerfDelay(I2CDELAY);     // spec requires clock to be high min of 4us
    WaitHighSCLLine(portID);
    SetSDALine(portID);
}


LwU8 i2cSendByte(LwU32 portID, LwU8 byte)
{
    LwU8 i;
    LwU8 save = byte;

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
        I2C_DEBUGOUT(NORMAL, ("%s", (byte & 0x80) ? "1" : "0"));
        byte <<= 1;
    }
    I2C_DEBUGOUT(NORMAL, ("(%02x)", save));

    i = i2cAck(portID);

    return (i);
}


LW_STATUS i2cReceiveByte(LwU32 portID, LwU8 *byte, LwU8 nack)
{
    LwU8 data=0;
    LwU8 i;
    LW_STATUS status;

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
        status = WaitHighSCLLine(portID) ? LW_ERR_GENERIC : LW_OK;
        if (status != LW_OK)
            goto done;
        osPerfDelay(I2CDELAY_CLK_HIGH);          // clock must be high at least 4us

        ReadSDA(portID, &data);
        I2C_DEBUGOUT(NORMAL, ("%s", (data) ? "1" : "0"));
        *byte <<= 1;
        *byte  |= (data == 1);
    }
    I2C_DEBUGOUT(NORMAL, ("(%02x)", *byte));

    ResetSCLLine(portID);
    if (nack)
    {
        SetSDALine(portID);         // send Nack
    }
    else
        ResetSDALine(portID);       // send Ack

    I2C_DEBUGOUT(NORMAL, ("%s", nack ? "N " : "A "));

    osPerfDelay(I2CDELAY_CLK_LOW);           // clock must be low at least 4.7 us
    SetSCLLine(portID);
    status = WaitHighSCLLine(portID) ? LW_ERR_GENERIC : LW_OK;
    osPerfDelay(I2CDELAY_CLK_HIGH);          // clock must be high at least 4us
    ResetSCLLine(portID);
    osPerfDelay(I2CDELAY_CLK_LOW);           // clock must be low at least 4.7 us

done:

    return status;
}


LwU32 i2cWrite(LwU32 portID, LwU8 ChipAdr, LwU16 AdrLen, LwU8 *Adr, LwU16 DataLen, LwU8 *Data)
{
    //
    // Enable writes to the I2C port
    //
    i2cHardwareInit(portID);

    i2cStart(portID);
    if ( i2cSendByte(portID, (LwU8)(ChipAdr<<1)) ) // send chip adr. with write bit
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


LW_STATUS i2cRead(LwU32 portID, LwU8 ChipAdr, LwU16 AdrLen, LwU8 *Adr, LwU16 DataLen, LwU8 *Data)
{
    LwU8 dat;
    LW_STATUS status = LW_ERR_GENERIC;        // pessimist

    //DBG_PRINT_STRING_VALUE(DEBUGLEVEL_TRACEINFO, "LWRM: ChipAdr ", (LwU32)ChipAdr);
    //DBG_PRINT_STRING_VALUE(DEBUGLEVEL_TRACEINFO, "LWRM: AdrLen ", (LwU32)AdrLen);
    //DBG_PRINT_STRING_VALUE(DEBUGLEVEL_TRACEINFO, "LWRM: Adr ", (LwU32)*Adr);
    //DBG_PRINT_STRING_VALUE(DEBUGLEVEL_TRACEINFO, "LWRM: DataLen ", (LwU32)DataLen);

    //
    // Enable writes to the I2C port
    //
    i2cHardwareInit(portID);

    i2cStart(portID);
    i2cSendByte(portID, (LwU8)(ChipAdr<<1));        // send chip adr. with write bit

    for ( ; AdrLen; AdrLen--)               // send sub-register address byte(s)
    {
        if ( i2cSendByte(portID, *Adr++) )
        {
            goto done;
        }
    }

    osPerfDelay(I2CDELAY);    // give the device some time to parse the subaddress

    i2cStart(portID);                             // send again chip address for switching to read mode
    if ( i2cSendByte(portID, (LwU8)((ChipAdr<<1) | 1)) )  // send chip adr. with read bit
    {
        goto done;
    }

    for (status = LW_OK; DataLen && (status == LW_OK); DataLen--)
    {
        status = i2cReceiveByte(portID,
                                (LwU8 *)&dat,
                                (LwU8)((DataLen == 1) ? NACK : ACK));         // receive byte(s)
        *Data++ = dat;
    }

done:
    i2cStop(portID);
    //DBG_PRINT_STRING_VALUE(DEBUGLEVEL_TRACEINFO, "LWRM: Data ", (LwU32)*Data);

    return status;
}

LW_STATUS i2cRead_EDDC(LwU32 portID, LwU8 SegmentAddr, LwU8 ChipAdr, LwU8 SubByteAddr, LwU16 DataLen, LwU8 *Data)
{
    //extended DDC compatibility not confirmed on date modified. No monitors with edids greater than 256 are
    //easily obtainable or locatable.

    LwU8 dat;
    LW_STATUS status = 0xFFFFFFFF;        // pessimist

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
    i2cSendByte(portID, (LwU8)(ChipAdr<<1));        // send chip adr. with write bit
    i2cSendByte(portID, (LwU8)SubByteAddr);

    i2cStart(portID);                             // send again chip address for switching to read mode
    if ( i2cSendByte(portID, (LwU8)((ChipAdr<<1) | 1)) )  // send chip adr. with read bit
    {
        goto done;
    }

    for (status = LW_OK; DataLen && (status == LW_OK); DataLen--)
    {
        status = i2cReceiveByte(portID,
                                (LwU8 *)&dat,
                                (LwU8)((DataLen == 1) ? NACK : ACK));         // receive byte(s)
        *Data++ = dat;
    }

done:
    i2cStop(portID);
    //DBG_PRINT_STRING_VALUE(DEBUGLEVEL_TRACEINFO, "LWRM: Data ", (LwU32)*Data);

    return status;
}

LwU32 i2cSend(LwU32 portID, LwU8 ChipAdr, LwU16 AdrLen, LwU8 *Adr, LwU16 DataLen, LwU8 *Data, LwU32 NoStopFlag)
{

    if ( ChipAdr ) {
        //
        // Enable writes to the I2C port
        //
        i2cHardwareInit(portID);

        i2cStart(portID);
        if ( i2cSendByte(portID, (LwU8)(ChipAdr<<1)) ) // send chip adr. with write bit
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


LwU32 i2cWrite_ALT(LwU32 portID, LwU8 ChipAdr, LwU16 AdrLen, LwU8 *Adr, LwU16 DataLen, LwU8 *Data)
{
    //
    // Enable writes to the I2C port
    //
    i2cHardwareInit(portID);

    i2cStart(portID);
    if ( i2cSendByte(portID, (LwU8)(ChipAdr<<1)) ) // send chip adr. with write bit
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


LwU32 i2cRead_ALT(LwU32 portID, LwU8 ChipAdr, LwU16 AdrLen, LwU8 *Adr, LwU16 DataLen, LwU8 *Data)
{
    LwU8 dat;

    //DBG_PRINT_STRING_VALUE(DEBUGLEVEL_TRACEINFO, "LWRM: ChipAdr ", (LwU32)ChipAdr);
    //DBG_PRINT_STRING_VALUE(DEBUGLEVEL_TRACEINFO, "LWRM: AdrLen ", (LwU32)AdrLen);
    //DBG_PRINT_STRING_VALUE(DEBUGLEVEL_TRACEINFO, "LWRM: Adr ", (LwU32)*Adr);
    //DBG_PRINT_STRING_VALUE(DEBUGLEVEL_TRACEINFO, "LWRM: DataLen ", (LwU32)DataLen);

    //
    // Enable writes to the I2C port
    //
    i2cHardwareInit(portID);

    i2cStart(portID);
    i2cSendByte(portID, (LwU8)((ChipAdr<<1) | 1));        // send chip adr. with write bit
    for ( ; DataLen ; DataLen--)
    {
        i2cReceiveByte(portID, (LwU8 *)&dat, (LwU8)((DataLen == 1) ? NACK : ACK));         // receive byte(s)
        *Data++ = dat;
    }
    i2cStop(portID);

    //DBG_PRINT_STRING_VALUE(DEBUGLEVEL_TRACEINFO, "LWRM: Data ", (LwU32)*Data);

    return 0;
}


LwU32 i2cRead_ALT2(LwU32 portID, LwU8 ChipAdr, LwU16 AdrLen, LwU8 *Adr, LwU16 DataLen, LwU8 *Data)
{
    LwU8 dat;

    //DBG_PRINT_STRING_VALUE(DEBUGLEVEL_TRACEINFO, "LWRM: ChipAdr ", (LwU32)ChipAdr);
    //DBG_PRINT_STRING_VALUE(DEBUGLEVEL_TRACEINFO, "LWRM: AdrLen ", (LwU32)AdrLen);
    //DBG_PRINT_STRING_VALUE(DEBUGLEVEL_TRACEINFO, "LWRM: Adr ", (LwU32)*Adr);
    //DBG_PRINT_STRING_VALUE(DEBUGLEVEL_TRACEINFO, "LWRM: DataLen ", (LwU32)DataLen);

    //
    // Enable writes to the I2C port
    //
    i2cHardwareInit(portID);

    i2cStart(portID);
    i2cSendByte(portID, (LwU8)(ChipAdr<<1));        // send chip adr. with write bit

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
    if ( i2cSendByte(portID, (LwU8)(( ChipAdr<<1) | 1)) )  // send chip adr. with read bit
    {
        i2cStop(portID);                         // ack failed --> generate stop condition
        return 0xFFFFFFFF;
    }

    for ( ; DataLen ; DataLen--)
    {
        i2cReceiveByte(portID, (LwU8 *)&dat, (LwU8)((DataLen == 1) ? NACK : ACK));         // receive byte(s)
        *Data++ = dat;
    }

    i2cStop(portID);

    //DBG_PRINT_STRING_VALUE(DEBUGLEVEL_TRACEINFO, "LWRM: Data ", (LwU32)*Data);

    return 0;
}

static LwU8 i2cReadDevice(LwU8 addr, LwU8* read, LwU16 count)
{
  LwU32 i;
  LwU8 nack;

  if (!read) return 0;

  *read = 0;

  i2cInit(gPortID);
  i2cStart(gPortID);

  nack  =   i2cSendByte(gPortID, (LwU8)(addr | I2C_READCYCLE));

  if(!nack)
  {
    for (i=0; i<count; i++)
    {
      dprintf("%sR", i%8 ? " " : "\n");
      nack = !!i2cReceiveByte(gPortID, (LwU8*)(&read[i]), (LwU8)(i<(LwU32)(count-1) ? 0 : 1));
    }
  }

  i2cStop(gPortID);

  dprintf("\n");

  return !nack;
}

static LwU8 i2cReadRegisterEx(LwU8 addr, LwU8 sublen, LwU8* subaddr, LwU8* read, LwU16 count)
{
  LwU32 i;
  LwU8 nack;

  if (!read) return 0;

  i2cInit(gPortID);
  i2cStart(gPortID);
  nack  = i2cSendByte(gPortID, (LwU8)addr);

  for(i=0; !nack && i<sublen; i++)
      nack |= i2cSendByte(gPortID, (LwU8)subaddr[i]);

  if (!nack)
  {
    i2cStart(gPortID);
    nack  =   i2cSendByte(gPortID, (LwU8)(addr | I2C_READCYCLE));

    for (i=0; i<count; i++)
    {
      dprintf("%sR", i%8 ? "" : "\n");
      nack |= !!i2cReceiveByte(gPortID, (LwU8*)(&read[i]), (LwU8)(i<(LwU32)(count-1) ? 0 : 1));
    }
  }

  i2cStop(gPortID);

  dprintf("\n");

  return !nack;
}

static LwU8 i2cReadRegister(LwU8 addr, LwU8 subaddr, LwU8* read, LwU16 count)
{
  LwU32 i;
  LwU8 nack;

  if (!read) return 0;

  *read = 0;
  i2cInit(gPortID);
  i2cStart(gPortID);
  nack  = i2cSendByte(gPortID, (LwU8)addr);
  if (!nack)
      nack = i2cSendByte(gPortID, (LwU8)subaddr);

  if (!nack)
  {
    i2cStart(gPortID);
    nack  =   i2cSendByte(gPortID, (LwU8)(addr | I2C_READCYCLE));

    for (i=0; i<count; i++)
    {
      dprintf("%sR", i%8 ? "" : "\n");
      nack |= !!i2cReceiveByte(gPortID, (LwU8*)(&read[i]), (LwU8)(i<(LwU32)(count-1) ? 0 : 1));
    }
  }

  i2cStop(gPortID);

  dprintf("\n");

  return !nack;
}

static LwU8 i2cWriteDevice(LwU8 addr, LwU8* data, LwU8 count)
{
  LwU16 i;
  LwU8 nack;

  i2cInit(gPortID);
  i2cStart(gPortID);
  nack  = i2cSendByte(gPortID, (LwU8)addr);


  for (i=0; !nack && i<count; i++)
  {
    dprintf("%sW", i%8 ? "" : "\n");
    nack |= i2cSendByte(gPortID, (LwU8)data[i]);
  }

  i2cStop(gPortID);

  dprintf("\n");

  return !nack;
}


#if 0
static LwU8 i2cWriteRegister(LwU8 addr, LwU8 subaddr, LwU8 data)
{
  LwU8 nack;

  i2cInit(gPortID);
  i2cStart(gPortID);
  nack  = i2cSendByte(gPortID, (LwU8)addr);
  nack |= i2cSendByte(gPortID, (LwU8)subaddr);
  nack |= i2cSendByte(gPortID, (LwU8)data);

  i2cStop(gPortID);
  return !nack;
}
#endif


//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


LwU32 getI2CCRTCOffset()
{
    dprintf("i2c not supported by lwwatch");
    return 0;
}

LwU8 hex2byte(char *input, LwU8* value, char* delims, char** stop)
{
    LwU8 i, temp;

    if (stop) *stop = (char*)NULL;

    skipDelims(&input, delims);

    *value = 0;
    for (i=2; i && *input && !isDelim(*input, delims); i--, input++)
    {
        *value *= 16;
        temp = TOUPPER(*input);

        if ((temp >= '0') && (temp <= '9'))
            *value += (LwU8)(temp - '0');
        else if ((temp >= 'A') && (temp <= 'F'))
            *value += (LwU8)(temp - 'A' +  10);
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


LwU8 dec2word(char *input, LwU16* value, char* delims, char** stop)
{
    LwU8 i, temp;

    if (stop) *stop = (char*)NULL;

    skipDelims(&input, delims);

    *value = 0;
    for (i=5; i && *input && !isDelim(*input, delims); i--, input++)
    {
        *value *= 10;
        temp = TOUPPER(*input);

        if ((temp >= '0') && (temp <= '9'))
            *value += (LwU8)(temp - '0');
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
    LwU8 port;
    char* input;

    input = input1023;

    if (!isDelim(*input, GENERIC_DELIMS))
    {
        dprintf("*** invalid format for 'change <p>ort ID'.\n");
        dprintf("    format is: 'p <portID>' where <portID> is 0-F (hex).");
        return;
    }

    skipDelims(&input, GENERIC_DELIMS);

    port = (LwU8)(*input - '0');
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
    LwU8 head;
    char* input;

    input = input1023;

    if (!isDelim(*input, GENERIC_DELIMS))
    {
        dprintf("*** invalid format for 'change <h>ead ID'.\n");
        dprintf("    format is: 'h <headID>' where <headID> is 0-F (hex).\n");
        return;
    }

    skipDelims(&input, GENERIC_DELIMS);

    head = (LwU8)(*input - '0');
    if (head > 0x0F)
    {
        dprintf("*** invalid format for 'change <p>ort ID'.\n");
        dprintf("    format is: 'p <portID>' where <portID> is 0-F (hex).\n");
        return;
    }

    dprintf("Current head has been changed to %u.\n", head);
    gHeadID = head;
}


void DumpI2C(LwU8* data, LwU8 numsubs, LwU8* subaddr, LwU16 count, LwU8 skip2)
{
    LwU32 i,j;

    I2C_DEBUGOUT(PARSING, ("PARSE: skip2='%u'\n", skip2));
    I2C_DEBUGOUT(PARSING, ("PARSE: numsubs='%u'\n", numsubs));
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


LwU8 ReadRegisterI2C(LwU8 addr, LwU8 subaddr, LwU8* data64K, LwU8 count)
{
    dprintf("*** reading (decimal) %u bytes from device %X, starting from subaddress %X.\n", (LwU32)count, (LwU32)addr, (LwU32)subaddr);
    return i2cReadRegister(addr, subaddr, data64K, count);
}

LwU8 ReadRegisterExI2C(LwU8 addr, LwU8 sublen, LwU8* subaddr, LwU8* data64K, LwU16 count)
{
    dprintf("*** reading (decimal) %u bytes from device %X, starting from subaddress %X (depth>1).\n", (LwU32)count, (LwU32)addr, (LwU32)subaddr[0]);
    return i2cReadRegisterEx(addr, sublen, subaddr, data64K, count);
}

LwU8 ReadDeviceI2C(LwU8 addr, LwU8* data64K, LwU16 count)
{
    dprintf("*** reading (decimal) %u bytes from device %X.\n", count, addr);
    return i2cReadDevice(addr, data64K, count);
}


LwU8 WriteDeviceI2C(LwU8 addr, LwU8* data256, LwU8 count)
{
    dprintf("*** sending (decimal) %u bytes to device %X.\n", count, addr);
    return i2cWriteDevice(addr, data256, count);
}

void SyntaxError(char* input1024, char* stop, char* error)
{
    LwU32 i,j;
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

    LwU32 portID = gPortID;
    LwU32 i2cAddr = 0;   //I2c Device address for DDC capable HDMI Monitors
    LwU64 tmp;

    //Backing up register values which will be used during this test
    LwU32 const i2cOverrideVal = GPU_REG_RD32(LW_PMGR_I2C_OVERRIDE(portID));
    LwU32 const i2cAddrVal = GPU_REG_RD32(LW_PMGR_I2C_ADDR(portID));
    LwU32 const i2cCntlVal = GPU_REG_RD32(LW_PMGR_I2C_CNTL(portID));
    LwU32 const i2cPollVal = GPU_REG_RD32(LW_PMGR_I2C_POLL(portID));
    LwU32 const i2cIntEnBit = GPU_REG_RD32(LW_PMGR_RM_INTR_EN_I2C) & BIT(portID);
    LwU32 const i2cIntMaskBit = GPU_REG_RD32(LW_PMGR_RM_INTR_MSK_I2C) & BIT(portID);
    LwU32 const i2cInterruptBit = GPU_REG_RD32(LW_PMGR_RM_INTR_I2C) & BIT(portID);

    LwU32 statusVal = 0;

    skipDelims(&input, GENERIC_DELIMS);

    if(FALSE == GetExpressionEx(input, &tmp, &stop))
    {
        SyntaxError(--input1023, stop, "bad format for I2C device (addr).");
        return;
    }
    if (LwU64_HI32(tmp) != 0) {
        dprintf("Invalid i2c address\n");
        return;
    }
    i2cAddr = (LwU32)tmp;

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
    LwU8 ack;
    char* stop;
    char* input;
    LwU8 addr, subaddr[10];
    LwU16 count=1;
    LwU8 numsubs=0;
    LwU8 read64K[65536];
    LwU16 countmax=128;
    LwU8 skip2=0;

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
    I2C_DEBUGOUT(PARSING, ("PARSE: stop='%s'\n", stop));
    I2C_DEBUGOUT(PARSING, ("PARSE: addr=0x%02X\n", addr));

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
        I2C_DEBUGOUT(PARSING, ("PARSE: subaddress=0x%02X\n", subaddr[numsubs]));
        input = stop;
        numsubs++;
    }

    I2C_DEBUGOUT(PARSING, ("PARSE: subaddress depth=0x%02X\n", numsubs));
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
        I2C_DEBUGOUT(PARSING, ("PARSE: count=0x%02X\n", count));
    }
    else
        I2C_DEBUGOUT(PARSING, ("PARSE: <no count>.\n"));


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
                I2C_DEBUGOUT(PARSING, ("PARSE: countmax override enabled.\n"));
                break;

            case '2':
                ++input;
                skip2 = 1;
                I2C_DEBUGOUT(PARSING, ("PARSE: skip-by-2 enabled.\n"));
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
        I2C_DEBUGOUT(PARSING, ("PARSE: this guard was added because I2C is slooowwww via serial WinDBG.\n"));
        SyntaxError(--input1023, stop, "read count questionably high, precede count with '/f' to override max to 65535.");
        return;
    }


    if (numsubs)
        ack = ReadRegisterExI2C(addr, numsubs, subaddr, read64K, count);
    else
        ack = ReadDeviceI2C(addr, read64K, count);

    if (ack)
        DumpI2C(read64K, numsubs, subaddr, count, skip2);
    else
        I2C_DEBUGOUT(NORMAL, ("*** communications error (device not present?)\n"));
}



void WriteI2C(char* input1023)
{
    LwU8 ack;
    char* stop;
    char* input;
    LwU8 addr, count=0;
    LwU8 write256[256];

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
    I2C_DEBUGOUT(PARSING, ("PARSE: addr=0x%02X\n", addr));

    input = stop;
    skipDelims(&input, ", \t");
    while (*input)
    {
        I2C_DEBUGOUT(PARSING, ("PARSE: begin data: '%s'\n", input));

        if (!hex2byte(input, &write256[count], GENERIC_DELIMS, &stop))
        {
            SyntaxError(--input1023, stop, "bad format for I2C subaddress and/or data.");
            return;
        }
        I2C_DEBUGOUT(PARSING, ("PARSE: subaddr (or data)=0x%02X\n", write256[count]));

        count++;
        input = stop;
        skipDelims(&input, GENERIC_DELIMS);
    }

    ack = WriteDeviceI2C(addr, write256, count);

    if (!ack)
        I2C_DEBUGOUT(NORMAL, ("*** communications error (device not present?)\n"));
}


void i2cMenu()
{
    LwU8 done=0;
    LwU8 oldindex;
    char input1024[1024];
    LwU32 i2cCRTCOffset = getI2CCRTCOffset();

    oldindex = GPU_REG_RD08(0x6013d4 + i2cCRTCOffset);

    dprintf("lw: Starting i2c Menu. (Type '?' for help)\n");

    while (!done)
    {
        dprintf("\n\n");
        memset(input1024, 0, sizeof(input1024));

        dprintf("current port: %u\n", gPortID);
        dprintf("current head: %u\n", gHeadID);
        if (osGetInputLine((LwU8 *)"i2c> ", (LwU8 *)input1024, sizeof(input1024)))
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
                    if (IsGM107orLater())
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

    GPU_REG_WR08(0x6013d4 + i2cCRTCOffset, oldindex);
}
