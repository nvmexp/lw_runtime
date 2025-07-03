//*****************************************************
//
// lwwatch WinDbg Extension
// retodd@lwpu.com - 2.1.2002
// i2c.h
//
//*****************************************************

#ifndef _I2C_H_
#define _I2C_H_

#include "os.h"


#define I2C_ADDRESS_RETRIES     8

// Serial Port Bits
#define I2C_SRCK        0x20    // Serial Clock write
#define I2C_SRD         0x10    // Serial Data  write
#define I2C_SRCK_IN     0x04    // Serial Clock read
#define I2C_SRD_IN      0x08    // Serial Data  read
#define I2C_ENABLE      0x01    // Enable Serial Port Function

///////////////////////////////////////////////////////////////////
// Constants used by I2C Protocol:

#define SDA_REG         0x02
#define SCL_REG         0x01

#define ACK             0   // I2C Acknowledge
#define NACK            1   // I2C No Acknowledge


#define I2CDELAY 10              // 10 usec
#define I2CDELAY_CLK_LOW   6     // low period of clock must be at least 4.7 usec - added .5 margin of error
#define I2CDELAY_CLK_HIGH  5     // high period of clock must be at least 4.0 usec - "

#define I2C_READCYCLE (LwU8)0x01

extern ICBENTRY ICBEntry[MAX_ICB_ENTRIES];

//
// state dump routines - I2C.c
//
LwU32 getI2CCRTCOffset(void);
void i2cWriteWrIndex(LwU32, LwU8);
LwU8 i2cReadWrIndex(LwU32);
LwU8 i2cReadStatusIndex(LwU32);
LwU8 i2cHardwareInit(LwU32);
void i2cWriteCtrl(LwU32, LwU8, LwU8);
LwU8 i2cReadCtrl(LwU32, LwU8);
void ReadSDA(LwU32, LwU8 *);
void ReadSCL(LwU32, LwU8 *);
void SetSCLLine(LwU32);
void ResetSCLLine(LwU32);
void SetSDALine(LwU32);
void ResetSDALine(LwU32);
LwU8 WaitHighSDALine(LwU32);
LwU8 WaitHighSCLLine(LwU32);
void i2cStart(LwU32);
void i2cStop(LwU32);
LwU8 i2cAck(LwU32);
void i2cInit(LwU32);
LwU8 i2cSendByte(LwU32, LwU8);
LW_STATUS i2cReceiveByte(LwU32, LwU8 *, LwU8);
LwU32 i2cWrite(    LwU32, LwU8, LwU16, LwU8 *, LwU16, LwU8 *);
LW_STATUS i2cRead(     LwU32, LwU8, LwU16, LwU8 *, LwU16, LwU8 *);
LW_STATUS i2cRead_EDDC(LwU32, LwU8, LwU8, LwU8,   LwU16, LwU8 *);
LwU32 i2cSend(     LwU32, LwU8, LwU16, LwU8 *, LwU16, LwU8 *, LwU32);
LwU32 i2cWrite_ALT(LwU32, LwU8, LwU16, LwU8 *, LwU16, LwU8 *);
LwU32 i2cRead_ALT( LwU32, LwU8, LwU16, LwU8 *, LwU16, LwU8 *);
LwU32 i2cRead_ALT2(LwU32, LwU8, LwU16, LwU8 *, LwU16, LwU8 *);

void i2cMenu(void);


#endif // _I2C_H_
