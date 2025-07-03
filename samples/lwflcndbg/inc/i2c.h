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

#ifdef MACOS
 #define MACI2CDELAY 5000   // 5 usec
#else
 #define MACI2CDELAY 0
#endif //(MACOS)

#define RM_OK                    0x00000000
#define RM_ERROR                 0xFFFFFFFF

#define I2C_READCYCLE (U008)0x01

extern ICBENTRY ICBEntry[MAX_ICB_ENTRIES];

//
// state dump routines - I2C.c
//
U032 getI2CCRTCOffset(void);
void i2cWriteWrIndex(U032, U008);
U008 i2cReadWrIndex(U032);
U008 i2cReadStatusIndex(U032);
U008 i2cHardwareInit(U032);
VOID i2cWriteCtrl(U032, U008, U008);
U008 i2cReadCtrl(U032, U008);
VOID ReadSDA(U032, U008 *);
VOID ReadSCL(U032, U008 *);
VOID SetSCLLine(U032);
VOID ResetSCLLine(U032);
VOID SetSDALine(U032);
VOID ResetSDALine(U032);
U008 WaitHighSDALine(U032);
U008 WaitHighSCLLine(U032);
VOID i2cStart(U032);
VOID i2cStop(U032);
U008 i2cAck(U032);
VOID i2cInit(U032);
U008 i2cSendByte(U032, U008);
U032 i2cReceiveByte(U032, U008 *, U008);
U032 i2cWrite(    U032, U008, U016, U008 *, U016, U008 *);
U032 i2cRead(     U032, U008, U016, U008 *, U016, U008 *);
U032 i2cRead_EDDC(U032, U008, U008, U008,   U016, U008 *);
U032 i2cSend(     U032, U008, U016, U008 *, U016, U008 *, U032);
U032 i2cWrite_ALT(U032, U008, U016, U008 *, U016, U008 *);
U032 i2cRead_ALT( U032, U008, U016, U008 *, U016, U008 *);
U032 i2cRead_ALT2(U032, U008, U016, U008 *, U016, U008 *);

VOID i2cMenu(void);


#endif // _I2C_H_
