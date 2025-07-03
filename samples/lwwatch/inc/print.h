//*****************************************************
//
// lwwatch WinDbg Extension
// retodd@lwpu.com - 2.1.2002
// print.h
//
//*****************************************************

#ifndef _PRINT_H_
#define _PRINT_H_

#include "os.h"

//
// print routines - print.c
//
void    printBuffer(char *buffer, LwU32 length, LwU64 offset, LwU8 size);
void    printData(PhysAddr addr, LwU32 sizeInBytes);
void    printDataByType(PhysAddr addr, LwU32 sizeInBytes, MEM_TYPE memoryType, LwU32 numColumns);
void    printDataColumns(PhysAddr addr, LwU32 sizeInBytes, LwU32 numColumns);
LW_STATUS    printClassName(LwU32 classNum);

#endif // _PRINT_H_
