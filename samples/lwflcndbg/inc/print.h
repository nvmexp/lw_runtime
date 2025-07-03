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
VOID    printBuffer(char *buffer, U032 length, LwU64 offset, U008 size);
VOID    printBufferEx(char *buffer, U032 length, LwU64 offset, U008 size, LwU64 base);
VOID    printData(PhysAddr addr, U032 sizeInBytes);
VOID    printDataByType(PhysAddr addr, U032 sizeInBytes, MEM_TYPE memoryType, U032 numColumns);
VOID    printDataColumns(PhysAddr addr, U032 sizeInBytes, U032 numColumns);
U032    printClassName(U032 classNum);

#endif // _PRINT_H_
