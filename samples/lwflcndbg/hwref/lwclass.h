//*****************************************************
//
// lwwatch WinDbg Extension
// retodd@lwpu.com - 2.1.2002
// lwclass.h
//
//*****************************************************

#ifndef _LWCLASS_H_
#define _LWCLASS_H_

//
// Also see printClassName in print.c
//

//
// Cases where the HW class number does not match the SW architecture class number
//
#define LW30_RANKINE_PRIMITIVE_HW_CLASSNUM 0x397
#define LW34_RANKINE_PRIMITIVE_HW_CLASSNUM 0x697
#define LW35_RANKINE_PRIMITIVE_HW_CLASSNUM 0x497

//
// LW Classes
//
#define MAX_CLASS_NUMBER_SUPPORTED  0x9F

#endif // _LWCLASS_H_
