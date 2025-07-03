 /************************ BEGIN COPYRIGHT NOTICE ***************************\
|*                                                                           *|
|* Copyright 2003-2010 by LWPU Corporation.  All rights reserved.  All     *|
|* information contained herein is proprietary and confidential to LWPU    *|
|* Corporation.  Any use, reproduction, or disclosure without the written    *|
|* permission of LWPU Corporation is prohibited.                           *|
|*                                                                           *|
 \************************** END COPYRIGHT NOTICE ***************************/

#if !defined(_H_LWPERFMONTABLECOMPRESS_GEN_H_)
#define _H_LWPERFMONTABLECOMPRESS_GEN_H_

// colwerts GPUSignalTable into CompressedGPUSignalTable 

bool lwCompressPerfmonTables(void **pPMSignalDataBaseData, LwU32 *pPMSignalDataBaseSize, const LwSignalDataBase::GPUSignalTable **ppPmSignalTable);

#endif // defined(_H_LWPERFMONTABLECOMPRESS_GEN_H_)
