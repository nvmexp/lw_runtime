/*
 * Copyright 2003-2012 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#ifndef _PARSE_H_
#define _PARSE_H_

char *getToken(char *ptr, char *token, char **tokenPtr);
int   parseCmd(char *args, char *op, int numParam, char **param);

#endif // _PARSE_H_

