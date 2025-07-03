/*
 * Copyright (c) 2008 - 2009 LWPU Corporation.  All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#ifndef __SUMS_H
#define __SUMS_H

int checksumFileOpen(FILE **outFile, const char *outputDir, const char *outputFile);
void checksumFileClose(FILE **outFile);
void checksumGoldFileRead(const char *goldDir, const char *fileName, unsigned altGold);

#endif // __SUMS_H
