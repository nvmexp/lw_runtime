/*
 * LWIDIA_COPYRIGHT_BEGIN
 *
 * Copyright 2009 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * LWIDIA_COPYRIGHT_END
 */

#ifndef __PRIO_H__
#define __PRIO_H__

#include <stdio.h>
#include <time.h>
#include "jspubtd.h"

#if defined(XP_UNIX)

#include <dirent.h>
typedef DIR PRDir;

#else

#include <Windows.h>

typedef struct
{
    HANDLE hdir;
    int32  firstFile;
    WIN32_FIND_DATAW FindFileData;
} PRDir;
#endif


// Mask bits for opening files
#define PR_RDONLY              0x01
#define PR_WRONLY              0x02
#define PR_RDWR                0x04
#define PR_CREATE_FILE         0x08
#define PR_APPEND              0x10
#define PR_TRUNCATE            0x20
#define PR_BINARY              0x40

// Explode type for PR_ExplodeTime
#define PR_LocalTimeParameters 0

// From values for PR_Seek                             
typedef enum
{
    PR_SEEK_LWR,
    PR_SEEK_SET
} PRSeekFrom;

// Access modes for PR_Access
typedef enum
{
    PR_ACCESS_EXISTS,
    PR_ACCESS_READ_OK,
    PR_ACCESS_WRITE_OK
} PRAccessMode;

// Skip values for PR_ReadDir                             
typedef enum
{
    PR_SKIP_BOTH,
    PR_SKIP_NONE
} PRReadDirMode;

// File type information for JS files.
typedef enum
{
    PR_FILE_FILE,
    PR_FILE_DIRECTORY,
    PR_FILE_OTHER
} PRFileType;

// A file descriptor is just a file
typedef FILE PRFileDesc;

// Compressed time is a time_t always
typedef time_t PRCompressedTime;

// File info structure for JS files.
typedef struct
{
    PRFileType type;
    int32 size;
    PRCompressedTime creationTime;
    PRCompressedTime modifyTime;
} PRFileInfo;

// Directory entry information for JS files.
typedef struct
{
    char *name;
} PRDirEntry;

// Exploded time information for JS files.
typedef struct
{
    int32 tm_year;
    int32 tm_month;
    int32 tm_mday;
    int32 tm_hour;
    int32 tm_min;
    int32 tm_sec;
} PRExplodedTime;

// File access functions
PRFileDesc * PR_Open(char *path, int32 mask, int32 perms);
int32 PR_Read(PRFileDesc *pFile, void *buf, int32 len);
int32 PR_Write(PRFileDesc *pFile, void *buf, int32 len);
int32 PR_Seek(PRFileDesc *pFile, int32 len, PRSeekFrom from);
int32 PR_Sync(PRFileDesc *pFile);
int32 PR_GetOpenFileInfo(PRFileDesc *pFile, PRFileInfo *pInfo);
int32 PR_Close(PRFileDesc *pFile);

// File system access functions
int32 PR_Access(char *path, PRAccessMode mode);
int32 PR_GetFileInfo(char *path, PRFileInfo *pInfo);
int32 PR_Delete(char *path);
int32 PR_Rename(char *src, char *dest);
int32 PR_MkDir(char *path, int32 perms);
int32 PR_RmDir(char *path);
PRDir * PR_OpenDir(char *path);
PRDirEntry * PR_ReadDir(PRDir *pDir, PRReadDirMode skip);
int32 PR_CloseDir(PRDir *pDir);

// Utility functions
void PR_ExplodeTime(PRCompressedTime compressedTime, int32 explodeType, PRExplodedTime *pExplodedTime);

#endif // __PRIO_H__

