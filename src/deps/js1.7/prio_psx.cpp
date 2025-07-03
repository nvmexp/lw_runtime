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

#include "prio.h"
#include "prerror.h"
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <string.h>

// This file provides a MODS implementation for Posix platforms  
// (Dos, Macosx, linux) for all the PR_ functions (normally from the  
// NSPR library) that allow the JS file objects to function

// Link list structure for connecting underlying directory entries with
// the JS structure needed.  JS wants the name field to be "name" and
// Posix uses "d_name"
struct OpenDir
{
    PRDir          *pDir;
    PRDirEntry      lwrDirEntry;
    struct OpenDir *pNext;
};

// Link list for open directories
struct OpenDir *m_pOpenDirs = NULL;

//-----------------------------------------------------------------------------
// Colwert a struct stat to a PRFileInfo structure
static void FStatToFileInfo(struct stat *pStat, PRFileInfo *pInfo)
{
    if (S_ISREG(pStat->st_mode)) 
    {
        pInfo->type = PR_FILE_FILE;
    }
    else if (S_ISDIR(pStat->st_mode)) 
    {
        pInfo->type = PR_FILE_DIRECTORY;
    }
    else
    {
        pInfo->type = PR_FILE_OTHER;
    }
    pInfo->size = (int32)pStat->st_size;
    pInfo->creationTime = pStat->st_ctime;
    pInfo->modifyTime = pStat->st_mtime;
}

//-----------------------------------------------------------------------------
// File Access Functions
//-----------------------------------------------------------------------------
PRFileDesc * PR_Open(char *path, int32 mask, int32 perms)
{
    char mode[4] = "r\0\0";

    if (mask & PR_RDWR) 
    {
        if (mask & PR_APPEND)
        {
            strcpy( mode, "a+" );
        }
        else if (!(mask & PR_CREATE_FILE)) 
        {
            strcpy( mode, "r+" );
        }
        else
        {
            strcpy( mode, "w+" );
        }
    }
    else if (mask & PR_WRONLY) 
    {
        if (mask & PR_APPEND)
        {
            strcpy( mode, "a" );
        }
        else if (!(mask & PR_CREATE_FILE)) 
        {
            strcpy( mode, "w" );
        }
        else
        {
            strcpy( mode, "w+" );
        }
    }
    else
    {
        strcpy( mode, "r" );
    }
    
    if ( mask & PR_BINARY )
        strcat( mode, "b" );

    return fopen(path, mode);
}

//-----------------------------------------------------------------------------
int32 PR_Read(PRFileDesc *pFile, void *buf, int32 len)
{
    return fread(buf, 1, len, pFile);
}

//-----------------------------------------------------------------------------
int32 PR_Write(PRFileDesc *pFile, void *buf, int32 len)
{
    return fwrite(buf, 1, len, pFile);
}

//-----------------------------------------------------------------------------
int32 PR_Seek(PRFileDesc *pFile, int32 len, PRSeekFrom from)
{
    int fromPos = SEEK_LWR;
    switch (from)
    {
        case PR_SEEK_LWR:
            fromPos = SEEK_LWR;
            break;
        case PR_SEEK_SET:
            fromPos = SEEK_SET;
            break;
        default:
            return PR_FAILURE;
    }

    fseek(pFile, len, fromPos);
    return ftell(pFile);
}

//-----------------------------------------------------------------------------
int32 PR_Sync(PRFileDesc *pFile)
{
    int status;
    status = fflush(pFile);
    if (status!=0)
      return PR_FAILURE;

    // Force a flush of the open file to disk by getting a duplicate
    // file descriptor, then closing the duplicate.
    // It's SLOW, but it is still much faster than closing and reopening.
    //
    // This is worthwhile because DOS crashes so often.  This way the log
    // file is up to date after the reboot.
    close(dup(fileno(pFile)));

    return PR_SUCCESS;
}

//-----------------------------------------------------------------------------
int32 PR_GetOpenFileInfo(PRFileDesc *pFile, PRFileInfo *pInfo)
{
    struct stat fStat;
    int ret;

    ret = fstat(fileno(pFile), &fStat);
    if (ret == 0) 
    {
        FStatToFileInfo(&fStat, pInfo);
    }

    return (ret == 0) ? PR_SUCCESS : PR_FAILURE;
}

//-----------------------------------------------------------------------------
int32 PR_Close(PRFileDesc *pFile)
{
    return (fclose(pFile) == 0) ? PR_SUCCESS : PR_FAILURE;
}

//-----------------------------------------------------------------------------
// File System Access Functions
//-----------------------------------------------------------------------------
int32 PR_Access(char *path, PRAccessMode mode)
{
    int accessMode;

    switch (mode) 
    {
        case PR_ACCESS_EXISTS:
            accessMode = F_OK;
            break;
        case PR_ACCESS_READ_OK:
            accessMode = R_OK;
            break;
        case PR_ACCESS_WRITE_OK:
            accessMode = W_OK;
            break;
        default:
            return PR_FAILURE;
    }

    return (access(path, accessMode) == 0) ? PR_SUCCESS : PR_FAILURE;
}

//-----------------------------------------------------------------------------
int32 PR_GetFileInfo(char *path, PRFileInfo *pInfo)
{
    struct stat fStat;
    int ret;

    ret = stat(path, &fStat);
    if (ret == 0) 
    {
        FStatToFileInfo(&fStat, pInfo);
    }

    return (ret == 0) ? PR_SUCCESS : PR_FAILURE;
}

//-----------------------------------------------------------------------------
int32 PR_Delete(char *path)
{
    return (remove(path) == 0) ? PR_SUCCESS : PR_FAILURE;
}

//-----------------------------------------------------------------------------
int32 PR_Rename(char *src, char *dest)
{
    return (rename(src, dest) == 0) ? PR_SUCCESS : PR_FAILURE;
}

//-----------------------------------------------------------------------------
int32 PR_MkDir(char *path, int32 perms)
{
    return (mkdir(path, perms) == 0) ? PR_SUCCESS : PR_FAILURE;
}

//-----------------------------------------------------------------------------
int32 PR_RmDir(char *path)
{
    return (rmdir(path) == 0) ? PR_SUCCESS : PR_FAILURE;
}

//-----------------------------------------------------------------------------
PRDir * PR_OpenDir(char *path)
{
    PRDir *pDir = opendir(path);
    if (pDir != NULL) 
    {
        // Create a link list entry for the new directory
        struct OpenDir *pNewDir = static_cast<OpenDir *>(malloc(sizeof(struct OpenDir)));
        pNewDir->pDir = pDir;
        pNewDir->lwrDirEntry.name = NULL;
        pNewDir->pNext = m_pOpenDirs;
        m_pOpenDirs = pNewDir;
    }

    return pDir;
}

//-----------------------------------------------------------------------------
PRDirEntry * PR_ReadDir(PRDir *pDir, PRReadDirMode skip)
{
    // Find the link list entry for the directory
    struct OpenDir *pDirStruct = m_pOpenDirs;
    while ((pDirStruct != NULL) && (pDirStruct->pDir != pDir)) 
    {
        pDirStruct = pDirStruct->pNext;
    }
    if (pDirStruct == NULL) 
        return NULL;

    struct dirent *pDirEnt;
    int bDone = 0;
    while (!bDone)
    {
        pDirEnt = readdir(pDir);
        if (!pDirEnt)
        {
            return NULL;
        }

        if ((skip == PR_SKIP_NONE) ||
            (strcmp(pDirEnt->d_name, ".") && strcmp(pDirEnt->d_name, "..")))
        {
            bDone = 1;
        }
    }

    pDirStruct->lwrDirEntry.name = pDirEnt->d_name;
    return &(pDirStruct->lwrDirEntry);
}

//-----------------------------------------------------------------------------
int32 PR_CloseDir(PRDir *pDir)
{
    // Find and delete the link list entry for the directory
    struct OpenDir *pDirStruct = m_pOpenDirs;
    struct OpenDir *pPrevDirStruct = NULL;
    while ((pDirStruct != NULL) && (pDirStruct->pDir != pDir)) 
    {
        pPrevDirStruct = pDirStruct;
        pDirStruct = pDirStruct->pNext;
    }

    if (pDirStruct != NULL) 
    {
        if (pDirStruct == m_pOpenDirs) 
        {
            m_pOpenDirs = pDirStruct->pNext;
        }
        else 
        {
            pPrevDirStruct->pNext = pDirStruct->pNext;
        }

        free(pDirStruct);
        return (closedir(pDir) == 0) ? PR_SUCCESS : PR_FAILURE;
    }

    return PR_SUCCESS;
}

//-----------------------------------------------------------------------------
// Utility Functions
//-----------------------------------------------------------------------------
void PR_ExplodeTime(PRCompressedTime compressedTime, int32 explodeType, PRExplodedTime *pExplodedTime)
{
    struct tm * pLocalTime = localtime(&compressedTime);

    pExplodedTime->tm_year = pLocalTime->tm_year + 1900;
    pExplodedTime->tm_month = pLocalTime->tm_mon;
    pExplodedTime->tm_mday = pLocalTime->tm_mday;
    pExplodedTime->tm_hour = pLocalTime->tm_hour;
    pExplodedTime->tm_min = pLocalTime->tm_min;
    pExplodedTime->tm_sec = pLocalTime->tm_sec;
}

