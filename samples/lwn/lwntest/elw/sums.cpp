/*
 * Copyright (c) 2008 - 2015 LWPU Corporation.  All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#include "ogtest.h"

#include "elw.h"
#include "str_util.h"
#include "sums.h"
#include "md5.h"
#include <ctype.h>

constexpr int CHECKSUM_MAX_LINELEN = 1024;

#ifndef crc_check
extern int shownotsup;
// IsTestSupported()
#include "testloop.h"
#endif

// For some reason, Win32 has problems fopening files
// with slashes when the current directory is a \\hostname
// directory (as best I can figure). -mjk
static void slash2backslash4win32(char *string)
{
#ifdef _WIN32
    while (*string != '\0') {
        if (*string == '/') {
            *string = '\\';
        }
        string++;
    }
#else
    // No need for this silliness.
#endif
}

#ifndef crc_check

static int __LWOG_MAX(int a, int b) {
    if (a>b) return a;
    return b;
}

// Read a line and strips any comments out.
// Returns 1 if there's any data left after comment removal, 0 otherwise.
static int commentStrip(char *line) {
    size_t i, len = strlen(line);
    char foundNonSpace = 0;
    if (len == 0) {
        return 0;
    }
    for (i = 0; i < len; i++) {
        if (line[i] == '#' || // Bash style # hash comment.
            (line[i] == '/' && line[i+1] == '/')) { // C++ style // comment.
            line[i] = '\0';
            break;
        }
        if (!isspace(line[i])) {
            foundNonSpace = 1;
        }
    }
    return foundNonSpace;
}

/****************************************************************************
 * checksumGoldFileRead
 * 
 * Reads the file specified by "fileName" into the TestStruct
 * 
 * This relies on the TestStruct being sorted by name.
 *
 * If a test appears multiple times in the checksum file, the checksum 
 * associated with the last one wins.  Since the checksum file is
 * lwmmulative across lwogtest runs, this means that the most recently run
 * instance of each test will be the one that is used.
 ****************************************************************************/
void checksumGoldFileRead(const char *goldDir, const char *fileName, unsigned altGold)
{
    int lo, hi, mid, cmp, found;
    unsigned char md5[MAX_MD5_GOLDS][16];
    char line[CHECKSUM_MAX_LINELEN];
    size_t namelen;
    FILE *f;
    char *name;
    int done = 0;

    if (!goldDir) {
        return;
    }

    // Allocate at least 256 bytes for the test name (blech) or more
    //  if needed for goldDir + filename + '/' + '\0'
    namelen = __LWOG_MAX(256, (int)strlen(goldDir) + (int)strlen(fileName) + 2);
    name = static_cast<char*>(__LWOG_MALLOC(namelen));

    lwog_snprintf(name, namelen, "%s/%s", goldDir, fileName);        
    slash2backslash4win32(name);

    {
        // Open the checksum file and read line-by-line.
        if ((f = fopen(name, "r")) == NULL) {
            goto fail;
        }
        done = feof(f);
    }

    while (!done) {
        {
            if (!fgets(line, CHECKSUM_MAX_LINELEN, f)) {
                break;
            }
        }

        if (!commentStrip(line)) {
            continue;
        }

        // This branching based on checksum type stuff is not as modular
        // as it could be.  If we ever add yet another checksum type
        // such as SHA1 to lwogtest, we may want to instead use two
        // function pointers.  One to read the value from the file, and
        // another to write the value into the TestStruct.
        memset(md5, 0, sizeof md5);
        if(!md5Read(line, name, md5)) {
            printf("lwogtest: warning: invalid md5 line format.\n");
            printf("%s", line);
            continue;
        }
        
        // Do a binary search over the entire test list using the test
        // name.
        found = 0;
        lo = 0;
        hi = TestCount - 1;
        do {
            OGTEST *midTest;
            mid = (lo + hi) / 2;
            midTest = GET_TEST(mid);
            cmp = stricmp(TEST_PROF_NAME(midTest), name);
            if (cmp == 0) {

                // If we have a match, save away the checksum.
                found = 1;
                memcpy(midTest->md5, md5, sizeof md5);
                midTest->md5ValidMask |= 1;
                break;
            } else if (cmp < 0) {
                lo = mid+1;
            } else {
                hi = mid-1;
            }
        } while (lo <= hi);

        if (!found) {
            printf("Unrecognized test '%s' found in file "
                   "%s\n", name, fileName);
        }

        {
            done = feof(f);
        }
    }

    {
        fclose(f);
    }

    __LWOG_FREE(name);

    return;

fail:
    printf("lwntest: warning: unable to open \"%s\" checksum file "
           "for reading.\n", name);

    __LWOG_FREE(name);
}

#endif //crc_check

/***********************************************************************
 *  checksumFileOpen
 *
 *  Sets and opens the file to which we're writing.
 ***********************************************************************/
int checksumFileOpen(FILE **outFile, const char *outputDir, const char *outputFile) {
    char name[300];

    if (outputDir) {
        lwog_snprintf(name, 300, "%s/%s", outputDir, outputFile);        
        slash2backslash4win32(name);
        *outFile = fopen(name, "a");
        if (!(*outFile)) {
            printf("warning: unable to open \"%s\" checksum file for writing.\n", name);
            return 0;
        }
    }

    return 1;
}


/***********************************************************************
 *  checksumFileClose
 *
 *  Closes the file to which we're writing.
 ***********************************************************************/
void checksumFileClose(FILE **outFile) {
    assert(*outFile);

    if(*outFile) {
        fclose(*outFile);
        *outFile = NULL;
    }
}
