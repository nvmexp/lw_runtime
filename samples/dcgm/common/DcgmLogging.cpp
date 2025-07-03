#include <sys/stat.h>
#include <time.h>
#include <string>
#include <iostream>
#include <sstream>
#include "DcgmLogging.h"

/*
 * rotate logFileName.(i-1) to logFileName.i
 * rotate logFileName to logFileName.1
 */
void dcgmLoggingRotate(char *logFileName)
{
    struct stat statBuf;
    if (!logFileName || (stat(logFileName, &statBuf) != 0))
    {
        /* log file does not exist */
        return;
    }

    std::stringstream oldFilename, newFilename;

    for (int i = DCGM_MAX_LOG_ROTATE; i > 1; i--)
    {
        oldFilename.str("");
        newFilename.str("");

        newFilename << logFileName << "." << i;
        oldFilename << logFileName << "." << (i - 1);

        if ((stat(newFilename.str().c_str(), &statBuf) == 0) &&
            (remove(newFilename.str().c_str()) < 0))
        {
            perror("Error deleting file");
            return;
        }

        if ((stat(oldFilename.str().c_str(), &statBuf) == 0) &&
            (rename(oldFilename.str().c_str(), newFilename.str().c_str()) < 0))
        {
            perror("Error renaming file");
            return;
        }
    }

    if ((rename(logFileName, oldFilename.str().c_str()) < 0))
    {
        perror("Error renaming file");
    }
}

void dcgmLoggingInit(char *elwDebugLevel, char *elwDebugAppend, char *elwDebugFile, char *elwDebugFileRotate)
{
    char buf[1024];

    if (lwosGetElw(elwDebugFileRotate, buf, sizeof(buf)) == 0)
    {
        if ((strncmp(buf, "true", 4) == 0) &&
            (lwosGetElw(elwDebugFile, buf, sizeof(buf)) == 0))
        {
            dcgmLoggingRotate(buf);
        }
    }

    loggingInit(elwDebugLevel, elwDebugAppend, elwDebugFile);
}
