#pragma once

namespace darkroom
{
    enum class Error : int
    {
        kSuccess = 0,
        kWrongSize = -1,
        kImageDimsNotEqual = -2,
        kImageTooSmall = -3,
        kIlwalidArgument = -4,
        kIlwalidArgumentCount = -5,
        kUnknownJobType = -6,
        kCouldntStartupTheProcess = -7,
        kNotEnoughFreeSpace = -8,
        kTargetPathNotWriteable = -9,
        kCouldntCreateFile = -10,
        kOperationFailed = -11,
        kOperationTimeout = -12,
        kDownloadFailed = -13,
        kInstallFailed = -14,
        kExceptionOclwred = -15,
        kIlwalidData = -16
    };

    /* 
        - Colwerts error code to a human-readable string
        - Do not free or modify memory returned by this function
    */
    const char* errToString(Error error);
}
