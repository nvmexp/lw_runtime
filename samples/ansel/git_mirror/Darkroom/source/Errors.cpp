#pragma warning(disable: 4514 4711 4710)

#include "darkroom/Errors.h"

namespace darkroom
{
    // colwert error code to error string
    const char* errToString(Error error)
    {
        switch (error)
        {
            case Error::kSuccess:
                return "Success";
            case Error::kWrongSize:
                return "Image size is not correct";
            case Error::kImageDimsNotEqual:
                return "Image dimensions are not equal";
            case Error::kImageTooSmall:
                return "Image size is too small";
            case Error::kIlwalidArgument:
                return "Invalid argument";
            case Error::kIlwalidArgumentCount:
                return "Invalid argument count";
            case Error::kUnknownJobType:
                return "Failed to identify job type (panorama type) by the folder contents";
            case Error::kCouldntStartupTheProcess:
                return "Failed to start the process";
            case Error::kNotEnoughFreeSpace:
                return "Not enough free space";
            case Error::kTargetPathNotWriteable:
                return "Couldn't write to the path specified";
            case Error::kCouldntCreateFile:
                return "Couldn't create file for writing";
            case Error::kOperationFailed:
                return "Internal error (operation failed)";
            case Error::kOperationTimeout:
                return "Internal error (operation timeout)";
            case Error::kDownloadFailed:
                return "Downloading failed";
            case Error::kInstallFailed:
                return "Installation failed";
            case Error::kExceptionOclwred:
                return "Internal exception oclwred";
            case Error::kIlwalidData:
                return "Invalid data";
            default:
                return "Unknown error";
        }
    }
}
