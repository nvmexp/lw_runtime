#pragma once

// Macro to export symbols on windows / gcc
#if defined _WIN32
    // No longer needed in windows, since we use def file or -DCMAKE_WINDOWS_EXPORT_ALL_SYMBOLS=TRUE
    #define EXPORT_SYMBOL
#else
    #define EXPORT_SYMBOL __attribute__ ((visibility ("default")))
#endif
#ifdef LWTENSOR_EXPOSE_INTERNAL
    #define EXPORT_INTERNAL_SYMBOL EXPORT_SYMBOL
#else
    #define EXPORT_INTERNAL_SYMBOL
#endif
