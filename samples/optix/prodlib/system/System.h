/*
 * Copyright (c) 2021, LWPU CORPORATION.  All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property and proprietary
 * rights in and to this software, related documentation and any modifications thereto.
 * Any use, reproduction, disclosure or distribution of this software and related
 * documentation without an express license agreement from LWPU Corporation is strictly
 * prohibited.
 *
 * TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED *AS IS*
 * AND LWPU AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS OR IMPLIED,
 * INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
 * PARTICULAR PURPOSE.  IN NO EVENT SHALL LWPU OR ITS SUPPLIERS BE LIABLE FOR ANY
 * SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES WHATSOEVER (INCLUDING, WITHOUT
 * LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS, BUSINESS INTERRUPTION, LOSS OF
 * BUSINESS INFORMATION, OR ANY OTHER PELWNIARY LOSS) ARISING OUT OF THE USE OF OR
 * INABILITY TO USE THIS SOFTWARE, EVEN IF LWPU HAS BEEN ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGES
 */

#pragma once

#include <string>

namespace prodlib {

// Return the name of the machine
std::string getHostName();

// Return number of CPU cores.
unsigned int getNumberOfCPUCores();

// Return the CPU clock rate
unsigned int getCPUClockRateInKhz();

// Return an estimate of the current available system memory
size_t getAvailableSystemMemoryInBytes();

// Return the total system memory
size_t getTotalSystemMemoryInBytes();

// Return the CPU name as reported by the operating system
std::string getCPUName();

// "Windows", "Linux", or "Mac"
std::string getPlatform();

// Return true if a given file exists.
bool fileExists( const char* file );

// Return true if a given file is writable.
bool fileIsWritable( const char* file );

// Check if a given directory exists.
bool dirExists( const char* path );

// Create the given directory, return false on error.
bool createDir( const char* path );

// Create the given directory and any intermediate directories, return false on error.
bool createDirectories( const char* path );

}  // end namespace prodlib
