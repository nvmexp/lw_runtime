// Copyright (c) 2017, LWPU CORPORATION.
// TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED
// *AS IS* AND LWPU AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS
// OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY
// AND FITNESS FOR A PARTICULAR PURPOSE.  IN NO EVENT SHALL LWPU OR ITS SUPPLIERS
// BE LIABLE FOR ANY SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES
// WHATSOEVER (INCLUDING, WITHOUT LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS,
// BUSINESS INTERRUPTION, LOSS OF BUSINESS INFORMATION, OR ANY OTHER PELWNIARY LOSS)
// ARISING OUT OF THE USE OF OR INABILITY TO USE THIS SOFTWARE, EVEN IF LWPU HAS
// BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES


#include <Control/ErrorManager.h>

#include <Exceptions/ExceptionHelpers.h>
#include <corelib/misc/String.h>
#include <prodlib/exceptions/Assert.h>
#include <prodlib/system/Knobs.h>
#include <prodlib/system/Logger.h>

#include <sstream>

using namespace optix;
using namespace corelib;
using namespace prodlib;


namespace {
// clang-format off
  Knob<bool> k_logErrors( RT_DSTRING("errorManager.logErrors"), true, RT_DSTRING("Log errors in addition to any reporting the app may make."));
// clang-format on
}

ErrorManager::ErrorManager()
    : m_lastErrorCode( RT_SUCCESS )
{
}

ErrorManager::~ErrorManager()
{
}

void ErrorManager::setErrorString( const std::string& funcname, const Exception& e )
{
    std::ostringstream out;
    out << "Function \"" << funcname << "\" caught exception: " << e.getDescription();

    if( !e.getStacktrace().empty() )
    {
        out << "\n================================================================================\n";
        out << "Backtrace:\n" << e.getStacktrace();
        out << "\n================================================================================\n";
    }

    m_lastErrorString = out.str();
    m_lastErrorCode   = getRTresultFromException( &e );

    if( k_logErrors.get() )
    {
        // log the event as "fatal" if it's caused by an assertion
        if( dynamic_cast<const prodlib::AssertionFailure*>( &e ) )
            lfatal << m_lastErrorString << std::endl;
        else
            lerr << m_lastErrorString << std::endl;
    }
}

void ErrorManager::setErrorString( const std::string& funcname, const std::exception& e )
{
    std::ostringstream out;
    out << "Function \"" << funcname << "\" caught C++ standard exception: " << e.what();
    m_lastErrorString = out.str();
    m_lastErrorCode   = RT_ERROR_UNKNOWN;
    lerr << m_lastErrorString;
}

void ErrorManager::setErrorString( const std::string& funcname, const std::string& error, RTresult errorCode )
{
    std::ostringstream out;
    out << "Function \"" << funcname << "\" detected error: " << error;
    m_lastErrorString = out.str();
    m_lastErrorCode   = errorCode;
    lerr << m_lastErrorString << std::endl;
}

std::string ErrorManager::getErrorString( RTresult code ) const
{
    std::ostringstream out;
    out << getErrorString_static( code );
    // We ensure that the code they are asking about matches the one that generated the error string.
    // This makes the mechanism mildly foolproof.  We do not add details if there is not a match.
    if( code != RT_SUCCESS && code == m_lastErrorCode )
        out << " (Details: " << m_lastErrorString << ")";

    return out.str();
}

const char* ErrorManager::getErrorString_static( RTresult code )
{
    switch( code )
    {
        case RT_SUCCESS:
            // Never add details for this one
            return "Success (no errors)";

        case RT_TIMEOUT_CALLBACK:
            return "Application's timeout callback requested the API call to terminate unfinished";

        case RT_ERROR_CONTEXT_CREATION_FAILED:
            return "Context creation failed";

        case RT_ERROR_OPTIX_NOT_LOADED:
            return "Failed to load OptiX library";

        case RT_ERROR_DENOISER_NOT_LOADED:
            return "Failed to load Denoiser library";

        case RT_ERROR_SSIM_PREDICTOR_NOT_LOADED:
            return "Failed to load SSIM predictor library";

        case RT_ERROR_DRIVER_VERSION_FAILED:
            return "Failed to retrieve the display driver version";

        case RT_ERROR_ILWALID_CONTEXT:
            return "Invalid context";

        case RT_ERROR_ILWALID_VALUE:
            return "Invalid value";

        case RT_ERROR_MEMORY_ALLOCATION_FAILED:
            return "Memory allocation failed";

        case RT_ERROR_TYPE_MISMATCH:
            return "Type mismatch";

        case RT_ERROR_VARIABLE_NOT_FOUND:
            return "Variable not found";

        case RT_ERROR_VARIABLE_REDECLARED:
            return "Variable redeclared";

        case RT_ERROR_ILLEGAL_SYMBOL:
            return "Illegal symbol";

        case RT_ERROR_ILWALID_SOURCE:
            return "Parse error";

        case RT_ERROR_VERSION_MISMATCH:
            return "Version mismatch";

        case RT_ERROR_OBJECT_CREATION_FAILED:
            return "Object creation failed";

        case RT_ERROR_LAUNCH_FAILED:
            return "Launch failed";

        case RT_ERROR_NO_DEVICE:
            return "A supported LWPU GPU could not be found";

        case RT_ERROR_ILWALID_DEVICE:
            return "Invalid device";

        case RT_ERROR_ILWALID_IMAGE:
            return "Invalid image";

        case RT_ERROR_FILE_NOT_FOUND:
            return "File not found";

        case RT_ERROR_ALREADY_MAPPED:
            return "Already mapped";

        case RT_ERROR_ILWALID_DRIVER_VERSION:
            return "A supported LWPU driver cannot be found. Please see the release notes for supported drivers.";

        case RT_ERROR_RESOURCE_NOT_REGISTERED:
            return "An OptiX resource was not registered properly";

        case RT_ERROR_RESOURCE_ALREADY_REGISTERED:
            return "An OptiX resource has already been registered";

        case RT_ERROR_NOT_SUPPORTED:
            return "The requested operation is not supported";

        case RT_ERROR_CONNECTION_FAILED:
            return "Connection to the remote device failed";

        case RT_ERROR_AUTHENTICATION_FAILED:
            return "Authentication failed. Please check your login credentials to the remote device";

        case RT_ERROR_CONNECTION_ALREADY_EXISTS:
            return "A connection to this remote device already exists";

        case RT_ERROR_NETWORK_LOAD_FAILED:
            return "Network library not found. Please check that libdice.so/.dll is accessible";

        case RT_ERROR_NETWORK_INIT_FAILED:
            return "There was an error initializing the network library. Please check that libdice.so/.dll is "
                   "accessible";

        case RT_ERROR_CLUSTER_NOT_RUNNING:
            return "There is no cluster running";

        case RT_ERROR_CLUSTER_ALREADY_RUNNING:
            return "Your user has another cluster already running in this remote device";

        case RT_ERROR_INSUFFICIENT_FREE_NODES:
            return "There are not sufficient free nodes in the remote device to allocate the requested number of nodes";

        case RT_ERROR_ILWALID_GLOBAL_ATTRIBUTE:
            return "Unknown global attribute";

        case RT_ERROR_DATABASE_FILE_PERMISSIONS:
            return "OptiX was unable to open the disk cache with sufficient privileges. "
                   "Please make sure the database file is writeable by the current user.";

        case RT_ERROR_UNKNOWN:
            return "Unknown error";
    }

    return "Unknown error";
}
