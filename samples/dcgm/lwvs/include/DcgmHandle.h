#pragma once

#include <string>

#include "dcgm_agent.h"
#include "dcgm_structs.h"

class DcgmHandle
{
public:
    DcgmHandle();
    ~DcgmHandle();

    /*
     * Translate the DCGM return code to a human-readable string
     *
     * @param ret      IN : the return code to translate to a string
     * @return
     * A string translation of the error code : On Success
     * 'Unknown error from DCGM: <ret>'       : On Failure
     */
    std::string  RetToString(dcgmReturn_t ret);

    /*
     * Establishes a connection with DCGM, saving the handle internally. This MUST be called before using 
     * the other methods
     * 
     * @param dcgmHostname    IN : the hostname of the lw-hostengine we should connect to. If "" is specified,
     *                             will connect to localhost.
     * @return
     * DCGM_ST_OK : On Success
     * DCGM_ST_*  : On Failure
     */
    dcgmReturn_t ConnectToDcgm(const std::string &dcgmHostname);

    /*
     * Get a string representation of the last error
     *
     * @return
     * The last DCGM return code as a string, or "" if the last call was successful
     */
    std::string  GetLastError();

    /*
     * Return the DCGM handle - it will still be owned by this class
     *
     * @return
     * the handle or 0 if it hasn't been initialized
     */
    dcgmHandle_t GetHandle();

    /*
     * Destroys the allocated members of this class if they exist
     */
    void         Cleanup();

private:
    dcgmReturn_t m_lastReturn;   // The last return code from DCGM
    dcgmHandle_t m_handle;       // The handle for the DCGM connection
};

