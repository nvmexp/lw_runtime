/*
 * LWIDIA_COPYRIGHT_BEGIN
 *
 * Copyright 2010-2019 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * LWIDIA_COPYRIGHT_END
 */

#ifndef INCLUDED_XP_H
#define INCLUDED_XP_H

#define MODULE_HANDLE void*

void * GetDLLProc(MODULE_HANDLE ModuleHandle, const char *FuncName);

namespace Xp
{
   void * GetDLLProc(void * ModuleHandle, const char * FuncName);
}

const int OK = 0;

class RC
{
private:
    int m_rc;
public:
    RC() : m_rc(0) {}
    RC(int rc)                  { m_rc = rc; }
    enum
    {
        EXIT_OK
       ,SIM_IFACE_NOT_FOUND
       ,DLL_LOAD_FAILED
       ,PCI_DEVICE_NOT_FOUND
       ,SOFTWARE_ERROR
       ,CANNOT_ALLOCATE_MEMORY
    };
    operator     int() const    { return m_rc; }
};

#define CHECK_RC(f)             \
        do {                        \
            if (OK != (rc = (f)))   \
            {                       \
                return rc;          \
            }                       \
        } while (0)

#endif
