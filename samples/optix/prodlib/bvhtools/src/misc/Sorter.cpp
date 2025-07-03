// Copyright LWPU Corporation 2013
// TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED
// *AS IS* AND LWPU AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS
// OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY
// AND FITNESS FOR A PARTICULAR PURPOSE.  IN NO EVENT SHALL LWPU OR ITS SUPPLIERS
// BE LIABLE FOR ANY SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES
// WHATSOEVER (INCLUDING, WITHOUT LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS,
// BUSINESS INTERRUPTION, LOSS OF BUSINESS INFORMATION, OR ANY OTHER PELWNIARY LOSS)
// ARISING OUT OF THE USE OF OR INABILITY TO USE THIS SOFTWARE, EVEN IF LWPU HAS
// BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES

#include "Sorter.hpp"
#include "SorterKernels.hpp"
#include <prodlib/exceptions/IlwalidValue.h>
#include <prodlib/exceptions/Assert.h>
#include <corelib/misc/Qsort.h>
#include <algorithm>
#include <memory.h>


using namespace prodlib::bvhtools;
using namespace prodlib;

//------------------------------------------------------------------------

void Sorter::configure(const Config& cfg)
{
    // Check for errors.

    RT_ASSERT(cfg.numItems >= 0);
    RT_ASSERT(cfg.inKeys.getNumBytes() >= cfg.numItems * cfg.bytesPerKey);
    RT_ASSERT(cfg.ilwalues.getNumBytes() >= cfg.numItems * cfg.bytesPerValue);
    RT_ASSERT(cfg.outKeys.getNumBytes() >= cfg.numItems * cfg.bytesPerKey);
    RT_ASSERT(cfg.outValues.getNumBytes() >= cfg.numItems * cfg.bytesPerValue);

    if (cfg.bytesPerKey != 4 && cfg.bytesPerKey != 8)
        throw IlwalidValue( RT_EXCEPTION_INFO, "Unsupported bytesPerKey!", cfg.bytesPerKey );

    if (cfg.bytesPerValue != 4)
        throw IlwalidValue( RT_EXCEPTION_INFO, "Unsupported bytesPerValue!", cfg.bytesPerValue );

    // Set config.

    m_cfg = cfg;

    // Set temp buffer size.
    
    if (!m_cfg.lwca || m_cfg.numItems <= 0)
        m_cfg.tempBuffer.setNumBytes(0);
    else
        m_cfg.tempBuffer.setNumBytes(radixSortTempSize(
            m_cfg.bytesPerKey, m_cfg.bytesPerValue, m_cfg.numItems));
}

//------------------------------------------------------------------------

void Sorter::execute(void)
{
    if (m_cfg.numItems <= 0)
        return;

    if (m_cfg.lwca)
    {
        m_cfg.lwca->beginTimer(getName());
        execDevice();
        m_cfg.lwca->endTimer();
    }
    else
    {
        execHost();
    }

    m_cfg.tempBuffer.markAsUninitialized();
}

//------------------------------------------------------------------------

void Sorter::execDevice(void)
{
    // Ilwoke radixSort
    if (m_cfg.bytesPerKey == 4)
    {
        radixSort(
            m_cfg.tempBuffer.reinterpretRaw().writeDiscardLWDA(),
            m_cfg.tempBuffer.getNumBytes(),
            m_cfg.inKeys.reinterpret<unsigned int>().readWriteLWDA(),
            m_cfg.outKeys.reinterpret<unsigned int>().writeDiscardLWDA(),
            m_cfg.ilwalues.reinterpret<unsigned int>().readWriteLWDA(),
            m_cfg.outValues.reinterpret<unsigned int>().writeDiscardLWDA(),
            m_cfg.numItems,
            m_cfg.lwca->getDefaultStream());
    }
    else
    {
        RT_ASSERT(m_cfg.bytesPerKey == 8);
        radixSort(
          m_cfg.tempBuffer.reinterpretRaw().writeDiscardLWDA(),
          m_cfg.tempBuffer.getNumBytes(),
          m_cfg.inKeys.reinterpret<unsigned long long>().readWriteLWDA(),
          m_cfg.outKeys.reinterpret<unsigned long long>().writeDiscardLWDA(),
          m_cfg.ilwalues.reinterpret<unsigned int>().readWriteLWDA(),
          m_cfg.outValues.reinterpret<unsigned int>().writeDiscardLWDA(),
          m_cfg.numItems,
          m_cfg.lwca->getDefaultStream());
    }
}

//------------------------------------------------------------------------

void Sorter::execHost(void)
{
    if (m_cfg.bytesPerKey == 4)
    {
        BufferRef<unsigned int> outKeys     = m_cfg.outKeys.reinterpret<unsigned int>();
        BufferRef<unsigned int> outValues   = m_cfg.outValues.reinterpret<unsigned int>();
        BufferRef<unsigned int> inKeys      = m_cfg.inKeys.reinterpret<unsigned int>();
        BufferRef<unsigned int> ilwalues    = m_cfg.ilwalues.reinterpret<unsigned int>();

        memcpy(outKeys.writeDiscardHost(), inKeys.readHost(), inKeys.getNumBytes());
        memcpy(outValues.writeDiscardHost(), ilwalues.readHost(), ilwalues.getNumBytes());

        corelib::qsort<unsigned int, unsigned int>(
            0, m_cfg.numItems, outKeys.readWriteHost(), outValues.readWriteHost());
    }
    else
    {
        RT_ASSERT(m_cfg.bytesPerKey == 8);
        BufferRef<unsigned long long>   outKeys     = m_cfg.outKeys.reinterpret<unsigned long long>();
        BufferRef<unsigned int>         outValues   = m_cfg.outValues.reinterpret<unsigned int>();
        BufferRef<unsigned long long>   inKeys      = m_cfg.inKeys.reinterpret<unsigned long long>();
        BufferRef<unsigned int>         ilwalues    = m_cfg.ilwalues.reinterpret<unsigned int>();

        memcpy(outKeys.writeDiscardHost(), inKeys.readHost(), inKeys.getNumBytes());
        memcpy(outValues.writeDiscardHost(), ilwalues.readHost(), ilwalues.getNumBytes());

        corelib::qsort<unsigned long long, unsigned int>(
            0, m_cfg.numItems, outKeys.readWriteHost(), outValues.readWriteHost());
    }
}

//------------------------------------------------------------------------
