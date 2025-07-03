#pragma once

/*
** Copyright (c) 2020 LWPU CORPORATION.  All rights reserved.
**
** LWPU CORPORATION and its licensors retain all intellectual property
** and proprietary rights in and to this software, related documentation
** and any modifications thereto.  Any use, reproduction, disclosure or
** distribution of this software and related documentation without an express
** license agreement from LWPU CORPORATION is strictly prohibited.
*/

#include <stdio.h>
#include <algorithm>
#include <math.h>
#include <time.h>
#include <string>

#define timespec_to_ms(t) (t.tv_sec*1000.0f + t.tv_nsec/1000000.0f)
inline float timespec_sub(struct timespec a, struct timespec b)
{
    a.tv_sec  -= b.tv_sec;
    a.tv_nsec -= b.tv_nsec;

    if (a.tv_nsec < 0) {
        a.tv_nsec += 1000000000;
        a.tv_sec  -= 1;
    }
    return timespec_to_ms(a);
}

class TimingStats
{
public:
    TimingStats(std::string name)
    {
        m_name = name;
        m_numData = -1;
    }

    ~TimingStats()
    {
        FlushData();
    }

    void AddTimingRecord()
    {
        struct timespec now;
        clock_gettime(CLOCK_MONOTONIC, &now);

        // On first call, just record the time but don't update the data
        if (m_numData >= 0) {
            m_data[m_numData++] = timespec_sub(now, m_prev);

            if (m_numData == MAX_DATA_COUNT) {
                FlushData();
                m_numData = 0;
            }
        } else {
            m_numData = 0;
        }
        m_prev = now;
    }

    void FlushData()
    {
        if (m_numData <= 0) {
            return;
        }
        float average = 0.0f;
        float stdDeviation = 0.0f;
        float min = m_data[0];
        float max = m_data[0];

        for (int i = 0; i < m_numData; i++) {
            float time = m_data[i];
            min = std::min(time, min);
            max = std::max(time, max);
            average += m_data[i];
        }

        average /= m_numData;

        for (int i = 0; i < m_numData; i++) {
            float elem = (m_data[i] - average);
            stdDeviation += elem * elem;
        }

        stdDeviation /= m_numData;
        stdDeviation = sqrt(stdDeviation);

        printf("%s: Iterations = %u, Average = %0.3f, Min = %0.3f, Max = %0.3f, "
               "StdDev = %0.3f, AverageFPS = %0.3f\n", m_name.c_str(),
                m_numData, average, min, max, stdDeviation,
                (average == 0.0f) ? 0.0f : 1000.0f/average);       // ms to fps
    }

private:
    static constexpr int MAX_DATA_COUNT = 1200;

    std::string m_name;

    float m_data[MAX_DATA_COUNT + 1];
    int m_numData;
    struct timespec m_prev;
};
