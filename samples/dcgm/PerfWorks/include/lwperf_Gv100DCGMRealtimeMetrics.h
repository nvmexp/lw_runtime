/*
 * Copyright 2014-2018  LWPU Corporation.  All rights reserved.
 *
 * NOTICE TO USER:
 *
 * This source code is subject to LWPU ownership rights under U.S. and
 * international Copyright laws.
 *
 * This software and the information contained herein is PROPRIETARY and
 * CONFIDENTIAL to LWPU and is being provided under the terms and conditions
 * of a form of LWPU software license agreement.
 *
 * LWPU MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE
 * CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR
 * IMPLIED WARRANTY OF ANY KIND.  LWPU DISCLAIMS ALL WARRANTIES WITH
 * REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * IN NO EVENT SHALL LWPU BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL,
 * OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
 * OF USE, DATA OR PROFITS,  WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
 * OR OTHER TORTIOUS ACTION,  ARISING OUT OF OR IN CONNECTION WITH THE USE
 * OR PERFORMANCE OF THIS SOURCE CODE.
 *
 * U.S. Government End Users.   This source code is a "commercial item" as
 * that term is defined at  48 C.F.R. 2.101 (OCT 1995), consisting  of
 * "commercial computer  software"  and "commercial computer software
 * documentation" as such terms are  used in 48 C.F.R. 12.212 (SEPT 1995)
 * and is provided to the U.S. Government only as a commercial end item.
 * Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through
 * 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the
 * source code with only those rights set forth herein.
 *
 * Any use of this source code in individual and commercial software must
 * include, in the user documentation and internal comments to the code,
 * the above Disclaimer and U.S. Government End Users Notice.
 */

#pragma once
#include <stdint.h>
#include "lwperf_RealtimeMetrics.h"

namespace lw { namespace metrics { namespace gv100 {


    namespace RawMetricIdx
    {
        enum Enum
        {
            sm__warps_active                                                , //      0
            tpc__cycles_elapsed                                             , //      1
            sm__cycles_active                                               , //      2
            sys__cycles_elapsed                                             , //      3
            smsp__pipe_tensor_op_hmma_cycles_active                         , //      4
            COUNT // = 5
        };
    }

    struct RawMetricsStorage : public RawMetricsContext
    {
        double values[RawMetricIdx::COUNT];
        uint16_t counts[RawMetricIdx::COUNT];

        MetricValue GetValue(RawMetricIdx::Enum rawMetricIdx, double sustainedRate, double cycles_elapsed)
        {
            MetricValue metricValue; // single exit-point to encourage RVO
            if (configuring)
            {
                counts[rawMetricIdx] = 1;
                // fill dummy data to avoid compiler or linter warnings about uninitialized data
                metricValue = {};
            }
            else
            {
                const double unitCount = counts[rawMetricIdx];
                metricValue.sum = values[rawMetricIdx];
                metricValue.avg = values[rawMetricIdx] / unitCount;
                metricValue.peak_sustained = sustainedRate;
                metricValue.cycles_elapsed = cycles_elapsed;
            }
            return metricValue;
        }

        // helpers for templating
        static const size_t NumRawMetricIds = RawMetricIdx::COUNT;
        static const uint64_t* GetRawMetricIds()
        {
            static const uint64_t s_rawMetricIds[] = {
                0x4fc35fed50b6a866, // sm__warps_active
                0x5d8a1887d485c0c8, // tpc__cycles_elapsed
                0x729172b0ec7860a7, // sm__cycles_active
                0x971c2144063dc931, // sys__cycles_elapsed
                0xd1b0b882710e4e60, // smsp__pipe_tensor_op_hmma_cycles_active
                0
            };
            return s_rawMetricIds;
        }
        static const char* const* GetRawMetricNames()
        {
            static const char* const s_rawMetricNames[] = {
                "sm__warps_active",
                "tpc__cycles_elapsed",
                "sm__cycles_active",
                "sys__cycles_elapsed",
                "smsp__pipe_tensor_op_hmma_cycles_active",
                0
            };
            return s_rawMetricNames;
        }

        RawMetricsStorage() {}
    };

    struct RawMetrics : public RawMetricsStorage
    {
        MetricValue sm__warps_active()
        {
            return GetValue(RawMetricIdx::sm__warps_active, 64.0, tpc__cycles_elapsed().avg);
        }

        MetricValue tpc__cycles_elapsed()
        {
            MetricValue metricValue;
            if (configuring)
            {
                counts[RawMetricIdx::tpc__cycles_elapsed] = 1;
                metricValue = {};
            }
            else
            {
                const double unitCount = counts[RawMetricIdx::tpc__cycles_elapsed];
                metricValue.sum = values[RawMetricIdx::tpc__cycles_elapsed];
                metricValue.avg = metricValue.sum / unitCount;
                metricValue.peak_sustained = 1.0;
                metricValue.cycles_elapsed = metricValue.avg;
            }
            return metricValue;
        }

        MetricValue sm__cycles_active()
        {
            return GetValue(RawMetricIdx::sm__cycles_active, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue sys__cycles_elapsed()
        {
            MetricValue metricValue;
            if (configuring)
            {
                counts[RawMetricIdx::sys__cycles_elapsed] = 1;
                metricValue = {};
            }
            else
            {
                const double unitCount = counts[RawMetricIdx::sys__cycles_elapsed];
                metricValue.sum = values[RawMetricIdx::sys__cycles_elapsed];
                metricValue.avg = metricValue.sum / unitCount;
                metricValue.peak_sustained = 1.0;
                metricValue.cycles_elapsed = metricValue.avg;
            }
            return metricValue;
        }

        MetricValue smsp__pipe_tensor_op_hmma_cycles_active()
        {
            return GetValue(RawMetricIdx::smsp__pipe_tensor_op_hmma_cycles_active, 1.0, tpc__cycles_elapsed().avg);
        }

    };

    struct AllMetrics : public RawMetrics
    {
        MetricValue attrs_per_vector()
        {
            const double value = 4;
            return MetricValue{ value, value, value, 0};
        }

        MetricValue bytes_per_sector()
        {
            const double value = 32;
            return MetricValue{ value, value, value, 0};
        }

        MetricValue four_attrs_per_vert()
        {
            const double value = 4;
            return MetricValue{ value, value, value, 0};
        }

        MetricValue pixels_per_quad()
        {
            const double value = 4;
            return MetricValue{ value, value, value, 0};
        }

        MetricValue samples_per_z_occluder()
        {
            const double value = 8;
            return MetricValue{ value, value, value, 0};
        }

        MetricValue threads_per_quad()
        {
            const double value = 4;
            return MetricValue{ value, value, value, 0};
        }

    };

#define LW_GV100_RAW_METRIC_DESCS(f_) \
    f_(sm__cycles_active, "# of cycles with at least one warp in flight") \
    f_(sm__warps_active, "sum of per-cycle # of warps in flight") \
    f_(smsp__pipe_tensor_op_hmma_cycles_active, "# of cycles where HMMA tensor cores were active") \
    f_(sys__cycles_elapsed, "# of cycles elapsed on SYS") \
    f_(tpc__cycles_elapsed, "# of cycles elapsed on TPC") \

#define LW_GV100_ALL_METRIC_DESCS(f_) \
    LW_GV100_RAW_METRIC_DESCS(f_) \


}}} // namespace
