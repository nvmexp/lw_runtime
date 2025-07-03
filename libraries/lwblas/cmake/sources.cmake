set(LWTENSOR_UTILS_SRC
  src/lwtensor.cpp
  src/types.cpp
  src/typesEx.cpp
  src/util.cpp
  src/utilEx.cpp
  src/context.cpp
  src/heuristicSimple.cpp
  src/heuristicEW.cpp
)

set(LWTENSOR_CONTRACTION_SRC
  src/tensorContraction_lwtlass_auto.lw
  src/tensorContraction.lw
  src/tensorColwolution.lw
  src/reduction.lw
)

set(LWTENSOR_SRC_SM61
  src/tensorContraction_sm61_cccc.lw
  src/tensorContraction_sm61_dddd.lw
  src/tensorContraction_sm61_ddds.lw
  src/tensorContraction_sm61_ssss.lw
  src/tensorContraction_sm61_ssss_tc.lw
  src/tensorContraction_sm61_zzzz.lw
  src/tensorContraction_sm61_dzzz.lw
  src/tensorContraction_sm61_zdzz.lw
  src/tensorContraction_sm61_zzzc.lw
  src/tensorContraction_sm61_hhhs_tc.lw
  src/elementwise_sm61.lw
)

set(LWTENSOR_SRC_SM70
  src/tensorContraction_sm70_cccc.lw
  src/tensorContraction_sm70_dddd.lw
  src/tensorContraction_sm70_ddds.lw
  src/tensorContraction_sm70_ssss.lw
  src/tensorContraction_sm70_ssss_tc.lw
  src/tensorContraction_sm70_ssss_tc.lw
  src/tensorContraction_sm70_zzzz.lw
  src/tensorContraction_sm70_dzzz.lw
  src/tensorContraction_sm70_zdzz.lw
  src/tensorContraction_sm70_zzzc.lw
  src/tensorContraction_sm70_hhhs_tc.lw

  src/tensorContraction_lwte_sm70_ssss.lw
  src/tensorContraction_lwte_sm70_dddd.lw
  src/elementwise_sm70.lw
)

set(LWTENSOR_SRC_SM75
  src/tensorContraction_sm75_cccc.lw
  src/tensorContraction_sm75_dddd.lw
  src/tensorContraction_sm75_ddds.lw
  src/tensorContraction_sm75_ssss.lw
  src/tensorContraction_sm75_ssss_tc.lw
  src/tensorContraction_sm75_zzzz.lw
  src/tensorContraction_sm75_dzzz.lw
  src/tensorContraction_sm75_zdzz.lw
  src/tensorContraction_sm75_zzzc.lw
  src/tensorContraction_sm75_hhhs_tc.lw
  src/elementwise_sm75.lw
)

set(LWTENSOR_SRC_SM80
  src/tensorContraction_sm80_cccc.lw
  src/tensorContraction_sm80_cccc_tc_tf32.lw
  src/tensorContraction_sm80_dddd.lw
  src/tensorContraction_sm80_ddds.lw
  src/tensorContraction_sm80_ssss.lw
  src/tensorContraction_sm80_ssss_tc.lw
  src/tensorContraction_sm80_ssss_tc_bf16.lw
  src/tensorContraction_sm80_ssss_tc_tf32.lw
  src/tensorContraction_sm80_zzzz.lw
  src/tensorContraction_sm80_dzzz.lw
  src/tensorContraction_sm80_zdzz.lw
  src/tensorContraction_sm80_zzzc.lw
  src/tensorContraction_sm80_hhhs_tc.lw
  src/tensorContraction_sm80_bbbs_tc_bf16.lw
  src/elementwise_sm80.lw
)

set(LWTENSOR_ELEMENTWISE_SRC
  src/elementwise.lw
  src/elementwise_auto.lw
)

if (SYS_ARCH STREQUAL "ppc64le")
    set(LWTENSOR_DNN_HEURISTIC_SRC
        external/dnnheuristic/gemm_noarch.cpp
    )
    set(LWTENSOR_DNN_HEURISTIC_INC
        external/dnnheuristic/gemm.h
        external/dnnheuristic/gemm_template.h
    )
elseif(SYS_ARCH STREQUAL "sbsa")
    set(LWTENSOR_DNN_HEURISTIC_SRC
        external/dnnheuristic/gemm_noarch.cpp
    )
    set(LWTENSOR_DNN_HEURISTIC_INC
        external/dnnheuristic/gemm.h
        external/dnnheuristic/gemm_template.h
    )
else()
    set(LWTENSOR_DNN_HEURISTIC_SRC
        external/dnnheuristic/dispatch_x86_64.cpp
        external/dnnheuristic/gemm_avx2.cpp
        external/dnnheuristic/gemm_avx512.cpp
        external/dnnheuristic/gemm_avx.cpp
        external/dnnheuristic/gemm_sse2.cpp
        external/dnnheuristic/gemm_noarch.cpp
    )
    set(LWTENSOR_DNN_HEURISTIC_INC
        external/dnnheuristic/gemm.h
        external/dnnheuristic/gemm_template.h
        external/dnnheuristic/arch/helperavx.h
        external/dnnheuristic/arch/helperavx2.h
        external/dnnheuristic/arch/helperavx512f.h
        external/dnnheuristic/arch/helpersse2.h
    )
endif()

set(LWTENSOR_INC
  include/lwtensor.h
  include/lwtensor/types.h
  include/lwtensor/internal/lwtensorEx.h
  include/lwtensor/internal/lwtensor.h
  include/lwtensor/internal/defines.h
  include/lwtensor/internal/dnnContractionWeights.h
  include/lwtensor/internal/dnnContractionWeightsDDDD.h
  include/lwtensor/internal/elementwise.h
  include/lwtensor/internal/elementwisePrototype.h
  include/lwtensor/internal/exceptions.h
  include/lwtensor/internal/featuresUtils.h
  include/lwtensor/internal/heuristicsLwtlass.h
  include/lwtensor/internal/heuristicDnn.h
  include/lwtensor/internal/operators.h
  include/lwtensor/internal/reduction.h
  include/lwtensor/internal/tensorContraction.h
  include/lwtensor/internal/tensorContractionLwtlass.h
  include/lwtensor/internal/ttgt.h
  include/lwtensor/internal/types.h
  include/lwtensor/internal/utilEx.h
  include/lwtensor/internal/util.h
)

set(LWTENSOR_TEST_SRC
  test/apiTest.lw
  test/lwtensorTest.lw
  test/unitTest.cpp
)

if (WIN32)
    set_property(
        SOURCE external/dnnheuristic/gemm_avx512.cpp APPEND_STRING PROPERTY
        COMPILE_FLAGS   " /arch:AVX512 /fp:fast"
    )
    set_property(
        SOURCE external/dnnheuristic/gemm_avx2.cpp APPEND_STRING PROPERTY
        COMPILE_FLAGS   " /arch:AVX2 /fp:fast"
    )
    set_property(
        SOURCE external/dnnheuristic/gemm_avx.cpp APPEND_STRING PROPERTY
        COMPILE_FLAGS   " /arch:AVX /fp:fast"
    )
    set_property(
        SOURCE external/dnnheuristic/gemm_sse2.cpp APPEND_STRING PROPERTY
        COMPILE_FLAGS   " /fp:fast"
    )
    set_property(
        SOURCE external/dnnheuristic/gemm_noarch.cpp APPEND_STRING PROPERTY
        COMPILE_FLAGS   " /fp:fast"
    )
else()
    if (SYS_ARCH STREQUAL "ppc64le")
    elseif (SYS_ARCH STREQUAL "sbsa")
    else()
        set_property(
            SOURCE external/dnnheuristic/gemm_avx512.cpp APPEND_STRING PROPERTY
            COMPILE_FLAGS " -O3 -mavx2 -ffast-math -mfma -mavx512dq -Wno-attributes"
        )
        set_property(
            SOURCE external/dnnheuristic/gemm_avx2.cpp APPEND_STRING PROPERTY
            COMPILE_FLAGS " -O3 -mavx2 -ffast-math -mfma -Wno-attributes"
        )
        set_property(
            SOURCE external/dnnheuristic/gemm_avx.cpp APPEND_STRING PROPERTY
            COMPILE_FLAGS " -O3 -mavx -ffast-math -Wno-attributes"
        )
        set_property(
            SOURCE external/dnnheuristic/gemm_sse2.cpp APPEND_STRING PROPERTY
            COMPILE_FLAGS " -O3 -msse2 -ffast-math"
        )
        set_property(
            SOURCE external/dnnheuristic/gemm_noarch.cpp APPEND_STRING PROPERTY
            COMPILE_FLAGS " -O3 -ffast-math"
        )
        set_property(
            SOURCE
            external/dnnheuristic/gemm_avx512.cpp
            external/dnnheuristic/gemm_avx2.cpp
            external/dnnheuristic/gemm_avx.cpp
            external/dnnheuristic/gemm_sse2.cpp
            external/dnnheuristic/gemm_noarch.cpp
            APPEND_STRING PROPERTY COMPILE_FLAGS " -Wno-unused-function"
        )
    endif()

endif()

