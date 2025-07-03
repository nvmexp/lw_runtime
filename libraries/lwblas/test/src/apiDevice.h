#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <ostream>
#include <limits>
#include <cstdint>

#include "apiTest.h"
#include "gtest/gtest.h"
#include "lwtensor.h"
#include "lwtensor/internal/operatorsPLC3.h"

namespace APITESTING{

     /**
      * \id lwGet_deviceCode
      * \brief validate the functionality of operator lwGet of device code
      * \depends None
      * \setup None
      * \testprocedure ./apiTest --gtest_filter=OPERATORS_H.lwGet_deviceCode
      * \teardown None
      * \testgroup OPERATORS_H
      * \inputs None
      * \outputs None
      * \expected States the expected value
      */
     TEST(OPERATORS_H, lwGet_deviceCode)
     {
         lwdaStream_t pstream;
         lwdaStreamCreateWithFlags(&pstream, lwdaStreamNonBlocking);

         signed char x0 = (signed char)2;
         bool result0 = lwGetTest<signed char, signed char>(&x0, pstream);
         EXPECT_TRUE(result0);

         signed char x1 = (signed char)2;
         bool result1 = lwGetTest<signed char, uint8_t>(&x1, pstream);
         EXPECT_TRUE(result1);

         signed char x2 = (signed char)2;
         bool result2 = lwGetTest<signed char, int>(&x2, pstream);
         EXPECT_TRUE(result2);

         signed char x3 = (signed char)3;
         bool result3 = lwGetTest<signed char, half>(&x3, pstream);
         EXPECT_TRUE(result3);

         signed char x4 = (signed char)4;
         bool result4 = lwGetTest<signed char, unsigned>(&x4, pstream);
         EXPECT_TRUE(result4);

         signed char x5 = (signed char)4;
         bool result5 = lwGetTest<signed char, double>(&x5, pstream);
         EXPECT_TRUE(result5);

         unsigned x6 = 2U;
         bool result6 = lwGetTest<unsigned, uint8_t>(&x6, pstream);
         EXPECT_TRUE(result6);

         unsigned x7 = 2U;
         bool result7 = lwGetTest<unsigned, signed char>(&x7, pstream);
         EXPECT_TRUE(result7);

         unsigned x8 = 2U;
         bool result8 = lwGetTest<unsigned, int>(&x8, pstream);
         EXPECT_TRUE(result8);

         unsigned x9 = 2U;
         bool result9 = lwGetTest<unsigned, half>(&x9, pstream);
         EXPECT_TRUE(result9);

         unsigned x10 = 2U;
         bool result10 = lwGetTest<unsigned, double>(&x10, pstream);
         EXPECT_TRUE(result10);

         half x11 = lwGet<half>(2.3f);
         bool result11 = lwGetTest<half, int>(&x11, pstream);
         EXPECT_TRUE(result11);

         half x12 = lwGet<half>(2.3f);
         bool result12 = lwGetTest<half, uint32_t>(&x12, pstream);
         EXPECT_TRUE(result12);

         half x13 = lwGet<half>(2.3f);
         bool result13 = lwGetTest<half, signed char>(&x13, pstream);
         EXPECT_TRUE(result13);

         half x14 = lwGet<half>(2.3f);
         bool result14 = lwGetTest<half, uint8_t>(&x14, pstream);
         EXPECT_TRUE(result14);

         half x15 = lwGet<half>(2.3f);
         bool result15 = lwGetTest<half, double>(&x15, pstream);
         EXPECT_TRUE(result15);

         double x16 = 2.3;
         bool result16 = lwGetTest<double, signed char>(&x16, pstream);
         EXPECT_TRUE(result16);

         double x17 = 2.3;
         bool result17 = lwGetTest<double, uint8_t>(&x17, pstream);
         EXPECT_TRUE(result17);

         double x18 = 2.3;
         bool result18 = lwGetTest<double, int>(&x18, pstream);
         EXPECT_TRUE(result18);

         double x19 = 2.3;
         bool result19 = lwGetTest<double, unsigned>(&x19, pstream);
         EXPECT_TRUE(result19);

         double x20 = 2.3;
         bool result20 = lwGetTest<double, half>(&x20, pstream);
         EXPECT_TRUE(result20);

         lwComplex x21 = make_lwComplex(1.2f, 2.3f);
         bool result21 = lwGetTest<lwComplex, lwComplex>(&x21, pstream);
         EXPECT_TRUE(result21);

         lwComplex x22 = make_lwComplex(1.2f, 2.3f);
         bool result22 = lwGetTest<lwComplex, int8_t>(&x22, pstream);
         EXPECT_TRUE(result22);

         lwComplex x23 = make_lwComplex(1.2f, 2.3f);
         bool result23 = lwGetTest<lwComplex, float>(&x23, pstream);
         EXPECT_TRUE(result23);

         int8_t x24 = 2;
         bool result24 = lwGetTest<int8_t, lwComplex>(&x24, pstream);
         EXPECT_TRUE(result24);

         float x25 = 2.3f;
         bool result25 = lwGetTest<float, lwComplex>(&x25, pstream);
         EXPECT_TRUE(result25);

         int x26 = 2;
         bool result26 = lwGetTest<int, lwComplex>(&x26, pstream);
         EXPECT_TRUE(result26);

         int32_t x27 = 2;
         bool result27 = lwGetTest<int32_t, uint8_t>(&x27, pstream);
         EXPECT_TRUE(result27);

         int32_t x28 = 2;
         bool result28 = lwGetTest<int32_t, unsigned>(&x28, pstream);
         EXPECT_TRUE(result28);

         int32_t x29 = 2;
         bool result29 = lwGetTest<int32_t, half>(&x29, pstream);
         EXPECT_TRUE(result29);

         unsigned x30 = 2U;
         bool result30 = lwGetTest<unsigned, uint32_t>(&x30, pstream);
         EXPECT_TRUE(result30);

         unsigned x31 = 2U;
         bool result31 = lwGetTest<unsigned, float>(&x31, pstream);
         EXPECT_TRUE(result31);

         unsigned x32 = 2U;
         bool result32 = lwGetTest<unsigned, float>(&x32, pstream);
         EXPECT_TRUE(result32);

         half x33 = lwGet<half>(2.3f);
         bool result33 = lwGetTest<half, half>(&x33, pstream);
         EXPECT_TRUE(result33);

         float x34 = 2.3f;
         bool result34 = lwGetTest<float, uint8_t>(&x34, pstream);
         EXPECT_TRUE(result34);

         lwdaStreamDestroy(pstream);
     }


     /**
      * \id lwExp_device
      * \brief validate the functionality of operator lwExp of device code
      * \depends None
      * \setup None
      * \testprocedure ./apiTest --gtest_filter=OPERATORS_H.lwExp_device
      * \teardown None
      * \testgroup OPERATORS_H
      * \inputs None
      * \outputs None
      * \expected States results from device and host are equal to each other
      */
     __device__ funPtr<half, half> dhalf_lwExp = lwExp;
     __device__ funPtr<float, float> dfloat_lwExp = lwExp;
     __device__ funPtr<double, double> ddouble_lwExp = lwExp;
     TEST(OPERATORS_H, lwExp_device)
     {
         funPtr<half, half> hhalf_lwExp;
         funPtr<float, float> hfloat_lwExp;
         funPtr<double, double> hdouble_lwExp;
         lwdaStream_t pstream;
         lwdaStreamCreateWithFlags(&pstream, lwdaStreamNonBlocking);

         lwdaMemcpyFromSymbolAsync(&hhalf_lwExp, dhalf_lwExp, sizeof(funPtr<half, half>), 0, lwdaMemcpyDefault, pstream);
         lwdaMemcpyFromSymbolAsync(&hfloat_lwExp, dfloat_lwExp, sizeof(funPtr<float, float>), 0, lwdaMemcpyDefault, pstream);
         lwdaMemcpyFromSymbolAsync(&hdouble_lwExp, ddouble_lwExp, sizeof(funPtr<double, double>), 0, lwdaMemcpyDefault, pstream);

         half x0 = lwGet<half>(2.3f);
         bool result0 = deviceCodeUnaryTest<half, half>(&x0, hhalf_lwExp, lwExp, pstream);
         EXPECT_TRUE(result0);

         float x1 = 2.3f;
         bool result1 = deviceCodeUnaryTest<float, float>(&x1, hfloat_lwExp, lwExp, pstream);
         EXPECT_TRUE(result1);

         double x2 = 2.3;
         bool result2 = deviceCodeUnaryTest<double, double>(&x2, hdouble_lwExp, lwExp, pstream);
         EXPECT_TRUE(result2);
         lwdaStreamDestroy(pstream);
     }

     /**
      * \id lwLn_device
      * \brief validate the functionality of operator lwLn of device code
      * \depends None
      * \setup None
      * \testprocedure ./apiTest --gtest_filter=OPERATORS_H.lwLn_device
      * \teardown None
      * \testgroup OPERATORS_H
      * \inputs None
      * \outputs None
      * \expected States results from device and host are equal to each other
      */
     __device__ funPtr<half, half> dhalf_lwLn = lwLn;
     __device__ funPtr<float, float> dfloat_lwLn = lwLn;
     __device__ funPtr<double, double> ddouble_lwLn = lwLn;
     TEST(OPERATORS_H, lwLn_device)
     {
         funPtr<half, half> hhalf_lwLn;
         funPtr<float, float> hfloat_lwLn;
         funPtr<double, double> hdouble_lwLn;
         lwdaStream_t pstream;
         lwdaStreamCreateWithFlags(&pstream, lwdaStreamNonBlocking);

         lwdaMemcpyFromSymbolAsync(&hhalf_lwLn, dhalf_lwLn, sizeof(funPtr<half, half>), 0, lwdaMemcpyDefault, pstream);
         lwdaMemcpyFromSymbolAsync(&hfloat_lwLn, dfloat_lwLn, sizeof(funPtr<float, float>), 0, lwdaMemcpyDefault, pstream);
         lwdaMemcpyFromSymbolAsync(&hdouble_lwLn, ddouble_lwLn, sizeof(funPtr<double, double>), 0, lwdaMemcpyDefault, pstream);

         half x0 = lwGet<half>(2.3f);
         bool result0 = deviceCodeUnaryTest<half, half>(&x0, hhalf_lwLn, lwLn, pstream);
         EXPECT_TRUE(result0);

         float x1 = 2.3f;
         bool result1 = deviceCodeUnaryTest<float, float>(&x1, hfloat_lwLn, lwLn, pstream);
         EXPECT_TRUE(result1);

         double x2 = 2.3;
         bool result2 = deviceCodeUnaryTest<double, double>(&x2, hdouble_lwLn, lwLn, pstream);
         EXPECT_TRUE(result2);
         lwdaStreamDestroy(pstream);
     }

     __device__ funPtr<half, half> dhalf_lwSin = lwSin;
     __device__ funPtr<float, float> dfloat_lwSin = lwSin;
     __device__ funPtr<double, double> ddouble_lwSin = lwSin;
     TEST(OPERATORS_H, lwSin_device)
     {
         funPtr<half, half> hhalf_lwSin;
         funPtr<float, float> hfloat_lwSin;
         funPtr<double, double> hdouble_lwSin;
         lwdaStream_t pstream;
         lwdaStreamCreateWithFlags(&pstream, lwdaStreamNonBlocking);

         lwdaMemcpyFromSymbolAsync(&hhalf_lwSin, dhalf_lwSin, sizeof(funPtr<half, half>), 0, lwdaMemcpyDefault, pstream);
         lwdaMemcpyFromSymbolAsync(&hfloat_lwSin, dfloat_lwSin, sizeof(funPtr<float, float>), 0, lwdaMemcpyDefault, pstream);
         lwdaMemcpyFromSymbolAsync(&hdouble_lwSin, ddouble_lwSin, sizeof(funPtr<double, double>), 0, lwdaMemcpyDefault, pstream);

         half x0 = lwGet<half>(2.3f);
         bool result0 = deviceCodeUnaryTest<half, half>(&x0, hhalf_lwSin, lwSin, pstream);
         EXPECT_TRUE(result0);

         float x1 = 2.3f;
         bool result1 = deviceCodeUnaryTest<float, float>(&x1, hfloat_lwSin, lwSin, pstream);
         EXPECT_TRUE(result1);

         double x2 = 2.3;
         bool result2 = deviceCodeUnaryTest<double, double>(&x2, hdouble_lwSin, lwSin, pstream);
         EXPECT_TRUE(result2);
         lwdaStreamDestroy(pstream);
     }

     /**
      * \id lwCos_device
      * \brief validate the functionality of operator lwCos of device code
      * \depends None
      * \setup None
      * \testprocedure ./apiTest --gtest_filter=OPERATORS_H.lwCos_device
      * \teardown None
      * \testgroup OPERATORS_H
      * \inputs None
      * \outputs None
      * \expected States results from device and host are equal to each other
      */
     __device__ funPtr<half, half> dhalf_lwCos = lwCos;
     __device__ funPtr<float, float> dfloat_lwCos = lwCos;
     __device__ funPtr<double, double> ddouble_lwCos = lwCos;
     TEST(OPERATORS_H, lwCos_device)
     {
         funPtr<half, half> hhalf_lwCos;
         funPtr<float, float> hfloat_lwCos;
         funPtr<double, double> hdouble_lwCos;
         lwdaStream_t pstream;
         lwdaStreamCreateWithFlags(&pstream, lwdaStreamNonBlocking);

         lwdaMemcpyFromSymbolAsync(&hhalf_lwCos, dhalf_lwCos, sizeof(funPtr<half, half>), 0, lwdaMemcpyDefault, pstream);
         lwdaMemcpyFromSymbolAsync(&hfloat_lwCos, dfloat_lwCos, sizeof(funPtr<float, float>), 0, lwdaMemcpyDefault, pstream);
         lwdaMemcpyFromSymbolAsync(&hdouble_lwCos, ddouble_lwCos, sizeof(funPtr<double, double>), 0, lwdaMemcpyDefault, pstream);

         half x0 = lwGet<half>(2.3f);
         bool result0 = deviceCodeUnaryTest<half, half>(&x0, hhalf_lwCos, lwCos, pstream);
         EXPECT_TRUE(result0);

         float x1 = 2.3f;
         bool result1 = deviceCodeUnaryTest<float, float>(&x1, hfloat_lwCos, lwCos, pstream);
         EXPECT_TRUE(result1);

         double x2 = 2.3;
         bool result2 = deviceCodeUnaryTest<double, double>(&x2, hdouble_lwCos, lwCos, pstream);
         EXPECT_TRUE(result2);
         lwdaStreamDestroy(pstream);
     }

     /**
      * \id lwTan_device
      * \brief validate the functionality of operator lwTan of device code
      * \depends None
      * \setup None
      * \testprocedure ./apiTest --gtest_filter=OPERATORS_H.lwTan_device
      * \teardown None
      * \testgroup OPERATORS_H
      * \inputs None
      * \outputs None
      * \expected States results from device and host are equal to each other
      */
     __device__ funPtr<half, half> dhalf_lwTan = lwTan;
     __device__ funPtr<float, float> dfloat_lwTan = lwTan;
     __device__ funPtr<double, double> ddouble_lwTan = lwTan;
     TEST(OPERATORS_H, lwTan_device)
     {
         funPtr<half, half> hhalf_lwTan;
         funPtr<float, float> hfloat_lwTan;
         funPtr<double, double> hdouble_lwTan;
         lwdaStream_t pstream;
         lwdaStreamCreateWithFlags(&pstream, lwdaStreamNonBlocking);

         lwdaMemcpyFromSymbolAsync(&hhalf_lwTan, dhalf_lwTan, sizeof(funPtr<half, half>), 0, lwdaMemcpyDefault, pstream);
         lwdaMemcpyFromSymbolAsync(&hfloat_lwTan, dfloat_lwTan, sizeof(funPtr<float, float>), 0, lwdaMemcpyDefault, pstream);
         lwdaMemcpyFromSymbolAsync(&hdouble_lwTan, ddouble_lwTan, sizeof(funPtr<double, double>), 0, lwdaMemcpyDefault, pstream);

         half x0 = lwGet<half>(2.3f);
         bool result0 = deviceCodeUnaryTest<half, half>(&x0, hhalf_lwTan, lwTan, pstream);
         EXPECT_TRUE(result0);

         float x1 = 2.3f;
         bool result1 = deviceCodeUnaryTest<float, float>(&x1, hfloat_lwTan, lwTan, pstream);
         EXPECT_TRUE(result1);

         double x2 = 2.3;
         bool result2 = deviceCodeUnaryTest<double, double>(&x2, hdouble_lwTan, lwTan, pstream);
         EXPECT_TRUE(result2);
         lwdaStreamDestroy(pstream);
     }

     /**
      * \id lwSinh_device
      * \brief validate the functionality of operator lwSinh of device code
      * \depends None
      * \setup None
      * \testprocedure ./apiTest --gtest_filter=OPERATORS_H.lwSinh_device
      * \teardown None
      * \testgroup OPERATORS_H
      * \inputs None
      * \outputs None
      * \expected States results from device and host are equal to each other
      */
     __device__ funPtr<half, half> dhalf_lwSinh = lwSinh;
     __device__ funPtr<float, float> dfloat_lwSinh = lwSinh;
     __device__ funPtr<double, double> ddouble_lwSinh = lwSinh;
     TEST(OPERATORS_H, lwSinh_device)
     {
         funPtr<half, half> hhalf_lwSinh;
         funPtr<float, float> hfloat_lwSinh;
         funPtr<double, double> hdouble_lwSinh;
         lwdaStream_t pstream;
         lwdaStreamCreateWithFlags(&pstream, lwdaStreamNonBlocking);

         lwdaMemcpyFromSymbolAsync(&hhalf_lwSinh, dhalf_lwSinh, sizeof(funPtr<half, half>), 0, lwdaMemcpyDefault, pstream);
         lwdaMemcpyFromSymbolAsync(&hfloat_lwSinh, dfloat_lwSinh, sizeof(funPtr<float, float>), 0, lwdaMemcpyDefault, pstream);
         lwdaMemcpyFromSymbolAsync(&hdouble_lwSinh, ddouble_lwSinh, sizeof(funPtr<double, double>), 0, lwdaMemcpyDefault, pstream);

         half x0 = lwGet<half>(2.3f);
         bool result0 = deviceCodeUnaryTest<half, half>(&x0, hhalf_lwSinh, lwSinh, pstream);
         EXPECT_TRUE(result0);

         float x1 = 2.3f;
         bool result1 = deviceCodeUnaryTest<float, float>(&x1, hfloat_lwSinh, lwSinh, pstream);
         EXPECT_TRUE(result1);

         double x2 = 2.3;
         bool result2 = deviceCodeUnaryTest<double, double>(&x2, hdouble_lwSinh, lwSinh, pstream);
         EXPECT_TRUE(result2);
         lwdaStreamDestroy(pstream);
     }

     /**
      * \id lwCosh_device
      * \brief validate the functionality of operator lwCosh of device code
      * \depends None
      * \setup None
      * \testprocedure ./apiTest --gtest_filter=OPERATORS_H.lwCosh_device
      * \teardown None
      * \testgroup OPERATORS_H
      * \inputs None
      * \outputs None
      * \expected States results from device and host are equal to each other
      */
     __device__ funPtr<half, half> dhalf_lwCosh = lwCosh;
     __device__ funPtr<float, float> dfloat_lwCosh = lwCosh;
     __device__ funPtr<double, double> ddouble_lwCosh = lwCosh;
     TEST(OPERATORS_H, lwCosh_device)
     {
         funPtr<half, half> hhalf_lwCosh;
         funPtr<float, float> hfloat_lwCosh;
         funPtr<double, double> hdouble_lwCosh;
         lwdaStream_t pstream;
         lwdaStreamCreateWithFlags(&pstream, lwdaStreamNonBlocking);

         lwdaMemcpyFromSymbolAsync(&hhalf_lwCosh, dhalf_lwCosh, sizeof(funPtr<half, half>), 0, lwdaMemcpyDefault, pstream);
         lwdaMemcpyFromSymbolAsync(&hfloat_lwCosh, dfloat_lwCosh, sizeof(funPtr<float, float>), 0, lwdaMemcpyDefault, pstream);
         lwdaMemcpyFromSymbolAsync(&hdouble_lwCosh, ddouble_lwCosh, sizeof(funPtr<double, double>), 0, lwdaMemcpyDefault, pstream);

         half x0 = lwGet<half>(2.3f);
         bool result0 = deviceCodeUnaryTest<half, half>(&x0, hhalf_lwCosh, lwCosh, pstream);
         EXPECT_TRUE(result0);

         float x1 = 2.3f;
         bool result1 = deviceCodeUnaryTest<float, float>(&x1, hfloat_lwCosh, lwCosh, pstream);
         EXPECT_TRUE(result1);

         double x2 = 2.3;
         bool result2 = deviceCodeUnaryTest<double, double>(&x2, hdouble_lwCosh, lwCosh, pstream);
         EXPECT_TRUE(result2);
         lwdaStreamDestroy(pstream);
     }

     /**
      * \id lwTanh_device
      * \brief validate the functionality of operator lwTanh of device code
      * \depends None
      * \setup None
      * \testprocedure ./apiTest --gtest_filter=OPERATORS_H.lwTanh_device
      * \teardown None
      * \testgroup OPERATORS_H
      * \inputs None
      * \outputs None
      * \expected States results from device and host are equal to each other
      */
     __device__ funPtr<half, half> dhalf_lwTanh = lwTanh;
     __device__ funPtr<float, float> dfloat_lwTanh = lwTanh;
     __device__ funPtr<double, double> ddouble_lwTanh = lwTanh;
     TEST(OPERATORS_H, lwTanh_device)
     {
         funPtr<half, half> hhalf_lwTanh;
         funPtr<float, float> hfloat_lwTanh;
         funPtr<double, double> hdouble_lwTanh;
         lwdaStream_t pstream;
         lwdaStreamCreateWithFlags(&pstream, lwdaStreamNonBlocking);

         lwdaMemcpyFromSymbolAsync(&hhalf_lwTanh, dhalf_lwTanh, sizeof(funPtr<half, half>), 0, lwdaMemcpyDefault, pstream);
         lwdaMemcpyFromSymbolAsync(&hfloat_lwTanh, dfloat_lwTanh, sizeof(funPtr<float, float>), 0, lwdaMemcpyDefault, pstream);
         lwdaMemcpyFromSymbolAsync(&hdouble_lwTanh, ddouble_lwTanh, sizeof(funPtr<double, double>), 0, lwdaMemcpyDefault, pstream);

         half x0 = lwGet<half>(2.3f);
         bool result0 = deviceCodeUnaryTest<half, half>(&x0, hhalf_lwTanh, lwTanh, pstream);
         EXPECT_TRUE(result0);

         float x1 = 2.3f;
         bool result1 = deviceCodeUnaryTest<float, float>(&x1, hfloat_lwTanh, lwTanh, pstream);
         EXPECT_TRUE(result1);

         double x2 = 2.3;
         bool result2 = deviceCodeUnaryTest<double, double>(&x2, hdouble_lwTanh, lwTanh, pstream);
         EXPECT_TRUE(result2);
         lwdaStreamDestroy(pstream);
     }

     /**
      * \id lwAsin_device
      * \brief validate the functionality of operator lwAsin of device code
      * \depends None
      * \setup None
      * \testprocedure ./apiTest --gtest_filter=OPERATORS_H.lwAsin_device
      * \teardown None
      * \testgroup OPERATORS_H
      * \inputs None
      * \outputs None
      * \expected States results from device and host are equal to each other
      */
     __device__ funPtr<half, half> dhalf_lwAsin = lwAsin;
     __device__ funPtr<float, float> dfloat_lwAsin = lwAsin;
     __device__ funPtr<double, double> ddouble_lwAsin = lwAsin;
     TEST(OPERATORS_H, lwAsin_device)
     {
         funPtr<half, half> hhalf_lwAsin;
         funPtr<float, float> hfloat_lwAsin;
         funPtr<double, double> hdouble_lwAsin;
         lwdaStream_t pstream;
         lwdaStreamCreateWithFlags(&pstream, lwdaStreamNonBlocking);

         lwdaMemcpyFromSymbolAsync(&hhalf_lwAsin, dhalf_lwAsin, sizeof(funPtr<half, half>), 0, lwdaMemcpyDefault, pstream);
         lwdaMemcpyFromSymbolAsync(&hfloat_lwAsin, dfloat_lwAsin, sizeof(funPtr<float, float>), 0, lwdaMemcpyDefault, pstream);
         lwdaMemcpyFromSymbolAsync(&hdouble_lwAsin, ddouble_lwAsin, sizeof(funPtr<double, double>), 0, lwdaMemcpyDefault, pstream);

         half x0 = lwGet<half>(0.3f);
         bool result0 = deviceCodeUnaryTest<half, half>(&x0, hhalf_lwAsin, lwAsin, pstream);
         EXPECT_TRUE(result0);

         float x1 = 0.3f;
         bool result1 = deviceCodeUnaryTest<float, float>(&x1, hfloat_lwAsin, lwAsin, pstream);
         EXPECT_TRUE(result1);

         double x2 = 0.3;
         bool result2 = deviceCodeUnaryTest<double, double>(&x2, hdouble_lwAsin, lwAsin, pstream);
         EXPECT_TRUE(result2);
         lwdaStreamDestroy(pstream);
     }

     /**
      * \id lwAcos_device
      * \brief validate the functionality of operator lwAcos of device code
      * \depends None
      * \setup None
      * \testprocedure ./apiTest --gtest_filter=OPERATORS_H.lwAcos_device
      * \teardown None
      * \testgroup OPERATORS_H
      * \inputs None
      * \outputs None
      * \expected States results from device and host are equal to each other
      */
     __device__ funPtr<half, half> dhalf_lwAcos = lwAcos;
     __device__ funPtr<float, float> dfloat_lwAcos = lwAcos;
     __device__ funPtr<double, double> ddouble_lwAcos = lwAcos;
     TEST(OPERATORS_H, lwAcos_device)
     {
         funPtr<half, half> hhalf_lwAcos;
         funPtr<float, float> hfloat_lwAcos;
         funPtr<double, double> hdouble_lwAcos;
         lwdaStream_t pstream;
         lwdaStreamCreateWithFlags(&pstream, lwdaStreamNonBlocking);

         lwdaMemcpyFromSymbolAsync(&hhalf_lwAcos, dhalf_lwAcos, sizeof(funPtr<half, half>), 0, lwdaMemcpyDefault, pstream);
         lwdaMemcpyFromSymbolAsync(&hfloat_lwAcos, dfloat_lwAcos, sizeof(funPtr<float, float>), 0, lwdaMemcpyDefault, pstream);
         lwdaMemcpyFromSymbolAsync(&hdouble_lwAcos, ddouble_lwAcos, sizeof(funPtr<double, double>), 0, lwdaMemcpyDefault, pstream);

         half x0 = lwGet<half>(0.3f);
         bool result0 = deviceCodeUnaryTest<half, half>(&x0, hhalf_lwAcos, lwAcos, pstream);
         EXPECT_TRUE(result0);

         float x1 = 0.3f;
         bool result1 = deviceCodeUnaryTest<float, float>(&x1, hfloat_lwAcos, lwAcos, pstream);
         EXPECT_TRUE(result1);

         double x2 = 0.3;
         bool result2 = deviceCodeUnaryTest<double, double>(&x2, hdouble_lwAcos, lwAcos, pstream);
         EXPECT_TRUE(result2);
         lwdaStreamDestroy(pstream);
     }

     /**
      * \id lwAtan_device
      * \brief validate the functionality of operator lwAtan of device code
      * \depends None
      * \setup None
      * \testprocedure ./apiTest --gtest_filter=OPERATORS_H.lwAtan_device
      * \teardown None
      * \testgroup OPERATORS_H
      * \inputs None
      * \outputs None
      * \expected States results from device and host are equal to each other
      */
     __device__ funPtr<half, half> dhalf_lwAtan = lwAtan;
     __device__ funPtr<float, float> dfloat_lwAtan = lwAtan;
     __device__ funPtr<double, double> ddouble_lwAtan = lwAtan;
     TEST(OPERATORS_H, lwAtan_device)
     {
         funPtr<half, half> hhalf_lwAtan;
         funPtr<float, float> hfloat_lwAtan;
         funPtr<double, double> hdouble_lwAtan;
         lwdaStream_t pstream;
         lwdaStreamCreateWithFlags(&pstream, lwdaStreamNonBlocking);

         lwdaMemcpyFromSymbolAsync(&hhalf_lwAtan, dhalf_lwAtan, sizeof(funPtr<half, half>), 0, lwdaMemcpyDefault, pstream);
         lwdaMemcpyFromSymbolAsync(&hfloat_lwAtan, dfloat_lwAtan, sizeof(funPtr<float, float>), 0, lwdaMemcpyDefault, pstream);
         lwdaMemcpyFromSymbolAsync(&hdouble_lwAtan, ddouble_lwAtan, sizeof(funPtr<double, double>), 0, lwdaMemcpyDefault, pstream);

         half x0 = lwGet<half>(0.3f);
         bool result0 = deviceCodeUnaryTest<half, half>(&x0, hhalf_lwAtan, lwAtan, pstream);
         EXPECT_TRUE(result0);

         float x1 = 0.3f;
         bool result1 = deviceCodeUnaryTest<float, float>(&x1, hfloat_lwAtan, lwAtan, pstream);
         EXPECT_TRUE(result1);

         double x2 = 0.3;
         bool result2 = deviceCodeUnaryTest<double, double>(&x2, hdouble_lwAtan, lwAtan, pstream);
         EXPECT_TRUE(result2);
         lwdaStreamDestroy(pstream);
     }

     /**
      * \id lwAsinh_device
      * \brief validate the functionality of operator lwAsinh of device code
      * \depends None
      * \setup None
      * \testprocedure ./apiTest --gtest_filter=OPERATORS_H.lwAsinh_device
      * \teardown None
      * \testgroup OPERATORS_H
      * \inputs None
      * \outputs None
      * \expected States results from device and host are equal to each other
      */
     __device__ funPtr<half, half> dhalf_lwAsinh = lwAsinh;
     __device__ funPtr<float, float> dfloat_lwAsinh = lwAsinh;
     __device__ funPtr<double, double> ddouble_lwAsinh = lwAsinh;
     TEST(OPERATORS_H, lwAsinh_device)
     {
         funPtr<half, half> hhalf_lwAsinh;
         funPtr<float, float> hfloat_lwAsinh;
         funPtr<double, double> hdouble_lwAsinh;
         lwdaStream_t pstream;
         lwdaStreamCreateWithFlags(&pstream, lwdaStreamNonBlocking);

         lwdaMemcpyFromSymbolAsync(&hhalf_lwAsinh, dhalf_lwAsinh, sizeof(funPtr<half, half>), 0, lwdaMemcpyDefault, pstream);
         lwdaMemcpyFromSymbolAsync(&hfloat_lwAsinh, dfloat_lwAsinh, sizeof(funPtr<float, float>), 0, lwdaMemcpyDefault, pstream);
         lwdaMemcpyFromSymbolAsync(&hdouble_lwAsinh, ddouble_lwAsinh, sizeof(funPtr<double, double>), 0, lwdaMemcpyDefault, pstream);

         half x0 = lwGet<half>(0.3f);
         bool result0 = deviceCodeUnaryTest<half, half>(&x0, hhalf_lwAsinh, lwAsinh, pstream);
         EXPECT_TRUE(result0);

         float x1 = 0.3f;
         bool result1 = deviceCodeUnaryTest<float, float>(&x1, hfloat_lwAsinh, lwAsinh, pstream);
         EXPECT_TRUE(result1);

         double x2 = 0.3;
         bool result2 = deviceCodeUnaryTest<double, double>(&x2, hdouble_lwAsinh, lwAsinh, pstream);
         EXPECT_TRUE(result2);
         lwdaStreamDestroy(pstream);
     }

     /**
      * \id lwAcosh_device
      * \brief validate the functionality of operator lwAcosh of device code
      * \depends None
      * \setup None
      * \testprocedure ./apiTest --gtest_filter=OPERATORS_H.lwAcosh_device
      * \teardown None
      * \testgroup OPERATORS_H
      * \inputs None
      * \outputs None
      * \expected States results from device and host are equal to each other
      */
     __device__ funPtr<half, half> dhalf_lwAcosh = lwAcosh;
     __device__ funPtr<float, float> dfloat_lwAcosh = lwAcosh;
     __device__ funPtr<double, double> ddouble_lwAcosh = lwAcosh;
     TEST(OPERATORS_H, lwAcosh_device)
     {
         funPtr<half, half> hhalf_lwAcosh;
         funPtr<float, float> hfloat_lwAcosh;
         funPtr<double, double> hdouble_lwAcosh;
         lwdaStream_t pstream;
         lwdaStreamCreateWithFlags(&pstream, lwdaStreamNonBlocking);

         lwdaMemcpyFromSymbolAsync(&hhalf_lwAcosh, dhalf_lwAcosh, sizeof(funPtr<half, half>), 0, lwdaMemcpyDefault, pstream);
         lwdaMemcpyFromSymbolAsync(&hfloat_lwAcosh, dfloat_lwAcosh, sizeof(funPtr<float, float>), 0, lwdaMemcpyDefault, pstream);
         lwdaMemcpyFromSymbolAsync(&hdouble_lwAcosh, ddouble_lwAcosh, sizeof(funPtr<double, double>), 0, lwdaMemcpyDefault, pstream);

         half x0 = lwGet<half>(2.3f);
         bool result0 = deviceCodeUnaryTest<half, half>(&x0, hhalf_lwAcosh, lwAcosh, pstream);
         EXPECT_TRUE(result0);

         float x1 = 2.3f;
         bool result1 = deviceCodeUnaryTest<float, float>(&x1, hfloat_lwAcosh, lwAcosh, pstream);
         EXPECT_TRUE(result1);

         double x2 = 2.3;
         bool result2 = deviceCodeUnaryTest<double, double>(&x2, hdouble_lwAcosh, lwAcosh, pstream);
         EXPECT_TRUE(result2);
         lwdaStreamDestroy(pstream);
     }

     /**
      * \id lwAtanh_device
      * \brief validate the functionality of operator lwAtanh of device code
      * \depends None
      * \setup None
      * \testprocedure ./apiTest --gtest_filter=OPERATORS_H.lwAtanh_device
      * \teardown None
      * \testgroup OPERATORS_H
      * \inputs None
      * \outputs None
      * \expected States results from device and host are equal to each other
      */
     __device__ funPtr<half, half> dhalf_lwAtanh = lwAtanh;
     __device__ funPtr<float, float> dfloat_lwAtanh = lwAtanh;
     __device__ funPtr<double, double> ddouble_lwAtanh = lwAtanh;
     TEST(OPERATORS_H, lwAtanh_device)
     {
         funPtr<half, half> hhalf_lwAtanh;
         funPtr<float, float> hfloat_lwAtanh;
         funPtr<double, double> hdouble_lwAtanh;
         lwdaStream_t pstream;
         lwdaStreamCreateWithFlags(&pstream, lwdaStreamNonBlocking);

         lwdaMemcpyFromSymbolAsync(&hhalf_lwAtanh, dhalf_lwAtanh, sizeof(funPtr<half, half>), 0, lwdaMemcpyDefault, pstream);
         lwdaMemcpyFromSymbolAsync(&hfloat_lwAtanh, dfloat_lwAtanh, sizeof(funPtr<float, float>), 0, lwdaMemcpyDefault, pstream);
         lwdaMemcpyFromSymbolAsync(&hdouble_lwAtanh, ddouble_lwAtanh, sizeof(funPtr<double, double>), 0, lwdaMemcpyDefault, pstream);

         half x0 = lwGet<half>(0.3f);
         bool result0 = deviceCodeUnaryTest<half, half>(&x0, hhalf_lwAtanh, lwAtanh, pstream);
         EXPECT_TRUE(result0);

         float x1 = 0.3f;
         bool result1 = deviceCodeUnaryTest<float, float>(&x1, hfloat_lwAtanh, lwAtanh, pstream);
         EXPECT_TRUE(result1);

         double x2 = 0.3;
         bool result2 = deviceCodeUnaryTest<double, double>(&x2, hdouble_lwAtanh, lwAtanh, pstream);
         EXPECT_TRUE(result2);
         lwdaStreamDestroy(pstream);
     }

     /**
      * \id lwFloor_device
      * \brief validate the functionality of operator lwFloor of device code
      * \depends None
      * \setup None
      * \testprocedure ./apiTest --gtest_filter=OPERATORS_H.lwFloor_device
      * \teardown None
      * \testgroup OPERATORS_H
      * \inputs None
      * \outputs None
      * \expected States results from device and host are equal to each other
      */
     __device__ funPtr<half, half> dhalf_lwFloor = lwFloor;
     __device__ funPtr<float, float> dfloat_lwFloor = lwFloor;
     __device__ funPtr<double, double> ddouble_lwFloor = lwFloor;
     TEST(OPERATORS_H, lwFloor_device)
     {
         funPtr<half, half> hhalf_lwFloor;
         funPtr<float, float> hfloat_lwFloor;
         funPtr<double, double> hdouble_lwFloor;
         lwdaStream_t pstream;
         lwdaStreamCreateWithFlags(&pstream, lwdaStreamNonBlocking);

         lwdaMemcpyFromSymbolAsync(&hhalf_lwFloor, dhalf_lwFloor, sizeof(funPtr<half, half>), 0, lwdaMemcpyDefault, pstream);
         lwdaMemcpyFromSymbolAsync(&hfloat_lwFloor, dfloat_lwFloor, sizeof(funPtr<float, float>), 0, lwdaMemcpyDefault, pstream);
         lwdaMemcpyFromSymbolAsync(&hdouble_lwFloor, ddouble_lwFloor, sizeof(funPtr<double, double>), 0, lwdaMemcpyDefault, pstream);

         half x0 = lwGet<half>(2.3f);
         bool result0 = deviceCodeUnaryTest<half, half>(&x0, hhalf_lwFloor, lwFloor, pstream);
         EXPECT_TRUE(result0);

         float x1 = 2.3f;
         bool result1 = deviceCodeUnaryTest<float, float>(&x1, hfloat_lwFloor, lwFloor, pstream);
         EXPECT_TRUE(result1);

         double x2 = 2.3;
         bool result2 = deviceCodeUnaryTest<double, double>(&x2, hdouble_lwFloor, lwFloor, pstream);
         EXPECT_TRUE(result2);
         lwdaStreamDestroy(pstream);
     }

     /**
      * \id lwCeil_device
      * \brief validate the functionality of operator lwCeil of device code
      * \depends None
      * \setup None
      * \testprocedure ./apiTest --gtest_filter=OPERATORS_H.lwCeil_device
      * \teardown None
      * \testgroup OPERATORS_H
      * \inputs None
      * \outputs None
      * \expected States results from device and host are equal to each other
      */
     __device__ funPtr<half, half> dhalf_lwCeil = lwCeil;
     __device__ funPtr<float, float> dfloat_lwCeil = lwCeil;
     __device__ funPtr<double, double> ddouble_lwCeil = lwCeil;
     TEST(OPERATORS_H, lwCeil_device)
     {
         funPtr<half, half> hhalf_lwCeil;
         funPtr<float, float> hfloat_lwCeil;
         funPtr<double, double> hdouble_lwCeil;
         lwdaStream_t pstream;
         lwdaStreamCreateWithFlags(&pstream, lwdaStreamNonBlocking);

         lwdaMemcpyFromSymbolAsync(&hhalf_lwCeil, dhalf_lwCeil, sizeof(funPtr<half, half>), 0, lwdaMemcpyDefault, pstream);
         lwdaMemcpyFromSymbolAsync(&hfloat_lwCeil, dfloat_lwCeil, sizeof(funPtr<float, float>), 0, lwdaMemcpyDefault, pstream);
         lwdaMemcpyFromSymbolAsync(&hdouble_lwCeil, ddouble_lwCeil, sizeof(funPtr<double, double>), 0, lwdaMemcpyDefault, pstream);

         half x0 = lwGet<half>(2.3f);
         bool result0 = deviceCodeUnaryTest<half, half>(&x0, hhalf_lwCeil, lwCeil, pstream);
         EXPECT_TRUE(result0);

         float x1 = 2.3f;
         bool result1 = deviceCodeUnaryTest<float, float>(&x1, hfloat_lwCeil, lwCeil, pstream);
         EXPECT_TRUE(result1);

         double x2 = 2.3;
         bool result2 = deviceCodeUnaryTest<double, double>(&x2, hdouble_lwCeil, lwCeil, pstream);
         EXPECT_TRUE(result2);
         lwdaStreamDestroy(pstream);
     }

     /**
      * \id lwSigmoid_device
      * \brief validate the functionality of operator lwSigmoid of device code
      * \depends None
      * \setup None
      * \testprocedure ./apiTest --gtest_filter=OPERATORS_H.lwSigmoid_device
      * \teardown None
      * \testgroup OPERATORS_H
      * \inputs None
      * \outputs None
      * \expected States results from device and host are equal to each other
      */
     __device__ funPtr<half, half> dhalf_lwSigmoid = lwSigmoid;
     __device__ funPtr<float, float> dfloat_lwSigmoid = lwSigmoid;
     __device__ funPtr<double, double> ddouble_lwSigmoid = lwSigmoid;
     TEST(OPERATORS_H, lwSigmoid_device)
     {
         funPtr<half, half> hhalf_lwSigmoid;
         funPtr<float, float> hfloat_lwSigmoid;
         funPtr<double, double> hdouble_lwSigmoid;
         lwdaStream_t pstream;
         lwdaStreamCreateWithFlags(&pstream, lwdaStreamNonBlocking);

         lwdaMemcpyFromSymbolAsync(&hhalf_lwSigmoid, dhalf_lwSigmoid, sizeof(funPtr<half, half>), 0, lwdaMemcpyDefault, pstream);
         lwdaMemcpyFromSymbolAsync(&hfloat_lwSigmoid, dfloat_lwSigmoid, sizeof(funPtr<float, float>), 0, lwdaMemcpyDefault, pstream);
         lwdaMemcpyFromSymbolAsync(&hdouble_lwSigmoid, ddouble_lwSigmoid, sizeof(funPtr<double, double>), 0, lwdaMemcpyDefault, pstream);

         half x0 = lwGet<half>(2.3f);
         bool result0 = deviceCodeUnaryTest<half, half>(&x0, hhalf_lwSigmoid, lwSigmoid, pstream);
         EXPECT_TRUE(result0);

         float x1 = 2.3f;
         bool result1 = deviceCodeUnaryTest<float, float>(&x1, hfloat_lwSigmoid, lwSigmoid, pstream);
         EXPECT_TRUE(result1);

         double x2 = 2.3;
         bool result2 = deviceCodeUnaryTest<double, double>(&x2, hdouble_lwSigmoid, lwSigmoid, pstream);
         EXPECT_TRUE(result2);
         lwdaStreamDestroy(pstream);
     }

     /**
      * \id lwNeg_device
      * \brief validate the functionality of operator lwNeg of device code
      * \depends None
      * \setup None
      * \testprocedure ./apiTest --gtest_filter=OPERATORS_H.lwNeg_device
      * \teardown None
      * \testgroup OPERATORS_H
      * \inputs None
      * \outputs None
      * \expected States results from device and host are equal to each other
      */
     __device__ funPtr<half, half> dhalf_lwNeg = lwNeg;
     __device__ funPtr<float, float> dfloat_lwNeg = lwNeg;
     __device__ funPtr<double, double> ddouble_lwNeg = lwNeg;
     TEST(OPERATORS_H, lwNeg_device)
     {
         funPtr<half, half> hhalf_lwNeg;
         funPtr<float, float> hfloat_lwNeg;
         funPtr<double, double> hdouble_lwNeg;
         lwdaStream_t pstream;
         lwdaStreamCreateWithFlags(&pstream, lwdaStreamNonBlocking);

         lwdaMemcpyFromSymbolAsync(&hhalf_lwNeg, dhalf_lwNeg, sizeof(funPtr<half, half>), 0, lwdaMemcpyDefault, pstream);
         lwdaMemcpyFromSymbolAsync(&hfloat_lwNeg, dfloat_lwNeg, sizeof(funPtr<float, float>), 0, lwdaMemcpyDefault, pstream);
         lwdaMemcpyFromSymbolAsync(&hdouble_lwNeg, ddouble_lwNeg, sizeof(funPtr<double, double>), 0, lwdaMemcpyDefault, pstream);

         half x0 = lwGet<half>(2.3f);
         bool result0 = deviceCodeUnaryTest<half, half>(&x0, hhalf_lwNeg, lwNeg, pstream);
         EXPECT_TRUE(result0);

         float x1 = 2.3f;
         bool result1 = deviceCodeUnaryTest<float, float>(&x1, hfloat_lwNeg, lwNeg, pstream);
         EXPECT_TRUE(result1);

         double x2 = 2.3;
         bool result2 = deviceCodeUnaryTest<double, double>(&x2, hdouble_lwNeg, lwNeg, pstream);
         EXPECT_TRUE(result2);
         lwdaStreamDestroy(pstream);
     }

     /**
      * \id lwAbs_device
      * \brief validate the functionality of operator lwAbs of device code
      * \depends None
      * \setup None
      * \testprocedure ./apiTest --gtest_filter=OPERATORS_H.lwAbs_device
      * \teardown None
      * \testgroup OPERATORS_H
      * \inputs None
      * \outputs None
      * \expected States results from device and host are equal to each other
      */
     __device__ funPtr<half, half> dhalf_lwAbs = lwAbs;
     __device__ funPtr<float, float> dfloat_lwAbs = lwAbs;
     __device__ funPtr<double, double> ddouble_lwAbs = lwAbs;
     TEST(OPERATORS_H, lwAbs_device)
     {
         funPtr<half, half> hhalf_lwAbs;
         funPtr<float, float> hfloat_lwAbs;
         funPtr<double, double> hdouble_lwAbs;
         lwdaStream_t pstream;
         lwdaStreamCreateWithFlags(&pstream, lwdaStreamNonBlocking);

         lwdaMemcpyFromSymbolAsync(&hhalf_lwAbs, dhalf_lwAbs, sizeof(funPtr<half, half>), 0, lwdaMemcpyDefault, pstream);
         lwdaMemcpyFromSymbolAsync(&hfloat_lwAbs, dfloat_lwAbs, sizeof(funPtr<float, float>), 0, lwdaMemcpyDefault, pstream);
         lwdaMemcpyFromSymbolAsync(&hdouble_lwAbs, ddouble_lwAbs, sizeof(funPtr<double, double>), 0, lwdaMemcpyDefault, pstream);

         half x0 = lwGet<half>(2.3f);
         bool result0 = deviceCodeUnaryTest<half, half>(&x0, hhalf_lwAbs, lwAbs, pstream);
         EXPECT_TRUE(result0);

         float x1 = 2.3f;
         bool result1 = deviceCodeUnaryTest<float, float>(&x1, hfloat_lwAbs, lwAbs, pstream);
         EXPECT_TRUE(result1);

         double x2 = 2.3;
         bool result2 = deviceCodeUnaryTest<double, double>(&x2, hdouble_lwAbs, lwAbs, pstream);
         EXPECT_TRUE(result2);
         lwdaStreamDestroy(pstream);
     }

     /**
      * \id lwSoftSign_device
      * \brief validate the functionality of operator lwSoftSign of device code
      * \depends None
      * \setup None
      * \testprocedure ./apiTest --gtest_filter=OPERATORS_H.lwSoftSign_device
      * \teardown None
      * \testgroup OPERATORS_H
      * \inputs None
      * \outputs None
      * \expected States results from device and host are equal to each other
      */
     __device__ funPtr<half, half> dhalf_lwSoftSign = lwSoftSign;
     __device__ funPtr<float, float> dfloat_lwSoftSign = lwSoftSign;
     __device__ funPtr<double, double> ddouble_lwSoftSign = lwSoftSign;
     TEST(OPERATORS_H, lwSoftSign_device)
     {
         funPtr<half, half> hhalf_lwSoftSign;
         funPtr<float, float> hfloat_lwSoftSign;
         funPtr<double, double> hdouble_lwSoftSign;
         lwdaStream_t pstream;
         lwdaStreamCreateWithFlags(&pstream, lwdaStreamNonBlocking);

         lwdaMemcpyFromSymbolAsync(&hhalf_lwSoftSign, dhalf_lwSoftSign, sizeof(funPtr<half, half>), 0, lwdaMemcpyDefault, pstream);
         lwdaMemcpyFromSymbolAsync(&hfloat_lwSoftSign, dfloat_lwSoftSign, sizeof(funPtr<float, float>), 0, lwdaMemcpyDefault, pstream);
         lwdaMemcpyFromSymbolAsync(&hdouble_lwSoftSign, ddouble_lwSoftSign, sizeof(funPtr<double, double>), 0, lwdaMemcpyDefault, pstream);

         half x0 = lwGet<half>(2.3f);
         bool result0 = deviceCodeUnaryTest<half, half>(&x0, hhalf_lwSoftSign, lwSoftSign, pstream);
         EXPECT_TRUE(result0);

         float x1 = 2.3f;
         bool result1 = deviceCodeUnaryTest<float, float>(&x1, hfloat_lwSoftSign, lwSoftSign, pstream);
         EXPECT_TRUE(result1);

         double x2 = 2.3;
         bool result2 = deviceCodeUnaryTest<double, double>(&x2, hdouble_lwSoftSign, lwSoftSign, pstream);
         EXPECT_TRUE(result2);
         lwdaStreamDestroy(pstream);
     }

     /**
      * \id lwSqrt_device
      * \brief validate the functionality of operator lwSqrt of device code
      * \depends None
      * \setup None
      * \testprocedure ./apiTest --gtest_filter=OPERATORS_H.lwSqrt_device
      * \teardown None
      * \testgroup OPERATORS_H
      * \inputs None
      * \outputs None
      * \expected States results from device and host are equal to each other
      */
     __device__ funPtr<half, half> dhalf_lwSqrt = lwSqrt;
     __device__ funPtr<float, float> dfloat_lwSqrt = lwSqrt;
     __device__ funPtr<double, double> ddouble_lwSqrt = lwSqrt;
     TEST(OPERATORS_H, lwSqrt_device)
     {
         funPtr<half, half> hhalf_lwSqrt;
         funPtr<float, float> hfloat_lwSqrt;
         funPtr<double, double> hdouble_lwSqrt;
         lwdaStream_t pstream;
         lwdaStreamCreateWithFlags(&pstream, lwdaStreamNonBlocking);

         lwdaMemcpyFromSymbolAsync(&hhalf_lwSqrt, dhalf_lwSqrt, sizeof(funPtr<half, half>), 0, lwdaMemcpyDefault, pstream);
         lwdaMemcpyFromSymbolAsync(&hfloat_lwSqrt, dfloat_lwSqrt, sizeof(funPtr<float, float>), 0, lwdaMemcpyDefault, pstream);
         lwdaMemcpyFromSymbolAsync(&hdouble_lwSqrt, ddouble_lwSqrt, sizeof(funPtr<double, double>), 0, lwdaMemcpyDefault, pstream);

         half x0 = lwGet<half>(2.3f);
         bool result0 = deviceCodeUnaryTest<half, half>(&x0, hhalf_lwSqrt, lwSqrt, pstream);
         EXPECT_TRUE(result0);

         float x1 = 2.3f;
         bool result1 = deviceCodeUnaryTest<float, float>(&x1, hfloat_lwSqrt, lwSqrt, pstream);
         EXPECT_TRUE(result1);

         double x2 = 2.3;
         bool result2 = deviceCodeUnaryTest<double, double>(&x2, hdouble_lwSqrt, lwSqrt, pstream);
         EXPECT_TRUE(result2);
         lwdaStreamDestroy(pstream);
     }

     /**
      * \id lwRcp_device
      * \brief validate the functionality of operator lwRcp of device code
      * \depends None
      * \setup None
      * \testprocedure ./apiTest --gtest_filter=OPERATORS_H.lwRcp_device
      * \teardown None
      * \testgroup OPERATORS_H
      * \inputs None
      * \outputs None
      * \expected States results from device and host are equal to each other
      */
     __device__ funPtr<half, half> dhalf_lwRcp = lwRcp;
     TEST(OPERATORS_H, lwRcp_device)
     {
         funPtr<half, half> hhalf_lwRcp;
         lwdaStream_t pstream;
         lwdaStreamCreateWithFlags(&pstream, lwdaStreamNonBlocking);

         lwdaMemcpyFromSymbolAsync(&hhalf_lwRcp, dhalf_lwRcp, sizeof(funPtr<half, half>), 0, lwdaMemcpyDefault, pstream);

         half x0 = lwGet<half>(2.3f);
         bool result0 = deviceCodeUnaryTest<half, half>(&x0, hhalf_lwRcp, lwRcp, pstream);
         EXPECT_TRUE(result0);

         lwdaStreamDestroy(pstream);
     }

     /**
      * \id lwElu_device
      * \brief validate the functionality of operator lwElu of device code
      * \depends None
      * \setup None
      * \testprocedure ./apiTest --gtest_filter=OPERATORS_H.lwElu_device
      * \teardown None
      * \testgroup OPERATORS_H
      * \inputs None
      * \outputs None
      * \expected States results from device and host are equal to each other
      */
     // TODO: this test passed locally, but failed on CI/CD
//     __device__ funPtr1<half,  float, half> dhalf_lwElu = lwElu;
//     __device__ funPtr1<float, float, float> dfloat_lwElu = lwElu;
//     __device__ funPtr1<double, float, double> ddouble_lwElu = lwElu;
//     TEST(OPERATORS_H, lwElu_device)
//     {
//         funPtr1<half, float, half> hhalf_lwElu;
//         funPtr1<float, float, float> hfloat_lwElu;
//         funPtr1<double, float, double> hdouble_lwElu;
//         lwdaStream_t pstream;
//         lwdaStreamCreateWithFlags(&pstream, lwdaStreamNonBlocking);
//
//         lwdaMemcpyFromSymbolAsync(&hhalf_lwElu, dhalf_lwElu, sizeof(funPtr1<half, float, half>), 0, lwdaMemcpyDefault, pstream);
//         lwdaMemcpyFromSymbolAsync(&hfloat_lwElu, dfloat_lwElu, sizeof(funPtr1<float, float, float>), 0, lwdaMemcpyDefault, pstream);
//         lwdaMemcpyFromSymbolAsync(&hdouble_lwElu, ddouble_lwElu, sizeof(funPtr1<double, float, double>), 0, lwdaMemcpyDefault, pstream);
//
//         float y = 1.2f;
//         half x0 = lwGet<half>(2.3f);
//         bool result0 = deviceCodeBinaryTest<half, float, half>(&x0, &y, hhalf_lwElu, lwElu, pstream);
//         EXPECT_TRUE(result0);
//
//         float x1 = 2.3f;
//         bool result1 = deviceCodeBinaryTest<float, float, float>(&x1, &y, hfloat_lwElu, lwElu, pstream);
//         EXPECT_TRUE(result1);
//
//         double x2 = 2.3;
//         bool result2 = deviceCodeBinaryTest<double, float, double>(&x2, &y, hdouble_lwElu, lwElu, pstream);
//         EXPECT_TRUE(result2);
//         lwdaStreamDestroy(pstream);
//     }

} //namespace APITESTING
