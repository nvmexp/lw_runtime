#pragma once 

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <ostream>
#include <limits>
#include <cstdint>
#include "apiTest.h"
#include "gtest/gtest.h"
#include "lwtensor.h"
#include "lwtensor/types.h"
#include "lwtensor/internal/lwtensorEx.h"
#include "lwtensor/internal/types.h"
#include "lwtensor/internal/util.h"
#include "lwtensor/internal/operators.h"
#include "lwtensor/internal/defines.h"
#include "lwtensor/internal/exceptions.h"
#include "lwtensor/internal/elementwise.h"
#include "lwtensor/internal/elementwisePrototype.h"

/**
 * @file
 * @brief This file contains all unit test.
 */

//namespace

namespace APITESTING
{
    /**
     * @brief Test the function lwGet<lwDoubleComplex>
     * pass criteria: output should be as expected
     */
    TEST(OPERATORS, lwGet_lwDoubleComplex0)
    {
        lwDoubleComplex a, b;
        a = make_lwDoubleComplex(1.2, -3.4);
        callingInfo("lwGet<lwDoubleComplex>");
        b = lwGet<lwDoubleComplex>(a);
        EXPECT_TRUE(lwIsEqual(a, b));
    }

    /**
     * @brief Test the function lwGet<lwComplex>, from lwDoubleComplex to lwComplex
     * pass criteria: output should be as expected
     */
    TEST(OPERATORS, lwGet_lwDoubleComplex1)
    {
        lwDoubleComplex a;
        lwComplex b;
        a = make_lwDoubleComplex(1.2, -3.4);
        callingInfo("lwGet<lwComplex>");
        b = lwGet<lwComplex>(a);
        EXPECT_TRUE(lwIsEqual(lwComplexDoubleToFloat(a), b));
    }

    /**
     * @brief Test the function lwGet<lwComplex>, from signed char to lwComplex,
     * pass criteria: output  should be as expected
     */
    TEST(OPERATORS, lwGet_signedChar7)
    {
        signed char a;
        lwComplex b;
        a = 'a';
        callingInfo("lwGet<double>");
        b = lwGet<lwComplex>(a);
        EXPECT_TRUE(lwIsEqual(make_lwComplex(float(a), 0.0f), b));
    }

    /**
     * @brief Test the function lwGet<lwDoubleComplex, from signed char to lwDoubleComplex
     * pass criteria: output  should be as expected
     */
    TEST(OPERATORS, lwGet_signedChar8)
    {
        signed char a;
        lwDoubleComplex b;
        a = 'a';
        callingInfo("lwGet<lwDoubleComplex>");
        b = lwGet<lwDoubleComplex>(a);
        EXPECT_TRUE(lwIsEqual(make_lwDoubleComplex(double(a), 0.0), b));
    }

    /**
     * @brief Test the function lwGet<lwComplex>, from int to lwComplex
     * pass criteria: output  should be as expected
     */
    TEST(OPERATORS, lwGet_int11)
    {
        int a;
        lwComplex b;
        a = 1;
        callingInfo("lwGet<lwComplex>");
        b = lwGet<lwComplex>(a);
        callingInfo("lwIsEqual");
        EXPECT_TRUE(lwIsEqual(make_lwComplex(float(a), 0.0f), b));
    }

    /**
     * @brief Test the function lwGet<lwDoubleComplex>, from int to lwDoubleComplex
     * pass criteria: output  should be as expected
     */
    TEST(OPERATORS, lwGet_int12)
    {
        int a;
        lwDoubleComplex b;
        a = 1;
        callingInfo("lwGet<lwDoubleComplex>");
        b = lwGet<lwDoubleComplex>(a);
        callingInfo("lwIsEqual");
        EXPECT_TRUE(lwIsEqual(make_lwDoubleComplex(double(a), 0.0), b));
    }


    /**
     * @brief Test the function lwGet<lwComplex>, from unsigned to lwComplex
     * pass criteria: output  should be as expected
     */
    TEST(OPERATORS, lwGet_unsigned7)
    {
        unsigned a;
        lwComplex b;
        a = 1;
        callingInfo("lwGet<lwComplex>");
        b = lwGet<lwComplex>(a);
        callingInfo("lwIsEqual");
        EXPECT_TRUE(lwIsEqual(make_lwComplex(float(a), 0.0f), b));
    }

    /**
     * @brief Test the function lwGet<lwDoubleComplex>, from unsigned to lwDoubleComplex
     * pass criteria: output  should be as expected
     */
    TEST(OPERATORS, lwGet_unsigned8)
    {
        unsigned a;
        lwDoubleComplex b;
        a = 1;
        callingInfo("lwGet<lwDoubleComplex>");
        b = lwGet<lwDoubleComplex>(a);
        callingInfo("lwIsEqual");
        EXPECT_TRUE(lwIsEqual(make_lwDoubleComplex(double(a), 0.0), b));
    }

    /**
     * @brief Test the function lwGet<lwComplex>, from float to lwComplex
     * pass criteria: output  should be as expected
     */
    TEST(OPERATORS, lwGet_float8)
    {
        float a;
        lwComplex b;
        a = -1.2f;
        callingInfo("lwGet<lwComplex>");
        b = lwGet<lwComplex>(a);
        callingInfo("lwIsEqual");
        EXPECT_TRUE(lwIsEqual(make_lwComplex(float(a), 0.0f), b));
    }

    /**
     * @brief Test the function lwGet<lwDoubleComplex>, from float to lwDoubleComplex
     * pass criteria: output  should be as expected
     */
    TEST(OPERATORS, lwGet_float9)
    {
        float a;
        lwDoubleComplex b;
        a = -1.2f;
        callingInfo("lwGet<lwDoubleComplex>");
        b = lwGet<lwDoubleComplex>(a);
        EXPECT_TRUE(lwIsEqual(make_lwDoubleComplex(double(a), 0.0), b));
    }

    /**
     * @brief Test the function lwGet<lwComplex>, from half to lwComplex
     * pass criteria: output  should be as expected
     */
    TEST(OPERATORS, lwGet_half7)
    {
        half a;
        lwComplex b;
        a = __float2half(1);
        callingInfo("lwGet<lwComplex>");
        b = lwGet<lwComplex>(a);
        callingInfo("lwIsEqual and make_lwComplex");
        EXPECT_TRUE(lwIsEqual(make_lwComplex(lwGet<float>(a), 0.0f), b));
    }

    /**
     * @brief Test the function lwGet<lwDoubleComplex>, from half to lwDoubleComplex
     * pass criteria: output  should be as expected
     */

    TEST(OPERATORS, lwGet_half8)
    {
        half a;
        lwDoubleComplex b;
        a = __float2half(1);
        callingInfo("lwGet<lwDoubleComplex>");
        b = lwGet<lwDoubleComplex>(a);
        callingInfo("lwIsEqual and make_lwDoubleComplex");
        EXPECT_TRUE(lwIsEqual(make_lwDoubleComplex(double(lwGet<float>(a)), 0.0), b));
    }

    /**
     * @brief Test the function lwGet<lwComplex>(a), from double to lwComplex
     * pass criteria: output  should be as expected
     */
    TEST(OPERATORS, lwGet_double11)
    {
        double a;
        lwComplex b;
        a = -1.2;
        callingInfo("lwGet<lwComplex>");
        b = lwGet<lwComplex>(a);
        callingInfo("lwIsEqual and make_lwComplex");
        EXPECT_TRUE(lwIsEqual(make_lwComplex(float(a), 0.0f), b));
    }

    /**
     * @brief Test the function lwGet<lwDoubleComplex>, from double to lwDoubleComplex
     * pass criteria: output  should be as expected
     */
    TEST(OPERATORS, lwGet_double12)
    {
        double a;
        lwDoubleComplex b;
        a = -1.2;
        callingInfo("lwGet<lwDoubleComplex>");
        b = lwGet<lwDoubleComplex>(a);
        callingInfo("lwIsEqual and make_lwDoubleComplex");
        EXPECT_TRUE(lwIsEqual(make_lwDoubleComplex(double(a), 0.0), b));
    }

    /**
     * @brief Test the function lwAdd<lwComplex>
     * pass criteria: output  should be as expected
     */
    TEST(OPERATORS, lwAddlwComplex)
    {
        float ar = 1.2f, ai = 3.4f;
        float br = 1.2f, bi = 2.2f;
        callingInfo("lwAdd<lwComplex>");
        EXPECT_TRUE(lwIsEqual<lwComplex>(
                    lwAdd<lwComplex>(
                        make_lwComplex(ar, ai),
                        make_lwComplex(br, bi)),
                    make_lwComplex(ar + br, ai + bi)));
    }

    /**
     * @brief Test the function lwAdd<lwDoubleComplex>
     * pass criteria: output  should be as expected
     */
    TEST(OPERATORS, lwAddlwDoubleComplex)
    {
        double ar = 1.2f, ai = 3.4f;
        double br = 1.2f, bi = 2.2f;
        callingInfo("lwAdd<lwDoubleComplex>");
        EXPECT_TRUE(lwIsEqual<lwDoubleComplex>(
                    lwAdd<lwDoubleComplex>(
                        make_lwDoubleComplex(ar, ai),
                        make_lwDoubleComplex(br, bi)),
                    make_lwDoubleComplex(ar + br, ai + bi)));
    }

    /**
     * @brief Test the function Operator<half, half, half, LWTENSOR_OP_MAX>
     * pass criteria: output  should be as expected
     */
    TEST(OPERATORS, OperatorHalf)
    {
        half A, B;
        A = UniformRandomNumber<half>(0.0, 10.0);
        B = UniformRandomNumber<half>(0.0, 10.0);

        callingInfo("lwMul");
        struct Operator<half, half, half, LWTENSOR_OP_MUL> mul;
        EXPECT_TRUE(lwIsEqual(mul.execute(A, B), lwMul(lwGet<half>(A), lwGet<half>(B))));

        callingInfo("lwAdd");
        struct Operator<half, half, half, LWTENSOR_OP_ADD> add;
        EXPECT_TRUE(lwIsEqual(add.execute(A, B), lwAdd(lwGet<half>(A), lwGet<half>(B))));

        /* CHANGE: OP_SUB is no longer supported. */
        //struct Operator<half, half, half, LWTENSOR_OP_SUB> sub;
        //EXPECT_TRUE(lwIsEqual(sub.execute(A, B), lwSub(lwGet<half>(A), lwGet<half>(B))));

        callingInfo("lwMax");
        struct Operator<half, half, half, LWTENSOR_OP_MAX> max;
        EXPECT_TRUE(lwIsEqual(max.execute(A, B), lwMax(lwGet<half>(A), lwGet<half>(B))));
    }

    /**
     * @brief Test the function Operator<float, float, float, LWTENSOR_OP_MAX>
     * pass criteria: output  should be as expected
     */
    TEST(OPERATORS, OperatorFloat)
    {
        float A, B;
        A = UniformRandomNumber<float>(0.0, 10.0);
        B = UniformRandomNumber<float>(0.0, 10.0);

        callingInfo("lwMul");
        struct Operator<float, float, float, LWTENSOR_OP_MUL> mul;
        EXPECT_TRUE(lwIsEqual(mul.execute(A, B), lwMul(lwGet<float>(A), lwGet<float>(B))));

        callingInfo("lwAdd");
        struct Operator<float, float, float, LWTENSOR_OP_ADD> add;
        EXPECT_TRUE(lwIsEqual(add.execute(A, B), lwAdd(lwGet<float>(A), lwGet<float>(B))));

        /* CHANGE: OP_SUB is no longer supported. */
        //struct Operator<float, float, float, LWTENSOR_OP_SUB> sub;
        //EXPECT_TRUE(lwIsEqual(sub.execute(A, B), lwSub(lwGet<float>(A), lwGet<float>(B))));

        callingInfo("lwMax");
        struct Operator<float, float, float, LWTENSOR_OP_MAX> max;
        EXPECT_TRUE(lwIsEqual(max.execute(A, B), lwMax(lwGet<float>(A), lwGet<float>(B))));
    }

    /**
     * @brief Test the function Operator<double, double, double, LWTENSOR_OP_MAX>
     * pass criteria: output  should be as expected
     */
    TEST(OPERATORS, OperatorDouble)
    {
        double A, B;
        A = UniformRandomNumber<double>(0.0, 10.0);
        B = UniformRandomNumber<double>(0.0, 10.0);

        callingInfo("lwMul");
        struct Operator<double, double, double, LWTENSOR_OP_MUL> mul;
        EXPECT_TRUE(lwIsEqual(mul.execute(A, B), lwMul(lwGet<double>(A), lwGet<double>(B))));

        callingInfo("lwAdd");
        struct Operator<double, double, double, LWTENSOR_OP_ADD> add;
        EXPECT_TRUE(lwIsEqual(add.execute(A, B), lwAdd(lwGet<double>(A), lwGet<double>(B))));

        /* CHANGE: OP_SUB is no longer supported. */
        //struct Operator<double, double, double, LWTENSOR_OP_SUB> sub;
        //EXPECT_TRUE(lwIsEqual(sub.execute(A, B), lwSub(lwGet<double>(A), lwGet<double>(B))));

        callingInfo("lwMax");
        struct Operator<double, double, double, LWTENSOR_OP_MAX> max;
        EXPECT_TRUE(lwIsEqual(max.execute(A, B), lwMax(lwGet<double>(A), lwGet<double>(B))));
    }

    /**
     * @brief Test the function Operator<lwComplex, lwComplex, lwComplex, LWTENSOR_OP_MAX>
     * pass criteria: output  should be as expected
     */
    TEST(OPERATORS, OperatorlwComplex)
    {
        lwComplex A, B;
        A = UniformRandomNumber<lwComplex>(0.0, 10.0);
        B = UniformRandomNumber<lwComplex>(0.0, 10.0);

        callingInfo("lwMul");
        struct Operator<lwComplex, lwComplex, lwComplex, LWTENSOR_OP_MUL> mul;
        EXPECT_TRUE(lwIsEqual(mul.execute(A, B), lwMul(lwGet<lwComplex>(A), lwGet<lwComplex>(B))));

        callingInfo("lwAdd");
        struct Operator<lwComplex, lwComplex, lwComplex, LWTENSOR_OP_ADD> add;
        EXPECT_TRUE(lwIsEqual(add.execute(A, B), lwAdd(lwGet<lwComplex>(A), lwGet<lwComplex>(B))));

        /* CHANGE: OP_SUB is no longer supported. */
        //struct Operator<lwComplex, lwComplex, lwComplex, LWTENSOR_OP_SUB> sub;
        //EXPECT_TRUE(lwIsEqual(sub.execute(A, B), lwSub(lwGet<lwComplex>(A), lwGet<lwComplex>(B))));

        callingInfo("lwMax");
        struct Operator<lwComplex, lwComplex, lwComplex, LWTENSOR_OP_MAX> max;
        EXPECT_TRUE(lwIsEqual(max.execute(A, B), lwMax(lwGet<lwComplex>(A), lwGet<lwComplex>(B))));
    }

    /**
     * @brief Test the function Operator<lwDoubleComplex, lwDoubleComplex, lwDoubleComplex, LWTENSOR_OP_MAX>
     * pass criteria: output  should be as expected
     */
    TEST(OPERATORS, OperatorlwDoubleComplex)
    {
        lwDoubleComplex A, B;
        A = UniformRandomNumber<lwDoubleComplex>(0.0, 10.0);
        B = UniformRandomNumber<lwDoubleComplex>(0.0, 10.0);

        callingInfo("lwMul");
        struct Operator<lwDoubleComplex, lwDoubleComplex, lwDoubleComplex, LWTENSOR_OP_MUL> mul;
        EXPECT_TRUE(lwIsEqual(mul.execute(A, B), lwMul(lwGet<lwDoubleComplex>(A), lwGet<lwDoubleComplex>(B))));

        callingInfo("lwAdd");
        struct Operator<lwDoubleComplex, lwDoubleComplex, lwDoubleComplex, LWTENSOR_OP_ADD> add;
        EXPECT_TRUE(lwIsEqual(add.execute(A, B), lwAdd(lwGet<lwDoubleComplex>(A), lwGet<lwDoubleComplex>(B))));

        /* CHANGE: OP_SUB is no longer supported. */
        //struct Operator<lwDoubleComplex, lwDoubleComplex, lwDoubleComplex, LWTENSOR_OP_SUB> sub;
        //EXPECT_TRUE(lwIsEqual(sub.execute(A, B), lwSub(lwGet<lwDoubleComplex>(A), lwGet<lwDoubleComplex>(B))));

        callingInfo("lwMax");
        struct Operator<lwDoubleComplex, lwDoubleComplex, lwDoubleComplex, LWTENSOR_OP_MAX> max;
        EXPECT_TRUE(lwIsEqual(max.execute(A, B), lwMax(lwGet<lwDoubleComplex>(A), lwGet<lwDoubleComplex>(B))));
    }

    /**
     * @brief Test the function lwtensorUnaryOp<lwComplex>
     * pass criteria: output  should be as expected
     */
    TEST(OPERATORS, lwtensorUnaryOplwComplex)
    {
        lwComplex ti = UniformRandomNumber<lwComplex>(0.0, 10.0);
        callingInfo("lwtensorUnaryOp<lwComplex>");
        lwComplex t1 = lwtensorUnaryOp<lwComplex>(ti, LWTENSOR_OP_IDENTITY);
        EXPECT_TRUE(lwIsEqual(t1, ti));

        /* CHANGE: OP_SQRT is now sqrt and does not support complex. */
        //callingInfo("lwtensorUnaryOp<lwComplex>");
        //lwComplex t2 = lwtensorUnaryOp<lwComplex>(ti, LWTENSOR_OP_SQRT);
        //EXPECT_TRUE(lwIsEqual(t2, lwCmulf(ti, ti)));

        callingInfo("lwtensorUnaryOp<lwComplex>");
        lwComplex t3 = lwtensorUnaryOp<lwComplex>(ti, LWTENSOR_OP_CONJ);
        EXPECT_TRUE(lwIsEqual(t3, lwConjf(ti)));

        /* TODO: RELU supported is depreciated from lwComplext */
        //callingInfo("lwtensorUnaryOp<lwComplex>");
        //lwComplex t3 = lwtensorUnaryOp<lwComplex>(ti, LWTENSOR_OP_RELU);
        //EXPECT_TRUE(lwIsEqual(t3, ti.x > 0 && ti.y > 0 ? ti : make_lwComplex(0.0,0.0)));

        //callingInfo("lwtensorUnaryOp<lwComplex>");
        //        lwComplex t4 = lwtensorUnaryOp<lwComplex>(ti, LWTENSOR_OP_ADD);
        //        EXPECT_TRUE(lwIsEqual(t4, make_lwComplex(0.0, 0.0)));
    }

    /**
     * \brief Test the function lwtensorUnaryOp
     * \test validate the functionality of operator lwtensorUnaryOp with type int
     * \depends None
     * \setup None
     * \testprocedure ./apiTest --gtest_filter=lwtensorUnaryOpIntTest.lwtensorUnaryOpInt
     * \teardown None
     * \inputs None
     * \outputs None
     * \expected States the expected value
     */
    TEST(OPERATORS, lwtensorUnaryOplwDoubleComplex)
    {
        callingInfo("lwtensorUnaryOp<lwDoubleComplex>");
        lwDoubleComplex ti = UniformRandomNumber<lwDoubleComplex>(0.0, 10.0);
        lwDoubleComplex t1 = lwtensorUnaryOp<lwDoubleComplex>(ti, LWTENSOR_OP_IDENTITY);
        EXPECT_TRUE(lwIsEqual(t1, ti));

        /* CHANGE: OP_SQRT is now sqrt and does not support complex. */
        //lwDoubleComplex t2 = lwtensorUnaryOp<lwDoubleComplex>(ti, LWTENSOR_OP_SQRT);
        //EXPECT_TRUE(lwIsEqual(t2, lwCmul(ti, ti)));

        callingInfo("lwtensorUnaryOp<lwDoubleComplex>");
        lwDoubleComplex t3 = lwtensorUnaryOp<lwDoubleComplex>(ti, LWTENSOR_OP_CONJ);
        EXPECT_TRUE(lwIsEqual(t3, lwConj(ti)));

        // lwDoubleComplex t3 = lwtensorUnaryOp<lwDoubleComplex>(ti, LWTENSOR_OP_RELU);
        // EXPECT_TRUE(lwIsEqual(t3, ti.x > 0 && ti.y > 0 ? ti : make_lwDoubleComplex(0.0,0.0)));

        //        lwDoubleComplex t4 = lwtensorUnaryOp<lwDoubleComplex>(ti, LWTENSOR_OP_ADD);
        //        EXPECT_TRUE(lwIsEqual(t4, make_lwDoubleComplex(0.0, 0.0)));
    }

    /**
     * @brief Test the function lwtensorBinaryOp<lwComplex >
     * pass criteria: output  should be as expected
     */
    TEST(OPERATORS, lwtensorBinaryOplwComplex )
    {
        lwComplex  ti1 = UniformRandomNumber<lwComplex >(0.0, 10.0);
        lwComplex  ti2 = UniformRandomNumber<lwComplex >(0.0, 10.0);
        callingInfo("lwtensorBinaryOp<lwComplex>");
        lwComplex  to1 = lwtensorBinaryOp<lwComplex >(ti1, ti2, LWTENSOR_OP_ADD);
        EXPECT_TRUE(lwIsEqual(to1, lwCaddf(ti1, ti2)));

        callingInfo("lwtensorBinaryOp<lwComplex>");
        lwComplex  to2 = lwtensorBinaryOp<lwComplex >(ti1, ti2, LWTENSOR_OP_MUL);
        EXPECT_TRUE(lwIsEqual(to2, lwCmulf(ti1, ti2)));

        //        lwComplex  to3 = lwtensorBinaryOp<lwComplex >(ti1, ti2, LWTENSOR_OP_IDENTITY, NULL);
        //        EXPECT_TRUE(lwIsEqual(to3, make_lwComplex(0.0, 0.0)));
    }

    /**
     * @brief Test the function lwtensorBinaryOp<lwDoubleComplex >
     * pass criteria: output  should be as expected
     */
    TEST(OPERATORS, lwtensorBinaryOplwDoubleComplex )
    {
        lwDoubleComplex  ti1 = UniformRandomNumber<lwDoubleComplex >(0.0, 10.0);
        lwDoubleComplex  ti2 = UniformRandomNumber<lwDoubleComplex >(0.0, 10.0);
        callingInfo("lwtensorBinaryOp<lwDoubleComplex>");
        lwDoubleComplex  to1 = lwtensorBinaryOp<lwDoubleComplex >(ti1, ti2, LWTENSOR_OP_ADD);
        EXPECT_TRUE(lwIsEqual(to1, lwCadd(ti1, ti2)));

        callingInfo("lwtensorBinaryOp<lwDoubleComplex>");
        lwDoubleComplex  to2 = lwtensorBinaryOp<lwDoubleComplex >(ti1, ti2, LWTENSOR_OP_MUL);
        EXPECT_TRUE(lwIsEqual(to2, lwCmul(ti1, ti2)));

        //        lwDoubleComplex  to3 = lwtensorBinaryOp<lwDoubleComplex >(ti1, ti2, LWTENSOR_OP_IDENTITY, NULL);
        //        EXPECT_TRUE(lwIsEqual(to3, make_lwDoubleComplex(0.0, 0.0)));
    }

    /**
     * @brief Test the function lw2Norm_lwComplex
     * pass criteria: output  should be as expected
     */
    TEST(OPERATORS, lw2Norm_lwComplex)
    {
        lwComplex a = make_lwComplex(1.0f, 2.0f);
        double b = std::sqrt(lwSquare2Norm(a));
        callingInfo("lw2Norm");
        double c = lw2Norm(a);
        EXPECT_EQ(b, c);
    }

    /**
     * @brief Test the function lw2Norm_lwDoubleComplex
     * pass criteria: output  should be as expected
     */
    TEST(OPERATORS, lw2Norm_lwDoubleComplex)
    {
        lwDoubleComplex a = make_lwDoubleComplex(1.0, 2.0);
        double b = std::sqrt(lwSquare2Norm(a));
        callingInfo("lw2Norm");
        double c = lw2Norm(a);
        EXPECT_EQ(b, c);
    }

} //namespace
