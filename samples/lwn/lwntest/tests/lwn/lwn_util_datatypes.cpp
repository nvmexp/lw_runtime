/*
 * Copyright (c) 2015 LWPU Corporation.  All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

//
// lwn_util_datatypes.cpp
//
// Basic unit tests for LWN data type classes defined in lwn_datatypes.h.
//
#include "lwntest_cpp.h"
#include "lwn_utils.h"
using namespace lwn;

#define LWN_DATATYPES_LOG_OUTPUT    0

#if LWN_DATATYPES_LOG_OUTPUT >= 2
#define SHOULD_LOG(_result)     true
#define LOG(x)                  printf x
#define LOG_INFO(x)             printf x
#elif LWN_DATATYPES_LOG_OUTPUT >= 1
#define SHOULD_LOG(_result)     (!(_result))
#define LOG(x)                  printf x
#define LOG_INFO(x)
#else
#define SHOULD_LOG(_result)     false
#define LOG(x)
#define LOG_INFO(x)
#endif

class LWNDataTypeTest
{
    // DataBuffer:  union holding storage for a few data values of various types.
    union DataBuffer {
        uint8_t  ub[32];
        int8_t   b[32];
        uint16_t us[32];
        int16_t  s[32];
        uint32_t ui[32];
        int32_t  i[32];
        float    f[32];
    };

    // Static variable holding the name of the data type used for the current
    // subtest.
    static const char *subtest;

    // Function to log results of an individual test to the console.
    static void log(bool result, int line, const char *cond);

    // checkFormatEnum:  Checks that the registered format enum for type <T>
    // matches <F>.
    template <typename T, LWNformat F> static bool checkFormatEnum();

    // checkRawMemory:  Checks that raw memory values in <ilwalues>, when read
    // as a vector type, are properly colwerted to "external" values in
    // <outValues> and that raw "external" values in <outValues> are properly
    // colwerted to a vector whose memory layout matches <repackedValues>.
    template <typename T, int N, bool isPacked>
    static bool checkRawMemory(int lwalues,
                               const typename T::StoredComponentType *ilwalues,
                               const typename T::ExternalScalarType *outValues,
                               const typename T::StoredComponentType *repackedValues);

    // checkCore*:  Checks that various operators and supported (no compile
    // errors) and behave properly.  <ref> is a reference vector with values
    // (4,3,2,1) or subsets thereof that exercise the "from scalars"
    // constructor.
    template <typename T, int N> static bool checkCore(T ref);
    template <typename T, int N> static bool checkCoreNormalized(T ref, bool isSigned);

    // check*:  Template functions to perform general checks for various
    // different data type classes.
    template <typename T, int N, LWNformat F> static bool checkBool(T ref);
    template <typename T, int N, LWNformat F> static bool checkF32(T ref);
    template <typename T, int N, LWNformat F> static bool checkU32(T ref);
    template <typename T, int N, LWNformat F> static bool checkS32(T ref);
    template <typename T, int N, LWNformat F> static bool checkF16(T ref);
    template <typename T, int N, LWNformat F> static bool checkUnorm(T ref);
    template <typename T, int N, LWNformat F> static bool checkSnorm(T ref);
    template <typename T, int N, LWNformat F> static bool checkUint(T ref);
    template <typename T, int N, LWNformat F> static bool checkSint(T ref);
    template <typename T, int N, LWNformat F> static bool checkU2F(T ref);
    template <typename T, int N, LWNformat F> static bool checkS2F(T ref);
    template <typename T, int N, LWNformat F> static bool checkPacked16(T ref);
    template <typename T, int N, LWNformat F> static bool checkPacked32F(T ref);
    template <typename T, int N, LWNformat F> static bool checkPacked32UI(T ref);
    template <typename T, int N, LWNformat F> static bool checkPacked32I(T ref);

public:
    LWNTEST_CppMethods();
};

const char * LWNDataTypeTest::subtest = "UNSET";

void LWNDataTypeTest::log(bool result, int line, const char *cond)
{
    if (SHOULD_LOG(result)) {
        LOG(("%s [%s, %d] %s\n", result ? "PASSED" : "FAILED", subtest, line, cond));
    }
}

// CHECK:  Perform an individual check of condition <cond>, setting the
// variable <result> to false if the check fails, and optionally logging the
// results of the test.
#define CHECK(cond)                     \
    do {                                \
        bool lresult = cond;            \
        log(lresult, __LINE__, #cond);  \
        result = result && lresult;     \
    } while (0)

lwString LWNDataTypeTest::getDescription() const
{
    lwStringBuf sb;
    sb << 
        "Basic functional/unit test for LWN data type classes in `lwn_datatypes.h`. "
        "Clears the screen to green if all tests pass; red if any test fails.  Use "
        "the LOG_OUTPUT entries at the top of the source code to get more detailed "
        "failure information.";
    return sb.str();
}

int LWNDataTypeTest::isSupported() const
{
    return lwogCheckLWNAPIVersion(5, 0);
}

template <typename T, LWNformat F>
bool LWNDataTypeTest::checkFormatEnum()
{
    bool result = true;
    T a;

    // Make sure the lwnFormat() functions returns the proper token.
    CHECK(dt::traits<T>::lwnFormat() == F);
    CHECK(lwnFormat(a) == F);

    return result;
}

template <typename T, int N, bool isPacked>
bool LWNDataTypeTest::checkRawMemory(int lwalues,
                                     const typename T::StoredComponentType *ilwalues,
                                     const typename T::ExternalScalarType *outValues,
                                     const typename T::StoredComponentType *repackedValues)
{
    bool result = true;

    // Staging buffers for input, output, and repacked values for each loop
    // iteration.
    typename T::StoredComponentType iv[4];
    typename T::ExternalScalarType ov[4];
    typename T::StoredComponentType rv[4];
        
    // Staging buffers to hold copies of <iv> and <ov> for packing.
    T storedVector;
    typename T::ExternalVectorType externalVector;

    // Staging buffers holding values for packed and unpacked data using the
    // vector type <T>.
    typename T::ExternalScalarType unpacked[4];
    typename T::StoredComponentType packed[4];

    // Normally, we are provided with arrays with <lwalues> inputs and outputs
    // and we generate <lwalues> separate vectors by swizzling values around.
    // For packed types, we have <lwalues> packed input values and
    // <lwalues>*<N> output component values.  We don't swizzle in this case.
    for (int i = 0; i < lwalues; i++) {
        if (isPacked) {
            iv[0] = ilwalues[i];
            for (int j = 0; j < N; j++) {
                ov[j] = outValues[N*i + j];
            }
            rv[0] = repackedValues[i];
        } else {
            for (int j = 0; j < N; j++) {
                iv[j] = ilwalues[(i + j) % lwalues];
                ov[j] = outValues[(i + j) % lwalues];
                rv[j] = repackedValues[(i + j) % lwalues];
            }
        }

        // We're going to memcpy() between types, so assert that we aren't
        // writing off the end of "raw" arrays.
        ct_assert(sizeof(storedVector) <= sizeof(iv));
        ct_assert(sizeof(storedVector) <= sizeof(packed));
        ct_assert(sizeof(externalVector) <= sizeof(ov));
        ct_assert(sizeof(externalVector) <= sizeof(unpacked));

        // Verify that making a copy of the input values in "raw" form into a
        // vector type and then unpacking to the external type produces the
        // correct output values.
        memcpy(&storedVector, iv, sizeof(storedVector));
        externalVector = (typename T::ExternalVectorType)(storedVector);
        memcpy(unpacked, &externalVector, sizeof(externalVector));
        for (int j = 0; j < N; j++) {
            CHECK(unpacked[j] == ov[j]);
        }

        // Verify that making a copy of the output values in "raw" form into
        // an external vector type and packing to the vector type produces the
        // correct repacked values.
        memcpy(&externalVector, ov, sizeof(externalVector));
        storedVector = T(externalVector);
        memcpy(packed, &storedVector, sizeof(storedVector));
        for (int j = 0; j < (isPacked ? 1 : N); j++) {
            CHECK(packed[j] == rv[j]);
        }
    }

    return result;
}

// Perform various checks to ensure that operators are defined and work
// properly for data types other than signed/unsigned normalized types.
template <typename T, int N>
bool LWNDataTypeTest::checkCore(T ref)
{
    bool result = true;
    T a, b;

    // Initialize a vector to (4,3,2,1) using the setComponent() methods and
    // compare against the (4,3,2,1) reference vector.
    a = T(0);
    for (int i = 0; i < N; i++) {
        a.setComponent(i, 4 - i);
    }
    CHECK(all(ref == a));

    // Compute (3,2,1,0) and perform various comparisons against the (4,3,2,1)
    // vector.
    b = 1;
    a = a - b;
    CHECK(!any(ref == a));
    CHECK(all(ref != a));
    CHECK(!any(ref < a));
    CHECK(!any(ref <= a));
    CHECK(all(ref > a));
    CHECK(all(ref >= a));

    // Test the (3,2,1,0) vector using component selectors.
    for (int i = 0; i < N; i++) {
        CHECK(a[i] == (typename T::ExternalScalarType)(3 - i));
    }

    // Go back to (4,3,2,1) using addition.
    b = 1;
    a = a + b;
    CHECK(all(ref == a));

    // Test multiplication and division; we skip the test for types with 1- or
    // 2-bit alpha channels due to out-of-bounds issues.
    if (N < 4 || a.componentBits(3) > 2) {
        b = 2;
        a = a * b;
        for (int i = 0; i < N; i++) {
            CHECK(a[i] == (typename T::ExternalScalarType)(2 * (4- i)));
        }
        a = a / b;
        CHECK(all(ref == a));
    }

    // Test the unary "+" operator.
    b = +a;
    CHECK(all(ref == b));

    // Test mixed scalar/vector arithmetic operators; we skip some tests for
    // types with 1- or 2-bit alpha channels due to out-of-bounds issues.
    for (int i = 0; i < N; i++) {
        a.setComponent(i, 3 - i);
    }
    CHECK(all(ref == a + 1));
    CHECK(all(ref == 1 + a));
    CHECK(all(a == ref - 1));
    if (N < 4 || a.componentBits(3) > 2) {
        for (int i = 0; i < N; i++) {
            a.setComponent(i, i);
        }
        CHECK(all(a == 4 - ref));
        for (int i = 0; i < N; i++) {
            a.setComponent(i, 2 * (4 - i));
        }
        CHECK(all(a == ref * 2));
        CHECK(all(a == 2 * ref));
        CHECK(all(ref == a / 2));
        for (int i = 0; i < N; i++) {
            a.setComponent(i, 4 - i);
            b.setComponent(i, 12 / (4 - i));
        }
        CHECK(all(b == 12 / a));
    }

    // Test mixed scalar/vector comparison operators.
    CHECK(!any(ref == 0));
    CHECK(!any(0 == ref));
    CHECK(all(ref != 0));
    CHECK(all(0 != ref));
    CHECK(all(ref < 5));
    CHECK(!any(5 < ref));
    CHECK(all(ref <= 4));
    CHECK(!any(5 <= ref));
    CHECK(!any(ref > 5));
    CHECK(all(5 > ref));
    CHECK(!any(ref >= 5));
    CHECK(all(4 >= ref));

    return result;
}

// Perform various checks to ensure that operators are defined and work
// properly for signed/unsigned normalized types.
template <typename T, int N>
bool LWNDataTypeTest::checkCoreNormalized(T ref, bool isSigned)
{
    bool result = true;
    int maxValue[N];
    T a, b;

    // Initialize a floating-point vector with 1 LSB values for each component.
    typename T::ExternalVectorType lsb = 1.0;
    for (int i = 0; i < N; i++) {
        int bits = T::componentBits(i);
        if (isSigned) bits = bits - 1;
        maxValue[i] = (1 << bits) - 1;
        lsb[i] /= maxValue[i];
    }

    // Initialize a vector to (4,3,2,1) using component selectors and compare
    // against the (4,3,2,1) reference vector.
    a = T(0.0);
    for (int i = 0; i < N; i++) {
        a.setComponent(i, float(4 - i) / maxValue[i]);
    }
    CHECK(all(ref == a));

    // Compute (3,2,1,0) and perform various comparisons against the (4,3,2,1)
    // vector.
    b = lsb;
    a = a - b;
    CHECK(!any(ref == a));
    CHECK(all(ref != a));
    CHECK(!any(ref < a));
    CHECK(!any(ref <= a));
    CHECK(all(ref > a));
    CHECK(all(ref >= a));

    // Test the (3,2,1,0) vector using component selectors.  We use screwy
    // math here to replicate the normalized component extraction logic and
    // avoid failures due to FP rounding errors.
    for (int i = 0; i < N; i++) {
        float ec = a[i];
        float rv = float(3 - i) / maxValue[i];
        CHECK(ec == rv);
    }

    // Go back to (1,2,3,4) using addition.
    b = lsb;
    a = a + b;
    CHECK(all(ref == a));

    // Test multiplication and division; we skip the test for types with 1- or
    // 2-bit alpha channels due to out-of-bounds issues.
    if (N < 4 || a.componentBits(3) > 2) {
        a = a * (typename T::ExternalVectorType)(2.0);
        for (int i = 0; i < N; i++) {
            float ec = a[i];
            float rv = float(2 * (4 - i)) / maxValue[i];
            CHECK(ec == rv);
        }
        a = a / (typename T::ExternalVectorType)(2.0);
        CHECK(all(ref == a));
    }

    // Test the unary "+" operator.
    b = +a;
    CHECK(all(ref == b));

    // Test a few scalar/vector arithmetic operators, which is trickier with
    // values constrained to small ranges.
    b = 1.0 - ref;
    for (int i = 0; i < N; i++) {
        float ec = b[i];
        float rv = float(maxValue[i] - (4 - i)) / maxValue[i];
        CHECK(ec == rv);
    }
    if (N < 4 || T::componentBits(3) > 2) {
        b = 2.0 * ref;
        for (int i = 0; i < N; i++) {
            float ec = b[i];
            float rv = float(2 * (4 - i)) / maxValue[i];
            CHECK(ec == rv);
        }
        b = b / 2.0;
        CHECK(all(ref == b));
    }

    // Test mixed scalar/vector comparison operators.  Note that "1" here is
    // 1.0 (maximum value).  We skip some tests for really small alpha
    // channels.
    CHECK(!any(ref == 0));
    CHECK(!any(0 == ref));
    CHECK(all(ref != 0));
    CHECK(all(0 != ref));
    if (N < 4 || T::componentBits(3) > 2) {
        CHECK(all(ref < 1));
        CHECK(!any(1 < ref));
    }
    CHECK(all(ref <= 1));
    if (N < 4 || T::componentBits(3) > 2) {
        CHECK(!any(1 <= ref));
        CHECK(!any(ref > 1));
        CHECK(all(1 > ref));
        CHECK(!any(ref >= 1));
    }
    CHECK(all(1 >= ref));

    return result;
}

// Perform various checks to ensure that operators are defined and work
// properly for our internal boolean types.
template <typename T, int N, LWNformat F>
bool LWNDataTypeTest::checkBool(T ref)
{
    bool result = true;
    T a, b;

    // No LWN formats are defined for boolean types, so don't call
    // checkFormatEnum().

    // Initialize <a> and <b> to all-true and all-false using component selector.
    for (int i = 0; i < N; i++) {
        a[i] = true;
        b[i] = false;
    }

    // Test for working any() and all() functions.
    CHECK(any(a));
    CHECK(all(a));
    CHECK(!any(b));
    CHECK(!all(b));

    // Test for correct scalar values extracted by component selectors.
    for (int i = 0; i < N; i++) {
        CHECK(a[0]);
        CHECK(!b[0]);
    }

    // Test the "&&" operator.
    CHECK(all(a && a));
    CHECK(!any(a && b));
    CHECK(!any(b && b));

    // Test the "||" operator.
    CHECK(all(a || a));
    CHECK(all(a || b));
    CHECK(!any(b || b));

    // Test the "!" (not) operator.
    a = !a;
    b = !b;
    CHECK(!any(a));
    CHECK(all(b));

    // Check mixed-value vectors.
    for (int i = 0; i < N; i++) {
        a[i] = ((i & 1) == 0);
        b[i] = ((i & 1) != 0);
    }
    CHECK(all(a == (!b)));
    CHECK(all((!a) == b));
    CHECK(all(a || b));
    CHECK(!any(a && b));

    return result;
}

template <typename T, int N, LWNformat F>
bool LWNDataTypeTest::checkF32(T ref)
{
    bool result = true;
    T a, b;
    if (!checkFormatEnum<T, F>()) {
        result = false;
    }
    if (!checkCore<T, N>(ref)) {
        result = false;
    }

    // Test that component selection operators work properly.
    for (int i = 0; i < N; i++) {
        a[i] = 4 - i;
    }
    CHECK(all(a == ref));
    for (int i = 0; i < N; i++) {
        CHECK(a[i] == 4 - i);
    }

    // Test the unary negation operator.
    CHECK(all(-ref < 0));
    CHECK(all(((-ref) * (-1)) == ref));

    // Test subtraction producing negative values.
    a = 0;
    b = a - ref;
    for (int i = 0; i < N; i++) {
        CHECK(b[i] == -(4 - i));
    }

    // Test operators with floating-point constants.
    a = ref - 1;
    CHECK(all(ref == 1.0 + a));
    CHECK(all(ref == 1.0f + a));
    CHECK(all(ref == a + 1.0));
    CHECK(all(ref == a + 1.0f));

    // Test that raw values are handled properly.
    LWNfloat v[] = { 0.0, 0.25, 1.0, 4.0, -0.0, -0.25, -1.0, -4.0 };
    if (!checkRawMemory<T, N, false>(8, v, v, v)) {
        result = false;
    }

    return result;
}

template <typename T, int N, LWNformat F>
bool LWNDataTypeTest::checkU32(T ref)
{
    bool result = true;
    T a, b;
    if (!checkFormatEnum<T, F>()) {
        result = false;
    }
    if (!checkCore<T, N>(ref)) {
        result = false;
    }

    // Test that component selection operators work properly.
    for (int i = 0; i < N; i++) {
        a[i] = 4 - i;
    }
    CHECK(all(a == ref));
    for (int i = 0; i < N; i++) {
        CHECK(a[i] == (typename T::ExternalScalarType)(4 - i));
    }

    // Test that raw values are handled properly.
    LWNuint v[] = { 0, 1, 2, 4, 7, 0x7FFFFFFF, 0x80000000, 0xFFFFFFFF };
    if (!checkRawMemory<T, N, false>(8, v, v, v)) {
        result = false;
    }

    return result;
}

template <typename T, int N, LWNformat F>
bool LWNDataTypeTest::checkS32(T ref)
{
    bool result = true;
    T a, b;

    if (!checkFormatEnum<T, F>()) {
        result = false;
    }
    if (!checkCore<T, N>(ref)) {
        result = false;
    }

    // Test that component selection operators work properly.
    for (int i = 0; i < N; i++) {
        a[i] = 4 - i;
    }
    CHECK(all(a == ref));
    for (int i = 0; i < N; i++) {
        CHECK(a[i] == 4 - i);
    }

    // Test the unary negation operator.
    CHECK(all(-ref < 0));
    CHECK(all(((-ref) * (-1)) == ref));

    // Test subtraction producing negative values.
    a = 0;
    b = a - ref;
    for (int i = 0; i < N; i++) {
        CHECK(b[i] == -(4 - i));
    }

    // Test that raw values are handled properly.
    LWNint v[] = { 0, 1, -1, 12, -31, 0x7FFFFFFF, -0x7FFFFFFF, -0x7FFFFFFF-1 };
    if (!checkRawMemory<T, N, false>(8, v, v, v)) {
        result = false;
    }

    return result;
}


template <typename T, int N, LWNformat F>
bool LWNDataTypeTest::checkF16(T ref)
{
    bool result = true;
    T a, b;
    typename T::ExternalVectorType a32, b32;

    if (!checkFormatEnum<T, F>()) {
        result = false;
    }
    if (!checkCore<T, N>(ref)) {
        result = false;
    }

    // Test that component selection operators work properly.
    for (int i = 0; i < N; i++) {
        a[i] = 4 - i;
    }
    CHECK(all(a == ref));
    for (int i = 0; i < N; i++) {
        CHECK(a[i] == 4 - i);
    }

    // Test the unary negation operator.
    CHECK(all(-ref < 0));
    CHECK(all(((-ref) * (-1)) == ref));

    // Test subtraction producing negative values.
    a = 0;
    b = a - ref;
    for (int i = 0; i < N; i++) {
        CHECK(b[i] == -(4 - i));
    }

    // Test operators with floating-point constants.
    a = ref - 1;
    CHECK(all(ref == 1.0 + a));
    CHECK(all(ref == 1.0f + a));
    CHECK(all(ref == a + 1.0));
    CHECK(all(ref == a + 1.0f));

    // Test that raw values are handled properly.
    uint16_t packed[] = { 0x0000, 0x3400, 0x3C00, 0x4400, 0x8000, 0xB400, 0xBC00, 0xC400 };
    LWNfloat unpacked[] = { 0.0, 0.25, 1.0, 4.0, -0.0, -0.25, -1.0, -4.0 };
    if (!checkRawMemory<T, N, false>(8, packed, unpacked, packed)) {
        result = false;
    }

    return result;
}

template <typename T, int N, LWNformat F>
bool LWNDataTypeTest::checkUnorm(T ref)
{
    bool result = true;
    typename T::StoredComponentType maxValue = (1U << T::componentBits(0)) - 1;
    T a;

    if (!checkFormatEnum<T, F>()) {
        result = false;
    }
    if (!checkCoreNormalized<T, N>(ref, false)) {
        result = false;
    }

    // Test that component selection operators work properly.
    for (int i = 0; i < N; i++) {
        a[i] = float(4 - i) / maxValue;
    }
    CHECK(all(a == ref));
    for (int i = 0; i < N; i++) {
        double rv = double(4 - i) / maxValue;
        CHECK(a[i] == rv);
    }

    // Test that raw values are handled properly.
    typename T::StoredComponentType packed[4];
    typename T::ExternalScalarType unpacked[4] = { 0.0, 1.0 / 3, 2.0 / 3, 1.0 };
    packed[0] = 0;
    packed[1] = maxValue / 3;
    packed[2] = 2 * maxValue / 3;
    packed[3] = maxValue;
    if (!checkRawMemory<T, N, false>(4, packed, unpacked, packed)) {
        result = false;
    }

    return result;
}

template <typename T, int N, LWNformat F>
bool LWNDataTypeTest::checkSnorm(T ref)
{
    bool result = true;
    T a, b;
    typename T::StoredComponentType maxValue = (1 << (T::componentBits(0) - 1)) - 1;

    if (!checkFormatEnum<T, F>()) {
        result = false;
    }
    if (!checkCoreNormalized<T, N>(ref, true)) {
        result = false;
    }

    // Test that component selection operators work properly.
    for (int i = 0; i < N; i++) {
        a[i] = float(4 - i) / maxValue;
    }
    CHECK(all(a == ref));
    for (int i = 0; i < N; i++) {
        double rv = double(4 - i) / maxValue;
        CHECK(a[i] == rv);
    }

    // Test the unary negation operator.
    CHECK(all(-ref < 0));
    CHECK(all(((-ref) * (-1)) == ref));

    // Test subtraction producing negative values.
    a = 0;
    b = a - ref;
    for (int i = 0; i < N; i++) {
        double rv = -double(4 - i) / maxValue;
        CHECK(b[i] == rv);
    }

    // Test that raw values are handled properly.
    typename T::StoredComponentType packed[6];
    typename T::ExternalScalarType unpacked[6];
    typename T::StoredComponentType repacked[6];
    packed[0] = 0;
    packed[1] = maxValue / 2;
    packed[2] = maxValue;
    packed[3] = -maxValue / 2;
    packed[4] = -maxValue;          // for S8, -127 (0x81) represents -1.0
    packed[5] = -maxValue - 1;      // for S8, -128 is clamped to also be -1.0
    for (int i = 0; i < 6; i++) {
        unpacked[i] = float(packed[i]) / maxValue;
        repacked[i] = packed[i];
    }
    unpacked[5] = -1.0;             // deal with clamping
    repacked[5] = repacked[4];
    if (!checkRawMemory<T, N, false>(6, packed, unpacked, repacked)) {
        result = false;
    }

    return result;
}

template <typename T, int N, LWNformat F>
bool LWNDataTypeTest::checkUint(T ref)
{
    bool result = true;
    T a;
    if (!checkFormatEnum<T, F>()) {
        result = false;
    }
    if (!checkCore<T, N>(ref)) {
        result = false;
    }

    // Test that component selection operators work properly.
    for (int i = 0; i < N; i++) {
        a[i] = 4 - i;
    }
    CHECK(all(a == ref));
    for (int i = 0; i < N; i++) {
        CHECK(a[i] == (typename T::ExternalScalarType)(4 - i));
    }

    // Test that raw values are handled properly.
    typename T::StoredComponentType packed[6];
    typename T::ExternalScalarType unpacked[6];
    typename T::StoredComponentType maxValue = ~0U >> (32 - T::componentBits(0));
    packed[0] = 0;
    packed[1] = 1;
    packed[2] = maxValue >> 2;
    packed[3] = maxValue >> 1;
    packed[4] = maxValue - 1;
    packed[5] = maxValue;
    for (int i = 0; i < 6; i++) {
        unpacked[i] = packed[i];
    }
    if (!checkRawMemory<T, N, false>(6, packed, unpacked, packed)) {
        result = false;
    }

    return result;
}

template <typename T, int N, LWNformat F>
bool LWNDataTypeTest::checkSint(T ref)
{
    bool result = true;
    T a, b;
    if (!checkFormatEnum<T, F>()) {
        result = false;
    }
    if (!checkCore<T, N>(ref)) {
        result = false;
    }

    // Test that component selection operators work properly.
    for (int i = 0; i < N; i++) {
        a[i] = 4 - i;
    }
    CHECK(all(a == ref));
    for (int i = 0; i < N; i++) {
        CHECK(a[i] == 4 - i);
    }

    // Test the unary negation operator.
    CHECK(all(-ref < 0));
    CHECK(all(((-ref) * (-1)) == ref));

    // Test subtraction producing negative values.
    a = 0;
    b = a - ref;
    for (int i = 0; i < N; i++) {
        CHECK(b[i] == -(4 - i));
    }

    // Test that raw values are handled properly.
    typename T::StoredComponentType packed[6];
    typename T::ExternalScalarType unpacked[6];
    typename T::StoredComponentType maxValue = ~0U >> (31 - T::componentBits(0));
    packed[0] = 0;
    packed[1] = 1;
    packed[2] = maxValue;
    packed[3] = -1;
    packed[4] = -maxValue;
    packed[5] = -maxValue - 1;
    for (int i = 0; i < 6; i++) {
        unpacked[i] = packed[i];
    }
    if (!checkRawMemory<T, N, false>(6, packed, unpacked, packed)) {
        result = false;
    }

    return result;
}

template <typename T, int N, LWNformat F>
bool LWNDataTypeTest::checkU2F(T ref)
{
    bool result = true;
    T a;
    if (!checkFormatEnum<T, F>()) {
        result = false;
    }
    if (!checkCore<T, N>(ref)) {
        result = false;
    }

    // Test that component selection operators work properly.
    for (int i = 0; i < N; i++) {
        a[i] = 4 - i;
    }
    CHECK(all(a == ref));
    for (int i = 0; i < N; i++) {
        CHECK(a[i] == 4 - i);
    }

    // Test operators with floating-point constants.
    a = ref - 1;
    CHECK(all(ref == 1.0 + a));
    CHECK(all(ref == 1.0f + a));
    CHECK(all(ref == a + 1.0));
    CHECK(all(ref == a + 1.0f));

    // Test that raw values are handled properly.
    typename T::StoredComponentType packed[6];
    typename T::ExternalScalarType unpacked[6];
    typename T::StoredComponentType maxValue = ~0U >> (32 - T::componentBits(0));
    if (T::componentBits(0) > 16) {
        maxValue = 1U << (T::componentBits(0) - 1);
    }
    packed[0] = 0;
    packed[1] = 1;
    packed[2] = 4;
    packed[3] = maxValue >> 2;
    packed[4] = maxValue >> 1;
    packed[5] = maxValue;
    for (int i = 0; i < 6; i++) {
        unpacked[i] = packed[i];
    }
    if (!checkRawMemory<T, N, false>(6, packed, unpacked, packed)) {
        result = false;
    }

    return result;
}

template <typename T, int N, LWNformat F>
bool LWNDataTypeTest::checkS2F(T ref)
{
    bool result = true;
    T a, b;
    if (!checkFormatEnum<T, F>()) {
        result = false;
    }
    if (!checkCore<T, N>(ref)) {
        result = false;
    }

    // Test that component selection operators work properly.
    for (int i = 0; i < N; i++) {
        a[i] = 4 - i;
    }
    CHECK(all(a == ref));
    for (int i = 0; i < N; i++) {
        CHECK(a[i] == 4 - i);
    }

    // Test the unary negation operator.
    CHECK(all(-ref < 0));
    CHECK(all(((-ref) * (-1)) == ref));

    // Test subtraction producing negative values.
    a = 0;
    b = a - ref;
    for (int i = 0; i < N; i++) {
        CHECK(b[i] == -(4 - i));
    }

    // Test operators with floating-point constants.
    a = ref - 1;
    CHECK(all(ref == 1.0 + a));
    CHECK(all(ref == 1.0f + a));
    CHECK(all(ref == a + 1.0));
    CHECK(all(ref == a + 1.0f));

    // Test that raw values are handled properly.
    typename T::StoredComponentType packed[6];
    typename T::ExternalScalarType unpacked[6];
    typename T::StoredComponentType maxValue = ~0U >> (31 - T::componentBits(0));
    if (T::componentBits(0) > 16) {
        maxValue = 1 << (T::componentBits(0) - 2);
    }
    packed[0] = 0;
    packed[1] = 1;
    packed[2] = 4;
    packed[3] = maxValue >> 2;
    packed[4] = maxValue >> 1;
    packed[5] = maxValue;
    for (int i = 0; i < 6; i++) {
        unpacked[i] = packed[i];
    }
    if (!checkRawMemory<T, N, false>(6, packed, unpacked, packed)) {
        result = false;
    }

    return result;
}

template <typename T, int N, LWNformat F>
bool LWNDataTypeTest::checkPacked16(T ref)
{
    bool result = true;
    if (!checkFormatEnum<T, F>()) {
        result = false;
    }
    if (!checkCoreNormalized<T, N>(ref, false)) {
        // All 16- bit packed types are UNORM.
        result = false;
    }

    // Test that component selection operators work properly.  For packed
    // types, component selectors return an external value that can only be
    // used as an r-value.
    for (int i = 0; i < N; i++) {
        int maxValue = (1 << T::componentBits(i)) - 1;
        float rv = float(4 - i) / maxValue;
        CHECK(ref[i] == rv);
    }

    // Test that raw values are handled properly -- logic is format-specific.
    DataBuffer packed;
    DataBuffer unpacked;
    switch (F) {
    case LWN_FORMAT_RGBA4:
        packed.us[0] = 0x3210;
        packed.us[1] = 0x7654;
        packed.us[2] = 0xBA98;
        packed.us[3] = 0xFEDC;
        for (int i = 0; i < 16; i++) {
            unpacked.f[i] = float(i) / 15;
        }
        break;
    case LWN_FORMAT_RGB5:
    case LWN_FORMAT_BGR5:
        packed.us[0] = (0x00 << 0) | (0x0A << 5) | (0x14 << 10);
        packed.us[1] = (0x0A << 0) | (0x14 << 5) | (0x1F << 10);
        packed.us[2] = (0x14 << 0) | (0x1F << 5) | (0x00 << 10);
        packed.us[3] = (0x1F << 0) | (0x00 << 5) | (0x0A << 10);
        for (int i = 0; i < 12; i++) {
            int value = ((i % 3) + (i / 3)) % 4;
            value = (value * 10) + (value == 3 ? 1 : 0);
            unpacked.f[i] = float(value) / 31;
        }
        break;
    case LWN_FORMAT_RGB5A1:
    case LWN_FORMAT_BGR5A1:
        packed.us[0] = (0x00 << 0) | (0x0A << 5) | (0x14 << 10);
        packed.us[1] = (0x0A << 0) | (0x14 << 5) | (0x1F << 10) | 0x8000;
        packed.us[2] = (0x14 << 0) | (0x1F << 5) | (0x00 << 10);
        packed.us[3] = (0x1F << 0) | (0x00 << 5) | (0x0A << 10) | 0x8000;
        for (int i = 0; i < 16; i++) {
            if ((i % 4) == 3) {
                unpacked.f[i] = (i & 0x4) ? 1.0 : 0.0;
            } else {
                int value = ((i % 4) + (i / 4)) % 4;
                value = (value * 10) + (value == 3 ? 1 : 0);
                unpacked.f[i] = float(value) / 31;
            }
        }
        break;
    case LWN_FORMAT_RGB565:
    case LWN_FORMAT_BGR565:
        packed.us[0] = (0x00 << 0) | (0x14 << 5) | (0x14 << 11);
        packed.us[1] = (0x0A << 0) | (0x28 << 5) | (0x1F << 11);
        packed.us[2] = (0x14 << 0) | (0x3F << 5) | (0x00 << 11);
        packed.us[3] = (0x1F << 0) | (0x00 << 5) | (0x0A << 11);
        for (int i = 0; i < 12; i++) {
            int value = ((i % 3) + (i / 3)) % 4;
            if ((i % 3) == 1) {
                value = (value * 20) + (value == 3 ? 3 : 0);
                unpacked.f[i] = float(value) / 63;
            } else {
                value = (value * 10) + (value == 3 ? 1 : 0);
                unpacked.f[i] = float(value) / 31;
            }
        }
        break;
    default:
        assert(0);
        break;
    }

    // Swap red and blue in the expected values for BGR formats.
    if (F == LWN_FORMAT_BGR5 || F == LWN_FORMAT_BGR565) {
        for (int i = 0; i < 12; i += 3) {
            float v = unpacked.f[i];
            unpacked.f[i] = unpacked.f[i + 2];
            unpacked.f[i + 2] = v;
        }
    } else if (F == LWN_FORMAT_BGR5A1) {
        for (int i = 0; i < 16; i += 4) {
            float v = unpacked.f[i];
            unpacked.f[i] = unpacked.f[i + 2];
            unpacked.f[i + 2] = v;
        }
    }

    if (!checkRawMemory<T, N, true>(4, packed.us, unpacked.f, packed.us)) {
        result = false;
    }

    return result;
}

// Check 32-bit 10/10/10/2 packed types that unpack to float.
template <typename T, int N, LWNformat F>
bool LWNDataTypeTest::checkPacked32F(T ref)
{
    bool result = true;
    if (!checkFormatEnum<T, F>()) {
        result = false;
    }

    // Do core data type/operator checks based on the format.
    bool isSigned = false;
    bool isNormalized = false;
    switch (F) {
    case LWN_FORMAT_RGB10A2:           isSigned = false; isNormalized = true; break;
    case LWN_FORMAT_RGB10A2SN:         isSigned = true;  isNormalized = true; break;
    case LWN_FORMAT_RGB10A2_UI2F:      isSigned = false; isNormalized = false; break;
    case LWN_FORMAT_RGB10A2_I2F:       isSigned = true;  isNormalized = false; break;
    default:                        assert(0); break;
    }
    if (isNormalized) {
        if (!checkCoreNormalized<T, N>(ref, isSigned)) {
            result = false;
        }
    } else {
        if (!checkCore<T, N>(ref)) {
            result = false;
        }
    }

    // Test that raw values are handled properly -- logic is format-specific.
    DataBuffer packed;
    DataBuffer unpacked;
    DataBuffer repacked;
    if (!isSigned) {
        packed.ui[0] = (0x000 << 0) | (0x150 << 10) | (0x2A0 << 20) | (0x0 << 30);
        packed.ui[1] = (0x150 << 0) | (0x2A0 << 10) | (0x3FF << 20) | (0x1 << 30);
        packed.ui[2] = (0x2A0 << 0) | (0x3FF << 10) | (0x000 << 20) | (0x2 << 30);
        packed.ui[3] = (0x3FF << 0) | (0x000 << 10) | (0x150 << 20) | (0x3 << 30);
        for (int i = 0; i < 4; i++) {
            repacked.ui[i] = packed.ui[i];
        }
        for (int i = 0; i < 16; i++) {
            if ((i % 4) == 3) {
                unpacked.f[i] = i / 4;
                if (isNormalized) {
                    unpacked.f[i] /= 3.0;
                }
            } else {
                int value = ((i % 4) + (i / 4)) % 4;
                value = 0x150 * value + ((value == 3) ? 0xF : 0x0);
                unpacked.f[i] = value;
                if (isNormalized) {
                    unpacked.f[i] /= 1023.0;
                }
            }
        }
    } else {
        packed.ui[0] = (0x000 << 0) | (0x100 << 10) | (0x1FF << 20) | (0x0 << 30);
        packed.ui[1] = (0x100 << 0) | (0x1FF << 10) | (0x201 << 20) | (0x1 << 30);
        packed.ui[2] = (0x1FF << 0) | (0x201 << 10) | (0x000 << 20) | (0x2 << 30);
        packed.ui[3] = (0x201 << 0) | (0x000 << 10) | (0x100 << 20) | (0x3 << 30);
        for (int i = 0; i < 4; i++) {
            repacked.ui[i] = packed.ui[i];
        }
        for (int i = 0; i < 16; i++) {
            if ((i % 4) == 3) {
                unpacked.f[i] = ((i / 4) ^ 2) - 2;
                if (isNormalized) {
                    if (unpacked.f[i] < -1.0) {
                        unpacked.f[i] = -1.0;
                        repacked.ui[i/4] |= 0x40000000;
                    }
                }
            } else {
                int value = packed.ui[i / 4] >> (10 * (i % 4));
                value = (value << 22) >> 22;
                unpacked.f[i] = value;
                if (isNormalized) {
                    unpacked.f[i] /= 511.0;
                }
            }
        }
    }
    if (!checkRawMemory<T, N, true>(4, packed.ui, unpacked.f, repacked.ui)) {
        result = false;
    }

    return result;
}

template <typename T, int N, LWNformat F>
bool LWNDataTypeTest::checkPacked32UI(T ref)
{
    bool result = true;
    if (!checkFormatEnum<T, F>()) {
        result = false;
    }
    if (!checkCore<T, N>(ref)) {
        result = false;
    }

    // Test that component selection operators work properly.  For packed
    // types, component selectors return an external value that can only be
    // used as an r-value.
    for (int i = 0; i < N; i++) {
        CHECK(ref[i] == (typename T::ExternalScalarType)(4 - i));
    }

    // Test that raw values are handled properly.
    DataBuffer packed;
    DataBuffer unpacked;
    assert(F == LWN_FORMAT_RGB10A2UI);
    packed.ui[0] = (0x000 << 0) | (0x150 << 10) | (0x2A0 << 20) | (0x0 << 30);
    packed.ui[1] = (0x150 << 0) | (0x2A0 << 10) | (0x3FF << 20) | (0x1 << 30);
    packed.ui[2] = (0x2A0 << 0) | (0x3FF << 10) | (0x000 << 20) | (0x2 << 30);
    packed.ui[3] = (0x3FF << 0) | (0x000 << 10) | (0x150 << 20) | (0x3 << 30);
    for (int i = 0; i < 16; i++) {
        unpacked.ui[i] = ((packed.ui[i / 4]) >> (10 * (i % 4))) & 0x3FF;
    }
    if (!checkRawMemory<T, N, true>(4, packed.ui, unpacked.ui, packed.ui)) {
        result = false;
    }

    return result;

}

template <typename T, int N, LWNformat F>
bool LWNDataTypeTest::checkPacked32I(T ref)
{
    bool result = true;
    if (!checkFormatEnum<T, F>()) {
        result = false;
    }
    if (!checkCore<T, N>(ref)) {
        result = false;
    }

    // Test that component selection operators work properly.  For packed
    // types, component selectors return an external value that can only be
    // used as an r-value.
    for (int i = 0; i < N; i++) {
        CHECK(ref[i] == 4 - i);
    }

    // Test that raw values are handled properly.
    DataBuffer packed;
    DataBuffer unpacked;
    assert(F == LWN_FORMAT_RGB10A2I);
    packed.ui[0] = (0x000 << 0) | (0x100 << 10) | (0x1FF << 20) | (0x0 << 30);
    packed.ui[1] = (0x100 << 0) | (0x1FF << 10) | (0x200 << 20) | (0x1 << 30);
    packed.ui[2] = (0x1FF << 0) | (0x200 << 10) | (0x000 << 20) | (0x2 << 30);
    packed.ui[3] = (0x200 << 0) | (0x000 << 10) | (0x100 << 20) | (0x3 << 30);
    for (int i = 0; i < 16; i++) {
        if ((i % 4) == 3) {
            unpacked.i[i] = ((i / 4) ^ 2) - 2;
        } else {
            int value = ((packed.ui[i / 4]) >> (10 * (i % 4))) & 0x3FF;
            value = (value << 22) >> 22;
            unpacked.i[i] = value;
        }
    }
    if (!checkRawMemory<T, N, true>(4, packed.ui, unpacked.i, packed.ui)) {
        result = false;
    }

    return result;
}


// CHECK_TYPE:  Use one of the "check" template functions (with suffix
// <_func>) to check for correct handling of type <_type> (with <_n>
// components).  The type should have a format enum of <_fmt>, and the
// (4,3,2,1) reference vector is given by <_ref>.
#define CHECK_TYPE(_func, _type, _n, _fmt, _ref)        \
    do {                                                \
        subtest = #_type;                               \
        if (!check##_func<dt::_type, _n, _fmt>(_ref)) { \
            result = false;                             \
                }                                       \
        } while (0)


void LWNDataTypeTest::doGraphics() const
{
    DeviceState *deviceState = DeviceState::GetActive();
    QueueCommandBuffer &queueCB = deviceState->getQueueCB();

    bool result = true;
    float s;

    CHECK_TYPE(Bool, bvec1, 1, LWN_FORMAT_NONE, dt::bvec1(true));
    CHECK_TYPE(Bool, bvec2, 2, LWN_FORMAT_NONE, dt::bvec2(true, false));
    CHECK_TYPE(Bool, bvec3, 3, LWN_FORMAT_NONE, dt::bvec3(true, false, true));
    CHECK_TYPE(Bool, bvec4, 4, LWN_FORMAT_NONE, dt::bvec4(true, false, true, false));

    CHECK_TYPE(F32, vec1, 1, LWN_FORMAT_R32F,    dt::vec1(4));
    CHECK_TYPE(F32, vec2, 2, LWN_FORMAT_RG32F,   dt::vec2(4, 3));
    CHECK_TYPE(F32, vec3, 3, LWN_FORMAT_RGB32F,  dt::vec3(4, 3, 2));
    CHECK_TYPE(F32, vec4, 4, LWN_FORMAT_RGBA32F, dt::vec4(4, 3, 2, 1));

    CHECK_TYPE(U32, uvec1, 1, LWN_FORMAT_R32UI,    dt::uvec1(4));
    CHECK_TYPE(U32, uvec2, 2, LWN_FORMAT_RG32UI,   dt::uvec2(4, 3));
    CHECK_TYPE(U32, uvec3, 3, LWN_FORMAT_RGB32UI,  dt::uvec3(4, 3, 2));
    CHECK_TYPE(U32, uvec4, 4, LWN_FORMAT_RGBA32UI, dt::uvec4(4, 3, 2, 1));

    CHECK_TYPE(S32, ivec1, 1, LWN_FORMAT_R32I,    dt::ivec1(4));
    CHECK_TYPE(S32, ivec2, 2, LWN_FORMAT_RG32I,   dt::ivec2(4, 3));
    CHECK_TYPE(S32, ivec3, 3, LWN_FORMAT_RGB32I,  dt::ivec3(4, 3, 2));
    CHECK_TYPE(S32, ivec4, 4, LWN_FORMAT_RGBA32I, dt::ivec4(4, 3, 2, 1));

    CHECK_TYPE(F16, f16vec1, 1, LWN_FORMAT_R16F, dt::f16vec1(4));
    CHECK_TYPE(F16, f16vec2, 2, LWN_FORMAT_RG16F, dt::f16vec2(4, 3));
    CHECK_TYPE(F16, f16vec3, 3, LWN_FORMAT_RGB16F, dt::f16vec3(4, 3, 2));
    CHECK_TYPE(F16, f16vec4, 4, LWN_FORMAT_RGBA16F, dt::f16vec4(4, 3, 2, 1));

    s = 1.0 / 0xFF;
    CHECK_TYPE(Unorm, u8lwec1, 1, LWN_FORMAT_R8,    dt::u8lwec1(s*4));
    CHECK_TYPE(Unorm, u8lwec2, 2, LWN_FORMAT_RG8,   dt::u8lwec2(s*4, s*3));
    CHECK_TYPE(Unorm, u8lwec3, 3, LWN_FORMAT_RGB8,  dt::u8lwec3(s*4, s*3, s*2));
    CHECK_TYPE(Unorm, u8lwec4, 4, LWN_FORMAT_RGBA8, dt::u8lwec4(s*4, s*3, s*2, s*1));

    s = 1.0 / 0xFFFF;
    CHECK_TYPE(Unorm, u16lwec1, 1, LWN_FORMAT_R16,    dt::u16lwec1(s*4));
    CHECK_TYPE(Unorm, u16lwec2, 2, LWN_FORMAT_RG16,   dt::u16lwec2(s*4, s*3));
    CHECK_TYPE(Unorm, u16lwec3, 3, LWN_FORMAT_RGB16,  dt::u16lwec3(s*4, s*3, s*2));
    CHECK_TYPE(Unorm, u16lwec4, 4, LWN_FORMAT_RGBA16, dt::u16lwec4(s*4, s*3, s*2, s*1));

    s = 1.0 / 0x7F;
    CHECK_TYPE(Snorm, i8lwec1, 1, LWN_FORMAT_R8SN,    dt::i8lwec1(s*4));
    CHECK_TYPE(Snorm, i8lwec2, 2, LWN_FORMAT_RG8SN,   dt::i8lwec2(s*4, s*3));
    CHECK_TYPE(Snorm, i8lwec3, 3, LWN_FORMAT_RGB8SN,  dt::i8lwec3(s*4, s*3, s*2));
    CHECK_TYPE(Snorm, i8lwec4, 4, LWN_FORMAT_RGBA8SN, dt::i8lwec4(s*4, s*3, s*2, s*1));

    s = 1.0 / 0x7FFF;
    CHECK_TYPE(Snorm, i16lwec1, 1, LWN_FORMAT_R16SN,    dt::i16lwec1(s*4));
    CHECK_TYPE(Snorm, i16lwec2, 2, LWN_FORMAT_RG16SN,   dt::i16lwec2(s*4, s*3));
    CHECK_TYPE(Snorm, i16lwec3, 3, LWN_FORMAT_RGB16SN,  dt::i16lwec3(s*4, s*3, s*2));
    CHECK_TYPE(Snorm, i16lwec4, 4, LWN_FORMAT_RGBA16SN, dt::i16lwec4(s*4, s*3, s*2, s*1));

    CHECK_TYPE(Uint, u8vec1, 1, LWN_FORMAT_R8UI,    dt::u8vec1(4));
    CHECK_TYPE(Uint, u8vec2, 2, LWN_FORMAT_RG8UI,   dt::u8vec2(4, 3));
    CHECK_TYPE(Uint, u8vec3, 3, LWN_FORMAT_RGB8UI,  dt::u8vec3(4, 3, 2));
    CHECK_TYPE(Uint, u8vec4, 4, LWN_FORMAT_RGBA8UI, dt::u8vec4(4, 3, 2, 1));

    CHECK_TYPE(Uint, u16vec1, 1, LWN_FORMAT_R16UI,    dt::u16vec1(4));
    CHECK_TYPE(Uint, u16vec2, 2, LWN_FORMAT_RG16UI,   dt::u16vec2(4, 3));
    CHECK_TYPE(Uint, u16vec3, 3, LWN_FORMAT_RGB16UI,  dt::u16vec3(4, 3, 2));
    CHECK_TYPE(Uint, u16vec4, 4, LWN_FORMAT_RGBA16UI, dt::u16vec4(4, 3, 2, 1));

    CHECK_TYPE(Sint, i8vec1, 1, LWN_FORMAT_R8I,    dt::i8vec1(4));
    CHECK_TYPE(Sint, i8vec2, 2, LWN_FORMAT_RG8I,   dt::i8vec2(4, 3));
    CHECK_TYPE(Sint, i8vec3, 3, LWN_FORMAT_RGB8I,  dt::i8vec3(4, 3, 2));
    CHECK_TYPE(Sint, i8vec4, 4, LWN_FORMAT_RGBA8I, dt::i8vec4(4, 3, 2, 1));

    CHECK_TYPE(Sint, i16vec1, 1, LWN_FORMAT_R16I,    dt::i16vec1(4));
    CHECK_TYPE(Sint, i16vec2, 2, LWN_FORMAT_RG16I,   dt::i16vec2(4, 3));
    CHECK_TYPE(Sint, i16vec3, 3, LWN_FORMAT_RGB16I,  dt::i16vec3(4, 3, 2));
    CHECK_TYPE(Sint, i16vec4, 4, LWN_FORMAT_RGBA16I, dt::i16vec4(4, 3, 2, 1));

    CHECK_TYPE(U2F, u2f8vec1, 1, LWN_FORMAT_R8_UI2F,    dt::u2f8vec1(4));
    CHECK_TYPE(U2F, u2f8vec2, 2, LWN_FORMAT_RG8_UI2F,   dt::u2f8vec2(4, 3));
    CHECK_TYPE(U2F, u2f8vec3, 3, LWN_FORMAT_RGB8_UI2F,  dt::u2f8vec3(4, 3, 2));
    CHECK_TYPE(U2F, u2f8vec4, 4, LWN_FORMAT_RGBA8_UI2F, dt::u2f8vec4(4, 3, 2, 1));

    CHECK_TYPE(U2F, u2f16vec1, 1, LWN_FORMAT_R16_UI2F,    dt::u2f16vec1(4));
    CHECK_TYPE(U2F, u2f16vec2, 2, LWN_FORMAT_RG16_UI2F,   dt::u2f16vec2(4, 3));
    CHECK_TYPE(U2F, u2f16vec3, 3, LWN_FORMAT_RGB16_UI2F,  dt::u2f16vec3(4, 3, 2));
    CHECK_TYPE(U2F, u2f16vec4, 4, LWN_FORMAT_RGBA16_UI2F, dt::u2f16vec4(4, 3, 2, 1));

    CHECK_TYPE(U2F, u2f32vec1, 1, LWN_FORMAT_R32_UI2F,    dt::u2f32vec1(4));
    CHECK_TYPE(U2F, u2f32vec2, 2, LWN_FORMAT_RG32_UI2F,   dt::u2f32vec2(4, 3));
    CHECK_TYPE(U2F, u2f32vec3, 3, LWN_FORMAT_RGB32_UI2F,  dt::u2f32vec3(4, 3, 2));
    CHECK_TYPE(U2F, u2f32vec4, 4, LWN_FORMAT_RGBA32_UI2F, dt::u2f32vec4(4, 3, 2, 1));

    CHECK_TYPE(S2F, i2f8vec1, 1, LWN_FORMAT_R8_I2F,    dt::i2f8vec1(4));
    CHECK_TYPE(S2F, i2f8vec2, 2, LWN_FORMAT_RG8_I2F,   dt::i2f8vec2(4, 3));
    CHECK_TYPE(S2F, i2f8vec3, 3, LWN_FORMAT_RGB8_I2F,  dt::i2f8vec3(4, 3, 2));
    CHECK_TYPE(S2F, i2f8vec4, 4, LWN_FORMAT_RGBA8_I2F, dt::i2f8vec4(4, 3, 2, 1));

    CHECK_TYPE(S2F, i2f16vec1, 1, LWN_FORMAT_R16_I2F,    dt::i2f16vec1(4));
    CHECK_TYPE(S2F, i2f16vec2, 2, LWN_FORMAT_RG16_I2F,   dt::i2f16vec2(4, 3));
    CHECK_TYPE(S2F, i2f16vec3, 3, LWN_FORMAT_RGB16_I2F,  dt::i2f16vec3(4, 3, 2));
    CHECK_TYPE(S2F, i2f16vec4, 4, LWN_FORMAT_RGBA16_I2F, dt::i2f16vec4(4, 3, 2, 1));

    CHECK_TYPE(S2F, i2f32vec1, 1, LWN_FORMAT_R32_I2F,    dt::i2f32vec1(4));
    CHECK_TYPE(S2F, i2f32vec2, 2, LWN_FORMAT_RG32_I2F,   dt::i2f32vec2(4, 3));
    CHECK_TYPE(S2F, i2f32vec3, 3, LWN_FORMAT_RGB32_I2F,  dt::i2f32vec3(4, 3, 2));
    CHECK_TYPE(S2F, i2f32vec4, 4, LWN_FORMAT_RGBA32_I2F, dt::i2f32vec4(4, 3, 2, 1));

    CHECK_TYPE(Packed16, vec4_rgba4,   4, LWN_FORMAT_RGBA4,   dt::vec4_rgba4(4.0/15, 3.0/15, 2.0/15, 1.0/15));
    CHECK_TYPE(Packed16, vec3_rgb5,    3, LWN_FORMAT_RGB5,    dt::vec3_rgb5(4.0/31, 3.0/31, 2.0/31));
    CHECK_TYPE(Packed16, vec4_rgb5a1,  4, LWN_FORMAT_RGB5A1,  dt::vec4_rgb5a1(4.0/31, 3.0/31, 2.0/31, 1.0));
    CHECK_TYPE(Packed16, vec3_rgb565,  3, LWN_FORMAT_RGB565,  dt::vec3_rgb565(4.0/31, 3.0/63, 2.0/31));
    CHECK_TYPE(Packed16, vec3_bgr5,    3, LWN_FORMAT_BGR5,    dt::vec3_bgr5(4.0/31, 3.0/31, 2.0/31));
    CHECK_TYPE(Packed16, vec4_bgr5a1,  4, LWN_FORMAT_BGR5A1,  dt::vec4_bgr5a1(4.0/31, 3.0/31, 2.0/31, 1.0));
    CHECK_TYPE(Packed16, vec3_bgr565,  3, LWN_FORMAT_BGR565,  dt::vec3_bgr565(4.0/31, 3.0/63, 2.0/31));

    CHECK_TYPE(Packed32F,  vec4_rgb10a2,        4, LWN_FORMAT_RGB10A2,      dt::vec4_rgb10a2(4.0/1023, 3.0/1023, 2.0/1023, 1.0/3));
    CHECK_TYPE(Packed32F,  vec4_rgb10a2sn,      4, LWN_FORMAT_RGB10A2SN,    dt::vec4_rgb10a2sn(4.0/511, 3.0/511, 2.0/511, 1.0/1));
    CHECK_TYPE(Packed32UI, vec4_rgb10a2ui,      4, LWN_FORMAT_RGB10A2UI,    dt::vec4_rgb10a2ui(4, 3, 2, 1));
    CHECK_TYPE(Packed32I,  vec4_rgb10a2i,       4, LWN_FORMAT_RGB10A2I,     dt::vec4_rgb10a2i(4, 3, 2, 1));
    CHECK_TYPE(Packed32F,  vec4_rgb10a2ui_to_f, 4, LWN_FORMAT_RGB10A2_UI2F, dt::vec4_rgb10a2ui_to_f(4, 3, 2, 1));
    CHECK_TYPE(Packed32F,  vec4_rgb10a2i_to_f,  4, LWN_FORMAT_RGB10A2_I2F,  dt::vec4_rgb10a2i_to_f(4, 3, 2, 1));

    if (result) {
        queueCB.ClearColor(0, 0.0, 1.0, 0.0, 1.0);
    } else {
        queueCB.ClearColor(0, 1.0, 0.0, 0.0, 1.0);
    }
    queueCB.submit();
}

OGTEST_CppTest(LWNDataTypeTest, lwn_util_datatypes, );
