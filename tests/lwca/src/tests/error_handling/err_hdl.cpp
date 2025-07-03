#include "gtest/gtest.h"
#include "lwda_utils/lwda_utils.h"
namespace
{

TEST(ErrorHandling, GetErrorName)
{
    const char *name = nullptr;

    // Negative tests
    EXPECT_EQ(lwGetErrorName(static_cast<LWresult>(-1), &name), LWDA_ERROR_ILWALID_VALUE);
    EXPECT_EQ(name, nullptr);

    for (size_t i = static_cast<size_t>(LWDA_SUCCESS); i < static_cast<size_t>(LWDA_ERROR_UNKNOWN); i++) {
        SCOPED_TRACE_STREAM("Error Code: " << i);

        LWresult status = lwGetErrorName(static_cast<LWresult>(i), &name);

        if (status == LWDA_SUCCESS) {
            EXPECT_NE(name, nullptr);
            size_t sz = strlen(name);
            EXPECT_NE(sz, 0);
            for (size_t j = 0; j < sz; j++) {
                EXPECT_PRED1(isascii, name[j]);
            }
        }
        else {
            EXPECT_EQ(status, LWDA_ERROR_ILWALID_VALUE);
            EXPECT_EQ(name, nullptr);
        }
    }
}

TEST(ErrorHandling, GetErrorString)
{
    const char *name = nullptr;

    // Negative tests
    EXPECT_EQ(lwGetErrorString(static_cast<LWresult>(-1), &name), LWDA_ERROR_ILWALID_VALUE);
    EXPECT_EQ(name, nullptr);

    for (size_t i = static_cast<size_t>(LWDA_SUCCESS); i < static_cast<size_t>(LWDA_ERROR_UNKNOWN); i++) {
        SCOPED_TRACE_STREAM("Error Code: " << i);

        LWresult status = lwGetErrorString(static_cast<LWresult>(i), &name);

        if (status == LWDA_SUCCESS) {
            EXPECT_NE(name, nullptr);
            size_t sz = strlen(name);
            EXPECT_NE(sz, 0);
            for (size_t j = 0; j < sz; j++) {
                EXPECT_TRUE(isascii(name[j]));
            }
        }
        else {
            EXPECT_EQ(status, LWDA_ERROR_ILWALID_VALUE);
            EXPECT_EQ(name, nullptr);
        }
    }
}

}
