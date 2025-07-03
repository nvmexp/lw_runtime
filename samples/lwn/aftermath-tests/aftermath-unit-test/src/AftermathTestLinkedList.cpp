/*
 * Copyright (c) 2020, LWPU CORPORATION.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 */

#include <AftermathTest.h>
#include <AftermathTestLogging.h>
#include <AftermathTestUtils.h>

#include <lwassert.h>
#include <AftermathUtils.h>

#include <type_traits>
#include <vector>

namespace AftermathTest {

using TestList = Aftermath::Utils::SimpleLinkedList<int>;

static bool operator==(const TestList& lhs, const std::vector<int>& rhs)
{
    auto node = lhs.Head();
    size_t n = 0;
    for (; n < rhs.size() && node != nullptr; ++n, node = node->next) {
        if (node->value != rhs[n]) {
            return false;
        }
    }
    return n == rhs.size();
}

static std::ostream& operator<<(std::ostream& os, const TestList& v)
{
    os << "{";
    for (auto node = v.Head(); node != nullptr;) {
        os << node->value;
        node = node->next;
        if (node) {
            os << ", ";
        }
    }
    os << "}";
    return os;
}

static std::ostream& operator<<(std::ostream& os, const std::vector<int>& v)
{
    os << "{";
    for (size_t n = 0; n < v.size(); ++n) {
        os << v[n];
        if (n != v.size() - 1) {
            os << ", ";
        }
    }
    os << "}";
    return os;
}

class SimpleLinkedListValidator {
public:

    bool Test();
};

bool SimpleLinkedListValidator::Test()
{
    TestList l;

#define EXPECT(...) TEST_EQ(l, std::vector<int>({__VA_ARGS__}))

    l.AddFront(1);
    EXPECT(1);

    l.AddFront(2);
    EXPECT(2, 1);

    l.AddFront(3);
    EXPECT(3, 2, 1);

    l.Clear();
    EXPECT();

    l.AddFront(1);
    EXPECT(1);

    l.AddFront(2);
    EXPECT(2, 1);

    l.AddFront(3);
    EXPECT(3, 2, 1);

    l.AddFront(2);
    EXPECT(2, 3, 2, 1);

    l.AddBack(3);
    EXPECT(2, 3, 2, 1, 3);

    l.AddFront(2);
    EXPECT(2, 2, 3, 2, 1, 3);

    l.AddFront(1);
    EXPECT(1, 2, 2, 3, 2, 1, 3);

    l.RemoveFirst(3);
    EXPECT(1, 2, 2, 2, 1, 3);

    l.Remove(l.FindFirst(1));
    EXPECT(2, 2, 2, 1, 3);

    l.Remove(l.FindFirst([](int v){return v == 1;}));
    EXPECT(2, 2, 2, 3);

    l.RemoveFirst(3);
    EXPECT(2, 2, 2);

    l.AddBack(1);
    EXPECT(2, 2, 2, 1);

    l.AddBack(2);
    EXPECT(2, 2, 2, 1, 2);

    l.AddBack(3);
    EXPECT(2, 2, 2, 1, 2, 3);

    l.AddBack(2);
    EXPECT(2, 2, 2, 1, 2, 3, 2);

    l.RemoveAll(2);
    EXPECT(1, 3);

    l.RemoveFirst(2);
    EXPECT(1, 3);

    l.RemoveAll(3);
    EXPECT(1);

    l.RemoveAll(1);
    EXPECT();

    l.RemoveAll(2);
    EXPECT();

    return true;

#undef EXPECT
}

AFTERMATH_DEFINE_TEST(SimpleLinkedList, UNIT,
    LwError Execute(const Options& options)
    {
        (void)options;
        SimpleLinkedListValidator validator;
        if (!validator.Test()) {
            return LwError_IlwalidState;
        } else {
            return LwSuccess;
        }
    }
);

} // namespace AftermathTest
