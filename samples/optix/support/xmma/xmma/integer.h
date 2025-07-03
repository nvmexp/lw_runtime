/***************************************************************************************************
 * Copyright (c) 2019, LWPU CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, are permitted
 * provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright notice, this list of
 *       conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright notice, this list of
 *       conditions and the following disclaimer in the documentation and/or other materials
 *       provided with the distribution.
 *     * Neither the name of the LWPU CORPORATION nor the names of its contributors may be used
 *       to endorse or promote products derived from this software without specific prior written
 *       permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL LWPU CORPORATION BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROLWREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TOR (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

#pragma once

#include <xmma/xmma.h>

namespace xmma {

///////////////////////////////////////////////////////////////////////////////////////////////////

// Compile time constants
template<int a, int b>
struct Div_round_up {
    enum { VALUE = (a + b - 1) / b };
};

template<int a, int alignment>
struct Align_up {
    enum { VALUE = Div_round_up<a, alignment>::value * alignment };
};

#if !defined(__LWDA_ARCH__)

// From signed to signed
template<
    typename T, typename S,
    typename std::enable_if<
        std::numeric_limits<T>::is_signed && std::numeric_limits<S>::is_signed,
        int>::type = 0
    >
XMMA_HOST T integer_cast(const S value) {
    if (value > std::numeric_limits<T>::max() || value < std::numeric_limits<T>::min()) {
        throw std::bad_cast();
    }

    return static_cast<T>(value);
}

// From signed to unsigned
template<
    typename T, typename S,
    typename std::enable_if<
        !std::numeric_limits<T>::is_signed && std::numeric_limits<S>::is_signed,
        int>::type = 0
    >
XMMA_HOST T integer_cast(const S value) {
    if ((std::numeric_limits<T>::digits < std::numeric_limits<S>::digits &&
         value > static_cast<S>(std::numeric_limits<T>::max())) ||
        value < 0) {
        throw std::bad_cast();
    }

    return static_cast<T>(value);
}

// From unsigned to unsigned
template<
    typename T, typename S,
    typename std::enable_if<
        !std::numeric_limits<T>::is_signed && !std::numeric_limits<S>::is_signed,
        int>::type = 0
    >
XMMA_HOST T integer_cast(const S value) {
    if (std::numeric_limits<T>::digits < std::numeric_limits<S>::digits &&
        value > std::numeric_limits<T>::max()) {
        throw std::bad_cast();
    }

    return static_cast<T>(value);
}

// From unsigned to signed
template<
    typename T, typename S,
    typename std::enable_if<
        std::numeric_limits<T>::is_signed && !std::numeric_limits<S>::is_signed,
        int>::type = 0
    >
XMMA_HOST T integer_cast(const S value) {
    if (std::numeric_limits<T>::digits <= std::numeric_limits<S>::digits &&
        value > static_cast<S>(std::numeric_limits<T>::max())) {
        throw std::bad_cast();
    }

    return static_cast<T>(value);
}


#else // !defined(__LWDA_ARCH__)

template<typename T, typename S>
struct _is_safe_cast {
#if defined(__LWDACC_RTC__)
    // Ignoring integer cast check for lwrtc
    static constexpr bool value = true;
#else
#if defined(XMMA_INTEGER_CAST_CHECK_ENABLED)
    static constexpr bool value =
        // From unsigned to signed
        (std::numeric_limits<T>::is_signed && !std::numeric_limits<S>::is_signed &&
         std::numeric_limits<T>::digits > std::numeric_limits<S>::digits) ||
        // From unsigned to unsigned
        (!std::numeric_limits<T>::is_signed && !std::numeric_limits<S>::is_signed &&
         std::numeric_limits<T>::digits >= std::numeric_limits<S>::digits) ||
        // From signed to signed
        (std::numeric_limits<T>::is_signed && std::numeric_limits<S>::is_signed &&
         std::numeric_limits<T>::digits >= std::numeric_limits<S>::digits);
#else
    static constexpr bool value = true;
#endif
#endif
};

template<typename T, typename S>
XMMA_DEVICE T integer_cast(const S value) {
    static_assert(_is_safe_cast<T, S>::value, "Cast from wider to narrower type is forbidden in device code");

    return static_cast<T>(value);
}

#endif // __LWDA_ARCH__

///////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T, typename S,
#if !defined(__LWDACC_RTC__)
         typename std::enable_if<std::is_colwertible<T, long long>::value, int>::type = 0,
         typename std::enable_if<std::is_colwertible<S, long long>::value, int>::type = 0,
#endif // __LWDACC_RTC__
         bool = true>
XMMA_HOST_DEVICE T div_round_up(const T m, const S n) {
    long long m_ = integer_cast<long long>(m);
    long long n_ = integer_cast<long long>(n);

#if !defined(__LWDA_ARCH__)
    if (n_ == 0) {
        throw std::logic_error("Divide by zero");
    }
#endif

    return integer_cast<T>((m_ + n_ - 1) / n_);
}

XMMA_HOST_DEVICE size_t align_up(size_t size, size_t alignment) {
    size_t chunks = div_round_up(size, alignment);
    return chunks * alignment;
}

inline int64_t ptr_to_int64(const void* ptr) {
    return static_cast<int64_t>(reinterpret_cast<uintptr_t>(const_cast<void *>(ptr)));
}

///////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace xmma
