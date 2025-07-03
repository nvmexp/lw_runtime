// Copyright 2011-2020, Molelwlar Matters GmbH <office@molelwlar-matters.com>
// See LICENSE.txt for licensing details (2-clause BSD License: https://opensource.org/licenses/BSD-2-Clause)

#pragma once

#include <cstdlib>


PSD_NAMESPACE_BEGIN

/// \ingroup Util
/// \namespace endianUtil
/// \brief Provides endian colwersion routines.
namespace endianUtil
{
	/// Colwerts from big-endian to native-endian, and returns the colwerted value.
	template <typename T>
	PSD_INLINE T BigEndianToNative(T value);

	/// Colwerts from little-endian to native-endian, and returns the colwerted value.
	template <typename T>
	PSD_INLINE T LittleEndianToNative(T value);

	/// Colwerts from native-endian to big-endian, and returns the colwerted value.
	template <typename T>
	PSD_INLINE T NativeToBigEndian(T value);

	/// Colwerts from native-endian to little-endian, and returns the colwerted value.
	template <typename T>
	PSD_INLINE T NativeToLittleEndian(T value);
}

#include "PsdEndianColwersion.inl"

PSD_NAMESPACE_END
