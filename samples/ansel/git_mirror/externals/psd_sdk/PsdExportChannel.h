// Copyright 2011-2020, Molelwlar Matters GmbH <office@molelwlar-matters.com>
// See LICENSE.txt for licensing details (2-clause BSD License: https://opensource.org/licenses/BSD-2-Clause)

#pragma once


PSD_NAMESPACE_BEGIN

/// \ingroup Types
/// \namespace exportChannel
/// \brief A namespace denoting a channel that is exported to the Layer Mask section.
namespace exportChannel
{
	enum Enum
	{
		// supported in Grayscale dolwments
		GRAY,

		// supported in RGB dolwments
		RED,
		GREEN,
		BLUE,

		// supported in all dolwments
		ALPHA
	};
}

PSD_NAMESPACE_END
