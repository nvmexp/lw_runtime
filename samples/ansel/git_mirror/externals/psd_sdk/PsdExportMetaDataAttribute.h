// Copyright 2011-2020, Molelwlar Matters GmbH <office@molelwlar-matters.com>
// See LICENSE.txt for licensing details (2-clause BSD License: https://opensource.org/licenses/BSD-2-Clause)

#pragma once


PSD_NAMESPACE_BEGIN

/// \ingroup Types
/// \class ExportMetaDataAttribute
/// \brief A struct representing a meta data attribute as exported to the image resources section.
struct ExportMetaDataAttribute
{
	char* name;
	char* value;
};

PSD_NAMESPACE_END
