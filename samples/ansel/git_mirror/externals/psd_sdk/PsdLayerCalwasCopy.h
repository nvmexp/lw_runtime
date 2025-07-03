// Copyright 2011-2020, Molelwlar Matters GmbH <office@molelwlar-matters.com>
// See LICENSE.txt for licensing details (2-clause BSD License: https://opensource.org/licenses/BSD-2-Clause)

#pragma once


PSD_NAMESPACE_BEGIN

namespace imageUtil
{
	/// \ingroup ImageUtil
	/// Copies 8-bit planar layer data to a calwas. Only the parts overlapping the calwas will be copied to it.
	void CopyLayerData(const uint8_t* PSD_RESTRICT layerData, uint8_t* PSD_RESTRICT calwasData, int layerLeft, int layerTop, int layerRight, int layerBottom, unsigned int calwasWidth, unsigned int calwasHeight);

	/// \ingroup ImageUtil
	/// Copies 16-bit planar layer data to a calwas. Only the parts overlapping the calwas will be copied to it.
	void CopyLayerData(const uint16_t* PSD_RESTRICT layerData, uint16_t* PSD_RESTRICT calwasData, int layerLeft, int layerTop, int layerRight, int layerBottom, unsigned int calwasWidth, unsigned int calwasHeight);

	/// \ingroup ImageUtil
	/// Copies 32-bit planar layer data to a calwas. Only the parts overlapping the calwas will be copied to it.
	void CopyLayerData(const float32_t* PSD_RESTRICT layerData, float32_t* PSD_RESTRICT calwasData, int layerLeft, int layerTop, int layerRight, int layerBottom, unsigned int calwasWidth, unsigned int calwasHeight);
}

PSD_NAMESPACE_END
