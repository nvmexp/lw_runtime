// Copyright 2011-2020, Molelwlar Matters GmbH <office@molelwlar-matters.com>
// See LICENSE.txt for licensing details (2-clause BSD License: https://opensource.org/licenses/BSD-2-Clause)

#pragma once

#include "PsdExportColorMode.h"
#include "PsdExportChannel.h"
#include "PsdCompressionType.h"
#include "PsdAlphaChannel.h"


PSD_NAMESPACE_BEGIN

struct ExportDolwment;
class File;
class Allocator;


/// \ingroup Exporter
/// Creates a new document suited for exporting a PSD file. The returned document needs to be freed
/// by a call to \ref DestroyExportDolwment.
ExportDolwment* CreateExportDolwment(Allocator* allocator, unsigned int calwasWidth, unsigned int calwasHeight, unsigned int bitsPerChannel, exportColorMode::Enum colorMode);

/// \ingroup Exporter
/// Destroys and nullifies the given \a document previously created by a call to \ref CreateExportDolwment.
void DestroyExportDolwment(ExportDolwment*& document, Allocator* allocator);


/// \ingroup Exporter
/// Adds meta data to a document. The contents of \a name and \a value are copied. The returned index can be used to update existing meta data
/// by a call to \ref UpdateMetaData.
unsigned int AddMetaData(ExportDolwment* document, Allocator* allocator, const char* name, const char* value);

/// \ingroup Exporter
/// Updates existing meta data at the given \a index.
void UpdateMetaData(ExportDolwment* document, Allocator* allocator, unsigned int index, const char* name, const char* value);


/// \ingroup Exporter
/// Sets the ICC profile of a document. The contents of \a rawProfileData are copied.
void SetICCProfile(ExportDolwment* document, Allocator* allocator, void* rawProfileData, uint32_t size);

/// \ingroup Exporter
/// Sets the EXIF data of a document. The contents of \a rawExitData are copied.
void SetEXIFData(ExportDolwment* document, Allocator* allocator, void* rawExifData, uint32_t size);

/// \ingroup Exporter
/// Sets the JPEG thumbnail of a document. The contents of \a rawJpegData are copied.
void SetJpegThumbnail(ExportDolwment* document, Allocator* allocator, uint32_t width, uint32_t height, void* rawJpegData, uint32_t size);


/// \ingroup Exporter
/// Adds a layer to a document. The returned index can be used to update layer data by a call to \ref UpdateLayer.
unsigned int AddLayer(ExportDolwment* document, Allocator* allocator, const char* name);

/// \ingroup Exporter
/// Updates a layer with planar 8-bit data. The function internally takes ownership over all data, so planar image data passed to this function can be freed afterwards.
/// Planar data must hold "width*height" bytes, where width = \a right - \a left and height = \a botttom - \a top.
/// Note that individual layers can be smaller and/or larger than the calwas in PSD dolwments.
void UpdateLayer(ExportDolwment* document, Allocator* allocator, unsigned int layerIndex, exportChannel::Enum channel, int left, int top, int right, int bottom, const uint8_t* planarData, compressionType::Enum compression);

/// \ingroup Exporter
/// Updates a layer with planar 16-bit data. The function internally takes ownership over all data, so planar image data passed to this function can be freed afterwards.
/// Planar data must hold "width*height*2" bytes, where width = \a right - \a left and height = \a botttom - \a top.
/// Note that individual layers can be smaller and/or larger than the calwas in PSD dolwments.
void UpdateLayer(ExportDolwment* document, Allocator* allocator, unsigned int layerIndex, exportChannel::Enum channel, int left, int top, int right, int bottom, const uint16_t* planarData, compressionType::Enum compression);

/// \ingroup Exporter
/// Updates a layer with planar 32-bit data. The function internally takes ownership over all data, so planar image data passed to this function can be freed afterwards.
/// Planar data must hold "width*height*4" bytes, where width = \a right - \a left and height = \a botttom - \a top.
/// Note that individual layers can be smaller and/or larger than the calwas in PSD dolwments.
void UpdateLayer(ExportDolwment* document, Allocator* allocator, unsigned int layerIndex, exportChannel::Enum channel, int left, int top, int right, int bottom, const float32_t* planarData, compressionType::Enum compression);


/// \ingroup Exporter
/// Adds an alpha channel to a document. The returned index can be used to update channel data by a call to \ref UpdateChannel.
unsigned int AddAlphaChannel(ExportDolwment* document, Allocator* allocator, const char* name, uint16_t r, uint16_t g, uint16_t b, uint16_t a, uint16_t opacity, AlphaChannel::Mode::Enum mode);

/// \ingroup Exporter
/// Updates a channel with 8-bit data. The function internally takes ownership over all data, so image data passed to this function can be freed afterwards.
/// Data must hold width*height bytes.
void UpdateChannel(ExportDolwment* document, Allocator* allocator, unsigned int channelIndex, const uint8_t* data);

/// \ingroup Exporter
/// Updates a channel with 16-bit data. The function internally takes ownership over all data, so image data passed to this function can be freed afterwards.
/// Data must hold width*height*2 bytes.
void UpdateChannel(ExportDolwment* document, Allocator* allocator, unsigned int channelIndex, const uint16_t* data);

/// \ingroup Exporter
/// Updates a channel with 32-bit data. The function internally takes ownership over all data, so image data passed to this function can be freed afterwards.
/// Data must hold width*height*4 bytes.
void UpdateChannel(ExportDolwment* document, Allocator* allocator, unsigned int channelIndex, const float32_t* data);


/// \ingroup Exporter
/// Updates the merged image data. The function internally takes ownership over all data, so planar image data passed to this function can be freed afterwards.
/// Planar data must hold width*height bytes.
void UpdateMergedImage(ExportDolwment* document, Allocator* allocator, const uint8_t* planarDataR, const uint8_t* planarDataG, const uint8_t* planarDataB);

/// \ingroup Exporter
/// Updates the merged image data. The function internally takes ownership over all data, so planar image data passed to this function can be freed afterwards.
/// Planar data must hold width*height*2 bytes.
void UpdateMergedImage(ExportDolwment* document, Allocator* allocator, const uint16_t* planarDataR, const uint16_t* planarDataG, const uint16_t* planarDataB);

/// \ingroup Exporter
/// Updates the merged image data. The function internally takes ownership over all data, so planar image data passed to this function can be freed afterwards.
/// Planar data must hold width*height*4 bytes.
void UpdateMergedImage(ExportDolwment* document, Allocator* allocator, const float32_t* planarDataR, const float32_t* planarDataG, const float32_t* planarDataB);


/// \ingroup Exporter
/// Exports a document to the given file.
void WriteDolwment(ExportDolwment* document, Allocator* allocator, File* file);

PSD_NAMESPACE_END
