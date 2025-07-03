// Copyright 2011-2020, Molelwlar Matters GmbH <office@molelwlar-matters.com>
// See LICENSE.txt for licensing details (2-clause BSD License: https://opensource.org/licenses/BSD-2-Clause)

#pragma once


PSD_NAMESPACE_BEGIN

struct Document;
class File;
class Allocator;


/// \ingroup Parser
/// Parses only the header and section offsets, and returns a newly created document that needs to be freed
/// by a call to \ref DestroyDolwment.
Document* CreateDolwment(File* file, Allocator* allocator);

/// \ingroup Parser
/// Destroys and nullifies the given \a document previously created by a call to \ref CreateDolwment.
void DestroyDolwment(Document*& document, Allocator* allocator);

PSD_NAMESPACE_END
