// Copyright LWPU Corporation 2021
// TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED
// *AS IS* AND LWPU AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS
// OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY
// AND FITNESS FOR A PARTICULAR PURPOSE.  IN NO EVENT SHALL LWPU OR ITS SUPPLIERS
// BE LIABLE FOR ANY SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES
// WHATSOEVER (INCLUDING, WITHOUT LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS,
// BUSINESS INTERRUPTION, LOSS OF BUSINESS INFORMATION, OR ANY OTHER PELWNIARY LOSS)
// ARISING OUT OF THE USE OF OR INABILITY TO USE THIS SOFTWARE, EVEN IF LWPU HAS
// BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES

// Phantom intersector, CatmullRom
#include <exp/builtinIS/CatmullRomPhantomIntersector_ptx_bin.h>
#include <exp/builtinIS/CatmullRomPhantomLowMemIntersector_ptx_bin.h>

// Phantom intersector, cubic Bspline
#include <exp/builtinIS/LwbicLwrvePhantomIntersector_ptx_bin.h>
#include <exp/builtinIS/LwbicLwrvePhantomLowMemIntersector_ptx_bin.h>

// Linear intersector
#include <exp/builtinIS/LinearLwrveIntersector_ptx_bin.h>
#include <exp/builtinIS/LinearLwrveLowMemIntersector_ptx_bin.h>

// Phantom intersector, quadratic Bspline
#include <exp/builtinIS/QuadraticLwrvePhantomIntersector_ptx_bin.h>
#include <exp/builtinIS/QuadraticLwrvePhantomLowMemIntersector_ptx_bin.h>
