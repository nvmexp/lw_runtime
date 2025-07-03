*************
Release Notes
*************

=================
lwTENSOR v1.3.1
=================

* Improved tensor contraction performance model (i.e., algo LWTENSOR_ALGO_DEFAULT)
* Improved performance for tensor contraction that have an overall large contracted
  dimension (i.e., a parallel reduction was added)

*Compatibility notes*:

* Binaries provided for LWCA 10.2/11.0/11.x (x>0) for x86_64 and OpenPower
* Binaries provided for LWCA 11.0/11.x (x>0) for ARM64

=================
lwTENSOR v1.3.0
=================

* Support up to 40-dimensional tensors
* Support 64-bit strides
* Up to 2x performance improvement across library
* Support for BF16 Element-wise operations

*Compatibility notes*:

* Not binary compatible with previous versions, due to added int64 stride support.
* Binaries provided for LWCA 10.2/11.0/11.x (x>0) for x86_64 and OpenPower
* Binaries provided for LWCA 11.0/11.x (x>0) for ARM64

*Resolved issues*:

* Fixed bug with mixed real-complex contraction and strided data

=================
lwTENSOR v1.2.2
=================

* Improved performance for Element-wise operations

*Compatibility notes*:

* Binaries provided for LWCA 10.1/10.2/11.x for x86_64 and OpenPower
* Binaries provided for LWCA 11.x for ARM64

=================
lwTENSOR v1.2.1
=================

* Added examples to https://github.com/LWPU/LWDALibrarySamples

*Compatibility notes*:

* Requires a sufficiently recent (GCC 5 or higher) libstdc++ when linking statically
* Binaries provided for LWCA 10.1/10.2/11.0/11.1 for x86_64 and OpenPower
* Binaries provided for LWCA 11.0/11.1 for ARM64

=================
lwTENSOR v1.2.0
=================

* Support for cache plans and autotuning
* Support BF16 for Elementwise and Reduction

*Compatibility notes*:

* Binaries provided for LWCA 10.1/10.2/11.0 for x86_64 and OpenPower
* Binaries provided for LWCA 11.0 for ARM64

=================
lwTENSOR v1.1.0
=================

* Support for LWCA 11.0
* Added support for `Windows 10 x86_64` and `Linux ARM64` platforms
* Added support for `SM 8.0`
* Support third generation Tensor Cores
* Improved performance

*Compatibility notes*:

* Binaries provided for LWCA 10.1/10.2/11.0 for x86_64 and OpenPower
* Binaries provided for LWCA 11.0 for ARM64

=================
lwTENSOR v1.0.1
=================

* Added support for `SM 6.0`

=================
lwTENSOR v1.0.0
=================

* Initial release
* Support for `SM 7.0`
* Support mixed-precision operations
* Support device-side alpha and beta
* Support C != D

*Compatibility notes*:

* *lwTENSOR* requires LWCA 10.1/10.2 for  for x86_64 and OpenPower
