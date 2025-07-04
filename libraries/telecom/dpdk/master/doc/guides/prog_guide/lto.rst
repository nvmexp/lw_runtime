..  SPDX-License-Identifier: BSD-3-Clause
    Copyright(c) 2019 Marvell International Ltd.

Link Time Optimization
======================

The DPDK supports compilation with link time optimization turned on.
This depends obviously on the ability of the compiler to do "whole
program" optimization at link time and is available only for compilers
that support that feature.
To be more specific, compiler (in addition to performing LTO) have to
support creation of ELF objects containing both normal code and internal
representation (called fat-lto-objects in gcc and icc).
This is required since during build some code is generated by parsing
produced ELF objects (pmdinfogen).

The amount of performance gain that one can get from LTO depends on the
compiler and the code that is being compiled.
However LTO is also useful for additional code analysis done by the
compiler.
In particular due to interprocedural analysis compiler can produce
additional warnings about variables that might be used uninitialized.
Some of these warnings might be "false positives" though and you might
need to explicitly initialize variable in order to silence the compiler.

Please note that turning LTO on causes considerable extension of
build time.

Link time optimization can be enabled by setting meson built-in 'b_lto' option:

.. code-block:: console

    meson build -Db_lto=true
