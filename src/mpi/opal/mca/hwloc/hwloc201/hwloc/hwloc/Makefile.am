# Copyright © 2009-2018 Inria.  All rights reserved.
# Copyright © 2009-2012 Université Bordeaux
# Copyright © 2009-2014 Cisco Systems, Inc.  All rights reserved.
# Copyright © 2011-2012 Oracle and/or its affiliates.  All rights reserved.
# See COPYING in top-level directory.

AM_CFLAGS = $(HWLOC_CFLAGS)
AM_CPPFLAGS = $(HWLOC_CPPFLAGS) -DHWLOC_INSIDE_LIBHWLOC
AM_LDFLAGS = $(HWLOC_LDFLAGS)

EXTRA_DIST = dolib.c

# If we're in standalone mode, build the installable library.
# Otherwise, build the embedded library.

if HWLOC_BUILD_STANDALONE
lib_LTLIBRARIES = libhwloc.la
else
noinst_LTLIBRARIES = libhwloc_embedded.la
endif

pluginsdir = @HWLOC_PLUGINS_DIR@
plugins_LTLIBRARIES =
plugins_ldflags = -module -avoid-version -lltdl
# Beware that files are not rebuilt automatically when reconfiguring with different paths in these flags.
AM_CPPFLAGS += -DHWLOC_PLUGINS_PATH=\"$(HWLOC_PLUGINS_PATH)\" -DRUNSTATEDIR=\"$(HWLOC_runstatedir)\"

# Sources and ldflags

sources = \
        topology.c \
        traversal.c \
        distances.c \
        components.c \
        bind.c \
        bitmap.c \
        pci-common.c \
        diff.c \
        shmem.c \
        misc.c \
        base64.c \
        topology-noos.c \
        topology-synthetic.c \
        topology-xml.c \
        topology-xml-nolibxml.c
ldflags =

# Conditionally add to the sources and ldflags

if HWLOC_HAVE_LIBXML2
if HWLOC_XML_LIBXML_BUILD_STATIC
sources += topology-xml-libxml.c
else
plugins_LTLIBRARIES += hwloc_xml_libxml.la
hwloc_xml_libxml_la_SOURCES = topology-xml-libxml.c
hwloc_xml_libxml_la_CFLAGS = $(AM_CFLAGS) $(HWLOC_LIBXML2_CFLAGS) -DHWLOC_INSIDE_PLUGIN
hwloc_xml_libxml_la_LDFLAGS = $(plugins_ldflags) $(HWLOC_LIBXML2_LIBS)
endif
endif HWLOC_HAVE_LIBXML2

if HWLOC_HAVE_PCIACCESS
if HWLOC_PCI_BUILD_STATIC
sources += topology-pci.c
else
plugins_LTLIBRARIES += hwloc_pci.la
hwloc_pci_la_SOURCES = topology-pci.c
hwloc_pci_la_CFLAGS = $(AM_CFLAGS) $(HWLOC_PCIACCESS_CFLAGS) -DHWLOC_INSIDE_PLUGIN
hwloc_pci_la_LDFLAGS = $(plugins_ldflags) $(HWLOC_PCIACCESS_LIBS)
endif
endif HWLOC_HAVE_PCIACCESS

if HWLOC_HAVE_OPENCL
if HWLOC_OPENCL_BUILD_STATIC
sources += topology-opencl.c
else
plugins_LTLIBRARIES += hwloc_opencl.la
hwloc_opencl_la_SOURCES = topology-opencl.c
hwloc_opencl_la_CFLAGS = $(AM_CFLAGS) $(HWLOC_OPENCL_CFLAGS) -DHWLOC_INSIDE_PLUGIN
hwloc_opencl_la_LDFLAGS = $(plugins_ldflags) $(HWLOC_OPENCL_LIBS) $(HWLOC_OPENCL_LDFLAGS)
endif
endif HWLOC_HAVE_OPENCL

if HWLOC_HAVE_LWDART
if HWLOC_LWDA_BUILD_STATIC
sources += topology-lwca.c
else
plugins_LTLIBRARIES += hwloc_lwda.la
hwloc_lwda_la_SOURCES = topology-lwca.c
hwloc_lwda_la_CFLAGS = $(AM_CFLAGS) $(HWLOC_LWDA_CFLAGS) -DHWLOC_INSIDE_PLUGIN
hwloc_lwda_la_LDFLAGS = $(plugins_ldflags) $(HWLOC_LWDA_LIBS)
endif
endif HWLOC_HAVE_LWDART

if HWLOC_HAVE_LWML
if HWLOC_LWML_BUILD_STATIC
sources += topology-lwml.c
else
plugins_LTLIBRARIES += hwloc_lwml.la
hwloc_lwml_la_SOURCES = topology-lwml.c
hwloc_lwml_la_CFLAGS = $(AM_CFLAGS) $(HWLOC_LWML_CFLAGS) -DHWLOC_INSIDE_PLUGIN
hwloc_lwml_la_LDFLAGS = $(plugins_ldflags) $(HWLOC_LWML_LIBS)
endif
endif HWLOC_HAVE_LWML

if HWLOC_HAVE_GL
if HWLOC_GL_BUILD_STATIC
sources += topology-gl.c
else
plugins_LTLIBRARIES += hwloc_gl.la
hwloc_gl_la_SOURCES = topology-gl.c
hwloc_gl_la_CFLAGS = $(AM_CFLAGS) $(HWLOC_GL_CFLAGS) -DHWLOC_INSIDE_PLUGIN
hwloc_gl_la_LDFLAGS = $(plugins_ldflags) $(HWLOC_GL_LIBS)
endif
endif HWLOC_HAVE_GL

if HWLOC_HAVE_SOLARIS
sources += topology-solaris.c
sources += topology-solaris-chiptype.c
endif HWLOC_HAVE_SOLARIS

if HWLOC_HAVE_LINUX
sources += topology-linux.c topology-hardwired.c
endif HWLOC_HAVE_LINUX

if HWLOC_HAVE_BGQ
sources += topology-bgq.c
endif HWLOC_HAVE_BGQ

if HWLOC_HAVE_AIX
sources += topology-aix.c
ldflags += -lpthread
endif HWLOC_HAVE_AIX

if HWLOC_HAVE_HPUX
sources += topology-hpux.c
ldflags += -lpthread
endif HWLOC_HAVE_HPUX

if HWLOC_HAVE_WINDOWS
sources += topology-windows.c
endif HWLOC_HAVE_WINDOWS

if HWLOC_HAVE_DARWIN
sources += topology-darwin.c
endif HWLOC_HAVE_DARWIN

if HWLOC_HAVE_FREEBSD
sources += topology-freebsd.c
endif HWLOC_HAVE_FREEBSD

if HWLOC_HAVE_NETBSD
sources += topology-netbsd.c
ldflags += -lpthread
endif HWLOC_HAVE_NETBSD

if HWLOC_HAVE_X86_CPUID
sources += topology-x86.c
endif HWLOC_HAVE_X86_CPUID

if HWLOC_HAVE_GCC
ldflags += -no-undefined
endif HWLOC_HAVE_GCC


if HWLOC_HAVE_WINDOWS
# Windows specific rules

LC_MESSAGES=C
export LC_MESSAGES
ldflags += -Xlinker --output-def -Xlinker .libs/libhwloc.def

if HWLOC_HAVE_MS_LIB
dolib$(EXEEXT): dolib.c
	$(CC_FOR_BUILD) $< -o $@
.libs/libhwloc.lib: libhwloc.la dolib$(EXEEXT)
	[ ! -r .libs/libhwloc.def ] || ./dolib$(EXEEXT) "$(HWLOC_MS_LIB)" $(HWLOC_MS_LIB_ARCH) .libs/libhwloc.def $(libhwloc_so_version) .libs/libhwloc.lib
all-local: .libs/libhwloc.lib
clean-local:
	$(RM) dolib$(EXEEXT)
endif HWLOC_HAVE_MS_LIB

install-exec-hook:
	[ ! -r .libs/libhwloc.def ] || $(INSTALL) .libs/libhwloc.def $(DESTDIR)$(libdir)
if HWLOC_HAVE_MS_LIB
	[ ! -r .libs/libhwloc.def ] || $(INSTALL) .libs/libhwloc.lib $(DESTDIR)$(libdir)
	[ ! -r .libs/libhwloc.def ] || $(INSTALL) .libs/libhwloc.exp $(DESTDIR)$(libdir)
endif HWLOC_HAVE_MS_LIB

uninstall-local:
	rm -f $(DESTDIR)$(libdir)/libhwloc.def
if HWLOC_HAVE_MS_LIB
	rm -f $(DESTDIR)$(libdir)/libhwloc.lib $(DESTDIR)$(libdir)/libhwloc.exp
endif HWLOC_HAVE_MS_LIB

# End of Windows specific rules
endif HWLOC_HAVE_WINDOWS


# Installable library

libhwloc_la_SOURCES = $(sources)
libhwloc_la_LDFLAGS = $(ldflags) -version-info $(libhwloc_so_version) $(HWLOC_LIBS) $(HWLOC_LDFLAGS)

if HWLOC_HAVE_PLUGINS
AM_CPPFLAGS += $(LTDLINCL)
libhwloc_la_LDFLAGS += -export-dynamic
libhwloc_la_LIBADD = $(LIBLTDL)
endif

# Embedded library (note the lack of a .so version number -- that
# intentionally only appears in the installable library).  Also note
# the lack of _LDFLAGS -- all libs are added by the upper layer (via
# HWLOC_EMBEDDED_LIBS).

libhwloc_embedded_la_SOURCES = $(sources)

# XML data (only install if we're building in standalone mode)

if HWLOC_BUILD_STANDALONE
xml_DATA = $(srcdir)/hwloc.dtd $(srcdir)/hwloc2.dtd $(srcdir)/hwloc2-diff.dtd
xmldir = $(pkgdatadir)
EXTRA_DIST += hwloc.dtd hwloc2.dtd hwloc2-diff.dtd
endif

DISTCLEANFILES = static-components.h

if HWLOC_HAVE_PLUGINS
check_LTLIBRARIES = hwloc_fake.la
hwloc_fake_la_SOURCES = topology-fake.c
hwloc_fake_la_LDFLAGS = $(plugins_ldflags) -rpath /nowhere # force libtool to build a shared-library even it's check-only
endif
