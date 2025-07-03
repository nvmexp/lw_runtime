#!/bin/bash

set -x

#Save the linux distro to a variable for later use
linuxdisto=`grep DISTRIB_ID /etc/*-release | awk -F '=' '{print $2}'`

# Default to lwmake style target of 'amd64'
if [ -z "${TARGET}" ]; then
    TARGET=amd64
fi

# Validate the target
if [ "${TARGET}" != "amd64" ] && [ "${TARGET}" != "ppc64le" ]; then
    echo "Can only target amd64 or ppc64le"
    exit 1
fi

# Set some rpm/deb specific values
if [ "${TARGET}" = "amd64" ]; then
    RPMARCH=x86_64
    DEBARCH=amd64
	DEBLIBINSTDIR=/usr/lib/x86_64-linux-gnu
elif [ "${TARGET}" = "ppc64le" ]; then
    RPMARCH=ppc64le
    DEBARCH=ppc64el
    DEBLIBINSTDIR=/usr/lib/powerpc64le-linux-gnu
fi

if [ -z ${VULCAN} ]; then
	dcgm_rootdir=$(readlink -f ..)
	dcgm_output_dir=${dcgm_rootdir}/_out/Linux_${TARGET}_release
else
	dcgm_rootdir=${VULCAN_COMPONENT_DIR}/..
	dcgm_output_dir=${VULCAN_INSTALL_DIR}/gdk/dcgm
fi

PKG_NAME=datacenter-gpu-manager
# Only change the EPOCH version to throw away old numbering schemes
EPOCH_VERSION=1

VERSION_HEADER=${dcgm_rootdir}/dcgmlib/dcgm_structs_internal.h
VERSION=$(grep DCGM_VERSION_STRING ${VERSION_HEADER} | awk '{print $3}' | sed 's/\"//g')

VERSIONING_ELW_VARS=
if [ -n "${LW_DVS_BLD}" ]; then
    VERSIONING_ELW_VARS="--elwvar LW_DVS_BLD"
fi
if [ -n "${CHANGELIST}" ]; then
    VERSIONING_ELW_VARS="${VERSIONING_ELW_VARS} --elwvar CHANGELIST"
fi
if [ -n "${DVS_SW_CHANGELIST}" ]; then
    VERSIONING_ELW_VARS="${VERSIONING_ELW_VARS} --elwvar DVS_SW_CHANGELIST"
fi
if [ -n "${LW_BUILD_CL}" ]; then
    VERSIONING_ELW_VARS="${VERSIONING_ELW_VARS} --elwvar LW_BUILD_CL"
fi

# Make DCGM
# If ilwoked by vulcan, this can be skipped as it is handled by building dcgm.vlcc
if [ -z ${VULCAN} ]; then
	if [ -z ${BUILD_TOOLS_DIR} ]; then
		cd ${dcgm_rootdir}; unix-build ${VERSIONING_ELW_VARS} lwmake release ${TARGET} clobber -j8
		cd ${dcgm_rootdir}; unix-build ${VERSIONING_ELW_VARS} lwmake release ${TARGET} -j8
	else
		cd ${dcgm_rootdir}; unix-build ${VERSIONING_ELW_VARS} --tools ${BUILD_TOOLS_DIR} lwmake release ${TARGET} clobber -j8
		cd ${dcgm_rootdir}; unix-build ${VERSIONING_ELW_VARS} --tools ${BUILD_TOOLS_DIR} lwmake release ${TARGET} -j8
	fi
fi

if ! [[ -d ${dcgm_output_dir} ]]; then
	echo "Can not find DCGM output directory. exiting..."
	exit 1
fi

# Clean
cd ${dcgm_output_dir} && rm -rf ../packaging


###############################################################################
# Join DCGM (and formerly LWVS)                                               #
###############################################################################
mkdir -p ../packaging/dcgm_merge
cp -R ${dcgm_output_dir} ../packaging/dcgm_merge/dcgm

###############################################################################
# Debian packages                                                             #
###############################################################################
# manually setup the dhbmake package locations
# set all the autoscripts path like postinst-systemd-enable
export DH_AUTOSCRIPTDIR=${BUILD_AUTOMATION_TOOLS_DIR}/debhelper-10.2.2/autoscripts/
# let python modules to have our location in @INC include path for modules
export PERL5LIB=${BUILD_AUTOMATION_TOOLS_DIR}/debhelper-10.2.2/
export PERL5LIB=$PERL5LIB:${BUILD_AUTOMATION_TOOLS_DIR}/devscripts-2.16.8/lib
# include dhmake and debhelper location to PATH
export PATH=/${BUILD_AUTOMATION_TOOLS_DIR}/debhelper-10.2.2/:$PATH
export PATH=/${BUILD_AUTOMATION_TOOLS_DIR}/dhmake-2.201608:$PATH
export PATH=/${BUILD_AUTOMATION_TOOLS_DIR}/devscripts-2.16.8:$PATH
export PATH=/${BUILD_AUTOMATION_TOOLS_DIR}/devscripts-2.16.8/scripts:$PATH
# start debuild with our PATH and keeping all our ELW variables.
# also let it run as fakeroot as dvs-builder account is not root
debuild_extra_options="--preserve-elw --preserve-elwvar PATH -b -rfakeroot"

# Set up debian packaging and create debian-specific formatted tarball
cd ../packaging
mkdir -p DEBS/${PKG_NAME}_${VERSION}
cp -R ${dcgm_rootdir}/packaging/DEBS/debian DEBS/${PKG_NAME}_${VERSION}
cp -R dcgm_merge/* DEBS/${PKG_NAME}_${VERSION}
cd dcgm_merge
tar -czf ../DEBS/${PKG_NAME}_${VERSION}.orig.tar.gz *

# Create the debian packages
cd ../DEBS/${PKG_NAME}_${VERSION}
make -f debian/rules fill_templates EPOCH=${EPOCH_VERSION} REVISION=${VERSION} DEBLIBINSTDIR=${DEBLIBINSTDIR}
debuild $debuild_extra_options -us -uc -d -a${DEBARCH}

###############################################################################
# RPM packages                                                                #
###############################################################################
if [ "$linuxdisto" = "Ubuntu" ] ; then
  echo "Skipping RPM generation for Ubuntu since ubuntu rpmbuild puts libs in the wrong folder"
  exit
fi

# Set up RPM packaging and create RPM-specific formatted tarball
# Fabric Manager will use same source tar
FM_PKG_NAME=$PKG_NAME-fabricmanager
FM_API_HEADER_PKG_NAME=$PKG_NAME-fabricmanager-internal-api-header

cd ${dcgm_output_dir}/../packaging
mkdir -p RPMS/{RPMS,SRPMS,BUILD,BUILDROOT,SPECS,SOURCES}
# copy all the SPEC files
cp ${dcgm_rootdir}/packaging/RPMS/${PKG_NAME}.spec RPMS/SPECS
cp ${dcgm_rootdir}/packaging/RPMS/${FM_PKG_NAME}.spec RPMS/SPECS
cp ${dcgm_rootdir}/packaging/RPMS/${FM_API_HEADER_PKG_NAME}.spec RPMS/SPECS

mkdir -p RPMS/SOURCES/${PKG_NAME}-${VERSION}
cd RPMS/SOURCES
cp -R ../../dcgm_merge/* ${PKG_NAME}-${VERSION}
tar -czf ${PKG_NAME}-${VERSION}.tar.gz ${PKG_NAME}-${VERSION}

cd ..

# Create the RPM packages
rpmbuild -ba SPECS/${PKG_NAME}.spec \
    --define "%version ${VERSION}" \
    --define "%_topdir ${PWD}" \
    --define "%_arch ${RPMARCH}" \
    --define "%_build_arch ${RPMARCH}" \
    --target=${RPMARCH}

# Create Fabric Manager RPM package
rpmbuild -ba SPECS/${FM_PKG_NAME}.spec \
    --define "%version ${VERSION}" \
    --define "%_topdir ${PWD}" \
    --define "%_arch ${RPMARCH}" \
    --define "%_build_arch ${RPMARCH}" \
    --target=${RPMARCH}

# Create Fabric Manager API Header package
rpmbuild -ba SPECS/${FM_API_HEADER_PKG_NAME}.spec \
    --define "%version ${VERSION}" \
    --define "%_topdir ${PWD}" \
    --define "%_arch ${RPMARCH}" \
    --define "%_build_arch ${RPMARCH}" \
    --target=${RPMARCH}    