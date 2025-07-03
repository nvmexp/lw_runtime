# Fabric Manager SPEC File.
# This SPEC file uses the same source tar created for vanilla DCGM package. 
# The main difference/files in this SPEC compared to vanilla DCGM package is as follows:
#  *copy lwpu-fabricmanager.service instead of vanilla dcgm.service
#  *copy Fabric Manager topology file dcgm/topology/topology 


%global _enable_debug_package 0
%global debug_package %{nil}
%global __os_install_post /usr/lib/rpm/brp-compress %{nil}

%global lwvsdir %{_datarootdir}/lwpu-validation-suite

Name:           datacenter-gpu-manager-fabricmanager
Version:        %{?version}
Release:        1
Summary:        Fabric Manager for LWSwitch based systems

License:        LWPU Proprietary
URL:            http://www.lwpu.com
Source0:        datacenter-gpu-manager-%{version}.tar.gz
Requires:       libgomp


%description
Fabric Manager for LWPU LWSwitch based systems.

%prep
%setup -q -n datacenter-gpu-manager-%{version}

%build

%install
export DONT_STRIP=1

rm -rf %{buildroot}

mkdir -p %{buildroot}%{_bindir}/
cp dcgm/dcgmi %{buildroot}%{_bindir}/
cp dcgm/lw-hostengine %{buildroot}%{_bindir}/
cp dcgm/lwswitch-audit %{buildroot}%{_bindir}/

mkdir -p %{buildroot}%{_libdir}/
cp dcgm/libdcgm*.so.1 %{buildroot}%{_libdir}/
cp -a dcgm/libdcgm*.so %{buildroot}%{_libdir}/
cp dcgm/libdcgm_stub.a %{buildroot}%{_libdir}/
cp dcgm/liblwperf_dcgm_host.so %{buildroot}%{_libdir}/

mkdir -p %{buildroot}%{_includedir}/
cp dcgm/dcgm_agent.h %{buildroot}%{_includedir}/
cp dcgm/dcgm_structs.h %{buildroot}%{_includedir}/
cp dcgm/dcgm_fields.h %{buildroot}%{_includedir}/
cp dcgm/dcgm_errors.h %{buildroot}%{_includedir}/
cp dcgm/lwml.h %{buildroot}%{_includedir}/

mkdir -p %{buildroot}%{_defaultdocdir}/datacenter-gpu-manager-fabricmanager
cp -R -a dcgm/doc/EULA.pdf %{buildroot}%{_defaultdocdir}/datacenter-gpu-manager-fabricmanager/
cp -R -a dcgm/doc/THIRD-PARTY-NOTICES.txt %{buildroot}%{_defaultdocdir}/datacenter-gpu-manager-fabricmanager/

mkdir -p %{buildroot}/usr/local/dcgm/
cp -R -a dcgm/sdk_samples/ %{buildroot}/usr/local/dcgm/samples/
cp -R -a dcgm/bindings/ %{buildroot}/usr/local/dcgm/bindings/

mkdir -p %{buildroot}%{lwvsdir}
cp -R -a dcgm/lwvs %{buildroot}%{lwvsdir}
cp -R -a dcgm/plugins %{buildroot}%{lwvsdir}
cp -R -a dcgm/configfile_examples %{buildroot}%{lwvsdir}
cp -R -a dcgm/python_examples %{buildroot}%{lwvsdir}
ln -s lwvs %{buildroot}%{lwvsdir}/lwpu-vs

mkdir -p %{buildroot}%{_sysconfdir}/lwpu-validation-suite
cp -a dcgm/lwvs.conf %{buildroot}%{_sysconfdir}/lwpu-validation-suite

mkdir -p %{buildroot}%{_bindir}
ln -s %{lwvsdir}/lwvs %{buildroot}%{_bindir}/lwvs

mkdir -p %{buildroot}/usr/lib/systemd/system
cp dcgm/systemd/lwpu-fabricmanager.service  %{buildroot}/usr/lib/systemd/system

mkdir -p %{buildroot}/usr/share/lwpu/lwswitch
cp dcgm/topology/topology %{buildroot}/usr/share/lwpu/lwswitch

%post -p /sbin/ldconfig

%postun -p /sbin/ldconfig

%files
%doc
%{_bindir}/*
%{_libdir}/*
%{_includedir}/*
%{_sysconfdir}/*
%{_defaultdocdir}/*
/usr/local/dcgm/*
/usr/lib/systemd/system/*
/usr/share/lwpu/lwswitch/*
%{lwvsdir}

%changelog
* Fri Jun 29 2018 Shibu Baby
- Initial Fabric Manager RPM packaging
