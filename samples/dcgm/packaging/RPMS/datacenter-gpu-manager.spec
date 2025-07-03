%global _enable_debug_package 0
%global debug_package %{nil}
%global __os_install_post /usr/lib/rpm/brp-compress %{nil}

%global lwvsdir %{_datarootdir}/lwpu-validation-suite

Name:           datacenter-gpu-manager
Version:        %{?version}
Release:        1
Summary:        LWPU Datacenter GPU Manager

License:        LWPU Proprietary
URL:            http://www.lwpu.com
Source0:        %{name}-%{version}.tar.gz
Requires:       libgomp


%description
LWPU Datacenter GPU Manager

%prep
%setup -q

%build

%install
export DONT_STRIP=1

rm -rf %{buildroot}

mkdir -p %{buildroot}%{_bindir}/
cp dcgm/dcgmi %{buildroot}%{_bindir}/
cp dcgm/lw-hostengine %{buildroot}%{_bindir}/

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

mkdir -p %{buildroot}%{_defaultdocdir}/datacenter-gpu-manager
cp -R -a dcgm/doc/EULA.pdf %{buildroot}%{_defaultdocdir}/datacenter-gpu-manager/
cp -R -a dcgm/doc/THIRD-PARTY-NOTICES.txt %{buildroot}%{_defaultdocdir}/datacenter-gpu-manager/

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
cp dcgm/systemd/dcgm.service %{buildroot}/usr/lib/systemd/system

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
%{lwvsdir}

%changelog
* Tue Dec 1 2015 Chris Hunter
- Added DCGM SDK samples 
* Mon Nov 2 2015 Rob Todd
- Add documentation and license pieces
* Wed Oct 7 2015 Rob Todd
- Renaming of LWCM to DCGM
* Mon Jun 29 2015 Andy Dick
- Initial LWCM RPM packaging
