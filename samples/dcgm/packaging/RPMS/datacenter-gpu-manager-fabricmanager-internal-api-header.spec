# Fabric Manager internal API's header file SPEC File.
# This SPEC file uses the same source tar created for vanilla DCGM package. 


%global _enable_debug_package 0
%global debug_package %{nil}
%global __os_install_post /usr/lib/rpm/brp-compress %{nil}

Name:           datacenter-gpu-manager-fabricmanager-internal-api-header
Version:        %{?version}
Release:        1
Summary:        Fabric Manager internal API header files

License:        LWPU Proprietary
URL:            http://www.lwpu.com
Source0:        datacenter-gpu-manager-%{version}.tar.gz


%description
Fabric Manager Internal API's header files.

%prep
%setup -q -n datacenter-gpu-manager-%{version}

%build

%install
export DONT_STRIP=1

rm -rf %{buildroot}

mkdir -p %{buildroot}%{_includedir}/
cp dcgm/dcgm_module_fm_internal.h %{buildroot}%{_includedir}/
cp dcgm/dcgm_module_fm_structs_internal.h %{buildroot}%{_includedir}/
cp dcgm/dcgm_uuid.h %{buildroot}%{_includedir}/


%files
%{_includedir}/*

%changelog
* Mon Nov 12 2018 Shibu Baby
- Initial version of Fabric Manager API header RPM package
