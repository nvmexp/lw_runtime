// This file is automatically generated by {{$name}} - DO NOT EDIT!
//
// defines to indicate enabled/disabled for all chips, features, classes, engines, and apis.
//
// Profile:  {{$PROFILE}}
// Template: {{$TEMPLATE_FILE}}
//
// Chips:    {{ CHIP_LIST() }}
//

#ifndef _{{$XXCFG}}_H_
#define _{{$XXCFG}}_H_


//
// CHIP families - enabled or disabled
//
{{ CHIP_FAMILY_DEFINES(":ALL") }}

//
// CHIPS - enabled or disabled
//
{{ CHIP_DEFINES(":ALL") }}

//
// Features - enabled or disabled
//
{{ GROUP_DEFINES ("FEATURES", ":ALL") }}
{{ GROUP_SPECIALS("FEATURES") }}

//
// List of all known chips, features, apis, classes, etc.
// Used by the {{$XXCFG}}_xxx_ENABLED macros to detect misspellings
// in param to RMCFG_FEATURE_ENABLED(), etc on some platforms.
//
{{ CHIP_DEFTEST_DEFINES()            }}
{{ GROUP_DEFTEST_DEFINES('FEATURES') }}
{{ GROUP_DEFTEST_DEFINES('ENGINES')  }}

// Make sure the specified feature is defined and not a misspelling
// by checking the "_def" forms above which are all set to '1' for
// each defined chip, feature, etc, irrespective of it's enable/disable
// state.
// NOTE: some compilers will warn on every instance of this trick, so
// only enable for GCC builds.
#define _{{$XXCFG}}_vet(x)  0

#if defined(LWRM) && defined(__GNUC__) && !defined(__clang__)
#  undef  _{{$XXCFG}}_vet
#  define _{{$XXCFG}}_vet(x)  ((__def_{{$XXCFG}} ## x) ? 0 : (0 * (1/0)))
#endif
//

// Compile-time constant macros to help with enabling or disabling code based
// on whether a chip is enabled.
// May be used by both C code ('if') and C-preprocessor directives ('#if')
//

#define {{$XXCFG}}_CHIP_ENABLED(CHIP)       ({{$XXCFG}}_CHIP_##CHIP)
#define {{$XXCFG}}_FEATURE_ENABLED(FEATURE) ({{$XXCFG}}_FEATURE_##FEATURE + _{{$XXCFG}}_vet(_FEATURE_##FEATURE))
#define {{$XXCFG}}_IS_PLATFORM(P)           ({{$XXCFG}}_FEATURE_PLATFORM_##P)

#endif // _LWWATCH_CONFIG_H_
