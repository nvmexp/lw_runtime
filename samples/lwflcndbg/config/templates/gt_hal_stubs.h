// This file is automatically generated by {{$name}} - DO NOT EDIT!
//
// HAL stubs, generated by {{$name}}.
//
// Profile:  {{$PROFILE}}
// Template: {{$TEMPLATE_FILE}}
//
// Chips:    {{ CHIP_LIST() }}
//

#ifndef _G_{{$XXCFG}}_HAL_STUBS_H_
#define _G_{{$XXCFG}}_HAL_STUBS_H_

// pull in private headers for each engine
{{ ENGINE_INCLUDE_PRIVATE_HEADER(":ALL_ENGINES") }}

#include "{{$HAL_PUBLIC_FILE}}"

{{ HAL_STUBS() }}

// "missing engine" setup sequences, if any.

{{ HAL_SETUP_MISSING_FUNCTION(":ALL_ENGINES") }}

#endif  // _G_{{$XXCFG}}_HAL_STUBS_H_
