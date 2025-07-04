// This file is automatically generated by lwwatch-config - DO NOT EDIT!
//
// Private HAL support for {{ENGINE}}.
// 
// Profile:  {{$PROFILE}}
// Haldef:   {{$DEF_FILE}}
// Template: {{$TEMPLATE_FILE}}
// 

#ifndef _G_{{$ENGINE}}_PRIVATE_H_
#define _G_{{$ENGINE}}_PRIVATE_H_

#include "{{$ENG_PUBLIC_FILE}}"

{{ HAL_INTERFACE_PROTOTYPES() }}
{{ HAL_INTERFACE_MISSING_PROTOTYPE() }}

#if defined({{$ENGINE_SETUP_DEFINE}})    // for use by hal init only

//
// Setup {{$ENGINE}}'s hal interface function pointers
//

{{ HAL_INTERFACES_SETUP_FUNCTION(":ENABLED:BY_CHIPFAMILY") }}

#endif  // {{$ENGINE_SETUP_DEFINE}}

#endif  // _G_{{$ENGINE}}_PRIVATE_H_
