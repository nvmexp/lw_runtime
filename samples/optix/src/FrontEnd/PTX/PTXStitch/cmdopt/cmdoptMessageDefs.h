/*
 * LWIDIA_COPYRIGHT_BEGIN
 *
 * Copyright (c) 2020, LWPU CORPORATION.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 *
 * LWIDIA_COPYRIGHT_END
 */
/*
 *  Module name              : cmdoptMessageDefs.h
 *
 *  Description              :
 *
 *         This file defines the messages generated
 *         by the command option parsing library.
 */

#ifndef cmdoptMessageDefs_INCLUDED
#define cmdoptMessageDefs_INCLUDED

#include <misc/stdMessageDefsBegin.h>

MSG( cmdoptMsgArgumentExpected,    Fatal,   "argument expected after '%s'"                )
MSG( cmdoptMsgNoArgumentExpected,  Fatal,   "no argument expected after '%s'"             )
MSG( cmdoptMsgRedefinedArgument,   Fatal,   "redefinition of argument '%s'"               )
MSG( cmdoptMsgOverriddenIncompatible,  Warning, "incompatible redefinition for option '%s', the last value of this option was used")
MSG( cmdoptMsgIsNoKeyword,         Fatal,   "'%s' is not in 'keyword=value' format"       )
MSG( cmdoptMsgRedefinedKeyword,    Fatal,   "redefinition of keyword '%s' "               )
MSG( cmdoptMsgNotABool,            Fatal,   "'%s': expected true or false"                )
MSG( cmdoptMsgNotANumber,          Fatal,   "'%s': expected a number"                     )
MSG( cmdoptMsgUnknownOption,       Fatal,   "Unknown option '%s'"                         )
MSG( cmdoptMsgKeywordNotInDomain,  Fatal,   "Keyword '%s' is not defined for option '%s'" )
MSG( cmdoptMsgValueNotInDomain,    Fatal,   "Value '%s' is not defined for option '%s'"   )
MSG( cmdoptMsgOptionNotDefined,    Fatal,   "No value specified for option '%s'"          )
MSG( cmdoptMsgOpenOptFailed,       Fatal,   "Could not open options file '%s'"            )
MSG( cmdoptMsgReadProblem,         Fatal,   "Failed to read contents of options file"     )
MSG( cmdoptMsgDeprecated,          Warning, "option '%s' has been deprecated"             )
MSG( cmdoptMsgTooManyOptFileOpened,Fatal,   "Too many options file opened '%s'"          )
MSG( cmdoptMsgOutOfRange,          Fatal,   "%s value (%s) out of range"                 )

#include <misc/stdMessageDefsEnd.h>

#endif

