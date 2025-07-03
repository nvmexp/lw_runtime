/*
 * LWIDIA_COPYRIGHT_BEGIN
 *
 * Copyright (c) 2006-2021, LWPU CORPORATION.  All rights reserved.
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
 *  Module name              : stdCmdOpt.c
 *
 *  Description              :
 */

/*-------------------------------- Includes ----------------------------------*/

#include "g_lwconfig.h"
#include "stdCmdOpt.h"
#include "stdLocal.h"
#include "stdMap.h"
#include "stdSet.h"
#include "stdList.h"
#include "stdString.h"
#include "stdMessages.h"
#include "stdStdFun.h"
#include "stdFileNames.h"
#include "stdUtils.h"
#include "cmdoptMessageDefs.h"

/*---------------------------------- Types -----------------------------------*/

typedef struct {
    listXList        (list);
} *ListValue;


typedef struct {
    String            name;
    String            shortName;
    cmdoptType        type;
    cmdoptMode        mode;
    uInt              flags;
    stdSet_t          valueDomain;
    stdSet_t          keywordDomain;
    String            description;
    String            valueFormat;
    Bool              hasValue;
    Bool              hasDefaultValue;
    Pointer           value;
    Pointer           defaultValue;
    Pointer           defaultWhenSpecified;
    uInt              position;     // where oclwrred in arg list
    stdList_t         positionList; // list of positions
} cmdoptOption, 
 *cmdoptOption_t;


typedef struct {
    String            description;
    cmdoptGroupFlags  flags;
    listXList         (options);
} *OptionGroup;


struct cmdoptDescr {
    stdMap_t          options;
    stdMap_t          shorts;
    OptionGroup       lwrrentGroup;
    Bool              saveUnknownOpts;
    listXList        (optionGroups);
    Int               optFileIncluded;
    Int               lwrOptPos;       /* current option position*/
};

/*---------------------------- Utility Functions -----------------------------*/

typedef union { Pointer p; Float f; } PFType;

static Float citof( Pointer value )
{
    PFType pf;
    pf.p= value;
    return pf.f;
}

static Pointer cftoi( Float value )
{
    PFType pf;
    pf.f= value;
    return pf.p;
}

/*------------------------------ API Functions -------------------------------*/

/*
 * Function        : Create a new command line option object.
 *                   Option descriptors have to be added to it
 *                   using repeated use of function cmdoptAddOption,
 *                   after which it can be used for parsing a command line.
 *                   The option is created with a default command option group
 *                   (see also function cmdoptAddGroup).
 * Parameters      : saveUnknownOpts      (I)  Save the unknown options encountered during parsing
 * Function Result : empty object
 */
cmdoptDescr_t cmdoptCreate(Bool saveUnknownOpts)
{
    cmdoptDescr_t result;
    
    stdNEW(result);
    
    result->saveUnknownOpts = saveUnknownOpts;
    result->options= mapNEW(String,10);
    result->shorts = mapNEW(String,10);
    result->lwrOptPos = 0;
    result->optFileIncluded = 0;
    listXInit( result->optionGroups );
    cmdoptAddGroup( result, cmdoptSortGroup, "Options" );
    
    cmdoptAddOption( result, cmdoptNormalArg, cmdoptNormalArg, cmdoptString, cmdoptListValue, cmdoptHidden, Nil, Nil, Nil, Nil, Nil, "" );
    cmdoptAddOption( result, cmdoptUnknownOptionArg, cmdoptUnknownOptionArg, cmdoptString, cmdoptListValue, cmdoptHidden, Nil, Nil, Nil, Nil, Nil, "" );
    
    return result;
}



/*
 * Function        : Add a new command option group to the specified options descriptor,
 *                   and make it 'current'. I.e., until a next call to this function,
 *                   every option added using cmdoptAddOption will become member of
 *                   this option group. 
 *                   Command option groups are used during command option printing (by using cmdoptPrint): 
 *                   all groups are printed in order of declaration.
 * Parameters      : options     (I)  Options descriptor to modify.
 *                   flags       (I)  Attributes of group.
 *                   description (I)  Description of added option group.
 * Function Result : 
 */
void cmdoptAddGroup( cmdoptDescr_t     options,
                     cmdoptGroupFlags  flags,
                     cString           description )
{
    OptionGroup group;

    stdNEW( group );
    
    group->flags       = flags;
    group->description = S(description);
    listXInit(group->options);
    
    listXPutAfter( options->optionGroups, group );
    
    options->lwrrentGroup= group;
}


    static ListValue newListValue(void)
    {
        ListValue result;

        stdNEW(result);

        listXInit(result->list);

        return result;   
    }

        static void addStringValueToSet( String value, stdSet_t set ) 
        { 
            setInsert(set, value);
        }

    static stdSet_t mkStringSet( String values, Bool caseInsensitive )
    {
        stdSet_t result;
        
        if (caseInsensitive) { result= setNEW(CIString,10); }
                        else { result= setNEW(String,  10); }
        
        stdTokenizeString( values, ",", False, False, (stdEltFun)addStringValueToSet, result, False, False );
        
        return result;
    }
    
        static void addIntValueToSet( String value, stdSet_t set ) 
        { 
            Char  *end;
            long result;
            errno = 0;
            result= strtol(value,&end,0);
            stdCHECK(errno == 0, (cmdoptMsgOutOfRange, "32-bit integer", value));
            stdCHECK(result <= INT_MAX && result >= INT_MIN, (cmdoptMsgOutOfRange, "32-bit integer", value));
            stdCHECK( *end == 0, (cmdoptMsgNotANumber, value) );
            
            setInsert(set, (Pointer)(Address)result);
        }

        static void adduIntValueToSet( String value, stdSet_t set ) 
        { 
            Char  *end;
            unsigned long result;
            errno = 0;
            result= strtoul(value,&end,0);
            stdCHECK(errno == 0, (cmdoptMsgOutOfRange, "32-bit unsigned integer", value));
            stdCHECK(result <= UINT_MAX, (cmdoptMsgOutOfRange, "32-bit unsigned integer", value));
            stdCHECK( *end == 0, (cmdoptMsgNotANumber, value) );
            
            setInsert(set, (Pointer)(Address)result);
        }

    static stdSet_t mkIntSet( String values, Bool isSigned )
    {
        stdSet_t result;
        
        result= setNEW(Int,10);
        
        if (isSigned) {
            stdTokenizeString( values, ",", False, False, (stdEltFun)addIntValueToSet,  result, False, False );
        } else {
            stdTokenizeString( values, ",", False, False, (stdEltFun)adduIntValueToSet, result, False, False );
        }
        
        return result;
    }
    
        static void addInt64ValueToSet( String value, stdSet_t set ) 
        { 
            Char  *end;
            uInt64 result;
            errno = 0;
            result= strtoll(value,&end,0);
            stdCHECK(errno == 0, (cmdoptMsgOutOfRange, "64-bit integer", value));
            stdCHECK( *end == 0, (cmdoptMsgNotANumber, value) );
            
            setInsert(set, stdCOPY(&result));
        }

        static void adduInt64ValueToSet( String value, stdSet_t set ) 
        { 
            Char  *end;
            uInt64 result;
            errno = 0;
            result= strtoull(value,&end,0);
            stdCHECK(errno == 0, (cmdoptMsgOutOfRange, "64-bit unsigned integer", value));
            stdCHECK( *end == 0, (cmdoptMsgNotANumber, value) );
            
            setInsert(set, stdCOPY(&result));
        }

    static stdSet_t mkInt64Set( String values, Bool isSigned )
    {
        stdSet_t result;
        
        result= setNEW(pInt64,10);
        
        if (isSigned) {
            stdTokenizeString( values, ",", False, False, (stdEltFun)addInt64ValueToSet,  result, False, False );
        } else {
            stdTokenizeString( values, ",", False, False, (stdEltFun)adduInt64ValueToSet, result, False, False );
        }
        
        return result;
    }
    
    static Pointer initOptiolwalue( cmdoptOption_t option )
    {
        switch (option->mode) {
        case cmdoptNoValue          : return Nil; 
        case cmdoptSingleValue      : return Nil;
        case cmdoptKeywordValue     : return Nil; 
        case cmdoptListValue        : return newListValue();
        case cmdoptListKeywordValue : return newListValue(); 
        default                     : stdASSERT( False, ("Case label out of range") );
        }
        return Nil; /* keep compiler happy */
    }
    
    static void addOptiolwalue( cmdoptDescr_t options, cmdoptOption_t option, String value, Bool addAsOne );


/*
 * Function        : Add a new option descriptor to specified option object.
 *                   The option will be member of the lwrrently active command option group.
 * Parameters      : options              (IO) Option descriptor to add to.
 *                   name                 (I)  Name of option.
 *                   shortName            (I)  Shortened name for option.
 *                   type                 (I)  Type of option; see description in header.
 *                   mode                 (I)  Mode of option; see description in header.
 *                   flags                (I)  Attributes of option; see description in header.
 *                   valueDomain          (I)  Value domain of option, or Nil for no restrictions
 *                                                             see description in header.
 *                   keywordDomain        (I)  Keyword domain of option, or Nil for no restrictions.
 *                                                             see description in header.
 *                   defaultValue         (I)  Default value of option when not specified
 *                   defaultWhenSpecified (I)  In case of Keyword or Single value, default value of option 
 *                                                             when keyword specified without value (using '=')
 *                   valueFormat          (I)  Format for option value, used in cmdoptPrint.
 *                   description          (I)  Textual description of option, used in cmdoptPrint.
 * Function Result : 
 */
void cmdoptAddOption( cmdoptDescr_t     options, 
                      cString           name, 
                      cString           shortName,
                      cmdoptType        type,  
                      cmdoptMode        mode,
                      uInt              flags,
                      cString           valueDomain,
                      cString           keywordDomain,
                      cString           defaultValue,
                      cString           defaultWhenSpecified,
                      cString           valueFormat,
                      cString           description )
{
    cmdoptOption_t option;
    
    stdNEW(option);
    
    option->name                 = S(name      ? name      : "");
    option->shortName            = S(shortName ? shortName : "");
    option->hasValue             = False;
    option->hasDefaultValue      = False;
    option->type                 = type;
    option->mode                 = mode;
    option->flags                = flags;
    option->description          = S(description);
    option->valueFormat          = S(valueFormat ? valueFormat : "");
    option->defaultWhenSpecified = S(defaultWhenSpecified);

    /* By default, allow redefinition of single value option. The
       option parser code will check for incompatible redefinition and
       issue a warning. 
       There is no way lwrrently to disallow redefinition; this can be added 
       as a new entry in cmdoptFlags and checked here. */
    if (option->mode == cmdoptSingleValue) {
      option->flags |= cmdoptRedefinition;
    }

    if (!(option->flags & cmdoptDisabled) ) { 
        listXPutAfter( options->lwrrentGroup->options, option );
    }
    
    if (flags & (cmdoptSingleLetter | cmdoptESingleLetter)) {
        stdASSERT( option->mode     != cmdoptNoValue, ("Single letter option '%s' cannot have mode 'NoValue'",            name) );
        stdASSERT( strlen(shortName)==             1, ("Single letter option '%s' must have a single= letter short name", name) );
    }
    
    if (valueDomain != Nil && !stdISEMPTYSTRING(valueDomain)) {
        switch (type) {
            
        case cmdoptHex         : 
        case cmdoptHex32       : option->valueDomain= mkIntSet   (S(valueDomain),False); break; 
        case cmdoptHex64       : option->valueDomain= mkInt64Set (S(valueDomain),False); break; 
        case cmdoptInt64       : option->valueDomain= mkInt64Set (S(valueDomain),True ); break; 
        case cmdoptInt         : option->valueDomain= mkIntSet   (S(valueDomain),True ); break;
        case cmdoptString      : option->valueDomain= mkStringSet(S(valueDomain),False); break;
        case cmdoptCIString    : option->valueDomain= mkStringSet(S(valueDomain),True ); break;

        case cmdoptFloat       : 
        case cmdoptBool        : 
        case cmdoptOptionsFile : break;

        default                : stdASSERT( False, ("Case label out of range") );
        }
    }
    
    if ( keywordDomain != Nil && !stdISEMPTYSTRING(keywordDomain) ) {
        option->keywordDomain= mkStringSet(S(keywordDomain),False);
    }
    
    option->value= initOptiolwalue(option);
    
    stdASSERT( !stdISEMPTYSTRING(option->name), ("No empty string for option names allowed") );
    
    if ( mapDefine(options->options, option->name, option) != Nil ) {
        stdASSERT( False, ("Duplicate option defined: %s", name) );
    }
    
    if ( !stdISEMPTYSTRING( option->shortName )
      && mapDefine(options->shorts, option->shortName, option) != Nil 
       ) {
        stdASSERT( False, ("Duplicate short option defined: %s", option->shortName) );
    }

    if ( defaultValue != Nil && !stdISEMPTYSTRING(defaultValue) ) {
        addOptiolwalue(options, option, S(defaultValue), False);
        
        if (option->mode == cmdoptListKeywordValue) { 
            ListValue list= option->value;
            option->defaultValue= list->list->head;
        } else {
            option->defaultValue = option->value;
        }
        
        option->hasDefaultValue  = True;
        option->hasValue         = False;
        option->value            = initOptiolwalue(option);
    }
}




/*
 * Function        : Retrieve value of named option. The result is delivered via
 *                   parameter 'value', in a format that is depending on the
 *                   mode and type of the option.
 *                   This value is only defined when the function returns
 *                   'True', indicating that the option was specified in the
 *                   parsed command line:
 *                       mode noValue:        undefined (oclwrrence of option 
 *                                                          via function result)
 *                       mode SingleValue:    Int, Bool or String, 
 *                                            depending on option type
 *                       mode ListValue:      stdList_t of Int, Bool or String, 
 *                                            depending on option type.
 *                                            The order of oclwrrence of of the
 *                                            values in the list is identical to 
 *                                            the order on which they oclwrred on
 *                                            the command line
 *                       mode KeywordValue:   stdMap_t from keyword String 
 *                                            to Int, Bool or String, 
 *                                            depending on option type
 *
 * Parameters      : options  (I)  options descriptor to inspect
 *                   name     (I)  name of option
 *                   value    (O)  retrieved option value, as described above
 * Function Result : True iff. an option value is returned (either parsed from
 *                   the command line, or a default value;
 *
 * NB              : The regular command line arguments (the non-options)
 *                   can be retrieved using cmdoptNormalArg
 */
static Bool __cmdoptGetOptiolwalue( cmdoptOption_t  option, 
                                    cString         name, 
                                    Pointer         value)
{
    Pointer val;
    Bool    result;
    
    if (option->flags & cmdoptRequiredOption) {
        stdCHECK( option->hasValue | option->hasDefaultValue, 
                    (cmdoptMsgOptionNotDefined, option->name) );
    }

    if ( option->hasValue ) {
       val    = option->value;
       result = True;
    } else 
    if ( option->hasDefaultValue  && (option->mode != cmdoptListKeywordValue) ) {
       val    = option->defaultValue;
       result = True;
    } else {
       val    = option->value;
       result = False;
    }
                        
    if ( value != Nil ) {
        switch (option->mode) {
        case cmdoptKeywordValue     : *(Pointer*)value = val;
                                       break;

        case cmdoptNoValue          :  val= (Pointer)(Address)result;  /* FALLTHROUGH */
        case cmdoptSingleValue      :  switch (option->type) {
                                       case cmdoptBool:
                                          *(Bool*) value = (Bool)(Address)val; 
                                          break;
                                          
                                       case cmdoptString:
                                       case cmdoptCIString:
                                          *(String*)value = (String)val; 
                                          break;

                                       case cmdoptFloat :
                                       case cmdoptInt   :
                                       case cmdoptHex   :
                                       case cmdoptHex32 :
                                          *(uInt32*)value = (uInt32)(Address)val; 
                                          break;
                                          
                                       case cmdoptInt64 :
                                       case cmdoptHex64 :
                                          *(uInt64*)value = val ? *(uInt64*)val : 0;
                                          break;
                                          
                                       case cmdoptOptionsFile:
                                          stdASSERT( False, ("Case label out of range") );
                                          break;
                                       }
                                       break;

        case cmdoptListKeywordValue :
        case cmdoptListValue        : { ListValue l= val;
                                       *(Pointer*)value = l->list;
                                        break;
                                      }
        default                     : stdASSERT( False, ("Case label out of range") );
        }
    }
    
    return result;
}

Bool _cmdoptGetOptiolwalue( cmdoptDescr_t   options, 
                            cString         name, 
                            Pointer         value,
                            uInt            valueSize)
{
    Bool   result;
    uInt64 tmp = 0;
    
    cmdoptOption_t option= mapApply( options->options, S(name) );
    stdASSERT( option, ("Unknown option requested: %s",  name) );

    switch (option->mode) {
    case cmdoptNoValue          : 
    case cmdoptSingleValue      : // scalars handled by next switch
                                  break; 
    
    case cmdoptKeywordValue     : 
    case cmdoptListKeywordValue :
    case cmdoptListValue        :
                                  stdASSERT( valueSize == sizeof(Pointer), 
                                            ("cmdoptGetOptiolwalue for %s: unexpected size for String result", name) );
                                  return __cmdoptGetOptiolwalue(option,name,value);
                                  
    default                     : stdASSERT( False, ("Case label out of range") );
    }


    switch (option->type) {
    case cmdoptString   :
    case cmdoptCIString :
       stdASSERT( valueSize == sizeof(String), ("cmdoptGetOptiolwalue for %s: unexpected size for String result", name) );
       return __cmdoptGetOptiolwalue(option,name,value);

    case cmdoptFloat  :
       stdASSERT( valueSize == sizeof(Float),  ("cmdoptGetOptiolwalue for %s: unexpected size for Float result", name) );
       return __cmdoptGetOptiolwalue(option,name,value);
       break;

    case cmdoptBool   :
       stdASSERT( valueSize == sizeof(Bool),   ("cmdoptGetOptiolwalue for %s: unexpected size for Bool result", name) );
       return __cmdoptGetOptiolwalue(option,name,value);
       break;
   
    case cmdoptInt    :
    case cmdoptInt64  :
    case cmdoptHex    :
    case cmdoptHex32  :
    case cmdoptHex64  :
       result =  __cmdoptGetOptiolwalue(option,name,(Pointer)&tmp);
       switch (valueSize) {
       case 1  : *(uInt8* )value = tmp; break;
       case 2  : *(uInt16*)value = tmp; break;
       case 4  : *(uInt32*)value = tmp; break;
       case 8  : *(uInt64*)value = tmp; break;
       default : stdASSERT( False, ("cmdoptGetOptiolwalue for %s: unexpected size for integer result", name) );
       }
       return result;
   
    case cmdoptOptionsFile:
       stdASSERT( False, ("cmdoptOptionsFile type not allowed for cmdoptGetOptiolwalue") );
       break;
    }
    
    return False;
}
 
uInt cmdoptGetOptionPosition( cmdoptDescr_t   options,
                              cString         name)
{
    cmdoptOption_t option= mapApply( options->options, S(name) );
    stdASSERT( option, ("Unknown option requested: %s",  name) );
    return option->position;
}

stdList_t cmdoptGetOptionPositionList( cmdoptDescr_t   options,
                                  cString         name)
{
    cmdoptOption_t option= mapApply( options->options, S(name) );
    stdASSERT( option, ("Unknown option requested: %s",  name) );
    stdASSERT( option->mode == cmdoptListValue, ("must be ListValue"));
    return option->positionList;
}

/*
 * Function        : Test if option was specified on command line
 * Parameters      : options  (I)  options descriptor to inspect
 *                   name     (I)  name of option
 * Function Result : True iff. an option value was specified on the
 *                   parsed command line.
 */
Bool cmdoptOnCommandLine( cmdoptDescr_t   options, 
                          cString         name )
{
    cmdoptOption_t option= mapApply( options->options, S(name) );
    
    stdASSERT( option, ("Unknown option requested: %s",  name) );

    return  option->hasValue;
}


/*
 * Function        : Set additional flags for command line option.
 * Parameters      : options  (I)  options descriptor to modify.
 *                   name     (I)  name of option.
 *                   flags    (I)  flags to set.
 * Function Result : Previous option flags.
 */
uInt32 cmdoptSetFlags( cmdoptDescr_t   options, 
                       cString         name,
                       uInt32          flags )
{
    uInt32 result;

    cmdoptOption_t option= mapApply( options->options, S(name) );
    
    stdASSERT( option, ("Unknown option requested: %s",  name) );

    result= option->flags;

    option->flags |= flags;

    return result;
}


/*-------------------------- Command Option Parsing --------------------------*/

static Pointer optiolwalue( cmdoptOption_t option, String value )
{
    switch (option->type) {

    case cmdoptBool    : 
            if (stdEQSTRING(value,"true" )) { 
                return (Pointer)True; 
            } else 
                    
            if (stdEQSTRING(value,"false")) { 
                return (Pointer)False; 
            } else {
                    
                stdCHECK( False, (cmdoptMsgNotABool, value) );
                return Nil;
            }
                         
    case cmdoptFloat   : 
            {
                Char   *end;
                Float  result;

                result= (Float)strtod(value,&end);
                stdCHECK( *end == 0, (cmdoptMsgNotANumber, value) );

                return (Pointer)cftoi(result);
            }
    
    case cmdoptInt     : 
            {
                Char *end;
                long result;

                errno = 0;
                result= strtol(value,&end,0); // should this be strtoul?
                stdCHECK(errno == 0, (cmdoptMsgOutOfRange, "32-bit integer", value));
                // strtol returns 64bit long on linux, 
                // so check that it fits in 32bits.
                stdCHECK(result <= INT_MAX && result >= INT_MIN, (cmdoptMsgOutOfRange, "32-bit integer", value));
                stdCHECK( *end == 0, (cmdoptMsgNotANumber, value) ) {
                    stdCHECK( 
                       stdIMPLIES( option->valueDomain, setContains(option->valueDomain, (Pointer)(Address)result) ),
                       (cmdoptMsgValueNotInDomain, value, option->name)
                    );
                }

                return (Pointer)(Address)result;
            }
    
    case cmdoptHex     : 
    case cmdoptHex32   : 
            {
                Char *end;
                unsigned long result;

                errno = 0;
                result= strtoul(value,&end,0);
                stdCHECK(errno == 0, (cmdoptMsgOutOfRange, "32-bit hex", value));
                stdCHECK(result <= UINT_MAX, (cmdoptMsgOutOfRange, "32-bit hex", value));
                stdCHECK( *end == 0, (cmdoptMsgNotANumber, value) ) {
                    stdCHECK( 
                       stdIMPLIES( option->valueDomain, setContains(option->valueDomain, (Pointer)(Address)result) ),
                       (cmdoptMsgValueNotInDomain, value, option->name)
                    );
                }
                return (Pointer)(Address)result;
            }
    
    case cmdoptInt64   : 
            {
                Char   *end;
                uInt64  result;

                errno = 0;
                result= strtoll(value,&end,0);
                stdCHECK(errno == 0, (cmdoptMsgOutOfRange, "64-bit integer", value));
                stdCHECK( *end == 0, (cmdoptMsgNotANumber, value) ) {
                    stdCHECK( 
                       stdIMPLIES( option->valueDomain, setContains(option->valueDomain, (Pointer)&result) ),
                       (cmdoptMsgValueNotInDomain, value, option->name)
                    );
                }
                return stdCOPY(&result);
            }
    
    case cmdoptHex64   : 
            {
                Char   *end;
                uInt64  result;

                errno = 0;
                result= strtoull(value,&end,0);
                stdCHECK(errno == 0, (cmdoptMsgOutOfRange, "64-bit hex", value));
                stdCHECK( *end == 0, (cmdoptMsgNotANumber, value) ) {
                    stdCHECK( 
                       stdIMPLIES( option->valueDomain, setContains(option->valueDomain, (Pointer)&result) ),
                       (cmdoptMsgValueNotInDomain, value, option->name)
                    );
                }
                return stdCOPY(&result);
            }
    
    case cmdoptOptionsFile : 
            return value;

    case cmdoptString      : 
    case cmdoptCIString    : 
            {
                if (option->valueDomain) {
                    String origValue = setElement(option->valueDomain, value);
                    
                    stdCHECK( origValue, (cmdoptMsgValueNotInDomain, value, option->name) ) {
                        value = origValue;
                    }
                }
                return value;
            }


    default : stdASSERT( False, ("Case label out of range") );
    }
    /*
     * Silence compiler.
     */
    return Nil;
}


static void addKeyword( stdMap_t map, cmdoptOption_t option, String value )
{
    String  keyword;
    Pointer keywordValue;

    Char *equals= strchr(value, '=');
    
    if (equals) {

       *equals= 0;
        keyword= stdCOPYSTRING(value); 
       *equals= '=';

        keywordValue = optiolwalue(option, equals+1);
        
    } else {
    
        if ( option->defaultValue
          && mapIsDefined(option->defaultValue,value)
          ) {
           keywordValue= mapApply(option->defaultValue,value);

        } else

        if (option->defaultWhenSpecified) {
           keywordValue = optiolwalue(option,option->defaultWhenSpecified);

        } else {
           stdCHECK( equals,(cmdoptMsgIsNoKeyword, value) );
           return;
        }
        
        keyword= stdCOPYSTRING(value);
    }
      
      
    stdCHECK( !mapIsDefined(map, keyword), (cmdoptMsgRedefinedKeyword, keyword) );
    
    stdCHECK( 
       stdIMPLIES( option->keywordDomain, setContains(option->keywordDomain, keyword) ),
       (cmdoptMsgKeywordNotInDomain, keyword, option->name)
    );
    
    mapDefine( map, keyword, keywordValue );       
}


static void addMapToList( stdMap_t value, cmdoptOption_t option ) 
{ 
    ListValue list= option->value;
    listXPutAfter(list->list, value);
}


static void addOptionToList( String value, cmdoptOption_t option ) 
{ 
    ListValue list= option->value;
    listXPutAfter(list->list, optiolwalue(option, value) );
    listAddTo((Pointer)(Address)option->position, &option->positionList);
}

    typedef struct {
        cmdoptOption_t option;
        stdMap_t       map;
    } MapRec;

    static void addToMap( String value, MapRec *rec ) 
    { 
        addKeyword(rec->map, rec->option, value );
    }

    static void addDefaultKeyword( String keyword, Pointer value, stdMap_t map )
    {
        if (!mapIsDefined(map, keyword)) {
            mapDefine(map,keyword,value);
        }
    }

static stdMap_t keywordMap( String value, cmdoptOption_t option )
{
    MapRec rec;
    rec.map    = mapNEW(String,8);
    rec.option = option;
    
    stdTokenizeString( value, ",", False, False, (stdEltFun)addToMap, &rec, 
                       False, ((option->flags&cmdoptKeepBraces) != 0));
    
    if (option->hasDefaultValue) {
        mapTraverse(option->defaultValue, (stdPairFun)addDefaultKeyword, rec.map);
    }
    
    return rec.map;
}


    static void addToArgvMap( String value, stdMap_t argvMap )
    {
        if (strchr(value, '"')) {
            // Quotes may have been used in option to control separation,
            // but now that it is a separate option it will have outer quotes
            // and inner quotes are not needed (and shell would have removed
            // these if given directly on command line).
            char *newV, *oldV;
            char prev = 0;
            for (oldV = value, newV = value; *oldV != 0; ++oldV) {
                if (*oldV == '"' && prev != '\\') {
                    ; // skip
                } else {
                    *newV = *oldV;
                    ++newV;
                }
                prev = *oldV;
            }
            *newV = 0;
        }
        mapDefine( argvMap, (Pointer)(Address)mapSize(argvMap), value );
    }

    static void addToArgv( Int index, String value, String* argv )
    {
        argv[index] = value;
    }

#define MAX_OPTION_FILE_TO_INCLUDE  15
    static void addFileValue( String optFileName, cmdoptDescr_t options )
    {
        stdMap_t argvMap;
        String   buf;
        Int      argc;
        String*  argv;

        /* check if option file included exceeds the MAX_OPTION_FILE_TO_INCLUDE */
        if (options->optFileIncluded >= MAX_OPTION_FILE_TO_INCLUDE) {
            msgReport(cmdoptMsgTooManyOptFileOpened, optFileName);
        }
        options->optFileIncluded++;
        /*
         * Read the options file into a string:
         */
        {
            FILE*       file;
            stdString_t contents = stringNEW();
            Char        buffer[1000];

            file= fopen(optFileName, "r");
            stdCHECK( file != Nil, (cmdoptMsgOpenOptFailed, optFileName) );
            
            while (fgets(buffer, sizeof(buffer), file)) {
                stringAddBuf (contents, buffer);
            }
            
            buf= stringStripToBuf(contents);
            
            fclose(file);
        }

        /*
         * Build map of options.
         */
        argvMap= mapNEW(Int, 10);
        stdTokenizeString( buf, " \t\r\n", False,
#ifdef STD_OS_win32
                           False, /* keep \ as needed in paths */
#else
                           True,
#endif
                           (stdEltFun)addToArgvMap, argvMap, False, False );

        /*
         * Build final argv for passing into cmdoptParse().
         */
        argc   = mapSize(argvMap) + 1;
        argv   = stdMALLOC((argc + 1) * sizeof(String*));
        argv[0]= argv[argc] = Nil;
        mapTraverse(argvMap, (stdPairFun)addToArgv, argv + 1);

        /*
         * Done! Parse it.
         */
        cmdoptParse( options, argc, argv);

        mapDelete(argvMap);
        stdFREE(buf);
        options->optFileIncluded--; 
    }

static void checkOptValueRedefinition(Pointer oldOptValue, cmdoptOption_t option)
{
  Bool IsCompatible = True;
  stdASSERT( option->mode == cmdoptSingleValue, ("unhandled option mode") );
  switch (option->type) {
  case cmdoptBool:
  case cmdoptHex:
  case cmdoptHex32:
  case cmdoptInt:
  case cmdoptFloat:     
      IsCompatible = oldOptValue == option->value;
      break;

  case cmdoptHex64:
  case cmdoptInt64:
      IsCompatible = (*(uInt64*)oldOptValue) == *((uInt64*)option->value);
      break;
    
  case cmdoptString: 
  case cmdoptCIString:
      IsCompatible = oldOptValue == option->value;  /* fast check */
      if (!IsCompatible) {
        IsCompatible = stdEQSTRING(oldOptValue, option->value);
      }
      break;

  default: 
      break;
  }

  stdCHECK( IsCompatible, (cmdoptMsgOverriddenIncompatible,option->name) );
}

static void addOptiolwalue( cmdoptDescr_t options, cmdoptOption_t option, String value, Bool addAsOne )
{               
    switch (option->mode) {
            
    case cmdoptNoValue      : 
          break;
    
    case cmdoptSingleValue  :           
          // note: by default, redefinitions are enabled, see cmdoptAddOption()
          if(!(option->flags & cmdoptRedefinition)) {
              // If option cannot have redefinition, issue error.
              stdCHECK( !option->hasValue, (cmdoptMsgRedefinedArgument,option->name) );
          } else {
              // If option is being redefined, check for incompatible redefinition.
              Bool doRedefinitionCheck = option->hasValue;
              Pointer oldOptValue = (option->hasValue) ? option->value : NULL;
              if (option->type == cmdoptOptionsFile ) {
                  addFileValue( optiolwalue(option, value), options );
                  option->value= optiolwalue(option, value);
              } else {
                  option->value= optiolwalue(option, value);
                  if (doRedefinitionCheck) {
                      checkOptValueRedefinition(oldOptValue, option);
                  }
              }
          }
          break;

    case cmdoptListValue    : 
          if (addAsOne) {
              addOptionToList( value, option );
          } else if (option->type == cmdoptOptionsFile ) {
              stdTokenizeString(value, ",", False, False, (stdEltFun)addFileValue, options, ((option->flags&cmdoptKeepQuote) != 0), ((option->flags&cmdoptKeepBraces) != 0));
          } else {
              stdTokenizeString(value, ",", False, (option->flags&cmdoptDoEscapes) != 0, (stdEltFun)addOptionToList, option, ((option->flags&cmdoptKeepQuote) != 0), ((option->flags&cmdoptKeepBraces) != 0));
          }
          break;
          
    case cmdoptKeywordValue :
          stdCHECK( !option->hasValue, (cmdoptMsgRedefinedArgument,option->name) );  
          option->value= keywordMap( value, option );
          break;

    case cmdoptListKeywordValue :
          addMapToList( keywordMap(value,option), option );
          break;

    default : stdASSERT( False, ("Case label out of range") );
   }
   
   option->hasValue= True;
}


static void addOptiolwalueX( cmdoptDescr_t options, cmdoptOption_t option, String value, Bool addAsOne )
{
    addOptiolwalue(options,option,value,addAsOne);
    
    if (option->flags & cmdoptNormalOption) {
        cmdoptOption_t normalOption = mapApply(options->options,cmdoptNormalArg);
        stdString_t    buffer       = stringNEW();
        
        stringAddFormat(buffer,"--%s",option->name);
        addOptiolwalue( options, normalOption, stringStripToBuf(buffer), True );
        addOptiolwalue( options, normalOption, value,                    True );
    }
}

/*
 * Function        : Parse specified command line according to the option descriptors
 *                   in the specified option object. Representation errors, domain errors
 *                   and  unknown options are flagged as (fatal) errors via module
 *                   stdMessages. Parsed options are aclwmulated in the option object,
 *                   to be retrieved via function cmdoptGetOptiolwalue.
 * Parameters      : options        (IO) option descriptor to parse in 
 *                   argv,argc      (I)  name of option
 * Function Result : -
 */
void cmdoptParse( cmdoptDescr_t options, Int argc, String argv[])
{
    Int     i;
    
    for (i=1; i<argc; i++) {
        Bool           shortForm;
        cmdoptOption_t option;

        String         lwrArg        = argv[i];
        String         optiolwalue   = Nil;
        
        if (lwrArg[0] == '-' && lwrArg[1]) {
            
            optiolwalue= strchr(lwrArg, '=');

            if (optiolwalue != Nil) {
               *optiolwalue= 0;
              ++optiolwalue;
            }
                
            if (lwrArg[1] == '-') {
                shortForm = False;
                option    = mapApply(options->options,lwrArg+2);
            } else {
                shortForm = True;
                option    = mapApply(options->shorts,lwrArg+1);
                
                if (!option) {
                    // map things like -DFOO to -D FOO
                    Char singleLetter[2];
                    singleLetter[0]= lwrArg[1];
                    singleLetter[1]=         0;
                    
                    option= mapApply(options->shorts, singleLetter);
                    if (option && (option->flags&(cmdoptSingleLetter|cmdoptESingleLetter))) {
                        if (optiolwalue != Nil) { *(--optiolwalue)= '=';}
                        optiolwalue= lwrArg+2;
                    } else {
                        option= Nil;
                    }
                }
            }
            
            if (option && (option->flags & cmdoptDisabled) ) {
                option= Nil;
            }

            if (option && (option->flags & cmdoptDeprecated) ) {
                stdCHECK( False, (cmdoptMsgDeprecated, option->name) );
            }

            if (option == Nil && options->saveUnknownOpts) {
                option = mapApply(options->options, cmdoptUnknownOptionArg);
                option->position = options->lwrOptPos++;
                if (optiolwalue != Nil) { *(--optiolwalue)= '='; optiolwalue= Nil; }
                addOptiolwalue( options, option, lwrArg, True );

            } else {
                stdCHECK( option, (cmdoptMsgUnknownOption,lwrArg+1) ) {
                    Bool addAsOne = (option->flags & cmdoptAddAsOne)!=0;
                    option->position = options->lwrOptPos++;
                
                    if (option->mode == cmdoptNoValue) {
                        stdCHECK( optiolwalue == Nil, (cmdoptMsgNoArgumentExpected, lwrArg) );
                        addOptiolwalueX(options, option, optiolwalue, addAsOne);  
                    } else {
                        if (optiolwalue != Nil && *optiolwalue == 0) {
                            optiolwalue = Nil; // no value given, e.g. -opt=
                        }
                        if (shortForm && option->flags&cmdoptESingleLetter) {
                            if (optiolwalue == Nil) {
                                option->hasValue= True;
                            } else {
                                addOptiolwalueX(options, option, optiolwalue, addAsOne);  
                            }
                        } else {
                            if (optiolwalue == Nil) {
                                if ( (i+1)<argc ) {
                                    optiolwalue= argv[i+1];

                                    if ( optiolwalue[0]=='-'
                                      && optiolwalue[1]
                                      && option->defaultWhenSpecified
                                       ) { 
                                        optiolwalue= Nil;
                                    } else { 
                                        i++;
                                    }
                                }

                                if (optiolwalue == Nil) {
                                    optiolwalue= option->defaultWhenSpecified;
                                }
                            }

                            stdCHECK( optiolwalue, (cmdoptMsgArgumentExpected, lwrArg) ) {
                                addOptiolwalueX(options, option, optiolwalue, addAsOne);  
                            }
                        }
                    }
                }
            }
        } else {
            option= mapApply(options->options,cmdoptNormalArg);
            option->position = options->lwrOptPos++;
            addOptiolwalue( options, option, lwrArg, True );
        }
    }
}



/*-------------------------- Command Option Printing -------------------------*/

#define TABSIZE  8

typedef struct {
    Bool            printHidden;
    Bool            printDefault; 
    Char            prefix;
    cmdoptOption_t  option;
    uInt            printPos;
    uInt            lmargin;
    uInt            rmargin;
    uInt            ldescriptionMargin;
    Bool            countTabs;
    uInt            tab1,tab2;
    String          format1, format2;
    Bool            doLmarginCheck;
} OptionPrintRec;


static Char *newLine( OptionPrintRec *rec, Char *p )
{
    Int i;

    printf("\n");
    i= rec->printPos= rec->lmargin;
    
    while (i-- > 0) { printf(" "); }
    
    return p;
}

        
static Char *printChar( OptionPrintRec *rec, Char *p )
{
    if (*p=='\t') {
        uInt lwrpos= rec->printPos;
        uInt newpos= stdROUNDUP( lwrpos + 1, TABSIZE );
        
        rec->printPos= newpos;
        
        while (lwrpos++ < newpos) { printf("%c",' '); }
        
        p++;
        
    } else 
           
    if (*p=='\n') {
        p= newLine(rec,p);
           
    } else {
        rec->printPos++;
        printf("%c",*p);
        p++;
    }
    
    return p;
}


static void oprintf( OptionPrintRec *rec, String format, ... )
{
    stdString_t buffer= stringNEW();
    
    Char   *p;
    String  handle;
    
    va_list ap;
    va_start (ap, format);
    stringAddVFormat(buffer, format, ap);
    va_end (ap);
    
    handle= p= stringStripToBuf(buffer);
    
    while (*p) {
        while (*p && (rec->printPos < rec->rmargin - strcspn(p, " \n")) ) {
            if (rec->doLmarginCheck && (rec->printPos == rec->lmargin)
                && (*p == ' ' || *p == '\n')
               ) {
                p++;
            } else {
                p= printChar(rec, p);
            }
        }
        if (*p) {
            while (*p && *p != ',' && !stdIsSpace(*p,True)) {
                p= printChar(rec, p) ;
            }
            if (*p == ',') { 
                p= printChar(rec, p); 
                p= newLine  (rec, p); 
            } else 
            if (*p) { 
                p= newLine  (rec, p); 
            }
        }
    }
    
    stdFREE(handle);
}


static Bool endsInChar( String line, Char terminator )
{
    if (*line == 0) {
       return False;
    } else {
        Char *p= &line[ strlen(line)-1 ];
    
        if (stdIsSpace(*p,True)) {
            return True;
        } else {        
            return *p == terminator;
        }
    }
}


static void printValue( Pointer value, OptionPrintRec *rec )
{
    switch (rec->option->type) {
    case cmdoptBool        : oprintf(rec, "%s",                        value ? "true" : "false" ); break;
    case cmdoptHex         : oprintf(rec, "0x%x",                      value ); break;
    case cmdoptHex32       : oprintf(rec, "0x%08x",                    value ); break;
    case cmdoptHex64       : oprintf(rec, "0x%" stdFMT_LLX, *(uInt64 *)value ); break;
    case cmdoptInt         : oprintf(rec, "%ld",                       value ); break;
    case cmdoptInt64       : oprintf(rec, "0x%" stdFMT_LLD, *(uInt64 *)value ); break;
    case cmdoptFloat       : oprintf(rec, "%e",                  citof(value )); break;
    case cmdoptString      : oprintf(rec, "'%s'",                      value ); break;
    case cmdoptCIString    : oprintf(rec, "'%s'",                      value ); break;
    case cmdoptOptionsFile : oprintf(rec, "'%s'",                      value ); break;

    default                : stdASSERT( False, ("Case label out of range") );
    }       
}

static void printKeyword( String keyword, Pointer value, OptionPrintRec *rec )
{
     oprintf(rec, "%c",  rec->prefix); rec->prefix= ',';
     oprintf(rec, "%s=", keyword);
     printValue(value,rec);
}

static void printKeywordValue(  Pointer value, OptionPrintRec *rec )
{
    rec->prefix= ' ';
    mapTraverse( value, (stdPairFun)printKeyword, rec );
}

static void printV( Pointer value, OptionPrintRec *rec )
{
     oprintf(rec, "%c",  rec->prefix); rec->prefix= ',';
     printValue(value,rec);
}

static void printOptiolwalue( cmdoptOption_t option, Bool printDefault, OptionPrintRec *rec )
{
    Pointer value = printDefault ? option->defaultValue : option->value;

    switch (option->mode) {
    case cmdoptSingleValue      : oprintf(rec, " ");
                                  printValue( value, rec );
                                  break;
                 
    case cmdoptListValue        : if ( value != Nil ) {
                                      listTraverse( 
                                         ((ListValue)value)->list, 
                                         (stdEltFun)printValue, 
                                         rec
                                      );
                                  }
                                  break;
                 
    case cmdoptKeywordValue     : if ( value != Nil ) {
                                      printKeywordValue( value, rec );
                                  }
                                  break;
                 
    case cmdoptListKeywordValue : if ( value != Nil ) {
                                      if (printDefault) {
                                          printKeywordValue( value, rec );
                                      } else {
                                          listTraverse( 
                                             ((ListValue)value)->list, 
                                             (stdEltFun)printKeywordValue, 
                                             rec
                                          );
                                      }
                                  }
                                  break;
                 
    case cmdoptNoValue          : oprintf(rec, " true");
                                  break;

    default                     : stdASSERT( False, ("Case label out of range") );
    }
}

    static void printText( OptionPrintRec *rec, String text, Char terminator )
    {
        oprintf(rec, "%c", stdToUpperCase( text[0]) );
        oprintf(rec, "%s", text+1);
        
        if (!endsInChar(text,terminator)) {
            oprintf(rec, "%c", terminator);
        }
    }
    
static void printOption( cmdoptOption_t option, OptionPrintRec *rec )
{
    Char   buffer[10000];
    Bool   emptyVf  = stdISEMPTYSTRING(option->valueFormat);
    Bool   listMode = (option->mode == cmdoptListValue)
                   || (option->mode == cmdoptListKeywordValue);
    String dotdot   = (listMode && !emptyVf)?",...":"";

    if ( (option->flags & (cmdoptHidden|cmdoptDeprecated)) && !rec->printHidden ) { return; }

    sprintf(buffer, "--%s%s%s%s", option->name, emptyVf?"":" ", option->valueFormat, dotdot);

    rec->option   = option;
    rec->printPos =      0;
    
    if ( rec->printDefault ) {
        if ( option->hasValue ) {
            if (rec->countTabs) { 
                rec->tab1= stdMAX(rec->tab1,strlen(buffer)); 
                return;
            } else {
                oprintf(rec, rec->format1, buffer);
            }

            printOptiolwalue(option, False, rec);
            printf("\n");
        }
        
    } else
    if ( !stdEQSTRING(option->name, cmdoptNormalArg) ) {
        if (rec->countTabs) { 
            rec->tab1= stdMAX(rec->tab1,strlen(buffer)); 
        } else {
            /* disable doLMarginCheck temporarily. The option name
               is supposed to be printed left-justified with trailing
               spaces, so that the following 'short name' for each
               option starts at the same column number. 
               But the lmargin check in oprintf() skips over  
               some of the trailing spaces (only if the option name 
               length is less than lmargin (e.g. '--ptx' with lmargin = 8, 
               and format = "%-43s"). This then makes the 'short name'
               print at different starting column numbers for such
               options, compared to options with larger names
               ("--default-stream"). */
            Bool doLmarginCheck = rec->doLmarginCheck;
            rec->doLmarginCheck = False;
            oprintf(rec, rec->format1, buffer);
            rec->doLmarginCheck = doLmarginCheck;
        }
            
        if ( !stdISEMPTYSTRING( option->shortName ) ) {
            sprintf(buffer, "(-%s)" , option->shortName);
        }

        if (rec->countTabs) { 
            rec->tab2= stdMAX(rec->tab2,strlen(buffer)); 
            return;
        } else {
            oprintf(rec, rec->format2, buffer);
        }
           
        printText(rec,option->description,'.'); 
                        
        if ( option->keywordDomain ) {
                stdList_t  keywords= setToList(option->keywordDomain);
                cmdoptType savedType= cmdoptString;
                stdSWAP(option->type, savedType, cmdoptType);
                oprintf(rec,  "\nAllowed keywords for this option: " );
                rec->prefix=' ';
                listSort( &keywords, (stdLessEqFun)stdStringLessEq);
                listTraverse( keywords, (stdEltFun)printV, rec );
                oprintf(rec,  "." );
                stdSWAP(option->type, savedType, cmdoptType);
                listDelete(keywords);
        }
        if ( option->valueDomain ) {
                stdList_t  values= setToList(option->valueDomain);
                oprintf(rec,  "\nAllowed values for this option: " );
                rec->prefix=' ';

                switch (rec->option->type) {
                case cmdoptBool        : 
                case cmdoptHex         : 
                case cmdoptHex32       : 
                case cmdoptInt         : listSort( &values, (stdLessEqFun)stdIntLessEq); break;
                
                case cmdoptInt64       : listSort( &values, (stdLessEqFun)stdpInt64LessEq); break;
                case cmdoptHex64       : listSort( &values, (stdLessEqFun)stdpuInt64LessEq); break;
                
                case cmdoptFloat       : listSort( &values, (stdLessEqFun)stdFloatLessEq); break;
                case cmdoptString      : listSort( &values, (stdLessEqFun)stdStringLessEq); break;
                case cmdoptCIString    : listSort( &values, (stdLessEqFun)stdCIStringLessEq); break;
                
                case cmdoptOptionsFile : break;

                default                : stdASSERT( False, ("Case label out of range") );
                }       

                listTraverse( values, (stdEltFun)printV, rec );
                oprintf(rec,  "." );
                listDelete(values);
        }
        if ( option->hasDefaultValue && !(option->flags & cmdoptDontPrintDefault) ) {
            oprintf(rec,  "\nDefault value: " );
            printOptiolwalue(option, True, rec);
            oprintf(rec,  "." );
        }
        printf("\n\n");
    }
}


static Bool optionLessEq( cmdoptOption_t o1, cmdoptOption_t o2 )
{
    return stdLEQSTRING( o1->name, o2->name );
}

static void countHidden( cmdoptOption_t option, uInt *hidden )
{
    if (option->flags & (cmdoptHidden|cmdoptDeprecated)) { (*hidden)++; }
}

static void printOptionGroup( OptionGroup group, OptionPrintRec *rec )
{
    uInt nrHidden  = 0;
    uInt nrOptions = listSize(group->options);

    listTraverse(group->options, (stdEltFun)countHidden, &nrHidden );  
    
    if ( (nrHidden==nrOptions) && !rec->printHidden ) { return; }


    if (!rec->countTabs) { 
        if (group->flags & cmdoptSortGroup) {
            listSort( &group->options, (stdLessEqFun)optionLessEq);
        }
        
        if (group->description && group->options) {
            Int   len,i;
            Char *dot= strchr(group->description,'.');
            if (dot != NULL && *(dot+1) == '\0') {
              /* if dot is last char of description,
               * then is terminator rather than separator. */
              dot = NULL;
            }
            
            if (dot) { len=  dot - group->description;  }
                else { len= strlen(group->description); }
                
            printf("\n");
            for (i=0; i<len; i++) { printf("%c", group->description[i]); }
            printf("\n");
            for (i=0; i<len; i++) { printf("="); }
            printf("\n");
            
            if (dot) {
                rec->option   = Nil;
                rec->printPos = 0;
                rec->lmargin  = 0;

                dot++;
                printText(rec,dot,'.'); 
                printf("\n");
            }
            printf("\n");
        }
    }
    
    rec->lmargin = rec->ldescriptionMargin;
    
    listTraverse(group->options, (stdEltFun)printOption, rec );  
}



/*
 * Function        : Print the options' command option groups in order of declaration.
 *                   For each group, a textual specification of the option list is printed,
 *                   sorted by long option name. The group is preceded by its description.
 * Parameters      : options    (I)  Options descriptor to print.
 *                   hidden     (I)  Print hidden options iff. this value is True.
 *                   parsed     (I)  Print parsed data (i.e. options which are set).
 * Function Result : 
 */
void cmdoptPrint( cmdoptDescr_t options, Bool hidden, Bool parsed )
{
    OptionPrintRec   rec;
    Char             format1[1000],format2[1000];    

    rec.printDefault= parsed;
    rec.printHidden = hidden;
    rec.doLmarginCheck = True;

    rec.countTabs   = True; 
    rec.tab1        = 0;
    rec.tab2        = 0; 

   /* determine print sizes */
    listTraverse(options->optionGroups, (stdEltFun)printOptionGroup, &rec );  
    
    rec.tab1 += 2;
    rec.tab2 += 2;
    sprintf(format1,"%%-%ds",   (int)rec.tab1);
    sprintf(format2,"%%-%ds\n", (int)rec.tab2);

    rec.format1            = format1;
    rec.format2            = format2;
    rec.countTabs          = False; 
    
#if 1
    rec.rmargin            = 80;
    rec.ldescriptionMargin = 8;
#else
    rec.rmargin            = rec.tab1 + rec.tab2 + 60;
    rec.ldescriptionMargin = rec.tab1 + rec.tab2;
#endif

   /* print them */
    listTraverse(options->optionGroups, (stdEltFun)printOptionGroup, &rec );  
}
