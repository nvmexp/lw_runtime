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
 *  Module name              : stdCmdOpt.h
 *
 *  Description              :
 *     
 *         This is a command line option parser library. It
 *         unravels a usual (argc,argv) pair according to a
 *         previously constructed 'command option specification object',
 *         and performs some generic checks. 
 *         The results of command line parsing are aclwmulated into
 *         the specification object and can be obtained using a retrieval
 *         functions.
 *
 *         The following concepts are used:
 *
 *         - command option           = Any argv element starting with a hyphen,
 *         - command option name        not to be considered as an argument of a
 *         - command option argument    previous option.
 *                                      For instance,  in   "make -f -g", the
 *                                      string "-g" is not an option, but an argument
 *                                      to option -f.
 *                                      The command option name is the representation
 *                                      of the option minux the leading hyphen.
 *         - command option type      = The type of an option argument; supported
 *                                      types are Bool, String, Int, Float, Hex and Hex32.
 *                                      The last three numeric types differ in how they
 *                                      are printed. A special type supported by this 
 *                                      library is the OptionsFile. Command options of this
 *                                      type are interpreted as names of files from which
 *                                      additional command line options should be included.
 *         - command option value     = The value of a command line option argument
 *         - command option keyword   = Sometimes, command line options build
 *                                      associations, as in 
 *                                             "ld -region start=0x0,end=0x100"
 *                                      The structure of the above command is:
 *                                      as follows"
 *                                             <command> <option> <keyword>=<value>,..                             
 *                     
 *         - command option mode      = The format of the command line option, plus
 *                                      its optional parameters:
 *                                        NoValue          : format : <option>
 *                                                           as in  : -help
 *                                        SingleValue      : format : <option> <value>
 *                                                           as in  : -f file
 *                                                           The value should have a 
 *                                                           representation according to the
 *                                                           type specified for the option.
 *                                                           Only one oclwrrence of the option
 *                                                           is allowed in one single command.
 *                                        ListValue        : format : <option> <value>[,..]
 *                                                           as in  : -undefined sym1,sym2,sym3
 *                                                           Options may occur multiple times,
 *                                                           as in  : -undefined sym1 -undefined sym2
 *                                                           The value should have a 
 *                                                           representation according to the
 *                                                           type specified for the option.
 *                                        KeywordValue     : format : <option> <keyword>=<value>[,..]
 *                                                           as in  : -region start=0x0,end=0x100
 *                                                           The value should have a 
 *                                                           representation according to the
 *                                                           type specified for the option.
 *                                        ListKeywordValue : A list of keyword values.
 *                     
 *         - command option flags     = Attributes that affect the behaviour of the
 *                                      command line option:
 *                                       RequiredOption  : The option must have a value after parsing
 *                                       DontPrintDefault: Do not print default value (if any) in
 *                                                         cmdoptPrint
 *                                       Hidden          : Do not print the option in cmdoptPrint
 *         - value domain             = Only for option type String: the allowed strings
 *                                      allowed for the value
 *         - keyword domain           = Only for option mode KeywordValue: the allowed strings
 *                                      allowed for the keywords
 */

#ifndef stdCmdOpt_INCLUDED
#define stdCmdOpt_INCLUDED

/*-------------------------------- Includes ----------------------------------*/

#include "stdTypes.h"
#include "stdList.h"

#ifdef __cplusplus
extern "C" {
#endif

/*---------------------------------- Types -----------------------------------*/

typedef enum {
   cmdoptOptionsFile,
   cmdoptBool,
   cmdoptString,
   cmdoptCIString,
   cmdoptInt,
   cmdoptInt64,
   cmdoptFloat,
   cmdoptHex,
   cmdoptHex32,
   cmdoptHex64,
} cmdoptType;

   
typedef enum {
   cmdoptNoValue,
   cmdoptSingleValue,
   cmdoptListValue,
   cmdoptKeywordValue,
   cmdoptListKeywordValue
} cmdoptMode;

typedef enum {
   cmdoptNoFlags           =  0,
   cmdoptRequiredOption    =  (1 <<  0),
   cmdoptDontPrintDefault  =  (1 <<  1),
   cmdoptHidden            =  (1 <<  2),
   cmdoptDisabled          =  (1 <<  3),
   cmdoptSingleLetter      =  (1 <<  4),
   cmdoptESingleLetter     =  (1 <<  5),
   cmdoptDoEscapes         =  (1 <<  6),
   cmdoptAddAsOne          =  (1 <<  7),
   cmdoptRedefinition      =  (1 <<  8),
   cmdoptDeprecated        =  (1 <<  9),
   cmdoptNormalOption      =  (1 << 10),
   cmdoptUnknownOption     =  (1 << 11),
   cmdoptKeepQuote         =  (1 << 12),  /* don't remove '\'', '\"' in the string */
   cmdoptKeepBraces        =  (1 << 13)   /* don't remove '\[', '\]' in the string */
} cmdoptFlags;


typedef enum {
   cmdoptNoGroup           = 0,
   cmdoptSortGroup         =  (1 << 0)
} cmdoptGroupFlags;

typedef struct cmdoptDescr *cmdoptDescr_t;

/*
 * The following is the name of the pseudo option
 * under which all non-options are grouped.
 * It has type String, Mode list.
 */
#define cmdoptNormalArg    " "
#define cmdoptUnknownOptionArg "__internal_unknown_opt"

/*-------------------------------- Functions ---------------------------------*/

/*
 * Function        : Create a new command line option object.
 *                   Option descriptors have to be added to it
 *                   using repeated use of function cmdoptAddOption,
 *                   after which it can be used for parsing a command line.
 *                   The option is created with a default command option group
 *                   (see also function cmdoptAddGroup).
 * Parameters      : saveUnknownOpts      (I)  Save the unknown options encountered during parsing
 * Function Result : Empty object.
 */
cmdoptDescr_t cmdoptCreate(Bool saveUnknownOpts);


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
                      cString           description );


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
                     cString           description );


/*
 * Function        : Parse specified command line according to the option descriptors
 *                   in the specified option object. Representation errors, domain errors
 *                   and  unknown options are flagged as (fatal) errors via module
 *                   stdMessages. Parsed options are aclwmulated in the option object,
 *                   to be retrieved via function cmdoptGetOptiolwalue.
 * Parameters      : options        (IO) Option descriptor to parse in .
 *                   argc           (I)  Option count.
 *                   argv           (I)  Name of options.
 * Function Result : 
 */
void cmdoptParse( cmdoptDescr_t options, Int argc, String argv[]);


/*
 * Function        : Retrieve value of named option. The result is delivered via
 *                   parameter 'value', in a format that is depending on the
 *                   mode and type of the option.
 *                   This value is only defined when the function returns
 *                   'True', indicating that the option was specified in the
 *                   parsed command line:
 *                       mode NoValue:        Undefined (oclwrrence of option 
 *                                                          via function result).
 *                       mode SingleValue:    Int, Bool or String, 
 *                                            depending on option type.
 *                       mode ListValue:      The stdList_t of Int, Bool or String, 
 *                                            depending on option type.
 *                                            The order of oclwrrence of of the
 *                                            values in the list is identical to 
 *                                            the order on which they oclwrred on
 *                                            the command line.
 *                       mode KeywordValue:   The stdMap_t from keyword String 
 *                                            to Int, Bool or String, 
 *                                            depending on option type.
 *                       mode FileValue:      Undefined, result is False.
 *                                            FileValue's cannot be `Get'.
 *
 * Parameters      : options  (I)  Options descriptor to inspect.
 *                   name     (I)  Name of option.
 *                   value    (O)  Retrieved option value, as described above.
 * Function Result : True iff. an option value is returned (either parsed from
 *                   the command line, or a default value.
 *
 * NB              : The regular command line arguments (the non-options)
 *                   can be retrieved using cmdoptNormalArg.
 */
Bool _cmdoptGetOptiolwalue( cmdoptDescr_t   options, 
                            cString         name, 
                            Pointer         value,
                            uInt            valueSize); 
#define cmdoptGetOptiolwalue(options,name,value) \
       _cmdoptGetOptiolwalue(options,name,(Pointer)(&(value)),sizeof (value))

#define cmdoptGetOptiolwalueNdf(options,name,value) \
       if (cmdoptOnCommandLine(options,name)) \
           { _cmdoptGetOptiolwalue(options,name,(Pointer)(&(value)),sizeof (value)); }

/*
 * Function        : Return command line position of named option.
 *                   Return 0 if option was not used.
 *                   If option is a list of values, then this function returns
 *                   the position of the last usage, and GetOptionPositionList
 *                   can be used to get positions of individual options in list.
 * Parameters      : options  (I)  Options descriptor to inspect.
 *                   name     (I)  Name of option.
 * Function Result : Position in command line of option.
 */
uInt cmdoptGetOptionPosition( cmdoptDescr_t   options, 
                              cString         name);

/* Return list of positions, where position corresponds to list of values.
 * Should only be used with cmdoptListValue mode */
stdList_t cmdoptGetOptionPositionList( cmdoptDescr_t   options, 
                                       cString         name);

/*
 * Function        : Test if option was specified on command line.
 * Parameters      : options  (I)  Options descriptor to inspect.
 *                   name     (I)  Name of option.
 * Function Result : True iff. an option value was specified on the
 *                   parsed command line.
 */
Bool cmdoptOnCommandLine( cmdoptDescr_t   options, 
                          cString         name );


/*
 * Function        : Set additional flags for command line option.
 * Parameters      : options  (I)  Options descriptor to modify.
 *                   name     (I)  Name of option.
 *                   flags    (I)  Flags to set.
 * Function Result : Previous option flags.
 */
uInt32 cmdoptSetFlags( cmdoptDescr_t   options, 
                       cString         name,
                       uInt32          flags );


/*
 * Function        : Print the options' command option groups in order of declaration.
 *                   For each group, a textual specification of the option list is printed,
 *                   sorted by long option name. The group is preceded by its description.
 * Parameters      : options    (I)  Options descriptor to print.
 *                   hidden     (I)  Print hidden options iff. this value is True.
 *                   parsed     (I)  Print parsed data (i.e. options which are set).
 * Function Result : 
 */
void cmdoptPrint( cmdoptDescr_t options, Bool hidden, Bool parsed );


#ifdef __cplusplus
}
#endif

#endif /* stdCmdOpt_INCLUDED */
