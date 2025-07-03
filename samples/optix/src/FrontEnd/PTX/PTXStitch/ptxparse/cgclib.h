/****************************************************************************\
Copyright (c) 2017, LWPU CORPORATION.

LWPU Corporation("LWPU") supplies this software to you in
consideration of your agreement to the following terms, and your use,
installation, modification or redistribution of this LWPU software
constitutes acceptance of these terms.  If you do not agree with these
terms, please do not use, install, modify or redistribute this LWPU
software.

In consideration of your agreement to abide by the following terms, and
subject to these terms, LWPU grants you a personal, non-exclusive
license, under LWPU's copyrights in this original LWPU software (the
"LWPU Software"), to use, reproduce, modify and redistribute the
LWPU Software, with or without modifications, in source and/or binary
forms; provided that if you redistribute the LWPU Software, you must
retain the copyright notice of LWPU, this notice and the following
text and disclaimers in all such redistributions of the LWPU Software.
Neither the name, trademarks, service marks nor logos of LWPU
Corporation may be used to endorse or promote products derived from the
LWPU Software without specific prior written permission from LWPU.
Except as expressly stated in this notice, no other rights or licenses
express or implied, are granted by LWPU herein, including but not
limited to any patent rights that may be infringed by your derivative
works or by other works in which the LWPU Software may be
incorporated. No hardware is licensed hereunder. 

THE LWPU SOFTWARE IS BEING PROVIDED ON AN "AS IS" BASIS, WITHOUT
WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING WITHOUT LIMITATION, WARRANTIES OR CONDITIONS OF TITLE,
NON-INFRINGEMENT, MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, OR
ITS USE AND OPERATION EITHER ALONE OR IN COMBINATION WITH OTHER
PRODUCTS.

IN NO EVENT SHALL LWPU BE LIABLE FOR ANY SPECIAL, INDIRECT,
INCIDENTAL, EXEMPLARY, CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
TO, LOST PROFITS; PROLWREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF
USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) OR ARISING IN ANY WAY
OUT OF THE USE, REPRODUCTION, MODIFICATION AND/OR DISTRIBUTION OF THE
LWPU SOFTWARE, HOWEVER CAUSED AND WHETHER UNDER THEORY OF CONTRACT,
TORT (INCLUDING NEGLIGENCE), STRICT LIABILITY OR OTHERWISE, EVEN IF
LWPU HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
\****************************************************************************/

/*
** cgclib.h
*/

#if !defined(__CGCLIB_H)
#define __CGCLIB_H 1

/****************************************************************************\
  cgclib interface -- using the compiler as a library

  The interface to the cgc compiler consists of 4 functions -- one startup
  initialization function, two `compile' functions and one final cleanup
  function.  The intent is that you call the init function to create a
  compilation context, then call the compile routines to compile programs,
  and finally call the ceanup function to clean everything up.  If you
  want to compile things in multiple threads, you should be able to call
  the init function multiple times (to create multiple contexts) and have
  each thread use its own context.

  FIXME -- lwrrently trying to compile more than one program in a context
  FIXME -- will fail.  So if you want to compile multiple programs, you need
  FIXME -- to create a new context for each one and destroy it afterwards
  FIXME -- In addition, trying to compile things simultaneously in multiple
  FIXME -- threads is likely to fail.
\****************************************************************************/

#if !defined(DONT_INCLUDE_COP)
#if !defined(DONT_INCLUDE_CGFX_DAG)
#include "cgfx_dag.h" /* must be before copi_inglobals.h */
#endif /* DONT_INCLUDE_CGFX_DAG */
#include "copi_inglobals.h"
typedef DagType DagType_t;
#else /* DONT_INCLUDE_COP */
typedef struct IMemPool_rec IMemPool;
typedef struct IAtomTable_Rec IAtomTable;
typedef struct Binding_Rec Binding;
typedef struct BindingList_Rec BindingList;
typedef struct BindingType_Rec BindingType;
typedef struct BindingTypeList_Rec BindingTypeList;
typedef struct BindingTypeMethod_Rec BindingTypeMethod;
typedef struct BindingTypeField_Rec BindingTypeField;
typedef struct IBasicBlock_rec IBasicBlock;
typedef struct IDag_rec IDag;
typedef int DagType_t;
#endif /* DONT_INCLUDE_COP */

#if defined(EXPORTSYMBOLS)
#if defined(WIN32)
#define DLLEXPORT __declspec(dllexport)
#elif defined(__GNUC__) && __GNUC__>=4
#define DLLEXPORT __attribute__ ((visibility("default")))
#elif defined(__SUNPRO_C) || defined(__SUNPRO_CC)
#define DLLEXPORT __global
#else
#define DLLEXPORT
#endif
#else
#define DLLEXPORT
#endif

#ifdef __cplusplus
extern "C" {
#endif /* c++ */

typedef struct CgStruct_Rec CgStruct;

DLLEXPORT const char *Cg_CompilerVersion(void);
/* returns a constant version identification string */

DLLEXPORT CgStruct *Cg_CompilerInit(int ac, const char * const *av, IMemPool *pool, char **listing);
/* main compiler initialization routine.  The arguments correspond to
 * the args of the cgc command line tool, and can be used to set
 * various options including the default profile and entry point.
 * 'listing' is an output set to a string containing warning/error
 * messages generated from parsing the supplied Cg code.  This string
 * (if it is non-NULL), is allocated using the memory pool passed to
 * Cg_CompilerInit, so must be freed (if required) by the caller.
 *
 * The pool passed to Cg_CompilerInit is used to allocate space for all
 * non-internal data structures and strings that are returned by the
 * compiler code.  Some functions below take an additional IMemPool
 * parameter which can be used to override the pool passed to CompilerInit --
 * it will be used for the data returned by those functions or if NULL
 * the one passed to CompilerInit will be used.
 */

DLLEXPORT void Cg_AllowTrapOnError(CgStruct *Cg);
/* Request that no signal handlers be registered to catch seg-faults, etc.
 */

DLLEXPORT IAtomTable *Cg_GetAtomTable(CgStruct *Cg);
/* provides access to the Cg context's atom table, used to look up
 * atoms present in e.g. Bindings returned by Cg_ExtractBindings.
 */

DLLEXPORT void Cg_SetOutputPool(CgStruct *Cg, IMemPool *pool);
/* change the pool used to allocate memory for objects and strings returned
 * by compiler functions.  If pool is NULL, objects will be allocated from
 * an internal pool freed when Cg_CompilerCleanup is called.  This call
 * overrides any pool set by a previous call to Cg_SetBindOutputPool; if
 * you want a seperate pool for bindings, you must call Cg_SetBindOutputPool
 * AFTER calling Cg_SetOutputPool
 */

DLLEXPORT void Cg_SetBindOutputPool(CgStruct *Cg, IMemPool *pool);
/* provide a seperate pool to be used for all Binding and BindingType
 * objects returned by calls to the compiler with this CgStruct.  If
 * ths is not called, or is called with NULL as the pool arg, all
 * such structs will be allocated by the default output pool (originally
 * passed to Cg_CompilerInit or Cg_SetOutputPool.
 */

DLLEXPORT IMemPool *Cg_GetOutputPool(CgStruct *Cg);
/* get the pool used to allocate memory for objects and strings returned
 * by compiler functions.
 */

DLLEXPORT IMemPool *Cg_GetBindOutputPool(CgStruct *Cg);
/* get the pool to used for all Binding and BindingType objects returned
 * by calls to the compiler with this CgStruct.
 */

DLLEXPORT int Cg_ParseFile(CgStruct *Cg, const char *filename, const char * const *opts, char **listing);
DLLEXPORT int Cg_ParseString(CgStruct *Cg, const char *input, int length, const char * const *opts, char **listing);
/* compiler entry points to parse Cg code and add it to the Cg
 * context.  This adds functions, types and variables to the Cg
 * context that may subsequently be used to generate programs.
 * Cg_ParseString reads Cg code from a string, Cg_ParseFile from a
 * file named by 'filename'.  'opts' is a NULL-terminated list of
 * options that may be used to override the options specified in
 * Cg_CompilerInit.  'listing' is an output set to a string containing
 * warning/error messages generated from parsing the supplied Cg code.
 * This string (if it is non-NULL), is allocated using the memory pool
 * passed to Cg_CompilerInit, so must be freed (if required) by the
 * caller.  The value returned is 0 upon successful parsing, non-zero
 * otherwise.
 */

DLLEXPORT char *Cg_PreprocFile(CgStruct *Cg, const char *filename, const char * const *opts, char **listing);
DLLEXPORT char *Cg_PreprocString(CgStruct *Cg, const char *input, int length, const char * const *opts, char **listing);
/* compiler entry points to run just the preprocessor on an input and return
 * the resulting string
 */

DLLEXPORT int Cg_ParseFileWithProfile(CgStruct *Cg, const char *profile, const char *filename, const char * const *opts, char **listing);
DLLEXPORT int Cg_ParseStringWithProfile(CgStruct *Cg, const char *profile, const char *input, int length, const char * const *opts, char **listing);
DLLEXPORT char *Cg_PreprocFileWithProfile(CgStruct *Cg, const char *profile, const char *filename, const char * const *opts, char **listing);
DLLEXPORT char *Cg_PreprocStringWithProfile(CgStruct *Cg, const char *profile, const char *input, int length, const char * const *opts, char **listing);
/* compiler entry points to parse/preprocess file/string inputs in a profile dependant manner.
 */

typedef int (*Cg_IncludeCallback)(CgStruct *Cg, const char *name, int system, void *arg);
/* A callback function to be called to handle #include directives:
 *      name    the name parsed after the #include
 *      system  true for #include <>, false for #include ""
 *      arg     extra user-supplied arg
 *  returns
 *      true    if the include is dealt with (nothing more to do)
 *      false   fall back on the normal #include processing (look for a file)
 */

DLLEXPORT Cg_IncludeCallback Cg_SetIncludeFileCallback(CgStruct *Cg, Cg_IncludeCallback fn, void *arg, void **old_arg);
/* sets the include file callback.  Returns the old callback */

DLLEXPORT int Cg_PushInputFile(CgStruct *Cg, const char *file);
DLLEXPORT int Cg_PushInputString(CgStruct *Cg, const char *input, int length, const char *name);
/* push input sources to be read.  Called from the include file callback,
 * these set up sources of input to be read by the parser.  After an input
 * source is fully parsed, it will continue parsing the previous source */

DLLEXPORT const char *Cg_LwrrentInput(CgStruct *Cg, int *line, void **iter);
DLLEXPORT const char *Cg_NextInput(CgStruct *Cg, int *line, void **iter);
/* returns the current input sources (filename, and optional linnumber if
 * 'line' is non-null.  Only valid from the include file callback.
 * Cg_LwrrentInput will return the 'top' of the input stack, which is where
 * the next input to be read after the callback returns will come from.
 * NOTE: after calling Cg_PushInput, this will return the input just pushed
 * If 'iter' is non-null, Cg_NextInput can be used to iterate down the stack
 * of inputs, and will return NULL after the top level source file is
 * reached */

DLLEXPORT const char * const *Cg_GetIncludePaths(CgStruct *Cg, int *count);
/* returns the array of NUL-terminated include paths, which is not 
   NULL-terminated. */

#if defined(LW_MACOSX_OPENGL) && !defined(LW_IN_CGC)
#include <OpenGL/gldPPStream.h>
#else
typedef union _PPStreamToken PPStreamToken;
#endif
DLLEXPORT int Cg_ParsePPStream(CgStruct *Cg, PPStreamToken *stream, const char * const *opts, char **listing);

DLLEXPORT const char *Cg_GetDepInfo(CgStruct *Cg);
/* return the dependency info (list of files, separated by spaces) from the
 * last call to Cg_ParseXXX.  This pointer will only be valid until the next
 * call to Cg_ParseXXX, or to Cg_CompilerCleanup
 */

DLLEXPORT void Cg_ClearParsedCode(CgStruct *Cg, char **listing);
/* compiler entry point to remove all previously parsed code from the
 * Cg context. */

DLLEXPORT int Cg_ExtractBindings(CgStruct *Cg, const char *profile, const char *entry,
                       const char * const *opts, BindingList **bindings, BindingTypeList **bindingTypes,
                       char **listing);
/* compiler entry point for extracting binding information for a
 * program with specified profile and entry point.  'opts' is a
 * NULL-terminated list of options that may be used to override the
 * options specified in Cg_CompilerInit.  'bindings' is an output set to
 * the list of bindings contained within the program, allocated using
 * the memory pool passed to Cg_CompilerInit.  'bindingTypes' is an
 * output set to the list of binding types contained within the
 * program, allocated using the memory pool passed to Cg_CompilerInit.
 * 'listing' is as in Cg_ParseString/Cg_ParseFile.  The value returned
 * is 0 upon success, non-zero otherwise.
 */

DLLEXPORT char *Cg_Compile(CgStruct *Cg, const char *profile, const char *entry, const char * const *opts,
                 const BindingList *inputBindings, BindingList **finalBindings,
                 char **listing, char **binout, int *binoutsize);
/* compiler entry point to compile a program with a given profile and
 * entry point.  'opts' is a NULL-terminated list of options that may
 * be used to override the options specified in Cg_CompilerInit.
 * 'inputBindings' is an input list of bindings as returned by
 * Cg_ExtractBindings or a modified version of the binding list
 * returned by Cg_ExtractBindings used to specialize the given
 * program.  'finalBindings', an output value, is the final list of
 * bindings after the program has been comiled.  'listing' is as in
 * Cg_ParseFile, Cg_ParseString.  The value returned is the text of
 * the compiled program, allocated using the memory pool passed to
 * Cg_CompilerInit.
 */

DLLEXPORT int Cg_FindProgramInfo(CgStruct *Cg, int option, int *value, int *isAtom);
/* extract named info param about last compiled program in this CgStruct.
 * 'option' is the name (atom) and *value is used to return the value, which
 * may be either an integer or an atom (*isAtom will be set to 0 for integers
 * and 1 for atoms).  Returns 1 if there is such a param, 0 if fails
 */

#define COMPILER_FIND_PROFILE_INFO_AVAILABLE 1
DLLEXPORT int Cg_FindProfileInfo(CgStruct *Cg, int option, int *value, int *isAtom);
/* extract named profile param about last compiled program in this CgStruct.
 * 'option' is the name (atom) and *value is used to return the value, which
 * may be either an integer or an atom (*isAtom will be set to 0 for integers
 * and 1 for atoms).  Returns 1 if there is such a param, 0 if fails
 */

DLLEXPORT int Cg_GetGLSLVersion(CgStruct *Cg);
/* Return the GLSL version from the compiled program
 */

DLLEXPORT char *Cg_ImportParsedCode(CgStruct *Cg, CgStruct *from);
/* import parsed code from an existing CgStruct, combining its global scope
 * into the global scope of the traget CgStruct.  This leaves references to
 * the `from' CgStruct in the target CgStruct until the target CgStruct is
 * cleaned up or has its parsed code cleared, so you can't modify the `from'
 * CgStruct until its done.  If you cleanup the `from' CgStruct that will be
 * automatcially deferred until its no longer referenced.
 */

DLLEXPORT int Cg_GetErrorCount(CgStruct *Cg);
/* Return the current error count 
 */

DLLEXPORT CgStruct *Cg_ReferenceCgStruct(CgStruct *Cg);
/* increment the reference count on the given CgStruct
 */

typedef union new_Symbol Cg_Symbol;
typedef struct Technique_Rec CgFX_Technique;
typedef struct Pass_Rec CgFX_Pass;
typedef struct State_Rec CgFX_StateAssign;
typedef CgFX_Pass CgFX_SamplerState;
    /* we make a CgFX_SamplerState the same as a CgFX_Pass so we can pass
     * either to Cg_GetStateInitializers.  It will never have a name or
     * annotations, so those functions will always return NULL */
typedef union new_Type CgFX_Annotations;
/* These handle types refer to internal data structures within the compiled
 * code.  They are only valid until Cg_ClearParsedCode or Cg_CompilerCleanup
 * is called.  The caller does not need to do any additional cleanup of
 * them */

DLLEXPORT void Cg_IterateGlobals(CgStruct *Cg,
                       void (*fn)(CgStruct *, Cg_Symbol *, void *),
                       void *arg);
/* Iterate over the global scope and call the provided function for every
 * symbol.  The third argument to Cg_IterateGlobals is used as the third
 * argument to the function
 */

DLLEXPORT void Cg_IterateLocals(CgStruct *Cg, Cg_Symbol *,
                       void (*fn)(CgStruct *, Cg_Symbol *, void *),
                       void *arg);
/* Iterate over the local scope of a function and call the provided function
 * for every symbol.  The third argument to Cg_IterateGlobals is used as
 * the third argument to the function
 */

DLLEXPORT void Cg_IterateTechniques(CgStruct *Cg,
                          void (*fn)(CgStruct *, CgFX_Technique *, void *),
                          void *arg);
/* Iterate over the techniques in the global scope */

DLLEXPORT void Cg_IteratePasses(CgStruct *Cg, CgFX_Technique *tech,
                      void (*fn)(CgStruct *, CgFX_Pass *, void *),
                      void *arg);
/* Iterate over the passes in a technique */

DLLEXPORT void Cg_IterateTechniqueSymbols(CgStruct *Cg, CgFX_Technique *tech,
                                void (*fn)(CgStruct *, Cg_Symbol *, void *),
                                void *arg);
/* Iterate over the symbols in a technique */

DLLEXPORT void Cg_IterateStateAssignments(CgStruct *Cg, CgFX_Pass *pass,
                                void (*fn)(CgStruct *, CgFX_StateAssign *, void *),
                                void *arg);
/* Iterate over the state assignments in a pass */

DLLEXPORT CgFX_Annotations *Cg_GetSymbolAnnotations(CgStruct *Cg, Cg_Symbol *sym);
DLLEXPORT CgFX_Annotations *Cg_GetTechniqueAnnotations(CgStruct *Cg, CgFX_Technique *tech);
DLLEXPORT CgFX_Annotations *Cg_GetPassAnnotations(CgStruct *Cg, CgFX_Pass *pass);
/* get the Annotations on a symbol, technique, or pass */

DLLEXPORT void Cg_IterateAnnotations(CgStruct *Cg, CgFX_Annotations *annot,
                          void (*fn)(CgStruct *, Cg_Symbol *, void *),
                          void *arg);
/* iterate through the symbols in the Annotations */

DLLEXPORT void Cg_IterateStructFields(CgStruct *Cg, Cg_Symbol *struct_sym,
                          void (*fn)(CgStruct *, Cg_Symbol *, int, void *),
                          void *arg);
/* iterate through the symbols in the struct */

DLLEXPORT int Cg_IsFunction(Cg_Symbol *);  /* test if a Cg_Symbol is a function */
DLLEXPORT int Cg_IsVariable(Cg_Symbol *);  /* test if a Cg_Symbol is a variable */
DLLEXPORT int Cg_GetSymbolName(Cg_Symbol *);  /* get the name of a symbol */
DLLEXPORT int Cg_GetSymbolSemantic(Cg_Symbol *); /* get symbol's semantic, if any */
DLLEXPORT int Cg_GetTechniqueName(CgFX_Technique *);  /* get the name of a technique */
DLLEXPORT int Cg_GetPassName(CgFX_Pass *);  /* get the name of a pass */

/* if a variable is initialized with a sampler_state, get it.  Otherwise
 * returns NULL */
DLLEXPORT CgFX_SamplerState *Cg_GetSamplerStateInitializer(CgStruct *Cg, Cg_Symbol *var);

DLLEXPORT BindingTypeField *Cg_GetVariableType(CgStruct *Cg, Cg_Symbol *var,
                                     IMemPool *alloc);
/* Get the type of a variable.  The provided alloc will be used to allocate
 * space for the returned data.  If NULL, the default alloc for the CgStruct
 * will be used */
DLLEXPORT BindingTypeMethod *Cg_GetFunctionType(CgStruct *Cg, Cg_Symbol *fun,
                                      IMemPool *alloc);
/* Get the type of a function.  The provided alloc will be used to allocate
 * space for the returned data.  If NULL, the default alloc for the CgStruct
 * will be used */

DLLEXPORT IBasicBlock *Cg_GetFunctionCode(CgStruct *Cg, Cg_Symbol *fun, const char *profile,
                                const char * const *opts, char **listing, IMemPool *alloc);
/* Generate code in IBasicBlock list format for a function.  If the given
 * alloc is NULL, the default alloc for the CgStruct will be used.  If
 * an error oclwrs, will return NULL and the listing arg will be set to
 * an error message */

DLLEXPORT IBasicBlock *Cg_GetInitializer(CgStruct *Cg, Cg_Symbol *var, const char *profile,
                               const char * const *opts, char **listing, IMemPool *alloc);
/* Generate code for the initializer on a single variable */

DLLEXPORT IBasicBlock *Cg_GetGlobalInitializers(CgStruct *Cg, const char *profile, const char * const *opts,
                                      char **listing, IMemPool *alloc);
/* Generate code for all the initializers in the global scope. */

DLLEXPORT IBasicBlock *Cg_GetTechniqueInitializers(CgStruct *Cg, CgFX_Technique *tech,
                                         const char *profile, const char * const *opts,
                                         char **listing, IMemPool *alloc);
/* Generate code for the initializers in a technique scope */

DLLEXPORT IBasicBlock *Cg_GetAnnotationsInitializers(CgStruct *Cg, CgFX_Annotations *annot,
                                          const char *profile, const char * const *opts,
                                          char **listing, IMemPool *alloc);
/* Generate code for the contents of the Annotations */

typedef struct CgFX_StateSet_rec CgFX_StateSet;
typedef struct CgFX_StateConstInfo_rec {
    const char  *name;
    DagType_t   type;
    int         size; /* 0 for scalar */
    union {
        int     i;
        float   f;
    } val[4];
    /* this is a performance addition for the Cg runtime */
    void * CgRuntimeAtom; /* DO NOT REMOVE THIS without updating the Cg runtime */
} CgFX_StateConstInfo;
typedef struct CgFX_StateInfo_rec {
    const char  *name;
    DagType_t   type;
    short       vsize;  /* vector size of state assignments */
    short       msize;  /* matrix # of rows */
    /* for scalar types, vsize = msize = 0
     * for vector types, msize = 0, vsize = vector size
     * for matrix types, msize = rows, vsize = columns
     * must always have 0 <= vsize <= 4 and 0 <= msize <= 4
     * msize != 0 when vsize = 0 is not allowed */
    int         asize;  /* array size for array state assignments */
    int                 const_count;
    CgFX_StateConstInfo *constant;
} CgFX_StateInfo;

DLLEXPORT CgFX_StateSet *Cg_InitStateSet(CgFX_StateInfo *states, int count);
DLLEXPORT void Cg_AddStateToSet(CgFX_StateSet *set, CgFX_StateInfo *state);
DLLEXPORT void Cg_DestroyStateSet(CgFX_StateSet *set);

DLLEXPORT IBasicBlock *Cg_GetStateInitializers(CgStruct *Cg, CgFX_Pass *pass,
                                     CgFX_StateSet *states,
                                     const char *profile, const char * const *opts, char **listing,
                                     IMemPool *alloc);
/* Generate code corresponding to the state expressions in a pass */

DLLEXPORT IBasicBlock *Cg_GetStateInitializer(CgStruct *Cg, CgFX_StateAssign *sa,
                                    CgFX_StateSet *states,
                                    const char *profile, const char * const *opts, char **listing,
                                    IMemPool *alloc);

DLLEXPORT void Cg_CompilerCleanup(CgStruct *, char **listing);
/* cleanup function -- this cleans up everything in the CGStruct.
 */

typedef void *(*Cg_TLSGetProc)(void);
typedef void  (*Cg_TLSSetProc)(void*);

DLLEXPORT void Cg_InstallTLSHooks(Cg_TLSGetProc getHook, Cg_TLSSetProc setHook);

#include <stdio.h>
DLLEXPORT void Cg_PrintProfilesAndOptions(FILE *fp);
/* print out the profiles and their options (for a debug message
 * FIXME -- on MSVC this will fail if its called across dll boundaries, as
 * FIXME -- MSVC is broken in this way
 */

#ifdef __cplusplus
}
#endif /* c++ */

#undef DLLEXPORT

#endif /* !defined(__CGCLIB_H) */
