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
 *  Module name              : stdFileNames.h
 *
 *  Description              :
 *
 *        This module provides an abstract data type 
 *        'file search path', by which files can be located.
 *
 *        It also provides functions for constructing file path names
 *        from their components (directory, base name and extension), 
 *        and the reverse: decomposing such file path names into their 
 *        components.
 */

#ifndef stdFileNames_INCLUDED
#define stdFileNames_INCLUDED

/*--------------------------------- Includes ---------------------------------*/

#include "stdTypes.h"
#include "stdLocal.h"
#include "stdStdFun.h"

#ifdef __cplusplus
extern "C" {
#endif

/*---------------------------------- Types -----------------------------------*/

#if defined(STD_OS_win32)
    #define fnamPATHSEP_CHAR ';'
#else
    #define fnamPATHSEP_CHAR ':'
#endif


typedef struct fnamSearchPath  *fnamSearchPath_t;


/*-------------------- Search Path Manipulation Functions --------------------*/

/*
 * Function        : Create a new searchpath.
 * Parameters      : 
 * Function Result : A new, empty searchpath.
 */
fnamSearchPath_t STD_CDECL fnamCreate(void);



/*
 * Function        : Delete a searchpath.
 * Parameters      : searchPath  (I) Search path to delete, or Nil
 *                   deleteNames (I) When True, this function will also
 *                                   delete the directory names that are
 *                                   held by searchPath.
 * Function Result : 
 */
void STD_CDECL  fnamDelete( fnamSearchPath_t searchPath, Bool deleteNames );



/*
 * Function        : Construct search path from path string.
 * Parameters      : path      (I) String containing separated directory paths.
 *                   separator (I) Separator character used in 'path'
 * Function Result : Search path colwerted from 'path', or the 
 *                   Nil search path in case path itself is equal to Nil
 */
fnamSearchPath_t STD_CDECL  fnamColwertSearchPath( cString path, Char separator );


/*
 * Function        : Add the specified directory name to 
 *                   the specified search path, to the end or front, 
 *                   respectively.
 * Parameters      : searchPath  (I) Search path to add to. 
 *                   dirName     (I) Directory name to add.
 */
void STD_CDECL  fnamAddDir       ( fnamSearchPath_t searchPath, String dirName );
void STD_CDECL  fnamAddDirToFront( fnamSearchPath_t searchPath, String dirName );



/*
 * Function        : Add the specified directory name to 
 *                   the specified search path, to the end or front, 
 *                   respectively.
 * Parameters      : dirName     (I) Directory name to add.
 *                   searchPath  (I) Search path to add to. 
 * NB              : This function is an analogon of mapDefine,
 *                   intended as traversal function.
 */
void STD_CDECL fnamAddToDir       ( String dirName, fnamSearchPath_t searchPath );
void STD_CDECL fnamAddToDirToFront( String dirName, fnamSearchPath_t searchPath );



/*
 * Function         : Locate specified file in the directories
 *                    contained by the specified search path,
 *                    in the order in which the directory names
 *                    were added.
 * Parameters       : searchPath     (I) Search path to use, or Nil for simple file testing.
 *                    fileName       (I) File to locate.
 *                    searchRelative (I) When filename contains a relative 
 *                                       directory path, the searchPath is searched
 *                                       iff. searchRelative is True
 *                    research       (I) In case the immediate search of the specified file
 *                                       in the search path fails, searching will continue with
 *                                       fileName changed to its base name
 *                    rejectFile     (I) Callback function to reject a located
 *                                       candidate file name, and instead continue 
 *                                       searching. Or Nil for never reject.
 *                    data           (I) Generic data element passed as additional
 *                                       parameter to every invocation of 'rejectFile'.
 * Function Result  : A new string containing the complete path by
 *                    which the file can be opened in case 
 *                    the file was found, or Nil otherwise.
 */
String STD_CDECL fnamLocateFile( fnamSearchPath_t searchPath, cString fileName, 
                        Bool searchRelative, Bool research, 
                        stdPropertyFun rejectFile, Pointer data );



/*
 * Function        : Traverse over the directories.
 * Parameters      : searchPath (I) Search path to traverse. 
 *                   f          (I) Traversal function.
 *                   data       (I) Data passed to traversal function.
 * Function Result : 
 */
void STD_CDECL  fnamTraverse( fnamSearchPath_t searchPath, stdEltFun f, Pointer data );



/*
 * Function        : Print the specified searchpath on stream.
 * Parameters      : stream     (I) Stream.
 *                   searchPath (I) Search path to print. 
 * Function Result : 
 */
void STD_CDECL  fnamPrint( FILE* stream, fnamSearchPath_t searchPath );


/*-------------------- Filename (de)composition Functions --------------------*/

/*
 * Function        : Build a path name from its components.          
 * Parameters      : dirName    (I) Directory name.
 *                   fileName   (I) File name.
 *                   extension  (I) File extension name.
 * Function Result : Constructed path.
 */
String STD_CDECL fnamComposePath( cString dirName, cString fileName, cString extension );



/*
 * Function         : Decompose a path name into its components          
 * Parameters       : path       (I) pathname
 *                    dirName    (O) directory part of pathname
 *                    fileName   (O) file base name part of pathname
 *                    extension  (O) extension part of pathname
 * Function result  : -
 */
void STD_CDECL  fnamDecomposePath( cString path, String *dirName, String *fileName, String *extension );



/*
 * Function         : Split a path name into its directory path and its filename          
 * Parameters       : path       (I) pathname
 *                    dirName    (O) directory part of pathname
 *                    fileName   (O) file part of pathname
 * Function result  : -
 */
void STD_CDECL  fnamSplitPath( cString path, String *dirName, String *fileName );



/*
 * Function        : Heuristic decomposition of the name of a shared library 
 *                   into its base name and its version string.
 *                   For example, on OSX, libssl.0.9.7.dylib
 *                   results in 'ssl' and '0.9.7'.
 * Parameters      : path       (I) path representing shared library file
 *                   dirName    (O) Directory part of pathname.
 *                   baseName   (O) Base name of shared library, or Nil 
 *                                  if procIsTempName(path)
 *                   version    (O) Version string of shared library, 
 *                                  or Nil when not present.
 * Function Result : 
 */
void STD_CDECL  fnamDecomposeSharedLibraryName( cString path, String *dirName, String *baseName, String *version );


/*---------------------------- Filename Predicates ---------------------------*/

/*
 * Function        : Test whether specified path is a directory name.          
 * Parameters      : fileName   (I) File name to inspect.
 * Function Result : True iff. the specified file name is a directory
 */
Bool STD_CDECL fnamIsDirectory( cString name );


/*
 * Function        : Test whether specified file exists as a non directory file.          
 * Parameters      : fileName   (I) File name to inspect.
 * Function Result : True iff. the specified file exists and is not a directory.
 */
Bool STD_CDECL fnamFileExists( cString name );


/*
 * Function        : Inspect path type.          
 * Parameters      : fileName   (I) File name to inspect.
 * Function Result : True iff. the specified file name is an absolute path
 */
Bool STD_CDECL fnamIsAbsolutePath( cString fileName );


/*
 * Function        : Lookup file in current exelwtable search path.          
 * Parameters      : fileName   (I) File to lookup.
 * Function Result : Full path of exelwtable, when found, or Nil
 */
String STD_CDECL fnamLocateExelwtable( cString fileName );

/*
 * Function        : Lookup file in current exelwtable search path, with
                     predicate callback function to reject candidates
 * Parameters      : fileName   (I) File to lookup.
                     rejectFile (I) Callback predicate function to check if
                                    candidate found is acceptable.
                     data       (I) User payload to pass to the callback
                                    function.
 * Function Result : Full path of exelwtable, when found, or Nil
 */
String STD_CDECL fnamLocateExelwtableWithPredicate( cString fileName,
                                                    stdPropertyFun rejectFile, 
                                                    Pointer data );
                                                    

/*----------------------------- Directory Search -----------------------------*/

/*
 * Function        : Apply specified function to all files whose name matches the
 *                   specified pattern, in the specified directory,
 *                   with specified generic data element as additional parameter.
 *                   Notes: 
 *                    - the names passed to 'traverse' will be deallocated by 
 *                      fnamTraverseDirectory immediately after 'traverse' terminates. 
 *                      Hence, it should be copied if needed afterwards.
 * Parameters      : path            (I) Directory to traverse, or Nil for the current directory
 *                   pattern         (I) File name pattern, or Nil for all.
 *                                       This pattern should contain name pattern using 
 *                                       wildcard character '*'.
 *                   persistentName  (I) When this parameter is equal to 'True', then persistent 
 *                                       copies of file names are passed to 'traverse;
 *                                       otherwise these names are valid only during the
 *                                       traversal call.
 *                   prependPath     (I) When this parameter is True, the names passed to
 *                                       the traverse function will be prepended with the directory
 *                                       path. In the other case, when it is False, the names will
 *                                       be passed without any path.
 *                   traverse        (I) Function to apply to all found file names.
 *                   data            (I) Generic data element passed as additional
 *                                       parameter to every invocation of 'traverse'.
 * Function Result :
 */
void STD_CDECL fnamTraverseDirectory( cString path, cString pattern, Bool persistentName, Bool prependPath,
                                stdEltFun  traverse, Pointer data );

/* Check if two files point to same file */
Bool STD_CDECL fnamEquivalent( cString file1, cString file2);

#ifdef __cplusplus
}
#endif

#endif
