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
 *  Module name              : stdFileNames.c
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

/*--------------------------------- Includes ---------------------------------*/

#include "stdFileNames.h"
#include "stdLocal.h"
#include "stdList.h"
#include "stdString.h"
#include "stdProcess.h"

#if defined(LWVM_WSL) && !defined(LWVM_WSA)
// FIXME: Remove this hack once DX moves to a sane linking model for Linux/WSL
int __xstat(int ver, const char* path, struct stat* stat_buf);
int __fxstat(int ver, int fildes, struct stat* stat_buf);

#define stat(x,y)       __xstat(3, x, y)
#define fstat(x,y)      __fxstat(3, x, y)
#endif

/*---------------------------------- Types -----------------------------------*/


typedef struct fnamSearchPath {
      listXList       (directories);
} fnamSearchPath;

#define SLASH_CHAR  '/'
#define BSLASH_CHAR '\\'

/*-------------------- Search Path Manipulation Functions --------------------*/

/*
 * Function        : Create a new searchpath
 * Parameters      : - 
 * Function result : A new, empty searchpath
 */
fnamSearchPath_t STD_CDECL fnamCreate( void )
{
    fnamSearchPath_t result;

    stdNEW(result);

    listXInit(result->directories);

    return result;
}


/*
 * Function        : Delete a searchpath.
 * Parameters      : searchPath  (I) Search path to delete, or Nil
 *                   deleteNames (I) When True, this function will also
 *                                   delete the directory names that are
 *                                   held by searchPath.
 * Function Result : 
 */
void STD_CDECL  fnamDelete( fnamSearchPath_t searchPath, Bool deleteNames )
{
    if (searchPath) {
        if (deleteNames) {
            listObliterate( searchPath->directories, stdFreeFun );
        } else {
            listDelete( searchPath->directories );
        }

        stdFREE( searchPath );
    }
}



/*
 * Function        : Construct search path from path string.
 * Parameters      : path      (I) String containing separated directory paths.
 *                   separator (I) Separator character used in 'path'
 * Function Result : Search path colwerted from 'path', or the 
 *                   Nil search path in case path itself is equal to Nil
 */
fnamSearchPath_t STD_CDECL fnamColwertSearchPath( cString path, Char separator )
{
    if (!path) {
        return Nil;
    } else {
        fnamSearchPath_t result= fnamCreate();

        while (path) {
            Char saved= 0;
            Char *c= strchr(path,separator);

          #ifdef STD_OS_win32
           /*
            * By windows convention, quotes may occur in search paths,
            * in order to allow directory/file names that contain
            * otherwise reserved characters. 
            * Until somebody raises a complaint, we do not even
            * try to handle the '\' here, because then it will be time
            * to introduce a general purpose string parser that handles
            * such metacharacters properly, and use it at various places
            * in stdProcess.c and lwcc.c:
            */
            Char *q= strchr(path,'"');

            while (c && q && c>q) {
                c= strchr(q+1,'"');
                if (c) { 
                    q= strchr(c+1,'"');
                    c= strchr(c+1,separator); 
                }
            }
          #endif

            if (c) { saved= *c; *c= 0; }

            if (*path) { 
              #ifdef STD_OS_win32
                stdString_t buf= stringNEW();

                while (*path) {
                    if (*path != '"') {
                        stringAddChar(buf,*path);
                    }
                    path++;
                }

                fnamAddDir(result, stringStripToBuf(buf) ); 
              #else
                fnamAddDir(result, stdCOPYSTRING(path) ); 
              #endif
            }

            if (c) { *(c++)= saved; }

            path= c;
        }

        return result;
    }
}



/*
 * Function        : Add the specified directory name to 
 *                   the specified search path, to the end or front, 
 *                   respectively.
 * Parameters      : searchPath  (I) Search path to add to. 
 *                   dirName     (I) Directory name to add.
 */
void STD_CDECL fnamAddDir( fnamSearchPath_t searchPath, String dirName )
{ listXPutAfter( searchPath->directories, dirName ); }

void STD_CDECL fnamAddDirToFront( fnamSearchPath_t searchPath, String dirName )
{ listXPutInFront( searchPath->directories, dirName ); }


/*
 * Function        : Add the specified directory name to 
 *                   the specified search path, to the end or front, 
 *                   respectively.
 * Parameters      : dirName     (I) Directory name to add.
 *                   searchPath  (I) Search path to add to. 
 * NB              : This function is an analogon of mapDefine,
 *                   intended as traversal function.
 */
void STD_CDECL fnamAddToDir       ( String dirName, fnamSearchPath_t searchPath )
{ listXPutAfter( searchPath->directories, dirName ); }

void STD_CDECL fnamAddToDirToFront( String dirName, fnamSearchPath_t searchPath )
{ listXPutInFront( searchPath->directories, dirName ); }





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
                        stdPropertyFun rejectFile, Pointer data )
{
    struct stat   statBuff;
    String        result;
    String        directory;
    Bool          testDirect;
    
    fnamDecomposePath( fileName, &directory, Nil, Nil );
    
    testDirect = !searchPath
              || (directory && (fnamIsAbsolutePath(directory) || !searchRelative))
               ;
    
    stdFREE(directory);
     

    if ( testDirect ) {
        if ( stat(fileName, &statBuff) == 0 ) { return stdCOPYSTRING(fileName); }
    
    } else {
        stdList_t sp= searchPath->directories;

        while (sp != Nil) {
            result= fnamComposePath(sp->head, fileName, Nil);
            
            if ( stat(result, &statBuff) == 0
              && (!rejectFile || !rejectFile(result,data))
               ) { return result; }
               
            stdFREE(result);
            
            sp= sp->tail;
        }
    }

    if (research && searchPath) {
        String result = Nil;
    
        String directory,name,ext;
        
        fnamDecomposePath( fileName, &directory, &name, &ext );
    
        if (directory) {
            String baseName = fnamComposePath(Nil,name,ext);
            
            result = fnamLocateFile(searchPath,baseName,searchRelative,False,rejectFile,data);
            
            stdFREE(baseName);
        }
        
        stdFREE3(directory,name,ext);
        
        return result;
    }

    return Nil;
}




/*
 * Function        : Traverse over the directories
 * Parameters      : searchPath (I) search path to traverse 
 *                   f          (I) traversal function
 *                   data       (I) data passed to traversal function
 * Function result : -
 */
void STD_CDECL  fnamTraverse( fnamSearchPath_t searchPath, stdEltFun f, Pointer data )
{
    listTraverse(searchPath->directories, f, data);
}



/*
 * Function        : Print the specified searchpath on stream
 * Parameters      : searchPath (I) search path to print 
 * Function result : -
 */
void STD_CDECL  fnamPrint( FILE* stream, fnamSearchPath_t searchPath )
{
    stdList_t sp  = searchPath->directories;
    String    sep = "";

    while (sp != Nil) {
        fprintf(stream, "%s%s", sep, (String)sp->head);
        sep = ":";
        sp  = sp->tail;
    }
}



/*-------------------- Filename (de)composition Functions --------------------*/

/*
 * Function        : Build a path name from its components          
 * Parameters      : dirName    (I) directory name
 *                   fileName   (I) file name
 *                   extension  (I) file extension name
 * Function result : Constructed path
 */
String STD_CDECL fnamComposePath( cString dirName, cString fileName, cString extension )
{
    stdString_t buf= stringNEW();
   
    if (dirName && *dirName) { 
        cString dirEnd= dirName + strlen(dirName);
        
        while ( dirEnd > dirName
             && ( *(dirEnd-1)==SLASH_CHAR
             #if defined(STD_OS_CygWin) || defined(STD_OS_win32) || defined(STD_OS_MinGW)
               || *(dirEnd-1)==BSLASH_CHAR
             #endif
              ) ) {
            dirEnd--;
        }

        while (dirName < dirEnd) {
            stringAddChar ( buf, *(dirName++) );
        }
            
        stringAddChar ( buf, SLASH_CHAR );
    }
    
        stringAddBuf  ( buf, fileName   );
        
    if (extension && *extension) { 
        stringAddChar ( buf, '.'        );
        stringAddBuf  ( buf, extension  );
    }

    return stringStripToBuf( buf );
}


/*
 * Function         : Decompose a path name into its components          
 * Parameters       : path       (I) pathname
 *                    dirName    (O) directory part of pathname
 *                    fileName   (O) file base name part of pathname
 *                    extension  (O) extension part of pathname
 * Function result  : -
 */
void STD_CDECL  fnamDecomposePath( cString path, String *dirName, String *fileName, String *extension )
{
   /*
    * Generate temporary copy of 'path', since 
    * this function temporarily modifies it. This 
    * is safer in multithreaded elwironments, and if path 
    * happens to be a string constant (which might
    * be mapped in a read only section:
    */
    String path2 = stdCOPYSTRING(path);
    {
        String  dotPos    = strrchr(path2, '.');
        String  slashPos  = strrchr(path2,  SLASH_CHAR);

      #if defined(STD_OS_CygWin) || defined(STD_OS_win32) || defined(STD_OS_MinGW)
        String  bSlashPos = strrchr(path2, BSLASH_CHAR);

        if (slashPos < bSlashPos) { slashPos= bSlashPos; }
      #endif

        if (dotPos   < slashPos ) { dotPos  = Nil;      }

        if (dirName) {
            if (slashPos) { 
               *slashPos = 0; 
               *dirName  = stdCOPYSTRING(path2);
               *slashPos = SLASH_CHAR; 
            } else {
               *dirName  = Nil;
            }
        }

        if (fileName) {
            if (dotPos) { *dotPos= 0;   }

            if (slashPos) { 
               *fileName= stdCOPYSTRING(slashPos+1);
            } else {
               *fileName= stdCOPYSTRING(path2);
            }

            if (dotPos) { *dotPos= '.'; }
        }

        if (extension) {
            if (dotPos) { 
               *extension = stdCOPYSTRING(dotPos+1); 
            } else { 
               *extension = Nil; 
            }
        }
    }
    stdFREE(path2);
}


/*
 * Function         : Split a path name into its directory path and its filename          
 * Parameters       : path       (I) pathname
 *                    dirName    (O) directory part of pathname
 *                    fileName   (O) file part of pathname
 * Function result  : -
 */
void STD_CDECL  fnamSplitPath( cString path, String *dirName, String *fileName )
{
   /*
    * Generate temporary copy of 'path', since 
    * this function temporarily modifies it. This 
    * is safer in multithreaded elwironments, and if path 
    * happens to be a string constant (which might
    * be mapped in a read only section:
    */
    String path2 = stdCOPYSTRING(path);
    {
        String  slashPos = strrchr(path2, SLASH_CHAR);

      #if defined(STD_OS_CygWin) || defined(STD_OS_win32) || defined(STD_OS_MinGW)
        String  bSlashPos = strrchr(path2, BSLASH_CHAR);

        if (slashPos < bSlashPos) { slashPos= bSlashPos; }
      #endif

        if (dirName) {
            if (slashPos) { 
               *slashPos = 0; 
               *dirName  = stdCOPYSTRING(path2);
               *slashPos = SLASH_CHAR; 
            } else {
               *dirName  = Nil;
            }
        }

        if (fileName) {
            if (slashPos) { 
               *fileName= stdCOPYSTRING(slashPos+1);
            } else {
               *fileName= stdCOPYSTRING(path2);
            }
        }
    }
    stdFREE(path2);
}



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
 
    #if  !defined  STD_OS_win32
    static Bool instring( Char p, cString s )
    {
        while (p != *s) {
            if (*s) { s++;          }
               else { return False; }
        }
        
        return True;
    }
    #endif 
 
void STD_CDECL  fnamDecomposeSharedLibraryName( cString path, String *dirName, String *baseName, String *version )
{
    if (dirName ) { *dirName  = Nil; }
    if (baseName) { *baseName = Nil; }
    if (version ) { *version  = Nil; }
    
    #if  defined  STD_OS_win32
    {
        String base,ext;

        fnamDecomposePath(path,dirName,&base,&ext);

        if (!ext || stdEQSTRING(ext,"dll")) {
            if (baseName) { *baseName=base; } else { stdFREE(base); }
            stdFREE(ext);
        } else {
            if (baseName) { *baseName= fnamComposePath(Nil,base,ext); }
            stdFREE2(base,ext);
        }
    }

    #else 
    {
        String base,ext;

        fnamDecomposePath(path,dirName,&base,&ext);

        if ( !ext
          || stdEQSTRING(ext,"so"    )
          || stdEQSTRING(ext,"dylib" )
           ) {
           stdFREE(ext);
        } else {
           String b     = fnamComposePath(Nil,base,ext);
           String so    = strstr(b,".so.");
           String dylib = strstr(b,".dylib.");

           stdFREE2(base,ext);
           base = b;

           if (so) {
               if (version) { *version= stdCOPYSTRING( so+strlen(".so.") ); }
              *so=0;
           } else 
           if (dylib) {
               if (version) { *version= stdCOPYSTRING( dylib+strlen(".dylib.") ); }
              *dylib=0;
           }
        }

        {
            Char *p= base + strlen(base);

            while (p>base && instring(*(p-1),"01-.23456789")) { p--; }

            if (*p) {
               Char *pp= p;
               if (*pp=='-' || *pp=='.') { pp++; }
               if (version) { *version = stdCOPYSTRING(pp); }
               *p=0;
            }
        }

        if (stdIS_PREFIX("lib",base)) {
            if (baseName) { *baseName= stdCOPYSTRING( base+strlen("lib") ); }
        } else {
            if (baseName) { *baseName= stdCOPYSTRING( base ); }
        }

        stdFREE(base);
    }
    #endif
}


/*---------------------------- Filename Predicates ---------------------------*/

/*
 * Function        : Test whether specified path is a directory name.          
 * Parameters      : fileName   (I) File name to inspect.
 * Function Result : True iff. the specified file name is a directory
 */
Bool STD_CDECL fnamIsDirectory( cString name )
{
    struct stat buf;
    
    return stat(name,&buf) == 0
  #ifdef STD_OS_win32
        && (buf.st_mode & _S_IFDIR);
  #else
        && S_ISDIR(buf.st_mode);
  #endif
}


/*
 * Function        : Test whether specified file exists as a non directory file.          
 * Parameters      : fileName   (I) File name to inspect.
 * Function Result : True iff. the specified file exists and is not a directory.
 */
Bool STD_CDECL fnamFileExists( cString name )
{
    struct stat buf;
    
    return stat(name,&buf) == 0
  #if defined(STD_OS_win32) || defined(STD_OS_MinGW)
        && !(buf.st_mode & _S_IFDIR);
  #else
        && !S_ISDIR(buf.st_mode);
  #endif
}


/*
 * Function        : Inspect path type.          
 * Parameters      : fileName   (I) File name to inspect.
 * Function Result : True iff. the specified file name is an absolute path
 */
Bool STD_CDECL  fnamIsAbsolutePath( cString fileName )
{
    if (fileName[0]==0          ) { return True;  }
    if (fileName[0]==SLASH_CHAR ) { return True;  }
#if defined(STD_OS_CygWin) || defined(STD_OS_win32) || defined(STD_OS_MinGW)
    if (fileName[0]==BSLASH_CHAR) { return True;  }
    if (fileName[1]==':'        ) { return True;  }
#endif
    return False;
}

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
                                                    Pointer data )
{
    fnamSearchPath_t path;
    String           result;
    String           fnam = stdCOPYSTRING(fileName);
    
#if defined(STD_OS_CygWin) || defined(STD_OS_win32) || defined(STD_OS_MinGW)
   /*
    * Force the fnam to have .exe extension:
    */
    {
        String dir,name,ext;
        fnamDecomposePath(fnam,&dir,&name,&ext);
        
        if (ext && !stdEQSTRING_C(ext,"exe")) {
            String tmp= fnamComposePath(Nil,name,ext);
            stdFREE2(name,ext);
            name=tmp; ext= Nil;
        } 
        
        if (ext==Nil) {
            stdFREE(fnam);
            fnam= fnamComposePath(dir,name,"exe");
        } 
        
        stdFREE3(dir,name,ext);
    }
#endif

    path   = fnamColwertSearchPath( getelw("PATH"), fnamPATHSEP_CHAR );
#if defined(STD_OS_win32) || defined(STD_OS_MinGW)
    fnamAddDirToFront(path,stdCOPYSTRING("."));
#endif
    result = fnamLocateFile( path, fnam, False, False, rejectFile, data );
    
    fnamDelete( path, True ); 
    
    stdFREE(fnam);
    
    return result;
}


/*
 * Function        : Lookup file in current exelwtable search path.          
 * Parameters      : fileName   (I) File to lookup.
 * Function Result : Full path of exelwtable, when found, or Nil
 */
String STD_CDECL fnamLocateExelwtable( cString fileName )
{
    return fnamLocateExelwtableWithPredicate(fileName, Nil, Nil);    
}




/*----------------------------- Directory Search -----------------------------*/

    #if defined(STD_OS_win32) || defined(STD_OS_MinGW)
    #else
    static Bool wcMatch( cString s, cString p )
    {
        if (!p[0]) { 
            return !s[0]; 
        } else

        if (!s[0]) { 
            while (p[0]=='*') { p++; }; 
            return !p[0];
        } else

        if ( s[0]==p[0]
          || p[0]=='?'
           ) {
            return wcMatch(s+1,p+1);
        } else

        if ( p[0]=='*' ) {
            while (p[0]=='*') { p++; }; 

            return wcMatch(s,p) || wcMatch(s+1,p-1);
        } else {

            return False;
        }
    }
    #endif

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
                                stdEltFun  traverse, Pointer data )
{    
    #if defined(STD_OS_win32) || defined(STD_OS_MinGW)
        if (!pattern) { pattern= "*"; }

        {
            String tmpl= fnamComposePath(path,pattern,Nil);
            struct __finddata64_t dir;
            intptr_t handle= _findfirst64(tmpl,&dir);

            if (handle != (intptr_t)-1) {
                do {
                    String enam= dir.name;
                    if (enam[0] != '.') {
                        if (prependPath) {
                            String penam= fnamComposePath(path,enam,Nil);
                            traverse(penam,data);
                            if (!persistentName) { stdFREE(penam);           }
                        } else {
                            if ( persistentName) { enam=stdCOPYSTRING(enam); }
                            traverse(enam,data);
                        }
                    }
                } while (_findnext64(handle,&dir)==0);
                _findclose(handle);
            }
            
            stdFREE(tmpl);
        }
    #else
        if (!path) { path= "."; }

        {
            DIR* dir= opendir(path);

            if (dir) {
                struct dirent *entry;

                while ( (entry= readdir(dir)) ) {
                    String enam= entry->d_name;
                    if (enam[0] != '.' && (!pattern || wcMatch(enam,pattern)) ) {
                        if (prependPath) {
                            String penam= fnamComposePath(path,enam,Nil);
                            traverse(penam,data);
                            if (!persistentName) { stdFREE(penam);           }
                        } else {
                            if ( persistentName) { enam=stdCOPYSTRING(enam); }
                            traverse(enam,data);
                        }
                    }
                }
                closedir(dir);
            }
        }
    #endif
}

// this is similar to C++ std::filesystem::equivalent
Bool STD_CDECL fnamEquivalent( cString file1, cString file2)
{    
#if defined(STD_OS_win32)
  // windows defines linux stat call, but it doesn't set ino field,
  // so instead use GetFileInformationByHandle() to compare information.
  Bool st1, st2;
  BY_HANDLE_FILE_INFORMATION i1, i2;
  HANDLE h1 = CreateFile(file1, GENERIC_READ, FILE_SHARE_READ, Nil, 
                         OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, Nil);
  HANDLE h2 = CreateFile(file2, GENERIC_READ, FILE_SHARE_READ, Nil, 
                         OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, Nil);
  if (h1 == ILWALID_HANDLE_VALUE || h2 == ILWALID_HANDLE_VALUE) {
    return False;
  }
  st1 = GetFileInformationByHandle(h1, &i1);
  st2 = GetFileInformationByHandle(h2, &i2);
  CloseHandle(h1);
  CloseHandle(h2);
  if ( ! st1 || ! st2) {
    return False;
  }
  return i1.nFileIndexHigh == i2.nFileIndexHigh
      && i1.nFileIndexLow == i2.nFileIndexLow
      && i1.nFileSizeHigh == i2.nFileSizeHigh 
      && i1.nFileSizeLow == i2.nFileSizeLow
      && i1.ftLastWriteTime.dwHighDateTime == i2.ftLastWriteTime.dwHighDateTime
      && i1.ftLastWriteTime.dwLowDateTime == i2.ftLastWriteTime.dwLowDateTime
      && i1.dwVolumeSerialNumber == i2.dwVolumeSerialNumber;

#else // Linux
    struct stat s1, s2;
    // if cannot stat for some reason, return False to be safe.
    if ( stat(file1, &s1) != 0 ) return False;
    if ( stat(file2, &s2) != 0 ) return False;
    return (s1.st_dev == s2.st_dev) && (s1.st_ino == s2.st_ino);
#endif
}

