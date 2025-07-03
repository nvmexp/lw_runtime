/*
 * Copyright (c) 2015 LWPU Corporation.  All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */
 
#include "lwntest_cpp.h"
#include "lwn_utils.h"
#include <fstream>
#include <TCHAR.H>
#include <sys/stat.h>
#include <io.h>
#include <array>
#include <string>
#include <direct.h>

// Using win32 i/o calls so guard 
#if defined (_WIN32) 

using namespace lwn;

BOOL IsDots(const TCHAR* );
BOOL DeleteDirectory(const TCHAR* );    
void find_and_replace( std::string & src, std::string const & find, std::string const & replace );

        
class LWNGlslcIncludeTest
{   
public:
    static const int total_tests = 13; 
        
    struct Vertex {
    dt::vec3 position;
    dt::vec3 color;
    };
        
    LWNTEST_CppMethods();

private:
    std::string writeHeaders(std::string include_path) const;
    void process_shader_test(Device *device, lwnTest::GLSLCHelper *g_glslcHelper,
                             QueueCommandBuffer &queueCB, BufferAddress vboAddr,
                             const VertexArrayState &vertex, VertexShader &vs,
                             int vertexData_size, int test_number,
                             std::string include_path) const;
};

//clean up \\'s in given string
void find_and_replace( std::string & src, std::string const & find, std::string const & replace ) 
{
    for( std::string::size_type i = 0; ( i = src.find( find, i ) ) != std::string::npos; i += replace.length() )
    {
        src.replace( i, find.length(), replace );
    }
}

//Ripped from the web for directory removal
BOOL IsDots(const TCHAR* str)  
{
    if(_tcscmp(str,".") && _tcscmp(str,"..")) return FALSE;
    return TRUE;
}

//Ripped from the web for directory removal
BOOL DeleteDirectory(const TCHAR* sPath) 
 {
    HANDLE hFind; // file handle
    WIN32_FIND_DATA FindFileData;

    TCHAR DirPath[MAX_PATH];
    TCHAR FileName[MAX_PATH];

    _tcscpy(DirPath,sPath);
    _tcscat(DirPath,_T("\\"));
    _tcscpy(FileName,sPath);
    _tcscat(FileName,_T("\\*")); // searching all files

    hFind = FindFirstFile(FileName, &FindFileData); // find the first file
    if( hFind != ILWALID_HANDLE_VALUE ) 
    {
        do
        {
                if( IsDots(FindFileData.cFileName) ) 
                    continue;

                _tcscpy(FileName + _tcslen(DirPath), FindFileData.cFileName);
                if((FindFileData.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) 
                    {
                        // we have found a directory, relwrse
                        if( !DeleteDirectory(FileName) ) 
                            break; // directory couldn't be deleted
                    }
                else 
                {
                    if(FindFileData.dwFileAttributes & FILE_ATTRIBUTE_READONLY)
                        _chmod(FileName, _S_IWRITE); // change read-only file mode
    
                    if( !DeleteFile(FileName) ) 
                        break; // file couldn't be deleted
                }

        }while( FindNextFile(hFind,&FindFileData) );

        FindClose(hFind); // closing file handle
    }

    return RemoveDirectory(sPath); // remove the empty (maybe not) directory
 }

//Do the shader compilation, code replacement when file not found, draw out results
void LWNGlslcIncludeTest::process_shader_test(Device *device, lwnTest::GLSLCHelper *g_glslcHelper,
                                              QueueCommandBuffer &queueCB, BufferAddress vboAddr,
                                              const VertexArrayState &vertex, VertexShader &vs,
                                              int vertexData_size, int test_number,
                                              std::string include_path) const
{
    //Get CWD 
    char *lwrwd = _getcwd( NULL, 0 );
    std::string  cwd = lwrwd;
    free(lwrwd);
    
    //Set up the absolute path to used for absolute path test
    std::string absolute_path = "#include \"" + cwd;
    absolute_path.append(include_path + "absolute_includes\\frag9.h\"\n\n");
    
    //clean up any spots where there are not enough slashes in the absolute path we grabbed
    find_and_replace(absolute_path, "\\", "\\\\");  

    std::array<std::string, total_tests> shader_content = {     // test 1 : #include <xxx>
                                                                // draws square [0][0]
                                                                "#include <frag.h>\n\n",
                                                                
                                                                //TEST 2 : #include "xxx"
                                                                // draws square [0][1]
                                                                "#include \"frag.h\"\n\n",

                                                                //Test 3 : nested relative path using "/"
                                                                // draws square [0][2]
                                                                "#include \"../frag3.h\"\n\n",
                                                                    
                                                                //Test 4 : nested relative path using "\"
                                                                // draws square [0][3]
                                                                "#include \"..\\\\frag3.h\"\n\n",
                                                                    
                                                                //Test 5 : MACRO in #included header (i.e. not forced header)
                                                                // draws square [0][4]
                                                                "#include \"frag5.h\"\n\n",

                                                                //Test 6 : Macro in force included file - Test for failure 
                                                                //i.e. No forced standard header set yet so this should fail. If this passes draw out red square
                                                                // draws square [0][5]
                                                                "#include \"frag6.h\"\n\n",

                                                                //Test 7 : Macro in force included file 
                                                                // draws square [0][6]
                                                                "#include \"frag7.h\"\n\n",
                                                                   
                                                                //Test 8 : Header file with UTF-8 character in it
                                                                // draws square [1][0]
                                                                "#include <frag8.h>\n\n",

                                                                //Test 9 : Header file path beginning with "/" e.g. /blah.h
                                                                // draws square [1][1]   
                                                                "#include \"/frag.h\"\n\n",
                                                                    
                                                                //Test 10 : Absolute path 
                                                                // draws square [1][2]   
                                                                absolute_path.c_str(),

                                                                //Test 11 : Nested relative path i.e. subfolder with "\" 
                                                                // draws square [1][3]  
                                                                "#include \"sub_includes\\\\frag11.h\"\n\n",
                                                                  
                                                                //Test 12 : Nested relative path i.e. subfolder with "/" 
                                                                // draws square [1][4]
                                                                "#include \"sub_includes/frag11.h\"\n\n",
                                                                   
                                                                //Test 13 : Combo Macros... One macro in forced header, One macro in #included header
                                                                // draws square [1][5]
                                                                "#include \"frag13.h\"\n\n",
    };    
    
    //Device device1 = g_lwnDevice;
    Program *pgm = device->CreateProgram();
    
    FragmentShader fs_x(440);
    fs_x << 
        shader_content[test_number - 1].c_str();
    
    if (!g_glslcHelper->CompileAndSetShaders(pgm, vs, fs_x)) {
        printf("\nTest %d : Include Shader ERROR:\n", test_number);
        printf("Infolog: %s\n", g_glslcHelper->GetInfoLog());
        
        // replace with known good non-include shaders
        //FragmentShader fs_x(440);
        fs_x <<
            "in vec3 ocolor;\n"
            "out vec4 fcolor;\n"
            "void main() {\n"
            "  fcolor = vec4(1.0, 0.0, 0.0, 1.0);\n"
            "}\n";
        if (!g_glslcHelper->CompileAndSetShaders(pgm, vs, fs_x)) {
            printf("Infolog: %s\n", g_glslcHelper->GetInfoLog());
        }
    }

    queueCB.BindProgram(pgm, ShaderStageBits::ALL_GRAPHICS_BITS);  
    queueCB.BindVertexArrayState(vertex);
    queueCB.BindVertexBuffer(0, vboAddr, vertexData_size);
    queueCB.DrawArrays(DrawPrimitive::TRIANGLES, (test_number - 1) * 6, 6);

}       
  
//Write out the header files that we need to use for the tests
std::string LWNGlslcIncludeTest::writeHeaders(std::string include_path) const
{
    std::ofstream os_file;
    const int header_count = 13;
    
    //Header names and content
    std::array<std::string, header_count> header_name = {   include_path + "frag3.h",
                                                            include_path + "forced_includes\\force.h",
                                                            include_path + "absolute_includes\\frag9.h",
                                                            include_path + "base_includes\\sub_includes\\frag11.h", 
                                                            include_path + "base_includes\\frag.h", 
                                                            include_path + "base_includes\\frag5.h",
                                                            include_path + "base_includes\\frag6.h",
                                                            include_path + "base_includes\\frag7.h",
                                                            include_path + "base_includes\\frag8.h",
                                                            include_path + "base_includes\\frag13.h", 
                                                            include_path + "base_includes\\frag14.h",
                                                            include_path + "base_includes\\vert.h",   
                                                            include_path + "base_includes\\macro_define.h"
                                            };
                                    
    std::array<std::string, header_count> header_content =  {   "in vec3 ocolor;\
                                                                 \nout vec4 fcolor;\
                                                                 \n\lwoid main() \
                                                                 {\n\n  fcolor = vec4(0.0, 1.0, 0.0, 1);\n}",  //frag3.h
                                                                 
                                                                "#define FORCE_INCLUDED 1\
                                                                 \n#define RELATIVE_FORCE_INCLUDED 1",  //force.h   
                                                                 
                                                                "in vec3 ocolor;\
                                                                 \nout vec4 fcolor;\
                                                                 \n\lwoid main() \
                                                                 {\n\n  fcolor = vec4(0.0, 1.0, 0.0, 1);\n}",  //frag9.h
                                                                 
                                                                "in vec3 ocolor;\
                                                                 \nout vec4 fcolor;\
                                                                 \n\lwoid main()\
                                                                 {\n\n  fcolor = vec4(0.0, 1.0, 0.0, 1);\n}",  //frag11.h
                                                                 
                                                                "in vec3 ocolor;\
                                                                 \nout vec4 fcolor;\
                                                                 \n\lwoid main()\
                                                                 {\n\n  fcolor = vec4(0.0, 1.0, 0.0, 1);\n}",  //frag.h
                                                                  
                                                                "#include \"macro_define.h\"\
                                                                 \n\nin vec3 ocolor2;\
                                                                 \nout vec4 fcolor2;\
                                                                 \n\lwoid main()\
                                                                 {\n\n  #ifdef MACRO_IN_INCLUDED_HEADER\
                                                                    \n      fcolor2 = vec4(0.0, 1.0, 0.0, 1);\
                                                                    \n  #else\
                                                                    \n      fcolor2 = vec4(1.0, 0.0, 0.0, 1);\
                                                                    \n  #endif\n\n}",                           //frag5.h
                                                                    
                                                                "in vec3 ocolor2;\
                                                                 \nout vec4 fcolor2;\
                                                                 \n\lwoid main()\
                                                                 {\n #ifdef FORCE_INCLUDED\
                                                                  \n    fcolor2 = vec4(1.0, 0.0, 0.0, 1);\
                                                                  \n #else\
                                                                  \n    fcolor2 = vec4(0.0, 1.0, 0.0, 1);\
                                                                  \n #endif\n}",                                //frag6.h
                                                                  
                                                                "in vec3 ocolor2;\
                                                                 \nout vec4 fcolor2;\
                                                                 \n\lwoid main()\
                                                                 {\n #ifdef FORCE_INCLUDED\
                                                                  \n    fcolor2 = vec4(0.0, 1.0, 0.0, 1);\
                                                                  \n #else\
                                                                  \n    fcolor2 = vec4(1.0, 0.0, 0.0, 1);\
                                                                  \n #endif\n}",                                //frag7.h
                                                                    
                                                                "in vec3 ocolor2;\
                                                                 \nout vec4 fcolor2;\
                                                                 \n\n// Ꮚ Ᏹ Ꮙ Ꭿ \n// 吉岡康博\
                                                                 \n\lwoid main()\
                                                                 {\n  fcolor2 = vec4(0.0, 1.0, 0.0, 1);\n}",   //frag8.h
                                                                 
                                                                "#include \"macro_define.h\"\
                                                                 \n\nin vec3 ocolor2;\
                                                                 \nout vec4 fcolor2;\
                                                                 \lwoid main() \
                                                                 {\n\n  #if defined RELATIVE_FORCE_INCLUDED && defined MACRO_IN_INCLUDED_HEADER\
                                                                 \n     fcolor2 = vec4(0.0, 1.0, 0.0, 1);\
                                                                 \n #else\
                                                                 \n     fcolor2 = vec4(1.0, 0.0, 0.0, 1);\
                                                                 \n #endif\n}",                                 //frag13.h
                                                                 
                                                                "#include \"macro_define.h\"\
                                                                 \n\nin vec3 ocolor2;\
                                                                 \nout vec4 fcolor2;\
                                                                 \lwoid main() \
                                                                 {\n\n  #ifdef RELATIVE_FORCE_INCLUDED || #ifdef MACRO_IN_INCLUDED_HEADER\
                                                                 \n     fcolor2 = vec4(0.0, 1.0, 0.0, 1);\
                                                                 \n #else\
                                                                 \n     fcolor2 = vec4(1.0, 0.0, 0.0, 1);\
                                                                 \n #endif\n}",                                 //frag14.h
                                                                 
                                                                "layout(location=0) in vec3 position;\
                                                                 \nlayout(location=1) in vec3 color;\
                                                                 \nout vec3 ocolor;\
                                                                 \lwoid main()\
                                                                 {\n    gl_Position = vec4(position, 5.0);\
                                                                 \n  ocolor = color;\n}",                       //vert.h
                                                                 
                                                                "#define MACRO_IN_INCLUDED_HEADER 1"        //macro_header.h
                                        };

    //Get CWD 
    char *lwrwd = _getcwd( NULL, 0 );
    std::string cwd = lwrwd;
    free(lwrwd);
                                        
    // See if old folder structure has been left around
    if( ! (ILWALID_FILE_ATTRIBUTES == GetFileAttributes((cwd + include_path).c_str())) && ! (GetLastError() == ERROR_FILE_NOT_FOUND))
        {
            //Go ahead and delete the directory if it is there
            if ( ! DeleteDirectory((cwd + include_path).c_str())) 
            {
                printf("\nCould not remove the directory: %s\n", (cwd + include_path).c_str());
                return cwd;
            }
        }
        
        // Create folder hierarchy 
        if ( ! CreateDirectory((cwd + include_path).c_str(), NULL))
            printf("\nCoudn't create: %s", (cwd + include_path).c_str());          
        if ( ! CreateDirectory((cwd + include_path + "base_includes").c_str(), NULL))
            printf("\nCoudn't create: %s", (cwd + include_path + "base_includes").c_str());
        if ( ! CreateDirectory((cwd + include_path + "base_includes\\sub_includes").c_str(), NULL))
            printf("\nCoudn't create: %s", (cwd + include_path + "base_includes\\sub_includes").c_str());
        if ( ! CreateDirectory((cwd + include_path + "absolute_includes").c_str(), NULL))
            printf("\nCoudn't create: %s", (cwd + include_path + "absolute_includes").c_str());
        if ( ! CreateDirectory((cwd + include_path + "forced_includes").c_str(), NULL))
            printf("\nCoudn't create: %s", (cwd + include_path + "forced_includes").c_str());
        
        // Write out header files to disk 
        for (int x = 0; x < header_count; x++)
        {   
            os_file.open( (cwd + header_name[x]).c_str() );
            if(! os_file.fail())
            {   
                os_file << header_content[x];
                os_file.close();
            }
            else
                printf("\nCould not create file: %s", (cwd + header_name[x]).c_str());
        }
    
    return cwd;
}

lwString LWNGlslcIncludeTest::getDescription() const
{
    lwStringBuf sb;
    sb << "Test for adding include paths and forced header files when using the GLSLC Dll."
          "\nThe test will create a small directory structure and write out header files to be used at runtime and removed on exit."
          "\nThese files when found at runtime via the new features will produce a green square to be output corresponding to each test case."
          "\nAny failing test case will draw out a red square. Green=good, Red=a failed test case."
          "\nThe resulting image should have a top row of 7 green squares and a bottom row of 6 green squares.";
    return sb.str();    
}

int LWNGlslcIncludeTest::isSupported() const
{
    return lwogCheckLWNAPIVersion(5, 0);
}

//The tests begin here...
void LWNGlslcIncludeTest::doGraphics() const
{
    DeviceState *deviceState = DeviceState::GetActive();
    Device *device = deviceState->getDevice();
    QueueCommandBuffer &queueCB = deviceState->getQueueCB();
    Queue *queue = deviceState->getQueue();

    // Disable binary cacheing for #include tests.  The problem is we can't cache files that will be
    // #included inside shaders properly.
    LWNboolean cacheReadEnabled = g_glslcHelper->GetAllowCacheRead();
    LWNboolean cacheWriteEnabled = g_glslcHelper->GetAllowCacheWrite();
    g_glslcHelper->SetAllowCacheRead(false);
    g_glslcHelper->SetAllowCacheWrite(false);

    // Set up an include path that isn't fully fixed, allowing the same test
    // to be run in multiple processes or threads conlwrrently.
    std::string include_path = std::string("\\lwogtest_include_path_") + std::to_string(lwogGetTimerValue()) + std::string("\\");

    //Write out headers and get CWD
    std::string cwd = writeHeaders(include_path); 
    
    //Set up the AddIncludePath folder
    std::string include_folder = cwd + include_path + "base_includes";
    //Set up the forced include file
    std::string forced_include = cwd + include_path + "forced_includes\\force.h"; 

    static const Vertex vertexData[] = {     
        //test 1 
        { dt::vec3(-5.0, +0.5, 0.0), dt::vec3(0.0, 0.0, 1.0) },
        { dt::vec3(-5.0, +1.5, 0.0), dt::vec3(0.0, 1.0, 0.0) },
        { dt::vec3(-4.0, +0.5, 0.0), dt::vec3(1.0, 0.0, 0.0) },
        { dt::vec3(-4.0, +0.5, 0.0), dt::vec3(1.0, 0.0, 0.0) },
        { dt::vec3(-5.0, +1.5, 0.0), dt::vec3(1.0, 0.0, 0.0) },
        { dt::vec3(-4.0, +1.5, 0.0), dt::vec3(1.0, 0.0, 0.0) },
        //test 2
        { dt::vec3(-3.50, +0.5, 0.0), dt::vec3(0.0, 0.0, 1.0) },
        { dt::vec3(-3.50, +1.5, 0.0), dt::vec3(0.0, 1.0, 0.0) },
        { dt::vec3(-2.50, +0.5, 0.0), dt::vec3(1.0, 0.0, 0.0) },
        { dt::vec3(-2.50, +0.5, 0.0), dt::vec3(1.0, 0.0, 0.0) },
        { dt::vec3(-3.50, +1.5, 0.0), dt::vec3(1.0, 0.0, 0.0) },
        { dt::vec3(-2.50, +1.5, 0.0), dt::vec3(1.0, 0.0, 0.0) },        
        //test 3
        { dt::vec3(-2.00, +0.5, 0.0), dt::vec3(0.0, 0.0, 1.0) },
        { dt::vec3(-2.00, +1.5, 0.0), dt::vec3(0.0, 1.0, 0.0) },
        { dt::vec3(-1.00, +0.5, 0.0), dt::vec3(1.0, 0.0, 0.0) },
        { dt::vec3(-1.00, +0.5, 0.0), dt::vec3(1.0, 0.0, 0.0) },
        { dt::vec3(-2.00, +1.5, 0.0), dt::vec3(1.0, 0.0, 0.0) },
        { dt::vec3(-1.00, +1.5, 0.0), dt::vec3(1.0, 0.0, 0.0) },
        //test 4
        { dt::vec3(-0.50, +0.5, 0.0), dt::vec3(0.0, 0.0, 1.0) },
        { dt::vec3(-0.50, +1.5, 0.0), dt::vec3(0.0, 1.0, 0.0) },
        { dt::vec3(+0.50, +0.5, 0.0), dt::vec3(1.0, 0.0, 0.0) },
        { dt::vec3(+0.50, +0.5, 0.0), dt::vec3(1.0, 0.0, 0.0) },
        { dt::vec3(-0.50, +1.5, 0.0), dt::vec3(1.0, 0.0, 0.0) },
        { dt::vec3(+0.50, +1.5, 0.0), dt::vec3(1.0, 0.0, 0.0) },
        //test 5 
        { dt::vec3(+1.00, +0.5, 0.0), dt::vec3(0.0, 0.0, 1.0) },
        { dt::vec3(+1.00, +1.5, 0.0), dt::vec3(0.0, 1.0, 0.0) },
        { dt::vec3(+2.00, +0.5, 0.0), dt::vec3(1.0, 0.0, 0.0) },
        { dt::vec3(+2.00, +0.5, 0.0), dt::vec3(1.0, 0.0, 0.0) },
        { dt::vec3(+1.00, +1.5, 0.0), dt::vec3(1.0, 0.0, 0.0) },
        { dt::vec3(+2.00, +1.5, 0.0), dt::vec3(1.0, 0.0, 0.0) },
        //test 6
        { dt::vec3(+2.50, +0.5, 0.0), dt::vec3(0.0, 0.0, 1.0) },
        { dt::vec3(+2.50, +1.5, 0.0), dt::vec3(0.0, 1.0, 0.0) },
        { dt::vec3(+3.50, +0.5, 0.0), dt::vec3(1.0, 0.0, 0.0) },
        { dt::vec3(+3.50, +0.5, 0.0), dt::vec3(1.0, 0.0, 0.0) },
        { dt::vec3(+2.50, +1.5, 0.0), dt::vec3(1.0, 0.0, 0.0) },
        { dt::vec3(+3.50, +1.5, 0.0), dt::vec3(1.0, 0.0, 0.0) },            
        //test 7
        { dt::vec3(+4.00, +0.5, 0.0), dt::vec3(0.0, 0.0, 1.0) },
        { dt::vec3(+4.00, +1.5, 0.0), dt::vec3(0.0, 1.0, 0.0) },
        { dt::vec3(+5.00, +0.5, 0.0), dt::vec3(1.0, 0.0, 0.0) },
        { dt::vec3(+5.00, +0.5, 0.0), dt::vec3(1.0, 0.0, 0.0) },
        { dt::vec3(+4.00, +1.5, 0.0), dt::vec3(1.0, 0.0, 0.0) },
        { dt::vec3(+5.00, +1.5, 0.0), dt::vec3(1.0, 0.0, 0.0) },  
        //test 8
        { dt::vec3(-5.0, -0.5, 0.0), dt::vec3(0.0, 0.0, 1.0) },
        { dt::vec3(-5.0, -1.5, 0.0), dt::vec3(0.0, 1.0, 0.0) },
        { dt::vec3(-4.0, -0.5, 0.0), dt::vec3(1.0, 0.0, 0.0) },
        { dt::vec3(-4.0, -0.5, 0.0), dt::vec3(1.0, 0.0, 0.0) },
        { dt::vec3(-5.0, -1.5, 0.0), dt::vec3(1.0, 0.0, 0.0) },
        { dt::vec3(-4.0, -1.5, 0.0), dt::vec3(1.0, 0.0, 0.0) },
        //test 9
        { dt::vec3(-3.50, -0.5, 0.0), dt::vec3(0.0, 0.0, 1.0) },
        { dt::vec3(-3.50, -1.5, 0.0), dt::vec3(0.0, 1.0, 0.0) },
        { dt::vec3(-2.50, -0.5, 0.0), dt::vec3(1.0, 0.0, 0.0) },
        { dt::vec3(-2.50, -0.5, 0.0), dt::vec3(1.0, 0.0, 0.0) },
        { dt::vec3(-3.50, -1.5, 0.0), dt::vec3(1.0, 0.0, 0.0) },
        { dt::vec3(-2.50, -1.5, 0.0), dt::vec3(1.0, 0.0, 0.0) },        
        //test 10
        { dt::vec3(-2.00, -0.5, 0.0), dt::vec3(0.0, 0.0, 1.0) },
        { dt::vec3(-2.00, -1.5, 0.0), dt::vec3(0.0, 1.0, 0.0) },
        { dt::vec3(-1.00, -0.5, 0.0), dt::vec3(1.0, 0.0, 0.0) },
        { dt::vec3(-1.00, -0.5, 0.0), dt::vec3(1.0, 0.0, 0.0) },
        { dt::vec3(-2.00, -1.5, 0.0), dt::vec3(1.0, 0.0, 0.0) },
        { dt::vec3(-1.00, -1.5, 0.0), dt::vec3(1.0, 0.0, 0.0) },
        //test 11
        { dt::vec3(-0.50, -0.5, 0.0), dt::vec3(0.0, 0.0, 1.0) },
        { dt::vec3(-0.50, -1.5, 0.0), dt::vec3(0.0, 1.0, 0.0) },
        { dt::vec3(+0.50, -0.5, 0.0), dt::vec3(1.0, 0.0, 0.0) },
        { dt::vec3(+0.50, -0.5, 0.0), dt::vec3(1.0, 0.0, 0.0) },
        { dt::vec3(-0.50, -1.5, 0.0), dt::vec3(1.0, 0.0, 0.0) },
        { dt::vec3(+0.50, -1.5, 0.0), dt::vec3(1.0, 0.0, 0.0) },
        //test 12 
        { dt::vec3(+1.00, -0.5, 0.0), dt::vec3(0.0, 0.0, 1.0) },
        { dt::vec3(+1.00, -1.5, 0.0), dt::vec3(0.0, 1.0, 0.0) },
        { dt::vec3(+2.00, -0.5, 0.0), dt::vec3(1.0, 0.0, 0.0) },
        { dt::vec3(+2.00, -0.5, 0.0), dt::vec3(1.0, 0.0, 0.0) },
        { dt::vec3(+1.00, -1.5, 0.0), dt::vec3(1.0, 0.0, 0.0) },
        { dt::vec3(+2.00, -1.5, 0.0), dt::vec3(1.0, 0.0, 0.0) },
        //test 13
        { dt::vec3(+2.50, -0.5, 0.0), dt::vec3(0.0, 0.0, 1.0) },
        { dt::vec3(+2.50, -1.5, 0.0), dt::vec3(0.0, 1.0, 0.0) },
        { dt::vec3(+3.50, -0.5, 0.0), dt::vec3(1.0, 0.0, 0.0) },
        { dt::vec3(+3.50, -0.5, 0.0), dt::vec3(1.0, 0.0, 0.0) },
        { dt::vec3(+2.50, -1.5, 0.0), dt::vec3(1.0, 0.0, 0.0) },
        { dt::vec3(+3.50, -1.5, 0.0), dt::vec3(1.0, 0.0, 0.0) },            
        //test 14 : lwrrently unused
        { dt::vec3(+4.00, -0.5, 0.0), dt::vec3(0.0, 0.0, 1.0) },
        { dt::vec3(+4.00, -1.5, 0.0), dt::vec3(0.0, 1.0, 0.0) },
        { dt::vec3(+5.00, -0.5, 0.0), dt::vec3(1.0, 0.0, 0.0) },
        { dt::vec3(+5.00, -0.5, 0.0), dt::vec3(1.0, 0.0, 0.0) },
        { dt::vec3(+4.00, -1.5, 0.0), dt::vec3(1.0, 0.0, 0.0) },
        { dt::vec3(+5.00, -1.5, 0.0), dt::vec3(1.0, 0.0, 0.0) },   
    };

    VertexShader vs(440);
    vs <<
    "#include <vert.h>\n\n";

    // Add the Include Path to be used 
    g_glslcHelper->AddIncludePath(include_folder.c_str());
    
    // allocator will create pool at first allocation
    MemoryPoolAllocator allocator(device, NULL, ((12 * total_tests) * sizeof(vertexData)), LWN_MEMORY_POOL_TYPE_CPU_COHERENT);   
    
    VertexStream stream(sizeof(Vertex));
    LWN_VERTEX_STREAM_ADD_MEMBER(stream, Vertex, position);
    LWN_VERTEX_STREAM_ADD_MEMBER(stream, Vertex, color);    
    VertexArrayState vertex = stream.CreateVertexArrayState();
    Buffer *vbo = stream.AllocateVertexBuffer(device, (12 * total_tests), allocator, vertexData);
    BufferAddress vboAddr = vbo->GetAddress();   
    
    queueCB.ClearColor(0, 0.4, 0.0, 0.0, 0.0);
    
    //Run the tests
    //descriptions detailed above in void LWNGlslcIncludeTest::process_shader_test
    for (int test_num = 1; test_num <= total_tests; test_num++)
        {
            //once we have tested the forced standard include file for failure (test #6)
            //need to add the forced include file and proceed
            if(test_num == 7)
                {
                        //Set the forced standard include file
                        g_glslcHelper->AddforceIncludeStdHeader(forced_include.c_str());
                }
                
            process_shader_test(device, g_glslcHelper, queueCB, vboAddr, vertex, vs, sizeof(vertexData), test_num, include_path); 
        }
    
    queueCB.submit();

    //Tests finished
    
    // we need to make sure everything is done because
    // the MemoryAllocator destructor will free its memory
    // pool immediately w/o sync.
    queue->Finish();

    //Delete the directory structure on exit
    if ( ! DeleteDirectory((cwd + include_path).c_str())) 
        {
            printf("\nCould not remove the directory: %s\n", (cwd + include_path).c_str());
        }   
    
    // Reset to use the cache.
    g_glslcHelper->SetAllowCacheRead(cacheReadEnabled);
    g_glslcHelper->SetAllowCacheWrite(cacheWriteEnabled);
}

OGTEST_CppTest(LWNGlslcIncludeTest, lwn_glslc_include, );

#endif
