/*
 * LWIDIA_COPYRIGHT_BEGIN
 *
 * Copyright 2014-2019 by LWPU Corporation. All rights reserved. All information
 * contained herein is proprietary and confidential to LWPU Corporation. Any
 * use, reproduction, or disclosure without the written permission of LWPU
 * Corporation is prohibited.
 *
 * LWIDIA_COPYRIGHT_END
 */

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <sstream>
#include <vector>

#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>
#include <boost/smart_ptr.hpp>

#include "align.h"
#include "backinsertiterator.h"
#include "preproc.h"
#include "lwdiagutils.h"
#include "core/include/types.h"

#include "jsapi.h"
#include "jsdbgapi.h"
#include "jsobj.h"
#include "jsscript.h"
#include "jsscriptio.h"
#include "objectidsmapper.h"
#include "validatekeyarg.h"

using namespace boost::system;
using namespace boost::xpressive;

namespace po = boost::program_options;
namespace fs = boost::filesystem;
namespace sy = boost::system;

static JSClass globalClass =
{
    "Global",
    0,
    JS_PropertyStub,
    JS_PropertyStub,
    JS_PropertyStub,
    JS_PropertyStub,
    JS_EnumerateStub,
    JS_ResolveStub,
    JS_ColwertStub,
    JS_FinalizeStub
};

void ErrorReporter(JSContext *cx, const char *message, JSErrorReport *report)
{
    printf("%s at line %u\n", message, report->lineno);
}

void PrintUsage()
{
    printf("Usage: jsc [option]... [--input=]input-file\n");
    printf("  -o, --output=FILE             save to FILE\n");
    printf("  -b, --generate-boundjs        generate a C header file with embedded JavaScript byte code\n");
    printf("  -E, --preprocess              generate a preprocesses file\n");
    printf("  -I, --include-path=PATH       include directory for preprocessing the input file\n");
    printf("  -Dmacro[=def]                 define a macro for preprocessor\n");
}

class ObjectHookSet
{
public:
    ObjectHookSet(
        boost::shared_ptr<ObjectIdsMapper> idsMapper,
        boost::shared_ptr<JSRuntime> rt
    )
      : m_idsMapper(idsMapper)
      , m_jsRt(rt)
    {
        JS_SetObjectHook(m_jsRt.get(), ObjectIdsMapper::ObjectIdsMapperCallback, m_idsMapper.get());
    }

    ~ObjectHookSet()
    {
        JS_SetObjectHook(m_jsRt.get(), nullptr, nullptr);
    }

private:
    boost::shared_ptr<ObjectIdsMapper> m_idsMapper;
    boost::shared_ptr<JSRuntime> m_jsRt;
};

int main(int argc, char* argv[])
{
    po::positional_options_description positional;
    po::options_description cmdLineOptions;
    po::variables_map vm;

    std::string                inFileName;
    std::string                outFileName;

    bool generateBoundJs = false;
    bool generatePreprocessed = false;

    bool boundJs = false;

    cmdLineOptions.add_options()
        (
            "input",
            po::value<std::string>(&inFileName)->required(),
            "input file")
        (
            "output,o",
            po::value<std::string>(&outFileName)->required(),
            "output file"
        )
        (
            "generate-boundjs,b",
            "generate a single JavaScript file suitable for bound JS"
        )
        (
            "preprocess,E",
            "generate a preprocessed file"
        )
        (
            "include-path,I",
            po::value<std::vector<std::string> >(),
            "include path"
        )
        (
            "define,D",
            po::value<std::vector<std::string> >(),
            "define preprocessor macro"
        )
      ;
    positional.add("input", 1);

    try
    {
        po::store(po::command_line_parser(argc, argv).
                  positional(positional).
                  options(cmdLineOptions).run(), vm);
        po::notify(vm);
    }
    catch (po::error &x)
    {
        fprintf(stderr, "Command line error: %s\n", x.what());
        PrintUsage();
        return 1;
    }

    if (!fs::exists(inFileName))
    {
        fprintf(stderr, "Input file %s doesn't exist\n", inFileName.c_str());
        return 1;
    }

    std::vector<std::string> includePaths;
    if (vm.count("include-path"))
    {
        includePaths = vm["include-path"].as<std::vector<std::string> >();
    }

    std::vector<std::string> preprocDefines;
    if (vm.count("define"))
    {
        preprocDefines = vm["define"].as<std::vector<std::string> >();
    }

    generateBoundJs = 0 != vm.count("generate-boundjs");
    generatePreprocessed = 0 != vm.count("preprocess");
    if (generateBoundJs)
    {
        boundJs = true;
    }

    LwDiagUtils::Preprocessor preproc;

    fs::path scriptPath = fs::path(inFileName).parent_path();
    if (scriptPath == "")
    {
        scriptPath = ".";
    }
    preproc.AddSearchPath(scriptPath.string());

    fs::path programPath = fs::path(argv[0]).parent_path();
    if (programPath == "")
    {
        programPath = ".";
    }
    preproc.AddSearchPath(programPath.string());

    for (const auto &inc : includePaths)
    {
        preproc.AddSearchPath(inc);
    }

    for (const auto &def : preprocDefines)
    {
        std::string::size_type eqPos;
        if (std::string::npos != (eqPos = def.find('=')))
        {
            preproc.AddMacro(def.substr(0, eqPos).c_str(), def.substr(eqPos + 1).c_str());
        }
        else
        {
            preproc.AddMacro(def.c_str(), "");
        }
    }

    preproc.SetLineCommandMode(LwDiagUtils::Preprocessor::LineCommandAt);

    LwDiagUtils::EC ec = preproc.LoadFile(inFileName);
    if (ec != LwDiagUtils::OK)
       return LwDiagUtils::PREPROCESS_ERROR;

    if (boundJs)
    {
        preproc.AddMacro("BOUND_JS", "true");
    }

    std::vector<char> preprocessedBuffer;
    ec = preproc.Process(&preprocessedBuffer);
    if (ec != LwDiagUtils::OK)
       return LwDiagUtils::PREPROCESS_ERROR;

    try
    {
        if (generatePreprocessed)
        {
            boost::shared_ptr<FILE> preprocessedTextFile(
                fopen((inFileName + ".i").c_str(), "w"), fclose
            );
            size_t written  = fwrite(&preprocessedBuffer[0], 1, preprocessedBuffer.size(), preprocessedTextFile.get());
            if (preprocessedBuffer.size() != written)
            {
                throw sy::system_error(errno, sy::system_category());
            }
        }

        boost::shared_ptr<JSRuntime> jsRuntime(
            JS_NewRuntime(8 * 1024 * 1024), JS_DestroyRuntime
        );
        boost::shared_ptr<JSContext> jsContext(
            JS_NewContext(jsRuntime.get(), 8 * 1024, nullptr, nullptr, nullptr, nullptr), JS_DestroyContext
        );
        JS_SetVersion(jsContext.get(), JSVERSION_1_7);

        boost::shared_ptr<ObjectIdsMapper> objMap(new ObjectIdsMapper);
        ObjectHookSet hookSetter(objMap, jsRuntime);

        JSObject *jsObject = JS_NewObject(jsContext.get(), &globalClass, nullptr, nullptr);

        if (!JS_InitStandardClasses(jsContext.get(), jsObject)) return 1;
        JS_SetErrorReporter(jsContext.get(), ErrorReporter);

        // will be destroyed by the GC
        JSScript *jsScript = JS_CompileScript(jsContext.get(), jsObject, &preprocessedBuffer[0], preprocessedBuffer.size(), "none", 1);
        if (nullptr == jsScript)
        {
            return 1;
        }

        std::vector<UINT08> encryptedScript;
        BackInsertIterator oit(encryptedScript);
        BitsIO::OutBitStream<BackInsertIterator> strm(oit);

        typedef JSScriptSaver<BackInsertIterator, ObjectIdsMapper> ScriptSaver;
        boost::scoped_ptr<ScriptSaver> oa(
            new ScriptSaver(strm, jsContext.get(), objMap.get())
        );
        oa->Save(jsScript);
        oa.reset();

        boost::shared_ptr<FILE> of(fopen(outFileName.c_str(), "wb"), fclose);
        if (!of) throw sy::system_error(errno, sy::system_category());

        size_t written = fwrite(
            &encryptedScript[0],
            sizeof(encryptedScript[0]),
            encryptedScript.size(),
            of.get()
        );
        if (encryptedScript.size() != written)
        {
            throw sy::system_error(errno, sy::system_category());
        }
    }
    catch (sy::system_error &x)
    {
        fprintf(stderr, "%s\n", x.what());
        return 1;
    }

    return 0;
}
