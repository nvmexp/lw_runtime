/*
 * LWIDIA_COPYRIGHT_BEGIN
 *
 * Copyright 2008-2011, 2013, 2015-2021 by LWPU Corporation. All rights
 * reserved. All information contained herein is proprietary and confidential to
 * LWPU Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * LWIDIA_COPYRIGHT_END
 */

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>

#include <boost/program_options.hpp>

#include "core/include/version.h"
#include "core/include/types.h"

#include "lwdiagutils.h"
#include "preproc.h"

#include "encryption.h"
#ifdef ENCRYPTOR
#include "encrypt.h"
#else
#include "decrypt_log.h"
#endif
#include "decrypt.h"

namespace po = boost::program_options;

//! Remember the program path for later
static string s_ProgramPath;

#if defined(ENCRYPTOR)
// A functional object to serve as a callback for libencryption when it
// preprocesses files before encryption. It's a class instead of just a
// function to avoid passing parameters via global variables as it was in the
// previous version.
class PreprocessFile
{
public:
    PreprocessFile() = default;

    PreprocessFile(bool defineBoundJs)
      : m_DefineBoundJs(defineBoundJs)
    {}

    bool GetDefineBoundJs() const
    {
        return m_DefineBoundJs;
    }

    void SetDefineBoundJs(bool val)
    {
        m_DefineBoundJs = val;
    }

    //! \brief Preprocess a file (in C preprocessor fashion) - based upon the
    //!        code in core/utility/preprocess.cpp - which is not compiled here
    //!        to avoid including many more MODS specific functions
    //!
    //! \param input               : Input file name to preprocess
    //! \param pPreprocessedBuffer : Output buffer that the input file is
    //!                              preprocessed into
    //! \param additionalPaths     : Additional search paths for the file
    //! \param numPaths            : Number of entries in the additional paths
    //! \param preprocDefs         : Preprocessor definitions
    //! \param numDefs             : Number of entries in the preprocessor definitions
    //!
    //! \return OK if successful
    LwDiagUtils::EC operator()(
        const char     *input,
        vector<UINT08> *pPreprocessedBuffer,
        char          **additionalPaths,
        UINT32          numPaths,
        char          **preprocDefs,
        UINT32          numDefs
    )
    {
        LWDASSERT(pPreprocessedBuffer != 0);

        //
        // Initialize the preprocessor.
        //
        LwDiagUtils::Preprocessor preproc;

        preproc.SetDecryptFile(Decryptor::DecryptFile);

        //
        // Specify the path of the #include directories.
        //
        //    Path1: Path of script file.
        //
        string scriptPath(input);

        string::size_type pos = scriptPath.find_last_of("/\\");
        if (string::npos == pos)
        {
            scriptPath = ".";
        }
        else
        {
            scriptPath = scriptPath.substr(0, pos);
        }

        preproc.AddSearchPath(scriptPath);

        preproc.AddSearchPath(s_ProgramPath);

        if (additionalPaths && numPaths)
        {
            for (UINT32 p = 0; p < numPaths; ++p)
            {
                preproc.AddSearchPath(additionalPaths[p]);
            }
        }

        if (nullptr != preprocDefs && 0 != numDefs)
        {
            for (UINT32 d = 0; d < numDefs; ++d)
            {
                const char *def = preprocDefs[d];
                const char *eq = strchr(def, '=');
                if (nullptr == eq)
                {
                    preproc.AddMacro(def, "");
                }
                else
                {
                    string defName(def, eq);
                    preproc.AddMacro(defName.c_str(), eq + 1);
                }
            }
        }

        preproc.SetLineCommandMode(LwDiagUtils::Preprocessor::LineCommandAt);

        LwDiagUtils::EC ec = preproc.LoadFile(input);
        if (ec != LwDiagUtils::OK)
        {
            return ec;
        }

#if defined(INCLUDE_BOARDS_DB)
        // Pass INCLUDE_BOARDS_DB to JS preprocessor. boards.js will use it to
        // decide whether to include boards.db. Therefore boards.js is going to be
        // the only place where boards.db is either included or not.
        preproc.AddMacro("INCLUDE_BOARDS_DB", "");
#endif
        // Add a define for whether or not GL is included in this mods package.
        // Allows us to conditionally include glrandom.js at preprocess time.
        if (g_INCLUDE_OGL)
        {
            preproc.AddMacro("INCLUDE_OGL", "true");
        }
        if (g_INCLUDE_OGL_ES)
        {
            preproc.AddMacro("INCLUDE_OGL_ES", "true");
        }
        if (g_INCLUDE_VULKAN)
        {
            preproc.AddMacro("INCLUDE_VULKAN", "true");
        }
        if (g_INCLUDE_LWDA)
        {
            preproc.AddMacro("INCLUDE_LWDA", "true");
        }
        if (g_INCLUDE_LWDART)
        {
            preproc.AddMacro("INCLUDE_LWDART", "true");
        }
        if (m_DefineBoundJs)
        {
            preproc.AddMacro("BOUND_JS", "true");
        }

        if (g_INCLUDE_GPU)
        {
            preproc.AddMacro("INCLUDE_GPU", "true");
        }

        if (g_BUILD_TEGRA_EMBEDDED)
        {
            preproc.AddMacro("BUILD_TEGRA_EMBEDDED", "true");
        }

        // Ensure that buffer size is zero so that data insertion starts at the
        // beginning.
        pPreprocessedBuffer->resize(0);

        //
        // Preprocess the file.
        //

        ec = preproc.Process(reinterpret_cast<vector<char>*>(pPreprocessedBuffer));

        return ec;
    }

private:
    bool m_DefineBoundJs = false;
};
#endif

extern "C"
{
    INT32 EncryptVAPrintf
    (
       INT32        Priority,
       UINT32       ModuleCode,
       UINT32       Sps,
       const char * Format,
       va_list      RestOfArgs
    )
    {
        if (Priority > LwDiagUtils::PriLow)
            return vprintf(Format, RestOfArgs);
        return 0;
    }
} // extern "C"

class LwDiagUtilsInitializer
{
public:
    LwDiagUtilsInitializer() { LwDiagUtils::Initialize(nullptr, EncryptVAPrintf); }
    ~LwDiagUtilsInitializer() { LwDiagUtils::Shutdown(); }
};

void PrintUsage(string exename)
{
#if defined(ENCRYPTOR)
    LwDiagUtils::Printf(LwDiagUtils::PriNormal,
            "Usage 1 : encrypt -o<directory> file1.js [file2.h] [...]\n\n"
            "  Encrypt a list files (without C++ preprocessing) into corresponding\n"
            "  encrypted files suitable for inclusion into a package\n"
            "    -o<directory> Directory for output files to be placed\n\n\n"
            "Usage 2 : encrypt -b [-i<directory>] input.js output.h\n\n"
            "  Encrypt a single JS file (with C++ preprocessing) into a single header\n"
            "  file suitable for inclusion into a build\n"
            "    -b            Create bound JS header file in output.h containing the\n"
            "                  encrypted, preprocessed contents of input.js.\n"
            "    -i<directory> Add search directory for preprocessing files, only used\n"
            "                  when \"-b\" is also present and may be specified multiple\n"
            "                  times\n");
#else
    LwDiagUtils::Printf(LwDiagUtils::PriNormal,
      "Usage %s [-t] inputfile outputfile\n",
      exename.c_str());
#endif
}
#if defined(ENFORCE_LW_NETWORK)

    bool IsBypassPresent(string fname)
    {
        if (fname.empty() || !LwDiagXp::DoesFileExist(fname.c_str()))
            return false;

        LwDiagUtils::FileHolder bypassFile;
        if (LwDiagUtils::OK != bypassFile.Open(fname, "rb"))
            return false;

        long fileSize = 0;
        if (LwDiagUtils::OK != LwDiagUtils::FileSize(bypassFile.GetFile(), &fileSize))
            return false;

        const long BYPASS_FILE_SIZE = 16;
        if (fileSize != BYPASS_FILE_SIZE)
            return false;

        UINT32 bypassHash;
        UINT32 internalBypassHash;

        if (0 == fread(&bypassHash, 4, 1, bypassFile.GetFile()) ||
            0 == fread(&internalBypassHash, 4, 1, bypassFile.GetFile()))
        {
            return false;
        }
        bypassHash           = ~bypassHash;
        internalBypassHash   = ~internalBypassHash;

        if ((bypassHash == g_BypassHash) && (internalBypassHash == g_InternalBypassHash))
            return true;

        return false;
    }
#endif

#if defined(DECRYPTOR)
static constexpr unsigned lines    = 3;
static constexpr unsigned lineSize = 16;
extern const unsigned char m_6[lines * lineSize];

static void DumpLogKey()
{
    LwDiagUtils::Printf(LwDiagUtils::PriNormal, "Keys:\n");
    for (unsigned line = 0; line < lines; line++)
    {
        LwDiagUtils::Printf(LwDiagUtils::PriNormal, "   ");
        for (unsigned byte = 0; byte < lineSize; byte++)
        {
            const unsigned idx = byte + line * lineSize;
            LwDiagUtils::Printf(LwDiagUtils::PriNormal, " 0x%02x", m_6[idx]);
            if (idx + 1 < sizeof(m_6))
            {
                LwDiagUtils::Printf(LwDiagUtils::PriNormal, ",");
            }
        }
        LwDiagUtils::Printf(LwDiagUtils::PriNormal, "\n");
    }
}
#endif

int main(
   int    argc,
   char * argv[]
)
{
    LwDiagUtilsInitializer lwdiagUtilsInit;

    string bypassFile;
#if defined(DECRYPTOR)
    bool traceFile = false;
    UINT08 inputFileIdx = 1;
    UINT08 outputFileIdx = 2;

    // Purposely hide the fact that the decryptor can take an optional bypass
    // file at the end for now
    if ((argc > 5) || ((argc < 3) && (argc != 1)))
    {
        PrintUsage(argv[0]);
        return (int)LwDiagUtils::BAD_COMMAND_LINE_ARGUMENT;
    }
    if (argc > 1)
    {
        if (!strcmp(argv[1], "-t"))
        {
            traceFile = true;
            inputFileIdx = 2;
            outputFileIdx = 3;
        }

        if (traceFile && (argc < 4))
        {
            PrintUsage(argv[0]);
            return (int)LwDiagUtils::BAD_COMMAND_LINE_ARGUMENT;
        }

        if (((argc == 4) && !traceFile) || ((argc == 5) && traceFile))
            bypassFile = argv[argc - 1];
        else
            bypassFile = LwDiagXp::GetElw("MODS_DECRYPT_BYPASS");
    }
#endif

    // Set search paths before trying network bypass, because the certificate
    // file for the auth server may need to be found in program path
    const char* const exe = argv[0];
    s_ProgramPath = exe;
    const auto slashPos = s_ProgramPath.find_last_of("/\\");
    if (slashPos != string::npos)
    {
        s_ProgramPath.resize(slashPos + 1);
    }
    LwDiagUtils::AddExtendedSearchPath(s_ProgramPath);

#if defined(ENFORCE_LW_NETWORK)
    bool bOnLwidiaNetwork = false;
    const UINT32 networkRetries = 5;
    const bool bIsBypassPresent = IsBypassPresent(bypassFile);

    LwDiagUtils::EnableVerboseNetwork(bIsBypassPresent);

    // Do at least one check for on network even when the bypass is present
    // which allows network connection verbose printing
    for (UINT32 retry = 0; (retry < networkRetries) && !bOnLwidiaNetwork; retry++)
    {
        bOnLwidiaNetwork = LwDiagUtils::IsOnLwidiaIntranet() || bIsBypassPresent;
    }
    LwDiagUtils::NetShutdown();
    if (!bOnLwidiaNetwork)
    {
        LwDiagUtils::Printf(LwDiagUtils::PriError,
                            "File format not recognized.\n");
        return 1;
    }

    if (bOnLwidiaNetwork)
    {
        // If already on the lwpu network, print out the version and changelist
        // so that if necessary a bypass file can be generated that unlocks
        // the decryptor
        LwDiagUtils::Printf(LwDiagUtils::PriNormal,
                            "Version:    %s\n"
                            "Changelist: %u\n\n",
                            g_Version, g_Changelist);
    }
#endif

    if (argc == 1)
    {
        PrintUsage(argv[0]);
        return 0;
    }

    {
#if defined(ENCRYPTOR)
        bool bound = true;
        bool binary = false;
        bool traceFile = false;
        string inFileName;
        string outPath;
        vector<string> inFileNames;
        vector<string> includePaths;
        vector<string> preprocDefines;

        po::options_description incPathsAndDefs;
        incPathsAndDefs.add_options()
            (
                "include-path,i",
                po::value<vector<string> >(&includePaths),
                "include path"
            )
            (
                "define,D",
                po::value<vector<string> >(&preprocDefines),
                "define preprocessor macro"
            )
            (
                "trace-file,t",
                po::bool_switch(&traceFile),
                "encrypt with trace file key"
            )
          ;

        po::options_description optBoundUsage;
        try
        {
            optBoundUsage.add_options()
                (
                    "generate-boundjs,b",
                    po::value<bool>(&bound)->implicit_value(true),
                    "create bound JS header file"
                )
                (
                    "input",
                    po::value<std::string>(&inFileName)->required(),
                    "input file"
                )
                (
                    "output,o",
                    po::value<std::string>(&outPath)->required(),
                    "directory for output files"
                )
                (
                    "binary",
                    po::value<bool>(&binary)->implicit_value(true),
                    "input file is binary"
                )
              ;
        }
        catch (...)
        {
            LwDiagUtils::Printf(LwDiagUtils::PriError,
                                "Cannot parse command-line arguments\n");
            return 1;
        }
        optBoundUsage.add(incPathsAndDefs);

        po::positional_options_description posBoundUsage;
        posBoundUsage.add("input", 1).add("output", 1);

        po::options_description optNotBoundUsage;
        optNotBoundUsage.add_options()
            (
                "input",
                po::value<std::vector<std::string> >(&inFileNames)->required(),
                "input file")
            (
                "output,o",
                // We cannot make -o required because Buildmeister cannot handle
                // exceptions.
                po::value<std::string>(&outPath)/*->required()*/,
                "directory for output files"
            )
          ;
        optNotBoundUsage.add(incPathsAndDefs);

        po::positional_options_description posNotBoundUsage;
        posNotBoundUsage.add("input", -1);

        po::variables_map vm;

        // We can't handle errors of program options library because of
        // Buildmeister. In the old Linux system that Buildmeister uses the
        // catch clause doesn't catch. Maybe it's possible to make it work, but
        // it's unclear for me (vandrejev) how to debug on a Buildmeister
        // system or even verify a fix.
        bound = false;
        auto parsed = po::command_line_parser(argc, argv).
                      positional(posNotBoundUsage).
                      options(optNotBoundUsage).allow_unregistered().run();
        auto potentialBound = po::collect_unrecognized(parsed.options, po::exclude_positional);
        if (potentialBound.end() != find(potentialBound.begin(), potentialBound.end(), "-b"))
        {
            bound = true;
        }
        else
        {
            po::store(parsed, vm);
            po::notify(vm);
        }

        if (bound)
        {
            try
            {
                po::store(po::command_line_parser(argc, argv).
                          positional(posBoundUsage).
                          options(optBoundUsage).run(), vm);
                po::notify(vm);
            }
            // catch won't work on Buildmesiter machine because of the old
            // Linux, though it will print std::exception::what() on terminate.
            catch (po::error_with_option_name &x)
            {
                if ("--input" == x.get_option_name() || "--output" == x.get_option_name())
                {
                    LwDiagUtils::Printf(LwDiagUtils::PriNormal,
                        "%s: error: -b requires 2 files: <in.js> <out.h>\n",
                        exe);
                    return 1;
                }
                else
                {
                    LwDiagUtils::Printf(LwDiagUtils::PriError,
                        "Failed to parse option %s\n", x.get_option_name().c_str());
                    return 1;
                }
            }
            catch (po::error &)
            {
                PrintUsage(argv[0]);
                return 0;
            }
        }

        vector<char *> additionalPaths;
        transform(
            includePaths.begin(),
            includePaths.end(),
            back_inserter(additionalPaths),
            [](const string& incPath) { return const_cast<char*>(incPath.c_str()); }
        );
        vector<char *> preprocDefs;
        transform(
            preprocDefines.begin(),
            preprocDefines.end(),
            back_inserter(preprocDefs),
            [](const string& def) { return const_cast<char*>(def.c_str()); }
        );
        if (bound)
        {
            // Initialize the encryption library preprocessor callback
            Encryption::Initialize(PreprocessFile(true));

            string boundjs = inFileName;
            LwDiagUtils::Printf(
                LwDiagUtils::PriNormal,
                "Encrypting %s and binding to %s\n", inFileName.c_str(), outPath.c_str());

            const auto ec = Encryptor::EncryptFile(
                inFileName, outPath,
                additionalPaths.empty() ? nullptr : &additionalPaths[0],
                additionalPaths.size(),
                preprocDefs.empty() ? nullptr : &preprocDefs[0],
                preprocDefs.size(),
                true,
                !binary
            );
            if (ec)
            {
                return 1;
            }
        }
        else
        {
            string path = outPath.empty() ? "." : outPath;
            for (const auto &inFileName : inFileNames)
            {
                string out = LwDiagUtils::JoinPaths(
                    path, LwDiagUtils::StripDirectory(inFileName.c_str()));

                // Truncate extensions longer than 2 characters
                const bool extensionPresent =
                    LwDiagUtils::StripDirectory(inFileName.c_str()).rfind('.') != string::npos;
                if (extensionPresent)
                {
                    const size_t dot = out.rfind('.');
                    const size_t extlen = (dot != string::npos) ? (out.size() - dot - 1) : 0;
                    // Do not truncate JSON files, otherwise we will collide with
                    // JavaScript files (.jse)
                    if (extlen > 0 &&
                        (out.substr(out.find_last_of(".") + 1) != "json") &&
                        (out.substr(out.find_last_of(".") + 1) != "lz4"))
                    {
                        if (extlen > 2)
                        {
                            out.resize(out.size() - (extlen - 2));
                            LwDiagUtils::Printf(LwDiagUtils::PriWarn,
                                "Will truncate the file name to be \"%se\".\n",
                                out.c_str());
                        }
                    }
                }
                else
                {
                    out += '.';
                }

                // Append 'e'
                out += 'e';

                LwDiagUtils::Printf(LwDiagUtils::PriNormal,
                    "Encrypting %s to %s\n", inFileName.c_str(), out.c_str());

                LwDiagUtils::EC ec;
                if (traceFile)
                {
                    ec = Encryptor::EncryptTraceFile(inFileName, out.c_str());
                }
                else
                {
                    ec = Encryptor::EncryptFile(
                        inFileName,
                        out.c_str(),
                        additionalPaths.empty() ? &additionalPaths[0] : nullptr,
                        additionalPaths.size(),
                        preprocDefs.empty() ? nullptr : &preprocDefs[0],
                        preprocDefs.size(),
                        false,
                        false
                    );
                }
                if (ec)
                {
                    return 1;
                }
            }
        }
#else
        LwDiagUtils::Printf(LwDiagUtils::PriNormal,
                            "Decrypting %s to %s\n", argv[inputFileIdx], argv[outputFileIdx]);

        LwDiagUtils::FileHolder inFile;
        if (inFile.Open(argv[inputFileIdx], "rb") != LwDiagUtils::OK)
        {
            LwDiagUtils::Printf(LwDiagUtils::PriError,
                                "Failed to open file %s\n", argv[inputFileIdx]);
            return 1;
        }

        vector<UINT08> output;

        const auto encryptType = LwDiagUtils::GetFileEncryption(inFile.GetFile());

#if defined(DECRYPT_LOG_ONLY)
        const LwDiagUtils::EC ec = Decryptor::DecryptLog(inFile.GetFile(), &output);
#else
        // For now, only all non-log file decryption if the bypass file is present
        // No one outside of the MODS group should be doing this in any case
        if (!bIsBypassPresent)
        {
            LwDiagUtils::Printf(LwDiagUtils::PriError,
                                "File format not recognized.\n");
            return 1;
        }

        LwDiagUtils::EC ec;
        if (encryptType == LwDiagUtils::ENCRYPTED_FILE_V3)
        {
            if (traceFile)
            {
                ec = Decryptor::DecryptTraceFile(inFile.GetFile(), &output);
            }
            else
            {
                ec = Decryptor::DecryptFile(inFile.GetFile(), &output);
            }
        }
        else if (encryptType == LwDiagUtils::ENCRYPTED_LOG_V3 ||
                 encryptType == LwDiagUtils::NOT_ENCRYPTED)
        {
            ec = Decryptor::DecryptLog(inFile.GetFile(), &output);
        }
        else
        {
            LwDiagUtils::Printf(LwDiagUtils::PriError, "File format not recognized.\n");
            ec = LwDiagUtils::ILWALID_FILE_FORMAT;
        }
#endif
        if (encryptType == LwDiagUtils::NOT_ENCRYPTED && !ec)
        {
            LwDiagUtils::Printf(LwDiagUtils::PriNormal, "File %s is not encrypted\n",
                                argv[outputFileIdx]);
        }

        if (ec)
        {
            if (ec == LwDiagUtils::DECRYPTION_ERROR)
            {
                DumpLogKey();
            }
            LwDiagUtils::Printf(LwDiagUtils::PriError,
                                "Failed to decrypt file %s\n", argv[outputFileIdx]);
            return 1;
        }

        LwDiagUtils::FileHolder outFile;
        if (outFile.Open(argv[outputFileIdx], "wb") != LwDiagUtils::OK)
        {
            LwDiagUtils::Printf(LwDiagUtils::PriError,
                                "Failed to open file %s\n", argv[outputFileIdx]);
            return 1;
        }

        const auto numWritten = fwrite(&output[0], 1, output.size(), outFile.GetFile());
        if (numWritten != output.size())
        {
            LwDiagUtils::Printf(LwDiagUtils::PriError,
                                "Failed to write to file %s\n", argv[outputFileIdx]);
            return 1;
        }
#endif
    }

    return 0;
}
