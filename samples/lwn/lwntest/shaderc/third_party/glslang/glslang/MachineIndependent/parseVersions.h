//
// Copyright (C) 2015-2018 Google, Inc.
// Copyright (C) 2017 ARM Limited.
//
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//
//    Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//
//    Redistributions in binary form must reproduce the above
//    copyright notice, this list of conditions and the following
//    disclaimer in the documentation and/or other materials provided
//    with the distribution.
//
//    Neither the name of 3Dlabs Inc. Ltd. nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
// FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
// COPYRIGHT HOLDERS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
// INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
// BUT NOT LIMITED TO, PROLWREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
// LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
// ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//

// This is implemented in Versions.cpp

#ifndef _PARSE_VERSIONS_INCLUDED_
#define _PARSE_VERSIONS_INCLUDED_

#include "../Public/ShaderLang.h"
#include "../Include/InfoSink.h"
#include "Scan.h"

#include <map>

namespace glslang {

//
// Base class for parse helpers.
// This just has version-related information and checking.
// This class should be sufficient for preprocessing.
//
class TParseVersions {
public:
    TParseVersions(TIntermediate& interm, int version, EProfile profile,
                   const SpvVersion& spvVersion, EShLanguage language, TInfoSink& infoSink,
                   bool forwardCompatible, EShMessages messages)
        :
#ifndef GLSLANG_WEB
        forwardCompatible(forwardCompatible),
        profile(profile),
#endif
        infoSink(infoSink), version(version), 
        language(language),
        spvVersion(spvVersion), 
        intermediate(interm), messages(messages), numErrors(0), lwrrentScanner(0) { }
    virtual ~TParseVersions() { }
    void requireStage(const TSourceLoc&, EShLanguageMask, const char* featureDesc);
    void requireStage(const TSourceLoc&, EShLanguage, const char* featureDesc);
#ifdef GLSLANG_WEB
    const EProfile profile = EEsProfile;
    bool isEsProfile() const { return true; }
    void requireProfile(const TSourceLoc& loc, int profileMask, const char* featureDesc)
    {
        if (! (EEsProfile & profileMask))
            error(loc, "not supported with this profile:", featureDesc, ProfileName(profile));
    }
    void profileRequires(const TSourceLoc& loc, int profileMask, int milwersion, int numExtensions,
        const char* const extensions[], const char* featureDesc)
    {
        if ((EEsProfile & profileMask) && (milwersion == 0 || version < milwersion))
            error(loc, "not supported for this version or the enabled extensions", featureDesc, "");
    }
    void profileRequires(const TSourceLoc& loc, int profileMask, int milwersion, const char* extension,
        const char* featureDesc)
    {
        profileRequires(loc, profileMask, milwersion, extension ? 1 : 0, &extension, featureDesc);
    }
    void initializeExtensionBehavior() { }
    void checkDeprecated(const TSourceLoc&, int queryProfiles, int depVersion, const char* featureDesc) { }
    void requireNotRemoved(const TSourceLoc&, int queryProfiles, int removedVersion, const char* featureDesc) { }
    void requireExtensions(const TSourceLoc&, int numExtensions, const char* const extensions[],
        const char* featureDesc) { }
    void ppRequireExtensions(const TSourceLoc&, int numExtensions, const char* const extensions[],
        const char* featureDesc) { }
    TExtensionBehavior getExtensionBehavior(const char*) { return EBhMissing; }
    bool extensionTurnedOn(const char* const extension) { return false; }
    bool extensionsTurnedOn(int numExtensions, const char* const extensions[]) { return false; }
    void updateExtensionBehavior(int line, const char* const extension, const char* behavior) { }
    void updateExtensionBehavior(const char* const extension, TExtensionBehavior) { }
    void checkExtensionStage(const TSourceLoc&, const char* const extension) { }
    void fullIntegerCheck(const TSourceLoc&, const char* op) { }
    void doubleCheck(const TSourceLoc&, const char* op) { }
    bool float16Arithmetic() { return false; }
    void requireFloat16Arithmetic(const TSourceLoc& loc, const char* op, const char* featureDesc) { }
    bool int16Arithmetic() { return false; }
    void requireInt16Arithmetic(const TSourceLoc& loc, const char* op, const char* featureDesc) { }
    bool int8Arithmetic() { return false; }
    void requireInt8Arithmetic(const TSourceLoc& loc, const char* op, const char* featureDesc) { }
    void int64Check(const TSourceLoc&, const char* op, bool builtIn = false) { }
    void explicitFloat32Check(const TSourceLoc&, const char* op, bool builtIn = false) { }
    void explicitFloat64Check(const TSourceLoc&, const char* op, bool builtIn = false) { }
    bool relaxedErrors()    const { return false; }
    bool suppressWarnings() const { return true; }
    bool isForwardCompatible() const { return false; }
#else
    bool forwardCompatible;      // true if errors are to be given for use of deprecated features
    EProfile profile;            // the declared profile in the shader (core by default)
    bool isEsProfile() const { return profile == EEsProfile; }
    void requireProfile(const TSourceLoc& loc, int profileMask, const char* featureDesc);
    void profileRequires(const TSourceLoc& loc, int profileMask, int milwersion, int numExtensions,
        const char* const extensions[], const char* featureDesc);
    void profileRequires(const TSourceLoc& loc, int profileMask, int milwersion, const char* extension,
        const char* featureDesc);
    virtual void initializeExtensionBehavior();
    virtual void checkDeprecated(const TSourceLoc&, int queryProfiles, int depVersion, const char* featureDesc);
    virtual void requireNotRemoved(const TSourceLoc&, int queryProfiles, int removedVersion, const char* featureDesc);
    virtual void requireExtensions(const TSourceLoc&, int numExtensions, const char* const extensions[],
        const char* featureDesc);
    virtual void ppRequireExtensions(const TSourceLoc&, int numExtensions, const char* const extensions[],
        const char* featureDesc);
    virtual TExtensionBehavior getExtensionBehavior(const char*);
    virtual bool extensionTurnedOn(const char* const extension);
    virtual bool extensionsTurnedOn(int numExtensions, const char* const extensions[]);
    virtual void updateExtensionBehavior(int line, const char* const extension, const char* behavior);
    virtual void updateExtensionBehavior(const char* const extension, TExtensionBehavior);
    virtual bool checkExtensionsRequested(const TSourceLoc&, int numExtensions, const char* const extensions[],
        const char* featureDesc);
    virtual void checkExtensionStage(const TSourceLoc&, const char* const extension);
    virtual void fullIntegerCheck(const TSourceLoc&, const char* op);

    virtual void unimplemented(const TSourceLoc&, const char* featureDesc);
    virtual void doubleCheck(const TSourceLoc&, const char* op);
    virtual void float16Check(const TSourceLoc&, const char* op, bool builtIn = false);
    virtual void float16ScalarVectorCheck(const TSourceLoc&, const char* op, bool builtIn = false);
    virtual bool float16Arithmetic();
    virtual void requireFloat16Arithmetic(const TSourceLoc& loc, const char* op, const char* featureDesc);
    virtual void int16ScalarVectorCheck(const TSourceLoc&, const char* op, bool builtIn = false);
    virtual bool int16Arithmetic();
    virtual void requireInt16Arithmetic(const TSourceLoc& loc, const char* op, const char* featureDesc);
    virtual void int8ScalarVectorCheck(const TSourceLoc&, const char* op, bool builtIn = false);
    virtual bool int8Arithmetic();
    virtual void requireInt8Arithmetic(const TSourceLoc& loc, const char* op, const char* featureDesc);
    virtual void float16OpaqueCheck(const TSourceLoc&, const char* op, bool builtIn = false);
    virtual void int64Check(const TSourceLoc&, const char* op, bool builtIn = false);
    virtual void explicitInt8Check(const TSourceLoc&, const char* op, bool builtIn = false);
    virtual void explicitInt16Check(const TSourceLoc&, const char* op, bool builtIn = false);
    virtual void explicitInt32Check(const TSourceLoc&, const char* op, bool builtIn = false);
    virtual void explicitFloat32Check(const TSourceLoc&, const char* op, bool builtIn = false);
    virtual void explicitFloat64Check(const TSourceLoc&, const char* op, bool builtIn = false);
    virtual void fcoopmatCheck(const TSourceLoc&, const char* op, bool builtIn = false);
    virtual void intcoopmatCheck(const TSourceLoc&, const char *op, bool builtIn = false);
    bool relaxedErrors()    const { return (messages & EShMsgRelaxedErrors) != 0; }
    bool suppressWarnings() const { return (messages & EShMsgSuppressWarnings) != 0; }
    bool isForwardCompatible() const { return forwardCompatible; }
#endif // GLSLANG_WEB
    virtual void spvRemoved(const TSourceLoc&, const char* op);
    virtual void vulkanRemoved(const TSourceLoc&, const char* op);
    virtual void requireVulkan(const TSourceLoc&, const char* op);
    virtual void requireSpv(const TSourceLoc&, const char* op);


#if defined(GLSLANG_WEB) && !defined(GLSLANG_WEB_DEVEL)
    void C_DECL   error(const TSourceLoc&, const char* szReason, const char* szToken,
                        const char* szExtraInfoFormat, ...) { addError(); }
    void C_DECL    warn(const TSourceLoc&, const char* szReason, const char* szToken,
                        const char* szExtraInfoFormat, ...) { }
    void C_DECL ppError(const TSourceLoc&, const char* szReason, const char* szToken,
                        const char* szExtraInfoFormat, ...) { addError(); }
    void C_DECL  ppWarn(const TSourceLoc&, const char* szReason, const char* szToken,
                        const char* szExtraInfoFormat, ...) { }
#else
    virtual void C_DECL error(const TSourceLoc&, const char* szReason, const char* szToken,
        const char* szExtraInfoFormat, ...) = 0;
    virtual void C_DECL  warn(const TSourceLoc&, const char* szReason, const char* szToken,
        const char* szExtraInfoFormat, ...) = 0;
    virtual void C_DECL ppError(const TSourceLoc&, const char* szReason, const char* szToken,
        const char* szExtraInfoFormat, ...) = 0;
    virtual void C_DECL ppWarn(const TSourceLoc&, const char* szReason, const char* szToken,
        const char* szExtraInfoFormat, ...) = 0;
#endif

    void addError() { ++numErrors; }
    int getNumErrors() const { return numErrors; }

    void setScanner(TInputScanner* scanner) { lwrrentScanner = scanner; }
    TInputScanner* getScanner() const { return lwrrentScanner; }
    const TSourceLoc& getLwrrentLoc() const { return lwrrentScanner->getSourceLoc(); }
    void setLwrrentLine(int line) { lwrrentScanner->setLine(line); }
    void setLwrrentColumn(int col) { lwrrentScanner->setColumn(col); }
    void setLwrrentSourceName(const char* name) { lwrrentScanner->setFile(name); }
    void setLwrrentString(int string) { lwrrentScanner->setString(string); }

    void getPreamble(std::string&);
#ifdef ENABLE_HLSL
    bool isReadingHLSL()    const { return (messages & EShMsgReadHlsl) == EShMsgReadHlsl; }
    bool hlslEnable16BitTypes() const { return (messages & EShMsgHlslEnable16BitTypes) != 0; }
    bool hlslDX9Compatible() const { return (messages & EShMsgHlslDX9Compatible) != 0; }
#else
    bool isReadingHLSL()    const { return false; }
#endif

    TInfoSink& infoSink;

    // compilation mode
    int version;                 // version, updated by #version in the shader
    EShLanguage language;        // really the stage
    SpvVersion spvVersion;
    TIntermediate& intermediate; // helper for making and hooking up pieces of the parse tree

protected:
    TMap<TString, TExtensionBehavior> extensionBehavior;    // for each extension string, what its current behavior is set to
    EShMessages messages;        // errors/warnings/rule-sets
    int numErrors;               // number of compile-time errors encountered
    TInputScanner* lwrrentScanner;

private:
    explicit TParseVersions(const TParseVersions&);
    TParseVersions& operator=(const TParseVersions&);
};

} // end namespace glslang

#endif // _PARSE_VERSIONS_INCLUDED_
