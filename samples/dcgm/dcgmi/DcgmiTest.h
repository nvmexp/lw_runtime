/*
 * DcgmTest.h
 *
 */
#pragma once

#include "Command.h"
#include "dcgm_structs_internal.h"

class DcgmiTest : public Command{
public:
    DcgmiTest();
    virtual ~DcgmiTest();

    /* Save and load caches on host specified by mDcgmHandle */
    dcgmReturn_t CacheFileLoadSave(dcgmHandle_t mDcgmHandle, std::string filename, bool loadSave);

    /* View information in cache on host specified by mDcgmHandle */
    dcgmReturn_t IntrospectCache(dcgmHandle_t mDcgmHandle, unsigned int gpuId, std::string fieldId, bool isGroup);

    /* Inject errors and information into the cache on host specified by mDcgmHandle */
    dcgmReturn_t InjectCache(dcgmHandle_t mDcgmHandle, unsigned int gpuId, std::string fieldId, unsigned int pTime, std::string &injectValue);

private:
    /* Helper function to display field info to stdout */
    void HelperDisplayField(dcgmCacheManagerFieldInfo_t &fieldInfo);

    /* Helper function to initialize and populate the needed data in the field value */
    dcgmReturn_t HelperInitFieldValue(dcgmInjectFieldValue_t &injectFieldValue, std::string &injectValue);

    /* Helper function to parse user input into field id  */
    dcgmReturn_t HelperParseForFieldId(std::string str, unsigned short &fieldId, dcgmHandle_t mDcgmHandle);

    /* Helper function to format timestamp into human readable format */
    std::string HelperFormatTimestamp(long long timestamp);
};


/*****************************************************************************
 * Define classes to extend commands
 ****************************************************************************/

/**
 * Load Cache Ilwoker
 */
class LoadSaveCache : public Command
{
public:
    LoadSaveCache(std::string hostname, std::string fileName, bool save);
    virtual ~LoadSaveCache();

    int Execute();

private:
    DcgmiTest adminObj;
    std::string fileName;
    bool save;
};

/**
 * Introspect Cache Ilwoker
 */
class IntrospectCache : public Command
{
public:
    IntrospectCache(std::string hostname, unsigned int gpuId, std::string fieldId, bool isGroup);
    virtual ~IntrospectCache();

    int Execute();

private:
    DcgmiTest adminObj;
    unsigned int mGpuId;
    std::string mFieldId;
    bool mIDisGroup;
};

/**
 * Inject Cache Ilwoker
 */
class InjectCache : public Command
{
public:
    InjectCache(std::string hostname, unsigned int gId, std::string fieldId, unsigned int pTime, std::string injectValue);
    virtual ~InjectCache();

    int Execute();

private:
    DcgmiTest adminObj;
    unsigned int mGId;
    std::string mFieldId;
    unsigned int mTime;
    std::string mInjectValue;
};


