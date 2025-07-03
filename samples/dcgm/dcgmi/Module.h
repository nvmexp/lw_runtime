#ifndef MODULE_H_
#define MODULE_H_

#include "Command.h"
#include "DcgmiOutput.h"
#include "dcgm_structs.h"

class Module {
public:
    Module();
    virtual ~Module();
    int RunBlacklistModule(dcgmHandle_t dcgmHandle, dcgmModuleId_t moduleId, DcgmiOutput& out);
    int RunListModule(dcgmHandle_t dcgmHandle, DcgmiOutput& out);
    dcgmReturn_t statusToStr(dcgmModuleStatus_t status, std::string& str);
    static dcgmReturn_t moduleIdToName(dcgmModuleId_t moduleId, std::string& str);
private:
};

class BlacklistModule : public Command
{
public:
    BlacklistModule(const std::string& hostname, const std::string& moduleName, bool json);
    virtual ~BlacklistModule();
    int Execute();
private:
    Module mModuleObj;
    const std::string mModuleName;
};

class ListModule : public Command
{
public:
    ListModule(const std::string& hostname, bool json);
    virtual ~ListModule();
    int Execute();
private:
    Module mModuleObj;
};

#endif /* MODULE_H_ */
