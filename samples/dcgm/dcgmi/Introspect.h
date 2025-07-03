#ifndef INTROSPECT_H_
#define INTROSPECT_H_

#include "Command.h"
#include "CommandOutputController.h"
#include <string.h>

using std::string;

/*****************************************************************************
 * Define classes to extend commands
 ****************************************************************************/

class Introspect {
public:
    Introspect();
    virtual ~Introspect();

    dcgmReturn_t EnableIntrospect(dcgmHandle_t handle);
    dcgmReturn_t DisableIntrospect(dcgmHandle_t handle);
    dcgmReturn_t DisplayStats(dcgmHandle_t handle,
                              bool forHostengine,
                              bool forAllFields,
                              bool forAllFieldGroups,
                              std::vector<dcgmFieldGrp_t>forFieldGroups);

private:
    string readableMemory(long long bytes);
    string readablePercent(double p);

    template <typename T>
    string readableTime(T usec);
};

/**
 * Toggle whether Introspection is enabled
 */
class ToggleIntrospect : public Command
{
public:
    ToggleIntrospect(string hostname, bool enabled);
    virtual ~ToggleIntrospect();

    int Execute();

private:
    Introspect introspectObj;
    bool enabled;
};

/**
 * Display a summary of introspection information
 */
class DisplayIntrospectSummary : public Command
{
public:
    DisplayIntrospectSummary(string hostname,
                             bool forHostengine,
                             bool forAllFields,
                             bool forAllFieldGroups,
                             std::vector<dcgmFieldGrp_t> forFieldGroups);
    virtual ~DisplayIntrospectSummary();

    int Execute();

private:
    Introspect introspectObj;
    bool forHostengine;
    bool forAllFields;
    bool forAllFieldGroups;
    std::vector<dcgmFieldGrp_t> forFieldGroups;
};


#endif /* INTROSPECT_H_ */
