
#define TCLAP_SETBASE_ZERO 1

#include <iostream>
#include <algorithm>
#include <stdexcept>
#include <sstream>
#include <signal.h>
#include "Config.h"
#include "Group.h"
#include "FieldGroup.h"
#include "Query.h"
#include "Policy.h"
#include "Health.h"
#include "Diag.h"
#include "Topo.h"
#include "Introspect.h"
#include "Lwlink.h"
#include "DeviceMonitor.h"
#include "Module.h"
#include <ctype.h>
#include "ProcessStats.h"
#include "dcgm_client_internal.h"
#include "timelib.h"
#include "CommandLineParser.h"
#include "LwcmTCLAP.h"
#include "dcgm_fields.h"
#include "DcgmStringColwersions.h"
#include "tclap/ArgException.h"
#include "dcgm_agent.h"
#include "dcgm_structs.h"
#include "DcgmiProfile.h"
#include "DcgmDiagCommon.h"
#include "../common/DcgmStringTokenize.h"

#ifdef DEBUG
#include "DcgmiTest.h"
#endif

#define CHECK_TCLAP_ARG_NEGATIVE_VALUE(arg, name)                                                \
    if (arg.getValue() < 0)                                                                         \
        throw(TCLAP::CmdLineParseException("Positive value expected, negative value found", name))

using namespace std; // TODO: refactor this file so that we no longer need to embed this namespace

static const string g_hostnameHelpText = "Connects to specified IP or fully-qualified domain name. To connect to a host engine that was started with -d (unix socket), prefix the unix socket filename with 'unix://'. [default = localhost]";

static const std::string HW_SLOWDOWN("hw_slowdown");
static const std::string SW_THERMAL("sw_thermal");
static const std::string HW_THERMAL("hw_thermal");
static const std::string HW_POWER_BRAKE("hw_power_brake");

/*****************************************************************************
 * The following classes/functions process the command line parameters and
 * display help if needed using the TCLAP library.
 */

CommandLineParser::CommandLineParser()
{
    // fill in the available subsystems and their appropriate pointers
    mFunctionMap.insert(std::make_pair("discovery", &CommandLineParser::processQueryCommandLine));
    mFunctionMap.insert(std::make_pair("policy", &CommandLineParser::processPolicyCommandLine));
    mFunctionMap.insert(std::make_pair("group", &CommandLineParser::processGroupCommandLine));
    mFunctionMap.insert(std::make_pair("fieldgroup", &CommandLineParser::processFieldGroupCommandLine));
    mFunctionMap.insert(std::make_pair("config", &CommandLineParser::processConfigCommandLine));
    mFunctionMap.insert(std::make_pair("health", &CommandLineParser::processHealthCommandLine));
    mFunctionMap.insert(std::make_pair("diag", &CommandLineParser::processDiagCommandLine));
    mFunctionMap.insert(std::make_pair("stats", &CommandLineParser::processStatsCommandLine));
    mFunctionMap.insert(std::make_pair("topo", &CommandLineParser::processTopoCommandLine));
    mFunctionMap.insert(std::make_pair("introspect", &CommandLineParser::processIntrospectCommandLine));
    mFunctionMap.insert(std::make_pair("lwlink", &CommandLineParser::processLwlinkCommandLine));
    mFunctionMap.insert(std::make_pair("dmon", &CommandLineParser::processDmonCommandLine));
    mFunctionMap.insert(std::make_pair("modules", &CommandLineParser::processModuleCommandLine));
    mFunctionMap.insert(std::make_pair("profile", &CommandLineParser::processProfileCommandLine));
    
#ifdef DEBUG
    mFunctionMap.insert(std::make_pair("test", &CommandLineParser::processAdminCommandLine));
#endif
}

/* Entry method into this class for a given command line provided by main()
 */
int CommandLineParser::processCommandLine(int argc, char *argv[])
{
    dcgmReturn_t result = DCGM_ST_OK;
    try
    {
        // declare the main class that will handle processing
        // the two macros will be displayed as part of the help information
        TCLAP::CmdLine cmd(_DCGMI_FORMAL_NAME, ' ', _DCGMI_VERSION);

        // override the output so that usage can display the lwdatools address
        DCGMEntryOutput lwout;
        cmd.setOutput(&lwout);

        // parameters
        TCLAP::UnlabeledValueArg<std::string> subsystemArg("subsystem", "The desired subsystem to be accessed."
                "\n Subsystems Available:", true, "", "subsystem", cmd);
#ifdef DEBUG
        TCLAP::ValueArg<std::string> adminArg("","test", "Tests and Cache Manager [dcgmi admin –h for more info]", false, "", "", cmd);
#endif
        TCLAP::ValueArg<std::string> topoArg("", "topo", "GPU Topology [dcgmi topo -h for more info]", false, "", "", cmd);
        TCLAP::ValueArg<std::string> statsArg("", "stats", "Process Statistics [dcgmi stats -h for more info]", false, "", "", cmd);
        TCLAP::ValueArg<std::string> diagArg("","diag", "System Validation/Diagnostic [dcgmi diag –h for more info]", false, "", "", cmd);
        TCLAP::ValueArg<std::string> policyArg("","policy", "Policy Management [dcgmi policy –h for more info]", false, "", "", cmd);
        TCLAP::ValueArg<std::string> healthArg("","health", "Health Monitoring [dcgmi health –h for more info]", false, "", "", cmd);
        TCLAP::ValueArg<std::string> configArg("","config", "Configuration Management [dcgmi config –h for more info]", false, "", "", cmd);
        TCLAP::ValueArg<std::string> groupArg("","group", "GPU Group Management [dcgmi group –h for more info]", false, "", "", cmd);
        TCLAP::ValueArg<std::string> fieldGroupArg("","fieldgroup", "Field Group Management [dcgmi fieldgroup –h for more info]", false, "", "", cmd);
        TCLAP::ValueArg<std::string> queryArg("","discovery", "Discover GPUs on the system [dcgmi discovery –h for more info]", false, "", "", cmd);
        TCLAP::ValueArg<std::string> introspectArg("","introspect", "Gather info about DCGM itself [dcgmi introspect –h for more info]", false, "", "", cmd);
        TCLAP::ValueArg<std::string> lwlinkArg("","lwlink", "Displays LwLink link statuses and error counts [dcgmi lwlink –h for more info]", false, "", "", cmd);
        TCLAP::ValueArg<std::string> dmonArg("","dmon", "Stats Monitoring of GPUs [dcgmi dmon –h for more info]", false, "", "", cmd);
        TCLAP::ValueArg<std::string> moduleArg("","modules", "Control and list DCGM modules", false, "", "", cmd);
        TCLAP::ValueArg<std::string> profileArg("","profile", "Control and list DCGM profiling metrics", false, "", "", cmd);

        lwout.addToGroup("1", &subsystemArg);
        lwout.addFooter("Please email dcgm-support@lwpu.com with any questions, bug reports, etc.\n");
        // normally we would pass all of argc and argv.  But for this, we want just the first two
        if (argc < 2)
            cmd.parse(argc, argv);

        std::vector<std::string> args;
        for (int i = 0; i < 2; i++)
            args.push_back(argv[i]);
        cmd.parse(args);

        // call the correct subsystem
        functionIterator it = mFunctionMap.find(subsystemArg.getValue());
        if (it != mFunctionMap.end())
        {
            result = (dcgmReturn_t) (this->*mFunctionMap[subsystemArg.getValue()])(argc, argv);
        } 
        else
        {
            std::cout << "ERROR: Invalid subsystem." << std::endl << std::endl;
            lwout.usage(cmd);
            return -1;
        }
    }
    catch (TCLAP::ArgException &e) 
    {   
        std::cerr << "Error: " << e.error();
        if (e.argId().size() > 10)
        {
            // Substring needed since `argId()` prepends 'Argument: ' to the argument name
            std::cerr << " for arg " << e.argId().substr(10);
        }
        // When raising CmdLineParseException, do not add trailing '.' to message since error handler will add it.
        std::cerr << "." << std::endl;
        throw std::runtime_error ("An error oclwrred trying to parse the command line.");
    }  
    return result;
}

// the below subsystem processing commands assume two things
// 1) that the calling structure is already in a try block
// 2) that the argc and argv commands have not been modified
int CommandLineParser::processQueryCommandLine(int argc, char *argv[])
{
    dcgmReturn_t result = DCGM_ST_OK;
    std::string myName = "discovery";
    DCGMOutput helpOutput;

    DCGMSubsystemCmdLine cmd(myName, _DCGMI_FORMAL_NAME, ' ', _DCGMI_VERSION);
    cmd.setOutput(&helpOutput);

    // args are displayed in reverse order to this list
    TCLAP::ValueArg<std::string> getInfo("i", "info", "Specify which information to return. [default = atp] \n a - device info\n p - power limits\n t - thermal limits\n c - clocks", false, "atp", "flags");
    TCLAP::ValueArg<int> gpuId("", "gpuid", "The GPU ID to query.", false, 0, "gpuId", cmd);
    TCLAP::ValueArg<int> groupId("g", "group", "The group ID to query.", false, DCGM_GROUP_ALL_GPUS, "groupId", cmd);
    TCLAP::SwitchArg verbose("v", "verbose", "Display policy information per GPU.", cmd, false);
    TCLAP::SwitchArg getList("l", "list", "List all GPUs discovered on the host.", false);
    TCLAP::ValueArg<std::string> hostAddress("", "host", g_hostnameHelpText, false, "localhost", "IP/FQDN");

    std::vector <TCLAP::Arg *> xors;
    xors.push_back(&getList);
    xors.push_back(&getInfo);
    cmd.xorAdd(xors);
    cmd.add(&hostAddress);

    // Set help output information
    helpOutput.addDescription("discovery -- Used to discover and identify GPUs and their attributes.");
    helpOutput.addToGroup("1", &hostAddress);
    helpOutput.addToGroup("1", &getList);

    helpOutput.addToGroup("2", &hostAddress);
    helpOutput.addToGroup("2", &getInfo);
    helpOutput.addToGroup("2", &groupId);
    helpOutput.addToGroup("2", &verbose);

    cmd.parse(argc, argv);

    // Check for negative (invalid) inputs
    CHECK_TCLAP_ARG_NEGATIVE_VALUE(gpuId, "gpuid");
    CHECK_TCLAP_ARG_NEGATIVE_VALUE(groupId, "group");

    if (groupId.isSet() && gpuId.isSet())
    {
        throw(TCLAP::CmdLineParseException("Both GPU and Group specified at command line"));
    }

    Command *cmdq = NULL;
    if (getList.isSet())
    {
        cmdq = new QueryDeviceList(hostAddress.getValue());
    } 
    else if (gpuId.isSet())
    {
        cmdq = new QueryDeviceInfo(hostAddress.getValue(), gpuId.getValue(), getInfo.getValue());
    } 
    else if (groupId.isSet())
    {
        cmdq = new QueryGroupInfo(hostAddress.getValue(), groupId.getValue(), getInfo.getValue(), verbose.isSet());
    } 
    else if (getInfo.isSet())
    {
        cmdq = new QueryGroupInfo(hostAddress.getValue(), DCGM_GROUP_ALL_GPUS, getInfo.getValue(), verbose.isSet());
    }

    if (cmdq)
    {
        result = (dcgmReturn_t) cmdq->Execute();
    }

    delete(cmdq);

    return result;
}

int CommandLineParser::processPolicyCommandLine(int argc, char *argv[])
{
    dcgmReturn_t result = DCGM_ST_OK;
    std::string myName = "policy";
    DCGMOutput helpOutput;

    DCGMSubsystemCmdLine cmd(myName, _DCGMI_FORMAL_NAME, ' ', _DCGMI_VERSION);

    cmd.setOutput(&helpOutput);
    // args are displayed in reverse order to this list

    TCLAP::ValueArg<std::string> hostAddress("", "host", g_hostnameHelpText, false, "localhost", "IP/FQDN");
    TCLAP::SwitchArg clearPolicy("", "clear", "Clear the current violation policy.", false);
    TCLAP::SwitchArg getPolicy("", "get", "Get the current violation policy.", false);
    TCLAP::SwitchArg regPolicy("", "reg", "Register this process for policy updates.  This process will sit in an infinite loop waiting for updates from the policy manager.", false);
    TCLAP::ValueArg<std::string> setPolicy("", "set", "Set the current violation policy. Use csv action,validation (ie. 1,2)"
            "\n-----"
            "\nAction to take when any of the violations specified occur.\n 0 - None\n 1 - GPU Reset"
            "\n-----"
            "\lwalidation to take after the violation action has been performed."
            "\n 0 - None\n 1 - System Validation (short)\n 2 - System Validation (medium)\n 3 - System Validation (long)", false, "0,1", "actn,val");
    TCLAP::ValueArg<int> groupId("g", "group", "The GPU group to query on the specified host.", false, DCGM_GROUP_ALL_GPUS, "groupId", cmd);
    TCLAP::SwitchArg verbose("v", "verbose", "Display policy information per GPU.", cmd, false);
    TCLAP::ValueArg<int> maxpages("M", "maxpages", "Specify the maximum number of retired pages that will trigger a violation.", false, 0, "max", cmd);
    TCLAP::ValueArg<int> thermal("T", "maxtemp", "Specify the maximum temperature a group's GPUs can reach before triggering a violation.", false, 0, "max", cmd);
    TCLAP::ValueArg<int> power("P", "maxpower", "Specify the maximum power a group's GPUs can reach before triggering a violation.", false, 0, "max", cmd);
    TCLAP::SwitchArg eccdbe("e", "eccerrors", "Add ECC double bit errors to the policy conditions.", cmd, false);
    TCLAP::SwitchArg pciError("p", "pcierrors", "Add PCIe replay errors to the policy conditions.", cmd, false);
    TCLAP::SwitchArg lwlinkError("n", "lwlinkerrors", "Add LWLink errors to the policy conditions.", cmd, false);
    TCLAP::SwitchArg xidError("x", "xiderrors", "Add XID errors to the policy conditions.", cmd, false);
    TCLAP::SwitchArg json("j", "json", "Print the output in a json format", cmd, false);

    std::vector <TCLAP::Arg *> xors;
    xors.push_back(&getPolicy);
    xors.push_back(&regPolicy);
    xors.push_back(&setPolicy);
    xors.push_back(&clearPolicy);
    cmd.xorAdd(xors);
    cmd.add(&hostAddress);

    helpOutput.addDescription("policy -- Used to control policies for groups of GPUs. Policies control actions which are triggered by specific events.");
    helpOutput.addToGroup("1", &hostAddress);
    helpOutput.addToGroup("1", &groupId);
    helpOutput.addToGroup("1", &getPolicy);
    helpOutput.addToGroup("1", &json);

    helpOutput.addToGroup("2", &hostAddress);
    helpOutput.addToGroup("2", &groupId);
    helpOutput.addToGroup("2", &regPolicy);

    helpOutput.addToGroup("3", &hostAddress);
    helpOutput.addToGroup("3", &groupId);
    helpOutput.addToGroup("3", &setPolicy);
    helpOutput.addToGroup("3", &maxpages);
    helpOutput.addToGroup("3", &thermal);
    helpOutput.addToGroup("3", &power);
    helpOutput.addToGroup("3", &eccdbe);
    helpOutput.addToGroup("3", &pciError);
    helpOutput.addToGroup("3", &lwlinkError);    
    helpOutput.addToGroup("3", &xidError);    

    cmd.parse(argc, argv);

    // Check for negative (invalid) inputs
    CHECK_TCLAP_ARG_NEGATIVE_VALUE(maxpages, "maxpages");
    CHECK_TCLAP_ARG_NEGATIVE_VALUE(thermal, "maxtemp");
    CHECK_TCLAP_ARG_NEGATIVE_VALUE(power, "maxpower");
    CHECK_TCLAP_ARG_NEGATIVE_VALUE(groupId, "group");

    Command *cmdp = NULL;

    if (setPolicy.isSet() && !(maxpages.isSet() || thermal.isSet() ||
            power.isSet() || eccdbe.isSet() || pciError.isSet() || lwlinkError.isSet() || xidError.isSet()))
    {
        throw(TCLAP::CmdLineParseException("No conditions specified", "set"));
    }

    if (setPolicy.isSet())
    {
        // Verify that the given value is in the correct action,validation format and has length 3
        if (setPolicy.getValue().find(',') == std::string::npos || setPolicy.getValue().size() != 3)
        {
            throw(TCLAP::CmdLineParseException("Must use action,validation (csv) format", "set"));
        } 
        // Verify the action is a valid integer between 0 and 1
        else if (!isdigit(setPolicy.getValue().at(0)) || (setPolicy.getValue().at(0) - '0') > 1 
                 || (setPolicy.getValue().at(0) - '0') < 0)
        {
            throw(TCLAP::CmdLineParseException("The action must be 0 or 1", "set"));
        }
        // Verify validation is a valid integer between 0 and 3
        else if (!isdigit(setPolicy.getValue().at(2)) || (setPolicy.getValue().at(2) - '0') > 3 
                 || (setPolicy.getValue().at(2) - '0') < 0)
        {
            throw(TCLAP::CmdLineParseException("The validation must be between 0 and 3 (inclusive)", "set"));
        }
    }

    if (verbose.isSet() && !getPolicy.isSet())
    {
        throw(TCLAP::CmdLineParseException("Verbose option only available with the get policy arg (--get)"));
    }

    if (getPolicy.getValue())
    {
        cmdp = new GetPolicy(hostAddress.getValue(), groupId.getValue(), verbose.isSet(), json.getValue());
    }

    int buffer = 0;

    if (clearPolicy.isSet())
    {
        dcgmPolicy_t setPolicy;
        setPolicy.version = dcgmPolicy_version;
        setPolicy.condition = (dcgmPolicyCondition_t) 0;
        setPolicy.mode = DCGM_POLICY_MODE_MANUAL; // ignored for now
        setPolicy.isolation = DCGM_POLICY_ISOLATION_NONE; // ignored for now
        setPolicy.action = DCGM_POLICY_ACTION_NONE;
        setPolicy.validation = DCGM_POLICY_VALID_NONE;
        setPolicy.response = DCGM_POLICY_FAILURE_NONE; // ignored for now
        cmdp = new SetPolicy(hostAddress.getValue(), setPolicy, groupId.getValue());
    } 
    else if (setPolicy.isSet())
    {
        dcgmPolicy_t setPol;
        setPol.version = dcgmPolicy_version;

        setPol.parms[0].tag = dcgmPolicyConditionParms_t::BOOL;
        if (eccdbe.isSet())
        {
            setPol.parms[0].val.boolean = true;
            buffer+= DCGM_POLICY_COND_DBE;
        }
        setPol.parms[1].tag = dcgmPolicyConditionParms_t::BOOL;
        if (pciError.isSet())
        {
            setPol.parms[1].val.boolean = true;
            buffer+= DCGM_POLICY_COND_PCI;
        }
        setPol.parms[2].tag = dcgmPolicyConditionParms_t::LLONG;
        if (maxpages.isSet())
        {
            setPol.parms[2].val.llval = maxpages.getValue();
            buffer+= DCGM_POLICY_COND_MAX_PAGES_RETIRED;
        }
        setPol.parms[3].tag = dcgmPolicyConditionParms_t::LLONG;
        if (thermal.isSet())
        {
            setPol.parms[3].val.llval = thermal.getValue();
            buffer+= DCGM_POLICY_COND_THERMAL;
        }
        setPol.parms[4].tag = dcgmPolicyConditionParms_t::LLONG;
        if (power.isSet())
        {
            setPol.parms[4].val.llval = power.getValue();
            buffer+= DCGM_POLICY_COND_POWER;
        }
        setPol.parms[5].tag = dcgmPolicyConditionParms_t::BOOL;
        if (lwlinkError.isSet())
        {
            setPol.parms[5].val.boolean = true;
            buffer+= DCGM_POLICY_COND_LWLINK;
        }
        setPol.parms[6].tag = dcgmPolicyConditionParms_t::BOOL;
        if (xidError.isSet()) 
        {
            setPol.parms[6].val.boolean = true;
            buffer+= DCGM_POLICY_COND_XID;
        }

        setPol.condition = (dcgmPolicyCondition_t) buffer;

        setPol.mode = DCGM_POLICY_MODE_MANUAL; // ignored for now
        setPol.isolation = DCGM_POLICY_ISOLATION_NONE; // ignored for now
        setPol.action = (dcgmPolicyAction_t) (setPolicy.getValue().at(0) - '0');
        setPol.validation = (dcgmPolicyValidation_t) (setPolicy.getValue().at(2) - '0');
        setPol.response = DCGM_POLICY_FAILURE_NONE; // ignored for now

        cmdp = new SetPolicy(hostAddress.getValue(), setPol, groupId.getValue());
    }
    else if (regPolicy.isSet())
    {
        if (pciError.isSet())
        {
            buffer+= DCGM_POLICY_COND_PCI;
        }
        if (eccdbe.isSet())
        {
            buffer+= DCGM_POLICY_COND_DBE;
        }
        if (maxpages.isSet())
        {
            buffer+= DCGM_POLICY_COND_MAX_PAGES_RETIRED;
        }
        if (thermal.isSet())
        {
            buffer+= DCGM_POLICY_COND_THERMAL;
        }
        if (power.isSet())
        {
            buffer+= DCGM_POLICY_COND_POWER;
        }
        if (lwlinkError.isSet())
        {
            buffer+= DCGM_POLICY_COND_LWLINK;
        }
        if (xidError.isSet())
        {
            buffer+= DCGM_POLICY_COND_XID;
        }
        cmdp = new RegPolicy(hostAddress.getValue(), groupId.getValue(), buffer);
    }

    if (cmdp)
    {
        result = (dcgmReturn_t) cmdp->Execute();
    }

    delete(cmdp);

    return result;
}

int CommandLineParser::processGroupCommandLine(int argc, char *argv[])
{
    dcgmReturn_t result = DCGM_ST_OK;
    std::string myName = "group";
    DCGMOutput helpOutput;
    Group groupObj;

    DCGMSubsystemCmdLine cmd(myName, _DCGMI_FORMAL_NAME, ' ', _DCGMI_VERSION);
    cmd.setOutput(&helpOutput);

    // args are displayed in reverse order to this list
    TCLAP::SwitchArg listGroup("l", "list", "List the groups that lwrrently exist for a host.", false);
    TCLAP::SwitchArg infoGroup("i", "info", "Display the information for the specified group ID.", false);
    TCLAP::SwitchArg defaultGroup("", "default", "Adds all available GPUs to the group being created.", false);
    TCLAP::SwitchArg defaultLwSwitchGroup("", "defaultlwswitches", "Adds all available LwSwitches to the group being created.", false);
    TCLAP::ValueArg<int> groupId("g", "group", "The GPU group to query on the specified host.", false, 1, "groupId");
    TCLAP::ValueArg<std::string> createGroup("c", "create", "Create a group on the remote host.", false, "generic", "groupName");
    TCLAP::ValueArg<int> deleteGroup("d", "delete", "Delete a group on the remote host.", false, 0, "groupId");
    TCLAP::ValueArg<std::string> addDevice("a", "add", "Add device(s) to group. (csv gpuIds or entityIds like gpu:0,lwswitch:994)", false, "", "entityId");
    TCLAP::ValueArg<std::string> removeDevice("r", "remove", "Remove device(s) from group. (csv gpuIds, or entityIds like gpu:0,lwswitch:994)", false, "", "entityId");
    TCLAP::ValueArg<std::string> hostAddress("", "host", g_hostnameHelpText, false, "localhost", "IP/FQDN");
    TCLAP::SwitchArg json("j", "json", "Print the output in a json format", false);

    cmd.add(&infoGroup);
    cmd.add(&removeDevice);
    cmd.add(&addDevice);
    cmd.add(&defaultGroup);
    cmd.add(&defaultLwSwitchGroup);
    cmd.add(&json);

    std::vector <TCLAP::Arg *> xors;
    xors.push_back(&listGroup);
    xors.push_back(&groupId);
    xors.push_back(&deleteGroup);
    xors.push_back(&createGroup);
    cmd.xorAdd(xors);
    cmd.add(&hostAddress);

    helpOutput.addDescription("group -- Used to create and maintain groups of GPUs. Groups of GPUs can then be uniformly controlled through other DCGMI subsystems.");
    helpOutput.addToGroup("1", &hostAddress);
    helpOutput.addToGroup("1", &listGroup);
    helpOutput.addToGroup("1", &json);

    helpOutput.addToGroup("2", &hostAddress);
    helpOutput.addToGroup("2", &createGroup);
    helpOutput.addToGroup("2", &defaultGroup);
    helpOutput.addToGroup("2", &defaultLwSwitchGroup);

    helpOutput.addToGroup("3", &hostAddress);
    helpOutput.addToGroup("3", &createGroup);
    helpOutput.addToGroup("3", &addDevice);

    helpOutput.addToGroup("4", &hostAddress);
    helpOutput.addToGroup("4", &deleteGroup);

    helpOutput.addToGroup("5", &hostAddress);
    helpOutput.addToGroup("5", &groupId);
    helpOutput.addToGroup("5", &infoGroup);
    helpOutput.addToGroup("5", &json);

    helpOutput.addToGroup("6", &hostAddress);
    helpOutput.addToGroup("6", &groupId);
    helpOutput.addToGroup("6", &addDevice);

    helpOutput.addToGroup("7", &hostAddress);
    helpOutput.addToGroup("7", &groupId);
    helpOutput.addToGroup("7", &removeDevice);

    cmd.parse(argc, argv);

    // Check for negative (invalid) inputs
    CHECK_TCLAP_ARG_NEGATIVE_VALUE(deleteGroup, "delete");
    CHECK_TCLAP_ARG_NEGATIVE_VALUE(groupId, "group");

    Command *cmdg = NULL;

    if (groupId.isSet() && !(infoGroup.isSet() || addDevice.isSet() || removeDevice.isSet() ))
    {
        throw(TCLAP::CmdLineParseException("No action has been specified for the group"));
    }
    if (addDevice.isSet() && (deleteGroup.isSet() || infoGroup.isSet()))
    {
        throw(TCLAP::CmdLineParseException("Add device(s) cannot be used with the delete group (--delete) "
                                           "or group info (--info) options"));
    }
    if (removeDevice.isSet() && (deleteGroup.isSet() || infoGroup.isSet()))
    {
        throw(TCLAP::CmdLineParseException("Remove device(s) cannot be used with the delete group (--delete) "
                                           "or group info (--info) options"));
    }
    if (removeDevice.isSet() && addDevice.isSet())
    {
        throw(TCLAP::CmdLineParseException("Both add and remove device are set. Please use one at a time"));
    }

    if (createGroup.isSet() && (!createGroup.getValue().compare("--default") ||
                                !createGroup.getValue().compare("--defaultlwswitches") || 
                                !createGroup.getValue().compare("-a") || 
                                !createGroup.getValue().compare("--add")))
    {
        throw(TCLAP::CmdLineParseException("No group name given", "create"));
    }

    if (listGroup.isSet())
    {
        cmdg = new GroupList(hostAddress.getValue(), json.getValue());
    }
    else if (createGroup.isSet())
    {
        dcgmGroupType_t groupType = DCGM_GROUP_EMPTY;
        groupObj.SetGroupName(createGroup.getValue());
        groupObj.SetGroupInfo(addDevice.getValue());

        if (defaultGroup.isSet() && defaultLwSwitchGroup.isSet())
        {
            throw(TCLAP::CmdLineParseException("Both --default and --defaultlwswitches are set. "
                                               "Please use one at a time"));
        }

        if (defaultGroup.isSet())
        {
            groupType = DCGM_GROUP_DEFAULT;
        }
        else if (defaultLwSwitchGroup.isSet())
        {
            groupType = DCGM_GROUP_DEFAULT_LWSWITCHES;
        }
        cmdg = new GroupCreate(hostAddress.getValue(), groupObj, groupType);
    }
    else if (deleteGroup.isSet())
    {
        groupObj.SetGroupId(deleteGroup.getValue());
        cmdg = new GroupDestroy(hostAddress.getValue(), groupObj);
    }
    else if (groupId.isSet())
    {
        groupObj.SetGroupId(groupId.getValue());

        if (infoGroup.isSet())
        {
            cmdg = new GroupInfo(hostAddress.getValue(), groupObj, json.getValue());
        } 
        else if (addDevice.isSet())
        {
            groupObj.SetGroupInfo(addDevice.getValue());
            cmdg = new GroupAddTo(hostAddress.getValue(), groupObj);
        } 
        else if (removeDevice.isSet())
        {
            groupObj.SetGroupInfo(removeDevice.getValue());
            cmdg = new GroupDeleteFrom(hostAddress.getValue(), groupObj);
        }
    }

    if (cmdg)
    {
        result =  (dcgmReturn_t) cmdg->Execute();
    }

    delete(cmdg);

    return result;
}

int CommandLineParser::processFieldGroupCommandLine(int argc, char *argv[])
{
    dcgmReturn_t result = DCGM_ST_OK;
    std::string myName = "fieldgroup";
    DCGMOutput helpOutput;
    FieldGroup fieldGroupObj;

    DCGMSubsystemCmdLine cmd(myName, _DCGMI_FORMAL_NAME, ' ', _DCGMI_VERSION);
    cmd.setOutput(&helpOutput);

    // args are displayed in reverse order to this list
    TCLAP::SwitchArg listGroup("l", "list", "List the field groups that lwrrently exist for a host.", false);
    TCLAP::SwitchArg infoGroup("i", "info", "Display the information for the specified field group ID.", false);
    TCLAP::ValueArg<int> fieldGroupId("g", "fieldgroup", "The field group to query on the specified host.", false, -1, "fieldGroupId");
    TCLAP::ValueArg<std::string> fieldIds("f", "fieldids", "Comma-separated list of the field ids to add to a field group when creating a new one.", false, "", "fieldIds");
    TCLAP::ValueArg<std::string> createFieldGroup("c", "create", "Create a field group on the remote host.", false, "generic", "fieldGroupName");
    TCLAP::SwitchArg deleteFieldGroup("d", "delete", "Delete a field group on the remote host.", false);
    TCLAP::ValueArg<std::string> hostAddress("", "host", g_hostnameHelpText, false, "localhost", "IP/FQDN");
    TCLAP::SwitchArg json("j", "json", "Print the output in a json format", false);

    std::vector <TCLAP::Arg *> xors;
    xors.push_back(&listGroup);
    xors.push_back(&infoGroup);
    xors.push_back(&deleteFieldGroup);
    xors.push_back(&createFieldGroup);
    cmd.xorAdd(xors);
    cmd.add(&hostAddress);
    cmd.add(&fieldGroupId);
    cmd.add(&fieldIds);
    cmd.add(&json);

    helpOutput.addDescription("fieldgroup -- Used to create and maintain groups of field IDs. "
                              "Groups of field IDs can then be uniformly controlled through "
                              "other DCGMI subsystems.");
    helpOutput.addToGroup("1", &hostAddress);
    helpOutput.addToGroup("1", &listGroup);
    helpOutput.addToGroup("1", &json);

    helpOutput.addToGroup("2", &hostAddress);
    helpOutput.addToGroup("2", &createFieldGroup);
    helpOutput.addToGroup("2", &fieldIds);

    helpOutput.addToGroup("3", &hostAddress);
    helpOutput.addToGroup("3", &infoGroup);
    helpOutput.addToGroup("3", &fieldGroupId);
    helpOutput.addToGroup("3", &json);

    helpOutput.addToGroup("4", &hostAddress);
    helpOutput.addToGroup("4", &deleteFieldGroup);
    helpOutput.addToGroup("4", &fieldGroupId);

    cmd.parse(argc, argv);

    Command *cmdg = NULL;

    /* Populate arguments that are used by multiple commands */
    if (fieldGroupId.isSet())
    {
        fieldGroupObj.SetFieldGroupId(fieldGroupId.getValue());
    }
    if (fieldIds.isSet())
    {
        fieldGroupObj.SetFieldIdsString(fieldIds.getValue());
    }
    if (createFieldGroup.isSet())
    {
        fieldGroupObj.SetFieldGroupName(createFieldGroup.getValue());
    }


    /* Decide on which command to run and do argument validation */
    if (listGroup.isSet())
    {
        cmdg = new FieldGroupListAll(hostAddress.getValue(), json.getValue());
    }
    else if (infoGroup.isSet())
    {
        if (!fieldGroupId.isSet())
        {
            throw(TCLAP::CmdLineParseException("No fieldGroupId given (specify with --fieldgroup)", "info"));
        }

        cmdg = new FieldGroupInfo(hostAddress.getValue(), fieldGroupObj, json.getValue());
    }
    else if (createFieldGroup.isSet())
    {
        if (!fieldIds.isSet())
        {
            throw(TCLAP::CmdLineParseException("No field IDs given (specify with --fieldids)", "create"));
        }

        cmdg = new FieldGroupCreate(hostAddress.getValue(), fieldGroupObj);
    }
    else if (deleteFieldGroup.isSet())
    {
        if (!fieldGroupId.isSet())
        {
            throw(TCLAP::CmdLineParseException("No fieldGroupId given (specify with --fieldgroup)", "delete"));
        }

        cmdg = new FieldGroupDestroy(hostAddress.getValue(), fieldGroupObj);
    }
    else
    {
        throw(TCLAP::CmdLineParseException("No command has been specified. Run dcgmi fieldgroup --help for "
                                           "supported options"));
    }

    if (cmdg)
    {
        result =  (dcgmReturn_t) cmdg->Execute();
    }

    delete(cmdg);

    return result;
}

int CommandLineParser::processConfigCommandLine(int argc, char* argv[])
{
    dcgmReturn_t result = DCGM_ST_OK;
    std::string myName = "config";
    DCGMOutput helpOutput;
    Config configObj;

    DCGMSubsystemCmdLine cmd(myName, _DCGMI_FORMAL_NAME, ' ', _DCGMI_VERSION);
    cmd.setOutput(&helpOutput);

    TCLAP::ValueArg<int> groupId("g", "group", "The GPU group to query on the specified host.",
            false, DCGM_GROUP_ALL_GPUS, "groupId", cmd);
    TCLAP::ValueArg<std::string> hostAddress("", "host", g_hostnameHelpText, false, "localhost", "IP/FQDN");
    TCLAP::SwitchArg setConfig("", "set", "Set configuration.", false);   // For Command-line to set
    TCLAP::ValueArg<bool> eccMode("e", "eccmode", "Configure Ecc mode. (1 to Enable, 0 to Disable)",
            false, 0, "0/1"); // Option 1 for set command-line

    TCLAP::ValueArg<bool> syncBoost("s", "syncboost", "Configure Syncboost. (1 to Enable, 0 to Disable)",
            false, 0, "0/1"); // Option 2 for set command-line

    TCLAP::ValueArg<std::string> appClocks("a", "appclocks", "Configure Application Clocks. Must use memory,proc clocks (csv) format(MHz). ",
            false, "", "mem,proc"); // Option 2 for set command-line
    TCLAP::ValueArg<int> powerLimit("P", "powerlimit", "Configure Power Limit (Watts).", false, 0, "limit");

    TCLAP::ValueArg<int> computeMode("c", "compmode", "Configure Compute Mode. Can be any of the following:"
            "\n0 - Unrestricted \n1 - Prohibited\n2 - Exclusive Process", false, 0, "mode");
    TCLAP::SwitchArg verbose("v", "verbose", "Display policy information per GPU.", cmd, false);
    TCLAP::SwitchArg json("j", "json", "Print the output in a json format", cmd, false);

    TCLAP::SwitchArg getConfig("", "get", "Get configuration. Displays the Target and the Current Configuration."
                            "\n------"
                            "\n 1.Sync Boost - Current and Target Sync Boost State"
                            "\n 2.SM Application Clock - Current and Target SM application clock values"
                            "\n 3.Memory Application Clock - Current and Target Memory application clock values"
                            "\n 4.ECC Mode - Current and Target ECC Mode"
                            "\n 5 Power Limit - Current and Target power limits"
                            "\n 6.Compute Mode - Current and Target compute mode", false);   // Different command-line
    // Add more options to get command-line

    TCLAP::SwitchArg enforceConfig("", "enforce", "Enforce configuration.", false); // Different command-line
    // Add more options to enforce command-line

    std::vector <TCLAP::Arg *> xors;
    xors.push_back(&setConfig);
    xors.push_back(&getConfig);
    xors.push_back(&enforceConfig);
    cmd.xorAdd(xors);    

    cmd.add(&eccMode);
    cmd.add(&syncBoost);
    cmd.add(&appClocks);
    cmd.add(&powerLimit);
    cmd.add(&computeMode);
    cmd.add(&hostAddress);

    helpOutput.addDescription("config -- Used to configure settings for groups of GPUs.");
    helpOutput.addToGroup("set", &hostAddress);
    helpOutput.addToGroup("set", &groupId);    
    helpOutput.addToGroup("set", &setConfig);
    helpOutput.addToGroup("set", &eccMode);
    helpOutput.addToGroup("set", &syncBoost);
    helpOutput.addToGroup("set", &appClocks);
    helpOutput.addToGroup("set", &powerLimit);
    helpOutput.addToGroup("set", &computeMode);

    helpOutput.addToGroup("get", &hostAddress);
    helpOutput.addToGroup("get", &groupId);
    helpOutput.addToGroup("get", &getConfig);
    helpOutput.addToGroup("get", &verbose);
    helpOutput.addToGroup("get", &json);

    helpOutput.addToGroup("enforce", &hostAddress);
    helpOutput.addToGroup("enforce", &groupId);
    helpOutput.addToGroup("enforce", &enforceConfig);


    cmd.parse(argc, argv);

    // Check for negative (invalid) inputs
    CHECK_TCLAP_ARG_NEGATIVE_VALUE(powerLimit, "powerlimit");
    CHECK_TCLAP_ARG_NEGATIVE_VALUE(groupId, "group");
    CHECK_TCLAP_ARG_NEGATIVE_VALUE(computeMode, "compmode");

    if (setConfig.isSet() && !(eccMode.isSet() || syncBoost.isSet() ||
            appClocks.isSet() || powerLimit.isSet() || computeMode.isSet()))
    {
        throw(TCLAP::CmdLineParseException("No configuration options given", "set"));
    }

    // Process Set Command
    if (setConfig.isSet())
    {
        dcgmConfig_t mDeviceConfig;

        /* Set all the Params as unknown */
        mDeviceConfig.eccMode = DCGM_INT32_BLANK;
        mDeviceConfig.perfState.syncBoost = DCGM_INT32_BLANK;
        mDeviceConfig.perfState.targetClocks.memClock = DCGM_INT32_BLANK;
        mDeviceConfig.perfState.targetClocks.smClock = DCGM_INT32_BLANK;
        mDeviceConfig.powerLimit.val = DCGM_INT32_BLANK;
        mDeviceConfig.computeMode = DCGM_INT32_BLANK;
        mDeviceConfig.gpuId = DCGM_INT32_BLANK;

        if (eccMode.isSet()) 
        {
            mDeviceConfig.eccMode = eccMode.getValue();
        }

        if (syncBoost.isSet()) 
        {
            mDeviceConfig.perfState.syncBoost = syncBoost.getValue();
        }

        if (appClocks.isSet()) 
        {
            string clocks = appClocks.getValue();
            unsigned int memClk, smClk;
            char tmp;

            if (sscanf(clocks.c_str(), "%d,%d%c", &memClk, &smClk, &tmp) != 2) 
            {
                throw(TCLAP::CmdLineParseException("Failed to parse application clocks"));
            }

            mDeviceConfig.perfState.targetClocks.memClock = memClk;
            mDeviceConfig.perfState.targetClocks.smClock = smClk;
        }

        if (powerLimit.isSet()) 
        {
            mDeviceConfig.powerLimit.type = DCGM_CONFIG_POWER_CAP_INDIVIDUAL;
            mDeviceConfig.powerLimit.val = powerLimit.getValue();
        }

        if (computeMode.isSet()) 
        {
            unsigned int val = computeMode.getValue();
            mDeviceConfig.computeMode = val;
        }

        configObj.SetArgs(groupId.getValue(), &mDeviceConfig);

        SetConfig cmd(hostAddress.getValue(), configObj);
        result = (dcgmReturn_t) cmd.Execute();
    } 
    else if (getConfig.isSet()) 
    {
        configObj.SetArgs(groupId.getValue(), 0);
        GetConfig cmd(hostAddress.getValue(), configObj, verbose.isSet(), json.isSet());

        if (eccMode.isSet() || (appClocks.isSet()) || (powerLimit.isSet()) || (computeMode.isSet())) 
        {
            throw(TCLAP::CmdLineParseException("Additional parameters specified at command line", "get"));
        }        

        result = (dcgmReturn_t) cmd.Execute();
    } 
    else if (enforceConfig.isSet()) 
    {

        if (eccMode.isSet() || (appClocks.isSet()) || (powerLimit.isSet()) || (computeMode.isSet())) 
        {
            throw(TCLAP::CmdLineParseException("Additional parameters specified at command line", "enforce"));
        }

        configObj.SetArgs(groupId.getValue(), 0);   // Set the required Args (groupId in this case)
        EnforceConfig cmd(hostAddress.getValue(), configObj);

        result = (dcgmReturn_t) cmd.Execute();
    }

    return result;
}

// Helper for checking duplicate flags given to the set watches argument for processHealthCommandLine
void checkDuplicateFlagsForSetWatches(unsigned int i, const std::string& setWatches)
{
    for (unsigned int j = i + 1; j < setWatches.length(); j++)
    {
        if (setWatches.at(i) == setWatches.at(j))
        {
            throw(TCLAP::CmdLineParseException("Duplicate flags detected", "set"));
        }
    }
}

int CommandLineParser::processHealthCommandLine(int argc, char *argv[])
{    
    dcgmReturn_t result = DCGM_ST_OK;
    std::string myName = "health";
    DCGMOutput helpOutput;

    DCGMSubsystemCmdLine cmd(myName, _DCGMI_FORMAL_NAME, ' ', _DCGMI_VERSION);
    cmd.setOutput(&helpOutput);

    TCLAP::ValueArg<int> groupId("g", "group", "The GPU group to query on the specified host.",
            false, DCGM_GROUP_ALL_GPUS, "groupId", cmd);
    TCLAP::ValueArg<std::string> hostAddress("", "host", g_hostnameHelpText, false, "localhost", "IP/FQDN");
    TCLAP::SwitchArg clearWatches("", "clear", "Disable all watches being monitored.", false);
    TCLAP::SwitchArg getWatches("f", "fetch", "Fetch the current watch status.", false);
    TCLAP::SwitchArg checkWatches("c", "check", "Check to see if any errors or warnings have oclwrred in the lwrrently monitored watches.", false);
    TCLAP::SwitchArg json("j", "json", "Print the output in a json format", cmd, false);
    TCLAP::ValueArg<std::string> setWatches("s", "set", "Set the watches to be monitored. [default = pm]\n a - all watches"
            "\n p - PCIe watches (*)"
            "\n m - memory watches (*)"
            "\n i - infoROM watches"
            "\n t - thermal and power watches (*)"
            "\n n - LWLink watches (*)"
            "\n (*) watch requires 60 sec before first query",
            false, "pm","flags");
    // \n p - MLW and PLW watches \n d - driver watches

    std::vector <TCLAP::Arg *> xors;
    xors.push_back(&getWatches);
    xors.push_back(&setWatches);
    xors.push_back(&clearWatches);
    xors.push_back(&checkWatches);
    cmd.xorAdd(xors);
    cmd.add(&hostAddress);

    helpOutput.addDescription("health --  Used to manage the health watches of a group. The health of the GPUs in a group can then be monitored during a process.");
    helpOutput.addToGroup("get", &hostAddress);
    helpOutput.addToGroup("get", &groupId);    
    helpOutput.addToGroup("get", &getWatches);
    helpOutput.addToGroup("get", &json);

    helpOutput.addToGroup("set", &hostAddress);
    helpOutput.addToGroup("set", &groupId);    
    helpOutput.addToGroup("set", &setWatches);
    helpOutput.addToGroup("get", &json);

    helpOutput.addToGroup("check", &hostAddress);
    helpOutput.addToGroup("check", &groupId);    
    helpOutput.addToGroup("check", &checkWatches);
    helpOutput.addToGroup("get", &json);

    cmd.parse(argc, argv);

    // Check for negative (invalid) inputs
    CHECK_TCLAP_ARG_NEGATIVE_VALUE(groupId, "group");

    Command *cmdh = NULL;

    // Process Get Command
    if (getWatches.isSet())
    {
        cmdh = new GetHealth(hostAddress.getValue(), groupId.getValue(), json.getValue());
    }
    else if (clearWatches.isSet())
    {
        cmdh = new SetHealth(hostAddress.getValue(), groupId.getValue(), 0);
    }
    else if (setWatches.isSet())
    {
        int bitwiseWatches = 0;

        // Some watches are commented out until supported
        for (unsigned int i = 0; i < setWatches.getValue().length(); i++)
        {

            switch(setWatches.getValue().at(i))
            {
                case 'p':
                    bitwiseWatches|= DCGM_HEALTH_WATCH_PCIE;
                    //bitwiseWatches|= DCGM_HEALTH_WATCH_LWLINK;

                    //Check for duplicates
                    checkDuplicateFlagsForSetWatches(i, setWatches.getValue());
                    break;
                case 'm':
                    bitwiseWatches|= DCGM_HEALTH_WATCH_MEM;
                    //bitwiseWatches|= DCGM_HEALTH_WATCH_SM;

                    //Check for duplicates
                    checkDuplicateFlagsForSetWatches(i, setWatches.getValue());
                    break;
                case 'i':
                    bitwiseWatches|= DCGM_HEALTH_WATCH_INFOROM;

                    //Check for duplicates
                    checkDuplicateFlagsForSetWatches(i, setWatches.getValue());
                    break;
                    /*
                case 'u':
                        //bitwiseWatches|= DCGM_HEALTH_WATCH_MLW;
                        //bitwiseWatches|= DCGM_HEALTH_WATCH_PMU;

                case 'd':
                        //bitwiseWatches|= DCGM_HEALTH_WATCH_DRIVER;
                     *
                     */
                case 't':
                    bitwiseWatches|= DCGM_HEALTH_WATCH_THERMAL;
                    bitwiseWatches|= DCGM_HEALTH_WATCH_POWER;

                    //Check for duplicates
                    checkDuplicateFlagsForSetWatches(i, setWatches.getValue());
                    break;
                case 'n':
                    bitwiseWatches|= DCGM_HEALTH_WATCH_LWLINK;
                    
                    //Check for duplicates
                    checkDuplicateFlagsForSetWatches(i, setWatches.getValue());
                    break;
                case 'a':
                    //bitwiseWatches = DCGM_HEALTH_WATCH_ALL;
                    bitwiseWatches|= DCGM_HEALTH_WATCH_PCIE;
                    bitwiseWatches|= DCGM_HEALTH_WATCH_MEM;
                    bitwiseWatches|= DCGM_HEALTH_WATCH_INFOROM;
                    bitwiseWatches|= DCGM_HEALTH_WATCH_THERMAL;
                    bitwiseWatches|= DCGM_HEALTH_WATCH_POWER;
                    bitwiseWatches|= DCGM_HEALTH_WATCH_LWLINK;
                    if (setWatches.getValue().length() > 1)
                    {
                        throw(TCLAP::CmdLineParseException("Invalid flags detected", "set"));
                    }

                    break;

                default:
                    throw(TCLAP::CmdLineParseException("Invalid flags detected", "set"));
            }
        }

        if (bitwiseWatches == 0)
        {
            throw(TCLAP::CmdLineParseException("No flags detected", "set"));
        }

        cmdh = new SetHealth(hostAddress.getValue(), groupId.getValue(), bitwiseWatches);
    }
    else if (checkWatches.isSet())
    {
        cmdh = new CheckHealth(hostAddress.getValue(), groupId.getValue(), json.getValue());
    }

    if (cmdh)
    {
        result = (dcgmReturn_t) cmdh->Execute();
    }

    delete(cmdh);

    return result;
}

void CommandLineParser::ValidateThrottleMask(const std::string &throttleMask)
{
    std::stringstream buf;

    if (throttleMask.size() > DCGM_THROTTLE_MASK_LEN-1)
    {
        throw(TCLAP::CmdLineParseException("throttle-mask has to be under 50 characters"));
    }

    int isdigitRet = std::isdigit(throttleMask[0]);

    for (size_t i = 1; i < throttleMask.size(); i++)
    {
        if (std::isdigit(throttleMask[i]) != isdigitRet)
        {
            buf << "throttle-mask cannot mix numbers and letters, but we detected a mix of letters and numbers '"
                << throttleMask << "'.";
            throw(TCLAP::CmdLineParseException(buf.str()));
        }
    }

    if (isdigitRet)
    {
        // Check for valid numeric input
        uint64_t mask = std::strtoull(throttleMask.c_str(), NULL, 10);

        // Make sure the mask sets only valid bits
        mask &= ~DCGM_CLOCKS_THROTTLE_REASON_HW_SLOWDOWN;
        mask &= ~DCGM_CLOCKS_THROTTLE_REASON_SW_THERMAL;
        mask &= ~DCGM_CLOCKS_THROTTLE_REASON_HW_THERMAL;
        mask &= ~DCGM_CLOCKS_THROTTLE_REASON_HW_POWER_BRAKE;

        if (mask != 0)
        {
            buf << "Detected invalid bits set in the throttle-mask. Valid values for throttle-mask are 0, 8, 32, "
                << "64 and 128, and their additive permutations.";
            throw(TCLAP::CmdLineParseException(buf.str()));
        }
    }
    else
    {
        // Check for valid string input
        std::vector<std::string> reasons;
        tokenizeString(throttleMask, ",", reasons);
        for (size_t i = 0; i < reasons.size(); i++)
        {
            std::string tmp(reasons[i]);
            std::transform(tmp.begin(), tmp.end(), tmp.begin(), ::tolower);
            if ((tmp != HW_SLOWDOWN) && (tmp != SW_THERMAL) && (tmp != HW_THERMAL) && (tmp != HW_POWER_BRAKE))
            {
                buf << "Found '" << reasons[i] << "' in throttle-mask. Valid throttle reasons are limited to "
                    << HW_SLOWDOWN << ", " << SW_THERMAL << ", " << HW_THERMAL << ", and " << HW_POWER_BRAKE
                    << ".";
                throw(TCLAP::CmdLineParseException(buf.str()));
            }
        }
    }
}

int CommandLineParser::processDiagCommandLine(int argc, char *argv[])
{
    // Check for stop diag request
    const char* value = NULL;
    value = std::getelw(STOP_DIAG_ELW_VARIABLE_NAME);
    if (value != NULL)
    {
        AbortDiag abortDiag = AbortDiag(std::string(value));
        return (dcgmReturn_t) abortDiag.Execute();
    }

    dcgmReturn_t result = DCGM_ST_OK;
    std::string myName = "diag";
    DCGMOutput helpOutput;

    DCGMSubsystemCmdLine cmd(myName, _DCGMI_FORMAL_NAME, ' ', _DCGMI_VERSION);
    cmd.setOutput(&helpOutput);

    // args are displayed in reverse order to this list
    //TCLAP::SwitchArg viewDiag("v", "view", "Display diagnostic results.", cmd, false);
    //TCLAP::SwitchArg abortDiag("a", "abort", "Abort the diagnostic. Kills the LWVS/EUD process(es) on entire group", cmd, false); 1 ⊂ 2 ⊂ 3
    TCLAP::ValueArg<std::string> startDiag("r", "run", 
                                           "Run a diagnostic. (Note: higher numbered tests include all beneath.)  \n"
                                           " 1 - Quick (System Validation ~ seconds) \n"
                                           " 2 - Medium (Extended System Validation ~ 2 minutes) \n"
                                           " 3 - Long (System HW Diagnostics ~ 15 minutes) \n"
                                           "Specific tests to run may be specified by name, and multiple tests may be "
                                           "specified as a comma separated list. For example, the command:\n\n"
                                           " dcgmi diag -r \"sm stress,diagnostic\" \n\n"
                                           "would run the SM Stress and Diagnostic tests together.", 
                                           false, "1", "diag", cmd);
    TCLAP::ValueArg<int> groupId("g", "group", "The group ID to query.", false, DCGM_GROUP_ALL_GPUS, "groupId", cmd);
    TCLAP::ValueArg<std::string> hostAddress("", "host", g_hostnameHelpText, false, "localhost", "IP/FQDN", cmd);
    TCLAP::ValueArg<std::string> parms("p", "parameters", "Test parameters to set for this run.", false, "", "test_name.variable_name=variable_value", cmd);
    TCLAP::ValueArg<std::string> config("c", "configfile", "Path to the configuration file.", false, "", "/full/path/to/config/file", cmd);
    TCLAP::ValueArg<std::string> gpuList("i", "gpuList", "A comma-separated list of the gpus on which the diagnostic should run. Cannot be used with -g.", false, "", "gpuList", cmd);
    TCLAP::SwitchArg verbose("v", "verbose", "Show information and warnings for each test.",cmd ,false);
    TCLAP::SwitchArg statsOnFail("", "statsonfail", "Only output the statistics files if there was a failure", cmd,false);
    TCLAP::ValueArg<std::string> debugLogFile("", "debugLogFile", "Encrypted log file for debug information. If a \
            debug level has been specified then \"lwvs.log\" is the default log file.",
            false, "", "debug file", cmd);
    TCLAP::ValueArg<std::string> statsPath("", "statspath", "Write the plugin statistics to the given path \
            rather than the current directory", false, "", "plugin statistics path", cmd);
    TCLAP::ValueArg<int> debugLevel("d", "debugLevel", "Debug level 0-5 with 5 being the most verbose. The \
            logfile can be specified by the --debugLogFile parameter.",
            false, 0, "debug level", cmd);
    TCLAP::SwitchArg json("j", "json", "Print the output in a json format", cmd, false);
    TCLAP::SwitchArg training("", "train",
            "Run the diagnostic iteratively and generate a configuration file of golden values based on those results.",
            cmd, false);
    TCLAP::SwitchArg forceTrain("", "force", "Ignore warnings and train the diagnostic.", cmd, false);

    TCLAP::ValueArg<std::string> throttleMask("", "throttle-mask",
        "Specify which throttling reasons should be ignored. You can provide a comma separated list of reasons. "
        "For example, specifying 'HW_SLOWDOWN,SW_THERMAL' would ignore the HW_SLOWDOWN and SW_THERMAL throttling "
        "reasons. Alternatively, you can specify the integer value of the ignore bitmask. For the bitmask, "
        "multiple reasons may be specified by the sum of their bit masks. For "
        "example, specifying '40' would ignore the HW_SLOWDOWN and SW_THERMAL throttling reasons.\n"
        "\lwalid throttling reasons and their corresponding bitmasks (given in parentheses) are:\n"
        "HW_SLOWDOWN (8)\nSW_THERMAL (32)\nHW_THERMAL (64)\nHW_POWER_BRAKE (128)", false, "", "", cmd);

    TCLAP::ValueArg<std::string> pluginPath("", "plugin-path", "Specify the full path to the DCGM GPU diagnostic \
                                            plugins. For example, to retrive the plugins from a custom dir: \
                                            --plugin-path /home/test/test_plugins_dir/", false, "", 
                                            "/full/path/to/plugin/dir", cmd);

    TCLAP::ValueArg<unsigned int> trainingIterations("", "training-iterations", "The number of iterations to use "
                                                     "while training the diagnostic. The default is 4.",
                                                     false, 4, "training iterations", cmd);
    TCLAP::ValueArg<unsigned int> trainingVariance("", "training-variance", "The coefficient of variance "
                                                   "as a percentage required to trust the data. The default is 5",
                                                   false, 5, "training variance", cmd);
    TCLAP::ValueArg<unsigned int> trainingTolerance("", "training-tolerance", "The percentage the golden value "
                                                    "should be scaled to allow some tolerance when running the "
                                                    "diagnostic later. For example, if the callwlated golden "
                                                    "value for a minimum bandwidth were 9000 and the tolerance "
                                                    "were set to 5, then the minimum bandwidth written to the "
                                                    "configuration file would be 8550, 95% of 9000. The "
                                                    "default value is 5.",
                                                    false, 5, "training tolerance", cmd);
    TCLAP::ValueArg<std::string> goldelwaluesFile("", "golden-values-filename", "Specify the relative path where " 
                                                  "the DCGM GPU diagnostic should save the golden values file "
                                                  "produced in training mode. The path will be appended to /tmp "
                                                  "and will need to be copied elsewhere", false, "",
                                                  "path to golden values file", cmd);
    TCLAP::SwitchArg failEarly("", "fail-early",
        "Enable early failure checks for the Targeted Power, Targeted Stress, SM Stress, and Diagnostic tests. "
        "When enabled, these tests check for a failure once every 5 seconds (can be modified by the "
        "--check-interval parameter) while the test is running instead of a single check performed after the "
        "test is complete. Disabled by default.", cmd, false);

    TCLAP::ValueArg<unsigned int> failCheckInterval("", "check-interval",
        "Specify the interval (in seconds) at which the early failure checks should occur for the "
        "Targeted Power, Targeted Stress, SM Stress, and Diagnostic tests when early failure checks are enabled. "
        "Default is once every 5 seconds. Interval must be between 1 and 300", false, 5, "failure check interval", cmd);

    // Set help output information
    helpOutput.addDescription("diag -- Used to run diagnostics on the system.");
    helpOutput.addFooter("Verbose diagnostic output is lwrrently limited on client, for full diagnostic and validation "
                         "logs please check your server.\n");

    helpOutput.addToGroup("1", &hostAddress);
    helpOutput.addToGroup("1", &groupId);
    helpOutput.addToGroup("1", &startDiag);
    helpOutput.addToGroup("1", &parms);
    helpOutput.addToGroup("1", &config);
    helpOutput.addToGroup("1", &gpuList);
    helpOutput.addToGroup("1", &verbose);
    helpOutput.addToGroup("1", &statsOnFail);
    helpOutput.addToGroup("1", &debugLogFile);
    helpOutput.addToGroup("1", &statsPath);
    helpOutput.addToGroup("1", &json);
    helpOutput.addToGroup("1", &training);
    helpOutput.addToGroup("1", &forceTrain);
    helpOutput.addToGroup("1", &throttleMask);
    helpOutput.addToGroup("1", &pluginPath);
    helpOutput.addToGroup("1", &trainingIterations);
    helpOutput.addToGroup("1", &trainingVariance);
    helpOutput.addToGroup("1", &trainingTolerance);
    helpOutput.addToGroup("1", &goldelwaluesFile);
    helpOutput.addToGroup("1", &failEarly);
    helpOutput.addToGroup("1", &failCheckInterval);

    cmd.parse(argc, argv);

    if (training.isSet() && startDiag.isSet())
    {
        throw(TCLAP::CmdLineParseException("Specifying training and a run option (-r) are mutually exclusive."));
    }
    else if (!training.isSet() && !startDiag.isSet())
    {
        throw(TCLAP::CmdLineParseException("The run option (-r) must always be set unless training mode is being run."));
    }

    // Check for negative (invalid) inputs
    CHECK_TCLAP_ARG_NEGATIVE_VALUE(groupId, "group");
    // Checking string value so macro CHECK_TCLAP_ARG_NEGATIVE_VALUE is not used
    if (strtol(startDiag.getValue().c_str(), NULL, 10) < 0)
    {
        throw(TCLAP::CmdLineParseException("Positive value expected, negative value found", "run"));
    }

    if (gpuList.isSet() && groupId.isSet())
    {
        throw(TCLAP::CmdLineParseException("Specifying a group id and a gpu list are mutually exclusive"));
    }

    if (debugLogFile.getValue().size() > DCGM_FILE_LEN-1)
    {
        throw(TCLAP::CmdLineParseException("debugLogFile has to be under 30 characters"));
    }
    
    if (statsPath.getValue().size() > DCGM_PATH_LEN-1)
    {
        throw(TCLAP::CmdLineParseException("statspath has to be under 128 characters"));
    }

    if (debugLevel.getValue() < 0 || debugLevel.getValue() > 5)
    {
        throw(TCLAP::CmdLineParseException("debugLevel must be a number between 0 and 5, inclusive"));
    }

    // This will throw an exception if an error oclwrs
    ValidateThrottleMask(throttleMask.getValue());

    if (!training.isSet())
    {
        if (forceTrain.isSet())
        {
            throw(TCLAP::CmdLineParseException("--force is only valid with --train"));
        }

        if (trainingIterations.isSet())
        {
            throw(TCLAP::CmdLineParseException("training-iterations is only valid with --train"));
        }
        
        if (trainingVariance.isSet())
        {
            throw(TCLAP::CmdLineParseException("training-variance is only valid with --train"));
        }
        
        if (trainingTolerance.isSet())
        {
            throw(TCLAP::CmdLineParseException("training-tolerance is only valid with --train"));
        }
        
        if (goldelwaluesFile.isSet())
        {
            throw(TCLAP::CmdLineParseException("golden-values-filename is only valid with --train"));
        }
    }

    if (failCheckInterval.isSet() && !failEarly.isSet())
    {
        throw(TCLAP::CmdLineParseException("The --fail-early option must be enabled", "check-interval"));
    }

    if (failCheckInterval.isSet() && (failCheckInterval.getValue() == 0 || failCheckInterval.getValue() > 300))
    {
        throw(TCLAP::CmdLineParseException("Interval value must be between 1 and 300", "check-interval"));
    }

    bool trailwal = training.getValue();
    bool forceTrailwal = forceTrain.getValue();

    std::string rulwalue;
    if (startDiag.isSet())
    {
        rulwalue = startDiag.getValue();
    }

    dcgmRunDiag_t drd = { 0 };
    drd.version = dcgmRunDiag_version;
    std::string error;
    result = dcgm_diag_common_populate_run_diag(drd, rulwalue, parms.getValue(), "",
                                                gpuList.getValue(), verbose.getValue(), statsOnFail.getValue(),
                                                debugLogFile.getValue(), statsPath.getValue(),
                                                debugLevel.getValue(), throttleMask.getValue(),
                                                pluginPath.getValue(), trailwal, forceTrailwal,
                                                trainingIterations.getValue(), trainingVariance.getValue(),
                                                trainingTolerance.getValue(), goldelwaluesFile.getValue(),
                                                groupId.getValue(), failEarly.getValue(),
                                                failCheckInterval.getValue(), error);

    if (result == DCGM_ST_BADPARAM)
    {
        throw(TCLAP::CmdLineParseException(error));
    }

    Command *cmdd = new StartDiag(hostAddress.getValue(), parms.getValue(), config.getValue(), json.getValue(),
                                  drd, argv[0]);

    if (cmdd)
    {
        result = (dcgmReturn_t) cmdd->Execute();
    }

    delete(cmdd);

    return result;
}

int CommandLineParser::processStatsCommandLine(int argc, char *argv[])
{
    dcgmReturn_t result = DCGM_ST_OK;
    std::string myName = "stats";
    DCGMOutput helpOutput;

    DCGMSubsystemCmdLine cmd(myName, _DCGMI_FORMAL_NAME, ' ', _DCGMI_VERSION);
    cmd.setOutput(&helpOutput);

    // args are displayed in reverse order to this list
    TCLAP::SwitchArg verbose("v", "verbose", "Show process information for each GPU.",cmd ,false);
    TCLAP::ValueArg<std::string> jobStart("s", "jstart", "Start recording job statistics.", false, "Default Job", "job id");
    TCLAP::ValueArg<std::string> jobStop("x", "jstop", "Stop recording job statistics.", false, "Default Job", "job id");
    TCLAP::ValueArg<std::string> jobStats("j", "job", "Display job statistics.", false, "Default Job", "job id");
    TCLAP::ValueArg<std::string> jobRemove("r", "jremove", "Remove job statistics.", false, "Default Job", "job id");
    TCLAP::ValueArg<int> pid("p", "pid", "View statistics for the specified pid.", false, 1, "pid");
    TCLAP::SwitchArg enableWatches("e", "enable", "Enable system watches and start recording information.",false);
    TCLAP::SwitchArg disableWatches("d", "disable", "Disable system watches and stop recording information.",false);
    TCLAP::SwitchArg jobRemoveAll("a", "jremoveall", "Remove all job statistics.",false);
    TCLAP::ValueArg<int> groupId("g", "group", "The GPU group to query on the specified host.", false, DCGM_GROUP_ALL_GPUS, "groupId", cmd);
    TCLAP::ValueArg<std::string> hostAddress("", "host", g_hostnameHelpText, false, "localhost", "IP/FQDN");

    std::vector <TCLAP::Arg *> xors;
    xors.push_back(&pid);
    xors.push_back(&enableWatches);
    xors.push_back(&disableWatches);
    xors.push_back(&jobStart);
    xors.push_back(&jobStop);
    xors.push_back(&jobStats);
    xors.push_back(&jobRemove);
    xors.push_back(&jobRemoveAll);
    cmd.xorAdd(xors);
    cmd.add(&hostAddress);

    // Set help output information

    std::string footer =

            "Process Statistics Information:\n"

            "\n--  Exelwtion Stats --\n"
            "Start Time                (*) - Process start time\n"
            "End Time                  (*) - Process end time\n"
            "Total Exelwtion Time      (*) - Total exelwtion time in seconds\n"
            "No. Conflicting Processes (*) - Number of other processes that ran\n"
            "Conflicting Compute PID       - PID of conflicting compute process\n"
            "Conflicting Graphics PID      - PID of conflicting graphics process\n"

            "\n--  Performance Stats --\n"
            "Energy Consumed         - Total energy consumed during process in joules\n"
            "Max GPU Memory Used (*) - Maximum amount of GPU memory used in bytes\n"
            "SM Clock                - Statistics for SM clocks(s) in MHz\n"
            "Memory Clock            - Statistics for memory clock(s) in MHz\n"
            "SM Utilization          - Utilization of the GPU's SMs in percent\n"
            "Memory Utilization      - Utilization of the GPU's memory in percent\n"
            "PCIe Rx Bandwidth       - PCIe bytes read from the GPU\n"
            "PCIe Tx Bandwidth       - PCIe bytes written to the GPU\n"

            "\n--  Event Stats --\n"
            "Single Bit ECC Errors - Number of ECC single bit errors that oclwrred\n"
            "Double Bit ECC Errors - Number of ECC double bit errors that oclwrred\n"
            "PCIe Replay Warnings  - Number of PCIe replay warnings\n"
            "Critical XID Errors   - Number of critical XID Errors\n"
            "XID                   - Time of XID error in since start of process\n"

            "\n--  Slowdown Stats --\n"
            "Power           - Runtime % at reduced clocks due to power violation\n"
            "Thermal         - Runtime % at reduced clocks due to thermal limit\n"
            "Reliability     - Runtime % at reduced clocks due to reliability limit\n"
            "Board Limit     - Runtime % at reduced clocks due to board's voltage limit\n"
            "Low Utilization - Runtime % at reduced clocks due to low utilization\n"
            "Sync Boost      - Runtime % at reduced clocks due to sync boost\n"
            "\n(*) Represents a process statistic. Otherwise device statistic during"
            "\n     process lifetime listed.";


    helpOutput.addFooter(footer);
    helpOutput.addDescription("stats -- Used to view process statistics.");

    helpOutput.addToGroup("1", &hostAddress);
    helpOutput.addToGroup("1", &groupId);
    helpOutput.addToGroup("1", &enableWatches);

    helpOutput.addToGroup("2", &hostAddress);
    helpOutput.addToGroup("2", &groupId);
    helpOutput.addToGroup("2", &disableWatches);

    helpOutput.addToGroup("3", &hostAddress);
    helpOutput.addToGroup("3", &groupId);
    helpOutput.addToGroup("3", &pid);
    helpOutput.addToGroup("3", &verbose);

    helpOutput.addToGroup("4", &hostAddress);
    helpOutput.addToGroup("4", &groupId);
    helpOutput.addToGroup("4", &jobStart);

    helpOutput.addToGroup("5", &hostAddress);
    helpOutput.addToGroup("5", &jobStop);

    helpOutput.addToGroup("6", &hostAddress);
    helpOutput.addToGroup("6", &jobStats);
    helpOutput.addToGroup("6", &verbose);

    helpOutput.addToGroup("6", &hostAddress);
    helpOutput.addToGroup("6", &jobRemove);
    helpOutput.addToGroup("6", &verbose);

    helpOutput.addToGroup("7", &hostAddress);
    helpOutput.addToGroup("7", &jobRemoveAll);
    helpOutput.addToGroup("7", &verbose);

    cmd.parse(argc, argv);

    // Check for negative (invalid) inputs
    CHECK_TCLAP_ARG_NEGATIVE_VALUE(groupId, "group");
    CHECK_TCLAP_ARG_NEGATIVE_VALUE(pid, "pid");

    Command *cmds = NULL;

    if (enableWatches.isSet())
    {
        cmds = new EnableWatches(hostAddress.getValue(),groupId.getValue());
    } 
    else if (disableWatches.isSet())
    {
        cmds = new DisableWatches(hostAddress.getValue(),groupId.getValue());
    } 
    else if (pid.isSet())
    {
        cmds = new ViewProcessStats(hostAddress.getValue(), groupId.getValue(), pid.getValue(), verbose.getValue());
    } 
    else if (jobStart.isSet())
    {
        cmds = new StartJob(hostAddress.getValue(), groupId.getValue(), jobStart.getValue());
    }
    else if (jobStop.isSet())
    {
        cmds = new StopJob(hostAddress.getValue(), jobStop.getValue());
    }
    else if (jobStats.isSet())
    {
        cmds = new ViewJobStats(hostAddress.getValue(), jobStats.getValue(), verbose.getValue());
    }
    else if (jobRemove.isSet())
    {
        cmds = new RemoveJob(hostAddress.getValue(), jobRemove.getValue());
    }
    else if (jobRemoveAll.isSet())
    {
        cmds = new RemoveAllJobs(hostAddress.getValue());
    }

    if (cmds){
        result = (dcgmReturn_t) cmds->Execute();
    }

    delete(cmds);

    return result;
}

int CommandLineParser::processTopoCommandLine(int argc, char *argv[])
{
    dcgmReturn_t result = DCGM_ST_OK;
    std::string myName = "topo";
    DCGMOutput helpOutput;

    DCGMSubsystemCmdLine cmd(myName, _DCGMI_FORMAL_NAME, ' ', _DCGMI_VERSION);
    cmd.setOutput(&helpOutput);

    // args are displayed in reverse order to this list
    TCLAP::ValueArg<int> groupId("g", "group", "The group ID to query.", false, DCGM_GROUP_ALL_GPUS, "groupId", cmd);
    TCLAP::ValueArg<int> gpuId("", "gpuid", "The GPU ID to query.", false, 1, "gpuId", cmd);
    TCLAP::ValueArg<std::string> hostAddress("", "host", g_hostnameHelpText, false, "localhost", "IP/FQDN", cmd);
    TCLAP::SwitchArg json("j", "json", "Print the output in a json format", cmd, false);

    // Set help output information
    helpOutput.addDescription("topo -- Used to find the topology of GPUs on the system.");

    helpOutput.addToGroup("1", &hostAddress);
    helpOutput.addToGroup("1", &groupId);
    helpOutput.addToGroup("1", &json);

    helpOutput.addToGroup("2", &hostAddress);
    helpOutput.addToGroup("2", &gpuId);
    helpOutput.addToGroup("2", &json);

    cmd.parse(argc, argv);

    // Check for negative (invalid) inputs
    CHECK_TCLAP_ARG_NEGATIVE_VALUE(groupId, "group");
    CHECK_TCLAP_ARG_NEGATIVE_VALUE(gpuId, "gpuid");

    Command *cmdt = NULL;

    if (gpuId.isSet() && groupId.isSet())
    {
        throw(TCLAP::CmdLineParseException("Both GPU and group ID are given"));
    }

    if (gpuId.isSet())
    {
        cmdt = new GetGPUTopo(hostAddress.getValue(), gpuId.getValue(), json.getValue());
    }
    else 
    {
        cmdt = new GetGroupTopo(hostAddress.getValue(), groupId.getValue(), json.getValue());
    }

    if (cmdt)
    {
        result = (dcgmReturn_t) cmdt->Execute();
    }

    delete(cmdt);

    return result;
}

int CommandLineParser::processIntrospectCommandLine(int argc, char *argv[])
{
    dcgmReturn_t result = DCGM_ST_OK;
    std::string myName = "introspect";
    DCGMOutput helpOutput;

    DCGMSubsystemCmdLine cmd(myName, _DCGMI_FORMAL_NAME, ' ', _DCGMI_VERSION);
    cmd.setOutput(&helpOutput);

    // args are displayed in reverse order to this list
    TCLAP::SwitchArg show("s", "show",
            "Show introspection info for the given target(s).  "
            "Must be accompanied by at least one of --hostengine, --all-fields, or --field-group.",false);
    TCLAP::SwitchArg enable("e", "enable",
            "Enable introspection and start recording information.",false);
    TCLAP::SwitchArg disable("d", "disable",
            "Disable introspection and stop recording information.",false);
    TCLAP::ValueArg<std::string> hostAddress("",
            "host", g_hostnameHelpText,
            false, "localhost", "IP/FQDN");

    std::vector <TCLAP::Arg *> cmdXors;
    cmdXors.push_back(&enable);
    cmdXors.push_back(&disable);
    cmdXors.push_back(&show);

    TCLAP::SwitchArg hostengineTarget("H", "hostengine",
            "Specify the hostengine process as a target to retrieve introspection stats for.",
            false);
    TCLAP::SwitchArg allFieldsTarget("F", "all-fields",
            "Specify an aggregation of all watched fields as a target to retrieve introspection stats for.",
            false);
    TCLAP::SwitchArg allFieldGroupsTarget("", "all-field-groups",
                "Specify an aggregation of all field groups as a target to retrieve introspection stats for.",
                false);

    stringstream fgDescr;
    fgDescr << "Specify a field group ID as a target to retrieve introspection stats for. Use \"all\" to specify all field groups.";

    TCLAP::MultiArg<string> fieldGroupTarget("f", "field-group",
            fgDescr.str(), false, "");

    cmd.xorAdd(cmdXors);
    cmd.add(&hostAddress);
    cmd.add(&hostengineTarget);
    cmd.add(&allFieldsTarget);
    cmd.add(&fieldGroupTarget);
    cmd.add(&allFieldGroupsTarget);

    // Set help output information
    helpOutput.addDescription("introspect -- Used to access info about DCGM itself.");

    helpOutput.addToGroup("enable", &hostAddress);
    helpOutput.addToGroup("enable", &enable);

    helpOutput.addToGroup("disable", &hostAddress);
    helpOutput.addToGroup("disable", &disable);

    helpOutput.addToGroup("summary", &hostAddress);
    helpOutput.addToGroup("summary", &show);
    helpOutput.addToGroup("summary", &hostengineTarget);
    helpOutput.addToGroup("summary", &allFieldsTarget);
    helpOutput.addToGroup("summary", &fieldGroupTarget);
    helpOutput.addToGroup("summary", &allFieldGroupsTarget);

    cmd.parse(argc, argv);

    Command *cmdt = NULL;


    if (enable.isSet())
    {
        cmdt = new ToggleIntrospect(hostAddress.getValue(), true);
    }
    else if (disable.isSet()) 
    {
        cmdt = new ToggleIntrospect(hostAddress.getValue(), false);
    }
    else if (show.isSet()) 
    {
        if (!hostengineTarget.isSet() && !allFieldsTarget.isSet() && !fieldGroupTarget.isSet() && !allFieldGroupsTarget.isSet()) 
        {
            throw(TCLAP::CmdLineParseException("No target specified to show introspection stats for. "
                                               "See \"dcgmi introspect --help\""));
        }

        std::vector<string> fgIds = fieldGroupTarget.getValue();
        std::vector<dcgmFieldGrp_t> fgTargets;
        bool showAllFieldGroups = allFieldGroupsTarget.getValue();

        // validate field group arguments
        if (fieldGroupTarget.isSet() && fgIds.size() > 0)
        {
            if (std::find(fgIds.begin(), fgIds.end(), "all") != fgIds.end() && fgIds.size() > 1)
            {
                throw(TCLAP::CmdLineParseException("Cannot specify other field groups when \"all\" field groups are "
                                                   "specified"));
            }
            else if (fgIds.at(0).compare("all") == 0)
            {
                showAllFieldGroups = true;
            }
            else
            {
                // parse the given field collections
                for (size_t i = 0; i < fgIds.size(); ++i)
                {
                    bool success = false;
                    unsigned long long fgId = strTo<unsigned long long>(fgIds.at(i), &success);
                    fgTargets.push_back((dcgmFieldGrp_t)fgId);
                }
            }
        }

        cmdt = new DisplayIntrospectSummary(hostAddress.getValue(),
                                            hostengineTarget.getValue(),
                                            allFieldsTarget.getValue(),
                                            showAllFieldGroups,
                                            fgTargets);
    }

    if (cmdt)
    {
        result = (dcgmReturn_t) cmdt->Execute();
        delete(cmdt);
    }

    return result;
}

int CommandLineParser::processLwlinkCommandLine(int argc,char * argv [ ])
{
    dcgmReturn_t result = DCGM_ST_NOT_CONFIGURED;
    std::string myName = "lwlink";
    DCGMOutput helpOutput;

    DCGMSubsystemCmdLine cmd(myName, _DCGMI_FORMAL_NAME, ' ', _DCGMI_VERSION);
    cmd.setOutput(&helpOutput);

    TCLAP::ValueArg<int> gpuId("g", "gpuid", "The GPU ID to query. Required for -e", false, 1, "gpuId", cmd);
    TCLAP::ValueArg<std::string> hostAddress("", "host", g_hostnameHelpText, false, "localhost", "IP/FQDN", cmd);
    TCLAP::SwitchArg json("j", "json", "Print the output in a json format", cmd, false);

    TCLAP::SwitchArg printLwLinkErrors("e", "errors", "Print LwLink errors for a given gpuId (-g).", false);
    TCLAP::SwitchArg printLinkStatus("s", "link-status", "Print LwLink link status for all GPUs and LwSwitches in the system.", false);

    std::vector <TCLAP::Arg *> xors;
    xors.push_back(&printLwLinkErrors);
    xors.push_back(&printLinkStatus);
    cmd.xorAdd(xors);
    
    helpOutput.addDescription(
        "lwlink -- Used to get LwLink link status or error counts for GPUs and LwSwitches in the system"
        "\n\n LWLINK Error description "
        "\n========================="
        "\n CRC FLIT Error => Data link receive flow control digit CRC error."
        "\n CRC Data Error => Data link receive data CRC error."
        "\n Replay Error   => Data link transmit replay error."
        "\n Recovery Error => Data link transmit recovery error.");

    
    helpOutput.addToGroup("1", &hostAddress);
    helpOutput.addToGroup("1", &gpuId);
    helpOutput.addToGroup("1", &printLwLinkErrors);
    helpOutput.addToGroup("1", &json);

    helpOutput.addToGroup("2", &hostAddress);
    helpOutput.addToGroup("2", &printLinkStatus);

    cmd.parse(argc, argv);

    if (gpuId.isSet() && gpuId.getValue() < 0) 
    {
        throw(TCLAP::CmdLineParseException("Positive value expected, negative value found", "gpuid"));
    }
    
    Command *cmdn = NULL;
    
    if (printLwLinkErrors.isSet()) 
    {
         if (!gpuId.isSet())
         {
            throw(TCLAP::CmdLineParseException("GPU ID is required (--gpuid)", "errors"));
         }

         cmdn = new GetGpuLwlinkErrorCounts(hostAddress.getValue(), gpuId.getValue(), json.getValue());
    }
    else if (printLinkStatus.isSet()) 
    {
        cmdn = new GetLwLinkLinkStatuses(hostAddress.getValue());
    }

    if (cmdn)
    {
        result = (dcgmReturn_t)cmdn->Execute();
    }

    delete(cmdn);
    return result;
}

int CommandLineParser::processDmonCommandLine(int argc,char * argv [])
{
    dcgmReturn_t result = DCGM_ST_OK;
    std::string myName = "dmon";
    DCGMOutput helpOutput;
    unsigned int gpuIdList[DCGM_MAX_NUM_DEVICES] = {0};
    int number_of_gpus;
    std::string gpu_id_str;
    Command *cmdn = NULL;

    DCGMSubsystemCmdLine cmd(myName, _DCGMI_FORMAL_NAME, ' ', _DCGMI_VERSION);
    cmd.setOutput(&helpOutput);

    TCLAP::ValueArg<std::string> hostAddress("", "host", g_hostnameHelpText, false, "localhost", "IP/FQDN", cmd);
    
    TCLAP::ValueArg<std::string> gpuId("i", "gpu-id", " The comma seperated list of GPU IDs to run the dmon on. "
                                       "Default is -1 which runs for all supported GPU. Run dcgmi discovery -l to "
                                       "check list of available GPUs", false, "-1", "gpuId", cmd);
    
    TCLAP::ValueArg<std::string> groupId("g", "group-id", " The group to query on the specified host.", false, "-1", "groupId", cmd);
    TCLAP::ValueArg<int> fieldGroupId("f", "field-group-id", " The field group to query on the specified host.", false, 0, "fieldGroupId");
    TCLAP::ValueArg<std::string> fieldId("e", "field-id", " Field identifier to view/inject.", false, "-1", "fieldId");
    TCLAP::ValueArg<int> delay("d", "delay", " In milli seconds. Integer representing how often to query results from DCGM and print them for all of the GPUs. [default = 1000 msec,  Minimum value = 1 msec.]", false, 1000, "delay");
    
    TCLAP::ValueArg<int> count("c", "count", " Integer representing How many times to loop before exiting. [default- runs forever.]", false, 0 , "count");
    TCLAP::SwitchArg list("l", "list", "List to look up the long names, short names and field ids.", false);
    TCLAP::SwitchArg noWatch("", "no-watch", "Do not watch the underlying fields. This is useful for observing fields that can only have a single watcher like DCP field IDs 1001-1012. "
                             "If these fields are not already watched, N/W will be displayed.", false);

    // Set help output information
    helpOutput.addDescription("dmon -- Used to monitor GPUs and their stats.");
    helpOutput.addToGroup("1", &gpuId);
    helpOutput.addToGroup("1", &groupId);
    helpOutput.addToGroup("1", &fieldGroupId);
    helpOutput.addToGroup("1", &fieldId);
    helpOutput.addToGroup("1", &delay);
    helpOutput.addToGroup("1", &count);
    helpOutput.addToGroup("1", &list);
    helpOutput.addToGroup("1", &noWatch);

    std::vector <TCLAP::Arg *> xors_fields;
    xors_fields.push_back(&fieldGroupId);
    xors_fields.push_back(&fieldId);
    xors_fields.push_back(&list);
    cmd.xorAdd(xors_fields);
    cmd.add(&delay);
    cmd.add(&count);
    cmd.add(&noWatch);

    cmd.parse(argc, argv);

    if (groupId.isSet() && gpuId.isSet())
    {
        throw(TCLAP::CmdLineParseException("Both gpu-id and group-id are given. Only one of the options must be present"));
    }
    
    if (fieldId.isSet() && fieldGroupId.isSet())
    {
        throw(TCLAP::CmdLineParseException("Both field-id and field-group-id specified at command line. "
                                           "Only one of the options must be present"));
    }

    if (list.isSet() && (groupId.isSet() || gpuId.isSet() || fieldId.isSet() || fieldGroupId.isSet() || fieldId.isSet() 
                         || fieldGroupId.isSet() || delay.isSet() || count.isSet()))
    {
        throw(TCLAP::CmdLineParseException("Invalid parameters with list arg. Usage : dmon -l"));
    }

    if (list.isSet())
    {
        cmdn = new DeviceInfo(hostAddress.getValue(), gpuId.getValue(), groupId.getValue(), 
                              fieldId.getValue(), fieldGroupId.getValue(), delay.getValue(), count.getValue(), true, !noWatch.isSet());

        if (cmdn)
        {
            result = (dcgmReturn_t)cmdn->Execute();
        }

        if (cmdn)
            delete(cmdn);

        return result;
    }

    // Check for negative + invalid inputs
    CHECK_TCLAP_ARG_NEGATIVE_VALUE(count, "count");
    CHECK_TCLAP_ARG_NEGATIVE_VALUE(fieldGroupId, "field-group-id");
    if (delay.getValue() < 1){
        throw(TCLAP::CmdLineParseException("Invalid value", "delay"));
    }
    if (fieldId.isSet() && (!fieldId.getValue().size() || !isdigit(fieldId.getValue().at(0)))) 
    {
        throw(TCLAP::CmdLineParseException("Invalid value", "field-id"));
    }

    cmdn = new DeviceInfo(hostAddress.getValue(), gpuId.getValue(), groupId.getValue(), fieldId.getValue(), 
                          fieldGroupId.getValue(), delay.getValue(), count.getValue(), false, !noWatch.isSet());
    
    if (cmdn)
    {
        result = (dcgmReturn_t)cmdn->Execute();
    }

    if (cmdn)
    {
        delete(cmdn);
    }

    return result;
}

int CommandLineParser::processProfileCommandLine(int argc,char * argv [])
{
    const std::string myName = "profile";
    dcgmReturn_t result = DCGM_ST_OK;
    DCGMOutput helpOutput;
    Command *cmdn = NULL;

    DCGMSubsystemCmdLine cmd(myName, _DCGMI_FORMAL_NAME, ' ', _DCGMI_VERSION);
    cmd.setOutput(&helpOutput);

    helpOutput.addDescription("profile -- View available profiling metrics for GPUs");

    TCLAP::ValueArg<std::string> hostAddress("", "host", g_hostnameHelpText, false, "localhost", "IP/FQDN", cmd);
    TCLAP::SwitchArg json("j", "json", "Print the output in a json format", cmd, false);

    /* Available subcommands */
    std::vector <TCLAP::Arg *> xors;
    TCLAP::SwitchArg list("l", "list", "List available profiling metrics for a GPU or group of GPUs", false);
    TCLAP::SwitchArg pauseArg("", "pause", " Pause DCGM profiling in order to run LWPU developer tools like lwperf, nsight compute, or nsight systems.", false);
    TCLAP::SwitchArg resumeArg("", "resume", " Resume DCGM profiling that was paused previously with --pause.", false);
    xors.push_back(&list);
    xors.push_back(&pauseArg);
    xors.push_back(&resumeArg);
    cmd.xorAdd(xors);

    /* A list of GPUs or a groupId could be provided. Otherwise, we assume all GPUs are desired */
    TCLAP::ValueArg<std::string> gpuIds("i", "gpu-id", " The comma seperated list of GPU IDs to query. "
                                        "Default is supported GPUs on the system. Run dcgmi discovery "
                                        "-l to check list of GPUs available", false, "-1", "gpuId",cmd);
    TCLAP::ValueArg<std::string> groupId("g", "group-id", " The group of GPUs to query on the specified host.", false, "-1", "groupId", cmd);

    cmd.parse(argc, argv);

    if (groupId.isSet() && gpuIds.isSet())
    {
        throw(TCLAP::CmdLineParseException("Both GPU and group ID specified. Please use only one at a time"));
    }

    if (pauseArg.isSet() && resumeArg.isSet())
    {
        throw(TCLAP::CmdLineParseException("Both --pause and --resume were specified. Please use only one at a time"));
    }

    if (pauseArg.isSet())
    {
        cmdn = new DcgmiProfileSetPause(hostAddress.getValue(), true);
    }
    else if (resumeArg.isSet())
    {
        cmdn = new DcgmiProfileSetPause(hostAddress.getValue(), false);
    }
    else if (list.isSet())
    {
        cmdn = new DcgmiProfileList(hostAddress.getValue(), gpuIds.getValue(), 
                                    groupId.getValue(), json.getValue());
    }
    else
    {
        throw(TCLAP::CmdLineParseException("No argument provided for the profile subsystem"));
    }

    if (cmdn) 
    {
        result = (dcgmReturn_t)cmdn->Execute();
        delete(cmdn);
    }

    return result;
}

int CommandLineParser::processModuleCommandLine(int argc,char * argv [])
{
    const std::string myName = "modules";
    dcgmReturn_t result = DCGM_ST_OK;
    DCGMOutput helpOutput;
    Command *cmdn = NULL;

    DCGMSubsystemCmdLine cmd(myName, _DCGMI_FORMAL_NAME, ' ', _DCGMI_VERSION);
    cmd.setOutput(&helpOutput);

    helpOutput.addDescription("modules -- Control and list DCGM modules");

    TCLAP::ValueArg<std::string> hostAddress("", "host", g_hostnameHelpText, false, "localhost", "IP/FQDN", cmd);
    TCLAP::SwitchArg json("j", "json", "Print the output in a json format", cmd, false);
    TCLAP::SwitchArg list("l", "list", "List modules on hostengine", false);
    TCLAP::ValueArg<std::string> blacklist("", "blacklist", "Blacklist provided module", false, "", "Name");

    std::vector <TCLAP::Arg *> xors;
    xors.push_back(&list);
    xors.push_back(&blacklist);
    cmd.xorAdd(xors);

    cmd.parse(argc, argv);

    if (list.isSet()) 
    {
        cmdn = new ListModule(hostAddress.getValue(), json.getValue());
    }
    else if (blacklist.isSet()) 
    {
        cmdn = new BlacklistModule(hostAddress.getValue(), blacklist.getValue(), json.getValue());
    }
    // No else clause since xors ensures one of list or blacklist is always set

    if (NULL != cmdn) 
    {
        result = (dcgmReturn_t)cmdn->Execute();
        delete(cmdn);
    }

    return result;
}

#ifdef DEBUG
int CommandLineParser::processAdminCommandLine(int argc, char *argv[])
{
    dcgmReturn_t result = DCGM_ST_OK;
    std::string myName = "test";
    DCGMOutput helpOutput;

    DCGMSubsystemCmdLine cmd(myName, _DCGMI_FORMAL_NAME, ' ', _DCGMI_VERSION);
    cmd.setOutput(&helpOutput);

    TCLAP::ValueArg<std::string> loadCache("l", "loadcache", "Load the cache values from a file.", true, "cache.file", "filename");
    TCLAP::ValueArg<std::string> saveCache("s", "savecache", "Save the cache values to a file.", true, "cache.file", "filename");
    TCLAP::ValueArg<int> gpuId("", "gpuid", "The GPU ID to query on the specified host.", false, 1, "gpuId", cmd);
    TCLAP::ValueArg<int> groupId("g", "group", "The group to query on the specified host.", false, DCGM_GROUP_ALL_GPUS, "groupId", cmd);
    TCLAP::ValueArg<std::string> fieldId("f", "field", "Field identifier to view/inject.", false, "1", "fieldId", cmd);
    TCLAP::ValueArg<std::string> injectValue("v", "value", "Value to inject.", false, "0", "value", cmd);
    TCLAP::ValueArg<int> timeVal("", "in", "Number of seconds into the future in which to inject the data.", false, 1, "sec", cmd);
    TCLAP::SwitchArg introspect("", "introspect", "View values (injected and non injected) in cache." ,false);
    TCLAP::SwitchArg inject("", "inject", "Inject values into cache.", false);
    TCLAP::ValueArg<std::string> hostAddress("", "host", g_hostnameHelpText, false, "localhost", "IP/FQDN", cmd);

    std::vector <TCLAP::Arg *> xors;
    xors.push_back(&saveCache);
    xors.push_back(&loadCache);
    xors.push_back(&introspect);
    xors.push_back(&inject);
    cmd.xorAdd(xors);

    helpOutput.addToGroup("1", &hostAddress);
    helpOutput.addToGroup("1", &loadCache);

    helpOutput.addToGroup("2", &hostAddress);
    helpOutput.addToGroup("2", &introspect);
    helpOutput.addToGroup("2", &gpuId);
    helpOutput.addToGroup("2", &fieldId);

    helpOutput.addToGroup("3", &hostAddress);
    helpOutput.addToGroup("3", &inject);
    helpOutput.addToGroup("3", &gpuId);
    helpOutput.addToGroup("3", &fieldId);
    helpOutput.addToGroup("3", &injectValue);

    // add an option here if there is anything other than DCGM_STATS_FILE_TYPE_JSON possible
    cmd.parse(argc, argv);

    // Check for negative (invalid) inputs
    CHECK_TCLAP_ARG_NEGATIVE_VALUE(groupId, "group");
    CHECK_TCLAP_ARG_NEGATIVE_VALUE(gpuId, "gpuid");
    CHECK_TCLAP_ARG_NEGATIVE_VALUE(timeVal, "in");

    Command *cmdd = NULL;

    if (gpuId.isSet() && groupId.isSet())
    {
        throw(TCLAP::CmdLineParseException("Both group and GPU ID set. Please use only one at a time"));
    }

    if (loadCache.isSet())
    {
        cmdd = new LoadSaveCache(hostAddress.getValue(), loadCache.getValue(), false);
    }
    else if (saveCache.isSet())
    {
        cmdd = new LoadSaveCache(hostAddress.getValue(), saveCache.getValue(), true);
    }
    else if (introspect.isSet())
    {
        unsigned int gId = groupId.isSet()? groupId.getValue() : gpuId.getValue();
        cmdd = new IntrospectCache(hostAddress.getValue(), gId, fieldId.getValue(), groupId.isSet());
    }
    else if (inject.isSet())
    {
        if (groupId.isSet())
        {
            throw(TCLAP::CmdLineParseException("Injection can only be specified for a single GPU"));
        }

        cmdd = new InjectCache(hostAddress.getValue(), gpuId.getValue(), fieldId.getValue(), timeVal.getValue(), injectValue.getValue());
    }

    if (cmdd)
    {
        result = (dcgmReturn_t) cmdd->Execute();
    }

    delete(cmdd);

    return result;
}
#endif
