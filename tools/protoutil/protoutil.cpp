/*
 * LWIDIA_COPYRIGHT_BEGIN
 *
 * Copyright 2017-2019 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * LWIDIA_COPYRIGHT_END
 */

#include "protoutil_common.h"
#include "protoutil_commands.h"

#include <string>
#include <iostream>
#include <fstream>
#include <iomanip>

#include "topology.pb.h"

// boost
#include <boost/program_options.hpp>
#include <boost/range/algorithm.hpp>

namespace po = boost::program_options;
using namespace std;

//------------------------------------------------------------------------------
static void PrintUsage
(
    string                                     exeName
   ,string                                     command
   ,const po::options_description &            visibleOptions
   ,const po::options_description &            hiddenOptions
   ,const po::positional_options_description & positionalOptions)
{
    string optionalNoParam = "[-";
    string optionalParam;
    string requiredParam;
    bool bOptionalNoParamFound = false;
    bool bAnyParamFound = false;

    string baseExeName = exeName;
    if (exeName.find_last_of('/') != string::npos)
    {
        baseExeName = exeName.substr(exeName.find_last_of('/') + 1);
    }
    // First create the usage string for the command line by iterating
    // through all the visible options and constructing appropriate strings for
    // each type of option
    for (auto const & lwrOption : visibleOptions.options())
    {
        string optStr =
            lwrOption->canonical_display_name(po::command_line_style::allow_dash_for_short);

        auto semantic = lwrOption->semantic();

        if (semantic->is_required())
        {
            if (semantic->max_tokens() > 0 )
            {
                requiredParam += " ";
                requiredParam += optStr + " ARG";
                bAnyParamFound = true;
            }
        }
        else
        {
            if (semantic->max_tokens() > 0 )
            {
                optionalParam += " ";
                optionalParam += "[" + optStr + " ARG]";
                bAnyParamFound = true;
            }
            else
            {
                optionalNoParam += optStr.substr(optStr.find_first_not_of('-'));
                bOptionalNoParamFound = true;
                bAnyParamFound = true;
            }
        }
    }

    // Print out the usage line of the exelwtable normal option types
    cout << "Protobuf file utility\n";
    cout << "Usage : " << baseExeName << " ";
    if (bOptionalNoParamFound)
    {
        cout << optionalNoParam << "]";
    }

    cout << optionalParam  << requiredParam;
    if (bAnyParamFound)
        cout << " ";

    // Now loop through all the positional options and print them
    set<string> printedPos;
    const size_t maxPoCount = hiddenOptions.options().size();
    int itemCount = 1;

    // Only print each positional option once (with an indicator of how many
    // times it can be specified
    bool bStopParsing = false;
    bool bArgOptional = false;
    for (unsigned lwrPos = 0;
         lwrPos < positionalOptions.max_total_count() &&  !bStopParsing;
         lwrPos++)
    {
        string name = positionalOptions.name_for_position(lwrPos);
        if (printedPos.count(name))
        {
            itemCount++;
            continue;
        }
        else
        {
            if (itemCount > 1)
                cout << " ...";
            if (bArgOptional)
                cout << "]";
            itemCount = 1;
        }

        boost::shared_ptr<const boost::program_options::value_semantic> semantic;
        bool bFoundOption = false;
        for (auto const & foundOption : hiddenOptions.options())
        {
            if (foundOption->canonical_display_name() == name)
            {
                bFoundOption = true;
                semantic = foundOption->semantic();
                break;
            }
        }
        if (!bFoundOption)
            continue;

        if (printedPos.size())
            cout << " ";

        printedPos.insert(name);
        bArgOptional = !semantic->is_required();
        if (bArgOptional)
            cout << "[";
        cout << name;
        if ((printedPos.size() == maxPoCount) &&
            (positionalOptions.max_total_count() == numeric_limits<unsigned>::max()))
        {
            cout << "...]";
            bStopParsing = true;
        }
    }
    if (itemCount > 1)
        cout << " ...";
    if (bArgOptional)
        cout << "]";

    // Now print the detailed description of each option
    cout << "\n\n" << visibleOptions;
    for (auto const & lwrOption : hiddenOptions.options())
    {
        string name = lwrOption->format_name();
        name = name.substr(name.find_first_not_of("-"));
        if (command == name)
            continue;
        cout << "  " << setw(visibleOptions.get_option_column_width() - 2)
             << left << name << lwrOption->description() << "\n";
    }

    if ((command == "") || (command == "help") || (command == "unknown"))
    {
        cout << "\nCommands:\n"
             << "  " << setw(hiddenOptions.get_option_column_width() - 2) << left
             << "help" << "Get help on a specific command\n"
             << "  " << setw(hiddenOptions.get_option_column_width() - 2) << left
             << "dump" << "Dump a topology file\n"
             << "  " << setw(hiddenOptions.get_option_column_width() - 2) << left
             << "setecids" << "Associate ecids with devices in a topology file\n"
             << "  " << setw(hiddenOptions.get_option_column_width() - 2) << left
             << "create" << "Create a topolgy file from a JSON description\n"
             << "  " << setw(hiddenOptions.get_option_column_width() - 2) << left
             << "totext" << "Colwert a binary topology file to text\n"
             << "  " << setw(hiddenOptions.get_option_column_width() - 2) << left
             << "tobinary" << "Colwert a text topology file to binary\n"
             << "  " << setw(hiddenOptions.get_option_column_width() - 2) << left
             << "tograph" << "Colwert a binary topology file to dot graph\n"
             << "  " << setw(hiddenOptions.get_option_column_width() - 2) << left
             << "maxtraffic" << "Find a set of traffic sources and sinks that will occupy all links\n\n"
             << "Try \"" << baseExeName << " help command\" for more information\n";
    }

    if ((command == "setecids") || (command == "create"))
    {
        cout << "\nFor additional information including file format for ECID files\n"
             << "see https://confluence.lwpu.com/display/GM/Protobuf+Utility\n";
    }
}

//------------------------------------------------------------------------------
int main(int argc, char** argv)
{
    string command;
    vector<string> cmdArgs;

    po::options_description desc("Options");
    desc.add_options()
        ("help,h", "Print help message and exit")
        ("verbose,v", "Enable verbose prints")
        ("relaxed,r", "When creating topology files, allow relaxed routing")
        ("overlap,o", "When creating topology files, allow address ranges to overlap")
        ("text,t", "Process topology files as text");

    po::options_description hidden("Positional Arguments");
    hidden.add_options()
        ("command",     po::value<string>(&command)->required(),         "Command to be performed")
        ("arg",         po::value<vector<string>>(&cmdArgs),             "Command arguments");

    po::positional_options_description p;
    p.add("command", 1);
    p.add("arg", -1);

    po::options_description cmdlineOptions;
    cmdlineOptions.add(desc).add(hidden);

    po::variables_map vm;

    bool bVerbose = false;
    bool bText = false;
    bool bRelaxedRouting = false;
    bool bAddressOverlap = false;

    map<string, pair<po::options_description *, po::positional_options_description *>> helpMap;
    po::options_description helpOptions("help");
    po::positional_options_description helpPositional;
    helpOptions.add_options()
        ("help",    po::value<string>()->required(), "Dummy option")
        ("command", po::value<string>()->required(), "Command to request help on");
    helpPositional.add("help", 1);
    helpPositional.add("command", 1);
    helpMap["help"] = { &helpOptions, &helpPositional };

    po::options_description dumpOptions("dump");
    po::positional_options_description dumpPositional;
    dumpOptions.add_options()
        ("dump",     po::value<string>()->required(), "Dummy option")
        ("topofile", po::value<string>()->required(), "Topology file to be dumped");
    dumpPositional.add("dump", 1);
    dumpPositional.add("topofile", 1);
    helpMap["dump"] = { &dumpOptions, &dumpPositional };

    po::options_description setEcidsOptions("setecids");
    po::positional_options_description setEcidsPositional;
    setEcidsOptions.add_options()
        ("setecids",    po::value<string>()->required(), "Dummy option")                                         //$
        ("topofilein",  po::value<string>()->required(), "Input topology file")                                  //$
        ("ecidfile",    po::value<string>()->required(), "JSON file containing ECIDS")                           //$
        ("topofileout", po::value<string>(),             "Output topology file (if unspecified input is used)"); //$
    setEcidsPositional.add("setecids", 1);
    setEcidsPositional.add("topofilein", 1);
    setEcidsPositional.add("ecidfile", 1);
    setEcidsPositional.add("topofileout", 1);
    helpMap["setecids"] = { &setEcidsOptions, &setEcidsPositional };

    po::options_description createOptions("create");
    po::positional_options_description createPositional;
    createOptions.add_options()
        ("create",      po::value<string>()->required(), "Dummy option")
        ("jsontopo",    po::value<string>()->required(), "Input JSON topology file")
        ("topofileout", po::value<string>()->required(), "Output protobuf topology file");
    createPositional.add("create", 1);
    createPositional.add("jsontopo", 1);
    createPositional.add("topofileout", 1);
    helpMap["create"] = { &createOptions, &createPositional };

    po::options_description totextOptions("totext");
    po::positional_options_description totextPositional;
    totextOptions.add_options()
        ("totext",         po::value<string>()->required(), "Dummy option")
        ("inputTopo",      po::value<string>()->required(), "Input binary topology file")
        ("outputTextTopo", po::value<string>()->required(), "Output text topology file");
    totextPositional.add("totext", 1);
    totextPositional.add("inputTopo", 1);
    totextPositional.add("outputTextTopo", 1);
    helpMap["totext"] = { &totextOptions, &totextPositional };

    po::options_description tobinaryOptions("tobinary");
    po::positional_options_description tobinaryPositional;
    tobinaryOptions.add_options()
        ("tobinary",         po::value<string>()->required(), "Dummy option")
        ("inputTopo",        po::value<string>()->required(), "Input text topology file")
        ("outputBinaryTopo", po::value<string>()->required(), "Output binary topology file");
    tobinaryPositional.add("tobinary", 1);
    tobinaryPositional.add("inputTopo", 1);
    tobinaryPositional.add("outputBinaryTopo", 1);
    helpMap["tobinary"] = { &tobinaryOptions, &tobinaryPositional };

    po::options_description toGraphOptions("tograph");
    po::positional_options_description toGraphPositional;
    toGraphOptions.add_options()
        ("tograph", po::value<string>()->required(), "Dummy option")
        ("topofilein", po::value<string>()->required(), "Input topology file")
        ("output", po::value<string>()->required(), "Output dot graph file")
        ("highlight", po::value<string>(), "Highlight data flow between two devices: "
                                           "\"GPUx->GPUy[:n]\". If n is present it's a request to "
                                           "write to data region n, otherwise it's a response to a "
                                           "read request.")
        ("rank", po::value<vector<string>>(), "A comma separated list of GPUx and SWx to align "
                                              "horizontally. Can be specified multiple times.")
        ("graphattr", po::value<string>(), "Additional graph attributes to pass to graphviz.");

    toGraphPositional.add("tograph", 1);
    toGraphPositional.add("topofilein", 1);
    toGraphPositional.add("output", 1);
    helpMap["tograph"] = { &toGraphOptions, &toGraphPositional };

    po::options_description maxTrafficOptions("maxtraffic");
    po::positional_options_description maxTrafficPositional;
    maxTrafficOptions.add_options()
        ("maxtraffic", po::value<string>()->required(), "Dummy option")
        ("topofilein", po::value<string>()->required(), "Input topology file")
        ("output",     po::value<string>(), "Output JSON file");
    maxTrafficPositional.add("maxtraffic", 1);
    maxTrafficPositional.add("topofilein", 1);
    maxTrafficPositional.add("output", 1);
    helpMap["maxtraffic"] = { &maxTrafficOptions, &maxTrafficPositional };

    helpMap["unknown"] = { &hidden, &p };

    struct Options
    {
        const po::options_description &optDesc;
        const po::positional_options_description &posOpt;
    };
    map<string, Options> allCommands =
    {
        { "help",       { helpOptions,       helpPositional } }
      , { "dump",       { dumpOptions,       dumpPositional } }
      , { "setecids",   { setEcidsOptions,   setEcidsPositional } }
      , { "create",     { createOptions,     createPositional } }
      , { "totext",     { totextOptions,     totextPositional } }
      , { "tobinary",   { tobinaryOptions,   tobinaryPositional } }
      , { "tograph",    { toGraphOptions,    toGraphPositional } }
      , { "maxtraffic", { maxTrafficOptions, maxTrafficPositional } }
    };

    try
    {
        // get the command part first
        po::store(
            po::command_line_parser(argc, argv)
              .options(cmdlineOptions)
              .positional(p)
              .allow_unregistered()
              .run(),
            vm
        );

        if (vm.count("help"))
        {
            PrintUsage(argv[0], "", desc, hidden, p);
            return 0;
        }
        po::notify(vm);
        if (vm.count("verbose"))
            bVerbose = true;
        if (vm.count("text"))
            bText = true;
        if (vm.count("relaxed"))
            bRelaxedRouting = true;
        if (vm.count("overlap"))
            bAddressOverlap = true;

        vector<string> allArgs(argv + 1, argv + argc);
        // erase everything until the command
        allArgs.erase(allArgs.begin(), boost::find(allArgs, command));

        auto cmdEntry = allCommands.find(command);
        if (cmdEntry != allCommands.end())
        {
            auto cmdArgs = get<1>(*cmdEntry);
            // parse command specific arguments
            store(
                po::command_line_parser(allArgs)
                  .options(cmdArgs.optDesc)
                  .positional(cmdArgs.posOpt)
                  .run(),
                vm);
            notify(vm);
        }
        else
        {
            std::cerr << "ERROR: Unknown command - " << command << "\n\n";
            command = "help";
        }
    }
    catch (boost::program_options::required_option & e)
    {
        string lwrrOptionName = e.get_option_name();
        e.set_option_name(lwrrOptionName.substr(lwrrOptionName.find_first_not_of("-")));
        std::cerr << "ERROR: " << e.what() << "\n\n";
        PrintUsage(argv[0], "", desc, hidden, p);
        return -1;
    }
    catch (po::error & e)
    {
        cerr << "ERROR: " << e.what() << "\n\n";
        PrintUsage(argv[0], "", desc, hidden, p);
        return -1;
    }

    int retval = 0;
    ::fabric topology;
    if (command == "help")
    {
        if (cmdArgs.size() != 1)
            retval = -1;
        else
        {
            PrintUsage(argv[0],
                       cmdArgs[0],
                       desc,
                       *helpMap[cmdArgs[0]].first,
                       *helpMap[cmdArgs[0]].second);
        }
    }
    else if (command == "dump")
    {
        if ((cmdArgs.size() != 1) || !ParseTopoFile(cmdArgs[0], bText, &topology))
            retval = -1;
        else
            retval = DumpTopology(&topology) ? 0 : -1;
    }
    else if (command == "setecids")
    {
        if ((cmdArgs.size() < 2) || (cmdArgs.size() > 3) ||
            !ParseTopoFile(cmdArgs[0], bText, &topology))
        {
            retval = -1;
        }
        else
        {
            string topoFileOut = (cmdArgs.size() != 3) ? cmdArgs[0] : cmdArgs[2];
            retval = SetEcids(&topology, cmdArgs[1], topoFileOut, bText, bVerbose) ? 0 : -1;
        }
    }
    else if (command == "create")
    {
        if (cmdArgs.size() != 2)
            retval = -1;
        else
            retval = CreateTopology(cmdArgs[0],
                                    cmdArgs[1],
                                    bRelaxedRouting,
                                    bAddressOverlap,
                                    bVerbose,
                                    bText) ? 0 : -1;
    }
    else if (command == "totext")
    {
        if ((cmdArgs.size() != 2) || !ParseTopoFile(cmdArgs[0], false, &topology))
            retval = -1;
        else
            retval = WriteTopoFile(cmdArgs[1], true, &topology, bVerbose) ? 0 : -1;
    }
    else if (command == "tobinary")
    {
        if ((cmdArgs.size() != 2) || !ParseTopoFile(cmdArgs[0], true, &topology))
            retval = -1;
        else
            retval = WriteTopoFile(cmdArgs[1], false, &topology, bVerbose) ? 0 : -1;
    }
    else if (command == "tograph")
    {
        if (!ParseTopoFile(vm["topofilein"].as<string>(), false, &topology))
            retval = -1;
        else
            retval = ToGraph(
                cmdArgs[1],
                vm["highlight"].empty() ? string() : vm["highlight"].as<string>(),
                vm["rank"].empty() ? vector<string>() : vm["rank"].as<vector<string>>(),
                vm["graphattr"].empty() ? string() : vm["graphattr"].as<string>(),
                &topology
            ) ? 0 : -1;
    }
    else if (command == "maxtraffic")
    {
        if (!ParseTopoFile(vm["topofilein"].as<string>(), false, &topology))
            retval = -1;
        else
            retval = MaxTraffic(
                vm["output"].empty() ? string() : vm["output"].as<string>(),
                &topology
            ) ? 0 : -1;
    }
    else
    {
        cerr << "Unknown command : " << command << "\n\n";
        command = "unknown";
        retval = -1;
    }

    if (retval == -1)
    {
        PrintUsage(argv[0],
                   command,
                   desc,
                   *helpMap[command].first,
                   *helpMap[command].second);
    }

    return retval;
}
