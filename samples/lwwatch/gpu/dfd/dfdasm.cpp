/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2021 by LWPU Corporation.  All rights reserved.  All information
 * contained herein is proprietary and confidential to LWPU Corporation.  Any
 * use, reproduction, or disclosure without the written permission of LWPU
 * Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

/*
 * DFD (Design for debug) Assembly Plugin
 *
 * Dfdasm is an abstraction layer between dfd tools and hardware environment
 * More info: https://p4hw-swarm.lwpu.com/files/hw/doc/gpu/hopper/hopper/design/IAS/DFD/dfd_assembly.md
 *
 */

#include <fstream>
#include <stdio.h>
#include <stdarg.h>
#include <stdint.h>
#include <list>
#include <string>
#include <vector>
#include <unordered_map>
#include <iostream>
#include <sstream>
#include <regex>
#include <algorithm>
#include "dfdasm.h"

using namespace std;

typedef enum DFDASM_METHOD_TYPE
{
    REG_METHOD,
    COND_METHOD
} DFDASM_METHOD_TYPE;

typedef enum DFDASM_METHOD_RETURN
{
    OK,
    RETURN,
    EXIT
} DFDASM_METHOD_RETURN;

// constant array to decode supported operations
static const string dfdasmValidConditionOperations[6] = {"==", "!=", "<", "<=", ">", ">="};
static const string dfdasmValidSetOperations[9] = {"+", "-", "*", "/", "<<", ">>", "&", "|", "~"};

// get next valid line
// will skip over comment and empty lines
bool DfdAsmGetNextLine(ifstream &fd, string &line, size_t &indent, int &lineNumber)
{
    while (getline(fd, line))
    {
        lineNumber += 1;
        indent = line.find_first_not_of(" ");
        // if npos returned the entire line is empty so continue
        if (indent == string::npos)
        {
            continue;
        }
        if (!line.empty() && line[indent] != '#')
        {
            return true;
        }
    }
    return false;
}

// check if is a new command
bool DfdAsmIsCommand(string line)
{
    return regex_match(line, regex("^[a-zA-Z0-9_]+:(\\s*)$"));
}

// parse name of a new command
string DfdAsmParseCommandName(string line, size_t indent)
{
    return line.substr(indent, line.find(":") - indent);
}

// split line to vector by space
vector<string> DfdAsmSplit(string line)
{
    vector<string> result;
    string word;
    istringstream ss(line);
    while (ss >> word) {
        result.push_back(word);
    }
    return result;
}

// check if string is valid variable
bool DfdAsmIsValidVariable(string token) 
{
    return regex_match(token, regex("\\$[a-zA-Z0-9_]+"));
}

// check if string is valid value
bool DfdAsmIsValidValue(string token)
{
    return regex_match(token, regex("(0x[a-fA-F0-9]{1,8})|(\\d+)"));
}

string DfdAsmLwU64ToString(LwU64 in) {
    char buffer[20];
    sprintf(buffer, "$0x%llux", in);
    return string(buffer);
}

// forward declaration of main classes
class DfdAsmMethod;
class DfdAsmCondition;
class DfdAsmCommand;
class DfdAsmRuntime;

class DfdAsmMethod
{
    public:
        virtual ~DfdAsmMethod() {}
        virtual bool init(vector<string> &splittedLine, int lineNumber) = 0;
        virtual DFDASM_METHOD_RETURN exec(DfdAsmRuntime *runtime) = 0;
        virtual DFDASM_METHOD_TYPE type() = 0;
};

class DfdAsmCondition
{
    private:
        string left;
        string op;
        string right;

    public:
        bool init(vector<string> &splittedLine, int lineNumber);
        bool evaluate(string method, DfdAsmRuntime *runtime);
};

class DfdAsmCommand
{
    public:
        vector<DfdAsmMethod *> methods;
        ~DfdAsmCommand()
        {
            for (size_t i = 0; i < methods.size(); i++)
            {
                delete methods[i];
            }
            methods.clear();
        }
        DFDASM_METHOD_RETURN exec(DfdAsmRuntime *runtime)
        {
            for (size_t i = 0; i < methods.size(); i++)
            {
                switch (methods[i]->exec(runtime))
                {
                case OK:
                    continue;
                case RETURN:
                    return RETURN;
                case EXIT:
                    return EXIT;
                }
            }
            return RETURN;
        }
};

class DfdAsmRuntime
{
    public:
        unordered_map<string, DfdAsmCommand *> commandMap;
        unordered_map<string, int64_t> variableMap;
        ofstream logFile;
        ifstream asmFile;
        bool verbose;
        bool test;
        bool verboseLog;
        DfdAsmRuntime(const char *asmFname, const char *logFname, int v, int vl, int t)
        {
            logFile.open(logFname);
            asmFile.open(asmFname);
            verbose = (v != 0);
            verboseLog = (vl != 0);
            test = (t != 0);
        }
        ~DfdAsmRuntime()
        {
            // call destructor on each command which will delete all commands and methods
            unordered_map<string, DfdAsmCommand *>::iterator it;
            for (it = commandMap.begin(); it != commandMap.end(); it++)
            {
                delete it->second;
            }
            // clear maps
            variableMap.clear();
            commandMap.clear();
            // close files if opened
            if (logFile.is_open())
            {
                logFile.close();
            }
            if (asmFile.is_open())
            {
                asmFile.close();
            }
        }
        bool init();
        bool decodeAndPushMethod(string line, vector<DfdAsmMethod *> *methods, int lineNumber);
        DFDASM_METHOD_RETURN exec(const char *command)
        {
            string cmd = string(command);
            if (commandMap.find(cmd) == commandMap.end())
            {
                dprintf("ERROR!! Command %s not found!\n", cmd.c_str());
                return EXIT;
            }
            return commandMap[cmd]->exec(this);
        }
        LwU32 TokenToLwU32(string token)
        {
            if (DfdAsmIsValidValue(token))
            {
                return (LwU32)stoll(token, nullptr, 0);
            }
            if (variableMap.find(token) == variableMap.end())
            {
                return 0;
            }
            return (LwU32)variableMap[token];
        }
        int64_t TokenToInt64(string token)
        {
            if (DfdAsmIsValidValue(token))
            {
                return (int64_t)stoll(token, nullptr, 0);
            }
            if (variableMap.find(token) == variableMap.end())
            {
                return 0;
            }
            return variableMap[token];
        }
        LwU64 TokenToLwU64(string token)
        {
            if (DfdAsmIsValidValue(token))
            {
                return (LwU64)stoll(token, nullptr, 0);
            }
            if (variableMap.find(token) == variableMap.end())
            {
                return 0;
            }
            return variableMap[token];
        }
        void saveToToken(string token, int64_t value)
        {
        
            variableMap[token] = value;
        }
        void log(const char *format, ...)
        {
            char buffer[256];
            va_list args;
            va_start(args, format);
            vsprintf(buffer, format, args);
            string msg(buffer);
            if (verbose)
            {
                dprintf("%s", msg.c_str());
            }
            logFile << msg;
            va_end(args);
        }
        void log(string msg)
        {
            if (verbose || verboseLog)
            {
                dprintf("%s", msg.c_str());
            }
            logFile << msg;
        }
};

class DfdAsmRegMethod : public DfdAsmMethod
{
    public:
        virtual ~DfdAsmRegMethod() {}
        virtual bool init(vector<string> &splittedLine, int lineNumber) = 0;
        virtual DFDASM_METHOD_RETURN exec(DfdAsmRuntime *runtime) = 0;
        DFDASM_METHOD_TYPE type()
        {
            return REG_METHOD;
        }
};

class DfdAsmConditionalMethod : public DfdAsmMethod
{
    public:
        vector<DfdAsmMethod *> methods;
        DfdAsmCondition *condition;
        virtual ~DfdAsmConditionalMethod()
        {
            delete condition;
            for (size_t i = 0; i < methods.size(); i++)
            {
                delete methods[i];
            }
            methods.clear();
        }
        virtual bool init(vector<string> &splittedLine, int lineNumber) = 0;
        virtual DFDASM_METHOD_RETURN exec(DfdAsmRuntime *runtime) = 0;
        DFDASM_METHOD_TYPE type()
        {
            return COND_METHOD;
        }
};

class DfdAsmSetMethod : public DfdAsmRegMethod
{
    private:
        string target;
        string op;
        string left;
        string right;

    public:
        bool init(vector<string> &splittedLine, int lineNumber)
        {
            // nop set
            if (splittedLine.size() == 4)
            {
                op = "NOP";
                target = splittedLine[2];
                left = splittedLine[3];
            }
            // set with unary op not
            else if (splittedLine.size() == 5)
            {
                target = splittedLine[2];
                op = splittedLine[3];
                left = splittedLine[4];
                // operation must be not
                if (op != "~")
                {
                    dprintf("ERROR!! Line %d: Invalid set operation!!  Unary set operation must be NOT!\n", lineNumber);
                    return false;
                }
            }
            // set with op
            else if (splittedLine.size() == 6)
            {
                target = splittedLine[2];
                left = splittedLine[3];
                op = splittedLine[4];
                right = splittedLine[5];
                // operation must be one of 9 valid set ops
                if (find(dfdasmValidSetOperations, dfdasmValidSetOperations + 9, op) == (dfdasmValidSetOperations + 9))
                {
                    dprintf("ERROR!! Line %d: Invalid set operation!!  Operation %s not valid!!\n", lineNumber, op.c_str());
                    return false;
                }
            }
            // wrong number of tokens
            else
            {
                dprintf("ERROR!! Line %d: Invalid set operation!! Invalid number of tokens!!\n", lineNumber);
                return false;
            }
            // more error checks
            // first operand must be a variable
            if (!DfdAsmIsValidVariable(target))
            {
                dprintf("ERROR!! Line %d: Invalid set operation!! Set target must be a variable!!\n", lineNumber);
                return false;
            }
            // left must be valid variable or value
            if (!(DfdAsmIsValidVariable(left) || DfdAsmIsValidValue(left)))
            {
                dprintf("ERROR!! Line %d: Invalid set operation!! First set operand must be a valid variable or value!!\n", lineNumber);
                return false;
            }
            if ((op != "NOP") && (op != "~"))
            {
                if (!(DfdAsmIsValidVariable(right) || DfdAsmIsValidValue(right)))
                {
                    dprintf("ERROR!! Line %d: Invalid set operation!! Second set operand must be a valid variable or value!!\n", lineNumber);
                    return false;
                }
            }
            return true;
        }
        DFDASM_METHOD_RETURN exec(DfdAsmRuntime *runtime)
        {
            int64_t result = 0;
            if (op == "NOP") {
                result = runtime->TokenToInt64(left);
            }
            else
            {
                // use index in valid_set_operation array
                switch (distance(dfdasmValidSetOperations, find(dfdasmValidSetOperations, dfdasmValidSetOperations + 9, op)))
                {
                    case 0:
                        result = runtime->TokenToInt64(left) + runtime->TokenToInt64(right);
                        break;
                    case 1:
                        result = runtime->TokenToInt64(left) - runtime->TokenToInt64(right);
                        break;
                    case 2:
                        result = runtime->TokenToInt64(left) * runtime->TokenToInt64(right);
                        break;
                    case 3:
                        result = runtime->TokenToInt64(left) / runtime->TokenToInt64(right);
                        break;
                    case 4:
                        result = runtime->TokenToInt64(left) << runtime->TokenToInt64(right);
                        break;
                    case 5:
                        result = runtime->TokenToInt64(left) >> runtime->TokenToInt64(right);
                        break;
                    case 6:
                        result = runtime->TokenToInt64(left) & runtime->TokenToInt64(right);
                        break;
                    case 7:
                        result = runtime->TokenToInt64(left) | runtime->TokenToInt64(right);
                        break;
                    case 8:
                        result = ~runtime->TokenToInt64(left);
                        break;
                    default:
                        return EXIT;
                }
            }
            runtime->saveToToken(target, result);
            runtime->log("set %s 0x%x\n", target.c_str(), result);
            return OK;
        }
};

class DfdAsmPrirdMethod : public DfdAsmRegMethod
{
    private:
        string addr;
    public:
        bool init(vector<string> &splittedLine, int lineNumber)
        {
            if (splittedLine.size() == 3) {
                addr = splittedLine[2];
            } else {
                return false;
            }
            // error check
            // addr must be valid variable or value
            if (!(DfdAsmIsValidVariable(addr) || DfdAsmIsValidValue(addr)))
            {
                dprintf("ERROR!! Line %d: Invalid prird operation!! Addr must be a valid variable or value!!\n", lineNumber);
                return false;
            }
            return true;
        }
        DFDASM_METHOD_RETURN exec(DfdAsmRuntime *runtime)
        {
            LwU32 addrHex = runtime->TokenToLwU32(addr);
            // under test mode treat all address as variable
            if (runtime->test)
            {
                runtime->saveToToken("$prird", runtime->TokenToLwU64(DfdAsmLwU64ToString(addrHex)));
            }
            else
            {
                runtime->saveToToken("$prird", (int64_t)GPU_REG_RD32(addrHex));
            }
            runtime->log("prird 0x%x 0x%x\n", addrHex, runtime->TokenToLwU64("$prird"));
            return OK;
        }
};

class DfdAsmPriwrMethod : public DfdAsmRegMethod
{
    private:
        string addr;
        string val;

    public:
        bool init(vector<string> &splittedLine, int lineNumber)
        {
            if (splittedLine.size() == 4)
            {
                addr = splittedLine[2];
                val = splittedLine[3];
            }
            else
            {
                dprintf("ERROR!! Line %d: Invalid priwr operation!! Invalid number of tokens!!\n", lineNumber);
                return false;
            }
            // error check
            // addr must be valid variable or value
            if (!(DfdAsmIsValidVariable(addr) || DfdAsmIsValidValue(addr)))
            {
                dprintf("ERROR!! Line %d: Invalid priwr operation!! Addr %s must be a valid variable or value!!\n", lineNumber, addr.c_str());
                return false;
            }
            // val must be valid variable or value
            if (!(DfdAsmIsValidVariable(val) || DfdAsmIsValidValue(val)))
            {
                dprintf("ERROR!! Line %d: Invalid priwr operation!! Val %s must be a valid variable or value!!\n", lineNumber, val.c_str());
                return false;
            }
            return true;
        }
        DFDASM_METHOD_RETURN exec(DfdAsmRuntime *runtime)
        {
            LwU64 addrHex = runtime->TokenToLwU64(addr);
            LwU32 valHex = runtime->TokenToLwU32(val);
            // under test mode treat all address as variable
            if (runtime->test)
            {
                runtime->saveToToken(DfdAsmLwU64ToString(addrHex), valHex);
            }
            else
            {
                GPU_REG_WR32(addrHex, valHex);
            }
            runtime->log("priwr 0x%x 0x%x\n", addrHex, valHex);
            return OK;
        }
};

class DfdAsmCallMethod : public DfdAsmRegMethod
{
    private:
        string cmd;
    public:
        bool init(vector<string> &splittedLine, int lineNumber)
        {
            if (splittedLine.size() == 3)
            {
                cmd = splittedLine[2];
            }
            else
            {
                dprintf("ERROR!! Line %d: Invalid call operation!! Invalid number of tokens!!\n", lineNumber);
                return false;
            }
            return true;
        }
        DFDASM_METHOD_RETURN exec(DfdAsmRuntime *runtime)
        {
            runtime->log("call " + cmd + "\n");
            if (runtime->commandMap.find(cmd) == runtime->commandMap.end())
            {
                dprintf("ERROR!! Command %s not found!\n", cmd.c_str());
                return EXIT;
            }
            if (runtime->commandMap[cmd]->exec(runtime) == EXIT)
            {
                return EXIT;
            }
            return OK;
        }
};

class DfdAsmLogMethod : public DfdAsmRegMethod
{
    private:
        string msg;
    public:
        bool init(vector<string> &splittedLine, int lineNumber)
        {
            ostringstream s;
            copy(splittedLine.begin() + 2, splittedLine.end(), ostream_iterator<string>(s, " "));
            msg = s.str();
            // check if log has single quote
            if (splittedLine[1] == "\'log")
            {
                // regex here match any string without special char or "/" or "'" and end with single quote
                // x20-x7f is normal ascii range, skipping x27 (') and x2f (/)
                if (!regex_match(msg, regex("^[\\x20-\\x26\\x28-\\x2e\\x30-\\x7f]+\'(\\s*)$")))
                {
                    dprintf("ERROR!! Line %d: Invalid log content [%s]!\n", lineNumber, msg.c_str());
                    return false;
                }
                // remove single quote at end
                msg = regex_replace(msg, regex("\'(\\s*)$"), string(""));
            }
            else
            {
                // regex here match any string without special char or "/" or "'"
                // x20-x7f is normal ascii range, skipping x27 (') and x2f (/)
                if (!regex_match(msg, regex("^[\\x20-\\x26\\x28-\\x2e\\x30-\\x7f]+$")))
                {
                    dprintf("ERROR!! Line %d: Invalid log content [%s]!\n", lineNumber, msg.c_str());
                    return false;
                }
            }
            if (msg.length() > 65535)
            {
                dprintf("ERROR!! Line %d: Oversized log line! Length %d > 65535!\n", lineNumber, (int)msg.length());
                return false;
            }
            return true;
        }
        DFDASM_METHOD_RETURN exec(DfdAsmRuntime *runtime)
        {
            runtime->log("log " + msg + "\n");
            return OK;
        }
};

class DfdAsmReturnMethod : public DfdAsmRegMethod
{
    public:
        bool init(vector<string> &splittedLine, int lineNumber)
        {
            if (splittedLine.size() != 2)
            {
                dprintf("ERROR!! Line %d: Invalid return operation!! Invalid number of tokens!!\n", lineNumber);
                return false;
            }
            return true;
        }
        DFDASM_METHOD_RETURN exec(DfdAsmRuntime *runtime)
        {
            runtime->log("return\n");
            return RETURN;
        }
};

class DfdAsmExitMethod : public DfdAsmRegMethod
{
    private:
        string retval;
    public:
        bool init(vector<string> &splittedLine, int lineNumber)
        {
            if (splittedLine.size() != 3)
            {
                dprintf("ERROR!! Line %d: Invalid exit operation!! Invalid number of token!!\n", lineNumber);
                return false;
            }
            else
            {
                retval = splittedLine[2];
                if (!(DfdAsmIsValidVariable(retval) || DfdAsmIsValidValue(retval)))
                {
                    dprintf("ERROR!! Line %d: Invalid exit operation!! Return code must be a valid variable or value!!\n", lineNumber);
                    return false;
                }
            }
            return true;
        }
        DFDASM_METHOD_RETURN exec(DfdAsmRuntime *runtime)
        {
            dprintf("WARNING!! EXIT with status code 0x%x\n", runtime->TokenToLwU32(retval));
            runtime->log("exit 0x%x\n", runtime->TokenToLwU32(retval));
            // ignore return code for lwwatch
            return EXIT;
        }
};

class DfdAsmIfMethod : public DfdAsmConditionalMethod
{
    public:
        bool init(vector<string> &splittedLine, int lineNumber)
        {
            condition = new DfdAsmCondition();
            return condition->init(splittedLine, lineNumber);
        }
        DFDASM_METHOD_RETURN exec(DfdAsmRuntime *runtime)
        {
            if (condition->evaluate("if", runtime))
            {
                for (size_t i = 0; i < methods.size(); i++)
                {
                    switch(methods[i]->exec(runtime))
                    {
                        case OK:
                            continue;
                        case RETURN:
                            return RETURN;
                        case EXIT:
                            return EXIT;
                    }
                }
            }
            return OK;
        }
};

class DfdAsmWhileMethod : public DfdAsmConditionalMethod
{
    public:
        bool init(vector<string> &splittedLine, int lineNumber)
        {
            condition = new DfdAsmCondition();
            return condition->init(splittedLine, lineNumber);
        }
        DFDASM_METHOD_RETURN exec(DfdAsmRuntime *runtime)
        {
            while (condition->evaluate("while", runtime))
            {
                for (size_t i = 0; i < methods.size(); i++)
                {
                    switch (methods[i]->exec(runtime))
                    {
                        case OK:
                            continue;
                        case RETURN:
                            return RETURN;
                        case EXIT:
                            return EXIT;
                    }
                }
            }
            return OK;
        }
};

bool DfdAsmCondition::init(vector<string> &splittedLine, int lineNumber)
{
    if (splittedLine.size() != 5)
    {
        dprintf("ERROR!! Line %d: Invalid condition!! Invalid number of tokens!!\n", lineNumber);
        return false;
    }
    left = splittedLine[2];
    op = splittedLine[3];
    right = splittedLine[4];
    if (right[right.length() - 1] != ':')
    {
        dprintf("ERROR!! Line %d: Invalid condition!!\n", lineNumber);
        return false;
    }
    right = right.substr(0, right.length() - 1);
    // error checks
    // first operand must be a variable
    if (!DfdAsmIsValidVariable(left))
    {
        dprintf("ERROR!! Line %d: Invalid condition!! Invalid left operand!!\n", lineNumber);
        return false;
    }
    // operation must be one of 6 valid cond ops
    if (find(dfdasmValidConditionOperations, dfdasmValidConditionOperations + 6, op) == (dfdasmValidConditionOperations + 6))
    {
        dprintf("ERROR!! Line %d: Invalid condition!! Invalid operation!!\n", lineNumber);
        return false;
    }
    // right must be valid variable or value
    if (!(DfdAsmIsValidVariable(right) || DfdAsmIsValidValue(right)))
    {
        dprintf("ERROR!! Line %d: Invalid condition!! Invalid right operand!!\n", lineNumber);
        return false;
    }
    return true;
}

bool DfdAsmCondition::evaluate(string method, DfdAsmRuntime *runtime)
{
    int64_t leftVal = runtime->TokenToInt64(left);
    int64_t rightVal = runtime->TokenToInt64(right);
    runtime->log("%s 0x%x %s 0x%x\n", method.c_str(), leftVal, op.c_str(), rightVal);
    switch (distance(dfdasmValidConditionOperations, find(dfdasmValidConditionOperations, dfdasmValidConditionOperations + 6, op)))
    {
        case 0:
            return leftVal == rightVal;
        case 1:
            return leftVal != rightVal;
        case 2:
            return leftVal < rightVal;
        case 3:
            return leftVal <= rightVal;
        case 4:
            return leftVal > rightVal;
        case 5:
            return leftVal >= rightVal;
        }
    return false;
}

// decode the line to a method and push the method to input method vector
bool DfdAsmRuntime::decodeAndPushMethod(string line, vector<DfdAsmMethod *> *methods, int lineNumber)
{
    vector<string> splittedLine = DfdAsmSplit(line);

    DfdAsmMethod *newMethod;
    // sanity check format
    if (splittedLine[0] != "-") {
        return false;
    }
    // decode the method
    if (splittedLine[1] == "set")
    {
        newMethod = new DfdAsmSetMethod();
    }
    else if (splittedLine[1] == "prird")
    {
        newMethod = new DfdAsmPrirdMethod();
    }
    else if (splittedLine[1] == "priwr")
    {
        newMethod = new DfdAsmPriwrMethod();
    }
    else if (splittedLine[1] == "call")
    {
        newMethod = new DfdAsmCallMethod();
    }
    else if (splittedLine[1] == "log")
    {
        newMethod = new DfdAsmLogMethod();
    }
    else if (splittedLine[1] == "\'log")
    {
        newMethod = new DfdAsmLogMethod();
    }
    else if (splittedLine[1] == "return")
    {
        newMethod = new DfdAsmReturnMethod();
    }
    else if (splittedLine[1] == "exit")
    {
        newMethod = new DfdAsmExitMethod();
    }
    else if (splittedLine[1] == "if")
    {
        newMethod = new DfdAsmIfMethod();
    }
    else if (splittedLine[1] == "while")
    {
        newMethod = new DfdAsmWhileMethod();
    }
    else
    {
        dprintf("ERROR!! Line %d: Unknown method!!\n", lineNumber);
        return false;
    }
    if (!newMethod->init(splittedLine, lineNumber))
    {
        delete newMethod;
        return false;
    }
    methods->push_back(newMethod);
    return true;
}

bool DfdAsmRuntime::init()
{
    // used to track command/if/while since these classes can have methods
    vector<vector<DfdAsmMethod *> *> compileStack;
    vector<size_t> indentStack;
    string line;
    size_t expectedIndent = 0;
    size_t lwrIndent = 0;
    int lineNumber = 0;
    bool expectedIndentChange = false;
    DFDASM_METHOD_TYPE type;
    DfdAsmMethod *lwrMethod;
    if (!asmFile.is_open())
    {
        dprintf("ERROR!! Cannot open dfd asm file!!\n");
        return NULL;
    }
    if (!logFile.is_open())
    {
        dprintf("ERROR!! Cannot open log file!!\n");
        return false;
    }
    while (DfdAsmGetNextLine(asmFile, line, lwrIndent, lineNumber))
    {
        // 1) check indent to see if last command/if/while has ended
        // last command/if/while has ended
        if (lwrIndent < expectedIndent) {
            // indent will always change here
            expectedIndentChange = true;
            // check if stack is empty, technically shouldn't be possible but if so then there must be 
            // an error somewhere
            if (indentStack.empty())
            {
                dprintf("ERROR!! Line %d: Indent stack empty\n", lineNumber);
                return false;
            }
            while (!indentStack.empty())
            {
                // pop decode stack till indent level matches, then pop the matched level
                if (indentStack.back() == lwrIndent)
                {
                    indentStack.pop_back();
                    compileStack.pop_back();
                    break;
                }
                // if current indent level is larger than last stack's indent level then there is a indent error
                else if (lwrIndent > indentStack.back())
                {
                    dprintf("ERROR!! Line %d: Indent error!!\n", lineNumber);
                    return false;
                }
                // keep popping if current indent is smaller
                else
                {
                    indentStack.pop_back();
                    compileStack.pop_back();
                }
            }
        }
        // handle any expected change in indent
        if (expectedIndentChange)
        {
            expectedIndent = lwrIndent;
            expectedIndentChange = false;
        }
        // check for any unexpected indent increase
        if (lwrIndent > expectedIndent)
        {
            dprintf("ERROR!! Line %d: Unexpected indent increase! Current indent %d, expected indent %d\n", lineNumber, (int)lwrIndent, (int)expectedIndent);
            return false;
        }
        // 2) parse and handle the new line
        // check if it is a new command
        if (DfdAsmIsCommand(line))
        {
            // sanity check compile stack should be empty
            if (!compileStack.empty())
            {
                dprintf("ERROR!! Line %d: Compile stack not empty when parsing command! Current size %d\n", lineNumber, (int)compileStack.size());
                return false;
            }
            // parse func name and create new command
            DfdAsmCommand *newCmd = new DfdAsmCommand();
            compileStack.push_back(&newCmd->methods);
            indentStack.push_back(lwrIndent);
            commandMap[DfdAsmParseCommandName(line, lwrIndent)] = newCmd;
            // expect indent level to increase
            expectedIndent = lwrIndent + 1;
            expectedIndentChange = true;
        }
        // this is a method, at this point the back of compile_stack should be what we are working on
        // so decode and push to the back of compile stack
        else
        {
            // decode and push the new method to back of compile stack
            if (!decodeAndPushMethod(line, compileStack.back(), lineNumber))
            {
                dprintf("ERROR!! Line %d: Bad decode\n", lineNumber);
                return false;
            }
            lwrMethod = compileStack.back()->back();
            // check if we encountered a new if or while method
            type = lwrMethod->type();
            if (type == COND_METHOD)
            {
                // fine to cast it here since type == COND_METHOD implies conditional method
                compileStack.push_back(&((DfdAsmConditionalMethod *)lwrMethod)->methods);
                indentStack.push_back(lwrIndent);
                // expect indent level to increase
                expectedIndent = lwrIndent + 1;
                expectedIndentChange = true;
            }
        }
    }
    return true;
}

void runDfdAsm(const char *asmFname, const char *logFname, const char *command, int verbose, int verboseLog, int test)
{
    DfdAsmRuntime *runtime = new DfdAsmRuntime(asmFname, logFname, verbose, verboseLog, test);
    if (!runtime->init())
    {
        dprintf("ERROR Initializing runtime!!\n");
        delete runtime;
        return;
    }
    dprintf("Runtime initialized. Exelwting command %s\n", command);
    switch (runtime->exec(command))
    {
        case OK:
            dprintf("Status - OK\n");
            break;
        case RETURN:
            dprintf("Status - OK\n");
            break;
        case EXIT:
            dprintf("Status - EXIT\n");
            break;
    }
    delete runtime;
}