#include "lwwatch.h"
#include <cctype>
#include <cstdlib>
#include <cstdarg>
#include <cstring>
#include <cassert>
#include "manual.h"
#include "utils.h"
#include "eval.h"

extern "C" {
#include "chip.h"
}

bool increment (vector<unsigned> &coords, vector<unsigned> & dimensions);

bool outputToFile = false;
bool appendToFile = false;
FILE *outputFilePtr;

bool openFileDesc(string fileName)
{
    assert(outputToFile);
    assert(!outputFilePtr);

    // Find out if the file already exists
    FILE *temp = fopen(fileName.c_str(), "r");
    bool fileExists = (temp != NULL);
    if (fileExists)
        fclose(temp);

    if (appendToFile)
        outputFilePtr = fopen(fileName.c_str(), "a");
    else if (!fileExists)
    {
        outputFilePtr = fopen(fileName.c_str(), "w");    
    }
    else
    {
        dprintf("** Cannot overwrite file. Please use append/write to a new file.\n");
        return false;
    }

    if (!outputFilePtr)
    {
        dprintf("** File %s could not be opened\n", fileName.c_str());
        return false;
    }

    return true;
}

void closeFileDesc()
{
    outputToFile = false;
    appendToFile = false;

    if(outputFilePtr)
    {
        fclose(outputFilePtr);
        outputFilePtr = NULL;
    }
}

void dprintf_file(const char *fmt, ...)
{
    string buffer = "";
    int len;

    va_list va;
    va_start(va, fmt);

#if LWWATCHCFG_IS_PLATFORM(UNIX) || LWWATCHCFG_FEATURE_ENABLED(MODS_UNIX)
    len = vsnprintf(NULL, 0, fmt, va);
    va_end(va);
    va_start(va, fmt);
#else
    len = _vscprintf(fmt, va);
#endif

    if (!len)
    {
        buffer = "";
        return;
    }
    buffer.resize(len+1);
    vsprintf(&buffer[0], fmt, va);
    va_end(va);

    if (outputToFile)
    {
        if (outputFilePtr)
            fprintf(outputFilePtr, "%s", buffer.c_str());
        else
            assert(0);
    }
    else
        dprintf("%s", buffer.c_str());
}

#if LWWATCHCFG_IS_PLATFORM(UNIX) || LWWATCHCFG_FEATURE_ENABLED(MODS_UNIX)
void makeLower(string& arg)
{
    int size = arg.size();

    for(int i=0;i<size;i++)
    {
        arg[i] = tolower(arg[i]);
    }
}
#endif

string get_name(priv_manual::entry * e) {
    string & i = e->name;

    if (!e->parent) 
        return "???";

    string & j = e->parent->name;

    if (i.size() <= j.size()) 
        return "???";

    // make sure we miss the leading underscore
    string result = i.substr(j.size()+1, i.size() - (j.size() + 1));

    if (result.empty()) 
        return "???";
#if LWWATCHCFG_IS_PLATFORM(UNIX) || LWWATCHCFG_FEATURE_ENABLED(MODS_UNIX)
    makeLower(result);
#else
    _strlwr(&result[0]);
#endif

    return result;
}

void format(string & buffer, const char * fmt, ...)
{
    int len;

    va_list va;
    va_start(va, fmt);

#if LWWATCHCFG_IS_PLATFORM(UNIX) || LWWATCHCFG_FEATURE_ENABLED(MODS_UNIX)
    len = vsnprintf(NULL, 0, fmt, va);
    va_end(va);
    va_start(va, fmt);
#else
    len = _vscprintf(fmt, va);
#endif

    if (!len)  {
        buffer = "";
        return;
    }
    buffer.resize(len+1);
    vsprintf(&buffer[0], fmt, va);
    va_end(va);
}

static string eatSpaces(const string &inputStr)
{
    string ret_str = "";
    for (int i=0;i<(int)inputStr.length();i++)
    {
        if (!isspace(inputStr[i]))
            ret_str = ret_str + inputStr[i];
    }
    return ret_str;
}

//
// A function to colwert a string expression
// to unsigned. Returns false if the input string
// is not a valid numerical expression. 
// 
bool strtoul_eval(unsigned int &value, const string str)
{
    evaluator myEvaluator;
    string strNoSpaces = eatSpaces(str);
    return myEvaluator.eval_expr(value, strNoSpaces.c_str());
}

static string toupper_str(const string inputStr)
{
    string ret_str = "";
    for(int i=0;i<(int)inputStr.length();i++)
    {
        ret_str = ret_str + ((char)(toupper(inputStr[i])));
    }
    return ret_str;
}

// 
// Tokenizes the input string w.r.t to the separator string
// Even the separator string is included in the output. 
// 
static void parse_args_pd(const char *params, vector<string> &args, const char *sep)
{
    string paramsStr = eatSpaces(string(params));
    params = paramsStr.c_str();
    args.clear();
    const char *match = strstr(params, sep);
    while ((*params) != '\0')
    {
        if (params == match)
        {
            args.push_back(string(params, strlen(sep)));
            params += strlen(sep);
        }
        else if (match == NULL)
        {
            args.push_back(string(params));
            break;
        }
        else
        {
            args.push_back(string(params, match-params));
            params = match;
        }
        match = strstr(params, sep);        
    }
}

static int indexOfNextToken(const char *params, int i)
{
    while (!isspace(params[i]) && params[i] != '\0')
    {
        i++;
    }
    // Now, skip all spaces. 
    while (isspace(params[i]) && params[i] != '\0')
    {
        i++;
    }

    return i;
}

// 
// For every token, try to make a valid expression
// with the rest of the list. Return the string
// starting from such token as the write data for !pe
// Return the last token as the write data if no
// such numerical expression is found. 
// input sep is used only when reading the !pe input from a file. 
// 
static void  parse_args_pe(const char *params, vector<string> &args, const char *sep)
{
    args.clear();
    const char *match;

    // Try and find out if sep is in params.  If it is, we put the next token in args and assume it's a file name
    if ((match = strstr(params, sep)) != NULL)
    {
        int nextIdx = indexOfNextToken(match, 0);
        if (*(match+nextIdx) == '\0')
        {
            return;
        }
        string fname = string(match+nextIdx);
        args.push_back(fname);
        return;
    }

    vector<string> tokens;
    tokens.clear();

    // Split the input at white spaces. 
    tokenize(params, tokens);
    string lastToken = tokens.back();

    // 
    // If the number of tokens is just 2, return the first 
    // token as register name/address and second token as the 
    // data to be written
    // 
    if (tokens.size() == 2)
    {
        args.push_back(tokens[0]);
        args.push_back(tokens[1]);
        return;
    }

    const char *mergedP = params;
    int i;
    for (i=0; mergedP[i]!='\0'; i = indexOfNextToken(params, i))
    {
        unsigned value;
        if (strtoul_eval(value, string(&(mergedP[i]))))
        {
            args.push_back(eatSpaces(string(mergedP, i)));
            args.push_back(eatSpaces(string(&(mergedP[i]))));
            break;
        }
    }
    // If no valid expression found in the input string
    if (mergedP[i] == '\0')
    {
        // Take the last token as the write data
        const char *lastTokenMatch = strstr(mergedP, lastToken.c_str());
        assert(lastTokenMatch);
        args.push_back(eatSpaces(string(mergedP, lastTokenMatch-mergedP)));
        args.push_back(eatSpaces(string(lastTokenMatch)));
    }
}

void print_usage_pd()
{
    dprintf("PrivDump: Dump a privileged register/array\n");
    dprintf("Usage: !pd REGISTER_NAME[.fieldName] [ >[>] filePath]\n");
    dprintf("       !pd ARRAY_NAME[(i,j,...).fieldName] [ >[>] filePath]\n");   
}

void print_usage_pe()
{
    dprintf("PrivEmit: Write to a privileged register/array\n");
    dprintf("Usage: !pe REGISTER_NAME[.fieldName] Data\n");
    dprintf("       !pe REGISTER_NAME[(a:b)] Data\n");
    dprintf("       !pe ARRAY_NAME(i,j,...)[.fieldName] Data\n"); 
    dprintf("       !pe ARRAY_NAME(i,j,...)[(a:b)] Data\n"); 
    dprintf("       !pe < filePath\n");
    dprintf("       Data can be a number or a name of a field value\n");
}

//
// A function to check if the current field
// value is an _INIT or __PROD.
// 
bool isInitOrProd(const string fieldValueName)
{
    const char *valueName = fieldValueName.c_str();
    const char *match = strrchr(valueName, '_');
    if(match == NULL)
        //return true if it matches one of init or prod. 
        return (fieldValueName == "INIT" || fieldValueName == "_PROD");
    else
        return (!strcasecmp(match,"_INIT") || !strcasecmp(match, "_PROD"));

}

void print_field(string & line, priv_manual::field_entry * field, unsigned value, vector<unsigned> coords) {

    if (field->kind != priv_manual::kind_field) 
        return;

    LwU32 low = field->low_bit;
    LwU32 high = field->high_bit;

    vector<string> values; 
    line = "";

    // check for enumerates..
    string fld_name = get_name(field);
    unsigned initial_value = 0;
    unsigned lwrrent_match_value = 0;
    bool matchFound = false;
    string lwrrent_match_name = "";
    for (vector<priv_manual::entry *>::iterator j = field->children.begin(); j!=field->children.end(); ++j) {
        if ( (*j)->kind == priv_manual::kind_value ) {
            priv_manual::value_entry * enumerate = ( priv_manual::value_entry *)*j;

            if(enumerate->is_initial_value)
                initial_value = enumerate->value;

            if (value == enumerate->value){
                string enum_name = toupper_str(get_name(enumerate));
                matchFound = true;

                // 
                // Multiple value entries can have the same numerical value.
                // Overwrite previously matched value only if it is an init or prod
                // 
                if(isInitOrProd(lwrrent_match_name) || lwrrent_match_name == "")
                { 
                    string fieldNamePrinter;
                    if (field->is_indexed) 
                    {
                        bool first = true;
                        string coordStr;
                        format(fieldNamePrinter, "%s", (toupper_str(fld_name)).c_str());
                        for (int i = 0; i<(int)coords.size(); i++) {
                            if (!first)
                            {
                                format(coordStr, "%s,", coordStr.c_str());
                            } 
                            first = false;
                            format(coordStr,"%s%d",coordStr.c_str(), coords[i]);
                        }
                        format(fieldNamePrinter, "%s(%s) (%d:%d)", (toupper_str(fld_name)).c_str(), coordStr.c_str(), 
                               high, low);
                    }
                    else 
                    {
                        format(fieldNamePrinter, "%s (%d:%d)", (toupper_str(fld_name)).c_str(), high, low);
                    }
                    format(line, "%-33s = <%s> [0x%04x]", fieldNamePrinter.c_str(),
                            enum_name.c_str(), value);
                    lwrrent_match_value = enumerate->value;
                    lwrrent_match_name = enum_name;
                }               
            }
        }
    }

    if(lwrrent_match_value == initial_value && matchFound)
    {
        line = line + " (i)";
    }

    // If a match is not found
    if (!matchFound) 
    {
        string fieldNamePrinter = "";
        format(fieldNamePrinter, "%s (%d:%d)", (toupper_str(fld_name)).c_str(), high, low);
        format(line, "%-33s = <0x%08x>", fieldNamePrinter.c_str(), value);
    }     
}


// finds a field of a register whose low bit is equal to the passed parameter. 
priv_manual::field_entry * findField(priv_manual::entry *reg_entry, unsigned int low)
{
    for (vector<priv_manual::entry *>::iterator j = reg_entry->children.begin(); j!=reg_entry->children.end(); ++j)
    {
        if ( (*j)->kind == priv_manual::kind_field ) 
        {
            priv_manual::field_entry * fld = (priv_manual::field_entry *)*j;
            if (fld->low_bit == low)
            {
                return fld;
            }
        }
    }
    return NULL;
}

void print_register(priv_manual::entry *entry, unsigned address)
{
    // Read the value
    unsigned value = GPU_REG_RD32(address);
    dprintf_file("%-25s @(0x%08x) = 0x%08x\n", entry->name.c_str(), address, value); 
    if( (value & 0xFFFF0000) == 0xBADF0000 ) {
        dprintf(" ** WARNING: likely bogus register value 0xBADFxxxx" );
    }

    for (unsigned int low = 0; low < 32;)
    {
        priv_manual::field_entry *fld = findField(entry, low);
        if (fld == NULL)
        {
            low++;
            continue;
        }
        low = fld->high_bit + 1;

        // Handling indexed fields. 
        vector<unsigned> coords(fld->dimensions.size(),0);   // start at 0,0,0...
        do
        {
            // Callwlate the coordinate..                
            string line;
            if (!coords.empty() && 
                !fld->value.eval_range(fld->high_bit, fld->low_bit, &coords[0]))
                break;
            unsigned field_value = (value >> fld->low_bit) & ((((1 << (fld->high_bit - fld->low_bit)) - 1) << 1) | 1);
            print_field(line, fld, field_value, coords); 
            dprintf_file("                %s\n", line.c_str());
        } while (increment(coords,fld->dimensions));
    }

    // just a sanity check. 
    for (vector<priv_manual::entry *>::iterator j = entry->children.begin(); j!=entry->children.end(); ++j)
    {
        if ( (*j)->kind == priv_manual::kind_register )
        {
            dprintf("**Warning: Register has another register as a child!!\n");
        }
    }
}

static void writeToRange(unsigned address, unsigned high, unsigned low, unsigned int newValue)
{
    LwU32 regOldValue = GPU_REG_RD32(address);

    if (newValue > (unsigned int)((((1<<(high-low))-1)<<1)|1)) {
        dprintf("Error: 0x%x is out of the range of this entry!\n", newValue);
        return;
    }
    unsigned ander = 0; //a number of form 11110001111
    ander = ((((1 << ((high-low)))-1)<<1)|1)  <<low; //00001100000
    ander = ~ander; //11110011111
    unsigned old_temp = regOldValue & ander; //setting low-high bits to 0 in old value
    unsigned new_temp = (newValue<<low) & (~ander); //00000new_Value0000
    unsigned regNewValue = old_temp | new_temp;

    GPU_REG_WR32(address, regNewValue);
}

void write_register(priv_manual::entry * r, unsigned address, LwS32 high, LwS32 low, 
                    const string fieldStr, unsigned int newValue)
{

    if(fieldStr.empty())
    {
        if (high == -1 && low == -1)
        {
            GPU_REG_WR32(address, newValue);
            return;
        }
        else 
        {
            assert(high != -1);
            assert(low != -1);
            writeToRange(address, high, low, newValue);
            return;
        }
    }

    // Read the register first. 
    priv_manual::field_entry * fEntry = NULL; 

    string matchStr;
    unsigned idx = 0;
    // Check for the case of writing indexed field.
    if (fieldStr.substr(fieldStr.size() - 1) == ")") {
        string idxStr = fieldStr.substr(fieldStr.find("("), (fieldStr.find(")")));
        idxStr = idxStr.substr(1,idxStr.size()-2);
        matchStr = fieldStr.substr(0, fieldStr.find("("));
        idx = atoi(idxStr.c_str());
    } else {
        matchStr = fieldStr;
    }

    for (vector<priv_manual::entry *>::iterator j = r->children.begin(); j!=r->children.end(); ++j) 
    {        
        if ( (*j)->kind == priv_manual::kind_field ) 
        {

            // Try to match the field names. 
            string fld_name = (get_name(*j));            
            if(matchStr == fld_name || matchStr == toupper_str(fld_name))
            {
                fEntry = (priv_manual::field_entry *)(*j);
                /*XXX Should we break here since there should only be one matching? */
            } 
        }
    }

    if(fEntry == NULL)
    {
        dprintf("Field %s not found\n", fieldStr.c_str());
        return;
    }

    if (fEntry->is_indexed)
    {
        if (idx >= fEntry->size)
        {
            dprintf("Index out of range!\n");
            return;
        }
        fEntry->value.eval_range(fEntry->high_bit, fEntry->low_bit, &idx);
    }
    else
    {
        if (idx > 0)
        {
            dprintf("Field %s is not a indexed field", fieldStr.c_str());
            return;
        }
    }

    writeToRange(address, fEntry->high_bit, fEntry->low_bit, newValue);
}


// returns false if overflow
bool increment(vector<unsigned> & coords, vector<unsigned> & dimensions) {
    // callwlate the next position
    bool carry = true;
    for (vector<unsigned>::reverse_iterator ri = coords.rbegin(), rid = dimensions.rbegin(); ri!=coords.rend(); ++ri, ++rid) {
        // perform the carry
        if (carry) {
            ++*ri;
            carry = false;
        }
        // bounds check and reset this dimension
        if (*ri >= *rid) {
            *ri = 0;
            carry = true;
        }
    }
    return !carry;
}


void print_array(priv_manual::array_entry * r)
{    
    if (r->kind != priv_manual::kind_array) 
        return;        

    vector<unsigned> coords(r->dimensions.size(),0);   // start at 0,0,0...
    do 
    {
        // Callwlate the coordinate..
        unsigned where = 0;
        if (!r->value.eval(where, &coords[0]))
            break;

        dprintf_file("     [");
        bool first = true;
        for (int i = 0; i<(int)coords.size(); i++) 
        {
            if (!first) 
                dprintf_file(",");
            first = false;
            dprintf_file("%d", coords[i]);
        }        
        dprintf_file("] : ");
        
        print_register(r, where);       
    } while (increment(coords,r->dimensions));
}

// Assumes no spaces in the input string. Remove spaces before calling. 
void extractField(string& regAddrStrName, string& regAddrStrField, const string &regAddrStr)
{
    // Search for a dot and find the fieldname.
    const char *fullName = regAddrStr.c_str();
    int i = 0;
    while(fullName[i] != '\0' && fullName[i] != '.')
        i++;
    if(fullName[i] == '\0')
    {
        regAddrStrName = regAddrStr;
        regAddrStrField = "";
        return;
    }
    else
    {
        regAddrStrName = string(fullName, i);
        regAddrStrField = string(fullName+i+1);        
        return;
    }
}

// Extract the high, low bit if the bit range of register is specified.
// If no bit range is not specified, high and low are set to -1.
static void extractRange(string &regBaseName, LwS32 &high, LwS32 &low, const string regName)
{
   const char *inputStr = regName.c_str();
   const char *openingBr = inputStr;
   const char *closingBr;
   bool hasColon = false;
 
   while (1)
   {
       while (*openingBr != '(' && *openingBr != '\0')
       {
           openingBr++;
       }

       closingBr = openingBr;

       while (*closingBr != ')' && *closingBr != '\0')
       {
           if (*closingBr == ':')
           {
               hasColon = true;
           }
           closingBr++;
       }
       if (hasColon && *closingBr == ')')
       {
           break;
       }
       else if (!hasColon && *closingBr != '\0' && *(closingBr+1) != '\0')
       {
           openingBr++;
           continue;
       }
       else 
       {
           high = -1;
           low = -1;
           return;
       }
   }

   regBaseName = string(regName, 0, openingBr-inputStr) + string(regName, closingBr-inputStr+1, regName.length());

   char *sep;
   int num1 = strtoul(openingBr+1, &sep, 0);
   int num2 = strtoul(sep+1, NULL, 0);

   high = num1 < num2 ? num2 : num1;
   low = num1 < num2 ? num1 : num2;
}


// 
// Sets the indices array after extracting the indices. 
// if its not in indexed mode, indices will be an empty list. 
// Assumes that this function is called after extracting the field. 
// Assumes no spaces in the input string. Remove spaces before calling. 
// 
static void extractIndices(string &regBaseName, vector <unsigned> &indices, const string regName)
{
    indices.clear();    
    const char *inputStr = regName.c_str();
    const char *openingBr = inputStr; 
    while(*openingBr != '(' && *openingBr != '\0')
    {
        openingBr++;
    }

    const char *closingBr;
    if (*openingBr == '\0')
        return;
    else
        closingBr = openingBr+1; 
    bool isValidIdx = true;
    while(*closingBr != ')' && *closingBr != '\0')
    {
        if (!isdigit(*closingBr) && *closingBr != ',')
        {
            isValidIdx = false;
        }
        closingBr++;
    }

    if (!(*closingBr == ')' && isValidIdx))
    {
        indices.clear();
        return;
    }

    // If some extra characters are there after closingBr, 
    // then give error. 
    if ((*(closingBr+1) != '.' && *(closingBr+1) != '\0' && *(closingBr+1) != '('))
    {        
        indices.clear();
        dprintf("** error parsing the index\n");        
        return;
    }

    // Set the regBaseName as the substring
    // upto the opening bracket and after the closing bracket
    regBaseName = string(regName, 0, openingBr-inputStr) + string(regName, closingBr-inputStr+1, regName.length()); 
    string indexListStr = string(regName, (openingBr-inputStr)+1, (closingBr-openingBr)-1);    

    // Extract indices from the comma separated list. 
    const char *regNamePtr = indexListStr.c_str();
    const char *indexEnd = indexListStr.c_str();
    const char *indexStart = indexEnd; 

    while(*indexEnd != '\0')
    {
        if(*indexEnd == ',') 
        {
            string lwrIndex = string(indexListStr, (indexStart-regNamePtr), (indexEnd-indexStart));
            indices.push_back(strtoul(lwrIndex.c_str(), NULL, 0));
            indexStart = indexEnd+1;
        }
        indexEnd++;
    }

    string lwrIndex = string(indexListStr, (indexStart-regNamePtr), (indexEnd-indexStart));
    indices.push_back(strtoul(lwrIndex.c_str(), NULL, 0));
}

// 
// Checks if the indices are out of bounds
// Returns true if the indices are valid
// 
bool checkIndexBounds(const vector<unsigned> & dimensions, const vector<unsigned> & indices)
{
    if(dimensions.size() != indices.size())
        return false;
    int i;
    for(i=0;i<(int)indices.size();i++)
    {
        if(indices[i] >= dimensions[i])
            return false;
    }
    return true;
}

//
// Given an address, it finds out if the address belongs
// to one of the entries of the array
// 
bool isIdxMatch(vector<unsigned> &ret_indices, priv_manual::entry *entry, unsigned regAddress)
{
    if(entry->kind != priv_manual::kind_array)
        return false;

    priv_manual::array_entry *r = (priv_manual::array_entry *)entry;

    vector<unsigned> coords(r->dimensions.size(),0);

    do {
        // callwlate the coordinate..
        unsigned where = 0;
        if (!r->value.eval(where, &coords[0]))
            break;
        if(where == regAddress)
        {
            ret_indices = coords;
            return true;
        }

    } while (increment(coords,r->dimensions));

    return false;
}

// A function useful in printing the list of indices
string vectorToString(const vector<unsigned> &indices)
{    
    string str = "",temp = "";       

    if(indices.empty())
        return str;

    str += "(";
    bool first = true;
    for(unsigned i=0;i<indices.size();i++)
    {
        if(!first)
        {
            str += ", ";
        }
        first = false;
        format(temp, "%d", indices[i]);        
        str += temp;
    }
    str += ")";    

    return str;
}


// 
// Given an address, finds a matching register/array
// Sets the indices array accordingly
// Returns NULL if a match is not found
// 
priv_manual::entry * matchAddress(vector<unsigned> &indices, priv_manual *chipman, unsigned regAddress)
{
    map<unsigned, priv_manual::entry *>::iterator itr;

    // Finding the upper bound of the element.
    itr = (chipman->address_to_register.upper_bound(regAddress));

    while(true)
    {
        indices.clear();
        if(isIdxMatch(indices, itr->second, regAddress))
        {
            return itr->second;
        }

        if(itr == chipman->address_to_register.begin())
        {            
            return NULL;
        }        
        itr--;
    }
}

//
// Finds a matching entry in the manual. If an address is entered
// by the user, tries to match it with an array and puts the 
// corresponding indices in the indices array. 
// Returns NULL if an entry is not found.
// 
priv_manual::entry * findManualEntry(bool &isDirectMode, vector<unsigned> &indices, string regBaseName, priv_manual *chipman)
{   
    unsigned regAddress = 0;

    // If the  user entered a number
    if(strtoul_eval(regAddress, regBaseName.c_str()))
    {      
        isDirectMode = true;
        if(regAddress % 4 != 0)
        {
            dprintf("** Address not aligned\n");
            return NULL;
        }

        map<unsigned, priv_manual::entry *>::iterator itr;      
        // Use the address_to_register map
        if((itr=chipman->address_to_register.find(regAddress)) != chipman->address_to_register.end())
        {
            // Return the entry found in the hashmap. 
            return (priv_manual::entry *)(itr->second);
        }        
        else if(indices.empty())
        {
            return matchAddress(indices, chipman, regAddress); 
        }
        else
        {
            return NULL;
        }
    }
    else
    {
        isDirectMode = false; 
        //use the name->entry map
        map<string, priv_manual::entry *>::iterator itr; 
        if((itr=chipman->name_to_register.find(regBaseName)) == chipman->name_to_register.end())
        {
            dprintf("** No matching register found for %s. Try using a register address\n", regBaseName.c_str()); 
            return NULL;
        }
        else
            return (priv_manual::entry *)(itr->second);
    }  
}

// Finds if the regEntry has a field/value that matches the wild card. 
bool matchesPattern(const char *pattern, priv_manual::entry *regEntry)
{
    if (regEntry->kind != priv_manual::kind_register && regEntry->kind != priv_manual::kind_array)
    {
        return false;
    }
    if (wild_compare(pattern, regEntry->name.c_str()))
    {
        return true;
    }
    // If the register name does not match, search for fields of the register.
    for (vector<priv_manual::entry *>::iterator field = regEntry->children.begin(); field!=regEntry->children.end(); ++field)
    {
        if ((*field)->kind == priv_manual::kind_field && 
            wild_compare(pattern, (*field)->name.c_str()))
        {
            return true;                        
        }
        // Try to match the enumerants of the field
        for (vector<priv_manual::entry *>::iterator value = (*field)->children.begin(); value!=(*field)->children.end(); ++value)
        {
            // Dont match with init or prod. There are too many of them. 
            if ((*value)->kind == priv_manual::kind_value &&
                wild_compare(pattern, (*value)->name.c_str()) &&
                !isInitOrProd((*value)->name)) 
            {
                    return true;             
            }
        }
    }
    return false;   
}

void wildCardPrint(const string regBaseName, priv_manual *chipman, vector<unsigned> &indices)
{
    bool first = true;

    for (map<string, priv_manual::entry *>::iterator  i = chipman->name_to_register.begin(); i!=chipman->name_to_register.end(); ++i) 
    {
        if (matchesPattern(regBaseName.c_str(), i->second)) 
        {
            priv_manual::entry* manualEntry = i->second;

            if (manualEntry->kind == priv_manual::kind_register)
            {
                if(!first)
                    dprintf_file("\n");
                first = false;
                priv_manual::register_entry *regEntry = (priv_manual::register_entry *)i->second;
                print_register(regEntry, regEntry->value);
            }
            else if (manualEntry->kind == priv_manual::kind_array) 
            {
                if(!first)
                    dprintf_file("\n");
                first = false;
                dprintf_file("%s\n", i->first.c_str());

                if ( indices.empty() )
                {
                    print_array((priv_manual::array_entry *)manualEntry);
                }
                else
                {
                    unsigned address = 0;
                    priv_manual::array_entry *arrayEntry = (priv_manual::array_entry *)manualEntry;
                    if(!checkIndexBounds(arrayEntry->dimensions, indices))
                    {
                        dprintf("** Error Indexing the array\n");
                        return;
                    }
                    if(!arrayEntry->value.eval(address, &(indices[0])))
                    {
                        dprintf("**Error getting address from indices\n");
                        return;
                    }
                    print_register(arrayEntry, address);
                }
            }
        }
    }
}

// Exelwtes the main operation of pd/pe
static void  exec_priv_dump_emit(priv_manual::entry * manualEntry, unsigned regWriteData,
        vector<unsigned> & indices, LwS32 rangeHigh, LwS32 rangeLow, const string  fieldName, bool isWrite)
{
    if(manualEntry == NULL)
        return;

    // Get the register address based on the indices and entry. 
    if(manualEntry->kind == priv_manual::kind_register)
    {
        // If indices is not empty, give an error and exit. 
        if(!indices.empty())
        {
            dprintf("** Trying to index a non-array type. Please remove the index\n");
            return;
        }
        priv_manual::register_entry *regEntry = (priv_manual::register_entry *)manualEntry;      
        if(isWrite == 0)
            print_register(regEntry, regEntry->value);
        else
            write_register(regEntry, regEntry->value, rangeHigh, rangeLow, fieldName, regWriteData);
    } //Register type

    // If it is an array type
    else if(manualEntry->kind == priv_manual::kind_array)
    {
        priv_manual::array_entry *arrayEntry = (priv_manual::array_entry *)manualEntry;

        if(indices.empty())
        {
            if(isWrite == 0)
                print_array(arrayEntry);
            else
            {
                dprintf("** Can't write to an entire Array. Please provide valid indices!!\n");
                return;
            }
        }

        // Callwlate the regEntry from indices. 
        else
        {
            unsigned address = 0;
            if(!checkIndexBounds(arrayEntry->dimensions, indices))
            {
                dprintf("** Error Indexing the array\n");
                return;
            }
            if(!arrayEntry->value.eval(address, &(indices[0])))
            {
                dprintf("**Error getting address from indices\n");
                return;
            }

            if(isWrite == 0)
                print_register(arrayEntry, address);
            else
                write_register(arrayEntry, address, rangeHigh, rangeLow, fieldName, regWriteData);
        }
    }// Array type

    else if(manualEntry->kind == priv_manual::kind_field)
    {  
        dprintf("** No matching register found!\n"); 
    }// Field type

    else
    {
        dprintf("** No support for the entry type\n"); 
    }
}

//
// Finds the number corresponding to the fieldName
// Puts the result in regWriteData
// Returns true if a valid fieldName is found
// 
bool evalFieldName(unsigned &regWriteData, priv_manual::entry * entry, const string regDataStr, const string fieldName)
{
    if (fieldName == "")
    {
        return false;
    }
    for (vector<priv_manual::entry *>::iterator fldItr = entry->children.begin(); fldItr!=entry->children.end(); fldItr++) 
    {
        if ((*fldItr)->kind != priv_manual::kind_field)
        {
            continue;             
        }
        priv_manual::field_entry *field = (priv_manual::field_entry *)(*fldItr);

        for (vector<priv_manual::entry *>::iterator valueItr = field->children.begin(); valueItr!=field->children.end(); valueItr++)
        {
            if((*valueItr)->kind != priv_manual::kind_value)
            {
                continue;
            }            
            priv_manual::value_entry *enumerate = (priv_manual::value_entry *)(*valueItr);
            if(regDataStr == toupper_str(get_name(enumerate)))
            {
                regWriteData = enumerate->value;
                return true;
            }
        }
    }
    return false;
}

// Helper function to read a file as input for !pe. Only matches address, don't care about the name of the register
static BOOL readFileForWrite(FILE *fPtr, unsigned &address, unsigned int &newValue)
{
    char line[128];
    const char *mark;
    char *sep;
    const char *lineSpec = "(@0x"; // Specifies what line we want to look at
    while (fgets(line, 128, fPtr) != NULL)
    {
        string lineNoSpace = eatSpaces(string(line));
        // Only look at lines that specify the address and value of registers
        if ((mark = strstr(lineNoSpace.c_str(), lineSpec)) != NULL)
        {
            address = strtoul((mark+strlen(lineSpec)), &sep, 16);
            newValue = strtoul(sep+1, NULL, 16);
            return true;
        }
    }
    return false;
}

#ifndef DRF_WINDOWS
extern "C" void priv_emit(const char *params)
{
    if (params == NULL || *params == '\0')
    {
        print_usage_pe();
        return;
    }
    // Colwert everything to upper case.     
    string params_upper_str= toupper_str(string(params));
    const char *params_uppercase = params_upper_str.c_str();
    vector<string> args;
    parse_args_pe(params_uppercase, args, "<");   

    // If args has size 1, we assume !pe is taking a input file
    if (args.size() == 1)
    {
        FILE *inFile = fopen(args[0].c_str(), "r");
        if (inFile == NULL)
        {
            dprintf("%s: Invalid input file name %s.\n", __FUNCTION__, args[0].c_str());
            return;
        }
        else 
        {
            unsigned address;
            unsigned int newValue;
            unsigned count = 0;
            while (readFileForWrite(inFile, address, newValue))
            {
                GPU_REG_WR32(address, newValue);
                dprintf("(@0x%08x = 0x%08x)\n", address, newValue); 
                count++;
            }
            dprintf("%d registers are set\n", count);
        }
        fclose(inFile);
        return;
    }
    else if (args.size() != 2) 
    {
        print_usage_pe();
        return;
    }    

    priv_manual * chipman = get_active_manual();

    string regBaseName = ""; 
    vector<unsigned> indices;
    string fieldName=""; 
    priv_manual::entry *manualEntry = NULL;  
    // Direct mode: When we want to write to a register using its address
    bool isDirectMode = false; 
    bool isWrite = true;

    //the string to store the user input. 
    string regNameArg = args[0];
    string regDataStr = args[1];
    LwS32 high, low;

    extractField(regBaseName, fieldName, regNameArg); 
    extractIndices(regBaseName, indices, regBaseName); 
    extractRange(regBaseName, high, low, regBaseName);

    manualEntry = findManualEntry(isDirectMode, indices, regBaseName, chipman);
    if(manualEntry == NULL)
    { 
        if(isDirectMode && indices.empty())
        {
            unsigned regAddress = 0, writeData = 0;

            strtoul_eval(regAddress, regBaseName);
            if(strtoul_eval(writeData, regDataStr))
            {
                dprintf("Writing to the location 0x%08x\n", regAddress);
                GPU_REG_WR32(regAddress, writeData);
                return;
            }

        }
        dprintf("** Cannot find a matching register\n");
        return;
    }

    unsigned int regWriteData = 0; 

    // Trying to match with a field name
    if(evalFieldName(regWriteData, manualEntry, regDataStr, fieldName))
    {
        // FieldStr should not be empty here. 
        if(fieldName == "")
        {
            dprintf("** Error. Did you forget a field name?\n");
            return;
        }
    }
    // Trying to parse a number
    else if(!strtoul_eval(regWriteData, regDataStr))
    {
        dprintf("** Argument must be a number/name of a field value\n");
        return;
    }

    string fieldPrintStr = "";
    fieldPrintStr = fieldName.empty()? "" :"."+fieldName;

    if(isDirectMode)
    {        
        // Corrects a small issue
        // If the address matches with an array entry and index is not specified,
        // push an index of 0
        // so that we can write to the 0'th element of the array. 
        if(manualEntry->kind == priv_manual::kind_array && indices.empty())
        {
            vector<unsigned> coords(((priv_manual::array_entry *)manualEntry)->dimensions.size(),0);
            indices = coords;
        }


        dprintf("Writing to %s%s%s\n", (manualEntry->name).c_str(), vectorToString(indices).c_str(), fieldPrintStr.c_str());
    }

    // Do the operation based on the extracted values. 
    exec_priv_dump_emit(manualEntry, regWriteData, indices, high, low, fieldName, isWrite);     
}

extern "C" void priv_dump(const char *params)
{
    if (params == NULL || *params == '\0')
    {
        print_usage_pd();
        return;
    }
    // Colwert everything to upper case.     
    string params_upper_str= toupper_str(string(params));
    const char *params_uppercase = params_upper_str.c_str();
    vector<string> args;
    parse_args_pd(params_uppercase, args, ">>");
    if(args.size() == 1)
    {
        parse_args_pd(params_uppercase, args, ">");
    }

    if (args.size() != 1 && args.size() != 3) {
        print_usage_pd();
        return;
    }    
    priv_manual * chipman = get_active_manual();

    string regBaseName = ""; 
    vector<unsigned> indices;
    string fieldName=""; 
    priv_manual::entry *manualEntry = NULL;
    bool isWrite = 0;
    bool isDirectMode = false;

    string regNameArg = args[0];

    // outputToFile is a global variable.
    outputToFile = false;
    if (args.size() == 3)
    {

        string redirectionOp = args[1];
        string outputFileName = args[2];

        if(redirectionOp == ">>")
        {
            appendToFile = true;
        }
        else if (redirectionOp != ">")
        {
            print_usage_pd();
            goto exit_function;
        }             
        outputToFile = true;
        if(!openFileDesc(outputFileName))
        {
            closeFileDesc();
            return;
        }
         
        dprintf_file("%s\n", params);
    }
     

    extractField(regBaseName, fieldName, regNameArg); 
    extractIndices(regBaseName, indices, regBaseName); 

    // Treat wild card as a special case     
    if(strchr(regBaseName.c_str(), '*') != NULL)
    {
        wildCardPrint(regBaseName, chipman, indices);          
    }
    else
    {        
        manualEntry = findManualEntry(isDirectMode, indices, regBaseName, chipman);
        if(manualEntry == NULL)
        {     
            if(isDirectMode)
            {
                unsigned regAddress; 
                strtoul_eval(regAddress, regBaseName);

                unsigned data = GPU_REG_RD32(regAddress);

                // Just try to read the register.. 
                dprintf("Could not find a matching register in the manuals\n");
                dprintf_file("Value read from 0x%08x: 0x%08x \n", regAddress, data);             
            }
            goto exit_function;
        }        

        else if(isDirectMode)
        {
            // Corrects a small output formatting issue
            // Should not print the entire array in direct mode when index is not specified
            // Print just the first element of the array
            if(manualEntry->kind == priv_manual::kind_array && indices.empty())
            {
                // Index the first element by making the indices array 
                // all zeroes
                vector<unsigned> coords(((priv_manual::array_entry *)manualEntry)->dimensions.size(),0);
                indices = coords;

            }

            dprintf_file("%s%s\n", (manualEntry->name).c_str(), vectorToString(indices).c_str());
        }

        // Do the operation based on the extracted values. 
        exec_priv_dump_emit(manualEntry, 0, indices, -1, -1, fieldName, isWrite); 
    }
    exit_function:
        dprintf_file("\n\n");
        //Close the opened file. 
        closeFileDesc();
}

// priv_dump_register is a new API in unix/common/manual.c for
// different output formatting.  It's based on the DRF parser.  For
// non-unix platforms, just have it call regular priv_dump and ignore
// the new formatting requests
extern "C" void priv_dump_register( const char *params, LwU32 skip_zeroes )
{
    priv_dump( params );
}
#endif

// Helper function for printMethod_dinjectmethod, handles multimapping of registers to the same address
static priv_manual::entry *findEntry_dinjectmethod(unsigned addr, const char *name, priv_manual *chipman)
{
    map<unsigned, priv_manual::entry *>::iterator itr;      
    char a, b;
    size_t b_size;
    for (itr = chipman->address_to_register.find(addr); itr != chipman->address_to_register.end(); ++itr)
    {
        string s;
        a = name[int (strlen(name)-1)];
        b_size = itr->second->name.length();
        b = (itr->second->name.c_str())[b_size -1];
        if (((priv_manual::entry *)(itr->second))->name.find(name) != string::npos && 
            a == b)
        {
            return itr->second;
        }
    }
    return NULL; 
}

// Print method for dinjectmethod, search in the state cache.
// This method is only intended to be used by !dinjectmethod (This is a hack because the data structure for dinjectmethod is
// out-dated.
extern "C" LwU32 print_sc_dinj(unsigned baseAddr, unsigned addr, unsigned chanNum, char *name)
{
    priv_manual * chipman = get_active_manual();
    vector<unsigned> indices;
    string scName;
    string className;
    priv_manual::entry *ent;

    if (chanNum == 0)
    {
        className = "D";
    }
    else if (chanNum == 1 || chanNum == 2)
    {
        className = "C";
    }
    else if (chanNum == 3 || chanNum == 4)
    {
        className = "E";
    }

    scName = className + "_SC_";
    scName += name;

    if ((ent = findEntry_dinjectmethod(addr, scName.c_str(), chipman)) == NULL)
    {
        if ((ent = findEntry_dinjectmethod(baseAddr, scName.c_str(), chipman)) == NULL)
        {
            dprintf("%s can NOT be found in state cache\n", name);
            return 0;
        }
    }

    dprintf("name=%s, addr=0x%x\n", (ent->name).c_str(), addr);
    if (ent->kind == priv_manual::kind_array)
    {
        if (((priv_manual::array_entry *)ent)->dimensions.size() == 0)
        {
            return 0;
        }
        print_array((priv_manual::array_entry *)ent);
    } 
    else if (ent->kind == priv_manual::kind_register)
    {
        print_register(ent, addr);
    } 
    dprintf("\n\n");
    return 1;
}
