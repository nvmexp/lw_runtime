/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2019 by LWPU Corporation.  All rights reserved.  All information
 * contained herein is proprietary and confidential to LWPU Corporation.  Any
 * use, reproduction, or disclosure without the written permission of LWPU
 * Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */



#include "lwwatch.h"
#if LWWATCHCFG_IS_PLATFORM(UNIX) || LWWATCHCFG_FEATURE_ENABLED(MODS_UNIX)
#include <cstddef>
#include <sys/types.h>
#include <dirent.h>
#else
#include <windows.h>
#endif

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <vector>
#include <string>

using namespace std;

#include "utils.h"
#include "eval.h"
#include "manual.h"

#include "dpparsedpcd.h"

extern "C" {
#include "chip.h"
}

#define LWDP_MASK(l,size) ((0xFFFFFFFF>>(32-size))<<l)

inline bool is_prefix(const string & prefix, const string & test)
{
    if (test.size() <=  prefix.size())
        return false;

    return strncmp(prefix.c_str(), test.c_str(), prefix.size()) == 0;
}

class parser 
{
    char * ibuf, *buffer;

    void skipwhite()
    {
        while (*ibuf == ' ' || *ibuf == '\t') 
            ++ibuf;
    }

    void skipline()
    {
        while (*ibuf && *ibuf!='\n' && *ibuf!='\r')
            ++ibuf;

        while (*ibuf == '\n' || *ibuf=='\r')
            ++ibuf;    
    }

    void match_end_of_line(string & tok) 
    {
        skipwhite();
        tok = "";
        while (*ibuf && *ibuf!='\r' && *ibuf!='\n')
            tok+= *ibuf++;
    }

    void match_ident(string & tok) 
    {
        skipwhite();
        tok = "";
        while ((*ibuf >= 'a' && *ibuf <='z') || (*ibuf >= 'A' && *ibuf <='Z') || *ibuf=='_' || (*ibuf >= '0' && *ibuf <='9'))
            tok+= *ibuf++;
    }

    void match_ident_raw(string & tok) 
    {
        skipwhite();
        tok = "";
        while ((*ibuf >= 'a' && *ibuf <='z') || (*ibuf >= 'A' && *ibuf <='Z') || *ibuf=='_' || (*ibuf >= '0' && *ibuf <='9'))
            tok+= *ibuf++;
    }

    bool match(const char * str) 
    {
        skipwhite();
        char * i = ibuf;
        while (*str && *str==*i)
            str++, i++;

        if (!*str) {
            ibuf = i;
            return true;
        }
        return false;
    }

    public:

    parser() 
    {
        ibuf = "";
        buffer = 0;
    }

    ~parser() 
    {
        delete[] buffer;
    }

    bool load(const string & path) 
    {
        FILE * f = fopen(path.c_str(), "rb");
        if (!f)
            return false;

        delete[] buffer;
        fseek(f, 0, SEEK_END);
        long size = ftell(f);
        fseek(f, 0, SEEK_SET);
        buffer = ibuf = new char[size+1];
        fread(ibuf, 1, size, f);
        ibuf[size] = 0;

        fclose(f);
        return true;
    }

    void parse_define(map<string, evaluator> & raw_entries)
    {
        string field, value;
        match_ident_raw(field);
        map<string, unsigned> params;
        if (*ibuf == '(') { // macro parameters
            ++ibuf;

            unsigned args = 0;

            while (*ibuf && *ibuf != ')') {
                string last_arg;
                match_ident_raw(last_arg);

                params[last_arg] = args++;

                skipwhite();
                if (*ibuf == ',')
                    ibuf++;
                skipwhite();
            }


            if (*ibuf != ')')       // skip
                return;
            ++ibuf;
        }
        match_end_of_line(value);

        evaluator & e = raw_entries[field];
        e.parameters.swap(params);
        e.expr = value;    
    }

    void go(map<string, evaluator> & raw_entries) 
    {
        while (*ibuf) {
            skipwhite();
            if (*ibuf=='#') {
                // eat the #
                ++ibuf;
                // make sure it was a #define
                string define;
                match_ident(define);
                if (define == "define")
                    parse_define(raw_entries);

                skipline();
            } 
            else 
                skipline();
        }
    }

};

struct post_processor 
{
    map<string, evaluator> raw_entries;

    struct prefix_entry 
    {
        string name;
        prefix_entry * parent;
        vector<priv_manual::entry *> entry;
        char type; //type is either 'R', 'A', 'F', 'V'
        map<string, prefix_entry *> children;

        prefix_entry * find_prefix(const prefix_entry *testEntry) 
        {
            string regname(testEntry->name);

            for(map<string, prefix_entry *>::reverse_iterator i(children.upper_bound(regname));
                    i!=children.rend(); ++i)
            {
                //Register can't be a child of register or an array entry. 
                if(testEntry->type == 'R' && (i->second->type == 'R' ||
                                             i->second->type == 'A'))
                {
                    continue;
                }

                // __SIZE_1 entries do not have a type. 
                // Treat them as a special case
                if(i->second->type != testEntry->type || 
                        (testEntry->type == 0 && i->second->type == 0))
                {
                    if (is_prefix(i->first, regname))
                        return i->second;
                }
            }
            return 0;
        }

        void insert(const string  &regname, char type)
        {
            children.insert(make_pair(regname, new prefix_entry(this, regname, type)));
        }

        prefix_entry * find_field(const string & str) 
        {
            string s = name + "_" + str;
            map<string, prefix_entry *>::iterator i = children.find(s);
            if (i!=children.end())
                return i->second;
            else
                return 0;
        }

        prefix_entry(prefix_entry * parent, const string & name, char type) : name(name), parent(parent), entry(0),type(type) {}
        prefix_entry() {}
    };

    priv_manual * target;
    prefix_entry root;

    post_processor(priv_manual * target) : target(target), root(&root, "", 0) {} 

    bool add(const string & where)
    {
        // process the manual
        parser stage;
        if(stage.load(where))
        {
            stage.go(raw_entries);
        }
        else
        {
            printf("Loading failed\n");
            return false;
        }

        return true;
    }

    bool get_specs(prefix_entry * e, char * specs) 
    {
        string & name = e->name;
        string & value = raw_entries[name].expr;

        if (const char * where = strstr(value.c_str(), "/*")) 
        {
            where+=2;
            while (*where == ' ' || *where=='\t')
                ++where;

            int i = 0;
            while (i<5 && *where != '*'  && *where)
                specs[i++] = *where++;
            return true;
        }    
        return false;
    }

    void compile_priv_manual(prefix_entry *e) 
    {
        char specs[5];
        if (get_specs(e, specs)) 
        {
            if (specs[4] == 'S')
            {
                return;
            }   
            else if (specs[4] == 'A') 
            {
                unsigned int size;
                char size_buffer[24]; 
                const char *tempStr;
                char newString[50];
                sprintf(size_buffer, "_SIZE");
                prefix_entry * szfield = e->find_field(string(size_buffer));

                if (szfield)
                {
                    raw_entries[szfield->name].eval(size, 0);
                }
                else
                {
                    dprintf("Entry not found!\n");
                }

                for(unsigned k = 0; k < size; k++)
                {
                    priv_manual::register_entry * n  = new priv_manual::register_entry();
                    e->entry.push_back(n);
                    tempStr = e->name.c_str();
                    sprintf(newString, "%s(%d)", tempStr, k);
                    n->name = string(newString);
                    n->value = 0;
                    raw_entries[e->name].eval(n->value, &k);
                    target->insert(e->parent->entry[0], e->entry[k]);
                }

            }
            else if (specs[4] == 'R') 
            {
                priv_manual::register_entry * n  = new priv_manual::register_entry();
                e->entry.push_back(n);
                n->name = e->name;
                n->value = 0;
                raw_entries[n->name].eval(n->value, 0);              
            }
            else if (specs[4] == 'V') 
            {
                priv_manual::value_entry * n = new priv_manual::value_entry();
                e->entry.push_back(n);
                n->is_initial_value = specs[2] == 'I';
                n->name = e->name;
                raw_entries[n->name].eval(n->value, 0);    
            }
            else if (specs[4] == 'F') 
            {
                priv_manual::field_entry * n = new priv_manual::field_entry();
                e->entry.push_back(n);
                n->name = e->name;
                n->value = raw_entries[n->name];
                n->is_indexed = false;
                n->size = 0;
                n->low_bit = n->high_bit = 0;
                raw_entries[n->name].eval_range(n->high_bit, n->low_bit, 0);                           

            }
            else
            {
                e->entry[0] = e->parent->entry[0];
            }
        }
        else
        {
            e->entry[0] = e->parent->entry[0];
        }//if get specs

        // hook up the tree
        if (specs[4] != 'A' && e->entry != e->parent->entry)
        {
            target->insert(e->parent->entry[0], e->entry[0]);
        }

        for (map<string, prefix_entry *>::iterator i = e->children.begin(); i != e->children.end(); ++i)
        {
            compile_priv_manual(i->second);
        }
    }

    char getType(string entryName)
    {
        prefix_entry pe(NULL, entryName, 0);
        char specs[5];
        get_specs(&pe, &(specs[0]));

        if(specs[4] == 'R' || specs[4] == 'V' || specs[4] == 'F' || specs[4] == 'A')
            return specs[4];
        else return 0;
    }

    void compile_priv_manual() 
    {
        // now analyze raw entries..
        for (map<string, evaluator>::iterator i = raw_entries.begin(); i!=raw_entries.end();++i) 
        {
            prefix_entry * parent = &root;
            prefix_entry lwrrentEntry(NULL, i->first, getType(i->first));

            while (prefix_entry * n = parent->find_prefix(&lwrrentEntry))
            {
                parent = n;
            }
            parent->insert(i->first, getType(i->first));
        };
        root.parent = &root;
        priv_manual::device_entry* n = new priv_manual::device_entry();
        root.entry.push_back(n);
        target->root = root.entry[0];
        compile_priv_manual(&root);
    }
};

class manual_cache 
{
    public:
        priv_manual * load(string path1, string path2)
        {
            priv_manual * db = new priv_manual();
            post_processor p(db);
            if (!p.add(path1))
                return NULL;
            if (path2.length() > 0)
                if (!p.add(path2))
                    return NULL;
            p.compile_priv_manual();
            return db;
        }

        manual_cache() {}
        ~manual_cache() {}
};


/*!
 * @brief dpDecodeAux - Function to print out strings representing the meaning
 *                      of a DP register and its data
 * 
 *  @param[in]  LwU32       addr        DP address (register)
 *  @param[in]  LwU8        *data       Data at that address
 *  @param[in]  LwU32       length      Length of the data
 *  @param[in]  LwU8        version     DP version
 *  @return     LwBool                  Success/failure       
 */
extern "C" LwBool dpDecodeAux(LwU32 addr, LwU8 *data, LwU32 length, LwU8 version)
{
    manual_cache                 *cache;
    priv_manual                  *dpManual;
    priv_manual::entry           *regEntry;
    //char                          dpcdFullPath[MAX_PATH];
    //char                          dpcdFullPath14[MAX_PATH];
    string                        dpcdFullPath;
    string                        dpcdFullPath14;
    vector<priv_manual::entry*>   regChildren;
    vector<priv_manual::entry*>::iterator it, it1;
    vector<priv_manual::entry*>   fieldChildren;
    LwU64                         testValue;
    LwU8                          offset;

    string dpcdPath("\\sw\\dev\\gpu_drv\\chips_a\\drivers\\common\\inc\\displayport\\dpcd.h");
    string dpcdPath14("\\sw\\dev\\gpu_drv\\chips_a\\drivers\\common\\inc\\displayport\\dpcd14.h");

    const char *testElw = getelw("P4ROOT");
    if (testElw == NULL)
    {
        dprintf("P4ROOT elw variable not set. Aborting.\n");
        return LW_FALSE;
    }
    string p4RootElw(testElw);
    dpcdFullPath = p4RootElw;
    dpcdFullPath14 = p4RootElw;
    dpcdFullPath += dpcdPath;
    dpcdFullPath14 += dpcdPath14;

    cache = new manual_cache(); 
    dpManual = cache->load(dpcdFullPath.c_str(), version == 0x14 ? dpcdFullPath14.c_str() : "");
    if (dpManual == NULL)
    {
        testElw = getelw("LWW_DP_MANUAL");
        if (testElw == NULL)
        {
            dprintf("Manual could not be loaded.\n"
                "Set environment variable LWW_DP_MANUAL to "
                "<your path>\\chips_a\\drivers\\common\\inc\\displayport\\\n");
            return LW_FALSE;
        }
        string manualElw(testElw);
        dpcdFullPath = manualElw;
        dpcdFullPath14 = manualElw;
        dpcdFullPath += "\\dpcd.h";
        dpcdFullPath14 += "\\dpcd14.h";

        dpManual = cache->load(dpcdFullPath.c_str(), version == 0x14 ? dpcdFullPath14.c_str() : "");
        if (dpManual == NULL)
        {
            dprintf("Manual could not be loaded.\n"
                "Make sure that %s and \n %s exists.\n", dpcdFullPath.c_str(), dpcdFullPath14.c_str());
            return LW_FALSE;
        }
    }

    for (offset = 0; offset < length; offset++)
    {
        testValue = data[offset];
        if ((regEntry = dpManual->address_to_register[addr+offset]))
        {
            dprintf("%s (0x%x)\n", (regEntry->name).c_str(), addr+offset);
            regChildren = regEntry->children;
            if(regChildren.size() > 0)
            {
                for(it = regChildren.begin(); it != regChildren.end(); it++)
                {
                    unsigned int low_bit = ((priv_manual::field_entry*)(*it))->low_bit;
                    unsigned int high_bit = ((priv_manual::field_entry*)(*it))->high_bit;
                    unsigned int size = high_bit-low_bit + 1;
                    unsigned int mask = LWDP_MASK(low_bit, size);
                    unsigned int result = testValue & mask;
                    result = result >> low_bit;
                    fieldChildren = (*it)->children;

                    if(fieldChildren.size() > 0)
                    {
                        for(it1 = fieldChildren.begin(); it1 != fieldChildren.end(); it1++)
                        {
                            priv_manual::value_entry *valueEntry = (priv_manual::value_entry*)(*it1);
                            if(valueEntry->value == result)
                            {
                                dprintf("\t\t%-45s [0x%04x]\n", (valueEntry->name).c_str(), result);
                                break;
                            }
                        }
                        // if corresponding value was not found in the manual
                        if (it1 == fieldChildren.end())
                        {
                            dprintf("\t\t%-45s = 0x%04x\n", ((*it)->name).c_str(), result);
                        }
                    }
                    else
                    {
                        dprintf("\t\t%-45s = 0x%04x\n", ((*it)->name).c_str(), result);
                    }
                }
            }
            else
            {
                dprintf("\t\t[0x%04llx]\n", testValue);
            }
        }
    }    

    delete(dpManual);
    delete(cache);
    return LW_TRUE;
}
