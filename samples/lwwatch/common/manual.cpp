/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2019 by LWPU Corporation.  All rights reserved.  All information
 * contained herein is proprietary and confidential to LWPU Corporation.  Any
 * use, reproduction, or disclosure without the written permission of LWPU
 * Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

// manparser.cpp : Defines the entry point for the console application.

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

extern "C" {
#include "chip.h"
}

    inline bool is_prefix(const string & prefix, const string & test) {
        if (test.size() <=  prefix.size())
            return false;

        return strncmp(prefix.c_str(), test.c_str(), prefix.size()) == 0;
    }

class parser {
    char * ibuf, *buffer;

    void skipwhite() {
        while (*ibuf == ' ' || *ibuf == '\t') 
            ++ibuf;
    }


    void skipline() {
        while (*ibuf && *ibuf!='\n' && *ibuf!='\r')
            ++ibuf;

        while (*ibuf == '\n' || *ibuf=='\r')
            ++ibuf;    
    }

    void match_end_of_line(string & tok) {
        skipwhite();
        tok = "";
        while (*ibuf && *ibuf!='\r' && *ibuf!='\n')
            tok+= *ibuf++;
    }

    void match_ident(string & tok) {
        skipwhite();
        tok = "";
        while ((*ibuf >= 'a' && *ibuf <='z') || (*ibuf >= 'A' && *ibuf <='Z') || *ibuf=='_' || (*ibuf >= '0' && *ibuf <='9'))
            tok+= *ibuf++;
    }

    void match_ident_raw(string & tok) {
        skipwhite();
        tok = "";
        while ((*ibuf >= 'a' && *ibuf <='z') || (*ibuf >= 'A' && *ibuf <='Z') || *ibuf=='_' || (*ibuf >= '0' && *ibuf <='9'))
            tok+= *ibuf++;
    }

    bool match(const char * str) {
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

    parser() {
        ibuf = "";
        buffer = 0;
    }

    ~parser() {
        delete[] buffer;
    }

    bool load(const string & path) {
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

    void parse_define(map<string, evaluator> & raw_entries) {
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

    void go(map<string, evaluator> & raw_entries) {
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

struct post_processor {
    map<string, evaluator> raw_entries;

    struct prefix_entry {
        string name;

        prefix_entry * parent;
        priv_manual::entry * entry;

        char type; //type is either 'R', 'A', 'F', 'V'

        map<string, prefix_entry *> children;

        prefix_entry * find_prefix(const prefix_entry *testEntry) {

            string regname(testEntry->name);

            for(map<string, prefix_entry *>::reverse_iterator i(children.upper_bound(regname));
                    i!=children.rend(); ++i){

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

            void insert(const string  &regname, char type) {
                children.insert(make_pair(regname, new prefix_entry(this, regname, type)));
            }

            prefix_entry * find_field(const string & str) {
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

        bool add(const string & where) {
            // process the manual
            parser stage;
            // stop if the file fails to open
            if (!stage.load(where)) {
                return false;
            }
            stage.go(raw_entries);
            return true;
        }

        bool get_specs(prefix_entry * e, char * specs) {
            string & name = e->name;
            string & value = raw_entries[name].expr;

            if (const char * where = strstr(value.c_str(), "/*")) {
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

        void compile_priv_manual(prefix_entry *e) {
            char specs[5];
            if (get_specs(e, specs)) {
                // Second condition is a hack because of the bug in the manual ('A' is now 'R')
                // Consider everything with parameters(i,j..) as an array except in case of indexed-fields
                if (specs[4] == 'A' || (raw_entries[e->name].parameters.size() > 0 && specs[4] != 'F')) {
                    priv_manual::array_entry * n  = new priv_manual::array_entry();
                    e->entry = n;
                    n->name = e->name;
                    n->value = raw_entries[n->name];

                    size_t dims = n->value.parameters.size();

                    vector<unsigned> & dimensions(n->dimensions);
                    for (unsigned int i =0; i!=dims; i++) {
                        // lets analyze the array dimensions.  
                        char size_buffer[24];
                        sprintf(size_buffer, "_SIZE_%d", i+1);
                        prefix_entry * szfield = e->find_field(size_buffer);

                        dimensions.push_back(0);

                        if (szfield)
                        {
                            // 
                            // Evaluate the value of the __SIZE_* field and 
                            // store that value in the last element of the dimensions array
                            // 
                            raw_entries[szfield->name].eval(dimensions.back(), 0);
                        }
                        else
                        {
                            // 
                            // The following state cache registers do not have __SIZE_ entries
                            // in the *_sc_addendum.h file. 
                            // a) Registers marked as private
                            // b) Registers that should not be used by SW
                            //

                            // 
                            // Also, there are a few special cases that our parser cannot handle. 
                            // LW_PMGR_CR_ALIAS(i)
                            // LW_PFIFO_CACHE1_GET_INDEX(G)
                            // Those registers will be marked as arrays with size 1. 
                            // Also, the rest of the dimensions are ignored. 
                            // 
                            break;
                        }
                    }

                    vector<unsigned> idx(dimensions.size());
                    for (unsigned i = 0; i<idx.size(); ++i)
                        if (idx[i])
                            idx[i]--;

                    n->low=n->high=0;

                    n->value.eval(n->high, idx.empty() ? 0 : &idx[0]);
                    n->high ++;

                    for (unsigned i = 0; i<idx.size(); ++i)
                        idx[i] = 0;

                    n->value.eval(n->low, idx.empty() ? 0 : &idx[0]);
                }
                else if (specs[4] == 'R') {
                    priv_manual::register_entry * n  = new priv_manual::register_entry();
                    e->entry = n;
                    n->name = e->name;
                    n->value = 0;
                    raw_entries[n->name].eval(n->value, 0);              
                }
                else if (specs[4] == 'V') {
                    priv_manual::value_entry * n = new priv_manual::value_entry();
                    e->entry = n;
                    n->is_initial_value = specs[2] == 'I';
                    n->name = e->name;
                    raw_entries[n->name].eval(n->value, 0);           
                }
                else if (specs[4] == 'F') {
                    priv_manual::field_entry * n = new priv_manual::field_entry();
                    e->entry = n;
                    n->name = e->name;
                    n->value = raw_entries[n->name];

                    //
                    // Explicitly initialize low_bit, high_bit to ~0.
                    // Otherwise, indexed fields are incorrectly being considered as 
                    // being fields at bit position 0. 
                    // 
                    n->low_bit = n->high_bit = ~0;
                    //Check for indexed field
                    if ((raw_entries[n->name]).parameters.size() > 0) {
                        n->is_indexed = true;

                        size_t dims = n->value.parameters.size();
                        vector<unsigned> & dimensions(n->dimensions);
                        for (unsigned int i =0; i!=dims; i++) {
                            // lets analyze the array dimensions.  
                            char size_buffer[24];
                            sprintf(size_buffer, "_SIZE_%d", i+1);
                            prefix_entry * szfield = e->find_field(size_buffer);

                            //Temporary Special cases, until the bugs in the manuals are fixed
                            if (szfield == NULL)
                            {
                            szfield = e->find_field("_SIZE");
                            }
                            if (szfield == NULL)
                            {
                                szfield = e->find_field("_SIZE__%d");
                            }
                            if (szfield != NULL)
                            {
                                raw_entries[szfield->name].eval(n->size, 0);
                            }
                            dimensions.push_back(0);
                            if (szfield) 
                            {
                                raw_entries[szfield->name].eval(dimensions.back(), 0);
                            }
                            else
                            {
                                // 
                                // Our parser cannot handle exceptional cases like,
                                // LW_RAMDVD_CTX_TABLE (63*32+31):( 0*32+ 0) /* RWXUF */
                                // Only three such cases exist in the entire manual tree. 
                                // 
                                break;
                            }
                        }
                    } else {
                        n->is_indexed = false;
                        n->size = 0;
                        n->low_bit = n->high_bit = 0;
                        raw_entries[n->name].eval_range(n->high_bit, n->low_bit, 0);                           
                    }
                }
                else
                    e->entry = e->parent->entry;
            } else {
                e->entry = e->parent->entry;
            }

            // hook up the tree
            if (e->entry != e->parent->entry)
                target->insert(e->parent->entry, e->entry);

            for (map<string, prefix_entry *>::iterator i = e->children.begin(); i!=e->children.end();++i) 
                compile_priv_manual(i->second);
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

        void compile_priv_manual() {
            // now analyze raw entries..
            for (map<string, evaluator>::iterator i = raw_entries.begin(); i!=raw_entries.end();++i) {
                prefix_entry * parent = &root;
                prefix_entry lwrrentEntry(NULL, i->first, getType(i->first));

                while (prefix_entry * n = parent->find_prefix(&lwrrentEntry)) 
                    parent = n;

                parent->insert(i->first, getType(i->first));
            }

            root.parent = &root;
            root.entry = new priv_manual::device_entry();
            target->root = root.entry;
            compile_priv_manual(&root);
        }

        void compile_class(prefix_entry *e, const string & filter_prefix) {

            // if the entry name isn't LW5097_.... (or whatever the class is)
            // then it must be a notifier entry.
            if (e->name[0] !='N' || e->name[1]!='V' || 
                    e->name.substr(2,filter_prefix.size()).compare(filter_prefix) != 0) {
                e->entry = e->parent->entry;
                //goto filtered;
            }
            else
            {

                evaluator & r = raw_entries[e->name];
                if (!r.parameters.empty()) {
                    // some sort of array
                    priv_manual::array_entry * n  = new priv_manual::array_entry();
                    e->entry = n;
                    n->name = e->name;
                    n->value = raw_entries[n->name];

                    size_t dims = n->value.parameters.size();

                    vector<unsigned> & dimensions(n->dimensions);
                    for (unsigned int i =0; i!=dims; i++) 
                        dimensions.push_back(0);
                    n->low=n->high=0;
                } 
                else {

                    unsigned low, high;

                    if (r.eval_range(high, low, 0)) {
                        priv_manual::field_entry * n = new priv_manual::field_entry();
                        e->entry = n;
                        n->name = e->name;

                        n->low_bit = low;
                        n->high_bit = high;
                    } 
                    else if (e->parent->entry->kind == priv_manual::kind_field ||
                            e->parent->entry->kind == priv_manual::kind_value /* some values are prefix's of others*/)
                    {
                        priv_manual::value_entry * n = new priv_manual::value_entry();
                        e->entry = n;
                        n->is_initial_value = false;
                        n->name = e->name;
                        r.eval(n->value, 0);          
                    } else if (e->parent->entry->kind == priv_manual::kind_device ||
                            e->parent->entry->kind == priv_manual::kind_register /* some registers are prefix's of others*/)
                    {
                        priv_manual::register_entry * n  = new priv_manual::register_entry();
                        e->entry = n;
                        n->name = e->name;
                        n->value = 0;
                        r.eval(n->value, 0);                
                    }
                    else
                        assert(0 && "Unknown event entry");
                }

                target->insert(e->parent->entry, e->entry);
            }
//filtered:
            for (map<string, prefix_entry *>::iterator i = e->children.begin(); i!=e->children.end();++i) 
                compile_class(i->second,filter_prefix);
        }

        void compile_class(const string & filter) {
            // now analyze raw entries..
            for (map<string, evaluator>::iterator i = raw_entries.begin(); i!=raw_entries.end();++i) {
                prefix_entry * parent = &root;
                prefix_entry lwr_entry(NULL, i->first, getType(i->first));

                while (prefix_entry * n = parent->find_prefix(&lwr_entry)) 
                    parent = n;

                parent->insert(i->first, getType(i->first));
            }

            root.parent = &root;
            root.entry = new priv_manual::device_entry();
            target->root = root.entry;
            compile_class(&root, filter);    
        }
    };


    class manual_cache {
        string sdkroot, kernelroot;

        priv_manual * load(unsigned classid) {
            char idname[64];
            sprintf(idname, "%04X", classid);
            string path = sdkroot;
            if (!path.empty() && path[path.size()-1]!='\\')
                path+='\\';
            path+="class\\cl";
            path+=idname;
            path+=".h";

            priv_manual * db = new priv_manual;
            post_processor p(db);
            p.add(path);
            p.compile_class(idname);

            return db;
        }

        priv_manual * load(const string & chip) {
            string path = kernelroot;


#if LWWATCHCFG_IS_PLATFORM(UNIX) || LWWATCHCFG_FEATURE_ENABLED(MODS_UNIX)

            if (!path.empty() && path[path.size()-1]!='/')
                path+='/';
            path+=chip;
            path+="/";
            string base = path;

#else
            if (!path.empty() && path[path.size()-1]!='\\')
                path+='\\';
            path+=chip;
            path+="\\";
            string base = path;
            path+="*";
#endif

            priv_manual * db = new priv_manual;
            post_processor p(db);

#if LWWATCHCFG_IS_PLATFORM(UNIX) || LWWATCHCFG_FEATURE_ENABLED(MODS_UNIX)
            DIR *dp;
            struct dirent *ep;

            dp = opendir (path.c_str());
            if (dp != NULL)
            {
                while ((ep = readdir (dp)))
                {
                    string tmpStr = ep->d_name;
                    if ((tmpStr == ".") || (tmpStr == "..") ||
                        (strncmp(tmpStr.c_str(), "dev_lw_xve", 10) == 0))
                        continue;
                    p.add(base + tmpStr);
                }
                (void) closedir (dp);
            }
            else
                dprintf ("Internal error enumerating files in %s : %s\n", path.c_str(), "Couldn't open the directory.");
#else
            // enumerate all files in the directory that match
            WIN32_FIND_DATAA FindFileData;
            HANDLE hFind = ILWALID_HANDLE_VALUE;
            DWORD dwError;

            hFind = FindFirstFileA(path.c_str(), &FindFileData);

            if (hFind != ILWALID_HANDLE_VALUE) 
            {
                p.add(base + FindFileData.cFileName);
                while (FindNextFileA(hFind, &FindFileData) ) 
                    p.add(base + FindFileData.cFileName);

                dwError = GetLastError();
                FindClose(hFind);
                if (dwError != ERROR_NO_MORE_FILES) 
                    dprintf ("Internal error enumerating files in %s : %u\n", path.c_str(), dwError);
            }
#endif
            p.compile_priv_manual();
            return db;
        }



        priv_manual * load(const string chip[], const unsigned numPaths) {
            string path;
            unsigned int i;

#if LWWATCHCFG_IS_PLATFORM(UNIX) || LWWATCHCFG_FEATURE_ENABLED(MODS_UNIX)
            DIR *dp;
            struct dirent *ep;
#else
            WIN32_FIND_DATAA FindFileData;
            HANDLE hFind = ILWALID_HANDLE_VALUE;
            DWORD dwError;
#endif

            priv_manual * db = new priv_manual;
            post_processor p(db);



            for (i = 0; i < numPaths; i++)
            {
               path = kernelroot;
#if LWWATCHCFG_IS_PLATFORM(UNIX) || LWWATCHCFG_FEATURE_ENABLED(MODS_UNIX)

                if (!path.empty() && path[path.size()-1]!='/')
                    path+='/';
                path+=chip[i];
                path+="/";
                string base = path;

#else
                if (!path.empty() && path[path.size()-1]!='\\')
                    path+='\\';
                path+=chip[i];
                path+="\\";
                string base = path;
                path+="*";
#endif


#if LWWATCHCFG_IS_PLATFORM(UNIX) || LWWATCHCFG_FEATURE_ENABLED(MODS_UNIX)

                dp = opendir (path.c_str());
                if (dp != NULL)
                {
                    while ((ep = readdir (dp)))
                    {
                        string tmpStr = ep->d_name;
                        if ((tmpStr == ".") || (tmpStr == "..") ||
                            (strncmp(tmpStr.c_str(), "dev_lw_xve", 10) == 0))
                            continue;
                        p.add(base + tmpStr);
                    }
                    (void) closedir (dp);
                }
                else
                    dprintf ("Internal error enumerating files in %s : %s\n", path.c_str(), "Couldn't open the directory.");
#else
                // enumerate all files in the directory that match
                hFind = ILWALID_HANDLE_VALUE;

                hFind = FindFirstFileA(path.c_str(), &FindFileData);

                if (hFind != ILWALID_HANDLE_VALUE) 
                {
                    p.add(base + FindFileData.cFileName);
                    while (FindNextFileA(hFind, &FindFileData) ) 
                        p.add(base + FindFileData.cFileName);

                    dwError = GetLastError();
                    FindClose(hFind);
                    if (dwError != ERROR_NO_MORE_FILES) 
                        dprintf ("Internal error enumerating files in %s : %u\n", path.c_str(), dwError);
                }
    #endif
            }
            p.compile_priv_manual();
            return db;
        }

        map<unsigned, priv_manual *> manuals;
        map<string, priv_manual * > chips;

        public:
        manual_cache(const string & sdkroot, const string & kernelroot) // eg C:\lw\sw\pvt\main_lw5x\sdk\lwpu\inc\class
            : sdkroot(sdkroot), kernelroot(kernelroot) {}
       
         priv_manual * find(const string & chip) {
            if (!chips[chip])
                return chips[chip] = load(chip);

            return chips[chip];       
        }

        priv_manual * find(unsigned classid) {
            if (!manuals[classid])
                return manuals[classid] = load(classid);

            return manuals[classid];
        }

        priv_manual * find(const string chip[], const unsigned numPaths) {
            if (!chips[chip[0]])
                return chips[chip[0]] = load(chip, numPaths);

            return chips[chip[0]];       
        }
    };

    static manual_cache * cache = 0;

    static void validate_cache() {
        if (cache)
            return;

        const char * path = getelw("LWW_MANUAL_SDK");

        if (!path) {
            dprintf("lw: Please set your LWW_MANUAL_SDK environment variable to point to your "
                    INC_DIR_EXAMPLE " directory\n");
            path = INC_DIR_EXAMPLE;
        }
        cache = new manual_cache(path, string(path));
    }

    priv_manual * get_manual(unsigned classid) {
        validate_cache();
        return cache->find(classid);
    }

    priv_manual * get_manual(const string & chip) {
        validate_cache();
        return cache->find(chip);
    }

    priv_manual * get_manual(const string chip[], const unsigned numPaths) {
        validate_cache();
        return cache->find(chip, numPaths);
    }

    priv_manual * get_active_manual() {
        char *pChipManualsDir[MAX_PATHS];
        char *pClassNum;
        int numPaths = 1;
        int i = 0;
        string strManualsDir[MAX_PATHS] = {""};
        priv_manual *privManual = NULL;

        for(i = 0; i < MAX_PATHS; i++)
        {
            pChipManualsDir[i] = (char *)malloc(32  * sizeof(char));
        }
        pClassNum = (char *)malloc(32 * sizeof(char));

        if(!GetManualsDir(pChipManualsDir, pClassNum, &numPaths))
        {
            dprintf("\n: Unknown or unsupported lWpu GPU. Ensure that %s() supports the chip you're working on.\n",
                __FUNCTION__);
            privManual =  NULL;
            goto Cleanup;
        }

        if (numPaths == 1)
        {
            privManual = get_manual(string(pChipManualsDir[0]));
        }
        else
        {
            for(i = 0; i < MAX_PATHS; i++)
            {
                strManualsDir[i] = pChipManualsDir[i];
            }
            privManual = get_manual(strManualsDir, numPaths);
        }

Cleanup:
        // Lets free the char array
        for(i = 0; i < MAX_PATHS; i++)
        {
            free(pChipManualsDir[i]);
        }
        free(pClassNum);

        return privManual;
    }


