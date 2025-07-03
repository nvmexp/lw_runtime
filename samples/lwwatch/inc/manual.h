#ifndef __LWWATCH_MANUAL_H
#define __LWWATCH_MANUAL_H

#include <vector>
#include <string>
#include <map>
using namespace std;

#include "eval.h"

struct priv_manual {
    enum kind_t{
            kind_device,
            kind_array,
            kind_register,
            kind_value,
            kind_field
        };

    struct entry {
        entry * parent;
        kind_t kind;
 
        string              name;
        virtual ~entry() {}
        vector<entry * >    children;

        entry(kind_t kind) : kind(kind) {}
    };

    struct field_entry : public entry{
        //If field is indexed, we can figure out the dimensions by using the high and low bits
        evaluator           value;
        vector<unsigned>    dimensions;
        unsigned            low_bit, high_bit, size;
        bool                is_indexed;
        field_entry() : entry(kind_field) {}
    };

    struct value_entry : public entry{
        bool        is_initial_value;
        unsigned    value;

        value_entry() : entry(kind_value) {}
    };

    struct array_entry : public entry {
        evaluator           value;
        vector<unsigned>    dimensions;
        unsigned            low, high;

        array_entry() : entry(kind_array) {}
    };

    struct device_entry : public entry{
        device_entry() : entry(kind_device) {}
    };

    struct register_entry : public entry {
        unsigned            value;
        register_entry() : entry(kind_register) {}
    };

    map<unsigned, entry *> address_to_register;
    map<string,   entry *> name_to_register;

    entry                * root;

    void insert(entry * parent, entry * child) {
        child->parent = parent;
        parent->children.push_back(child);

        if (child->kind == kind_register)
            address_to_register[((register_entry*)child)->value] = child;
        else if (child->kind == kind_array)
            address_to_register[((array_entry*)child)->low] = child;

        name_to_register[child->name] = child;
    }

};

priv_manual * get_manual(unsigned classid);
priv_manual * get_manual(const string & chip);
priv_manual * get_active_manual(void);

#endif
