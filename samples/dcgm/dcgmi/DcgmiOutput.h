#ifndef DCGMI_OUTPUT_H_
#define DCGMI_OUTPUT_H_

#include "CommandOutputController.h"
#include <iostream>
#include <json/json.h>
#include <sstream>

/* *****************************************************************************
 * The classes declared in this file generate human- and machine-readable output
 * for DCGMI.
 *
 * To use DcgmiOutput, create an instance of a concrete DcgmiOutput
 * implementation, add values to it, then print instance.str()
 * *****************************************************************************
 */

class DcgmiOutput;

/* Internal class. Do not instantiate directly.
 *
 * Box can contain any type of "boxed" value
 */
class DcgmiOutputBox {
protected:
    // Abstract class
    DcgmiOutputBox();

public:
    // Colwerts boxed value to string
    virtual std::string str() const = 0;
    virtual ~DcgmiOutputBox();
    virtual DcgmiOutputBox* clone() const = 0;
};

/* Internal class. Do not instantiate directly.
 *
 * Concrete implementations of DcgmiOutputBox
 */
template <typename T>
class ConcreteDcgmiOutputBox : public DcgmiOutputBox {
private:
    T value;

public:
    ConcreteDcgmiOutputBox(const T& t);
    virtual ~ConcreteDcgmiOutputBox();
    ConcreteDcgmiOutputBox* clone() const;
    std::string str() const;
};

/* Internal class. Do not instantiate directly.
 *
 * Note: This is only a data structure. It does not control boxing the output */
class DcgmiOutputBoxer {
private:
    static CommandOutputController cmdView;
    DcgmiOutputBox* pb;
    Json::Value& json;
    std::vector<DcgmiOutputBox*> overflow;
    std::map<std::string, DcgmiOutputBoxer*> children;
    std::vector<std::string> childrenOrder;

    template <typename T>
    void set(const T& x);

public:
    DcgmiOutputBoxer(Json::Value&);
    virtual ~DcgmiOutputBoxer();
    DcgmiOutputBoxer(const DcgmiOutputBoxer& other);

    template <typename T>
    DcgmiOutputBoxer& operator=(const T& x);
    DcgmiOutputBoxer& operator[](const std::string& childName);
    template <typename T>
    void addOverflow(const T& x);
    template <typename T>
    void setOrAppend(const T& x);
    std::string str() const;

    const DcgmiOutputBox* getBox() const { return pb; };
    const std::vector<DcgmiOutputBox*>& getOverflow() const { return overflow; };
    const std::map<std::string, DcgmiOutputBoxer*> getChildren() const { return children; };
    std::vector<std::string> getChildrenOrder() const { return childrenOrder; };
};

class DcgmiOutputFieldSelector {
private:
    std::vector<std::string> selectorStrings;
    DcgmiOutputBoxer& run(DcgmiOutputBoxer&, std::vector<std::string>::const_iterator) const;

public:
    DcgmiOutputFieldSelector();
    DcgmiOutputFieldSelector& child(const std::string& childName);
    DcgmiOutputBoxer& run(DcgmiOutputBoxer& out) const;
};

class DcgmiOutputColumnClass {
private:
    unsigned int width;
    std::string name;
    DcgmiOutputFieldSelector selector;

public:
    unsigned int getWidth() const;
    const std::string& getName() const;
    const DcgmiOutputFieldSelector& getSelector() const;
    DcgmiOutputColumnClass(const unsigned int width,
                           const std::string& name,
                           const DcgmiOutputFieldSelector& selector);
};

class DcgmiOutput {
protected:
    std::map<std::string, DcgmiOutputBoxer*> sections;
    std::vector<std::string> sectionOrder;
    std::vector<std::string> header;
    Json::Value json;

    DcgmiOutput();

public:
    DcgmiOutput(const DcgmiOutput& other);
    virtual ~DcgmiOutput();
    DcgmiOutputBoxer& operator[](const std::string& sectionName);
    void addHeader(const std::string& headerStr);
    virtual void addColumn(const unsigned int width,
                           const std::string& columnName,
                           const DcgmiOutputFieldSelector& selector) {}
    virtual std::string str() = 0;
};

class DcgmiOutputTree : public DcgmiOutput {
private:
    unsigned int fullWidth;
    unsigned int rightWidth;

    std::string headerStr(const std::string& line) const;
    std::string levelStr(int level,
                         const std::string& label,
                         const DcgmiOutputBoxer& node) const;
    std::string rowStr(int level,
                       const std::string& prefix,
                       const std::string& label,
                       const std::string& value) const;

public:
    DcgmiOutputTree(unsigned int leftWidth, unsigned int rightWidth);
    virtual ~DcgmiOutputTree();
    std::string str();
};

class DcgmiOutputJson : public DcgmiOutput {
public:
    DcgmiOutputJson();
    virtual ~DcgmiOutputJson();
    std::string str();
};

class DcgmiOutputColumns : public DcgmiOutput {
private:
    unsigned int fullWidth;
    std::vector<DcgmiOutputColumnClass> columns;

    std::string columnLabelsStr() const;
    std::string headerStr(const std::string& line) const;
    std::string overflowStr(DcgmiOutputBoxer& boxer) const;
    std::string rowStr(const std::vector<std::string>& strs) const;
    std::string sectionStr(const std::string& sectionName, DcgmiOutputBoxer& boxer) const;

public:
    DcgmiOutputColumns();
    virtual ~DcgmiOutputColumns();
    void addColumn(const unsigned int width,
                   const std::string& columnName,
                   const DcgmiOutputFieldSelector& selector);
    std::string str();
};

/* ************************************************************************** */
/* ****************** This is the end of the declarations. ****************** */
/* ******************     (Template) definitions below     ****************** */
/* ************************************************************************** */

// Templates have to be visible to the compiler

template <typename T>
void deleteNotNull(T*& obj)
{
    if (NULL != obj) {
        delete obj;
        obj = NULL;
    }
}

/******* ConcreteDcgmiOutputBox *******/

template <typename T>
ConcreteDcgmiOutputBox<T>::ConcreteDcgmiOutputBox(const T& t)
    : value(t){};

template <typename T>
ConcreteDcgmiOutputBox<T>::~ConcreteDcgmiOutputBox() {}

template <typename T>
ConcreteDcgmiOutputBox<T>* ConcreteDcgmiOutputBox<T>::clone() const
{
    return new ConcreteDcgmiOutputBox<T>(*this);
}

template <typename T>
std::string ConcreteDcgmiOutputBox<T>::str() const
{
    std::ostringstream ss;
    ss << value;
    return ss.str();
}

/******* DcgmiOutputBoxer *******/

template <typename T>
void DcgmiOutputBoxer::addOverflow(const T& x)
{
    std::string str = cmdView.HelperDisplayValue(x);
    overflow.push_back(new ConcreteDcgmiOutputBox<std::string>(str));
    json["overflow"].append(str);
}

template <typename T>
void DcgmiOutputBoxer::set(const T& x)
{
    std::string str = cmdView.HelperDisplayValue(x);
    deleteNotNull(pb);
    pb = (new ConcreteDcgmiOutputBox<std::string>(str));
    json["value"] = str;
}

template <typename T>
void DcgmiOutputBoxer::setOrAppend(const T& x)
{
    if (NULL == pb) {
        set(x);
    } else {
        addOverflow(x);
    }
}

template <typename T>
DcgmiOutputBoxer& DcgmiOutputBoxer::operator=(const T& x)
{
    set(x);
    return *this;
}

#endif /* DCGMI_OUTPUT_H_ */
