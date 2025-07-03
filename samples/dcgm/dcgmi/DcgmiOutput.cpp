#include "DcgmiOutput.h"
#include <dcgm_structs_internal.h>
#include <iomanip>
#include <iostream>

std::string makeHline(char c, const std::vector<unsigned int>& widths)
{
    std::ostringstream ss;
    ss << '+';
    for (std::vector<unsigned int>::const_iterator lwr = widths.begin(); lwr != widths.end(); lwr++) {
        ss << std::string(*lwr, c);
        ss << '+';
    }
    ss << '\n';
    return ss.str();
}

/******* DcgmiOutputBox *******/

DcgmiOutputBox::DcgmiOutputBox() {}

DcgmiOutputBox::~DcgmiOutputBox() {}

/******* DcgmiOutputBoxer *******/

CommandOutputController DcgmiOutputBoxer::cmdView;

DcgmiOutputBoxer::DcgmiOutputBoxer(Json::Value& json)
    : pb(NULL)
    , json(json)
{
}

DcgmiOutputBoxer::DcgmiOutputBoxer(const DcgmiOutputBoxer& other)
    : json(other.json)
{
    std::map<std::string, DcgmiOutputBoxer*>::const_iterator maplwr;
    std::vector<DcgmiOutputBox*>::const_iterator vectorlwr;

    pb = NULL;

    if (NULL != other.pb) {
        pb = other.pb->clone();
    }

    childrenOrder = other.childrenOrder;

    for (maplwr = other.children.begin(); maplwr != other.children.end(); maplwr++) {
        children[maplwr->first] = new DcgmiOutputBoxer(*maplwr->second);
    }

    for (vectorlwr = other.overflow.begin(); vectorlwr != other.overflow.end(); vectorlwr++) {
        overflow.push_back((*vectorlwr)->clone());
    }
}

DcgmiOutputBoxer::~DcgmiOutputBoxer()
{
    std::map<std::string, DcgmiOutputBoxer*>::iterator maplwr;
    std::vector<DcgmiOutputBox*>::iterator vectorlwr;

    deleteNotNull(pb);

    for (maplwr = children.begin(); maplwr != children.end(); maplwr++) {
        delete maplwr->second;
    }

    for (vectorlwr = overflow.begin(); vectorlwr != overflow.end(); vectorlwr++) {
        delete *vectorlwr;
    }
}

DcgmiOutputBoxer& DcgmiOutputBoxer::operator[](const std::string& childName)
{
    std::map<std::string, DcgmiOutputBoxer*>::iterator lwr;
    Json::Value& jsonChild = json["children"][childName];

    lwr = children.find(childName);
    if (lwr == children.end()) {
        // It does not exist. Allocate memory for it
        children[childName] = new DcgmiOutputBoxer(jsonChild);
        // Also add it to order
        childrenOrder.push_back(childName);
    }
    return *children[childName];
};

std::string DcgmiOutputBoxer::str() const
{
    if (NULL != pb) {
        return pb->str();
    } else {
        return "";
    }
}

/******* DcgmiOutputFieldSelector *******/

DcgmiOutputFieldSelector::DcgmiOutputFieldSelector() {}

DcgmiOutputFieldSelector& DcgmiOutputFieldSelector::child(const std::string& childName)
{
    selectorStrings.push_back(childName);
    return *this;
}

DcgmiOutputBoxer&
DcgmiOutputFieldSelector::run(DcgmiOutputBoxer& out) const
{
    DcgmiOutputBoxer* result = &out;
    std::vector<std::string>::const_iterator lwr;

    for (lwr = selectorStrings.begin(); lwr != selectorStrings.end(); lwr++) {
        // First dereference the boxer (*result)
        // Then get the sub-boxer with the name [*lwr]
        // Then get the address of that sub-boxer
        result = &(*result)[*lwr];
    }
    return *result;
}

/******* DcgmiOutputColumnClass *******/

unsigned int DcgmiOutputColumnClass::getWidth() const
{
    return width;
}

const std::string& DcgmiOutputColumnClass::getName() const
{
    return name;
}

const DcgmiOutputFieldSelector& DcgmiOutputColumnClass::getSelector() const
{
    return selector;
}

DcgmiOutputColumnClass::DcgmiOutputColumnClass(const unsigned int width,
    const std::string& name,
    const DcgmiOutputFieldSelector& selector)
    : width(width)
    , name(name)
    , selector(selector)
{
}

/******* DcgmiOutput *******/

DcgmiOutput::DcgmiOutput() {}

DcgmiOutput::~DcgmiOutput()
{
    std::map<std::string, DcgmiOutputBoxer*>::iterator lwr;
    for (lwr = sections.begin(); lwr != sections.end(); lwr++) {
        delete lwr->second;
    }
}

DcgmiOutput::DcgmiOutput(const DcgmiOutput& other)
{
    json = other.json;
    header = other.header;
    sectionOrder = other.sectionOrder;
    std::map<std::string, DcgmiOutputBoxer*>::const_iterator lwr;
    for (lwr = other.sections.begin(); lwr != other.sections.end(); lwr++) {
        sections[lwr->first] = new DcgmiOutputBoxer(*lwr->second);
    }
}

DcgmiOutputBoxer& DcgmiOutput::operator[](const std::string& sectionName)
{
    std::map<std::string, DcgmiOutputBoxer*>::iterator lwr;
    Json::Value& jsonChild = json["body"][sectionName];

    lwr = sections.find(sectionName);
    if (lwr == sections.end()) {
        // It does not exist. Allocate memory for it
        sections[sectionName] = new DcgmiOutputBoxer(jsonChild);
        // Also add it to order
        sectionOrder.push_back(sectionName);
    }

    return *sections[sectionName];
};

void DcgmiOutput::addHeader(const std::string& headerStr)
{
    Json::Value& jsonHeader = json["header"];
    header.push_back(headerStr);
    jsonHeader.append(headerStr);
};

/******* DcgmiOutputTree *******/

DcgmiOutputTree::DcgmiOutputTree(unsigned int leftWidth, unsigned int rightWidth)
    : fullWidth(leftWidth + rightWidth)
    , rightWidth(rightWidth)
{
}

DcgmiOutputTree::~DcgmiOutputTree() {}

std::string DcgmiOutputTree::str()
{
    std::string result;
    std::vector<std::string>::const_iterator lwr;
    const unsigned int separatorPosition = fullWidth - rightWidth;

    std::vector<unsigned int> widths;
    widths.push_back(separatorPosition - 1);
    widths.push_back(rightWidth - 2);
    const std::string hline = makeHline('-', widths);

    result += hline;

    if (header.size() > 0) {
        for (lwr = header.begin(); lwr != header.end(); lwr++) {
            result += headerStr(*lwr);
        }
        result += makeHline('=', widths);
    }

    for (lwr = sectionOrder.begin(); lwr != sectionOrder.end(); lwr++) {
        result += levelStr(0, *lwr, *sections[*lwr]);
    }

    // Footer
    result += hline;
    return result;
}

std::string DcgmiOutputTree::headerStr(const std::string& line) const
{
    std::ostringstream ss;
    ss << "| ";
    ss << std::left << std::setw(fullWidth - 3) << line;
    ss << "|\n";
    return ss.str();
}

std::string DcgmiOutputTree::levelStr(int level, const std::string& label, const DcgmiOutputBoxer& node) const
{
    std::string prefix;
    std::ostringstream ss;
    std::vector<DcgmiOutputBox*>::const_iterator boxlwr;
    std::vector<std::string>::iterator strlwr;

    std::vector<std::string> childrenOrder = node.getChildrenOrder();
    const std::vector<DcgmiOutputBox*>& overflow = node.getOverflow();
    std::map<std::string, DcgmiOutputBoxer*> children = node.getChildren();

    // 0 is a special case
    if (level != 0) {
        prefix = "-> ";
    } else {
        prefix = "";
    }

    ss << rowStr(level, prefix, label, node.str());

    for (boxlwr = overflow.begin(); boxlwr != overflow.end(); boxlwr++) {
        // No prefix or label for overflow
        ss << rowStr(level, "", "", (*boxlwr)->str());
    }

    for (strlwr = childrenOrder.begin(); strlwr != childrenOrder.end(); strlwr++) {
        ss << levelStr(level + 1, *strlwr, *(children[*strlwr]));
    }

    return ss.str();
}

std::string DcgmiOutputTree::rowStr(int level, const std::string& prefix, const std::string& label, const std::string& value) const
{
    const unsigned int leftWidth = fullWidth - rightWidth;
    const unsigned int leftTextArea = leftWidth - 3;
    const unsigned int rightTextArea = rightWidth - 4;
    std::string left;
    std::string right;
    std::ostringstream ss;

    ss << std::setw(level * 3) << prefix;
    ss << label;
    left = ss.str();

    ss.str(std::string());
    ss.clear();
    ss << std::left << std::setw(rightTextArea) << value;
    right = ss.str();

    ss.str(std::string());
    ss.clear();
    ss << "| ";
    ss << std::left << std::setw(leftTextArea) << left;
    ss << " | ";
    ss << std::left << std::setw(rightTextArea) << right;
    ss << " |\n";

    return ss.str();
}

/******* DcgmiOutputColumns *******/

DcgmiOutputColumns::DcgmiOutputColumns()
    : fullWidth(0)
{
}

DcgmiOutputColumns::~DcgmiOutputColumns() {}

void DcgmiOutputColumns::addColumn(const unsigned int width,
    const std::string& columnName,
    const DcgmiOutputFieldSelector& selector)
{
    const DcgmiOutputColumnClass column(width, columnName, selector);
    fullWidth += 1 + width;
    columns.push_back(column);
}

std::string DcgmiOutputColumns::str()
{
    std::string result;
    std::vector<std::string>::iterator lwr;
    std::vector<unsigned int> widths;

    for (std::vector<DcgmiOutputColumnClass>::const_iterator lwr = columns.begin();
         lwr != columns.end();
         lwr++) {
        widths.push_back(lwr->getWidth());
    }

    const std::string hline = makeHline('-', widths);

    result += hline;

    if (header.size() > 0) {
        for (lwr = header.begin(); lwr != header.end(); lwr++) {
            result += headerStr(*lwr);
        }
        result += makeHline('=', widths);
    }

    result += columnLabelsStr();
    result += hline;

    for (lwr = sectionOrder.begin(); lwr != sectionOrder.end(); lwr++) {
        result += sectionStr(*lwr, *sections[*lwr]);
    }

    // Footer
    result += hline;
    return result;
}

std::string DcgmiOutputColumns::columnLabelsStr() const
{
    std::vector<std::string> strs;

    for (std::vector<DcgmiOutputColumnClass>::const_iterator lwr = columns.begin();
         lwr != columns.end();
         lwr++) {
        strs.push_back(lwr->getName());
    }

    return rowStr(strs);
}

std::string DcgmiOutputColumns::headerStr(const std::string& line) const
{
    std::ostringstream ss;
    ss << "| ";
    ss << std::left << std::setw(fullWidth - 2) << line;
    ss << "|\n";
    return ss.str();
}

std::string DcgmiOutputColumns::overflowStr(DcgmiOutputBoxer& boxer) const
{
    size_t maxsize = 0;
    std::string result;
    std::vector<std::string> strs;
    std::vector<const DcgmiOutputBoxer*> subboxes;
    std::vector<const DcgmiOutputBoxer*>::const_iterator boxlwr;
    std::vector<DcgmiOutputColumnClass>::const_iterator columnlwr;
    DcgmiOutputBoxer* pboxer;

    for (columnlwr = columns.begin();
         columnlwr != columns.end();
         columnlwr++) {
        pboxer = &columnlwr->getSelector().run(boxer);
        subboxes.push_back(pboxer);
        maxsize = DCGM_MAX(maxsize, pboxer->getOverflow().size());
    }

    for (size_t overflowLine = 0; overflowLine < maxsize; overflowLine++) {
        strs.clear();
        for (boxlwr = subboxes.begin(); boxlwr != subboxes.end(); boxlwr++) {
            if (overflowLine < (*boxlwr)->getOverflow().size()) {
                strs.push_back((*boxlwr)->getOverflow()[overflowLine]->str());
            } else {
                // If no overflow, push an empty string so we have something to
                // print for this field
                strs.push_back("");
            }
        }
        result += rowStr(strs);
    }

    return result;
}

std::string DcgmiOutputColumns::rowStr(const std::vector<std::string>& strs) const
{
    std::ostringstream ss;
    std::vector<DcgmiOutputColumnClass>::const_iterator columnlwr;
    std::vector<std::string>::const_iterator strlwr;
    bool firstCol = true;

    for (strlwr = strs.begin(), columnlwr = columns.begin();
         columnlwr != columns.end();
         columnlwr++, strlwr++) {
        ss << (firstCol ? "| " : " | ")
           << std::setw(columnlwr->getWidth() - 2)
           << std::left << *strlwr;
        firstCol = false;
    }

    ss << " |\n";

    return ss.str();
}

std::string DcgmiOutputColumns::sectionStr(const std::string& sectionName, DcgmiOutputBoxer& boxer) const
{
    std::string result;
    std::vector<std::string> strs;

    for (std::vector<DcgmiOutputColumnClass>::const_iterator lwr = columns.begin();
         lwr != columns.end();
         lwr++) {
        strs.push_back(lwr->getSelector().run(boxer).str());
    }

    result = rowStr(strs);
    result += overflowStr(boxer);

    return result;
}

/******* DcgmiOutputJson *******/

DcgmiOutputJson::DcgmiOutputJson() {}

DcgmiOutputJson::~DcgmiOutputJson() {}

std::string DcgmiOutputJson::str()
{
    return json.toStyledString();
}
