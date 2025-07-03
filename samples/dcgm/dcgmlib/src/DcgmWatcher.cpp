
#include "DcgmWatcher.h"

/*****************************************************************************/
bool DcgmWatcher::operator==(const DcgmWatcher& other)
{
    return (this->watcherType == other.watcherType) && (this->connectionId == other.connectionId);
}

/*****************************************************************************/
bool DcgmWatcher::operator!=(const DcgmWatcher& other)
{
    return !((*this) == other);
}

/*****************************************************************************/

