#ifndef DCGMWATCHER_H
#define DCGMWATCHER_H

#include "LwcmConnection.h"

/* DcgmWatcherType is defined in dcgm_structs_internal.h */

/*****************************************************************************/
class DcgmWatcher
{
public:
    /* Constructor */
    DcgmWatcher(DcgmWatcherType_t watcherType, dcgm_connection_id_t connectionId = DCGM_CONNECTION_ID_NONE) : 
                watcherType(watcherType), connectionId(connectionId) {};
    DcgmWatcher() : watcherType(DcgmWatcherTypeClient), connectionId(DCGM_CONNECTION_ID_NONE) {};
    
    /* Destructor */
    ~DcgmWatcher() {};

    DcgmWatcherType_t watcherType;     /* Watcher type */
    dcgm_connection_id_t connectionId; /* Connection associated with this watcher */

    /* Operators */
    bool operator==(const DcgmWatcher& other);
    bool operator!=(const DcgmWatcher& other);
};


/*****************************************************************************/

#endif //DCGMWATCHER_H

