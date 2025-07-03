/*
 * LWIDIA_COPYRIGHT_BEGIN
 *
 * Copyright 2017-2018 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * LWIDIA_COPYRIGHT_END
 */
 
#include <iostream>
#include <fstream>
 
namespace
{
    class VerboseLog
    {
        public:
            VerboseLog() : m_bPrint(false) { }

            template <class T>
            VerboseLog &operator<<(const T &v)
            {
                if (m_bPrint)
                    cout << v;
                return *this;
            }
            void Enable(bool bEnable) { m_bPrint = bEnable; }

        private:
            bool m_bPrint;
    };
    VerboseLog vout;
}

struct DeviceInfo
{
    string                     devType;            // Device type ("gpu", "switch")
    int                        node;               // Node of the device within the protobuf
                                                   // topology description
    int                        index;              // Index of the device within its appropriate
                                                   // device type container in the protobuf
                                                   // topology description
    int                        peerId;
    int                        maxPorts;           // Maximum number of ports on the device
    int                        requesterBase;      // Requester link ID base for the device.
                                                   // Requester link IDs for the device start at
                                                   // the base and are assigned sequentially and
                                                   // based on the port number
    int                        targetId;           // Corresponding targetId for this device (if applicable)
    map<int, pair<string, int>> remoteConnections; // Map of ports on the device to the
                                                   // remote deviceTag/port

    map<int, map<int, set<int>>> routedRequesterLinkIds; // For each port a map of requester link
                                                         // ids to the port that they are routed
                                                         // to

    // Map of ports on the device to the response requirements for the port
    map<int, map<string, set<int>>> responseRequirements;

    map<int, set<string>>      requiredRouting;    // For each port a set of device tags that must
                                                   // contain routes since data is being routed
                                                   // into this port bound for the device tag
    map<int, set<string>>      actualRouting;      // For each port a set of device tags that are
                                                   // actually being routed
    vector<int>                rlanIds;            // The RlanIDs assigned to each port
};