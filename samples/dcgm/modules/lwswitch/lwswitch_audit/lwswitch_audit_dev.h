#pragma once
/*
 * File:   lwswitch_audit_dev.h
 */

#include <stdbool.h>
#include <sys/types.h>

#define SWITCH_PATH_LEN         64

/*
dev_id: Switch instance in /dev/
Return: Corresponding switch id in HWLinks table
*/
int naGetDevToSwitchID(int dev_id);

/*
switch_id: Switch ID in HWLinks table
Return: Corresponding switch instance in /dev/
*/
int naGetSwitchToDevID(int switch_id);

/*
Opens lwswitch device
dev_id: Switch instance in /dev/
Return: File descriptor for opened device
*/
int naOpenSwitchDev(int dev_id);

/*
Read switch physical IDs from switches and populate the
switch ID <-> Switch instance maps
*/
bool naReadSwitchIDs(int num_switches);

/*Get port Mast for lwswitch device
fd: file descriptor for opened switch device
mask: reference variable in which value of 64bitmask is returned
Return: true = successfully read mask 
        false = failed to read mask
*/
bool naReadPortMask(int fd, uint64_t &port_mask);

/*Is port enabled
mask: Port mask previously read from device
port_num: port number to check if enabled disabled
Return: true=enabled, false=disabled
*/
bool naIsPortEnabled(uint64_t mask, int port_num);

/*Read requestor link ID from switch
fd: File desciptor for switch
switch_port: port from which to read the requestor link ID
Return: requestor link ID
*/
int naReadReqLinkId(int fd, uint32_t switch_port);

