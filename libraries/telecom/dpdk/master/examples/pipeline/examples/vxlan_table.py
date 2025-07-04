#!/usr/bin/elw python3
# SPDX-License-Identifier: BSD-3-Clause
# Copyright(c) 2020 Intel Corporation
#

"""
A Python program that generates the VXLAN tunnels for this example.
"""

import argparse

DESCRIPTION = 'Table Generator'

KEY = '0xaabbccdd{0:04x}'
ACTION = 'vxlan_encap'
ETHERNET_HEADER = 'ethernet_dst_addr N(0xa0a1a2a3{0:04x}) ' \
    'ethernet_src_addr N(0xb0b1b2b3{0:04x}) ' \
    'ethernet_ether_type N(0x0800)'
IPV4_HEADER = 'ipv4_ver_ihl N(0x45) ' \
    'ipv4_diffserv N(0) ' \
    'ipv4_total_len N(50) ' \
    'ipv4_identification N(0) ' \
    'ipv4_flags_offset N(0) ' \
    'ipv4_ttl N(64) ' \
    'ipv4_protocol N(17) ' \
    'ipv4_hdr_checksum N(0x{1:04x}) ' \
    'ipv4_src_addr N(0xc0c1{0:04x}) ' \
    'ipv4_dst_addr N(0xd0d1{0:04x})'
UDP_HEADER = 'udp_src_port N(0xe0{0:02x}) ' \
    'udp_dst_port N(4789) ' \
    'udp_length N(30) ' \
    'udp_checksum N(0)'
VXLAN_HEADER = 'vxlan_flags N(0) ' \
    'vxlan_reserved N(0) ' \
    'vxlan_vni N({0:d}) ' \
    'vxlan_reserved2 N(0)'
PORT_OUT = 'port_out H({0:d})'

def ipv4_header_checksum(i):
    cksum = (0x4500 + 0x0032) + (0x0000 + 0x0000) + (0x4011 + 0x0000) + (0xc0c1 + i) + (0xd0d1 + i)
    cksum = (cksum & 0xFFFF) + (cksum >> 16)
    cksum = (cksum & 0xFFFF) + (cksum >> 16)
    cksum = ~cksum & 0xFFFF
    return cksum

def table_generate(n, p):
    for i in range(0, n):
        print("match %s action %s %s %s %s %s %s" %
              (KEY.format(i), ACTION,
               ETHERNET_HEADER.format(i),
               IPV4_HEADER.format(i, ipv4_header_checksum(i)),
               UDP_HEADER.format(i % 256),
               VXLAN_HEADER.format(i),
               PORT_OUT.format(i % p)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=DESCRIPTION)

    parser.add_argument(
        '-n',
        help='number of table entries (default: 65536)',
        required=False,
        default=65536)

    parser.add_argument(
        '-p',
        help='number of network ports (default: 4)',
        required=False,
        default=4)

    args = parser.parse_args()
    table_generate(int(args.n), int(args.p))
