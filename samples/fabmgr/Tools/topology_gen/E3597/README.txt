The .csv and .sh files located in this directory are used to create FM topology files for WOLF bringup and validation work.

Phase1: Wolf Production Testing Using MODS
==========================================
Wolf LWLink Connections for MODS Loopback Testing (See Section 2.1)
- https://docs.google.com/document/d/1D74W-x-_NYAF-WIMZKxeDWw5kbhlv0mJ5ApTyUeilrU/edit#heading=h.mgb0ztd9cf8p

Bug
- https://lwbugswb.lwpu.com/LwBugs5/SWBug.aspx?bugid=3399986 

Source files:
- e3597_mods_loopout.csv
- e3597_mods_loopout.sh

Generated files:
- e3597_mods_loopout.topo.txt
- e3597_mods_loopout.topo_table.txt
- e3597_mods_loopout.paths.txt
- e3597_mods_loopout.topo.bin

Please gunzip and copy the e3597_mods_loopout.topo.bin file to the target wolf system under /usr/share/lwswitch/lwpu.

Phase2a: Wolf LWOS Loopout Bringup
==================================
Wolf LWLink connections for Looput Configuration (See section 3.1)
- https://docs.google.com/document/d/1D74W-x-_NYAF-WIMZKxeDWw5kbhlv0mJ5ApTyUeilrU/edit#heading=h.6g75o6e0xtzw

Source files:
- e3597_lwos_loopout.csv
- e3597_lwos_loopout.sh

Generated files:
- e3597_lwos_loopout.topo.txt
- e3597_lwos_loopout.topo_table.txt
- e3597_lwos_loopout.paths.txt
- e3597_lwos_loopout.topo.bin

Please gunzip and copy the e3597_lwos_loopout.topo.bin file to the target wolf system under /usr/share/lwswitch/lwpu.

Phase2b: Wolf LWOS Back2back Bringup
====================================
Wolf LWLink connections for back2back Configuration (See section 3.2)
- https://docs.google.com/document/d/1D74W-x-_NYAF-WIMZKxeDWw5kbhlv0mJ5ApTyUeilrU/edit#heading=h.4ijt7v3yh9sg

Source files:
- e3597_lwos_b2b.csv
- e3597_lwos_b2b.sh

Generated files:
- e3597_lwos_b2b.topo.txt
- e3597_lwos_b2b.topo_table.txt
- e3597_lwos_b2b.paths.txt
- e3597_lwos_b2b.topo.bin

Please gunzip and copy the e3597_lwos_b2b.topo.bin file to the target wolf system under /usr/share/lwswitch/lwpu.

Phase3: 1 Prospector + 3 Wolfs
==============================
Wolf LWLink connections for Single Prospector Configuration (See section 4.1)
- https://docs.google.com/document/d/1D74W-x-_NYAF-WIMZKxeDWw5kbhlv0mJ5ApTyUeilrU/edit#heading=h.grzshtbiwhrb

Source files:
- topology_gen_WOLF_LR.py
- e3597_prospector_1node.csv
- e3597_prospector_1node.sh

Generated files:
- e3597_prospector_1node.topo.txt
- e3597_prospector_1node.topo_table.txt
- e3597_prospector_1node.paths.txt
- e3597_prospector_1node.topo.bin

Please gunzip and copy the e3597_prospector_1node.topo.bin file to the target wolf system under /usr/share/lwswitch/lwpu.

Phase4: 2 Prospector + 3 Wolfs
==============================
Wolf LWLink connections for Two Prospector Configuration (See section 5.1)
- https://docs.google.com/document/d/1D74W-x-_NYAF-WIMZKxeDWw5kbhlv0mJ5ApTyUeilrU/edit#heading=h.i3hv62dfeove

Source files:
- e3597_prospector_2node.csv
- e3597_prospector_2node.sh

Generated files:
- e3597_prospector_2node.topo.txt
- e3597_prospector_2node.topo_table.txt
- e3597_prospector_2node.paths.txt
- e3597_prospector_2node.topo.bin

Please gunzip and copy the e3597_prospector_2node.topo.bin file to the target wolf system under /usr/share/lwswitch/lwpu.

Phase5: 4 Prospectors + 6 Wolfs
===============================
Wolf LWLink connections for Four Prospector Configuration (See section 6.1)
- https://docs.google.com/document/d/1D74W-x-_NYAF-WIMZKxeDWw5kbhlv0mJ5ApTyUeilrU/edit#heading=h.jysw42561nin

Source files:
- e3597_prospector_4node.csv
- e3597_prospector_4node.sh

Generated files:
- e3597_prospector_4node.topo.txt
- e3597_prospector_4node.topo_table.txt
- e3597_prospector_4node.paths.txt
- e3597_prospector_4node.topo.bin

Please gunzip and copy the e3597_prospector_4node.topo.bin file to the target wolf system under /usr/share/lwswitch/lwpu.
