sigdump:
Adding tabularize.pl which colwerts mux write and signal read sequence
provided by HW to individual tables used by the sigdump command.
Based on transform.pl used for pre-fermi sigdump.

For each chip, HW creates the input files with a flow called
"pmlsplitter", and checks them into
//hw/tools/pmlsplitter/<chip>/sigdump.txt.  Usually there is a file
for each chip, but sometimes one file from the largest chip in a
litter works for the smaller chips.  Ask the owner of pmlsplitter
(notes below on how to file the bug).  Sigdump uses the lwwatch HAL
chip identification macros such as IsGF100() in
sigdump_gf100.c:openSigdumpFiles() to determine which
signals_<chip>.sig and regsinfo_<chip>.sig files to use.  When adding
new chips, extend openSigdumpFiles() accordingly.

For example, the gm107 signals and regsinfo files are used for gm108.
Sigdump ignores the 'missing' chip instances (this is also how it
handles floorswept parts).  Lwwatch also only wires up a GM107 HAL.
The 3 GM20x chips (GM204, GM206, GM200) each require unique sigdump
files due to internal wiring differences, however lwwatch only wires
up the GM200 HAL as it's the largest chip in the litter and is a
logical superset of all 3.

File an Architecture bug on module "Architecture Tools - pmlsplitter"
to find out which sigdump.txt file to use for new chips.  

File a SW bug on module "sw-lwwatch" and ARB the chip SW manager (if
known) to confirm that the RM team will configure the lwwatch HAL to
match the sigdump expectations.

The tabularize.pl flow expects the "sigdump.txt" file to be renamed
"sigdump_<chip>.c".  The file is parsed to create the register or
signal table per the command line option.

Usage: 
tabularize.pl FILENAME.c --regsinfo  //For ouputing the register table
tabularize.pl FILENAME.c --siginfo  //For outputing the signal table

Example for GM204
# Assumes you have a HW client with pmlsplitter files.  You can also
# get them via http://p4viewer.lwpu.com//get/hw/tools/pmlsplitter/
cd <lwwatch client path>/tools
cp <HW client path>/hw/tools/pmlsplitter/gm204/sigdump.txt sigdump_gm204.c
tabularize.pl sigdump_gm204.c --siginfo > signals_gm204.sig
tabularize.pl sigdump_gm204.c --regsinfo > regsinfo_gm204.sig
cp signals_gm204.sig regsinfo_gm204.sig ..   # They are checked into here

Expected input file format : 
...
RegWrite(addr,value,mask);
RegWrite(addr,value,mask);
...
RegWrite(addr,value,mask);
OutputSignal(fp,"instance_class.signal_name[chiplet,instance]: ", RegBitRead(addr,lsb,msb));
...

Output formats:
signals_<chip>.sig:
  instanceClass.signame[chiplet,instance]: signal_reg_addr lsb msb num_writes
    - signal_reg_addr: (perfmon) PRI to read for the signal value
    - lsb: lsb of register value field in the PRI
    - msb: msb of register value field in the PRI
    - num_writes: number of register setup writes needed.  Sigdump 
      reads <num_writes> lines from the regsinfo_<chip>.sig file
      and performs those writes prior to reading this signal

regsinfo_<chip>.sig:
  reg_addr reg_mask reg_value
    - reg_addr: PRI write address
    - reg_mask: Mask for the written value.  Sigdump does a read-modify-write
      to preserve the rest of the register
    - reg_value: value to write in the masked field
