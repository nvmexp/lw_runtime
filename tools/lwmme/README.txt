                               README for LWMME
                         LWPU Method Macro Expander

This directory contains source code for the MME assembler, which is a tool
used to generate macros to be exelwted by the MME unit on Fermi/GF100 and
subsequent chips.  

This unit allows us to define "MME methods" in software that don't need to be
hard-wired in software.  Each MME method triggers the invocation of a program
that can read incoming method data from the pushbuffer, read the current value
of state methods (via a "shadow RAM"), perform simple computations including
conditional branching, and emit regular class methods to be processed by the
rest of the GPU.  The MME assembler is used to generate code for these
programs.

The capabilities of the MME and the assembly instruction set implemented by
the assembler can be found on the LWMME wiki page at:

  https://engwiki/index.php/LWMME

In this directory, there are three subdirectories:

* parser:  lex/yacc definitions for the text assembly language, plus related C
  code to implement the assembler itself.  Includes a "tests" subdirectory
  with sample MME files used as unit tests to ensure proper operation of the
  assembler.

* playback:  Source code to implement a playback capability that takes an
  assembled MME program and input data and generates the set of methods
  emitted.  Used by the assembler to verify proper operation of test cases
  embedded in the assembly code.

* common:  Headers/source code common to the parser and playback units.
