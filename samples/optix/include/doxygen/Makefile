# Creating the HTML and PDF version of the Doxygen API documentation

# By default, the doxygen command will be found using the current PATH
# environment variable.  If the doxygen command (or an appropriate
# version) is not available, define DOXYGEN_COMMAND to change the
# location of the depot's doxygen command for your workspace.

# By default, doxygen uses a file named "Doxyfile" for the command
# options, which is the filename used in this directory.  If you are
# testing another set of options, you can change the value of the
# DOXYFILE variable.

# The PDF version is created by LaTeX. You should define TEXINPUTS to
# be the top-level of the TeX Live or other distribution.  The 2011
# Tex Live distribution is also available in the depot.

# For Linux:

# The location of the doxgen command in the Perforce depot is:
#    //sw/tools/linux/doxygen/1.8.6/bin/doxygen

# To build the PDF version, you will also need to define TEXINPUTS
# appropriately.
# For example, if your workspace directory is <workspace> (and
# depending on how your workspace is set up), define TEXINPUTS like
# this (for Bash):
#    export TEXINPUTS=.:<workspace>/sw/tools/unix/hosts/Linux-x86/texlive-2011//:


DOXYGEN_COMMAND ?= doxygen
DOXYGEN_VERSION := $(shell $(DOXYGEN_COMMAND) --version)
VERSION_CHECK := $(shell ./valid_doxygen_version.sh $(DOXYGEN_VERSION))
DOXYFILE ?= Doxyfile

all : html pdf

html : html/index.html

pdf : latex/refman.pdf

html/index.html : $(DOXYFILE)
ifneq '$(VERSION_CHECK)' 'true'
	@echo $(VERSION_CHECK)
	exit 1
endif
	@rm -rf html
	$(DOXYGEN_COMMAND) $(DOXYFILE)

latex/refman.pdf : $(DOXYFILE)
	cd latex ; make clean ; cp ../lwda_header.png . ; make

clean:
	rm -rf html latex *.tag
