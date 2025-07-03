Using Doxygen to create API documentation in HTML and PDF format

The raytracing-docs.lwpu.com website provides both HTML and PDF
version of the API documentation of OptiX.  The method of generating
the API documentation defined in this directory has been updated to
provide both formats.

The files used for building the documentation are defined by the
INPUT option in file Doxyfile.  To test the current API build, enter:

   make html

Any warnings for undolwmented parameters, etc., are displayed.

To create the PDF version, enter:

   make pdf

This creates the file latex/refman.pdf.

When the API documentation is built for the raytracing-docs website,
the doxygen command uses the Doxyfile in this directory.  To match the
style in the website, formatting options are changed, but the
content-specific options (like INPUT) are not.  In this way,
developers can test the current state of the API documentation without
needing to test the website build.

The Makefile contains additional information about defining the
location of the doxygen command and the requirements for creating the
PDF version.
