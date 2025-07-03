/********************************************************************/
/*                   OptiX Doxygen Documentation                    */
/*                        LWPU Corporation                        */
/*                      generation instructions                     */
/*                     written by Marco Alesiani                    */
/********************************************************************/

OptiX uses Doxygen to generate its documentation including all sub-
products like OptiX Prime, wrappers like OptiXpp and modules like OptiXu

Make sure to have a tex installation (on Windows I used texlive or MikTex)
and have it added to your PATH variable before any other which may interfere
(e.g. cygwin's one) with the one you intend to use. This is needed by doxygen 
to re-generate the pdf.


= PREREQUISITES =
In order to generate a Doxygen documentation for OptiX one needs to make
sure of the following:

- All headers doxygen-commented that provide data to the documentation should
  reside in the "rtmain/include" directory or in any of its subdirectories
  
- All files which are meant to be processed by Doxygen should retain a
  doxygen-file directive at the top, similar to the following:
  
/**
 * @file   optix.h
 * @author LWPU Corporation
 * @brief  OptiX public API header
 *
 * Includes the host api if compiling host code, includes the lwca api if compiling 
 * device code.
 *
 * For the math library routines include optix_math.h
 */


= GROUPING =
All OptiX main modules are grouped within a doxygen group defined with

/**
 *  @defgroup NEW_MODULE
 */

any subsection should follow a hierarchy of inclusions

/**
 *  @defgroup NEW_MODULE_SUBSECTION
 *  @ingroup NEW_MODULE
 */

in order to appear as a chapter/subchapter hierarchy in the documentation. Similar APIs
can also be grouped (e.g. rtContextLaunch1D, rtContextLaunch2D and rtContextLaunch3D since
they share most of their documentation)

Every group is defined in the doxygen_hierarchy.h which is a fake header which helps
doxygen give a order to all the modules.

=====================================WARNING=================================================
Doxygen sometimes gets confused by big files full of #ifdef, #ifndef, #else, extern "C" { and things
like that. When getting weird errors it is highly recommended NOT to include an entire file like

/**
* This file has a lot of code ifdefs
*
* @{
*/
...
...
...
/** @} */

and prefer something like including everything singularly

/**
* @ingroup mygroup
*/
void myFunction();

/**
* @ingroup mygroup
*/
void myFunction2();

=========================================================================================

= REGENERATING THE PDF =
In order to re-generate the documentation PDF some changes must be made. First ensure that
everything is doxygen-commented and that there are no doxygen warnings/errors by running

  $ doxygen doxygen_optix_dolwmentation

Doxygen will generate files inside a "latex" folder in this directory containing the .tex files necessary
to create the pdf. 

A footer.tex and header.tex which moves "generated with doxygen" at the end and emphasizes "LWPU"
have been added, feel free to modify them as it suits you. These headers also put the front image on the
main page of the documentation and tweak the TOC (table of contents).

After modifying the refman.tex, use make (pdflatex might help) or the supplied make.bat file on
Windows (MikTex might help). The result is the refman.pdf file.


= CAVEATS =
Doxygen supports C and C++ comment styles, but C++ comments doxygen-style are known to
cause problems with C compilers (especially old ones) so as a rule of thumb if a code
is meant to be compiled in both C and C++, it is highly recommended to use C-style
doxygen comments

e.g. 
  /** @brief bla bla.. */
instead of
  /// @brief bla bla  