<?php
require("template.php");

# Now set the title of the page:
setPageTitle("About");

# Make the header.
generateHeader($_SERVER['PHP_SELF']);
?>
<div class="img-right">
	<img src="images/chessdemo.png" alt="chess demo"/><br/>
	GLUT's "Chess" demo<br/>
	running with FreeGLUT.
</div>

<p></p>

<div class="textheader">What?</div>
<p>FreeGLUT is a completely OpenSourced alternative to the OpenGL Utility Toolkit (GLUT) library. GLUT was originally written by Mark Kilgard to support the sample programs in the second edition OpenGL 'RedBook'. Since then, GLUT has been used in a wide variety of practical applications because it is simple, widely available and highly portable.</p>
<p>GLUT (and hence FreeGLUT) allows the user to create and manage windows containing OpenGL contexts on a wide range of platforms and also read the mouse, keyboard and joystick functions.</p>
<p>FreeGLUT is released under the X-Consortium license.</p>

<div class="textheader">Why?</div>
<p>The original GLUT library seems to have been abandoned with the most recent version (3.7) dating back to August 1998. Its license does not allow anyone to distribute modified library code. This would be OK, if not for the fact that GLUT is getting old and really needs improvement. Also, GLUT's license is incompatible with some software distributions (e.g., XFree86).</p>

<div class="textheader">Who?</div>
<p>FreeGLUT was originally written by Pawel W. Olszta with contributions from Andreas Umbach and Steve Baker. Steve is now the official owner/maintainer of FreeGLUT.</p>

<div class="textheader">When?</div>
<p>Pawel started FreeGLUT development on December 1st, 1999. The project is now virtually a 100% replacement for the original GLUT with only a few departures (such as the abandonment of SGI-specific features such as the Dials&amp;Buttons box and Dynamic Video Resolution) and a shrinking set of bugs.</p>
<p>There are some additional features such as a larger set of predefined shapes for use in applications.</p>

<div class="textheader"><a name="download"></a>Help out!</div>
<p>FreeGLUT 3.0 is in active development, and will feature ports to
Android and BlackBerry 10 as well as a host of other enhancements. We are looking
for developers to help out with further work on the Android and BlackBerry 10
ports. Furthermore, ports to Cocoa/Carbon on OSX, and maybe even Wayland
are planned, along with some enhancements to the API and implementation.
See <a href="progress.php">here</a> for an overview of the major points
on our todo list. You can easily help out by forking the unoffical clone
of our <a
href="https://sourceforge.net/p/freeglut/code/HEAD/tree/">sourceforge.net
SVN repository</a> on <a
href="https://github.com/dcnieho/FreeGLUT">github</a>. For more
information about helping out, see the <a href="help.php">Help Out
page</a> and <a
href="http://lists.sourceforge.net/lists/listinfo/freeglut-developer">join</a>
the <a
href="mailto:freeglut-developer@lists.sourceforge.net">freeglut-developer</a>
mailing list.</p>

<div class="textheader"><a name="download"></a>Downloads...</div>
<p>Below are file links for the FreeGLUT project. README files are included. Have fun!</p>

<div class="indent">
	<div class="textheader">Testing Releases</div>
    <p>The ports to Android and BlackBerry 10 as well as other API and
    implementation enhancements (e.g., move to CMake build system, VBO
    and shader support for geometry) are lwrrently only available in
    trunk. Feel free to test by downloading a <a
    href="https://sourceforge.net/p/freeglut/code/HEAD/tarball?path=/trunk/freeglut/freeglut">tarball
    of current trunk</a>, or <a href="help.php#svn">grabbing a copy from
    svn</a>, and give us feedback on how it worked for you. All this
    will eventually become a FreeGLUT 3.0 release.</p>
<!--
	<p>Version 2.8.0, Release Candidate 4 was released on Wednesday, December 21, 2011.</p>
	<p>
                <a href="http://freeglut.svn.sourceforge.net/viewvc/freeglut/tags/FG_2_8_0_RC4.tar.gz">Freeglut 2.8.0 Release Candidate 4</a> [<i>Released: 21 Dec 2011</i>]<br/>
        </p>
	<p>There are no presently active testing releases.
	</p>
-->
</div>


<div class="indent">
        <div class="textheader">Stable Releases</div>
        <p>
	<a href="http://prdownloads.sourceforge.net/freeglut/freeglut-2.8.1.tar.gz?download">Freeglut 2.8.1</a> [<i>Released: 5 April 2013</i>]<br/>
    <a href="http://prdownloads.sourceforge.net/freeglut/freeglut-2.8.0.tar.gz?download">Freeglut 2.8.0</a> [<i>Released: 2 January 2012</i>]<br/>
	<a href="http://prdownloads.sourceforge.net/freeglut/freeglut-2.6.0.tar.gz?download">Freeglut 2.6.0</a> [<i>Released: 27 November 2009</i>]<br/>
	<a href="http://prdownloads.sourceforge.net/freeglut/freeglut-2.4.0.tar.gz?download">Freeglut 2.4.0</a> [<i>Released: 9 June 2005</i>]<br/>
	<a href="http://prdownloads.sourceforge.net/freeglut/freeglut-2.2.0.tar.gz?download">Freeglut 2.2.0</a> [<i>Released: 12 December 2003</i>]<br/>
	<a href="http://prdownloads.sourceforge.net/freeglut/freeglut-2.0.1.tar.gz?download">Freeglut 2.0.1</a> [<i>Released: 23 October 2003</i>]
	</p>

	<div class="textheader">Prepackaged Releases</div>
	<p>The FreeGLUT project does not support packaged versions of FreeGLUT excepting, of course, the tarballs distributed here. However, various members of the community have put time and effort into providing source or binary rollups, and we thank them for their efforts. Here's a list which is likely incomplete:</p>
<!--
	<p>
		Andy Piper's <a href="http://jumpgate.homelinux.net/random/freeglut-fedora/">RedHat Fedora RPMs</a><br/>
		Gentoo <a href="http://bugs.gentoo.org/show_bug.cgi?id=36783">freeglut-2.2.0.ebuild</a><br/>
		Nigel Stewart's <a href="http://www.nigels.com/glt/devpak/">DevPak</a> for <a href="http://www.bloodshed.net/dev/devcpp.html">Dev C++</a>
	</p>
-->
	<p>
		<a href="http://www.transmissionzero.co.uk/software/freeglut-devel/">Martin Payne's Windows binaries (MSVC and MinGW)</a><br/>
		<a href="http://tisch.sf.net/freeglut-2.6.0-mpx-latest.patch">Florian Echtler's MPX Patch</a>
	</p>

	<p>If you have problems with these packages, please contact their maintainers - we of the FreeGLUT team probably can't help.</p>
	
	<div class="textheader">Development Releases</div>
	<p>
		<a href="https://sourceforge.net/p/freeglut/code/HEAD/tarball?path=/trunk/freeglut/freeglut">SVN trunk tarball</a><br/>
		<a href="help.php#svn">Anonymous SVN Instructions</a>
	</p>
</div>

<div class="textheader">Questions?</div>
<p>Don't be afraid to ask for help. We don't bite. Much.</p>
<p>Send FreeGLUT related questions to the appropriate FreeGLUT mailing list:</p>

<ul>
	<li><a href="mailto:freeglut-developer@lists.sourceforge.net">freeglut-developer</a> [<a href="http://lists.sourceforge.net/lists/listinfo/freeglut-developer">Subscribe</a>],</li>
	<li><a href="mailto:freeglut-announce@lists.sourceforge.net">freeglut-announce</a> [<a href="http://lists.sourceforge.net/lists/listinfo/freeglut-announce">Subscribe</a>], and</li>
	<li><a href="mailto:freeglut-bugs@lists.sourceforge.net">freeglut-bugs</a> [<a href="http://lists.sourceforge.net/lists/listinfo/freeglut-bugs">Subscribe</a>]</li>
</ul>

<p>Please note that <a href="http://sourceforge.net/p/freeglut/mailman/?source=navbar">you must subscribe before you can post</a> to our mailing lists. Sorry for the incolwenience.</p>


<?php generateFooter(); ?>
