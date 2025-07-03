lwmake build test suite.

This test suite should cover all OptiX white box unit tests. For perf tests we continue to rely on OptiXMark.
The test suite is based on Robot Framework (www.robotframework.org), a small Python-based testing utility.


External dependencies
---------------------

    - Python 2.7
	
	
Internal dependencies in addition to your lwmake tree
-----------------------------------------------------

    - An lwmake build
    - To run the unit test suite: builds of the test_* projects in the bin folder of the lwmake build


Running
-------

To run the pre-checkin smoke tests, simply execute "test_smoke[.sh|.bat]". You can execute this from within
any directory.

The output will be written to a directory called "report", relative to the current working directory.

There are other runner scripts to run different pieces of the test suite. Either use them as-is or as templates
for custom RF ilwocations.


Editing tests
-------------

Robot Framework defines tests as tables in .tsv files (tab separated values). It's easiest to edit them
using Excel (the "auto-size column width" function is your friend). The main thing to watch out for is that
when newly opening a file, Excel can get confused about fields that contain hyphen args to commands (e.g. the
"-o" in an oac command), because it tries to interpret the field as a formula. Just change it to text and
make sure Excel didn't mess something up when the file was opened.

Excel also sometimes decides to put quotes around cell content (e.g. a long comment). RF handles this, so
you don't have to worry about it.

The .tsv files can also be edited using a regular text editor. When doing that, make sure that tabs are
preserved correctly -- it's tabs and not 4  spaces, and double-tabs are not allowed (that counts as 2 columns).
It's easiest to use an editor that can visually display tabs.


Adding tests
------------

Lwrrently, the lwmake test suite supports Optix unit tests (runs the test and checks the exit code).
To add one of these, choose the testlist it should go into.  Then it's essentially monkey-see-monkey-do
to add the actual test case. If it's not clear what arguments to the run keywords are available,
reference their implementation in the "tools" directory.
