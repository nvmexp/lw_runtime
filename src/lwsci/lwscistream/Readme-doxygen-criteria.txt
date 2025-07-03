Criteria for dolwmenting private members of classes
===================================================
As per our SWAD/SWUD guidelines, when a class is exposed as a inter-unit
datatype to Tier-4 SWAD, all the private data members of the class need
not to be exported to Jama, the private data members are filtered based
on the following criteria:
    i. The members which are accessible through accessor functions.
    ii. Members which are used for the interaction with other units
        (ex. instances of TrackCount, TrackArrayCount utility classes).

All the other private data members need to be skipped while exporting
to Jama. But the DPJF tool doesn't support filtering specific members,
so the tool (our lwstomized DPJF tool) is modifed to skip the private
data members which don't have any doxygen comment. So the doxygen
style comments for such private data members should be updated as
normal C++ style comments.

For example, "state" member of SafeConnection class represents internal
state of the connection, so it need not to be exported to Tier-4 SWAD.
Instead it will be covered under "Internal Data and Structures" section
of SafeConnection SWUD.
