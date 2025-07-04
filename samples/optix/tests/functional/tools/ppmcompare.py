
import sys
import math
import os.path

def PPM_Compare( name1, name2, diffname, diff_threshold_in, allowed_percentage_in ):

    # make sure we have these as float
    diff_threshold     = float(diff_threshold_in)
    allowed_percentage = float(allowed_percentage_in)
    
    print "Comparing images \"" + name1 + "\" and \"" + name2 + "\" with a per-channel difference threshold of " + str(diff_threshold) + " and " + str(allowed_percentage) + "% of pixels allowed to differ."
    
    if not os.path.isfile( name1 ):
        print "File not found: " + name1
        return 1
    if not os.path.isfile( name2 ):
        print "File not found: " + name2
        return 1

    file1 = open( name1, "rb" )
    file2 = open( name2, "rb" )

    # Read file contents.
    cont1 = file1.read()
    cont2 = file2.read()

    # Split headers and data. We want 4 splits to get 5 parts (id,xres,yres,255,data).
    splt1 = cont1.split( None, 4 )
    splt2 = cont2.split( None, 4 )

    # If we didn't get the expected number of strings, we don't have a valid PPM file. Return failure,
    # because very likely this wasn't intended, even if both files match (a different comparison method
    # should be used in that case).
    if len(splt1) != 5 or len(splt2) != 5:
        print "At least one file is not a valid PPM file."
        return(1)

    # Compare headers.
    if splt1[0] != splt2[0] or splt1[1] != splt2[1] or splt1[2] != splt2[2] or splt1[3] != splt2[3]:
        print "File headers don't match."
        return(1)
        
    # Make extra sure the size of the binary blobs matches.
    if len(splt1[4]) != len(splt2[4]):
        print "Binary data sizes don't match."
        return(1)
    
    # Compute diff
    n = len(splt1[4]) / 3
    diffcnt = 0
    diffdata = [chr(0)] * n * 3

    for i in range(0,n):

        maxchanneldiff = float(0)

        for j in range(0,3):
            diff = float( abs( ord(splt1[4][i*3+j]) - ord(splt2[4][i*3+j]) ) )
            maxchanneldiff = max( diff, maxchanneldiff )

        if maxchanneldiff > diff_threshold:
            diffcnt+=1
            diffdata[i*3+0] = chr(255)
            diffdata[i*3+1] = chr(0)
            diffdata[i*3+2] = chr(0)
        elif maxchanneldiff > 0:
            diffdata[i*3+0] = chr(255)
            diffdata[i*3+1] = chr(255)
            diffdata[i*3+2] = chr(0)

    diff_percentage = 100.0 * float(diffcnt) / float(n)
    print str(diff_percentage) + "% of pixels exceeds diff threshold."
    
    diff_file = open( diffname, "wb" )
    diff_file.write( splt1[0] + "\n" )
    diff_file.write( splt1[1] + " " )
    diff_file.write( splt1[2] + "\n" )
    diff_file.write( splt1[3] + "\n" )
    diff_file.write( "".join(diffdata) )
    diff_file.close()

    if diff_percentage > allowed_percentage:
        print "Images considered different."
        return(1)

    print "Images considered equivalent."
    return(0)


#PPM_Compare( sys.argv[1], sys.argv[2], "diff.ppm", int(sys.argv[3]), float(sys.argv[4]) )
