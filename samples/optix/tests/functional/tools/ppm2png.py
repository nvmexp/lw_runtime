
import png
import os.path
import shutil

def PPM_To_PNG( ppmname, pngname, defaultpng ):
	
	print "Colwerting " + ppmname + " to " + pngname
	
	if not os.path.isfile( ppmname ):
		print "File not found: " + ppmname + ". Outputting default placeholder PNG."
		shutil.copyfile( defaultpng, pngname )
		return 1
	
	ppmfile = open( ppmname, "rb" )
	pngfile = open( pngname, "wb" )
	
	strdata = ppmfile.read()
	
	# Split headers and data. We want 4 splits to get 5 parts (id,xres,yres,255,data).
	splt = strdata.split( None, 4 )
	
	if len(splt) != 5:
		print "Input is not a valid PPM file."
		return 1
	
	data = map( ord, splt[4] )
	
	writer = png.Writer(width=int(splt[1]),height=int(splt[2]),bitdepth=8,greyscale=False,alpha=False,compression=9)
	writer.write_array( pngfile, data )

	return 0