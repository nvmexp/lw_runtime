import os
import shutil

def process_metrics(input, output, create_empty_files):
    """ Copy file from input to output if it exists. Otherwise create an empty file if requested"""

    if os.path.exists(input):
        shutil.move(input, output)
        return

    if create_empty_files == "True":
        print 'File does not exist'
        open(output,'wb').write('') # create zero sized file
