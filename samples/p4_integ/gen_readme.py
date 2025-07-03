import os
from pathlib import Path
import xml.etree.ElementTree as ET

sample_dirs = os.listdir('./Samples')
sample_dirs = sorted(sample_dirs, key=str.casefold)

for d in sample_dirs:
    sample_dir = './Samples/'+str(d)
    readme_file = open(sample_dir + '/README.md', 'w')
    readme_file.write('# ' + d.replace('_', '. ', 1).replace('_', ' '))
    readme_file.write('\n\n\n')

    print('### [' + d.replace('_', '. ', 1).replace('_', ' ') + '](./' + str(d) + ')')

    pathlist = os.listdir(sample_dir)
    pathlist = sorted(pathlist, key=str.casefold)

    for sample_name in pathlist:
        if Path(sample_dir + '/' + sample_name).is_dir():
            root = ET.parse(sample_dir + '/' + sample_name + '/info.xml').getroot()
            desc = root.find('description').text

            readme_file.write('### [' + str(sample_name) + '](./' + str(sample_name) + ')')
            readme_file.write('\n')
            readme_file.write(desc)
            readme_file.write('\n\n')

    readme_file.close()
