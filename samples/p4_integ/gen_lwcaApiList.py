import os
from pathlib import Path
import xml.etree.ElementTree as ET

sample_dirs = os.listdir('./Samples')
sample_dirs = sorted(sample_dirs, key=str.casefold)

for d in sample_dirs:
    sample_dir = './Samples/'+str(d)

    pathlist = os.listdir(sample_dir)
    pathlist = sorted(pathlist, key=str.casefold)

    for sample_name in pathlist:
        if Path(sample_dir + '/' + sample_name).is_dir():
            print(sample_dir + '/' + sample_name + '/info.xml')
            tree = ET.parse(sample_dir + '/' + sample_name + '/info.xml')
            root = tree.getroot()

            desc = root.find('description')
            desc.text = '<![CDATA[' + desc.text + ']]>'

            entry = root.find('lwda_api_list')

            if (entry):
                entry.clear()
            else:
                entry = ET.SubElement(root, 'lwda_api_list')

            # o = only matching
            # h - no file name
            os.system('cd ' + sample_dir + '/' + sample_name + ' && grep -oshR \"lwca[a-zA-Z]\+[(]\" *.cpp > ./'+sample_name+'toolkit.txt')
            os.system('cd ' + sample_dir + '/' + sample_name + ' && grep -oshR \"lwca[a-zA-Z]\+[(]\" *.lw >> ./'+sample_name+'toolkit.txt')
            os.system('cd ' + sample_dir + '/' + sample_name + ' && grep -oshR \"lwca[a-zA-Z]\+[(]\" src/*.cpp >> ./'+sample_name+'toolkit.txt')
            os.system('cd ' + sample_dir + '/' + sample_name + ' && grep -oshR \"lwca[a-zA-Z]\+[(]\" src/*.lw >> ./'+sample_name+'toolkit.txt')

            toolkitAPIFile = open(sample_dir + '/' + sample_name+'/'+sample_name+'toolkit.txt', 'r')
            toolkitAPIs = toolkitAPIFile.read().replace('(', '').split()
            toolkitAPIs = sorted(toolkitAPIs, key=str.casefold) 
            toolkitAPIs = set(toolkitAPIs)
            for toolkitAPI in toolkitAPIs:
                toolkitNode = ET.SubElement(entry, 'toolkit')
                toolkitNode.text = toolkitAPI
            
            toolkitAPIFile.close()
            os.system('rm ' + sample_dir + '/' + sample_name +'/' + sample_name + 'toolkit.txt')

            os.system('cd ' + sample_dir + '/' + sample_name + ' && grep -oshR \"lw[A-Z][a-zA-Z]\+[(]\" *.cpp > ./'+sample_name+'driver.txt')
            os.system('cd ' + sample_dir + '/' + sample_name + ' && grep -oshR \"lw[A-Z][a-zA-Z]\+[(]\" *.lw >> ./'+sample_name+'driver.txt')
            os.system('cd ' + sample_dir + '/' + sample_name + ' && grep -oshR \"lw[A-Z][a-zA-Z]\+[(]\" src/*.cpp >> ./'+sample_name+'driver.txt')
            os.system('cd ' + sample_dir + '/' + sample_name + ' && grep -oshR \"lw[A-Z][a-zA-Z]\+[(]\" src/*.lw >> ./'+sample_name+'driver.txt')

            driverAPIFile = open(sample_dir + '/' + sample_name+'/'+sample_name+'driver.txt', 'r')
            driverAPIs = driverAPIFile.read().replace('(', '').split()
            driverAPIs = sorted(driverAPIs, key=str.casefold)
            driverAPIs = set(driverAPIs)
            for driverAPI in driverAPIs:
                driverNode = ET.SubElement(entry, 'driver')
                driverNode.text = driverAPI

            driverAPIFile.close()
            os.system('rm ' + sample_dir + '/' + sample_name +'/' + sample_name + 'driver.txt')


            ET.indent(tree, space="    ", level=0)
            tree.write(sample_dir + '/' + sample_name + '/info.xml', encoding="UTF-8")

            xmlFile = open(sample_dir + '/' + sample_name + '/info.xml', 'r')
            xmlData = xmlFile.read().replace('&lt;', '<')
            xmlData = xmlData.replace('&gt;', '>')

            xmlFile.close()
            xmlFile = open(sample_dir + '/' + sample_name + '/info.xml', 'w')
            xmlFile.seek(0)
            xmlFile.write("""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE entry SYSTEM "SamplesInfo.dtd">
""")
            xmlFile.write(xmlData)
            xmlFile.write('\n')
            xmlFile.close()


