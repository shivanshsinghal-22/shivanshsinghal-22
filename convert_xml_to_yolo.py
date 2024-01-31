import os
import xml.etree.ElementTree as ET

# The following fucntion converts XML annotations to YOLO format( .txt )
# xml_path : gives the absolute path to the XML files that needs to be modified
# yolo_path : gives the path of the folder to save the modified( new ) annotation file
# XML uses class Names instead of class id's. Thus we need to define class Id's 

def convert_xml_to_yolo( xml_path , yolo_path , class_dict ):
    
    """ The below lines are used to access XML files in organised ways. 
        The parse function along with the XMl file path returns an Elementree that contains 
        an organised form of XML file. 

        The root of the XML file tells the top level hierarchy. Here it is annotation
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    image_width = int(root.find('size/width').text)
    image_height = int(root.find('size/height').text)

    yolo_file_path = os.path.join(yolo_path,os.path.splitext(os.path.basename(xml_path))[0] + '.txt')

    # We have created a .txt file with the same name as .Xml file

    with open(yolo_file_path,'w') as yolo_file:
        for obj in root.findall('object'):
            class_name = obj.find('name').text
            class_id = class_dict.get(class_name,-1)

            if class_id==-1:
                print(f"Warning no such class found")
                continue

            bbox = obj.find('bndbox')
            xmin = float(bbox.find('xmin').text)
            xmax = float(bbox.find('xmax').text)
            ymin = float(bbox.find('ymin').text)
            ymax = float(bbox.find('ymax').text)

            x_center = (xmin + xmax)/(2.0 * image_width)
            y_center = (ymin + ymax)/(2.0 * image_height)
            width = (xmax - xmin) / image_width
            height = (ymax - ymin) / image_height

            yolo_file.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

xml_folder = r'C:\Users\singh\Desktop\Project\data\annotations'
yolo_folder = r'C:\Users\singh\Desktop\Project\data\labels'
class_dict = {'with_mask':0 , 'without_mask':1 , 'mask_weared_incorrect':2}

for xml_file in os.listdir(xml_folder):
    if xml_file.endswith('.xml'):
        xml_file_path = os.path.join(xml_folder,xml_file)
        convert_xml_to_yolo(xml_file_path,yolo_folder,class_dict)