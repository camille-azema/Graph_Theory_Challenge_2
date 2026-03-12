import zipfile
import xml.etree.ElementTree as ET
import sys
import io

def read_docx(path):
    try:
        document = zipfile.ZipFile(path)
        xml_content = document.read('word/document.xml')
        tree = ET.XML(xml_content)
        PARA = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}p'
        TEXT = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t'
        paragraphs = []
        for paragraph in tree.iter(PARA):
            texts = [node.text for node in paragraph.iter(TEXT) if node.text]
            if texts:
                paragraphs.append(''.join(texts))
        return '\n'.join(paragraphs)
    except Exception as e:
        return str(e)

content = read_docx(sys.argv[1])
with open(sys.argv[2], 'w', encoding='utf-8') as f:
    f.write(content)
