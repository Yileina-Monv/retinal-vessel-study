"""Read metadata.xlsx from the mirror archive without editing the workbook."""
from pathlib import Path
from zipfile import ZipFile
from io import BytesIO
from collections import Counter
import xml.etree.ElementTree as ET
import json

ROOT=Path(__file__).resolve().parent
NS={'x':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
with ZipFile(ROOT/'archives/fundus_avseg_kaggle_v1.zip') as z:
    with ZipFile(BytesIO(z.read('Fundus-AVSeg/metadata.xlsx'))) as book:
        strings=[]
        if 'xl/sharedStrings.xml' in book.namelist():
            for node in ET.fromstring(book.read('xl/sharedStrings.xml')).findall('x:si',NS):
                strings.append(''.join(t.text or '' for t in node.findall('.//x:t',NS)))
        sheet_sizes={}
        records=[]
        formula_count=0
        for name in book.namelist():
            if not (name.startswith('xl/worksheets/sheet') and name.endswith('.xml')):continue
            root=ET.fromstring(book.read(name))
            rows=[]
            for row in root.findall('x:sheetData/x:row',NS):
                values={}
                for c in row.findall('x:c',NS):
                    formula_count+=c.find('x:f',NS) is not None
                    node=c.find('x:v',NS)
                    value=node.text if node is not None else ''
                    if c.attrib.get('t')=='s':value=strings[int(value)]
                    elif c.attrib.get('t')=='inlineStr':value=''.join(t.text or '' for t in c.findall('.//x:t',NS))
                    if value:values[''.join(i for i in c.attrib['r'] if i.isalpha())]=value
                if values:rows.append(values)
            sheet_sizes[name]=len(rows)
            if rows:
                header=rows[0]
                assert header=={'A':'image name','B':'eye id','C':'disease type','D':'image quality'}
                records.extend({header[col]:r.get(col,'') for col in header} for r in rows[1:])
        assert formula_count==0
        assert len(records)==100 and len({r['image name'] for r in records})==100
        assert all(all(r.values()) for r in records)
        report={'status':'PASS_READ_ONLY_METADATA_AUDIT','row_count':len(records),'sheet_nonempty_row_counts':sheet_sizes,
                'formula_count':formula_count,'disease_counts':dict(Counter(r['disease type'] for r in records)),
                'quality_counts':dict(Counter(r['image quality'] for r in records)),
                'eye_id_values':dict(Counter(r['eye id'] for r in records)),
                'identity_limit':'eye id contains laterality, not a unique patient identifier; patient independence cannot be verified from this workbook.',
                'records':records}
(ROOT/'fundus_metadata_evidence.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:v for k,v in report.items() if k!='records'},ensure_ascii=False,indent=2))
