"""Read and reconcile two public copies of the FIVES quality workbook; no edits."""
from pathlib import Path
from zipfile import ZipFile
from collections import Counter
import xml.etree.ElementTree as ET
import hashlib
import json

ROOT=Path(__file__).resolve().parent
NS={'x':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}

def read_book(path):
    records=[]
    with ZipFile(path) as z:
        strings=[]
        if 'xl/sharedStrings.xml' in z.namelist():
            strings=[''.join(t.text or '' for t in n.findall('.//x:t',NS)) for n in ET.fromstring(z.read('xl/sharedStrings.xml')).findall('x:si',NS)]
        wb=ET.fromstring(z.read('xl/workbook.xml'))
        rels={n.attrib['Id']:n.attrib['Target'] for n in ET.fromstring(z.read('xl/_rels/workbook.xml.rels'))}
        for sheet in wb.findall('x:sheets/x:sheet',NS):
            split=sheet.attrib['name'].lower()
            assert split in {'train','test'}
            target=rels[sheet.attrib['{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id']]
            xml_path=target.lstrip('/') if target.startswith('/') else 'xl/'+target
            cells=[]
            for row in ET.fromstring(z.read(xml_path)).findall('x:sheetData/x:row',NS):
                values={}
                for c in row.findall('x:c',NS):
                    assert c.find('x:f',NS) is None,'Unexpected formula in quality data'
                    v=c.find('x:v',NS);value=v.text if v is not None else ''
                    if c.attrib.get('t')=='s':value=strings[int(value)]
                    elif c.attrib.get('t')=='inlineStr':value=''.join(t.text or '' for t in c.findall('.//x:t',NS))
                    if value:values[''.join(x for x in c.attrib['r'] if x.isalpha())]=value
                if values:cells.append(values)
            assert cells[0]=={'A':'Disease','B':'Number','C':'IC','D':'Blur','E':'LC'}
            assert len(cells)-1=={'train':600,'test':200}[split]
            for row in cells[1:]:
                assert set(row)==set('ABCDE')
                disease=row['A'];num=int(row['B'])
                assert disease in 'ADGN'
                flags={k:int(row[col]) for col,k in [('C','IC'),('D','Blur'),('E','LC')]}
                assert set(flags.values())<={0,1}
                records.append({'key':f'{split}/{num}_{disease}.png','split':split,'filename':f'{num}_{disease}.png','disease':disease,**flags})
    assert len(records)==len({r['key'] for r in records})==800
    return records

primary=read_book(ROOT/'archives/quality_primary.xlsx')
comparison=read_book(ROOT/'archives/quality_comparison.xlsx')
p={r['key']:r for r in primary};c={r['key']:r for r in comparison}
differences=[{'key':key,'primary':p.get(key),'comparison':c.get(key)} for key in sorted(set(p)|set(c)) if p.get(key)!=c.get(key)]
groups=[]
for split in ['train','test','all']:
    for disease in ['A','D','G','N','all']:
        rows=[r for r in primary if (split=='all' or r['split']==split) and (disease=='all' or r['disease']==disease)]
        groups.append({'split':split,'disease':disease,'count':len(rows),
                       'IC_good':sum(r['IC'] for r in rows),'Blur_good':sum(r['Blur'] for r in rows),'LC_good':sum(r['LC'] for r in rows),
                       'all_three_good':sum(all(r[k]==1 for k in ['IC','Blur','LC']) for r in rows),
                       'quality_patterns_IC_Blur_LC':dict(Counter(''.join(str(r[k]) for k in ['IC','Blur','LC']) for r in rows))})
report={'status':'PASS_800_QUALITY_ROWS','row_count':800,'fields':['Disease','Number','IC','Blur','LC'],
        'coding':'Per original paper: 1 = good quality in that aspect; 0 = poor quality. IC = illumination/color distortion; LC = low contrast.',
        'paper_source':'https://pmc.ncbi.nlm.nih.gov/articles/PMC9352679/',
        'primary_mirror':{'ref':'nikitamanaenkov/fundus-image-dataset-for-vessel-segmentation','version':6},
        'comparison_mirror':{'ref':'sushanthreddypotu/fives-dataset','version':2},
        'workbook_files_byte_identical':(ROOT/'archives/quality_primary.xlsx').read_bytes()==(ROOT/'archives/quality_comparison.xlsx').read_bytes(),
        'all_decoded_quality_records_identical':not differences,'quality_record_differences':differences,
        'identity_fields_present':False,'identity_limit':'The workbook provides image numbers and disease/quality fields, not patient IDs or eye laterality.',
        'group_summaries':groups,'records':primary}
(ROOT/'quality_evidence.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:v for k,v in report.items() if k not in ['records','group_summaries']},ensure_ascii=False,indent=2))
print(json.dumps([g for g in groups if g['disease']=='all'],ensure_ascii=False,indent=2))
