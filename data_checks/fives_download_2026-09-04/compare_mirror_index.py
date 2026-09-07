"""Read a second public mirror's ZIP index using bounded HTTP ranges, without bulk download."""
from pathlib import Path, PurePosixPath
from zipfile import ZipFile
from urllib.parse import urlsplit
import io
import json
import requests

ROOT=Path(__file__).resolve().parent
URL='https://www.kaggle.com/api/v1/datasets/download/sushanthreddypotu/fives-dataset?datasetVersionNumber=2'

class RemoteZip(io.RawIOBase):
    def __init__(self,url):
        self.session=requests.Session()
        with self.session.get(url,stream=True,timeout=(10,30)) as r:
            r.raise_for_status()
            self.url=r.url
            self.size=int(r.headers['content-length'])
            self.etag=r.headers.get('etag')
        self.position=0;self.cache={};self.blocksize=256*1024;self.received=0
    def readable(self):return True
    def seekable(self):return True
    def tell(self):return self.position
    def seek(self,offset,whence=0):
        self.position=offset if whence==0 else self.position+offset if whence==1 else self.size+offset
        if not 0<=self.position<=self.size:raise ValueError('Seek outside archive')
        return self.position
    def read(self,size=-1):
        if size<0:size=self.size-self.position
        size=min(size,self.size-self.position)
        if size>4*1024**2:raise ValueError('Unexpected large index read')
        output=[]
        while size:
            block=self.position//self.blocksize
            start=block*self.blocksize
            if block not in self.cache:
                end=min(start+self.blocksize-1,self.size-1)
                with self.session.get(self.url,headers={'Range':f'bytes={start}-{end}'},stream=True,timeout=(10,30)) as r:
                    if r.status_code!=206:raise ValueError('Range not supported')
                    if r.headers.get('content-range')!=f'bytes {start}-{end}/{self.size}':raise ValueError('Wrong range returned')
                    if r.headers.get('etag')!=self.etag:raise ValueError('Resource changed')
                    data=r.content
                    if len(data)!=end-start+1:raise ValueError('Short range response')
                    self.cache[block]=data;self.received+=len(data)
                    if self.received>4*1024**2:raise ValueError('Index-transfer cap exceeded')
            offset=self.position-start
            n=min(size,len(self.cache[block])-offset)
            output.append(self.cache[block][offset:offset+n]);self.position+=n;size-=n
        return b''.join(output)

def normalized(name):
    p=PurePosixPath(name)
    for i,part in enumerate(p.parts):
        if part in ['train','test']:return '/'.join(p.parts[i:])
    return p.name

remote=RemoteZip(URL)
with ZipFile(remote) as z:
    other={normalized(m.filename):{'bytes':m.file_size,'crc32':f'{m.CRC:08x}'} for m in z.infolist() if not m.is_dir() and m.filename.lower().endswith('.png')}
assert len(other)==1600
report={'source':URL,'remote_archive_bytes':remote.size,'downloaded_index_bytes':remote.received,
        'signed_query_retained':False,'comparison_png_entries':len(other),'comparison_members':other}
local_path=ROOT/'archive_members.json'
if local_path.exists():
    local={normalized(r['name']):{'bytes':r['bytes'],'crc32':r['crc32']} for r in json.loads(local_path.read_text(encoding='utf-8')) if r['name'].lower().endswith('.png')}
    report['local_png_entries']=len(local)
    report['differences']=[{'name':key,'primary':local.get(key),'comparison':other.get(key)} for key in sorted(set(local)|set(other)) if local.get(key)!=other.get(key)]
    report['all_png_sizes_and_crc32_match']=not report['differences']
report['limit']='Agreement of two mirror indexes is corroboration, not cryptographic proof of identity with the official Figshare archive.'
(ROOT/'mirror_comparison_evidence.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:v for k,v in report.items() if k!='comparison_members'},ensure_ascii=False,indent=2))
