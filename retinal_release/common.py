import hashlib
import json
import os
from pathlib import Path
import platform
import uuid

ROOT=Path(__file__).resolve().parents[1]

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b''): h.update(chunk)
    return h.hexdigest()

def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode('utf-8')

def identity(value): return hashlib.sha256(canonical(value)).hexdigest()

def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))

def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    with temp.open('wb') as f: f.write(canonical(value));f.flush();os.fsync(f.fileno())
    os.replace(temp,path)

def safe(root,relative):
    # Portable paths are POSIX relative paths; reject drive names, traversal and aliases.
    if not isinstance(relative,str) or '\\' in relative or ':' in relative: raise ValueError('Nonportable path')
    p=Path(relative)
    if p.is_absolute() or not relative or any(x in ('','..','.') for x in relative.split('/')): raise ValueError('Unsafe path')
    target=(Path(root)/p).resolve()
    if not target.is_relative_to(Path(root).resolve()): raise ValueError('Path escapes root')
    return target

def machine():
    value=platform.node()
    if os.name=='nt':
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,r'SOFTWARE\Microsoft\Cryptography') as k:
            value+='|'+winreg.QueryValueEx(k,'MachineGuid')[0]
    return identity(value)
