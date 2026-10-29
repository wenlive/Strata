"""Bounded HTTP Range fallback for immutable official Qwen checkpoint files.

Resume segment files, validate Content-Range, stream-assemble, verify full LFS
SHA256 and atomically publish. Never load a whole tensor or file into memory.
"""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import time
import urllib.request


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024**2),b''):h.update(block)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--destination',type=Path,required=True);p.add_argument('--files',nargs='+',required=True)
    p.add_argument('--workers',type=int,default=8);p.add_argument('--segment-mib',type=int,default=128)
    a=p.parse_args();assert 1<=a.workers<=8 and 8<=a.segment_mib<=512
    source=json.loads((a.destination/'source-manifest.json').read_text());revision=source['sha']
    assert revision=='995ad96eacd98c81ed38be0c5b274b04031597b0'
    api={x['rfilename']:x for x in source['siblings']};selected=[];results=[];jobs=[];segment=a.segment_mib*1024**2
    staging=a.destination/'.incomplete'/'ranges';staging.mkdir(parents=True,exist_ok=True)
    for name in dict.fromkeys(a.files):
        assert name in api and Path(name).name==name and name.endswith('.safetensors')
        info=api[name];sha=info['lfs']['sha256'];target=a.destination/name
        if target.exists() and target.stat().st_size==info['size'] and digest(target)==sha:
            results.append({'file':name,'sha256':sha,'already_verified':True});continue
        selected.append(info)
        for start in range(0,info['size'],segment):
            size=min(segment,info['size']-start);part=staging/(sha+f'.{start:012d}.part')
            jobs.append((name,start,size,part))
    required=2*sum(x['size'] for x in selected)+200*1024**3
    if shutil.disk_usage(a.destination).free<required:raise OSError('Insufficient staging and system reserve space')

    def fetch(job):
        name,start,size,part=job
        for attempt in range(6):
            have=part.stat().st_size if part.exists() else 0
            if have>size:raise ValueError(f'Invalid partial segment size: {name} at {start}')
            if have==size:return
            at=start+have;end=start+size-1
            url=f'https://huggingface.co/Qwen/Qwen3.6-35B-A3B/resolve/{revision}/{name}?download=true&strata_range={at}'
            try:
                request=urllib.request.Request(url,headers={'Range':f'bytes={at}-{end}'})
                with urllib.request.urlopen(request,timeout=60) as response:
                    header=response.headers.get('Content-Range','');match=re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)',header)
                    if response.status!=206 or not match or tuple(map(int,match.groups()))!=(at,end,api[name]['size']):raise ValueError(f'Server returned wrong byte range: {name} at {at}')
                    with part.open('ab') as out:
                        remaining=size-have
                        while remaining:
                            block=response.read(min(1024**2,remaining))
                            if not block:raise OSError('Short segment response')
                            out.write(block);remaining-=len(block)
                print(f'segment verified-range {name} {start}:{start+size}',flush=True)
                return
            except (OSError,TimeoutError) as error:
                print(f'retry {name} range={at} attempt={attempt+1} type={type(error).__name__}',flush=True)
                if attempt==5:raise
                time.sleep(min(2**attempt,20))

    with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:list(pool.map(fetch,jobs))
    for info in selected:
        name=info['rfilename'];sha=info['lfs']['sha256'];target=a.destination/name
        if target.exists() and target.stat().st_size==info['size'] and digest(target)==sha:
            results.append({'file':name,'sha256':sha,'already_verified':True});continue
        temp=staging/(name+'.assembled.incomplete');h=hashlib.sha256();written=0
        with temp.open('wb') as out:
            for start in range(0,info['size'],segment):
                part=staging/(sha+f'.{start:012d}.part');assert part.stat().st_size==min(segment,info['size']-start)
                with part.open('rb') as f:
                    for block in iter(lambda:f.read(4*1024**2),b''):out.write(block);h.update(block);written+=len(block)
            out.flush();os.fsync(out.fileno())
        if written!=info['size'] or h.hexdigest()!=sha:raise ValueError(f'Full file SHA256/size mismatch; not published: {name}')
        os.replace(temp,target);results.append({'file':name,'sha256':sha,'bytes':written,'assembled_from_ranges':True})
        print(f'published-sha256 {name} {written} {sha}',flush=True)
    marker=a.destination/'range-subset-verified.json';temp=marker.with_suffix('.incomplete')
    temp.write_text(json.dumps({'revision':revision,'complete_checkpoint':False,'scope':'subset-only','files':results},indent=2));os.replace(temp,marker)
    print('SUBSET VERIFIED; full checkpoint still requires index verification',flush=True)


if __name__=='__main__':main()
