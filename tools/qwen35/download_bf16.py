"""Bounded-memory, resumable download with SHA256 validation of official BF16 files."""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import shutil
import time
import urllib.request


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--destination', type=Path, required=True)
    p.add_argument('--revision', default='995ad96eacd98c81ed38be0c5b274b04031597b0')
    p.add_argument('--workers', type=int, default=3)
    p.add_argument('--metadata-only',action='store_true',help='Fetch configuration/tokenizer/source files only; never mark checkpoint complete')
    p.add_argument('--only-files',nargs='+',help='Fetch an explicit subset of official filenames; write a subset manifest, never mark full checkpoint complete')
    p.add_argument('--backend', choices=('http','hf'), default='http',
                   help='http: bounded 1 MiB streaming/resume; hf: optional HF/Xet transport with the same SHA256 checks')
    a = p.parse_args()
    if not 1 <= a.workers <= 8:
        p.error('--workers must be between 1 and 8')
    a.destination.mkdir(parents=True, exist_ok=True)
    repo = 'Qwen/Qwen3.6-35B-A3B'
    api = json.load(urllib.request.urlopen(f'https://huggingface.co/api/models/{repo}/revision/{a.revision}?blobs=true',timeout=90))
    if api['sha'] != a.revision:
        raise ValueError('Use an immutable full revision SHA')
    (a.destination/'source-manifest.json').write_text(json.dumps(api,indent=2),encoding='utf-8')
    files = api['siblings']
    if a.metadata_only:files=[x for x in files if not x['rfilename'].endswith('.safetensors')]
    if a.only_files:
        selected=set(a.only_files)
        if a.metadata_only or not selected.issubset({x['rfilename'] for x in files}):p.error('Unknown subset filename or incompatible --metadata-only')
        files=[x for x in files if x['rfilename'] in selected]
    need = sum(max(0, x['size'] - ((a.destination/x['rfilename']).stat().st_size if (a.destination/x['rfilename']).exists() else 0)) for x in files)
    free = shutil.disk_usage(a.destination).free
    print(json.dumps({'revision':a.revision,'download_bytes':need,'free_bytes':free,'reserve_bytes':200*1024**3}),flush=True)
    if free < need + 200*1024**3:
        raise OSError('Insufficient disk space including 200 GiB system reserve')

    def digest(path):
        h = hashlib.sha256()
        with path.open('rb') as f:
            for block in iter(lambda:f.read(1024**2), b''):
                h.update(block)
        return h.hexdigest()

    def fetch(x):
        name, size = x['rfilename'], x['size']
        out = a.destination/name
        sha = (x.get('lfs') or {}).get('sha256')
        if out.exists() and out.stat().st_size == size and (not sha or digest(out)==sha):
            return {'file':name,'status':'verified-existing','bytes':size,'sha256':sha}
        if a.backend=='hf':
            from huggingface_hub import hf_hub_download
            started=time.perf_counter()
            resolved=Path(hf_hub_download(repo,filename=name,revision=a.revision,local_dir=a.destination))
            actual=digest(resolved)
            if resolved.stat().st_size!=size or (sha and actual!=sha):
                raise ValueError(f'HF transport size/SHA256 mismatch: {name}')
            print(f'verified-hf {name} {size} seconds={time.perf_counter()-started:.3f} {actual}',flush=True)
            return {'file':name,'status':'hf-downloaded-verified','bytes':size,'sha256':actual}
        part = out.with_name(out.name+'.incomplete')
        out.parent.mkdir(parents=True,exist_ok=True)
        for attempt in range(6):
            try:
                offset = part.stat().st_size if part.exists() else 0
                if offset > size:
                    raise ValueError(f'Oversized partial {part}')
                if offset != size:
                    headers = {'Accept-Encoding':'identity'}
                    if offset:
                        headers['Range'] = f'bytes={offset}-'
                    url = f'https://huggingface.co/{repo}/resolve/{a.revision}/{name}?download=true&resume={offset}'
                    with urllib.request.urlopen(urllib.request.Request(url,headers=headers),timeout=120) as r:
                        if offset and r.status != 206:
                            raise OSError(f'Server refused resume at {offset}; status={r.status}')
                        if offset and not r.headers.get('Content-Range','').startswith(f'bytes {offset}-'):
                            raise OSError('Content-Range mismatch')
                        print(f'download {name} from {offset}/{size}',flush=True)
                        last = time.monotonic()
                        with part.open('ab' if offset else 'wb') as f:
                            while block := r.read(1024**2):
                                if f.tell()+len(block)>size:
                                    raise OSError('Response exceeds declared file length')
                                f.write(block)
                                if time.monotonic()-last>30:
                                    print(f'progress {name} {f.tell()}/{size}',flush=True)
                                    last=time.monotonic()
                if part.stat().st_size != size:
                    raise OSError(f'Truncated download {name}')
                actual=digest(part)
                if sha and actual!=sha:
                    raise ValueError(f'SHA256 mismatch {name}: {actual} != {sha}')
                part.replace(out)
                print(f'verified {name} {size} {actual}',flush=True)
                return {'file':name,'status':'downloaded-verified','bytes':size,'sha256':actual}
            except (OSError, TimeoutError) as e:
                print(f'retry {name} attempt={attempt+1}: {e!r}',flush=True)
                if attempt==5: raise
                time.sleep(min(2**attempt,20))

    with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:
        result=list(pool.map(fetch,files))
    partial=bool(a.metadata_only or a.only_files)
    manifest='metadata-verified.json' if a.metadata_only else 'subset-verified.json' if a.only_files else 'download-verified.json'
    tmp=a.destination/(manifest+'.incomplete')
    tmp.write_text(json.dumps({'revision':a.revision,'complete_checkpoint':not partial,'files':result},indent=2),encoding='utf-8')
    tmp.replace(a.destination/manifest)
    print('SUBSET/METADATA VERIFIED (not a full checkpoint)' if partial else 'ALL FILES VERIFIED',flush=True)


if __name__=='__main__':
    main()
