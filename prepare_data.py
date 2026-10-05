"""Convert CVAT polygon:lane to YOLO-seg and chronological 65/35 splits."""
import argparse
from collections import Counter
import hashlib
from pathlib import Path
import re
import shutil
import zipfile
import xml.etree.ElementTree as ET
from PIL import Image
import numpy as np
from common import write_json


def frame_number(name):
    match = re.search(r'frame_(\d+)', name)
    if not match:
        raise ValueError(f'Cannot determine chronological frame index: {name}')
    return int(match.group(1))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', required=True)
    p.add_argument('--output', default='data')
    p.add_argument('--train-ratio', type=float, default=.65)
    p.add_argument('--archive-cache', default='archive-cache')
    p.add_argument('--skip-missing', action='store_true', help='Explicitly audit and exclude missing images')
    args = p.parse_args()
    if not 0 < args.train_ratio < 1:
        p.error('train-ratio must be in (0,1)')
    source, out = Path(args.source), Path(args.output)
    if out.exists() and any(out.iterdir()):
        raise ValueError('Prepared output must be empty')
    records, ignored, missing = [], Counter(), []
    for group in ['Test','Train']:
        xmls = sorted((source/group).rglob('*.xml'))
        if not xmls:
            raise FileNotFoundError(f'No CVAT XML in {source/group}')
        for xml in xmls:
            images = ET.parse(xml).getroot().findall('image')
            if not images:
                raise ValueError(f'{xml}: only CVAT image annotation format supported')
            index = {}
            for path in xml.parent.rglob('*'):
                if path.suffix.lower() in {'.jpg','.jpeg','.png'}:
                    index.setdefault(path.name,[]).append(path)
            for archive in xml.parent.rglob('*.zip'):
                cache=Path(args.archive_cache)/hashlib.sha256(str(archive).encode()).hexdigest()[:12]
                cache.mkdir(parents=True,exist_ok=True)
                with zipfile.ZipFile(archive) as zipped:
                    for member in zipped.infolist():
                        name=Path(member.filename).name
                        if Path(name).suffix.lower() not in {'.jpg','.jpeg','.png'}:
                            continue
                        destination=cache/name
                        if not destination.exists():
                            with zipped.open(member) as src, destination.open('wb') as dst:
                                shutil.copyfileobj(src,dst)
                        index.setdefault(name,[]).append(destination)
            for item in images:
                name = Path(item.attrib['name']).name
                matches = index.get(name,[])
                if not matches and args.skip_missing:
                    missing.append({'name':name,'annotation':str(xml),'group':group})
                    continue
                if len(matches)!=1:
                    raise ValueError(f'{xml}: expected one source image for {name}, found {len(matches)}')
                image = matches[0]
                with Image.open(image) as im:
                    w,h = im.size
                if (w,h)!=(int(item.attrib['width']),int(item.attrib['height'])):
                    raise ValueError(f'{image}: XML size mismatch')
                lines, clipped = [], 0
                for shape in item:
                    if shape.tag != 'polygon' or shape.get('label') != 'lane':
                        ignored[f'{shape.tag}:{shape.get("label")}'] += 1
                        continue
                    coords = np.array([[float(v) for v in point.split(',')] for point in shape.attrib['points'].split(';')])
                    if coords.ndim!=2 or coords.shape[1]!=2 or len(coords)<3 or not np.isfinite(coords).all():
                        raise ValueError(f'{xml}:{name}: invalid lane polygon')
                    normalized = coords / np.array([w,h])
                    clipped += int(((normalized<0)|(normalized>1)).any())
                    normalized = np.clip(normalized,0,1)
                    lines.append('0 '+' '.join(f'{v:.9f}' for v in normalized.ravel()))
                records.append({'group':group,'frame':frame_number(name),'name':name,
                    'source':image,'annotation':xml,'lines':lines,'clipped':clipped})
    manifest, hashes, names, duplicates, unique = [], {}, set(), [], []
    for r in records:
        digest = hashlib.sha256(r['source'].read_bytes()).hexdigest()
        if digest in hashes:
            duplicates.append({'name':r['name'],'same_as':hashes[digest],'group':r['group']})
            continue
        hashes[digest]=r['name']
        r['sha256']=digest
        unique.append(r)
    unique.sort(key=lambda r:(r['frame'],r['name']))
    train_count=int(len(unique)*args.train_ratio)
    if not 0 < train_count < len(unique):
        raise ValueError('Not enough unique images for train/test')
    for index,r in enumerate(unique):
        split='train' if index < train_count else 'test'
        key=r['name']
        if key in names:
            raise ValueError(f'Duplicate name with different bytes: {key}')
        names.add(key)
        row = {'split':split,'frame':r['frame'],'name':r['name'],'source':str(r['source']),
               'annotation':str(r['annotation']),'sha256':r['sha256'],'lane_polygons':len(r['lines']),
               'clipped_polygons':r['clipped']}
        manifest.append(row)
        image_dir,label_dir = out/'images'/split,out/'labels'/split
        image_dir.mkdir(parents=True,exist_ok=True); label_dir.mkdir(parents=True,exist_ok=True)
        shutil.copy2(r['source'],image_dir/r['name'])
        (label_dir/(Path(r['name']).stem+'.txt')).write_text('\n'.join(r['lines'])+'\n',encoding='utf-8')
    summary = {'counts':dict(Counter(r['split'] for r in manifest)),
               'ignored_shapes':dict(ignored),'train_ratio_requested':args.train_ratio,
               'train_ratio_actual':train_count/len(unique),'test_start_frame':unique[train_count]['frame'],
               'split_method':'chronological across all available unique images; original Train/Test regrouped',
               'clipped_polygons':sum(r['clipped_polygons'] for r in manifest),
               'selected_label':'polygon:lane','source':str(source.resolve()),
               'missing_images':len(missing),'excluded_duplicates':len(duplicates)}
    write_json(out/'manifest.json',{'summary':summary,'images':manifest,'missing':missing,'duplicates':duplicates})
    print(summary)


if __name__=='__main__':
    main()
