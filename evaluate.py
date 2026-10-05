"""Pixel metrics at network resolution; missing predictions are errors."""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
from dataset import list_images,native_mask
from common import write_json


def counts(pred,truth):
    pred,truth=pred>0,truth>0
    tp=int(np.count_nonzero(pred & truth))
    fp=int(np.count_nonzero(pred & ~truth))
    fn=int(np.count_nonzero(~pred & truth))
    tn=int(np.count_nonzero(~pred & ~truth))
    return tp,fp,fn,tn


def ratio(n,d,empty=0.):
    return n/d if d else empty


def is_detected(iou,positive_ground_truth,threshold=.6):
    return bool(positive_ground_truth and iou>threshold)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--images-dir',required=True)
    p.add_argument('--labels-dir',required=True)
    p.add_argument('--pred-masks-dir',required=True)
    p.add_argument('--iou-threshold',type=float,default=.6)
    p.add_argument('--height',type=int,default=36)
    p.add_argument('--width',type=int,default=64)
    p.add_argument('--output',default='run/test64x36/metrics.json')
    args=p.parse_args()
    if not 0<=args.iou_threshold<=1:
        p.error('IoU threshold in [0,1]')
    if min(args.height,args.width)<1:
        p.error('height and width must be positive')
    images=list_images(args.images_dir)
    totals=np.zeros(4,dtype=np.int64); rows=[]
    for path in images:
        with Image.open(path) as image:
            size=image.size
        truth=np.asarray(native_mask(Path(args.labels_dir)/(path.stem+'.txt'),size).resize(
            (args.width,args.height),Image.Resampling.NEAREST))
        predicted=Path(args.pred_masks_dir)/(path.stem+'.png')
        with Image.open(predicted) as mask:
            if mask.size!=(args.width,args.height) or mask.mode!='L':
                raise ValueError(f'{predicted}: expected {args.width}x{args.height} grayscale binary mask')
            pred=np.asarray(mask)
        if not np.isin(pred,[0,255]).all():
            raise ValueError(f'{predicted}: mask must contain only 0 and 255')
        tp,fp,fn,tn=counts(pred,truth); totals+=np.array([tp,fp,fn,tn])
        iou=ratio(tp,tp+fp+fn,1.)
        positive=bool((truth>0).any())
        rows.append({'image':path.name,'iou':iou,'positive_ground_truth':positive,
            'detected':is_detected(iou,positive,args.iou_threshold),'tp':tp,'fp':fp,'fn':fn,'tn':tn})
    tp,fp,fn,tn=[int(x) for x in totals]
    detected=[r for r in rows if r['detected']]
    positives=sum(r['positive_ground_truth'] for r in rows)
    summary={'num_images':len(rows),'num_positive_images':positives,'num_detected':len(detected),
        'detection_rate':ratio(len(detected),positives),'iou_threshold':args.iou_threshold,
        'avg_iou_detected':float(np.mean([r['iou'] for r in detected])) if detected else None,
        'mean_image_iou':float(np.mean([r['iou'] for r in rows])),
        'global_iou':ratio(tp,tp+fp+fn,1.),'dice':ratio(2*tp,2*tp+fp+fn,1.),
        'precision':ratio(tp,tp+fp),'recall':ratio(tp,tp+fn),'pixel_accuracy':ratio(tp+tn,tp+fp+fn+tn),
        'tp':tp,'fp':fp,'fn':fn,'tn':tn,'resolution':f'{args.width}x{args.height} network mask',
        'empty_policy':'Both masks empty: IoU 1, excluded from positive-image detection rate'}
    write_json(args.output,{'summary':summary,'per_image':rows})
    print(summary)


if __name__=='__main__':
    main()
