"""Batch-1 inference, binary masks, snapshot and measured RAM/VRAM footprint."""
import argparse
from pathlib import Path
import platform
import threading
import time
import numpy as np
from PIL import Image, ImageDraw
import psutil
import torch
from common import get_device, load_checkpoint, write_json
from dataset import list_images, image_tensor, native_mask


class MemorySampler:
    def __init__(self):
        self.process=psutil.Process()
        self.baseline=self.process.memory_info().rss
        self.peak=self.baseline
        self.stop=threading.Event()
        self.thread=threading.Thread(target=self.loop,daemon=True)

    def sample(self):
        self.peak=max(self.peak,self.process.memory_info().rss)

    def loop(self):
        while not self.stop.wait(.01):
            self.sample()

    def __enter__(self):
        self.thread.start(); return self

    def __exit__(self,*args):
        self.sample(); self.stop.set(); self.thread.join()


def snapshot(image, mask, truth, destination):
    rgb=np.asarray(image).copy()
    overlay=rgb.copy()
    region=np.asarray(mask)>0
    overlay[region]=(overlay[region]*.5+np.array([0,255,70])*.5).astype(np.uint8)
    panels=[image,mask.convert('RGB'),Image.fromarray(overlay)]
    titles=['Original','Predicted lane mask','Prediction overlay']
    if truth is not None:
        panels.append(truth.convert('RGB')); titles.append('Ground truth: lane polygons only')
    w,h=640,360
    canvas=Image.new('RGB',(len(panels)*w,h+32),'white')
    draw=ImageDraw.Draw(canvas)
    for i,(panel,title) in enumerate(zip(panels,titles)):
        canvas.paste(panel.resize((w,h)),(i*w,32)); draw.text((i*w+10,8),title,fill='black')
    canvas.save(destination)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--images-dir',required=True)
    p.add_argument('--labels-dir',help='Optional ground truth for snapshot only')
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--output',default='run/test')
    p.add_argument('--threshold',type=float,default=.5)
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    args=p.parse_args()
    if not 0<args.threshold<1 or args.threads<1:
        p.error('threshold in (0,1), threads >= 1')
    torch.set_num_threads(args.threads)
    images=list_images(args.images_dir)
    out=Path(args.output)
    if out.exists() and any(out.iterdir()):
        raise ValueError('Inference output must be empty')
    (out/'masks').mkdir(parents=True,exist_ok=True)
    device=get_device(args.device)
    if device.type=='cuda':
        torch.cuda.reset_peak_memory_stats(device)
    times=[]
    with MemorySampler() as memory:
        model,state=load_checkpoint(args.checkpoint,device)
        config=state['config']; height,width=config['height'],config['width']
        with torch.inference_mode():
            dummy=torch.zeros(1,3,height,width,device=device)
            for _ in range(5):
                model(dummy)
            if device.type=='cuda':
                torch.cuda.synchronize(device)
            del dummy
            for i,path in enumerate(images):
                with Image.open(path) as source:
                    image=source.convert('RGB')
                tensor=image_tensor(image,height,width).unsqueeze(0).to(device)
                if device.type=='cuda':
                    torch.cuda.synchronize(device)
                start=time.perf_counter()
                logits=model(tensor)
                if device.type=='cuda':
                    torch.cuda.synchronize(device)
                times.append((time.perf_counter()-start)*1000)
                binary=(logits.sigmoid()[0,0].cpu().numpy()>args.threshold).astype(np.uint8)*255
                mask=Image.fromarray(binary).resize(image.size,Image.Resampling.NEAREST)
                mask.save(out/'masks'/(path.stem+'.png'))
                if i==0:
                    truth=native_mask(Path(args.labels_dir)/(path.stem+'.txt'),image.size) if args.labels_dir else None
                    snapshot(image,mask,truth,out/'snapshot.png')
                if (i+1)%50==0:
                    print(f'Inference {i+1}/{len(images)}',flush=True)
                memory.sample()
        os_peak=getattr(memory.process.memory_info(),'peak_wset',None)
    model_bytes=sum(x.numel()*x.element_size() for x in list(model.parameters())+list(model.buffers()))
    report={'device':str(device),'platform':platform.platform(),'torch':str(torch.__version__),
        'cpu':platform.processor(),'threads':args.threads,'height':height,'width':width,'batch_size':1,
        'checkpoint':str(Path(args.checkpoint).resolve()),'checkpoint_epoch':state['epoch'],
        'threshold':args.threshold,'num_images':len(images),'parameters':sum(p.numel() for p in model.parameters()),
        'model_mib':model_bytes/1024**2,'baseline_rss_mib':memory.baseline/1024**2,
        'sampled_peak_rss_mib':memory.peak/1024**2,'rss_increase_mib':(memory.peak-memory.baseline)/1024**2,
        'os_peak_working_set_mib':os_peak/1024**2 if os_peak is not None else None,
        'cuda_peak_allocated_mib':torch.cuda.max_memory_allocated(device)/1024**2 if device.type=='cuda' else None,
        'cuda_peak_reserved_mib':torch.cuda.max_memory_reserved(device)/1024**2 if device.type=='cuda' else None,
        'mean_forward_ms':float(np.mean(times)),'median_forward_ms':float(np.median(times)),
        'rss_scope':'model load, five warmups, all batch-1 forwards, decoding, mask export, one snapshot; baseline after imports',
        'latency_scope':'forward only after 5 warmups; excludes decode, transfer, sigmoid, resize and export',
        'rss_sampling_interval_ms':10}
    write_json(out/'memory.json',report)
    write_json(out/'predictions.json',{'images':[x.name for x in images],'checkpoint_epoch':state['epoch'],
        'threshold':args.threshold,'height':height,'width':width})
    print(report)


if __name__=='__main__':
    main()
