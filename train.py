"""Train every layer from scratch using explicit train and validation splits."""
import argparse
import csv
import time
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from dataset import LaneDataset
from model import LaneNet
from common import seed_everything, get_device, loss_function, write_json


def run_epoch(model, loader, device, optimizer=None):
    model.train(optimizer is not None)
    total, count, intersection, union = 0., 0, 0, 0
    with torch.enable_grad() if optimizer is not None else torch.inference_mode():
        for images, truth in loader:
            images, truth = images.to(device), truth.to(device)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = loss_function(logits, truth)
            if optimizer is not None:
                loss.backward()
                optimizer.step()
            total += loss.item()*images.size(0)
            count += images.size(0)
            pred, gt = logits.sigmoid()>.5, truth>.5
            intersection += (pred & gt).sum().item()
            union += (pred | gt).sum().item()
    return total/count, intersection/union if union else 1.


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', required=True)
    p.add_argument('--output', default='runs/lane')
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--height', type=int, default=96)
    p.add_argument('--width', type=int, default=160)
    p.add_argument('--base', type=int, default=6)
    p.add_argument('--lr', type=float, default=.001)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    args = p.parse_args()
    if min(args.height, args.width)<8 or min(args.epochs, args.batch_size, args.threads)<1 or args.lr<=0:
        p.error('invalid size, epoch, batch, thread or learning rate')
    torch.set_num_threads(args.threads)
    seed_everything(args.seed)
    device = get_device(args.device)
    root, out = Path(args.data_root), Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise ValueError('Output must be empty; choose a fresh run directory')
    datasets = [LaneDataset(root/'images'/split, root/'labels'/split, args.height, args.width, split=='train')
                for split in ['train', 'val']]
    if {p.name for p in datasets[0].images} & {p.name for p in datasets[1].images}:
        raise ValueError('Duplicate train/val filenames')
    write_json(out/'split_manifest.json', {s:[str(x.resolve()) for x in ds.images] for s,ds in zip(['train','val'],datasets)})
    loaders = [DataLoader(ds, batch_size=args.batch_size, shuffle=i==0, num_workers=0,
                         generator=torch.Generator().manual_seed(args.seed+i)) for i,ds in enumerate(datasets)]
    model = LaneNet(args.base).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    write_json(out/'config.json', vars(args))
    write_json(out/'environment.json', {'torch':str(torch.__version__), 'device':str(device),
        'device_name':torch.cuda.get_device_name() if device.type=='cuda' else 'CPU',
        'parameters':sum(p.numel() for p in model.parameters())})
    best, start = float('inf'), time.perf_counter()
    with SummaryWriter(str(out/'tensorboard')) as writer, (out/'history.csv').open('w',newline='') as f:
        history = csv.writer(f)
        history.writerow(['epoch','train_loss','val_loss','train_iou','val_iou','elapsed_seconds'])
        for epoch in range(1,args.epochs+1):
            tl,ti = run_epoch(model,loaders[0],device,optimizer)
            vl,vi = run_epoch(model,loaders[1],device)
            elapsed = time.perf_counter()-start
            history.writerow([epoch,tl,vl,ti,vi,elapsed]); f.flush()
            for key,value in [('loss/train',tl),('loss/val',vl),('iou/train',ti),('iou/val',vi)]:
                writer.add_scalar(key,value,epoch)
            state = {'architecture':'LaneNet-v1','config':vars(args),'epoch':epoch,
                     'val_loss':vl,'model_state':model.state_dict(),'optimizer_state':optimizer.state_dict()}
            torch.save(state,out/'last.pt')
            if vl<best:
                best=vl; torch.save(state,out/'best.pt')
            print(f'{epoch}/{args.epochs}: train loss {tl:.5f}; val loss {vl:.5f}; val IoU {vi:.4f}; elapsed {elapsed:.1f}s',flush=True)
    from report import plot_history
    plot_history(out/'history.csv',out/'loss.png')


if __name__=='__main__':
    main()
