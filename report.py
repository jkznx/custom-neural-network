"""Render plots and insert measured test results into README."""
import argparse
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_history(history, output):
    with Path(history).open() as f:
        rows=list(csv.DictReader(f))
    epochs=[int(r['epoch']) for r in rows]
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for split in ['train','val']:
        axes[0].plot(epochs,[float(r[f'{split}_loss']) for r in rows],label=split)
        axes[1].plot(epochs,[float(r[f'{split}_iou']) for r in rows],label=split)
    for ax,title in zip(axes,['BCE + Dice loss','Global lane IoU (network resolution)']):
        ax.set(xlabel='Epoch',title=title); ax.grid(alpha=.25); ax.legend()
    fig.tight_layout(); fig.savefig(output,dpi=160); plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',default='runs/lane')
    p.add_argument('--evaluation',default='run/test/metrics.json')
    p.add_argument('--memory',default='run/test/memory.json')
    p.add_argument('--readme',default='README.md')
    args=p.parse_args()
    run=Path(args.run)
    with (run/'history.csv').open() as f:
        rows=list(csv.DictReader(f))
    metrics=json.loads(Path(args.evaluation).read_text())['summary']
    memory=json.loads(Path(args.memory).read_text())
    plot_history(run/'history.csv',run/'loss.png')
    text=f'''ผลจากการรันจริงบน {memory['device']}: {len(rows)} epochs, input {memory['height']}×{memory['width']}, inference batch = 1

| รายการ | ผลจริง |
|---|---:|
| Test images | {metrics['num_images']} |
| Selected checkpoint epoch (minimum validation loss) | {memory['checkpoint_epoch']} |
| Final train loss (epoch {len(rows)}) | {float(rows[-1]['train_loss']):.6f} |
| Final validation loss (epoch {len(rows)}) | {float(rows[-1]['val_loss']):.6f} |
| Global lane IoU | {metrics['global_iou']:.6f} |
| Global Dice / F1 | {metrics['dice']:.6f} |
| Pixel precision | {metrics['precision']:.6f} |
| Pixel recall | {metrics['recall']:.6f} |
| Mean per-image IoU | {metrics['mean_image_iou']:.6f} |
| Images with lane and IoU > 0.6 | {metrics['num_detected']} / {metrics['num_positive_images']} |
| Detection rate (IoU > 0.6, positive images) | {metrics['detection_rate']:.6f} |
| Mean IoU of detected positive images | {metrics['avg_iou_detected']} |
| Parameters | {memory['parameters']:,} |
| FP32 model parameters + buffers | {memory['model_mib']:.4f} MiB |
| Sampled peak process RSS | {memory['sampled_peak_rss_mib']:.2f} MiB |
| Process peak working set (Windows, lifetime) | {memory['os_peak_working_set_mib']} MiB |
| Peak RSS increase over pre-model baseline | {memory['rss_increase_mib']:.2f} MiB |
| Mean forward latency (warm model) | {memory['mean_forward_ms']:.2f} ms/image |

![Training and validation loss]({run.as_posix()}/loss.png)

![Original, prediction, overlay, ground truth](run/test/snapshot.png)

Training elapsed: {float(rows[-1]['elapsed_seconds']):.1f} seconds. Checkpoint selection uses minimum validation loss; test data were not used for optimization or threshold selection. Pixel metrics use original image resolution after nearest-neighbor resizing of predicted masks. RSS includes Python, PyTorch, image buffers and output processing; model size alone is not total inference memory. Sampling every 10 ms can miss brief peaks. Windows lifetime peak working set is reported separately. CUDA allocation is unavailable in this CPU run.

กราฟแสดงว่า train loss ลดลงและ validation loss เข้าสู่ช่วงแกว่งตัวหลังช่วงแรก ค่า validation loss ต่ำสุดอยู่ที่ epoch {memory['checkpoint_epoch']} ไม่ใช่ epoch สุดท้าย การเทรนต่อทำให้ train loss ลดลงแต่ validation loss ไม่ดีขึ้นสม่ำเสมอ จึงใช้ best checkpoint เพื่อลดผลของ overfitting ภาพ snapshot เลือกภาพแรกตามชื่อไฟล์ (`frame_0002_f52.jpg`) ก่อนคำนวณคะแนน ไม่ได้เลือกภาพที่มีคะแนนสูงสุด ทั้ง 400 ภาพ Test มี lane จึงยังไม่มีหลักฐานจากชุดทดสอบภาพที่ไม่มี lane
'''
    path=Path(args.readme)
    readme=path.read_text(encoding='utf-8')
    start,end='<!-- RESULTS_START -->','<!-- RESULTS_END -->'
    before=readme.split(start)[0]; after=readme.split(end)[1]
    path.write_text(before+start+'\n'+text+'\n'+end+after,encoding='utf-8')
    (run/'RESULTS.md').write_text(text,encoding='utf-8')
    print('README updated with measured results')


if __name__=='__main__':
    main()
