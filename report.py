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
        if f'{split}_loss' not in rows[0]:
            continue
        axes[0].plot(epochs,[float(r[f'{split}_loss']) for r in rows],label=split)
        axes[1].plot(epochs,[float(r[f'{split}_iou']) for r in rows],label=split)
    for ax,title in zip(axes,['BCE + Dice loss','Global lane IoU (network resolution)']):
        ax.set(xlabel='Epoch',title=title); ax.grid(alpha=.25); ax.legend()
    fig.tight_layout(); fig.savefig(output,dpi=160); plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',default='runs/lane64x36')
    p.add_argument('--evaluation',default='run/test64x36/metrics.json')
    p.add_argument('--memory',default='run/test64x36/memory.json')
    p.add_argument('--readme',default='README.md')
    args=p.parse_args(); run=Path(args.run); inf=Path(args.memory).parent
    with (run/'history.csv').open() as f:
        rows=list(csv.DictReader(f))
    m=json.loads(Path(args.evaluation).read_text())['summary']
    mem=json.loads(Path(args.memory).read_text())
    preds=json.loads((inf/'predictions.json').read_text())
    avg=f"{m['avg_iou_detected']:.6f}" if m['avg_iou_detected'] is not None else 'N/A'
    plot_history(run/'history.csv',run/'loss.png')
    text=f'''### กราฟ loss และการลู่เข้า

![Training loss and IoU]({run.as_posix()}/loss.png)

เทรนจริง {len(rows)} epochs บน CPU, batch 4, float32, seed 42 ค่า train loss ลดจาก **{float(rows[0]['train_loss']):.6f}** เป็น **{float(rows[-1]['train_loss']):.6f}** เวลา epoch loop **{float(rows[-1]['elapsed_seconds']):.1f} วินาที** บันทึกทุก epoch ใน [history.csv]({run.as_posix()}/history.csv)

กราฟนี้เป็น training loss การรันนี้ใช้ Train 65% ทั้งหมดและ Test 35% ไม่มี validation split แยก เลือก checkpoint epoch {mem['checkpoint_epoch']} ซึ่งเป็น epoch สุดท้ายตามที่กำหนดก่อนเทรน ไม่ใช้ Test เลือก checkpoint หรือ threshold การลดลงของ training loss แสดงการเรียนรู้ชุด Train; ประสิทธิภาพกับข้อมูลที่ไม่ได้ใช้เทรนดูจาก Test ด้านล่าง

### ค่าประสิทธิภาพของโมเดล

ประเมิน binary masks ที่ความละเอียด **width {mem['width']} × height {mem['height']}** โดย rasterize GT polygon ที่ขนาดต้นฉบับก่อน resize ด้วย nearest-neighbor ใช้ sigmoid > 0.5 เป็น lane

| Metric | ผลจริง |
|---|---:|
| Test images | {m['num_images']} |
| Global pixel-wise lane IoU | {m['global_iou']:.6f} |
| Mean per-image IoU | {m['mean_image_iou']:.6f} |
| Dice / F1 | {m['dice']:.6f} |
| Precision | {m['precision']:.6f} |
| Recall | {m['recall']:.6f} |
| Detected: GT lane and IoU > 0.6 | {m['num_detected']} / {m['num_positive_images']} |
| Detection rate | {m['detection_rate']*100:.2f}% |
| Average IoU of detected results | {avg} |

ผล Yes/No และคะแนน IoU รายภาพอยู่ใน [metrics.json]({Path(args.evaluation).as_posix()}) Detection rate คือการผ่านเกณฑ์ IoU > 0.6 ไม่ใช่ความถูกต้องทุก pixel Global IoU รวม TP/FP/FN ทุกภาพ ส่วน mean per-image IoU เฉลี่ยรายภาพ

### ตัวอย่างก่อนและหลัง inference

![Original, predicted mask, overlay and polygon ground truth]({inf.as_posix()}/snapshot.png)

ภาพ `{preds['images'][0]}` เป็นภาพแรกตามชื่อไฟล์ที่เลือกก่อนคำนวณคะแนน จากซ้าย: RGB ต้นฉบับ, binary lane mask, overlay สีเขียวหลัง inference และ GT จาก polygon lane ภาพแสดงผลขยายกลับขนาดเดิมเพื่อดูง่าย ไฟล์ผลหลักใน masks/ มีขนาด 64×36 และมี native_masks/ สำหรับภาพขนาดเดิม

### Memory footprint ที่วัดจริงระหว่าง inference

ระบบ **{mem['platform']}**, PyTorch **{mem['torch']}**, CPU **{mem['cpu']}**, {mem['threads']} threads, batch 1, float32, input width {mem['width']} × height {mem['height']} ใช้ inference_mode และ warmup 5 ครั้ง

| รายการ | ค่าที่วัดได้ |
|---|---:|
| Parameters | {mem['parameters']:,} |
| FP32 model parameters + buffers | {mem['model_mib']:.4f} MiB |
| Baseline RSS ก่อนโหลดโมเดล | {mem['baseline_rss_mib']:.2f} MiB |
| Sampled peak process RSS | {mem['sampled_peak_rss_mib']:.2f} MiB |
| Peak RSS increase over baseline | {mem['rss_increase_mib']:.2f} MiB |
| Windows lifetime peak working set | {mem['os_peak_working_set_mib']:.2f} MiB |
| Mean forward latency หลัง warmup | {mem['mean_forward_ms']:.2f} ms/image |

RSS รวม Python, PyTorch, โหลด checkpoint, warmup, inference, image buffers, mask export และ snapshot อ่านทุก 10 ms จึงอาจพลาด peak สั้น ๆ Windows lifetime peak working set เป็นค่า OS ตลอด process รวมช่วงเริ่มโปรแกรมด้วย ขนาด parameter อย่างเดียวไม่ใช่ inference memory ทั้ง process ค่า latency วัดเฉพาะ forward ไม่รวม decode, preprocessing, sigmoid, resize และบันทึกไฟล์ การรันบน CPU จึงไม่มีค่า CUDA VRAM รายละเอียดใน [memory.json]({Path(args.memory).as_posix()})
'''
    path=Path(args.readme); readme=path.read_text(encoding='utf-8')
    start,end='<!-- RESULTS_START -->','<!-- RESULTS_END -->'
    if readme.count(start)!=1 or readme.count(end)!=1:
        raise ValueError('README requires one RESULTS_START/RESULTS_END pair')
    path.write_text(readme.split(start)[0]+start+'\n'+text+'\n'+end+readme.split(end)[1],encoding='utf-8')
    (run/'RESULTS.md').write_text(text,encoding='utf-8')
    print('README updated with current results')


if __name__=='__main__':
    main()
