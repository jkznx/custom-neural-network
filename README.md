# PSU Reservoir — Custom Lane Segmentation from Scratch

โปรเจกต์ Assignment-10 สำหรับ binary semantic segmentation ของพื้นที่ `lane` ใน PSU-reservoir dataset ออกแบบโมเดลเองด้วย PyTorch และเริ่มเทรนทุก layer จาก random initialization ไม่มี pretrained backbone ไม่มีการ freeze layer ใช้เฉพาะ CVAT `<polygon label="lane">` รวม polygon ทุกชิ้นในภาพเป็น foreground เดียว

**ขอบเขต label:** ใน dataset นี้ `lane` เป็นพื้นที่เลนตาม polygon ไม่ใช่เส้นตีถนน `track-line` ส่วน `sideway` ถือเป็น background เช่นเดียวกับพื้นที่อื่น ๆ โดยไม่ใช้ polyline หรือ bounding box มาสร้าง target

## โครงสร้างและเหตุผล

โมเดล **LaneNet-v1** เป็น residual encoder-decoder รูปตัว U ที่เขียนขึ้นเองใน `model.py` มี encoder 3 ระดับและ bottleneck ใช้ channel 6 → 12 → 24 → 48 decoder ใช้ bilinear interpolation, concat skip features และ residual block ก่อน convolution 1×1 ที่ให้ logits หนึ่ง channel

แต่ละ residual block = Conv3×3 → GroupNorm(3 groups) → ReLU → Conv3×3 → GroupNorm บวก shortcut (identity หรือ Conv1×1 เมื่อจำนวน channel เปลี่ยน) แล้ว ReLU น้ำหนัก convolution ทุกชั้นเริ่มด้วย Kaiming random initialization และ trainable ทั้งหมด

| Stage | Operation | Tensor shape (batch omitted) |
|---|---|---|
| Input | RGB / 255 | 3×96×160 |
| E1 | Residual block | 6×96×160 |
| E2 | MaxPool2 + residual block | 12×48×80 |
| E3 | MaxPool2 + residual block | 24×24×40 |
| Bottleneck | MaxPool2 + residual block | 48×12×20 |
| D3 | Bilinear upsample + E3 concat + block | 24×24×40 |
| D2 | Bilinear upsample + E2 concat + block | 12×48×80 |
| D1 | Bilinear upsample + E1 concat + block | 6×96×160 |
| Output | Conv1×1 | 1×96×160 logits |

เหตุผลในการออกแบบ:

- Channel น้อยและ downsample เพียง 3 ครั้ง ช่วยลด parameter และ activation memory เพื่อทดลองบน laptop/PC
- Skip connections ส่งรายละเอียดขอบเลนให้ decoder และ residual connections ช่วยการไหลของ gradient เมื่อเริ่มเทรนจากศูนย์
- GroupNorm ไม่อาศัยสถิติของ batch จึงเหมาะกับ batch 2–4
- Bilinear upsampling ไม่มี parameter เพิ่มและหลีกเลี่ยงการใช้ transposed convolution ในขั้น upsample
- Loss = 0.5 BCEWithLogits + 0.5 soft Dice เพื่อให้ทั้งการจำแนก pixel และพื้นที่ foreground มีส่วนใน optimization
- ความละเอียด 96×160 เป็นค่าเริ่มต้นสำหรับการเทรนจริงบน CPU; อาจทำให้ขอบเลนละเอียดหรือพื้นที่แคบสูญหาย หากเครื่องมีทรัพยากรพอให้เพิ่มเป็น 192×320 และเทรนใหม่ ค่า 48×48 ในไฟล์แนบเป็นตัวอย่าง ไม่ใช่ข้อกำหนดใน brief

แผนภาพเพิ่มเติมอยู่ใน [custom-unet-architecture.md](custom-unet-architecture.md)

```mermaid
flowchart TD
  X[RGB 3 x 96 x 160] --> E1[Residual 6 channels]
  E1 --> P1[MaxPool 2] --> E2[Residual 12 channels]
  E2 --> P2[MaxPool 2] --> E3[Residual 24 channels]
  E3 --> P3[MaxPool 2] --> B[Residual 48 channels]
  B --> U3[Bilinear + concat E3] --> D3[Residual 24 channels]
  D3 --> U2[Bilinear + concat E2] --> D2[Residual 12 channels]
  D2 --> U1[Bilinear + concat E1] --> D1[Residual 6 channels]
  D1 --> H[Conv 1x1 to 1 logit channel]
  H --> M[Sigmoid + threshold 0.5]
  E3 -. skip .-> U3
  E2 -. skip .-> U2
  E1 -. skip .-> U1
```

## Dataset และการป้องกัน data leakage

ต้นฉบับคือ `Dataset-Assignment-8` ของผู้ใช้ มี CVAT image XML, ภาพใน subfolders และ ZIP ภาพ 400 รูป `prepare_data.py` อ่านเฉพาะ polygon label `lane` แปลงเป็น YOLO-seg class 0 ที่มี normalized coordinates แล้วคัดลอกภาพเข้า working directory ใหม่ ไม่แก้ต้นฉบับ

ผลการตรวจและ split ของข้อมูลที่มีจริง:

| Set | Images | การใช้ |
|---|---:|---|
| Train | 528 | Optimization และ augmentation |
| Validation | 100 | Frames 1103–1202; เลือก checkpoint จาก loss ต่ำสุด |
| Excluded temporal gap | 20 | Frames 1083–1102; ไม่ใช้เทรน/ประเมิน |
| Test | 400 | Test เดิม frames 2–401; ใช้หลังจบการเทรน |

XML ในช่วง 450–499 และ 600–649 อ้างถึงภาพรวม 100 รูปที่ไม่พบในชุดข้อมูลที่ให้มา จึงบันทึกไว้ใน `data/manifest.json` และไม่นำมาเทรน ภาพซ้ำจาก Train ที่ตรงกับ Test 2 รูปถูกตัดออกตาม SHA-256 ของไฟล์ภาพ คง Test ไว้ทุกภาพ ใช้ validation ช่วงต่อเนื่องแทนการสุ่มเฟรมใกล้กัน ผลตรวจยังไม่สามารถยืนยันความเป็นอิสระของวิดีโอต้นทางหรือ near-duplicate ที่ไฟล์ต่างกันได้ จึงควรตีความผลเป็นผลบนเส้นทาง/ชุดข้อมูลนี้ และเพิ่มข้อมูลคนละการถ่ายหากต้องการวัด generalization

สำเนา manifest สำหรับแพ็กส่งอยู่ใน [dataset_manifest.json](dataset_manifest.json) มีรายชื่อภาพที่ใช้ ภาพหาย ภาพซ้ำที่ถูกตัดออก และ SHA-256 ของภาพต้นฉบับ

```text
data/
  images/{train,val,test}/*.jpg
  labels/{train,val,test}/*.txt
  manifest.json
```

Label 1 บรรทัดต่อ polygon: `0 x1 y1 x2 y2 ... xn yn` ต้องมีอย่างน้อย 3 vertices ใน [0,1] Rasterize ที่ความละเอียดภาพต้นฉบับก่อน resize mask แบบ nearest-neighbor ใช้การวาดแต่ละ polygon เป็น union เพื่อไม่ให้พื้นที่ซ้อนกลายเป็นรู ไฟล์ annotation ที่หายถือเป็น error; ภาพ negative ต้องมีไฟล์ label ว่าง รองรับภาพขนาดต่างกัน Polylines ในรูปแบบ YOLO txt ไม่สามารถแยกจาก polygon ได้ด้วย syntax เพียงอย่างเดียว จึงต้อง export เฉพาะ polygon จากต้นทาง; ตัวแปลง CVAT ในโปรเจกต์นี้เลือกชนิดข้อมูลอย่างชัดเจน

## ติดตั้งและรัน

ต้องมี Python 3.10+ (การรันที่ส่งใช้ Python 3.12) เปิด terminal ในโฟลเดอร์ repo บน Windows:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

สำหรับ CPU สามารถติดตั้ง torch จาก CPU index ก่อน:

```powershell
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

ไฟล์ `requirements-lock.txt` บันทึกเวอร์ชันจากการรันจริง การติดตั้ง CUDA ต้องเลือก PyTorch build ที่เข้ากับเครื่อง จากนั้นใช้ `--device cuda` แทน `cpu`

### 1. เตรียมข้อมูล

```powershell
python prepare_data.py --source "C:\Users\ACER\Desktop\1-2569\353-Ecosystem\P_RATTACHAI\Dataset-Assignment-8" --output data --archive-cache archive-cache --skip-missing
```

`--skip-missing` เป็นการเลือกตัดภาพที่ไม่มีต้นฉบับออกโดยมี audit log หากไม่ระบุ flag นี้ script จะหยุดเมื่อพบภาพหาย ตั้งค่า validation และ temporal gap ได้ด้วย `--val-count` และ `--gap` Output ต้องว่างเพื่อไม่ให้ข้อมูลรันเก่าปะปน

### 2. เทรนทุก layer ใหม่

```powershell
python train.py --data-root data --output runs/lane --epochs 30 --batch-size 4 --height 96 --width 160 --base 6 --threads 4 --device cpu
tensorboard --logdir runs/lane/tensorboard
```

ใช้ AdamW (lr 0.001, weight decay 0.0001), seed 42, batch 4, float32 ไม่โหลด weights ใดก่อนเทรน Augmentation ใช้เฉพาะ Train: white balance ±6%, brightness ±15/255 และ Gaussian blur เล็กน้อย Validation/Test ไม่ augment จัดเก็บ image/mask ที่ resize แล้วใน RAM เป็น uint8 เพื่อไม่ถอดรหัสภาพซ้ำทุก epoch; cache เป็นส่วนของ training memory ไม่ใช่ inference

Artifacts: `best.pt` (validation loss ต่ำสุด), `last.pt`, `history.csv`, `loss.png`, `config.json`, `environment.json`, TensorBoard และ split manifest ไม่เลือก threshold จาก Test ใช้ sigmoid > 0.5 คงที่ เริ่ม run ใหม่ด้วย `--output` ชื่อใหม่

### 3. Inference พร้อมวัด memory

```powershell
python inference.py --images-dir data/images/test --labels-dir data/labels/test --checkpoint runs/lane/best.pt --output run/test --threads 4 --device cpu
```

ขนาด input และ base channels อ่านจาก checkpoint โดยตรง Output เป็น PNG binary mask 0/255 ที่ความละเอียดภาพเดิมใน `run/test/masks/` พร้อม `snapshot.png`, `predictions.json`, `memory.json` มี warmup 5 ครั้งก่อนวัด forward latency ใช้ `torch.inference_mode()` และ batch 1 ค่า memory แยก parameter bytes, process RSS และ CUDA peak allocated/reserved (เมื่อใช้ CUDA) จากกัน Model parameter MiB ไม่ใช่ memory footprint ทั้ง process

### 4. ประเมินผลและอัปเดต README

```powershell
python evaluate.py --images-dir data/images/test --labels-dir data/labels/test --pred-masks-dir run/test/masks --output run/test/metrics.json
python report.py --run runs/lane --evaluation run/test/metrics.json --memory run/test/memory.json
python -m unittest discover -s tests -v
```

IoU = TP/(TP+FP+FN), Dice = 2TP/(2TP+FP+FN), precision = TP/(TP+FP), recall = TP/(TP+FN) รายงานทั้ง global pixel metrics และ mean per-image IoU เพื่อให้เห็นวิธีเฉลี่ยที่ต่างกัน Pixel accuracy มีใน JSON แต่ background มากอาจทำให้ค่านี้สูงโดยโมเดลไม่ได้แบ่งเลนดี

นิยามเพิ่มเติมตามตัวอย่างแนบ: ภาพที่มี GT lane และ IoU **มากกว่า** 0.6 ถือว่า detected; detection rate หารด้วยจำนวนภาพที่มี lane; average IoU detected เฉลี่ยเฉพาะกลุ่มนี้ ไม่ใช้ค่านี้เป็นหลักเพราะไม่นับภาพที่โมเดลทำได้ไม่ดี เมื่อทั้ง prediction/GT ว่างให้ IoU = 1 แต่ไม่นับเป็น lane detection เมื่อไม่มี detected image ให้ค่าเฉลี่ยเป็น null Script จะหยุดเมื่อ prediction หาย ขนาดผิด หรือไม่ใช่ binary เพื่อไม่รายงานผลจาก test subset โดยไม่ตั้งใจ

## ผลการเทรนและประเมินจริง

<!-- RESULTS_START -->
ผลจากการรันจริงบน cpu: 30 epochs, input 96×160, inference batch = 1

| รายการ | ผลจริง |
|---|---:|
| Test images | 400 |
| Selected checkpoint epoch (minimum validation loss) | 20 |
| Final train loss (epoch 30) | 0.022217 |
| Final validation loss (epoch 30) | 0.040079 |
| Global lane IoU | 0.933014 |
| Global Dice / F1 | 0.965347 |
| Pixel precision | 0.965736 |
| Pixel recall | 0.964957 |
| Mean per-image IoU | 0.933073 |
| Images with lane and IoU > 0.6 | 400 / 400 |
| Detection rate (IoU > 0.6, positive images) | 1.000000 |
| Mean IoU of detected positive images | 0.9330731375425491 |
| Parameters | 72,859 |
| FP32 model parameters + buffers | 0.2779 MiB |
| Sampled peak process RSS | 246.82 MiB |
| Process peak working set (Windows, lifetime) | 252.7578125 MiB |
| Peak RSS increase over pre-model baseline | 49.25 MiB |
| Mean forward latency (warm model) | 5.57 ms/image |

![Training and validation loss](runs/lane/loss.png)

![Original, prediction, overlay, ground truth](run/test/snapshot.png)

Training elapsed: 245.4 seconds. Checkpoint selection uses minimum validation loss; test data were not used for optimization or threshold selection. Pixel metrics use original image resolution after nearest-neighbor resizing of predicted masks. RSS includes Python, PyTorch, image buffers and output processing; model size alone is not total inference memory. Sampling every 10 ms can miss brief peaks. Windows lifetime peak working set is reported separately. CUDA allocation is unavailable in this CPU run.

กราฟแสดงว่า train loss ลดลงและ validation loss เข้าสู่ช่วงแกว่งตัวหลังช่วงแรก ค่า validation loss ต่ำสุดอยู่ที่ epoch 20 ไม่ใช่ epoch สุดท้าย การเทรนต่อทำให้ train loss ลดลงแต่ validation loss ไม่ดีขึ้นสม่ำเสมอ จึงใช้ best checkpoint เพื่อลดผลของ overfitting ภาพ snapshot เลือกภาพแรกตามชื่อไฟล์ (`frame_0002_f52.jpg`) ก่อนคำนวณคะแนน ไม่ได้เลือกภาพที่มีคะแนนสูงสุด ทั้ง 400 ภาพ Test มี lane จึงยังไม่มีหลักฐานจากชุดทดสอบภาพที่ไม่มี lane

<!-- RESULTS_END -->

## ไฟล์สำหรับส่งงาน

แนบ repo นี้พร้อม README, model.py, dataset.py, prepare_data.py, train.py, inference.py, evaluate.py, common.py, report.py, requirements, tests, `runs/lane/best.pt`, history, กราฟ loss และ `run/test/{metrics.json,memory.json,snapshot.png,predictions.json}` สามารถอัปโหลด ZIP หรือ repo ไป GitHub/PSU storage เพื่อส่ง URL ได้ ภาพ dataset และ prediction masks จำนวนมากไม่จำเป็นต้องอยู่ใน Git แต่ควรเก็บ manifest และระบุที่มาข้อมูลให้ผู้ตรวจรันซ้ำได้ ห้ามอ้างกราฟ/metrics จาก run อื่นเป็นผลของ checkpoint นี้

ตัวอย่างที่ผู้สอนให้ศึกษาประกอบ: [Ultrafast Lane Detection](https://github.com/ibaiGorordo/Ultrafast-Lane-Detection-Inference-Pytorch-) และ [YOLOTL](https://github.com/Highsky7/YOLOTL) โปรเจกต์นี้ใช้สถาปัตยกรรมที่เขียนเอง ไม่ใช้ weights ของตัวอย่างดังกล่าว
