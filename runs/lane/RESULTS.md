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
