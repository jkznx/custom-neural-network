### กราฟ loss และการลู่เข้า

![Training loss and IoU](runs/lane64x36/loss.png)

เทรนจริง 30 epochs บน CPU, batch 4, float32, seed 42 ค่า train loss ลดจาก **0.120468** เป็น **0.012987** เวลา epoch loop **76.1 วินาที** บันทึกทุก epoch ใน [history.csv](runs/lane64x36/history.csv)

กราฟนี้เป็น training loss การรันนี้ใช้ Train 65% ทั้งหมดและ Test 35% ไม่มี validation split แยก เลือก checkpoint epoch 30 ซึ่งเป็น epoch สุดท้ายตามที่กำหนดก่อนเทรน ไม่ใช้ Test เลือก checkpoint หรือ threshold การลดลงของ training loss แสดงการเรียนรู้ชุด Train; ประสิทธิภาพกับข้อมูลที่ไม่ได้ใช้เทรนดูจาก Test ด้านล่าง

### ค่าประสิทธิภาพของโมเดล

ประเมิน binary masks ที่ความละเอียด **width 64 × height 36** โดย rasterize GT polygon ที่ขนาดต้นฉบับก่อน resize ด้วย nearest-neighbor ใช้ sigmoid > 0.5 เป็น lane

| Metric | ผลจริง |
|---|---:|
| Test images | 367 |
| Global pixel-wise lane IoU | 0.828676 |
| Mean per-image IoU | 0.828612 |
| Dice / F1 | 0.906313 |
| Precision | 0.864659 |
| Recall | 0.952183 |
| Detected: GT lane and IoU > 0.6 | 367 / 367 |
| Detection rate | 100.00% |
| Average IoU of detected results | 0.828612 |

ผล Yes/No และคะแนน IoU รายภาพอยู่ใน [metrics.json](run/test64x36/metrics.json) Detection rate คือการผ่านเกณฑ์ IoU > 0.6 ไม่ใช่ความถูกต้องทุก pixel Global IoU รวม TP/FP/FN ทุกภาพ ส่วน mean per-image IoU เฉลี่ยรายภาพ

### ตัวอย่างก่อนและหลัง inference

![Original, predicted mask, overlay and polygon ground truth](run/test64x36/snapshot.png)

ภาพ `frame_0836_f21681.jpg` เป็นภาพแรกตามชื่อไฟล์ที่เลือกก่อนคำนวณคะแนน จากซ้าย: RGB ต้นฉบับ, binary lane mask, overlay สีเขียวหลัง inference และ GT จาก polygon lane ภาพแสดงผลขยายกลับขนาดเดิมเพื่อดูง่าย ไฟล์ผลหลักใน masks/ มีขนาด 64×36 และมี native_masks/ สำหรับภาพขนาดเดิม

### Memory footprint ที่วัดจริงระหว่าง inference

ระบบ **Windows-11-10.0.26300-SP0**, PyTorch **2.14.1+cpu**, CPU **Intel64 Family 6 Model 186 Stepping 2, GenuineIntel**, 4 threads, batch 1, float32, input width 64 × height 36 ใช้ inference_mode และ warmup 5 ครั้ง

| รายการ | ค่าที่วัดได้ |
|---|---:|
| Parameters | 72,859 |
| FP32 model parameters + buffers | 0.2779 MiB |
| Baseline RSS ก่อนโหลดโมเดล | 197.61 MiB |
| Sampled peak process RSS | 243.13 MiB |
| Peak RSS increase over baseline | 45.52 MiB |
| Windows lifetime peak working set | 248.23 MiB |
| Mean forward latency หลัง warmup | 1.97 ms/image |

RSS รวม Python, PyTorch, โหลด checkpoint, warmup, inference, image buffers, mask export และ snapshot อ่านทุก 10 ms จึงอาจพลาด peak สั้น ๆ Windows lifetime peak working set เป็นค่า OS ตลอด process รวมช่วงเริ่มโปรแกรมด้วย ขนาด parameter อย่างเดียวไม่ใช่ inference memory ทั้ง process ค่า latency วัดเฉพาะ forward ไม่รวม decode, preprocessing, sigmoid, resize และบันทึกไฟล์ การรันบน CPU จึงไม่มีค่า CUDA VRAM รายละเอียดใน [memory.json](run/test64x36/memory.json)
