# LaneNet-v1 — Mermaid Architecture

แผนภาพนี้อ้างอิง implementation ใน [model.py](model.py) โดยตรง ใช้ `base=6`, input RGB 96×160 และแสดง tensor shape ในรูป **C × H × W** โดยไม่รวม batch dimension มี trainable parameters **72,859 ตัว** ทุก layer เทรนใหม่ตั้งแต่ต้น ไม่มี pretrained weights

## Network และ skip connections

```mermaid
flowchart TD
    INPUT["RGB input<br/>3 x 96 x 160"]

    subgraph ENC["Encoder"]
        E1["Residual block: 3 to 6<br/>6 x 96 x 160"]
        P1["MaxPool 2 x 2"]
        E2["Residual block: 6 to 12<br/>12 x 48 x 80"]
        P2["MaxPool 2 x 2"]
        E3["Residual block: 12 to 24<br/>24 x 24 x 40"]
        P3["MaxPool 2 x 2"]
        E1 --> P1 --> E2 --> P2 --> E3 --> P3
    end

    BRIDGE["Bottleneck residual block: 24 to 48<br/>48 x 12 x 20"]

    subgraph DEC["Decoder"]
        U3["Bilinear upsample<br/>48 x 24 x 40"]
        C3["Concat with E3<br/>72 x 24 x 40"]
        D3["Residual block: 72 to 24<br/>24 x 24 x 40"]
        U2["Bilinear upsample<br/>24 x 48 x 80"]
        C2["Concat with E2<br/>36 x 48 x 80"]
        D2["Residual block: 36 to 12<br/>12 x 48 x 80"]
        U1["Bilinear upsample<br/>12 x 96 x 160"]
        C1["Concat with E1<br/>18 x 96 x 160"]
        D1["Residual block: 18 to 6<br/>6 x 96 x 160"]
        U3 --> C3 --> D3 --> U2 --> C2 --> D2 --> U1 --> C1 --> D1
    end

    HEAD["Conv 1 x 1: 6 to 1<br/>Logits: 1 x 96 x 160"]
    MASK["Sigmoid, then threshold greater than 0.5<br/>Binary lane mask: 96 x 160"]
    NATIVE["Nearest-neighbor resize<br/>Mask at original image resolution"]

    INPUT --> E1
    P3 --> BRIDGE --> U3
    E3 -. "skip features" .-> C3
    E2 -. "skip features" .-> C2
    E1 -. "skip features" .-> C1
    D1 --> HEAD --> MASK --> NATIVE
```

เส้นทึบแสดงลำดับการคำนวณ เส้นประแสดง skip features จาก encoder ที่นำไป concat ใน decoder การ upsample เป็น bilinear และไม่ลดจำนวน channel จนกว่าจะผ่าน residual block ถัดไป ดังนั้น concat ใน decoder มี channel 48+24=72, 24+12=36 และ 12+6=18 ตามลำดับ

ส่วน sigmoid, threshold และ nearest-neighbor resize เป็นขั้น inference หลัง network ส่วนการเทรนส่ง raw logits เข้า BCEWithLogitsLoss โดยตรง ไม่ threshold ก่อนคำนวณ loss

## ภายใน residual block

```mermaid
flowchart LR
    X["Input: Cin channels"] --> CONV1["Conv 3 x 3<br/>Cin to Cout"]
    CONV1 --> GN1["GroupNorm<br/>3 groups"] --> R1["ReLU"]
    R1 --> CONV2["Conv 3 x 3<br/>Cout to Cout"] --> GN2["GroupNorm<br/>3 groups"]
    X --> SHORTCUT["Shortcut<br/>Identity if Cin equals Cout<br/>Conv 1 x 1 otherwise"]
    GN2 --> ADD["Element-wise addition"]
    SHORTCUT --> ADD --> R2["ReLU"] --> Y["Output: Cout channels"]
```

Convolution 3×3 ใช้ padding 1 เพื่อรักษาขนาด spatial ส่วน shortcut ใช้ identity เมื่อจำนวน channel เท่ากัน และ Conv1×1 เมื่อจำนวน channel เปลี่ยน GroupNorm ใช้ 3 groups เพื่อรองรับ batch ขนาดเล็ก

## เหตุผลในการออกแบบ

- Encoder channel 6 → 12 → 24 → 48 และ downsample 3 ครั้ง ช่วยจำกัด parameter และ activation memory สำหรับ local machine
- Skip connections นำรายละเอียด spatial จาก encoder กลับไปยัง decoder
- Residual connections ช่วยให้ gradient ไหลผ่าน network ระหว่างการเทรนจากศูนย์
- Bilinear upsampling ไม่มี trainable parameter เพิ่ม
- GroupNorm ไม่ใช้ batch statistics จึงเหมาะกับการเทรน batch 2–4

ผลการเทรน กราฟ loss ภาพก่อน/หลัง และ memory footprint ที่วัดจริงอยู่ใน [README.md](README.md)
