# Custom U-Net: LaneNet-v1 (64×36)

ขนาดใน assignment ใช้ convention **width × height × channels**: RGB 64×36×3 และ binary mask 64×36×1 PyTorch ใช้ tensor N×C×H×W จึงเป็น input N×3×36×64 และ logits N×1×36×64

## Network Mermaid

```mermaid
flowchart TD
    INPUT["RGB input<br/>3 x 36 x 64"]

    subgraph ENC["Encoder"]
        E1["Residual block: 3 to 6<br/>6 x 36 x 64"]
        P1["MaxPool 2 x 2"]
        E2["Residual block: 6 to 12<br/>12 x 18 x 32"]
        P2["MaxPool 2 x 2"]
        E3["Residual block: 12 to 24<br/>24 x 9 x 16"]
        P3["MaxPool 2 x 2"]
        E1 --> P1 --> E2 --> P2 --> E3 --> P3
    end

    BRIDGE["Bottleneck residual block: 24 to 48<br/>48 x 4 x 8"]

    subgraph DEC["Decoder"]
        U3["Bilinear upsample<br/>48 x 9 x 16"]
        C3["Concat with E3<br/>72 x 9 x 16"]
        D3["Residual block: 72 to 24<br/>24 x 9 x 16"]
        U2["Bilinear upsample<br/>24 x 18 x 32"]
        C2["Concat with E2<br/>36 x 18 x 32"]
        D2["Residual block: 36 to 12<br/>12 x 18 x 32"]
        U1["Bilinear upsample<br/>12 x 36 x 64"]
        C1["Concat with E1<br/>18 x 36 x 64"]
        D1["Residual block: 18 to 6<br/>6 x 36 x 64"]
        U3 --> C3 --> D3 --> U2 --> C2 --> D2 --> U1 --> C1 --> D1
    end

    HEAD["Conv 1 x 1: 6 to 1<br/>Logits: 1 x 36 x 64"]
    MASK["Sigmoid, then threshold greater than 0.5<br/>Binary lane mask: 36 x 64"]
    NATIVE["Nearest-neighbor resize<br/>Mask at original image resolution"]

    INPUT --> E1
    P3 --> BRIDGE --> U3
    E3 -. "skip features" .-> C3
    E2 -. "skip features" .-> C2
    E1 -. "skip features" .-> C1
    D1 --> HEAD --> MASK --> NATIVE
```

แผนภาพแสดง C×H×W ไม่รวม batch Concat มี channel 72, 36, 18 ตามลำดับ MaxPool ทำให้ความสูง 9 กลายเป็น 4 ที่ bottleneck; decoder จึง interpolate ไปยังขนาด skip โดยตรง (4→9) เพื่อคืน output 36×64 อย่างถูกต้อง ขั้น native resize เป็นเพียงการแสดงผลและส่งออกสำเนาภาพ ไม่เปลี่ยนขนาด output หลัก 64×36

## Residual block Mermaid

```mermaid
flowchart LR
    X["Input: Cin channels"] --> CONV1["Conv 3 x 3<br/>Cin to Cout"]
    CONV1 --> GN1["GroupNorm<br/>3 groups"] --> R1["ReLU"]
    R1 --> CONV2["Conv 3 x 3<br/>Cout to Cout"] --> GN2["GroupNorm<br/>3 groups"]
    X --> SHORTCUT["Shortcut<br/>Identity if Cin equals Cout<br/>Conv 1 x 1 otherwise"]
    GN2 --> ADD["Element-wise addition"]
    SHORTCUT --> ADD --> R2["ReLU"] --> Y["Output: Cout channels"]
```

มี 72,859 trainable parameters, GroupNorm 3 groups, random Kaiming convolution weights และไม่มี pretrained backbone ดู implementation ใน [model.py](model.py) และผลจริงใน [README.md](README.md)
