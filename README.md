# CARLA VSAIT: Unpaired Image Translation via Vector Symbolic Architectures

This repository contains a sample code and a pretrained model in order to use VSAIT (VSAIT: Unpaired Image Translation via Vector Symbolic Architectures) image-to-image translation method in the CARLA simulator to enhance the photorealism of the Camera sensor.

## Dependencies

```
pip install carla
pip install onnxruntime-gpu
pip install torch
pip install opencv-python
pip install numpy
```

## Executing program

```
python carla_vsait.py
```

## Training

In order to train your own models download the official [VSAIT code](https://github.com/facebookresearch/vsait/tree/main) and follow the provided instruction. To export the ONNX model add the following blocks of code in the provided test.py script. The dataset used for training VSAIT can be found [here](https://www.kaggle.com/datasets/stefanospasios/carla2real-enhancing-the-photorealism-of-carla).
```
import torch
def export_to_onnx(model, export_path="model.onnx", input_size=(1, 3, 540, 960), device="cuda"):
    model.eval()
    model.to(device)

    dummy_input = torch.randn(*input_size).to(device)

    torch.onnx.export(
        model,
        dummy_input,
        export_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
    )

    print(f"Model exported to {export_path}")

# after trainer.test(solver)
export_to_onnx(solver)
```

## Translation Results (CARLA2Cityscapes)

![FinalColor-115504](https://github.com/user-attachments/assets/e21b58cc-2a49-4d20-a110-989b2620f876)

![FinalColor-046118](https://github.com/user-attachments/assets/2d0b25f6-f3d9-466e-b04f-0c5678d7f10e)

