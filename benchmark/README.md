# Upstream benchmarking tools


## YOLO latency examples
The following examples use [YOLOv8](https://github.com/ultralytics/ultralytics).

<details open>
<summary>1. ONNX export</summary>

Use `export_onnx` in `yolov8_onnx.py` to export the model with the required output format.
</details>


<details>
<summary>2. NMS insertion</summary>

Use `yolo_insert_nms` in `utils.py`. Inspect the exported ONNX graph with [Netron](https://netron.app/).
</details>


<details>
<summary>3. TensorRT conversion</summary>

Use the commands in `trtexec.md` or the Python functions in `utils.py`.
```bash
trtexec --onnx=./yolov8l_w_nms.onnx --saveEngine=yolov8l_w_nms.engine --buildOnly --fp16
```
</details>


<details>
<summary>4. Latency measurement</summary>

Run the relevant command in `trtexec.md` without `--buildOnly`.

</details>



<details>
<summary>5. Profiling</summary>

Prefix the inference command with:
```bash
nsys profile --force-overwrite=true  -t 'nvtx,cuda,osrt,cudnn' -c cudaProfilerApi -o yolov8l_w_nms 
```
Inspect the result with Nsight Systems.
<img width="1090" alt="image" src="https://github.com/lyuwenyu/RT-DETR/assets/17582080/507d8bde-9e7c-4ae5-b571-976c540ef2c6">

</details>


<details>
<summary>6. Python inference</summary>

Use `trtinfer.py` for inference and latency measurement on COCO validation images.

</details>

<!-- Modified for the PFD-DETR distribution. -->
