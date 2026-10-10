#!/bin/bash
# Modified for the PFD-DETR distribution.

                                                                       
                                                                                     

                                                             
set -e

                                     
if [ -z "$1" ]; then
    echo "Error: No ONNX file provided."
    echo "Usage: $0 /path/to/your/model.onnx"
    exit 1
fi

ONNX_FILE=$1
                                                            
ENGINE_FILE="${ONNX_FILE%.onnx}.trt"

echo "==> Converting ONNX to TensorRT Engine <=="
echo "  - Input ONNX:  $ONNX_FILE"
echo "  - Output TRT:  $ENGINE_FILE"
echo "  - Precision:   FP16"
echo "=========================================="

                          
                                                                      
                                                                   
trtexec --onnx="$ONNX_FILE" \
        --saveEngine="$ENGINE_FILE" \
        --fp16 \
        --verbose

echo "=========================================="
echo "✅ Successfully created TensorRT engine: $ENGINE_FILE"