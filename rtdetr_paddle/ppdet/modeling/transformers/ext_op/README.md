# Multi-scale deformable attention custom operator
This implementation follows the [Paddle custom-operator guide](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/custom_op/new_cpp_op_cn.html).

## 1. Dependencies
- Paddle >= 2.3.2
- gcc 8.2

## 2. Installation
Build the operator from its source directory:
```
cd rtdetr_paddle/ppdet/modeling/transformers/ext_op/
python setup_ms_deformable_attn_op.py install
```

After installation, use `ms_deformable_attn` as follows:
```
import paddle
from deformable_detr_ops import ms_deformable_attn

bs, n_heads, c = 2, 8, 8
query_length, n_levels, n_points = 2, 2, 2
spatial_shapes = paddle.to_tensor([(6, 4), (3, 2)], dtype=paddle.int64)
level_start_index = paddle.concat((paddle.to_tensor(
    [0], dtype=paddle.int64), spatial_shapes.prod(1).cumsum(0)[:-1]))
value_length = sum([(H * W).item() for H, W in spatial_shapes])

def get_test_tensors(channels):
    value = paddle.rand(
        [bs, value_length, n_heads, channels], dtype=paddle.float32) * 0.01
    sampling_locations = paddle.rand(
        [bs, query_length, n_heads, n_levels, n_points, 2],
        dtype=paddle.float32)
    attention_weights = paddle.rand(
        [bs, query_length, n_heads, n_levels, n_points],
        dtype=paddle.float32) + 1e-5
    attention_weights /= attention_weights.sum(-1, keepdim=True).sum(
        -2, keepdim=True)
    return [value, sampling_locations, attention_weights]

value, sampling_locations, attention_weights = get_test_tensors(c)

output = ms_deformable_attn(value,
                            spatial_shapes,
                            level_start_index,
                            sampling_locations,
                            attention_weights)
```

## 3. Unit tests
Run the operator tests:
```
python test_ms_deformable_attn_op.py
```
Example output from the upstream documentation:
```
*True check_forward_equal_with_paddle_float: max_abs_err 6.98e-10 max_rel_err 2.03e-07
*tensor1 True check_gradient_numerical(D=30)
*tensor2 True check_gradient_numerical(D=30)
*tensor3 True check_gradient_numerical(D=30)
*tensor1 True check_gradient_numerical(D=32)
*tensor2 True check_gradient_numerical(D=32)
*tensor3 True check_gradient_numerical(D=32)
*tensor1 True check_gradient_numerical(D=64)
*tensor2 True check_gradient_numerical(D=64)
*tensor3 True check_gradient_numerical(D=64)
*tensor1 True check_gradient_numerical(D=71)
*tensor2 True check_gradient_numerical(D=71)
*tensor3 True check_gradient_numerical(D=71)
*tensor1 True check_gradient_numerical(D=128)
*tensor2 True check_gradient_numerical(D=128)
*tensor3 True check_gradient_numerical(D=128)
*tensor1 True check_gradient_numerical(D=1024)
*tensor2 True check_gradient_numerical(D=1024)
*tensor3 True check_gradient_numerical(D=1024)
*tensor1 True check_gradient_numerical(D=1025)
*tensor2 True check_gradient_numerical(D=1025)
*tensor3 True check_gradient_numerical(D=1025)
*tensor1 True check_gradient_numerical(D=2048)
*tensor2 True check_gradient_numerical(D=2048)
*tensor3 True check_gradient_numerical(D=2048)
*tensor1 True check_gradient_numerical(D=3096)
*tensor2 True check_gradient_numerical(D=3096)
*tensor3 True check_gradient_numerical(D=3096)
```

<!-- Modified for the PFD-DETR distribution. -->
