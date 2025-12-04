# 昇思模型开发挑战赛（S1赛季)--MultiModal赛题

## 注意力计算优化

+ 问题：原来的注意力计算需要大量的换轴操作。
+ 方案：使用FlashAttention融合算子就可以用BSND布局进行张量计算
+ 收益：规避掉首尾的两次换轴操作。减少计算时间，同时减少显存占用。

例如task1/mindnlp/llm/inference/janus_pro/janus/models/siglip_vit.py中的Attention操作：

```python
# 原版：做完投影后，就是手写的Attention操作，需要换轴为BNSD，做完scaled_dot_product_attention又需要换轴回去BSND
def forward(self, x: mindspore.Tensor) -> mindspore.Tensor:
    B, N, C = x.shape
    qkv = (
        self.qkv(x)
        .reshape(B, N, 3, self.num_heads, self.head_dim)
        .permute(2, 0, 3, 1, 4)#需要换轴
    )
    q, k, v = qkv.unbind(0)
    q, k = self.q_norm(q), self.k_norm(k)

    if self.fused_attn:
        x = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            is_causal=False,
            dropout_p=self.attn_drop.p if self.training else 0.0,
        )
    else:
        q = q * self.scale
        attn = q @ k.swapaxes(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ v

    x = x.swapaxes(1, 2).reshape(B, N, C)
    x = self.proj(x)
    x = self.proj_drop(x)
    return x
# 修改版：使用flash_attention_score，直接一步完成
def forward(self, x: mindspore.Tensor) -> mindspore.Tensor:
    B, S, C = x.shape
    qkv = (
        self.qkv(x)
        .reshape(B, S, 3, self.num_heads, self.head_dim)#不需要换轴
    )
    q, k, v = qkv.unbind(2)
    q, k = self.q_norm(q), self.k_norm(k)
    x = flash_attention_score(
        q,
        k,
        v,
        head_num=self.num_heads, input_layout='BSND', real_shift=None, padding_mask=None, attn_mask=None,
                        scalar_value=self.scale
    )
    x = x.reshape(B, S, C)
    x = self.proj(x)
    x = self.proj_drop(x)
    return x
```

## 融合算子替换

### rotary_pos_emb计算优化

+ 问题：手写的旋转位置编码太慢
+ 方案：使用融合算子rotary_position_embedding替换
+ 收益：减少计算时间

例如task1/mindnlp/mindnlp/transformers/models/llama/modeling_llama.py：

```python
def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    # 原版：
    # cos = cos.unsqueeze(unsqueeze_dim)
    # sin = sin.unsqueeze(unsqueeze_dim)
    # q_embed = (q * cos) + (rotate_half(q) * sin)
    # k_embed = (k * cos) + (rotate_half(k) * sin)

    q_embed = mindspore.ops.rotary_position_embedding(q, cos, sin, 0)   
    k_embed = mindspore.ops.rotary_position_embedding(k, cos, sin, 0)   
    return q_embed, k_embed
```

### RMSNorm计算优化

+ 问题：手写的RMSNorm太慢
+ 方案：使用融合算子rms_norm替换
+ 收益：减少计算时间

例如task1/mindnlp/mindnlp/transformers/models/qwen2_vl/modeling_qwen2_vl.py中的Qwen2RMSNorm：

```python
def forward(self, hidden_states):
    # 原版：
    # input_dtype = hidden_states.dtype
    # hidden_states = hidden_states.to(mindspore.float32)
    # variance = ops.mean(hidden_states.pow(2), -1, keepdim=True)
    # hidden_states = hidden_states * ops.rsqrt(variance + self.variance_epsilon)
    # return self.weight * hidden_states.to(input_dtype)
    return F.rms_norm(hidden_states, self.weight, self.variance_epsilon)
```

## Qwen2-VL 视觉语言模型

### Conv3D4_to_Conv3dv2算子替换

+ 问题：Conv3D4_to_Conv3dv2算子太耗时，会拖长NPU时间
+ 方案：使用mint的conv3d函数替换aclnnConvolution_BatchMatMulNd_BatchMatMulV2
+ 收益：减少prefill的计算时间

路径：task1/mindnlp/mindnlp/core/nn/modules/conv.py中的class Conv3d：

```python
def forward(self, input):
    if self.padding_mode != 'zeros':
        input = ops.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode)
    #使用使用mint的conv3d函数替换原来的mops.Conv3D
    output = mint_f.conv3d(input=input,
                   weight=self.weight,
                   stride=self.stride,
                   dilation=self.dilation,
                   groups=self.groups)
    if self.bias is not None:
        output = mops.bias_add(output, self.bias)
    return output
```

### rotary_pos_emb计算优化

+ 问题：decode会重复拼接cos和sin。
+ 方案：将cos和sin拼接计算提出到apply_rotary_pos_emb函数外，比如Qwen2VLModel中运行
+ 收益：减少计算时间

```python
def apply_multimodal_rotary_pos_emb(q, k, cos, sin, mrope_section, unsqueeze_dim=1):
    # 原版：
    # mrope_section = mrope_section * 2
    # cos = ops.cat([m[i % 3] for i, m in enumerate(ops.split(cos, mrope_section, dim=-1))], dim=-1).unsqueeze(
    #     unsqueeze_dim
    # )
    # sin = ops.cat([m[i % 3] for i, m in enumerate(ops.split(sin, mrope_section, dim=-1))], dim=-1).unsqueeze(
    #     unsqueeze_dim
    # )

    # q_embed = (q * cos) + (rotate_half(q) * sin)
    # k_embed = (k * cos) + (rotate_half(k) * sin)
    q_embed = mindspore.ops.rotary_position_embedding(q, cos, sin, 0)   
    k_embed = mindspore.ops.rotary_position_embedding(k, cos, sin, 0)
    return q_embed, k_embed
    return q_embed, k_embed
```



### rescale函数优化

+ 问题：uint8的numpy数组和float的字面量相乘，结果是float64的numpy数组。而后续用float64计算比较慢
+ 方案：做完这个操作后，就cast到float32
+ 收益：减少计算时间

```sh
import numpy as np
img = np.array([10, 20], dtype=np.uint8)
scale = 0.5          # Python float 字面量 → float64
out = img * scale
print(out.dtype)     # float64
```

## Janus-Pro 多模态模型

### self.tokenizer.vocab.get缓存优化

+ 问题：VLChatProcessor中的几个@property会调用self.tokenizer.vocab.get获取数据，这个操作非常耗时
+ 方案：对@property做缓存，只在第一次访问的时候使用self.tokenizer.vocab.get
+ 收益：大幅度减少prefill计算时间

路径：task1/mindnlp/llm/inference/janus_pro/janus/models/processing_vlm.py中的VLChatProcessor

```python
@property
def image_start_id(self):
    if self.my_image_start_id == None:
        self.my_image_start_id = self.tokenizer.vocab.get(self.image_start_tag)
    return self.my_image_start_id
```

## 其他可能的优化

+ 避免相同图片重复编码，对相同的图片做缓存。可以降低测试程序中的decode时间，但是会增加一些prefill时间
+ Janus-Pro VLMImageProcessor中的ms.dataset.vision.Resize优化，当前是使用CPU上做的，应该可以放在NPU上加速



## 最终收益

| model_name            | memory_reserved | memory_allocated | avg_prefill_latency | avg_decode_latency   |
| :-------------------- | :-------------- | :--------------- | :------------------ | :------------------- |
| Qwen2-VL-2B-Instruct  | 5.36870912      | 4.919692288      | 0.18102014064788818 | 0.04882878065109253  |
| deepseek-moe-16b-chat | 16.10612736     | 15.21473536      | 0.11808335781097412 | 0.029616401195526124 |


## 评测结果

| 评测指标        | 平均得分     |
| --------------- | ------------ |
| 峰值显存得分    | 133.3333     |
| Prefill时延得分 | 490.1253     |
| Decode时延得分  | 215.2412     |
| **总分**        | **279.5666** |

