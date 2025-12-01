# 昇思模型开发挑战赛（S1赛季)--MultiModal赛题

## 注意力计算优化

+ 问题：原来的注意力计算需要大量的换轴操作。
+ 方案：使用FlashAttention融合算子就可以用BSND布局进行张量计算
+ 收益：规避掉首尾的两次换轴操作。减少计算时间，同时减少显存占用。

## 融合算子替换

### rotary_pos_emb计算优化

+ 问题：手写的旋转位置编码太慢
+ 方案：使用融合算子rotary_position_embedding替换
+ 收益：减少计算时间

### RMSNorm计算优化

+ 问题：手写的RMSNorm太慢
+ 方案：使用融合算子rms_norm替换
+ 收益：减少计算时间

## Qwen2-VL 视觉语言模型

### Conv3D4_to_Conv3dv2算子替换

+ 问题：Conv3D4_to_Conv3dv2算子太耗时，会拖长NPU时间
+ 方案：使用mint的conv3d函数替换aclnnConvolution_BatchMatMulNd_BatchMatMulV2
+ 收益：减少prefill的计算时间

### rotary_pos_emb计算优化

+ 问题：decode会重复计算相同freqs的cos和sin。
+ 方案：将freqs的cos和sin计算提出到apply_rotary_pos_emb函数外
+ 收益：减少计算时间

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



## 其他可能的优化

+ 避免相同图片重复编码，对相同的图片做缓存。可以降低测试程序中的decode时间，但是会增加一些prefill时间
+ Janus-Pro VLMImageProcessor中的ms.dataset.vision.Resize优化，当前是使用CPU上做的，应该可以放在NPU上加速