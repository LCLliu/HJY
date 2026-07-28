# CoRePrune 三项代码核查

正式调用链：`run_unlearning.py:1088-1102` 通过 `METHOD_REGISTRY[args.unlearn_method]` 实例化并调用 `method.run()`；`unlearning/methods/__init__.py:16-31` 将 `"retain_prior_cf_lora_prune"` 注册到 `RetainPriorCFLoraPruneMethod`。以下只核查该正式实现，未采用 `geometry_prune.py`、`retain_prioritized_cbr_unlearning.py` 等其他方法文件作为结论依据。

## 1. 残留筛选方式
- 文件与行号：
  - `run_unlearning.py:193-199`：`rp_cf_top_m` 与 `rp_cf_residual_selection_mode`，默认 `topk`。
  - `unlearning/methods/retain_prior_cf_lora_prune.py:474-482`：读取筛选模式、Top-M 与阈值。
  - `unlearning/methods/retain_prior_cf_lora_prune.py:535-536`：计算 `raw` 与 `z_score`。
  - `unlearning/methods/retain_prior_cf_lora_prune.py:584-591`：实际排序与筛选。
- 实际代码：

```python
raw = float(score_original[key] - score_cf[key])
z_score = float((raw - mean) / (std + eps))
group_records.sort(key=lambda r: float(r.get("residual_z", 0.0)), reverse=True)
if selection_mode == "threshold":
    selected = [
        r for r in group_records
        if float(r.get("residual_z", 0.0)) > threshold
    ]
else:
    selected = group_records[:top_m] if top_m > 0 else []
```

- 单侧或双侧：阈值分支是单侧 `z > threshold`，没有 `abs(z)`。因此阈值模式只保留高于阈值的正向异常变化，不会保留负向显著变化。默认正式参数是 `topk`，按 `residual_z` 降序取 Top-M，也不是双侧；它偏向最大正向 `z`，但若某组全为非正值且 `top_m > 0`，仍会取该组最大的若干个 `z`。
- 是否使用 Top-M：是。默认 `rp_cf_residual_selection_mode="topk"`，`top_m` 来自 `rp_cf_top_m`，未设置时回落到 `residual_top_m`，见 `unlearning/methods/retain_prior_cf_lora_prune.py:1526-1530`。
- 数学公式：
  - 对候选物品 \(i\)：\(r_i=s_i(H)-s_i(H\setminus\{f\})\)。
  - 对同用户保留交互 null control \(q\)：\(d_{i,q}=s_i(H)-s_i(H\setminus\{q\})\)，\(\mu_i=\operatorname{mean}_q d_{i,q}\)，\(\sigma_i=\operatorname{std}_q d_{i,q}\)。
  - 代码中的残留标准分：\(z_i=(r_i-\mu_i)/(\sigma_i+\varepsilon)\)。
  - 默认筛选：\(S_g=\operatorname{TopM}_{i\in C_g}(z_i)\)。
  - 阈值模式：\(S_g=\{i\in C_g\mid z_i>\tau\}\)。
- 论文应采用的写法：按当前默认正式实现，应写成按 \(z_i\) 降序 Top-M；若描述阈值配置，应写 \(z_i>\tau\)。不应写 \(|z_i|>\tau\)，除非代码改成双侧筛选。

## 2. 保留支持聚合
- 文件与行号：
  - `run_unlearning.py:206-208`：`rp_cf_retain_aggregation` 默认 `mean_positive_delta`，可选 `mean_positive_delta`、`max_positive_delta`、`sum_positive_delta`。
  - `unlearning/methods/retain_prior_cf_lora_prune.py:644-686`：`calibrate_retain_support` 中删除单个保留交互后的候选得分变化与聚合调用。
  - `unlearning/methods/retain_prior_cf_lora_prune.py:718-744`：全局归一化并生成 `S_ret`、`W_unl`、`W_prot`。
  - `unlearning/methods/retain_prior_cf_lora_prune.py:1140-1150`：`aggregate_retain_support` 聚合实现。
  - `unlearning/methods/retain_prior_cf_lora_prune.py:1153-1170`：`_normalize_global_nonnegative` 归一化。
  - `unlearning/methods/retain_prior_cf_lora_prune.py:1544-1576`：保留交互采样依赖用户历史。
- 单个保留贡献的减法方向：`score_original - score_minus_retain`，即原历史候选得分减去删除某个保留交互后的候选得分，见 `unlearning/methods/retain_prior_cf_lora_prune.py:680`。
- 是否截断负值：是。`calibrate_retain_support` 在写入贡献时使用 `max(delta, 0.0)`；`aggregate_retain_support` 又对输入执行 `max(0.0, float(v))`。这是先截断再聚合。
- 聚合方式：默认求均值 `mean_positive_delta`；代码还支持最大值 `max_positive_delta` 和求和 `sum_positive_delta`。没有 Top-m 均值、`log1p`、`logsumexp`、加权求和。
- 是否归一化：是。聚合后的 `support_raw_values` 通过 `_normalize_global_nonnegative` 做全局非负 min-max 归一化，结果写入 `S_ret`。
- 关键代码片段，最多 15 行：

```python
aggregation = str(getattr(self.args, "rp_cf_retain_aggregation", "mean_positive_delta") or "mean_positive_delta")
delta = float(record["score_original"]) - float(scores["scores"].get(key, 0.0))
support_by_id[record["record_id"]].append(max(delta, 0.0))
support = self.aggregate_retain_support(contributions, aggregation)
s_ret_norm = self._normalize_global_nonnegative(support_raw_values)
w_unl = float(i_unl * math.exp(-gamma * s_ret))
w_prot = float((1.0 - w_unl) * _sigmoid(s_ret))
values = [max(0.0, float(v)) for v in contributions]
if mode == "max_positive_delta":
    return float(max(values))
if mode == "sum_positive_delta":
    return float(sum(values))
return float(np.mean(values))
```

- 数学公式：
  - 对组 \(g\)、候选 \(i\)、采样保留交互 \(q\in R_g\)：\(\delta_{i,q}=s_i(H_g)-s_i(H_g\setminus\{q\})\)。
  - 截断贡献：\(c_{i,q}=\max(\delta_{i,q},0)\)。
  - 默认 raw 保留支持：\(\widetilde S_i=\frac{1}{|R_g|}\sum_{q\in R_g}c_{i,q}\)，若无贡献则为 0。
  - 可选模式：`max_positive_delta` 为 \(\max_q c_{i,q}\)，`sum_positive_delta` 为 \(\sum_q c_{i,q}\)。
  - 全局非负归一化：令 \(a_i=\max(\widetilde S_i,0)\)，\(P=\{a_i\mid a_i>0\}\)。若 \(P=\varnothing\)，\(S_i=0\)；若 \(\max P\le \min P+10^{-12}\)，\(S_i=\mathbf 1[a_i>0]\)；否则 \(S_i=0\) 当 \(a_i\le0\)，否则 \(S_i=(a_i-\min P)/(\max P-\min P)\)。
  - 后续权重：\(W_{\mathrm{unl},i}=\operatorname{clip}(I_{\mathrm{unl},i}e^{-\gamma S_i},0,1)\)，\(W_{\mathrm{prot},i}=\operatorname{clip}((1-W_{\mathrm{unl},i})\sigma(S_i),0,1)\)。
- 是否受用户历史长度影响：受影响。`_sample_retain_interactions` 从 `remove_forget_item_from_history(history, forget_iid)` 得到用户可用保留历史，再最多采样 `requested` 个；若可用池小于请求数则全取。因此贡献个数和采样集合受用户历史中合格保留交互数量影响。默认均值不会把样本数线性累加到 raw support，但样本数会影响估计；`sum_positive_delta` 模式会直接随样本数变化。
- 论文应采用的写法：默认应写“删除单个保留交互后的正向得分下降贡献 \( \max(s_i(H)-s_i(H\setminus\{q\}),0) \) 的均值，再做全局非负 min-max 归一化”。只有在明确说明命令行改为 `max_positive_delta` 或 `sum_positive_delta` 时，才写最大值或求和。

## 3. LoRA 缩放系数
- 文件与行号：
  - `run_unlearning.py:575-587`：加载基础模型并用 `PeftModel.from_pretrained` 加载 LoRA adapter，LoRA 前向由 PEFT 模块处理。
  - `train_ranker.py:50-58`：训练时 `LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, ...)`。
  - `config.py:140-143`：默认 `lora_r=8`、`lora_alpha=32`。
  - `experiments/Llama-2-7b-hf/ml-100k/checkpoint-4400/adapter_config.json:10-14`：当前 checkpoint 配置 `lora_alpha=32`、`r=8`。
  - `/home/hjy/miniconda3/envs/llamarec/lib/python3.10/site-packages/peft/tuners/lora.py:75-83`：PEFT `LoraConfig` 字段 `r`、`lora_alpha`。
  - `/home/hjy/miniconda3/envs/llamarec/lib/python3.10/site-packages/peft/tuners/lora.py:124-155`：`LoraLayer.update_layer` 中保存 rank、alpha 并计算 scaling。
  - `/home/hjy/miniconda3/envs/llamarec/lib/python3.10/site-packages/peft/tuners/lora.py:1207-1230`：默认 4bit 路径 `Linear4bit.forward` 中实际 LoRA 前向增量。
  - `/home/hjy/miniconda3/envs/llamarec/lib/python3.10/site-packages/peft/tuners/lora.py:893-910`：非 4bit `Linear.forward` 中同样乘 `self.scaling`。
  - `unlearning/methods/retain_prior_cf_lora_prune.py:57-68`、`:85-115`、`:261-266`：项目剪枝掩码作用在 `lora_B` 输入或 `B` 列上。
- rank 变量：PEFT 使用 `r` 和 `self.r[adapter_name]`；项目收集 rank 时使用 `rank = int(A.shape[0])`，见 `unlearning/methods/retain_prior_cf_lora_prune.py:109`。当前 checkpoint `r=8`。
- alpha 变量：PEFT 使用 `lora_alpha` 和 `self.lora_alpha[adapter_name]`。当前 checkpoint `lora_alpha=32`。
- scaling 表达式：`self.scaling[adapter_name] = lora_alpha / r`，见 PEFT `lora.py:155`。不是 `alpha / sqrt(rank)`。
- 是否使用 RS-LoRA：否。当前 PEFT 0.5.0 的 `LoraConfig` 中未发现 `use_rslora` 字段，当前 checkpoint `adapter_config.json` 也没有 RS-LoRA 配置；实际 scaling 为 `alpha / r`。
- 掩码与缩放的先后关系：掩码先于缩放。项目通过 `lora_B` 的 forward pre-hook 把 `A` 的输出乘 `rank_mask`，见 `retain_prior_cf_lora_prune.py:57-68`；PEFT 随后在 forward 中对 `lora_B(lora_A(...))` 乘 `self.scaling`，见 PEFT `lora.py:1217-1222`。物化时 `layer.B.mul_(mask.view(1, -1))` 零掉 `B` 的列，也是在 PEFT scaling 之前改变 \(B\)。
- 实际代码表达式：

```python
self.r[adapter_name] = r
self.lora_alpha[adapter_name] = lora_alpha
self.scaling[adapter_name] = lora_alpha / r
output = (
    self.lora_B[self.active_adapter](
        self.lora_A[self.active_adapter](self.lora_dropout[self.active_adapter](x))
    ).to(expected_dtype)
    * self.scaling[self.active_adapter]
)
masked_hidden = hidden * rank_mask.to(device=hidden.device, dtype=hidden.dtype).view(*view_shape)
```

- 数学公式：
  - 未剪枝 LoRA 增量：\(\Delta W_\ell=(\alpha_\ell/r_\ell)B_\ell A_\ell\)。
  - 当前 rank mask 生效时：\(\Delta W_\ell(m)=(\alpha_\ell/r_\ell)B_\ell\operatorname{diag}(m_\ell)A_\ell\)。
  - 由于 `model.eval()` 后 dropout 不生效，正式核查中的推理前向可按上式写；PEFT 前向代码仍保留 `lora_dropout` 调用。
- 论文应采用的写法：如果论文写有效 LoRA 增量，应显式写 \(\alpha_\ell/r_\ell\)。对 CoRePrune 的 rank mask，应写 \((\alpha_\ell/r_\ell)B_\ell\operatorname{diag}(m_\ell)A_\ell\)。只有在正文明确定义 \(B_\ell A_\ell\) 已吸收 PEFT scaling 时，才可省略缩放系数。

## 4. 最终结论

| 检查项 | 代码实际实现 | 论文建议写法 | 是否一致 |
|---|---|---|---|
| 残留筛选 | 默认按 `residual_z` 降序 Top-M；阈值分支为 `z > threshold`；无 `abs(z)` | 默认写 \(S_g=\operatorname{TopM}(z_i)\)；阈值配置写 \(z_i>\tau\)，不写 \(|z_i|>\tau\) | 若论文写 Top-M/单侧阈值则一致；若写双侧 \(|z|\) 则不一致 |
| 保留支持聚合 | `score_original - score_minus_retain`，先 `max(delta,0)`，默认均值；支持最大值/求和；随后全局非负 min-max 归一化 | 写正向贡献均值 \( \operatorname{mean}_q\max(s_i(H)-s_i(H\setminus\{q\}),0) \) 后归一化；仅在改参数时写 max/sum | 若论文写默认均值正贡献加归一化则一致；若写 log1p/logsumexp/Top-m/加权求和则不一致 |
| LoRA 缩放系数 | PEFT 内部前向乘 `self.scaling = lora_alpha / r`；rank mask 作用在缩放前；无 RS-LoRA；无重复缩放 | 写 \(\Delta W_\ell=(\alpha_\ell/r_\ell)B_\ell\operatorname{diag}(m_\ell)A_\ell\) | 若论文显式写 \(\alpha/r\) 则一致；若写裸 \(BA\) 且未说明吸收缩放则不一致 |
