import os.path

import clip
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.mixture import GaussianMixture

from datasets import get_specific_dataset
import os
os.environ["OMP_NUM_THREADS"] = '1'


class LoRALinear(torch.nn.Module):
    """A small LoRA adapter around an existing Linear layer."""

    def __init__(self, base_layer, rank=4, alpha=8.0, dropout=0.0):
        super().__init__()
        if not isinstance(base_layer, torch.nn.Linear):
            raise TypeError("LoRALinear can only wrap torch.nn.Linear.")
        if int(rank) <= 0:
            raise ValueError("rank must be positive.")

        self.base = base_layer
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / float(self.rank)
        self.dropout = torch.nn.Dropout(float(dropout)) if dropout and dropout > 0 else torch.nn.Identity()

        for param in self.base.parameters():
            param.requires_grad_(False)

        self.lora_A = torch.nn.Linear(self.base.in_features, self.rank, bias=False)
        self.lora_B = torch.nn.Linear(self.rank, self.base.out_features, bias=False)
        self.lora_A.to(device=self.base.weight.device, dtype=torch.float32)
        self.lora_B.to(device=self.base.weight.device, dtype=torch.float32)
        torch.nn.init.kaiming_uniform_(self.lora_A.weight, a=np.sqrt(5))
        torch.nn.init.zeros_(self.lora_B.weight)
    
    @property
    def weight(self):
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    def forward(self, x):
        base_out = self.base(x)
        lora_x = self.dropout(x).to(dtype=torch.float32)
        lora_out = self.lora_B(self.lora_A(lora_x)) * self.scaling
        return base_out + lora_out.to(dtype=base_out.dtype)


def freeze_module(module):
    for param in module.parameters():
        param.requires_grad_(False)
    return module


def _name_matches_keywords(name, target_keywords):
    if not target_keywords:
        return True
    return any(keyword in name for keyword in target_keywords)


def inject_lora_linear_modules(
    module,
    target_keywords=("visual",),
    rank=4,
    alpha=8.0,
    dropout=0.0,
    prefix="",
):
    """Replace matching Linear children with LoRALinear and return replaced names."""
    replaced = []
    for child_name, child in list(module.named_children()):
        full_name = f"{prefix}.{child_name}" if prefix else child_name
        if isinstance(child, LoRALinear):
            continue
        if "attn.out_proj" in full_name:
            continue
        if isinstance(child, torch.nn.Linear) and _name_matches_keywords(full_name, target_keywords):
            setattr(module, child_name, LoRALinear(child, rank=rank, alpha=alpha, dropout=dropout))
            replaced.append(full_name)
        else:
            replaced.extend(
                inject_lora_linear_modules(
                    child,
                    target_keywords=target_keywords,
                    rank=rank,
                    alpha=alpha,
                    dropout=dropout,
                    prefix=full_name,
                )
            )
    return replaced


def iter_lora_parameters(module):
    for submodule in module.modules():
        if isinstance(submodule, LoRALinear):
            yield from submodule.lora_A.parameters()
            yield from submodule.lora_B.parameters()


def collect_lora_state_dict(module):
    return {
        key: value.detach().cpu()
        for key, value in module.state_dict().items()
        if ".lora_A." in key or ".lora_B." in key
    }


def _as_numpy_indices(indices):
    if isinstance(indices, torch.Tensor):
        return indices.detach().cpu().numpy().astype(np.int64)
    return np.asarray(indices, dtype=np.int64)

def _initial_selection_checkpoint(selected_indices, fixed_indices, fixed_metrics):
    fixed_np = np.unique(_as_numpy_indices(fixed_indices))
    selected_np = np.unique(_as_numpy_indices(selected_indices))
    return {
        "kind": "fixed",
        "selected_indices": selected_np,
        "fixed_indices": fixed_np,
        "fixed_metrics": fixed_metrics,
    }


class CLIPNoiseDiscriminator(torch.nn.Module):
    """Binary clean/noisy head over an image feature and its noisy-label text feature."""

    def __init__(self, feature_dim, hidden_dim=512, dropout=0.1):
        super().__init__()
        pair_dim = int(feature_dim) * 4
        self.head = torch.nn.Sequential(
            torch.nn.Linear(pair_dim, int(hidden_dim)),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(float(dropout)),
            torch.nn.Linear(int(hidden_dim), 2),
        )

    def forward(self, image_features, text_features):
        image_features = torch.nan_to_num(image_features.float(), nan=0.0, posinf=0.0, neginf=0.0)
        text_features = torch.nan_to_num(text_features.float(), nan=0.0, posinf=0.0, neginf=0.0)
        image_features = F.normalize(image_features, dim=1)
        text_features = F.normalize(text_features, dim=1)
        pair_features = torch.cat(
            [
                image_features,
                text_features,
                image_features * text_features,
                torch.abs(image_features - text_features),
            ],
            dim=1,
        )
        return self.head(pair_features)


def _torch_device_from_args(args=None):
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _canonical_class_name(class_name):
    if isinstance(class_name, (list, tuple)):
        class_name = class_name[0]
    return str(class_name).lower()


def build_label_text_features(
    clip_model,
    class_names,
    suffix=".",
    use_cot=False,
    cot_suffix=" Let's think step by step.",
    device=None,
    template="a photo of a {}",
):
    device = device or next(clip_model.parameters()).device
    texts = []
    for class_name in class_names:
        text = template.format(_canonical_class_name(class_name))
        if suffix:
            text = text + str(suffix)
        if use_cot:
            text = text + str(cot_suffix)
        texts.append(text)

    tokenized = clip.tokenize(texts).to(device)
    with torch.no_grad():
        text_features = clip_model.encode_text(tokenized).float()
        text_features = torch.nan_to_num(text_features, nan=0.0, posinf=0.0, neginf=0.0)
        text_features = F.normalize(text_features, dim=1)
    return text_features


def _extract_images_labels_indices(batch):
    images, labels, indices = batch[0], batch[1], batch[2]
    if isinstance(images, (list, tuple)):
        images = images[0]
    return images, labels, indices


@torch.no_grad()
def score_lora_noise_discriminator(
    clip_model,
    discriminator,
    data_loader,
    text_features,
    device=None,
):
    device = device or next(clip_model.parameters()).device
    discriminator = discriminator.to(device)
    text_features = text_features.to(device)
    scores = torch.zeros(len(data_loader.dataset), dtype=torch.float32)

    clip_model.eval()
    discriminator.eval()
    for batch in data_loader:
        images, noisy_labels, indices = _extract_images_labels_indices(batch)
        images = images.to(device, non_blocking=True)
        noisy_labels = noisy_labels.long().to(device, non_blocking=True)
        image_features = clip_model.encode_image(images).float()
        image_features = torch.nan_to_num(image_features, nan=0.0, posinf=0.0, neginf=0.0)
        logits = discriminator(image_features, text_features[noisy_labels])
        logits = torch.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4)
        probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu()
        probs = torch.nan_to_num(probs, nan=0.0, posinf=1.0, neginf=0.0)
        scores[indices.long().cpu()] = probs
    return scores


def _pack_combined_selection_return(selected_indices, pred_fixed, all_labels, checkpoint=None, return_prompt=False):
    selected_np = _as_numpy_indices(selected_indices)
    if isinstance(all_labels, torch.Tensor):
        label_values = all_labels.detach().cpu().numpy()
    else:
        label_values = np.asarray(all_labels)
    selected_labels = label_values[selected_np]
    result = (selected_np, pred_fixed.detach().cpu(), selected_labels)
    if return_prompt:
        result = result + (checkpoint,)
    return result


def _contrastive_loss(h_high, h_low, mode="cosine"):
    """
    h_high/h_low: [B, D] float tensors (already on same device)
    Push them apart: distance >= margin
    """
    import torch
    import torch.nn.functional as F

    h_high = torch.nan_to_num(h_high.float(), nan=0.0, posinf=0.0, neginf=0.0)
    h_low  = torch.nan_to_num(h_low.float(),  nan=0.0, posinf=0.0, neginf=0.0)

    if mode == "euclidean":
        distances = torch.norm(h_high - h_low, p=2, dim=-1)
    elif mode == "cosine":
        h_high = F.normalize(h_high, dim=-1)
        h_low  = F.normalize(h_low, dim=-1)
        distances = torch.cosine_similarity(h_high, h_low, dim=-1)
    else:
        raise ValueError(f"Unsupported contrastive mode: {mode}")

    # loss = torch.clamp(margin - distances, min=0).mean()
    loss = torch.clamp(distances, min=0).mean()
    return loss


def get_score(prediction, labels, mode='celoss'):
    """
    prediction: torch tensor, shape [N, C], probabilities (should be in (0,1))
    labels: torch tensor, shape [N]
    """
    import numpy as np
    import torch

    num_classes = len(np.unique(labels))

    # ---- make log safe: avoid log(0) => -inf ----
    pred_safe = prediction.clamp(min=1e-12)

    if mode == 'celoss':
        loss = torch.log(pred_safe)
        score = torch.gather(loss, 1, labels.view(-1, 1)).squeeze()

    elif mode == 'perclass_celoss':
        loss = torch.log(pred_safe)
        score = torch.gather(loss, 1, labels.view(-1, 1)).squeeze()

        id_by_label = [np.where(labels.cpu().numpy() == i)[0] for i in range(num_classes)]
        for ids in id_by_label:
            if len(ids) == 0:
                continue
            s = score[ids]
            s_min = s.min()
            s_max = s.max()
            denom = (s_max - s_min)

            # if denom == 0, this class is constant -> set to zeros (or keep as-is)
            if torch.isfinite(denom) and denom > 0:
                score[ids] = (s - s_min) / (denom + 1e-12)
            else:
                score[ids] = 0.0

    else:  # consistency
        vote_y = torch.gather(prediction, 1, labels.view(-1, 1)).squeeze()
        vote_max = prediction.max(dim=1)[0]
        score = vote_y / (vote_max + 1e-12)

    # ---- final safety: remove any nan/inf just in case ----
    score = torch.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    return score


'''
multi_class：分类方式选择参数，str类型，可选参数为ovr和multinomial，默认为ovr。ovr即前面提到的one-vs-rest(OvR)，而multinomial即前面
提到的many-vs-many(MvM)。如果是二元逻辑回归，ovr和multinomial并没有任何区别，区别主要在多元逻辑回归上。
OvR和MvM有什么不同*？*
OvR的思想很简单，无论你是多少元逻辑回归，我们都可以看做二元逻辑回归。具体做法是，对于第K类的分类决策，我们把所有第K类的样本作为正例，除了第K类
样本以外的所有样本都作为负例，然后在上面做二元逻辑回归，得到第K类的分类模型。其他类的分类模型获得以此类推。
而MvM则相对复杂，这里举MvM的特例one-vs-one(OvO)作讲解。如果模型有T类，我们每次在所有的T类样本里面选择两类样本出来，不妨记为T1类和T2类，
把所有的输出为T1和T2的样本放在一起，把T1作为正例，T2作为负例，进行二元逻辑回归，得到模型参数。我们一共需要T(T-1)/2次分类。
可以看出OvR相对简单，但分类效果相对略差（这里指大多数样本分布情况，某些样本分布下OvR可能更好）。而MvM分类相对精确，但是分类速度没有OvR快。
如果选择了ovr，则4种损失函数的优化方法liblinear，newton-cg,lbfgs和sag都可以选择。但是如果选择了multinomial,则只能选择newton-cg, lbfgs和sag了。

class_weight：用于标示分类模型中各种类型的权重，可以是一个字典或者’balanced’字符串，默认为不输入，也就是不考虑权重，即为None。
如果选择输入的话，可以选择balanced让类库自己计算类型权重，或者自己输入各个类型的权重。举个例子，比如对于0,1的二元模型，我们可以定义
class_weight={0:0.9,1:0.1}，这样类型0的权重为90%，而类型1的权重为10%。如果class_weight选择balanced，那么类库会根据训练样本量来
计算权重。某种类型样本量越多，则权重越低，样本量越少，则权重越高。当class_weight为balanced时，类权重计算方法如下：
n_samples / (n_classes * np.bincount(y))。n_samples为样本数，n_classes为类别数量，np.bincount(y)会输出每个类的样本数，
例如y=[1,0,0,1,1],则np.bincount(y)=[2,3]。

那么class_weight有什么作用呢？
在分类模型中，我们经常会遇到两类问题：
第一种是误分类的代价很高。比如对合法用户和非法用户进行分类，将非法用户分类为合法用户的代价很高，我们宁愿将合法用户分类为非法用户，
这时可以人工再甄别，但是却不愿将非法用户分类为合法用户。这时，我们可以适当提高非法用户的权重。

第二种是样本是高度失衡的，比如我们有合法用户和非法用户的二元样本数据10000条，里面合法用户有9995条，非法用户只有5条，如果我们不考虑权重，
则我们可以将所有的测试集都预测为合法用户，这样预测准确率理论上有99.95%，但是却没有任何意义。这时，我们可以选择balanced，让类库自动
提高非法用户样本的权重。提高了某种分类的权重，相比不考虑权重，会有更多的样本分类划分到高权重的类别，从而可以解决上面两类问题。
'''
# def logistic_regression(features, labels):
#     # by default, we use weighted logistic regression to counter for the class imbalance
#     classifier = LogisticRegression(random_state=0, max_iter=10000,
#                                     class_weight='balanced').fit(features.cpu(), labels)  # .cpu())
#     # generate probability estimates
#     prediction = torch.tensor(classifier.predict_proba(features.cpu()))
#     return prediction


import types
def _ensure_encode_text_learn(clip_model):
    if hasattr(clip_model, "encode_text_learn"):
        return clip_model

    def encode_text_learn(self, prompts, tokenized_prompts):
        # --- robust cast dtype: match transformer weights dtype (fix fp16/fp32 mismatch) ---
        try:
            cast_dtype = self.transformer.resblocks[0].attn.in_proj_weight.dtype
        except Exception:
            # fallback: mimic AdaptCLIP get_cast_dtype idea
            try:
                cast_dtype = self.transformer.resblocks[0].mlp.c_fc.weight.dtype
            except Exception:
                cast_dtype = prompts.dtype

        x = prompts.to(cast_dtype) + self.positional_embedding.to(cast_dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)

        x = x.permute(1, 0, 2)      # LND -> NLD
        x = self.ln_final(x).float()  # ln 输出转 fp32
        eot = tokenized_prompts.argmax(dim=-1)
        x = x[torch.arange(x.shape[0], device=x.device), eot]
        x = x @ self.text_projection.float()
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x = x.clamp(min=-1e4, max=1e4)  # 防止爆
        return x

    clip_model.encode_text_learn = types.MethodType(encode_text_learn, clip_model)
    return clip_model


class _ClassPromptLearner(torch.nn.Module):
    """
    AdaptCLIP / CoOp-style class prompt learner.

    Two-branch design:
      - fixed branch: uses your detailed_features templates (handled elsewhere)
      - learnable branch: ONLY learns ctx tokens inserted between template prefix and class name

    Token layout:
      [SOS] + prefix("a photo of a") + [CTX x n_ctx] + class_name + suffix(+CoT) + [EOS] + padding
    """
    def __init__(
        self,
        clip_model,
        class_names,
        suffix=".",
        n_ctx=8,
        template="a photo of a {}",
        use_cot=False,
        cot_suffix=" Let's think step by step."
    ):
        super().__init__()
        import clip
        import torch

        self.clip_model = clip_model
        self.class_names = [cn[0] if isinstance(cn, list) else cn for cn in class_names]
        self.suffix = suffix
        self.n_ctx = int(n_ctx)
        self.template = template
        self.use_cot = bool(use_cot)
        self.cot_suffix = cot_suffix
        ctx_dim = clip_model.ln_final.weight.shape[0]

        # make ctx dtype consistent with CLIP text transformer weights
        try:
            dtype = clip_model.transformer.resblocks[0].attn.in_proj_weight.dtype
        except Exception:
            dtype = getattr(clip_model, "dtype", torch.float16)

        ctx = torch.empty(len(self.class_names), self.n_ctx, ctx_dim, dtype=dtype)
        torch.nn.init.normal_(ctx, std=0.02)
        self.ctx = torch.nn.Parameter(ctx)
        self.clip_dtype = dtype

        # ---- build learnable-branch texts (NO detailed_features here!) ----
        # prefix_text: template without class placeholder
        prefix_text = self.template.format("").replace("  ", " ").strip()
        # put X tokens between prefix and class name
        ctx_placeholders = " ".join(["X"] * self.n_ctx)

        texts = []
        for cn in self.class_names:
            cn = cn.lower()
            # IMPORTANT: ctx placeholders are between prefix and class name
            # "a photo of a X X X ... dog."
            # text = f"{prefix_text} {cn} {ctx_placeholders}{self.suffix}"
            text = f"{ctx_placeholders} {cn}{self.suffix}"
            if self.use_cot:
                text = text + self.cot_suffix
            texts.append(text)

        # ---- tokenize & move to same device as CLIP ----
        tokenized = clip.tokenize(texts)  # [K,77] on CPU
        device = next(clip_model.parameters()).device
        tokenized = tokenized.to(device)
        self.register_buffer("tokenized_prompts", tokenized)

        # ---- get token embeddings from CLIP ----
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized).type(dtype)  # [K,77,dim]

        # ---- compute prefix_len correctly (exclude EOS) ----
        # In OpenAI CLIP, EOS token id is 49407.
        eos_id = 49407
        tokenized_prefix = clip.tokenize([prefix_text]).to(device)  # [1,77]
        # position of EOS in prefix-only sequence
        eos_pos = (tokenized_prefix[0] == eos_id).nonzero(as_tuple=False)
        if eos_pos.numel() == 0:
            # fallback: use first zero (padding) position
            prefix_len = int((tokenized_prefix[0] != 0).sum().item())
        else:
            prefix_len = int(eos_pos[0].item())  # exclude EOS itself

        self.prefix_len = prefix_len

        # ---- slice token prefix/suffix around ctx span ----
        # prefix: [SOS + prefix_text tokens]
        self.register_buffer("token_prefix", embedding[:, :prefix_len, :])  # [K, prefix_len, dim]
        # suffix: [class_name + suffix + (CoT) + EOS + padding]
        self.register_buffer("token_suffix", embedding[:, prefix_len + self.n_ctx :, :])  # [K, 77-prefix_len-n_ctx, dim]

    def forward(self):
        ctx = self.ctx.to(self.token_prefix.dtype)  # cast to clip dtype for forward
        prompts = torch.cat([self.token_prefix, ctx, self.token_suffix], dim=1)
        return prompts, self.tokenized_prompts


def _train_prompt_learner(
    clip_model,
    prompt_learner,
    train_loader,
    steps=2000,
    lr=5e-4,
    wd=0.0,
    temp=0.07,
    # ===== new for robust prompt learning =====
    contrastive_weight=1.,
):
    """
    Warm-up training for learnable prompt tokens.
    Now: CE(clean label) + contrastive loss (clean vs simulated-noise label) to reduce FN.
    Only prompt_learner params trainable; CLIP frozen.

    - CE forces correct class alignment (like before)
    - Contrastive: for each clean sample, sample a wrong label y_noise != y
      and push text_feat[y] far from text_feat[y_noise] with margin
      (prompt robustness against label noise)
    """
    import torch
    import torch.nn.functional as F

    device = next(clip_model.parameters()).device
    clip_dtype = getattr(clip_model, "dtype", torch.float16)

    clip_model.eval()
    prompt_learner.train()

    # freeze clip
    for p in clip_model.parameters():
        p.requires_grad_(False)
    for p in prompt_learner.parameters():
        p.requires_grad_(True)

    # opt = torch.optim.AdamW(prompt_learner.parameters(), lr=lr, weight_decay=wd)
    opt = torch.optim.AdamW(
        prompt_learner.parameters(),
        lr=lr,
        weight_decay=wd,
        betas=(0.9, 0.98),
        eps=1e-6
    )

    it = iter(train_loader)
    for step in range(int(steps)):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(train_loader)
            batch = next(it)

        # your loader: (images, labels, index)
        images, labels = batch[0], batch[1]
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # ---- image features (frozen) ----
        with torch.no_grad():
            img_feat = clip_model.encode_image(images)
            img_feat = torch.nan_to_num(img_feat, nan=0.0, posinf=0.0, neginf=0.0).float()
            img_feat = F.normalize(img_feat, dim=1)

        # ---- text features from learnable prompts ----
        prompts_emb, tokenized = prompt_learner()
        # make sure dtype/device consistent with CLIP text tower
        prompts_emb = prompts_emb.to(device=device, dtype=clip_dtype)
        tokenized   = tokenized.to(device=device)

        txt_feat = clip_model.encode_text_learn(prompts_emb, tokenized)
        txt_feat = torch.nan_to_num(txt_feat, nan=0.0, posinf=0.0, neginf=0.0).float()
        txt_feat = F.normalize(txt_feat, dim=1)  # [K, D]

        # ---- CE on clean labels ----
        logits = (img_feat @ txt_feat.t()) / float(temp)
        logits = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)
        ce_loss = F.cross_entropy(logits, labels)

        # ---- simulated noisy label: y_noise != y ----
        num_classes = txt_feat.shape[0]
        if num_classes <= 1:
            con_loss = torch.zeros_like(ce_loss)
            print("num class less than one")
        else:
            B = labels.size(0)
            # pos similarity: sim(img_i, txt_yi)
            sim_pos = (img_feat * txt_feat[labels]).sum(dim=1)  # [B] (cos sim because normalized)
            neg_mode='random'
            neg_topk = 1
            infonce_tau = 0.07
            if neg_mode == "random":
                with torch.no_grad():
                    logits_neg = logits.clone()
                    logits_neg[torch.arange(labels.size(0), device=device), labels] = -1e9  # 禁掉正确类
                    y_noise = logits_neg.argmax(dim=1)  # 最像的错误类
                    # y_noise = torch.topk(logits_neg, k=neg_topk, dim=1).indices
                # "high quality" = text feat of y ; "low quality" = text feat of y_noise
                h_high = txt_feat[labels]  # [B, D]
                h_low = txt_feat[y_noise]  # [B, D]
                # B, N, D = h_low.shape
                # h_high_rep = h_high[:, None, :].expand(B, N, D).reshape(B * N, D)
                # h_low_flat = h_low.reshape(B * N, D)
                con_loss = _contrastive_loss(
                    h_high, h_low,
                    mode='euclidean'
                )
                # con_loss = _contrastive_loss(
                #     h_high_rep, h_low_flat,
                #     margin=contrastive_margin,
                #     mode='euclidean'
                # )

            else:
                def _infonce_pair_loss(sim_pos, sim_neg, tau=0.07):
                    """
                    sim_pos, sim_neg: [B] similarity scores (higher = more similar)
                    InfoNCE for 1 positive vs 1 negative:
                        L = -log exp(sim_pos/tau) / (exp(sim_pos/tau) + exp(sim_neg/tau))
                          = softplus((sim_neg - sim_pos)/tau)
                    """
                    import torch.nn.functional as F
                    return F.softplus((sim_neg - sim_pos) / float(tau)).mean()

                def _infonce_multi_neg_loss(sim_pos, sim_negs, tau=0.07):
                    """
                    sim_pos: [B]
                    sim_negs: [B, M]  M个负样本
                    L = -log exp(pos/tau) / (exp(pos/tau) + sum_m exp(neg_m/tau))
                    """
                    import torch
                    pos = (sim_pos / float(tau)).unsqueeze(1)  # [B,1]
                    neg = (sim_negs / float(tau))  # [B,M]
                    logits = torch.cat([pos, neg], dim=1)  # [B,1+M]
                    # label=0 表示第一列是正样本
                    import torch.nn.functional as F
                    return F.cross_entropy(logits, torch.zeros(logits.size(0), dtype=torch.long, device=logits.device))

                # hard negative: choose top-k most confusing wrong classes by logits
                with torch.no_grad():
                    logits_neg = logits.detach().clone()
                    logits_neg[torch.arange(B, device=device), labels] = -1e9

                    if int(neg_topk) <= 1:
                        y_noise = logits_neg.argmax(dim=1)  # [B]
                        sim_neg = (img_feat * txt_feat[y_noise]).sum(dim=1)
                        con_loss = _infonce_pair_loss(sim_pos, sim_neg, tau=infonce_tau)
                    else:
                        topk = min(int(neg_topk), num_classes - 1)
                        y_noises = torch.topk(logits_neg, k=topk, dim=1).indices  # [B,topk]
                        # compute sim for each negative
                        sim_negs = torch.einsum("bd,bkd->bk", img_feat, txt_feat[y_noises])  # [B,topk]
                        con_loss = _infonce_multi_neg_loss(sim_pos, sim_negs, tau=infonce_tau)

        loss = ce_loss + contrastive_weight * con_loss

        # ---- optimize ----
        if (not torch.isfinite(loss)) or torch.isnan(loss):
            print("[prompt-warmup] loss is NaN/Inf, abort warm-up")
            break

        opt.zero_grad(set_to_none=True)
        loss.backward()

        # --- new: skip step if any grad is NaN/Inf ---
        bad_grad = False
        for p in prompt_learner.parameters():
            if p.grad is not None and (not torch.isfinite(p.grad).all()):
                bad_grad = True
                break
        if bad_grad:
            print("[prompt-warmup] non-finite grad, skip step")
            opt.zero_grad(set_to_none=True)
            continue

        torch.nn.utils.clip_grad_norm_(prompt_learner.parameters(), max_norm=0.1)  # 从 1.0 改到 0.1 更稳
        opt.step()

        # optional debug
        if step % 50 == 0:
            print(f"[prompt-warmup] step={step} ce={ce_loss.item():.4f} con={con_loss.item():.4f} total={loss.item():.4f}")


def compute_selection_metrics(select_idx, is_clean_gt, labels=None, num_classes=None):
    """
    select_idx: 1D array-like, 被选中的样本 index
    is_clean_gt: torch.Tensor [N], 1=clean, 0=noisy
    labels: torch.Tensor [N], 多分类标签 (0~C-1)，可选
    num_classes: int，可选

    return: dict
      - overall metrics
      - confusion matrix
      - per_class_metrics (if labels is not None)
    """
    import numpy as np

    select_idx = np.asarray(select_idx)
    N = len(is_clean_gt)

    # ---------- binary prediction: selected or not ----------
    y_pred = np.zeros(N, dtype=int)
    y_pred[select_idx] = 1
    y_true = is_clean_gt.cpu().numpy()

    TP = ((y_pred == 1) & (y_true == 1)).sum()
    FP = ((y_pred == 1) & (y_true == 0)).sum()
    FN = ((y_pred == 0) & (y_true == 1)).sum()
    TN = ((y_pred == 0) & (y_true == 0)).sum()

    precision = TP / (TP + FP + 1e-12)
    recall    = TP / (TP + FN + 1e-12)
    f1        = 2 * precision * recall / (precision + recall + 1e-12)
    acc       = (TP + TN) / (TP + FP + FN + TN + 1e-12)

    results = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": acc,
        "selected_ratio": y_pred.mean(),
        "confusion_matrix": np.array([[TN, FP],
                                      [FN, TP]], dtype=int)
    }

    # ---------- per-class metrics ----------
    if labels is not None:
        labels_np = labels.cpu().numpy()
        if num_classes is None:
            num_classes = int(labels_np.max()) + 1

        per_class = {}

        for c in range(num_classes):
            cls_mask = (labels_np == c)

            # ground truth clean in this class
            clean_c = (y_true == 1) & cls_mask
            n_clean_c = clean_c.sum()

            # selected samples in this class
            selected_c = (y_pred == 1) & cls_mask
            n_selected_c = selected_c.sum()

            # true clean & selected
            tp_c = (selected_c & clean_c).sum()
            fp_c = (selected_c & (~clean_c)).sum()
            fn_c = ((~selected_c) & clean_c).sum()

            prec_c = tp_c / (tp_c + fp_c + 1e-12)
            rec_c  = tp_c / (tp_c + fn_c + 1e-12)
            f1_c   = 2 * prec_c * rec_c / (prec_c + rec_c + 1e-12)

            per_class[c] = {
                "selected": int(n_selected_c),
                "clean_selected": int(tp_c),
                "clean_total": int(n_clean_c),
                "precision": float(prec_c),
                "recall": float(rec_c),
                "f1": float(f1_c),
            }

        results["per_class_metrics"] = per_class
        print("len of selected idx:", len(select_idx))

    return results

def write_selection_metrics_to_file(metrics, filepath, title=None,epoch=-1):
    """
    metrics: compute_selection_metrics 返回的 dict
    filepath: 写入文件路径
    title: 可选标题（如 FIXED PROMPT / LEARNED PROMPT）
    """
    dirpath = os.path.dirname(filepath)
    if dirpath != "" and (not os.path.exists(dirpath)):
        os.makedirs(dirpath, exist_ok=True)
    with open(filepath, "a") as f:
        if title is not None:
            f.write(f"\n===== {title} =====\n")

        # ---- overall ----
        f.write(
            f"Epoch: {epoch}\t"
            f"[Overall] "
            f"Precision={metrics['precision']:.4f} | "
            f"Recall={metrics['recall']:.4f} | "
            f"F1={metrics['f1']:.4f} | "
            f"Acc={metrics['accuracy']:.4f} | "
            f"SelectRatio={metrics['selected_ratio']:.4f}\n"
        )

        # ---- confusion matrix ----
        cm = metrics["confusion_matrix"]
        f.write("Confusion Matrix [[TN, FP],[FN, TP]]:\n")
        f.write(f"{cm}\n")

        # ---- per-class ----
        if "per_class_metrics" in metrics:
            f.write("\n[Per-Class Metrics]\n")
            for c, m in metrics["per_class_metrics"].items():
                f.write(
                    f"Class {c:02d}: "
                    f"selected={m['selected']:5d} | "
                    f"clean_selected={m['clean_selected']:5d}/{m['clean_total']:5d} | "
                    f"precision={m['precision']:.4f} | "
                    f"recall={m['recall']:.4f} | "
                    f"f1={m['f1']:.4f}\n"
                )

        f.write("\n")

# ---------------- logistic regression (same as your original) ----------------
def logistic_regression(features, labels):
    clf = LogisticRegression(random_state=0, max_iter=10000,
                             class_weight="balanced").fit(features.cpu(), labels)
    return torch.tensor(clf.predict_proba(features.cpu()))

def _maybe_save_prompt(prompt_learner, save_path: str):
    """
    save_path: e.g.  {dataset}/{run_path}/prompt_from_combined_selection.pth
    """
    if prompt_learner is None:
        return
    ckpt = {
        "prompt_state_dict": prompt_learner.state_dict(),
        # 记录一下关键超参，方便你 debug / 防止结构不一致
        "n_ctx": getattr(prompt_learner, "n_ctx", None),
        "use_cot": getattr(prompt_learner, "use_cot", None),
        "cot_suffix": getattr(prompt_learner, "cot_suffix", None),
        "suffix": getattr(prompt_learner, "suffix", None),
        "template": getattr(prompt_learner, "template", None),
    }
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(ckpt, save_path)


def _save_selection_checkpoint(checkpoint, save_path):
    if checkpoint is None or not save_path:
        return
    dirpath = os.path.dirname(save_path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
    torch.save(checkpoint, save_path)


def fit_prompt_on_clean_subset_and_predict(
    args,
    pred_fixed,
    clean_indices,
    pseduo_labels=None,
    init_prompt_ckpt=None,
    strict_load=False,
):
    """
    Backward-compatible hook used by older experiments in main_cifar_new.py.

    The current pipeline performs LoRA refinement inside combined_selection.
    If older code calls this hook after stage 2, keep the run alive by returning
    the provided clean subset and checkpoint unchanged.
    """
    selected = np.unique(_as_numpy_indices(clean_indices))
    checkpoint = init_prompt_ckpt or {
        "kind": "clean_subset_passthrough",
        "selected_indices": selected,
    }
    return selected, None, checkpoint

def combined_selection(args, sed_args, pseduo_labels=None, return_prompt=False, save_prompt=False):
    """
    Patch version:
    - Keeps your original 4-score + per-class GMM/threshold + intersection logic fileciteturn10file4
    - Upgrades CLIP prompt to: fixed | learnable | hybrid, and enables *training* learnable prompts
      via encode_text_learn (AdaptCLIP-style). fileciteturn10file1
    """
    import clip
    import numpy as np
    import torch
    import torch.nn.functional as F
    from sklearn.mixture import GaussianMixture
    from sklearn.linear_model import LogisticRegression
    from datasets import get_specific_dataset
    metrics_file = os.path.join(
        args.dataset,
        args.run_path,
        "selection_metrics.txt"
    )
    # ---------------- defaults (new args) ----------------
    # prompt mode
    prompt_mode = getattr(args, "clip_prompt_mode", "hybrid")   # fixed|learnable|hybrid
    alpha = float(getattr(args, "clip_prompt_alpha", 0.5))      # hybrid weight for learnable
    n_ctx = int(getattr(args, "clip_prompt_len", 8))
    use_cot = bool(getattr(args, "clip_use_cot", False))
    cot_suffix = getattr(args, "clip_cot_suffix", " Let's think step by step.")

    # prompt warm-up
    train_steps = int(getattr(args, "clip_prompt_train_steps", 0))   # 0 means no training
    train_lr = float(getattr(args, "clip_prompt_train_lr", 5e-4))
    train_wd = float(getattr(args, "clip_prompt_train_wd", 0.0))
    train_bs = int(getattr(args, "clip_prompt_train_bs", 256))
    train_topk = int(getattr(args, "clip_prompt_train_topk", 200))   # per-class topk from fixed-prompt confidence
    temperature = float(getattr(args, "clip_temperature", 0.07))
    device = _torch_device_from_args(args)
    num_workers = int(getattr(args, "clip_num_workers", 8))

    # ---------------- load CLIP (patched to support encode_text_learn) ----------------
    if args.model == 'small':
        clip_model, preprocess = clip.load("ViT-B/32", device=device)
    elif args.model == 'tiny':
        clip_model, preprocess = clip.load("RN50", device=device)
    else:
        clip_model, preprocess = clip.load("ViT-L/14@336px", device=device)

    clip_model = _ensure_encode_text_learn(clip_model).to(device).eval()

    # ---------------- load dataset ----------------
    if pseduo_labels is not None:
        clip_data = get_specific_dataset(sed_args, preprocess)
        clip_data.label = pseduo_labels
    else:
        clip_data = get_specific_dataset(sed_args, preprocess)

    num_classes = clip_data.num_classes
    # detailed_features = clip_data.detailed_features
    class_names = clip_data.class_names
    suffix = clip_data.suffix

    clip_loader = torch.utils.data.DataLoader(
        clip_data,
        batch_size=int(getattr(args, "clip_eval_batch_size", 1024)),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )
    all_labels_tensor = torch.tensor(clip_data.label, dtype=torch.long)
    all_labels_np = all_labels_tensor.numpy()
    all_features = []

    # === GT clean labels (for metrics only) ===
    # dataloader_cifar.py 里定义的
    clean_labels = torch.tensor(clip_data.clean_label, dtype=torch.long)

    # 二分类 GT：1=干净样本，0=噪声样本
    if args.dataset == 'red_imagenet':
        is_clean_gt = torch.tensor(clip_data.is_clean_gt)
    else:
        is_clean_gt = (all_labels_tensor == clean_labels).long()  # shape [N]

    with torch.no_grad():
        for data, label, index in clip_loader:
            image_features = clip_model.encode_image(data.to(device, non_blocking=True)).float()
            all_features.append(image_features)

    all_features = torch.cat(all_features, dim=0).detach()
    all_features = F.normalize(all_features, dim=1)  # N x d

    # ---------------- fixed prompt prediction (your original) ---------------- fileciteturn10file0
    def fixed_prediction():
        # text_tokens = [[clip.tokenize(
        #     (f"a photo of a {(class_names[i][0] if isinstance(class_names[0], list) else class_names[i]).lower()}  {j}"
        #      + suffix
        #      + (cot_suffix if use_cot else ""))
        # ).cuda() for j in detailed_features[i]] for i in range(num_classes)]

        text_tokens = [[clip.tokenize(
            (f"a photo of a {(class_names[i][0] if isinstance(class_names[0], list) else class_names[i]).lower()} "
             + suffix
             + (cot_suffix if use_cot else ""))
        ).to(device) ] for i in range(num_classes)]

        text_features = [[clip_model.encode_text(text_token_i).float().detach()
                          for text_token_i in text_token] for text_token in text_tokens]

        similarity = torch.zeros(len(all_features), num_classes, device=all_features.device)
        with torch.no_grad():
            for i in range(num_classes):
                text_features_i = torch.cat(text_features[i], dim=0)  # NumPrompts x d
                text_features_i = F.normalize(text_features_i, dim=1)
                sim_ip = torch.einsum('ac, bc->ab', all_features, text_features_i)
                similarity[:, i] = torch.exp(sim_ip / temperature).sum(1)

        pred = (similarity / similarity.sum(1, keepdim=True)).detach()
        return pred

    pred_fixed = fixed_prediction()
    # Option 2: logistic regression with only visual features, 公式8
    prediction_lr = logistic_regression(all_features.cpu(), all_labels_tensor)
    CLIP_similarityprob_loss = get_score(pred_fixed.cpu(), all_labels_tensor, 'perclass_celoss')

    # Extract sample selection scores.
    CLIP_similarityprob_consistency = get_score(pred_fixed.cpu(), all_labels_tensor, 'consistency')
    CLIP_visuallr_loss = get_score(prediction_lr, all_labels_tensor, 'perclass_celoss')
    CLIP_visuallr_consistency = get_score(prediction_lr, all_labels_tensor, 'consistency')

    #  save scores
    method_score = [
        CLIP_similarityprob_loss, CLIP_similarityprob_consistency,
        CLIP_visuallr_loss, CLIP_visuallr_consistency]

    method_name = [
        'zeroshot_perclassgmm', 'zeroshot_consistency',
        'visuallr_perclassgmm', 'visuallr_consistency']

    # take intersection by default
    types = ['loss', 'consistency',
             'loss', 'consistency']

    id_by_label = [np.where(all_labels_np == i)[0] for i in range(num_classes)]
    missing = [k for k in range(num_classes) if len(id_by_label[k]) == 0]
    if len(missing) > 0:
        print(f"[warn] missing classes in labels: {missing[:20]} ... total={len(missing)}")

    select_i = []
    # loss-based scores ---> GMM
    for i, score in enumerate(method_score):
        clean_id_all = []
        for k in range(num_classes):  # per-class sample selection
            if types[i] == 'loss':
                gmm = GaussianMixture(2)
                gmm.fit(score[id_by_label[k]].reshape(-1, 1))
                # -loss 均值最小就是loss 均值最大的分量
                prob = gmm.predict_proba(score[id_by_label[k]].reshape(-1, 1))[:, gmm.means_.argmax()]
                clean_id = id_by_label[k][np.where(prob >= args.theta_gmm)[0]]
            else:
                clean_id = id_by_label[k][np.where(score[id_by_label[k]] >= args.theta_cons)[0]]
            clean_id_all.append(clean_id)
        clean_id_all = np.concatenate(clean_id_all)
        select_i.append(clean_id_all)  # 这里应该存了四组clean_id分别对应zero loss. zero consist，logist_loss, logist consist

    # aggregate final sample selection results
    select = select_i[0]
    for i in range(len(select_i) - 1):
        # 求select_i的四组数据的交集
        select = np.intersect1d(select, select_i[i + 1])


    # ****************** To avoid empty class, fill in with clip zeroshot classifier selection *************************#
    num_by_class = np.array([np.sum(all_labels_np[select] == i) for i in range(num_classes)])

    zero_class = np.where(num_by_class == 0)[0]
    if len(zero_class) != 0:
        non_zero_class = np.where(num_by_class != 0)[0]
        if len(non_zero_class) == 0:
            num_smallest = max(1, int(len(all_labels_np) / num_classes / num_classes))
        else:
            num_smallest = num_by_class[non_zero_class].min()
            if num_smallest < len(all_labels_np) / num_classes / num_classes:
                num_smallest = int(len(all_labels_np) / num_classes / num_classes)
                num_smallest = max(1, num_smallest)
        # print(num_smallest)
        # print(zero_class, non_zero_class, num_smallest)
        all_by_class = [np.where(all_labels_np == i)[0] for i in range(num_classes)]
        for clx in zero_class:
            clx_prob = method_score[0][all_by_class[clx]]
            # prob_rank = clx_prob.argsort(descending=True)
            prob_rank = clx_prob.argsort()
            if num_smallest == 1:
                selected_clx = np.array(all_by_class[clx][prob_rank[:num_smallest]])
            else:
                selected_clx = all_by_class[clx][prob_rank[:num_smallest]]
            select = np.concatenate([select, selected_clx])
    select_fixed = np.unique(select)

    # metrics_fixed = compute_selection_metrics(select_fixed, is_clean_gt)
    metrics_fixed = compute_selection_metrics(
        select_fixed,
        is_clean_gt,
        labels=all_labels_tensor,
        num_classes=num_classes
    )
    write_selection_metrics_to_file(
        metrics_fixed,
        metrics_file,
        title="FIXED PROMPT"
    )
    print(
        "[Selection Metrics | FIXED PROMPT] "
        f"Precision={metrics_fixed['precision']:.4f} | "
        f"Recall={metrics_fixed['recall']:.4f} | "
        f"F1={metrics_fixed['f1']:.4f} | "
        f"Acc={metrics_fixed['accuracy']:.4f} | "
        f"SelectRatio={metrics_fixed['selected_ratio']:.4f}"
    )
    
    print("Confusion matrix [[TN, FP],[FN, TP]]:\n", metrics_fixed["confusion_matrix"])
    print(f"per class {metrics_fixed['per_class_metrics']}")
    selected_final = select_fixed.copy()
    selection_ckpt = _initial_selection_checkpoint(
        selected_indices=selected_final,
        fixed_indices=select_fixed,
        fixed_metrics=metrics_fixed,
    )

    simple_lora_refine = bool(getattr(args, "clip_lora_refine", False))
    if simple_lora_refine:
        try:
            clean_indices = np.unique(_as_numpy_indices(select_fixed))
            num_samples = len(all_labels_np)
            clean_indices = clean_indices[(clean_indices >= 0) & (clean_indices < num_samples)]
            noise_indices = np.setdiff1d(np.arange(num_samples), clean_indices, assume_unique=False)
            print(
                "[LoRA refine simple] "
                f"clean={len(clean_indices)} noise={len(noise_indices)}"
            )

            if len(clean_indices) == 0 or len(noise_indices) == 0:
                print("[LoRA refine simple] empty clean/noise set; fallback to fixed selection.")
            else:
                target_keywords = tuple(
                    item.strip()
                    for item in str(getattr(args, "clip_lora_target_keywords", "visual")).split(",")
                    if item.strip()
                )
                freeze_module(clip_model)
                injected_modules = inject_lora_linear_modules(
                    clip_model,
                    target_keywords=target_keywords,
                    rank=int(getattr(args, "clip_lora_rank", 4)),
                    alpha=float(getattr(args, "clip_lora_alpha", 8.0)),
                    dropout=float(getattr(args, "clip_lora_dropout", 0.0)),
                )
                if len(injected_modules) == 0:
                    print("[LoRA refine simple] no Linear modules matched target keywords; fallback to fixed selection.")
                else:
                    batch_size = max(1, int(getattr(args, "clip_lora_batch_size", 128)))
                    batch_size = min(batch_size, len(clean_indices), len(noise_indices))
                    pin_memory = device.type == "cuda"
                    clean_train_loader = torch.utils.data.DataLoader(
                        torch.utils.data.Subset(clip_data, clean_indices.tolist()),
                        batch_size=batch_size,
                        shuffle=True,
                        num_workers=num_workers,
                        pin_memory=pin_memory,
                        drop_last=True,
                    )
                    noise_train_loader = torch.utils.data.DataLoader(
                        torch.utils.data.Subset(clip_data, noise_indices.tolist()),
                        batch_size=batch_size,
                        shuffle=True,
                        num_workers=num_workers,
                        pin_memory=pin_memory,
                        drop_last=True,
                    )

                    text_features = build_label_text_features(
                        clip_model,
                        class_names=class_names,
                        suffix=suffix,
                        use_cot=use_cot,
                        cot_suffix=cot_suffix,
                        device=device,
                    ).to(device)
                    discriminator = CLIPNoiseDiscriminator(
                        feature_dim=int(all_features.shape[1]),
                        hidden_dim=int(getattr(args, "clip_lora_hidden_dim", 512)),
                        dropout=float(getattr(args, "clip_lora_head_dropout", 0.1)),
                    ).to(device)

                    for param in clip_model.parameters():
                        param.requires_grad_(False)
                    for param in iter_lora_parameters(clip_model):
                        param.requires_grad_(True)
                    for param in discriminator.parameters():
                        param.requires_grad_(True)

                    train_params = [
                        param
                        for param in list(iter_lora_parameters(clip_model)) + list(discriminator.parameters())
                        if param.requires_grad
                    ]
                    if len(train_params) == 0:
                        raise RuntimeError("No trainable LoRA/discriminator parameters found.")

                    optimizer = torch.optim.AdamW(
                        train_params,
                        lr=float(getattr(args, "clip_lora_lr", 1e-4)),
                        weight_decay=float(getattr(args, "clip_lora_weight_decay", 0.0)),
                    )
                    grad_clip = float(getattr(args, "clip_lora_grad_clip", 1.0))
                    train_stats = {
                        "loss": 0.0,
                        "steps": 0,
                        "batch_size": int(batch_size),
                        "num_clean": int(len(clean_indices)),
                        "num_noise": int(len(noise_indices)),
                    }

                    clip_model.train()
                    discriminator.train()
                    for _ in range(int(getattr(args, "clip_lora_epochs", 3))):
                        noise_train_iter = iter(noise_train_loader)
                        for clean_batch in clean_train_loader:
                            try:
                                noise_batch = next(noise_train_iter)
                            except StopIteration:
                                noise_train_iter = iter(noise_train_loader)
                                noise_batch = next(noise_train_iter)

                            clean_images, clean_labels, _ = _extract_images_labels_indices(clean_batch)
                            noise_images, noise_labels, _ = _extract_images_labels_indices(noise_batch)
                            step_batch_size = min(clean_images.shape[0], noise_images.shape[0])
                            if step_batch_size <= 0:
                                continue

                            clean_images = clean_images[:step_batch_size].to(device, non_blocking=True)
                            noise_images = noise_images[:step_batch_size].to(device, non_blocking=True)
                            clean_labels = clean_labels[:step_batch_size].long().to(device, non_blocking=True)
                            noise_labels = noise_labels[:step_batch_size].long().to(device, non_blocking=True)
                            clean_targets = torch.ones(step_batch_size, dtype=torch.long, device=device)
                            noise_targets = torch.zeros(step_batch_size, dtype=torch.long, device=device)

                            images = torch.cat([clean_images, noise_images], dim=0)
                            labels_for_text = torch.cat([clean_labels, noise_labels], dim=0)
                            targets = torch.cat([clean_targets, noise_targets], dim=0)

                            image_features = clip_model.encode_image(images).float()
                            image_features = torch.nan_to_num(
                                image_features,
                                nan=0.0,
                                posinf=0.0,
                                neginf=0.0,
                            )
                            logits = discriminator(image_features, text_features[labels_for_text])
                            logits = torch.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4)
                            loss = F.cross_entropy(logits, targets)
                            if not torch.isfinite(loss):
                                continue

                            optimizer.zero_grad(set_to_none=True)
                            loss.backward()
                            bad_grad = False
                            for param in train_params:
                                if param.grad is not None and (not torch.isfinite(param.grad).all()):
                                    bad_grad = True
                                    break
                            if bad_grad:
                                optimizer.zero_grad(set_to_none=True)
                                continue
                            if grad_clip > 0:
                                torch.nn.utils.clip_grad_norm_(train_params, grad_clip)
                            optimizer.step()

                            train_stats["loss"] = float(loss.detach().cpu().item())
                            train_stats["steps"] += 1

                    clean_scores = score_lora_noise_discriminator(
                        clip_model=clip_model,
                        discriminator=discriminator,
                        data_loader=clip_loader,
                        text_features=text_features,
                        device=device,
                    )
                    score_threshold = float(getattr(args, "clip_lora_select_threshold", 0.5))
                    selected_tensor = torch.where(clean_scores >= score_threshold)[0].long()
                    selected_final = selected_tensor.cpu().numpy().astype(np.int64)
                    selected_pred_fixed = pred_fixed.index_select(
                        0,
                        selected_tensor.to(pred_fixed.device),
                    ).detach().cpu()
                    selection_kind = "lora_simple_refined"

                    print(
                        "[LoRA refine simple score]",
                        "min=", float(clean_scores.min()),
                        "max=", float(clean_scores.max()),
                        "mean=", float(clean_scores.mean()),
                        "std=", float(clean_scores.std()),
                        "threshold=", score_threshold,
                        "selected=", len(selected_final),
                    )
                    metrics_lora = compute_selection_metrics(
                        selected_final,
                        is_clean_gt,
                        labels=all_labels_tensor,
                        num_classes=num_classes,
                    )
                    write_selection_metrics_to_file(
                        metrics_lora,
                        metrics_file,
                        title="LORA SIMPLE REFINED"
                    )
                    print(
                        "[Selection Metrics | LORA SIMPLE REFINED] "
                        f"Precision={metrics_lora['precision']:.4f} | "
                        f"Recall={metrics_lora['recall']:.4f} | "
                        f"F1={metrics_lora['f1']:.4f} | "
                        f"Acc={metrics_lora['accuracy']:.4f} | "
                        f"SelectRatio={metrics_lora['selected_ratio']:.4f}"
                    )
                    print("Confusion matrix [[TN, FP],[FN, TP]]:\n", metrics_lora["confusion_matrix"])

                    selection_ckpt = {
                        "kind": selection_kind,
                        "selected_indices": selected_final,
                        "selected_pred_fixed": selected_pred_fixed,
                        "fixed_indices": select_fixed,
                        "clean_train_indices": clean_indices,
                        "noise_train_indices": noise_indices,
                        "train_stats": train_stats,
                        "injected_modules": injected_modules,
                        "lora_state_dict": collect_lora_state_dict(clip_model),
                        "discriminator_state_dict": discriminator.state_dict(),
                        "clean_scores": clean_scores.cpu(),
                        "fixed_metrics": metrics_fixed,
                        "lora_metrics": metrics_lora,
                    }
        except Exception as exc:
            print(f"[LoRA refine simple] failed with {type(exc).__name__}: {exc}. Fallback to fixed selection.")

    if save_prompt:
        metrics_final = compute_selection_metrics(
            selected_final,
            is_clean_gt,
            labels=all_labels_tensor,
            num_classes=num_classes,
        )
        write_selection_metrics_to_file(
            metrics_final,
            metrics_file,
            title="FINAL SELECTED"
        )
        print(
            "[Selection Metrics | FINAL SELECTED] "
            f"Precision={metrics_final['precision']:.4f} | "
            f"Recall={metrics_final['recall']:.4f} | "
            f"F1={metrics_final['f1']:.4f} | "
            f"Acc={metrics_final['accuracy']:.4f} | "
            f"SelectRatio={metrics_final['selected_ratio']:.4f}"
        )
        print("Confusion matrix [[TN, FP],[FN, TP]]:\n", metrics_final["confusion_matrix"])

        selection_ckpt["final_metrics"] = metrics_final
        _save_selection_checkpoint(
            selection_ckpt,
            os.path.join(args.dataset, args.run_path, "lora_noise_filter.pth")
        )

    return _pack_combined_selection_return(
        selected_indices=selected_final,
        pred_fixed=pred_fixed,
        all_labels=all_labels_tensor,
        checkpoint=selection_ckpt,
        return_prompt=return_prompt,
    )
