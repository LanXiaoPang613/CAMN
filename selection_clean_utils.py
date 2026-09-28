import os
from types import SimpleNamespace

import torch


def _as_long_tensor(values):
    if torch.is_tensor(values):
        return values.detach().cpu().long().view(-1)
    return torch.as_tensor(values, dtype=torch.long).view(-1)


def compute_selection_metrics(select_idx, is_clean_gt, labels=None, num_classes=None):
    clean_gt = _as_long_tensor(is_clean_gt).bool()
    selected = torch.zeros(clean_gt.numel(), dtype=torch.bool)
    select_idx = _as_long_tensor(select_idx)

    if select_idx.numel() > 0:
        if int(select_idx.min()) < 0 or int(select_idx.max()) >= clean_gt.numel():
            raise ValueError("selected indices outside dataset range")
        selected[select_idx] = True

    tp = int((selected & clean_gt).sum().item())
    fp = int((selected & ~clean_gt).sum().item())
    fn = int((~selected & clean_gt).sum().item())
    tn = int((~selected & ~clean_gt).sum().item())

    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    accuracy = (tp + tn) / clean_gt.numel() if clean_gt.numel() > 0 else 0.0
    selected_ratio = selected.float().mean().item() if clean_gt.numel() > 0 else 0.0

    metrics = {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(accuracy),
        "selected_ratio": float(selected_ratio),
        "selected_count": int(selected.sum().item()),
        "num_samples": int(clean_gt.numel()),
        "confusion_matrix": [[tn, fp], [fn, tp]],
    }

    if labels is not None and num_classes is not None:
        labels = _as_long_tensor(labels)
        if labels.numel() != clean_gt.numel():
            raise ValueError("labels and is_clean_gt must have the same length")

        per_class = []
        for class_idx in range(int(num_classes)):
            class_mask = labels == class_idx
            if class_mask.any():
                class_selected = selected[class_mask]
                class_clean = clean_gt[class_mask]
                class_tp = int((class_selected & class_clean).sum().item())
                class_fp = int((class_selected & ~class_clean).sum().item())
                class_fn = int((~class_selected & class_clean).sum().item())
                class_precision = class_tp / (class_tp + class_fp) if class_tp + class_fp > 0 else 0.0
                class_recall = class_tp / (class_tp + class_fn) if class_tp + class_fn > 0 else 0.0
                per_class.append(
                    {
                        "class": class_idx,
                        "precision": float(class_precision),
                        "recall": float(class_recall),
                        "selected": int(class_selected.sum().item()),
                        "clean_total": int(class_clean.sum().item()),
                        "clean_selected": int(class_tp),
                    }
                )
        metrics["per_class"] = per_class

    return metrics


def indices_to_mask(select_idx, n_samples):
    mask = torch.zeros(int(n_samples), dtype=torch.bool)
    select_idx = _as_long_tensor(select_idx)
    if select_idx.numel() == 0:
        return mask
    if int(select_idx.min()) < 0 or int(select_idx.max()) >= int(n_samples):
        raise ValueError("selected indices outside dataset range")
    mask[select_idx] = True
    return mask


def mask_to_indices(mask):
    if not torch.is_tensor(mask):
        mask = torch.as_tensor(mask)
    return torch.where(mask.detach().cpu().bool().view(-1))[0].long()


def combine_clean_indices(scs_clean_idx, clip_clean_idx, n_samples, mode="intersection"):
    if clip_clean_idx is None or mode == "scs":
        return _as_long_tensor(scs_clean_idx)

    scs_mask = indices_to_mask(scs_clean_idx, n_samples)
    clip_mask = indices_to_mask(clip_clean_idx, n_samples)

    if mode == "intersection":
        combined_mask = scs_mask & clip_mask
    elif mode == "union":
        combined_mask = scs_mask | clip_mask
    elif mode == "clip":
        combined_mask = clip_mask
    else:
        raise ValueError(f"unsupported clip combine mode: {mode}")

    return mask_to_indices(combined_mask)


def format_selection_metrics(title, metrics):
    return (
        f"{title}: precision {metrics['precision']:.4f}, "
        f"recall {metrics['recall']:.4f}, f1 {metrics['f1']:.4f}, "
        f"accuracy {metrics['accuracy']:.4f}, "
        f"selected_ratio {metrics['selected_ratio']:.4f}, "
        f"selected {metrics['selected_count']}/{metrics['num_samples']}, "
        f"confusion_matrix {metrics['confusion_matrix']},"
    )


def log_selection_metrics(result_dir, epoch, metrics_by_name):
    if result_dir is None:
        return
    epoch_text = int(epoch) + 1 if epoch is not None else "unknown"
    with open(os.path.join(result_dir, "selection_metrics.txt"), "a") as f:
        for name, metrics in metrics_by_name:
            f.write(f"Epoch {epoch_text} [{name}] {format_selection_metrics('', metrics).lstrip(': ')}\n")
            if "per_class" in metrics:
                f.write("\n[Per-Class Metrics]\n")
                for m in metrics["per_class"]:
                    '''
                    "class": class_idx,
                        "precision": float(class_precision),
                        "recall": float(class_recall),
                        "selected": int(class_selected.sum().item()),
                        "clean": int(class_clean.sum().item()),
                    '''
                    c=m['class']
                    f.write(
                        f"Class {c:02d}: "
                        f"selected={m['selected']:5d} | "
                        f"clean_selected={m['clean_selected']:5d}/{m['clean_total']:5d} | "
                        f"precision={m['precision']:.4f} | "
                        f"recall={m['recall']:.4f} \n"
                        # f"f1={m['f1']:.4f}\n"
                    )


def resolve_webfg_dataset_root(web_data_root, dataset_name):
    candidates = [
        os.path.join(web_data_root, dataset_name),
        os.path.join(web_data_root, f"{dataset_name}.tar", dataset_name),
        os.path.join(web_data_root, f"{dataset_name}.tar"),
    ]
    for candidate in candidates:
        if os.path.isdir(os.path.join(candidate, "train")) and os.path.isdir(os.path.join(candidate, "val")):
            return candidate
    return candidates[0]


def build_clipcleaner_args(params, result_dir):
    dataset_map = {
        "cifar100nc": "cifar100",
        "cifar80no": "cifar100",
        "cifar10nc": "cifar10",
    }
    dataset_name = dataset_map.get(params.dataset, params.dataset)
    if str(dataset_name).startswith("web-"):
        default_path = resolve_webfg_dataset_root(
            getattr(params, "web_data_root", "D:/2_Dataset_All/webFG-496"),
            dataset_name,
        )
    else:
        default_path = "./data/cifar100" if dataset_name == "cifar100" else "./data/cifar10"
    run_path = getattr(params, "clip_run_path", None)
    if run_path is None:
        run_name = os.path.basename(result_dir.rstrip(os.sep)) if result_dir else "sed_fine_clip"
        run_path = os.path.join("sed_fine_clip", run_name)

    return SimpleNamespace(
        dataset=dataset_name,
        dataset_path=getattr(params, "clip_dataset_path", None) or default_path,
        noise_ratio=float(params.closeset_ratio),
        noise_mode=params.noise_type,
        open_ratio=float(params.openset_ratio),
        gpuid=int(str(params.gpu).split(",")[0]),
        model=getattr(params, "clip_model", "small"),
        theta_gmm=float(getattr(params, "theta_gmm", 0.5)),
        theta_cons=float(getattr(params, "theta_cons", 0.8)),
        run_path=run_path,
        seed=int(params.seed),
        clip_eval_batch_size=int(getattr(params, "clip_eval_batch_size", 1024)),
        clip_num_workers=int(getattr(params, "clip_num_workers", 4)),
        clip_lora_refine=bool(getattr(params, "clip_lora_refine", False)),
        clip_lora_rank=int(getattr(params, "clip_lora_rank", 4)),
        clip_lora_alpha=float(getattr(params, "clip_lora_alpha", 8.0)),
        clip_lora_dropout=float(getattr(params, "clip_lora_dropout", 0.0)),
        clip_lora_head_dropout=float(getattr(params, "clip_lora_head_dropout", 0.1)),
        clip_lora_lr=float(getattr(params, "clip_lora_lr", 1e-4)),
        clip_lora_weight_decay=float(getattr(params, "clip_lora_weight_decay", 0.0)),
        clip_lora_epochs=int(getattr(params, "clip_lora_epochs", 3)),
        clip_lora_batch_size=int(getattr(params, "clip_lora_batch_size", 128)),
        clip_lora_hidden_dim=int(getattr(params, "clip_lora_hidden_dim", 512)),
        clip_lora_grad_clip=float(getattr(params, "clip_lora_grad_clip", 1.0)),
        clip_lora_select_threshold=float(getattr(params, "clip_lora_select_threshold", 0.56)),
    )


def run_clipcleaner_selection(params, result_dir, n_samples):
    if not getattr(params, "use_clip_cleaner", False):
        return None

    from clipcleaner_new import combined_selection

    clip_args = build_clipcleaner_args(params, result_dir)
    os.makedirs(os.path.join(clip_args.dataset, clip_args.run_path), exist_ok=True)
    print("[CLIPCleaner] running combined_selection before SED training...")
    selected_indices = combined_selection(
        clip_args,
        params,
        return_prompt=True,
        save_prompt=bool(getattr(params, "clip_save_prompt", False)),
    )[0]
    selected_indices = _as_long_tensor(selected_indices)
    if selected_indices.numel() > 0 and int(selected_indices.max()) >= int(n_samples):
        raise ValueError(
            f"CLIPCleaner selected index {int(selected_indices.max())}, "
            f"but train set has {n_samples} samples"
        )
    print(
        "[CLIPCleaner] clean/noisy split: "
        f"clean {selected_indices.numel()}/{n_samples}, "
        f"noisy {int(n_samples) - selected_indices.numel()}/{n_samples}"
    )
    return selected_indices
