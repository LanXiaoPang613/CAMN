import os
import sys
import argparse
import math
import time
import json
import copy
from types import SimpleNamespace
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from utils import *
from utils.builder import *
from model.MLPHeader import MLPHead
from util import *
from utils.eval import *
from model.SevenCNN import CNN
from torch.utils.data import DataLoader
from utils.ema import EMA
from utils.SCS import SCS
from utils.SCR import SCR
from utils.loss import cross_entropy_MUL
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import datetime
from utils.logger import Logger
# from ptflops import get_model_complexity_info


def save_current_script(log_dir):
    current_script_path = __file__
    shutil.copy(current_script_path, log_dir)


def build_logger(params):
    if params.ablation:
        logger_root = f'Ablation/{params.dataset}'
    else:
        logger_root = f'Results/{params.dataset}'
    logger_root = str(params.model) + logger_root
    if not os.path.isdir(logger_root):
        os.makedirs(logger_root, exist_ok=True)
    percentile = int(params.closeset_ratio * 100)
    noise_condition = f'symm_{percentile:2d}' if params.noise_type == 'symmetric' else f'asym_{percentile:2d}'
    logtime = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    if params.ablation:
        result_dir = os.path.join(logger_root, noise_condition, f'{params.log}-{logtime}')
    else:
        result_dir = os.path.join(logger_root, noise_condition, f'{params.log}-{logtime}')
    logger = Logger(logging_dir=result_dir, DEBUG=True)
    logger.set_logfile(logfile_name='log.txt')
    # save_config(params, f'{result_dir}/params.cfg')
    save_params(params, f'{result_dir}/params.json', json_format=True)
    save_current_script(result_dir)
    logger.msg(f'Result Path: {result_dir}')
    return logger, result_dir


def get_baseline_stats(result_file):
    with open(result_file, 'r') as f:
        lines = f.readlines()
    test_acc_list = []
    test_acc_list2 = []
    valid_epoch = []
    # valid_epoch = [191, 192, 193, 194, 195, 196, 197, 198, 199, 200]
    for idx in range(1, min(11, len(lines) + 1)):
        line = lines[-idx].strip()
        epoch, test_acc = line.split(': ')[0].split(' ')[-1], line.split('test acc ')[1].split(', ')[0]
        ep = int(epoch)
        valid_epoch.append(ep)
        # assert ep in valid_epoch, ep
        if '/' not in test_acc:
            test_acc_list.append(float(test_acc))
        else:
            test_acc1, test_acc2 = map(lambda x: float(x), test_acc.split(': ')[1].lstrip('(').rstrip(')').split('/'))
            test_acc_list.append(test_acc1)
            test_acc_list2.append(test_acc2)
    if len(test_acc_list2) == 0:
        test_acc_list = np.array(test_acc_list)
        print(valid_epoch)
        print(f'mean: {test_acc_list.mean():.2f}, std: {test_acc_list.std():.2f}')
        print(f' {test_acc_list.mean():.2f}±{test_acc_list.std():.2f}')
        return {'mean': test_acc_list.mean(), 'std': test_acc_list.std(), 'valid_epoch': valid_epoch}
    else:
        test_acc_list = np.array(test_acc_list)
        test_acc_list2 = np.array(test_acc_list2)
        print(valid_epoch)
        print(f'mean: {test_acc_list.mean():.2f} , std: {test_acc_list.std():.2f}')
        print(f'mean: {test_acc_list2.mean():.2f} , std: {test_acc_list2.std():.2f}')
        print(
            f' {test_acc_list.mean():.2f}±{test_acc_list.std():.2f}  ,  {test_acc_list2.mean():.2f}±{test_acc_list2.std():.2f} ')
        return {'mean1': test_acc_list.mean(), 'std1': test_acc_list.std(),
                'mean2': test_acc_list2.mean(), 'std2': test_acc_list2.std(),
                'valid_epoch': valid_epoch}


def wrapup_training(result_dir, best_accuracy):
    stats = get_baseline_stats(f'{result_dir}/log.txt')
    with open(f'{result_dir}/result_stats.txt', 'w') as f:
        f.write(f"valid epochs: {stats['valid_epoch']}\n")
        if 'mean' in stats.keys():
            f.write(f"mean: {stats['mean']:.4f}, std: {stats['std']:.4f}\n")
        else:
            f.write(f"mean1: {stats['mean1']:.4f}, std2: {stats['std1']:.4f}\n")
            f.write(f"mean2: {stats['mean2']:.4f}, std2: {stats['std2']:.4f}\n")
    os.rename(result_dir, f"{result_dir}-bestAcc_{best_accuracy:.4f}-lastAcc_{stats['mean']:.4f}")


def conf_penalty(outputs):
    outputs = outputs.clamp(min=1e-12)
    probs = torch.softmax(outputs, dim=1)
    return torch.mean(torch.sum(probs.log() * probs, dim=1))


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
                        "clean": int(class_clean.sum().item()),
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
        f"sed fine, "
        f"{title}: precision {metrics['precision']:.4f}, "
        f"recall {metrics['recall']:.4f}, f1 {metrics['f1']:.4f}, "
        f"accuracy {metrics['accuracy']:.4f}, "
        f"selected_ratio {metrics['selected_ratio']:.4f}, "
        f"selected {metrics['selected_count']}/{metrics['num_samples']}, "
        f"confusion_matrix {metrics['confusion_matrix']}, "
        f"per class {metrics['per_class']}"
    )


def log_selection_metrics(result_dir, epoch, metrics_by_name):
    if result_dir is None:
        return
    epoch_text = int(epoch) + 1 if epoch is not None else "unknown"
    with open(os.path.join(result_dir, "selection_metrics.txt"), "a") as f:
        for name, metrics in metrics_by_name:
            f.write(f"Epoch {epoch_text} [{name}] {format_selection_metrics('', metrics).lstrip(': ')}\n")
            f.write("\n")
            f.write(f"per-class {metrics['per_class']}\n")


def build_clipcleaner_args(params, result_dir):
    dataset_map = {
        "cifar100nc": "cifar100",
        "cifar80no": "cifar100",
        "cifar10nc": "cifar10",
    }
    dataset_name = dataset_map.get(params.dataset, params.dataset)
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
        print("use-clip-cleaner")
        return None

    from clipcleaner_new import combined_selection

    clip_args = build_clipcleaner_args(params, result_dir)
    os.makedirs(os.path.join(clip_args.dataset, clip_args.run_path), exist_ok=True)
    print("[CLIPCleaner] running combined_selection before SED training...")
    selected_indices = combined_selection(
        clip_args,params,
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


from selection_clean_utils import (
    _as_long_tensor,
    build_clipcleaner_args,
    combine_clean_indices,
    compute_selection_metrics,
    format_selection_metrics,
    indices_to_mask,
    log_selection_metrics,
    mask_to_indices,
)


def warmup(net, scs, scr, net_ema, ema, optimizer, trainloader, dev, train_loss_meter,train_accuracy_meter):
    net.train()
    pbar = tqdm(trainloader, ncols=150, ascii=' >', leave=False, desc='WARMUP TRAINING')
    for it, sample in enumerate(pbar):
        curr_lr = [group['lr'] for group in optimizer.param_groups][0]

        _, x, _ = sample['data']
        x = x.to(device)
        y = sample['label'].to(device).long()
        outputs = net(x)
        logits = outputs['logits'] if type(outputs) is dict else outputs
        loss_ce = F.cross_entropy(logits, y)
        penalty = conf_penalty(logits)
        loss = loss_ce + penalty

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        ema.update_params(net)
        # 把更新后的ema model加载给net_ema
        ema.apply_shadow(net_ema)

        train_acc = accuracy(logits, y, topk=(1,))
        train_accuracy_meter.update(train_acc[0], x.size(0))
        train_loss_meter.update(loss.detach().cpu().item(), x.size(0))
        pbar.set_postfix_str(f'TrainAcc: {train_accuracy_meter.avg:3.2f}%; TrainLoss: {train_loss_meter.avg:3.2f}')
        pbar.set_description(f'WARMUP TRAINING (lr={curr_lr:.3e})')


def robust_train(net, scs, scr, n_samples, net_ema, ema, optimizer, trainloader, train_loss_meter,train_accuracy_meter, num_class, params, epoch=None, result_dir=None, clip_clean_idx=None):
    net.train()
    pbar = tqdm(trainloader, ncols=150, ascii=' >', leave=False, desc='eval training')

    temp_logits = torch.zeros((n_samples,num_class)).cuda()
    temp_logits_ema = torch.zeros((n_samples, num_class)).cuda()
    label = torch.zeros(n_samples, dtype=torch.long).cuda()
    true_label = torch.zeros(n_samples, dtype=torch.long).cuda()
    psedu_label = torch.zeros(n_samples).cuda()
    with torch.no_grad():
        for it, sample in enumerate(pbar):
            indices = sample['index']
            _, x, _  = sample['data']
            x= x.to(device)
            y = sample['label'].to(device).long()
            y_true = sample['label_true'].to(device).long()
            outputs = net(x)

            outputs_ema = net_ema(x)
            logits_ema = outputs_ema['logits'] if type(outputs_ema) is dict else outputs_ema
            px = logits_ema.softmax(dim=1)
            temp_logits_ema[indices] = px
            _, pesudo = torch.max(px, dim=-1)
            psedu_label[indices] = pesudo.float()

            logits = outputs['logits'] if type(outputs) is dict else outputs
            px = logits.softmax(dim=1)
            temp_logits[indices] = px
            label[indices] = y
            true_label[indices] = y_true

    # 这里统计的是noise_idx和clean_idx结果
    scs_clean_idx, _ = scs.forward(config, temp_logits, label)
    clean_gt = label.cpu().long() == true_label.cpu().long()
    clean_idx = combine_clean_indices(
        scs_clean_idx,
        clip_clean_idx,
        n_samples,
        mode=getattr(params, "clip_combine_mode", "intersection"),
    )
    if len(clean_idx) == 0:
        print("[Selection Merge] merged clean set is empty; falling back to SCS clean set for this epoch.")
        clean_idx = _as_long_tensor(scs_clean_idx)
    noise_idx = mask_to_indices(~indices_to_mask(clean_idx, n_samples))
    print("the length of clean subset and noisy subset are {} and {}".format(len(clean_idx),len(noise_idx)))

    scs_metrics = compute_selection_metrics(
        scs_clean_idx,
        clean_gt,
        labels=true_label,
        num_classes=num_class,
    )
    metrics_to_log = [("SCS", scs_metrics)]
    print(format_selection_metrics("SCS clean selection metrics", scs_metrics))

    if clip_clean_idx is not None:
        clip_metrics = compute_selection_metrics(
            clip_clean_idx,
            clean_gt,
            labels=true_label,
            num_classes=num_class,
        )
        combined_metrics = compute_selection_metrics(
            clean_idx,
            clean_gt,
            labels=true_label,
            num_classes=num_class,
        )
        metrics_to_log.extend([("CLIP", clip_metrics), ("MERGED", combined_metrics)])
        print(format_selection_metrics("CLIP clean selection metrics", clip_metrics))
        print(
            format_selection_metrics(
                f"merged clean selection metrics ({getattr(params, 'clip_combine_mode', 'intersection')})",
                combined_metrics,
            )
        )

    log_selection_metrics(result_dir, epoch, metrics_to_log)
    weight = scr.forward(temp_logits_ema)


    pbar = tqdm(trainloader, ncols=150, ascii=' >', leave=False, desc='robust training')
    clean_lookup = set(_as_long_tensor(clean_idx).tolist())
    for it, sample in enumerate(pbar):
        indices = sample['index']
        _, x, x_s = sample['data']
        x, x_s = x.to(device), x_s.to(device)
        y = sample['label'].to(device).long()
        y_true = sample['label_true'].to(device).long()
        outputs = net(x)
        outputs_s = net(x_s)
        pesudo = psedu_label[indices].long()

        logits = outputs['logits'] if type(outputs) is dict else outputs
        logits_s = outputs_s['logits'] if type(outputs_s) is dict else outputs_s

        ind_in_clean=[]
        ind_in_noise=[]
        for i in range(len(indices)):
            if int(indices[i]) in clean_lookup:
                ind_in_clean.append(int(i))
            else:
                ind_in_noise.append(int(i))

        if config.use_mixup:
            l = np.random.beta(4, 4)
            l = max(l, 1 - l)
            idx2 = torch.randperm(len(ind_in_clean))
            loss_clean = torch.mean(
                F.cross_entropy(logits[ind_in_clean], y[ind_in_clean], reduction="none") * l + (
                            1 - l) * F.cross_entropy(
                    logits[ind_in_clean][idx2], y[ind_in_clean][idx2], reduction="none"))
        else:
            loss_clean = F.cross_entropy(logits[ind_in_clean],y[ind_in_clean])


        loss = loss_clean

        loss_SSL = F.cross_entropy(logits_s, pesudo, reduction="none") * weight[indices]
        loss += loss_SSL.mean() * config.alpha

        one_hot_y = torch.full(size=(y.size(0), n_classes), fill_value=0)
        one_hot_y.scatter_(dim=1, index=torch.unsqueeze(y, dim=1).cpu(), value=1)

        ind_forget = (y[ind_in_noise] != pesudo[ind_in_noise]).cpu()

        loss_MUL = -torch.mean(torch.sum(torch.log(1.0000001 - F.softmax(logits[ind_in_noise][ind_forget], dim=1)) * one_hot_y[ind_in_noise][ind_forget].cuda(), dim=1))

        loss_NLL= cross_entropy_MUL(logits[ind_in_noise][ind_forget], one_hot_y[ind_in_noise][ind_forget].cuda())
        loss = loss + loss_MUL * config.beta + loss_NLL * config.gamma


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        ema.update_params(net)
        ema.apply_shadow(net_ema)

        train_acc = accuracy(logits, y, topk=(1,))
        train_accuracy_meter.update(train_acc[0], x.size(0))
        train_loss_meter.update(loss.detach().cpu().item(), x.size(0))
        pbar.set_postfix_str(f'TrainAcc: {train_accuracy_meter.avg:3.2f}%; TrainLoss: {train_loss_meter.avg:3.2f}')

class ResNet(nn.Module):
    def __init__(self, arch='resnet18', num_classes=200, pretrained=True, activation='tanh', classifier='linear'):
        super().__init__()
        assert arch in torchvision.models.__dict__.keys(), f'{arch} is not supported!'
        resnet = torchvision.models.__dict__[arch](pretrained=pretrained)
        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
        )
        self.feat_dim = resnet.fc.in_features
        self.neck = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        if classifier == 'linear':
            self.classfier_head = nn.Linear(in_features=self.feat_dim, out_features=num_classes)
            init_weights(self.classfier_head, init_method='He')
        elif classifier.startswith('mlp'):
            sf = float(classifier.split('-')[1])
            self.classfier_head = MLPHead(self.feat_dim, mlp_scale_factor=sf, projection_size=num_classes, init_method='He', activation='relu')
        else:
            raise AssertionError(f'{classifier} classifier is not supported.')
        self.proba_head = torch.nn.Sequential(
            MLPHead(self.feat_dim, mlp_scale_factor=1, projection_size=3, init_method='He', activation=activation),
            torch.nn.Sigmoid(),
        )

    def forward(self, x):
        N = x.size(0)
        x = self.backbone(x)
        x = self.neck(x).view(N, -1)
        logits = self.classfier_head(x)
        prob = self.proba_head(x)
        return {'logits': logits, 'prob': prob}


def build_model(num_classes, params_init, dev, config):
    if config.dataset.startswith('web-'):
        net = ResNet(arch="resnet50", num_classes=num_classes, pretrained=True)
    else:
        net = CNN(input_channel=3, n_outputs=n_classes)

    return net.cuda()


def build_optimizer(net, params):
    if params.opt == 'adam':
        return build_adam_optimizer(net.parameters(), params.lr, params.weight_decay, amsgrad=False)
    elif params.opt == 'sgd':
        return build_sgd_optimizer(net.parameters(), params.lr, params.weight_decay, nesterov=True)
    else:
        raise AssertionError(f'{params.opt} optimizer is not supported yet!')


def resolve_num_workers(params, default_workers):
    if params.num_workers is not None:
        return params.num_workers
    return 0 if os.name == 'nt' else default_workers


def build_loader(params):
    dataset_n = params.dataset
    if dataset_n in ["cifar100nc", "cifar80no"]:
        num_workers = resolve_num_workers(params, 8)
        num_classes = int(100 * (1 - config.openset_ratio))
        transform = build_transform(rescale_size=32, crop_size=32)
        dataset = build_cifar100n_dataset("./data/cifar100",
                                          CLDataTransform(transform['cifar_test'],transform['cifar_train'],transform['cifar_train_strong_aug']),
                                          transform['cifar_test'], noise_type=params.noise_type,
                                          openset_ratio=params.openset_ratio, closeset_ratio=params.closeset_ratio,seed=params.seed)
        trainloader = DataLoader(dataset['train'], batch_size=params.batch_size, shuffle=True, num_workers=num_workers,
                                 pin_memory=True)
        test_loader = DataLoader(dataset['test'], batch_size=16, shuffle=False, num_workers=num_workers, pin_memory=False)
    if dataset_n.startswith('web-'):
        num_workers = resolve_num_workers(params, 4)
        class_ = {"web-aircraft": 100, "web-bird": 200, "web-car": 196}
        num_classes = class_[dataset_n]
        transform = build_transform(rescale_size=448, crop_size=448)
        dataset = build_webfg_dataset(os.path.join('Datasets', dataset_n),
                                      CLDataTransform(transform['train'], transform["train_strong_aug"]),
                                      transform['test'])
        trainloader = DataLoader(dataset["train"], batch_size=params.batch_size, shuffle=True, num_workers=num_workers,
                                 pin_memory=True)
        test_loader = DataLoader(dataset['test'], batch_size=16, shuffle=False, num_workers=num_workers,
                                 pin_memory=False)

    num_samples = len(trainloader.dataset)
    return_dict = {'trainloader': trainloader, 'num_classes': num_classes, 'num_samples': num_samples, 'dataset': dataset_n}
    return_dict['test_loader'] = test_loader
    return return_dict

'''
python SED_FINE.py --warmup-epoch 200 --epoch 300 --batch-size 128 --lr 0.05 --warmup-lr 0.1  --noise-type symmetric 
--closeset-ratio 0.2 --lr-decay cosine:200,5e-4,300  --opt sgd --dataset cifar100nc --gpu 4 
--momentum-scs 0.999 --momentum-scr 0.99 --aph 0.95 --alpha 1.0 --beta 0.1 --gamma 0.002
'''
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default="0")
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.05)
    parser.add_argument('--lr-decay', type=str, default='cosine:200,5e-4,300')
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--opt', type=str, default='sgd')
    parser.add_argument('--warmup-epochs', type=int, default=200)
    parser.add_argument('--warmup-lr', type=float, default=0.1)
    parser.add_argument('--warmup-gradual', action='store_true')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--params-init', type=str, default='none')
    parser.add_argument('--aph', type=float, default=0.95)
    parser.add_argument('--dataset', type=str, default='cifar100nc')
    parser.add_argument('--noise-type', type=str, default='symmetric')
    parser.add_argument('--closeset-ratio', type=float, default=0.8)
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--beta', type=float, default=0.1)
    parser.add_argument('--gamma', type=float, default=0.002)
    parser.add_argument('--save-weights', type=bool, default=False)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--restart-epoch', type=int, default=0)
    parser.add_argument('--use-quantile', type=bool, default=True)
    parser.add_argument('--clip-thresh', type=bool, default=True)
    parser.add_argument('--use-mixup', type=bool, default=False)
    parser.add_argument('--momentum-scs', type=float, default=0.999)
    parser.add_argument('--momentum-scr', type=float, default=0.99)
    parser.add_argument('--ablation', type=bool, default=False)
    parser.add_argument('--log', type=str, default=None)
    parser.add_argument('--model', type=str, default='CNN')
    parser.add_argument('--num-workers', type=int, default=None)
    parser.add_argument('--use-clip-cleaner', action='store_true', default=False)
    parser.add_argument('--clip-combine-mode', type=str, default='intersection',
                        choices=['intersection', 'union', 'clip', 'scs'])
    parser.add_argument('--clip-dataset-path', type=str, default=None)
    parser.add_argument('--clip-run-path', type=str, default=None)
    parser.add_argument('--clip-model', type=str, default='small', choices=['small', 'tiny', 'large'])
    parser.add_argument('--clip-save-prompt', action='store_true', default=False)
    parser.add_argument('--theta-gmm', type=float, default=0.7)
    parser.add_argument('--theta-cons', type=float, default=0.9)
    parser.add_argument('--clip-eval-batch-size', type=int, default=256)
    parser.add_argument('--clip-num-workers', type=int, default=4)
    parser.add_argument('--clip-lora-refine', action='store_true', default=False)
    parser.add_argument('--clip-lora-rank', type=int, default=4)
    parser.add_argument('--clip-lora-alpha', type=float, default=8.0)
    parser.add_argument('--clip-lora-dropout', type=float, default=0.0)
    parser.add_argument('--clip-lora-head-dropout', type=float, default=0.1)
    parser.add_argument('--clip-lora-lr', type=float, default=1e-4)
    parser.add_argument('--clip-lora-weight-decay', type=float, default=0.0)
    parser.add_argument('--clip-lora-epochs', type=int, default=3)
    parser.add_argument('--clip-lora-batch-size', type=int, default=128)
    parser.add_argument('--clip-lora-hidden-dim', type=int, default=512)
    parser.add_argument('--clip-lora-grad-clip', type=float, default=1.0)
    parser.add_argument('--clip-lora-select-threshold', type=float, default=0.56)
    args = parser.parse_args()
    print(args)
    return args

if __name__ == '__main__':
    config = parse_args()
    config.openset_ratio = 0.0 if config.dataset == 'cifar100nc' else 0.2
    init_seeds(config.seed)
    device = set_device(config.gpu)
    # bulid logger
    logger, result_dir = build_logger(config)
    logger.msg(str(config))
    # create dataloader
    loader_dict = build_loader(config)
    dataset_name, n_classes, n_samples = loader_dict['dataset'], loader_dict['num_classes'], loader_dict['num_samples']
    clip_clean_idx = run_clipcleaner_selection(config, result_dir, n_samples)

    scs = SCS(num_classes=n_classes,momentum=config.momentum_scs)
    scr = SCR(num_classes=n_classes, momentum=config.momentum_scr)

    # create model
    model = build_model(n_classes, config.params_init, device, config)

    if config.resume!=None:
        path = config.resume
        dict_s = torch.load(path, map_location='cpu')
        model.load_state_dict(dict_s)
        model.cuda()

    # create optimizer & lr_plan or lr_scheduler
    optim = build_optimizer(model, config)
    # 返回所有training epochs的lr,每个epoch一个
    lr_plan = build_lr_plan(config.lr, config.epochs, config.warmup_epochs, config.warmup_lr, decay=config.lr_decay,
                            warmup_gradual=config.warmup_gradual)
    model_ema = copy.deepcopy(model)

    ema = EMA(model_ema, alpha=config.aph)
    ema.apply_shadow(model_ema)

    targets_all = None
    best_accuracy, best_epoch = 0.0, None
    train_loss_meter = AverageMeter()
    train_accuracy_meter = AverageMeter()
    epoch = 0
    last_ten =0
    if config.restart_epoch != 0:
        epoch = config.restart_epoch
        config.restart_epoch = 0
    while epoch < config.epochs:
        train_loss_meter.reset()
        train_accuracy_meter.reset()
        adjust_lr(optim, lr_plan[epoch])
        input_loader = loader_dict['trainloader']
        if epoch < config.warmup_epochs:
            warmup(model, scs, scr, model_ema, ema, optim, input_loader, device, train_loss_meter,train_accuracy_meter)
        else:
            robust_train(model, scs, scr, n_samples, model_ema, ema, optim, input_loader, train_loss_meter,train_accuracy_meter, n_classes, config, epoch, result_dir, clip_clean_idx)
        eval_result = evaluate_cls_acc(loader_dict['test_loader'], model, device)
        test_accuracy = eval_result['accuracy']
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            best_epoch = epoch + 1
            if config.save_weights:
                torch.save(model.state_dict(), f'./result_log/'+dataset_name+"_"+str(epoch)+'_best_model.pth')
        if epoch >= config.epochs-10:
            last_ten+=test_accuracy
        logger.info(
            f'>> Epoch {epoch}: loss {train_loss_meter.avg:.2f} ,train acc {train_accuracy_meter.avg:.2f} ,test acc {test_accuracy:.2f}, best acc {best_accuracy:.2f}')
        epoch+=1
    print("last ten accuracy is ",float(last_ten/10))
    wrapup_training(result_dir, best_accuracy)
