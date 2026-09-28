import os
import sys
import argparse
import math
import time
import json
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from utils import *
from utils.builder import *
from model.MLPHeader import MLPHead
from util import *
from utils.loss import cross_entropy_MUL,label_smoothing_cross_entropy_MUL
from utils.eval import *
from model.SevenCNN import CNN
from torch.utils.data import DataLoader
from utils.ema import EMA
from utils.SCS import SCS
from utils.SCR import SCR
from selection_clean_utils import (
    _as_long_tensor,
    combine_clean_indices,
    indices_to_mask,
    mask_to_indices,
    resolve_webfg_dataset_root,
    run_clipcleaner_selection,
)

from torch.cuda.amp import autocast as autocast
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import datetime
from utils.logger import Logger


try:
    from thop import profile
except ImportError:
    profile = None

def save_current_script(log_dir):
    current_script_path = __file__
    shutil.copy(current_script_path, log_dir)


def build_logger(params):
    if params.ablation:
        logger_root = f'Ablation/{params.dataset}'
    else:
        logger_root = f'Results/{params.dataset}'
    if not os.path.isdir(logger_root):
        os.makedirs(logger_root, exist_ok=True)
    logtime = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    if params.ablation:
        result_dir = os.path.join(logger_root, f'{params.log}-{logtime}')
    else:
        result_dir = os.path.join(logger_root, f'{params.log}-{logtime}')
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


import warnings
warnings.filterwarnings('ignore')

def conf_penalty(outputs):
    outputs = outputs.clamp(min=1e-12)
    probs = torch.softmax(outputs, dim=1)
    return torch.mean(torch.sum(probs.log() * probs, dim=1))
def warmup(net, scs, scr, net_ema, ema, optimizer, trainloader, dev, train_loss_meter,train_accuracy_meter,params):
    net.train()
    pbar = tqdm(trainloader, ncols=150, ascii=' >', leave=False, desc='WARMUP TRAINING')
    for it, sample in enumerate(pbar):
        with autocast():
            curr_lr = [group['lr'] for group in optimizer.param_groups][0]

            x, _ = sample['data']
            x = x.to(device)
            y = sample['label'].to(device)
            outputs = net(x)
            logits = outputs['logits'] if type(outputs) is dict else outputs
            loss_ce = F.cross_entropy(logits, y, label_smoothing=params.labelsmo)
            penalty = conf_penalty(logits)
            loss = loss_ce + penalty

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            ema.update_params(net)
            ema.apply_shadow(net_ema)

        train_acc = accuracy(logits, y, topk=(1,))
        train_accuracy_meter.update(train_acc[0], x.size(0))
        train_loss_meter.update(loss.detach().cpu().item(), x.size(0))
        pbar.set_postfix_str(f'TrainAcc: {train_accuracy_meter.avg:3.2f}%; TrainLoss: {train_loss_meter.avg:3.2f}')
        pbar.set_description(f'WARMUP TRAINING (lr={curr_lr:.3e})')


def robust_train(net, scs, scr, n_samples, net_ema, ema, optimizer, trainloader, train_loss_meter,train_accuracy_meter, num_class, params, clip_clean_idx=None):
    net.train()
    pbar = tqdm(trainloader, ncols=150, ascii=' >', leave=False, desc='eval training')

    temp_logits = torch.zeros((n_samples,num_class)).cuda()
    temp_logits_ema = torch.zeros((n_samples, num_class)).cuda()
    label = torch.zeros(n_samples).cuda()
    psedu_label = torch.zeros(n_samples).cuda()
    with autocast():
        with torch.no_grad():
            for it, sample in enumerate(pbar):
                indices = sample['index']
                x, _ = sample['data']
                x= x.to(device)
                y = sample['label'].to(device)
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
                label[indices] = y.float()

    scs_clean_idx, _ = scs.forward(config, temp_logits, label)
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
    if clip_clean_idx is not None:
        print(
            "[Selection Merge] "
            f"SCS clean {len(scs_clean_idx)}/{n_samples}, "
            f"CLIP clean {len(clip_clean_idx)}/{n_samples}, "
            f"merged clean {len(clean_idx)}/{n_samples} "
            f"({getattr(params, 'clip_combine_mode', 'intersection')})"
        )
    weight = scr.forward(temp_logits_ema)


    pbar = tqdm(trainloader, ncols=150, ascii=' >', leave=False, desc='robust training')
    clean_lookup = set(_as_long_tensor(clean_idx).tolist())
    for it, sample in enumerate(pbar):
        with autocast():
            indices = sample['index']
            x, x_s = sample['data']
            x, x_s = x.to(device), x_s.to(device)
            y = sample['label'].to(device)
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
                    F.cross_entropy(logits[ind_in_clean], y[ind_in_clean], reduction="none",label_smoothing=params.labelsmo) * l + (
                                1 - l) * F.cross_entropy(
                        logits[ind_in_clean][idx2], y[ind_in_clean][idx2], reduction="none",label_smoothing=params.labelsmo))
            else:
                loss_clean = F.cross_entropy(logits[ind_in_clean],y[ind_in_clean],label_smoothing=params.labelsmo)


            loss = loss_clean

            loss_SSL = F.cross_entropy(logits_s, pesudo, reduction="none",label_smoothing=params.labelsmo) * weight[indices]
            loss += loss_SSL.mean() * config.alpha

            one_hot_y = torch.full(size=(y.size(0), n_classes), fill_value=0)
            one_hot_y.scatter_(dim=1, index=torch.unsqueeze(y, dim=1).cpu(), value=1)

            """
                尝试1：直接使用loss_MUL作为一个正则化损失项失效；猜测可能是部分错误选择的噪声样本，破坏性能
                尝试2：采用进一步的挑选，校正标签和给定的噪声标签不一致的样本进行遗忘；依旧失效
                尝试3：损失项用消极学习的方法遗忘掉噪声标签的样本;有用但是效果提升不大
            """

            ind_forget = (y[ind_in_noise] != pesudo[ind_in_noise]).cpu()

            loss_NLL = -torch.mean(torch.sum(
                torch.log(1.0000001 - F.softmax(logits[ind_in_noise][ind_forget], dim=1)) * one_hot_y[ind_in_noise][
                    ind_forget].cuda(), dim=1))

            loss_MUL = torch.mean(label_smoothing_cross_entropy_MUL(logits[ind_in_noise][ind_forget], y[ind_in_noise][ind_forget].cuda(), epsilon= 0.35, reduction="none"))
            loss = loss + loss_NLL * config.beta + loss_MUL * config.gamma


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


def build_loader(params):
    dataset_n = params.dataset
    if dataset_n in ["cifar100nc", "cifar80no"]:
        num_classes = int(100 * (1 - config.openset_ratio))
        transform = build_transform(rescale_size=32, crop_size=32)
        dataset = build_cifar100n_dataset("./data/cifar100",
                                          CLDataTransform(transform['cifar_train'],
                                                          transform['cifar_train_strong_aug']),
                                          transform['cifar_test'], noise_type=params.noise_type,
                                          openset_ratio=params.openset_ratio, closeset_ratio=params.closeset_ratio,seed=params.seed)
        trainloader = DataLoader(dataset['train'], batch_size=params.batch_size, shuffle=True, num_workers=8,
                                 pin_memory=True)
        test_loader = DataLoader(dataset['test'], batch_size=16, shuffle=False, num_workers=8, pin_memory=False)
    if dataset_n.startswith('web-'):
        class_ = {"web-aircraft": 100, "web-bird": 200, "web-car": 196}
        num_classes = class_[dataset_n]
        transform = build_transform(rescale_size=448, crop_size=448)
        dataset_root = resolve_webfg_dataset_root(getattr(params, "web_data_root", "D:/2_Dataset_All/webFG-496"), dataset_n)
        dataset = build_webfg_dataset(dataset_root,
                                      CLDataTransform(transform['train'], transform["train_strong_aug"]),
                                      transform['To_Tensor'])
        trainloader = DataLoader(dataset["train"], batch_size=params.batch_size, shuffle=True, num_workers=4,
                                 pin_memory=True)
        test_loader = DataLoader(dataset['test'], batch_size=16, shuffle=False, num_workers=4,
                                 pin_memory=False)

    num_samples = len(trainloader.dataset)
    return_dict = {'trainloader': trainloader, 'num_classes': num_classes, 'num_samples': num_samples, 'dataset': dataset_n}
    return_dict['test_loader'] = test_loader
    return return_dict

'''
python SED_FINE_web.py --warmup-epochs 5 --epoch 50 --batch-size 32 --lr 0.005  --warmup-lr 0.005  
--lr-decay cosine:5,5e-4,50 --weight-decay 5e-4 --seed 123 --opt sgd --dataset web-aircraft --gpu 9 
--momentum_scs 0.999 --momentum_scr 0.99 --alpha 1 --aph 0.95 --beta 0.1 --gamma 0.001 --log AIR_BEST
'''
# parse arguments
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default="0")
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--lr-decay', type=str, default='cosine:5,5e-4,50')
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--opt', type=str, default='sgd')
    parser.add_argument('--warmup-epochs', type=int, default=5)
    parser.add_argument('--warmup-lr', type=float, default=0.005)
    parser.add_argument('--warmup-gradual', action='store_true')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--params-init', type=str, default='none')

    parser.add_argument('--aph', type=float, default=0.95)

    parser.add_argument('--dataset', type=str, default='web-aircraft')
    parser.add_argument('--noise-type', type=str, default='symmetric')
    parser.add_argument('--closeset-ratio', type=float, default=0.2)

    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--save-weights', type=bool, default=False)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--restart-epoch', type=int, default=0)

    parser.add_argument('--use-quantile', type=bool, default=True)
    parser.add_argument('--clip-thresh', type=bool, default=True)
    parser.add_argument('--use-mixup', type=bool, default=False)
    parser.add_argument('--momentum_scs', type=float, default=0.999)
    parser.add_argument('--momentum_scr', type=float, default=0.99)
    parser.add_argument('--labelsmo', type=float, default=0.8)

    parser.add_argument('--beta', type=float, default=0.1)
    parser.add_argument('--gamma', type=float, default=0.001)
    parser.add_argument('--ablation', type=bool, default=False)
    parser.add_argument('--log', type=str, default='AIR_BEST')
    parser.add_argument('--web-data-root', type=str, default='D:/2_Dataset_All/webFG-496')
    parser.add_argument('--use-clip-cleaner', action='store_true', default=False)
    parser.add_argument('--clip-combine-mode', type=str, default='intersection',
                        choices=['intersection', 'union', 'clip', 'scs'])
    parser.add_argument('--clip-dataset-path', type=str, default=None)
    parser.add_argument('--clip-run-path', type=str, default=None)
    parser.add_argument('--clip-model', type=str, default='large', choices=['small', 'tiny', 'large'])
    parser.add_argument('--clip-save-prompt', action='store_true', default=False)
    parser.add_argument('--theta-gmm', type=float, default=0.5)
    parser.add_argument('--theta-cons', type=float, default=0)
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
    # eval_result = evaluate_cls_acc_TTA(loader_dict['test_loader'], model, device)
    if config.restart_epoch != 0:
        epoch = config.restart_epoch
        config.restart_epoch = 0
    while epoch < config.epochs:
        train_loss_meter.reset()
        train_accuracy_meter.reset()
        adjust_lr(optim, lr_plan[epoch])
        input_loader = loader_dict['trainloader']
        if epoch < config.warmup_epochs:
            warmup(model, scs, scr, model_ema, ema, optim, input_loader, device, train_loss_meter,train_accuracy_meter,config)
        else:
            robust_train(model, scs, scr, n_samples, model_ema, ema, optim, input_loader, train_loss_meter,train_accuracy_meter, n_classes, config, clip_clean_idx)

        eval_result = evaluate_cls_acc_TTA(loader_dict['test_loader'], model, device)
        test_accuracy = eval_result['accuracy']

        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            best_epoch = epoch + 1
            if config.save_weights:
                torch.save(model.state_dict(), f'./result_log/'+dataset_name+"_"+str(epoch)+'_best_model.pth')
        logger.info(
            f'>> Epoch {epoch}: loss {train_loss_meter.avg:.2f} ,train acc {train_accuracy_meter.avg:.2f} ,test acc {test_accuracy:.2f}, best acc {best_accuracy:.2f}')
        epoch+=1

    wrapup_training(result_dir, best_accuracy)
