import os
import argparse
import json
from pathlib import Path
import torch
import numpy as np
from CSCF import CSCF
from MultiViewDataset import MultiViewDataset, custom_collate_fn
from torch.utils.data import DataLoader

def get_parser(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--data_root', type=str, default='data')
    parser.add_argument('--save_dir', type=str, default='trained_models')
    parser.add_argument('--dataset', type=str, default='UCF', help='ActivityNet, Food101, SUNRGBD')
    parser.add_argument('--test_type', type=str, default='same', help='same, contra, data1, data2, total or all')
    parser.add_argument('--text_features', type=str, default='clipExtend', help='clip or clipExtend')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=None, help='Override the dataset learning rate')
    parser.add_argument('--modal_lrs', type=float, nargs='+', default=None,
                        help='Optional learning rates for each modality (RGB, depth for SUNRGBD)')
    parser.add_argument('--modal_weight_decays', type=float, nargs='+', default=None,
                        help='Optional Adam weight decay for each modality')
    profile = parser.add_mutually_exclusive_group()
    profile.add_argument('--config', type=str, default=None, help='JSON training hyperparameters')
    profile.add_argument('--baseline', action='store_true', help='Use original dataset training defaults')
    parser.add_argument('--batch_size', type=int, default=None)
    shuffling = parser.add_mutually_exclusive_group()
    shuffling.add_argument('--shuffle', dest='shuffle', action='store_true', help='Shuffle training batches each epoch')
    shuffling.add_argument('--no_shuffle', dest='shuffle', action='store_false')
    parser.set_defaults(shuffle=None)
    final_batch = parser.add_mutually_exclusive_group()
    final_batch.add_argument('--drop_last', dest='drop_last', action='store_true')
    final_batch.add_argument('--keep_last', dest='drop_last', action='store_false')
    parser.set_defaults(drop_last=None)
    parser.add_argument('--scheduler', choices=['none', 'cosine'], default=None)
    parser.add_argument('--warmup_epochs', type=int, default=None)
    parser.add_argument('--min_lr_ratio', type=float, default=None)
    parser.add_argument('--grad_clip', type=float, default=None)
    parser.add_argument('--weight_decay', type=float, default=None)
    parser.add_argument('--topk', type=int, default=1)

    args = parser.parse_args(argv)
    defaults = dict(lr=None, modal_lrs=None, modal_weight_decays=None, batch_size=2048,
                    shuffle=False, scheduler='none', warmup_epochs=0, min_lr_ratio=0.05,
                    grad_clip=None, weight_decay=0.0001, drop_last=False)
    config_path = Path(args.config) if args.config else Path(__file__).with_name('train_config.json')
    args.training_profile = 'original_defaults'
    args.config_source = None
    if args.config or (args.dataset == 'SUNRGBD' and not args.baseline):
        if not config_path.is_file():
            parser.error(f'Training config does not exist: {config_path}')
        with config_path.open(encoding='utf-8') as stream:
            config = json.load(stream)
        unknown = set(config) - set(defaults)
        if unknown:
            parser.error(f'Unsupported training config fields: {sorted(unknown)}')
        defaults.update(config)
        args.training_profile = 'explicit_config' if args.config else 'SUNRGBD'
        args.config_source = str(config_path.resolve())
    # An explicit shared rate/decay overrides corresponding per-modality defaults.
    if args.lr is not None and args.modal_lrs is None:
        defaults['modal_lrs'] = None
    if args.weight_decay is not None and args.modal_weight_decays is None:
        defaults['modal_weight_decays'] = None
    for name, value in defaults.items():
        if getattr(args, name) is None:
            setattr(args, name, value)
    return args

def train(args, save_model=True):
    if args.dataset == 'SUNRGBD' and args.epochs != 50:
        raise ValueError('SUNRGBD parameter tuning requires exactly 50 epochs')
    run_folder = os.getcwd()
    data_folder = os.path.join(run_folder, args.data_root)
    save_dir = os.path.join(run_folder, args.save_dir)
    seed = args.seed
    dataset = args.dataset
    epochs = args.epochs
    device = args.device
    text_features = args.text_features

    if dataset == 'ActivityNet':
        lr = 3e-3
    elif dataset == 'Food101':
        lr = 1e-3
    elif dataset == 'SUNRGBD':
        lr = 1e-4
    else:
        raise ValueError('Invalid dataset name')

    if args.lr is not None:
        lr = args.lr

    np.random.seed(seed)
    torch.manual_seed(seed)

    print(f'------------------ Loading train data ------------------')
    train_dataset = MultiViewDataset(data_folder=data_folder, dataset=dataset, text_features=text_features, type='same', mode='multi', train=True)
    attrs_multi = train_dataset.attrs_multi
    similarity_mat = train_dataset.similarity_mat
    modal_data = train_dataset.inputs_multi[:-2]
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=args.shuffle,
                                  drop_last=args.drop_last, collate_fn=custom_collate_fn)

    print(f'------------------ Create model ------------------')
    num_class = attrs_multi.shape[0]
    attr_dim = attrs_multi.shape[1]
    hid_dim = 1024
    proto_dim = [datum.shape[1] for datum in modal_data]
    modal_num = len(modal_data)

    model = CSCF(attr_dim=attr_dim, hid_dim=hid_dim, proto_dim=proto_dim, modal_num=modal_num, lr=lr,
                 weight_decay=args.weight_decay, device=device, topk=args.topk, num_class=num_class)
    if args.modal_lrs is not None or args.modal_weight_decays is not None:
        modal_lrs = args.modal_lrs if args.modal_lrs is not None else [lr] * modal_num
        modal_weight_decays = (args.modal_weight_decays if args.modal_weight_decays is not None
                               else [args.weight_decay] * modal_num)
        if len(modal_lrs) != modal_num or any(value <= 0 for value in modal_lrs):
            raise ValueError('modal_lrs must contain one positive learning rate per modality')
        if len(modal_weight_decays) != modal_num or any(value < 0 for value in modal_weight_decays):
            raise ValueError('modal_weight_decays must contain one nonnegative value per modality')
        model.optim = torch.optim.Adam([
            {'params': module.params_to_update, 'lr': value, 'weight_decay': decay}
            for module, value, decay in zip(model.single_mode_modules, modal_lrs, modal_weight_decays)
        ], weight_decay=args.weight_decay)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total trainable parameters: {total_params}")
    print(f"Training profile: {args.training_profile}, config: {args.config_source}")
    print(f"Training: epochs={epochs}, batch_size={args.batch_size}, shuffle={args.shuffle}, "
          f"modal_lrs={[group['lr'] for group in model.optim.param_groups]}, "
          f"weight_decays={[group['weight_decay'] for group in model.optim.param_groups]}, "
          f"scheduler={args.scheduler}, drop_last={args.drop_last}")

    print(f'------------------ Start training ------------------')
    model.train_loop(train_dataloader, attrs_multi, similarity_mat, epochs=epochs,
                     scheduler=args.scheduler, warmup_epochs=args.warmup_epochs,
                     min_lr_ratio=args.min_lr_ratio, grad_clip=args.grad_clip)

    model.training_config = dict(vars(args), resolved_lr=lr,
                                 resolved_modal_lrs=(args.modal_lrs if args.modal_lrs is not None
                                                    else [lr] * modal_num),
                                 resolved_weight_decays=[group['weight_decay'] for group in model.optim.param_groups])
    if save_model:
        os.makedirs(save_dir, exist_ok=True)
        torch.save(model, os.path.join(save_dir, f'{args.dataset}_{args.text_features}.pth'))
        with open(os.path.join(save_dir, f'{args.dataset}_{args.text_features}.json'), 'w', encoding='utf-8') as stream:
            json.dump({'config': model.training_config, 'history': model.training_history}, stream, indent=2)
        print(f'------------------ Finished training, saved to {save_dir} ------------------')
    else:
        print('------------------ Finished training ------------------')

    return model

def test_step(args, model, data_folder, dataset, text_features, type, mode):
    test_dataset = MultiViewDataset(data_folder=data_folder, dataset=dataset, text_features=text_features, type=type, mode=mode, train=False)
    test_attrs_multi = test_dataset.attrs_multi
    test_similarity_mat = test_dataset.similarity_mat
    seen_masks = test_dataset.seen_masks
    test_dataloader = DataLoader(test_dataset, batch_size=2048, collate_fn=custom_collate_fn)

    if dataset == 'ActivityNet':
        balanced_weight = 5e-3
    elif dataset == 'Food101':
        balanced_weight = 3e-3
    elif dataset == 'SUNRGBD':
        balanced_weight = 1.5e-2
    else:
        raise ValueError('Invalid dataset name')

    test_preds = []
    test_targets = []

    with torch.no_grad():
        for batch in test_dataloader:
            inputs = [inputs for inputs in batch[0]]
            masks = [masks for masks in batch[2]]
            targets = torch.tensor(batch[1])

            probs = model(inputs, masks, test_attrs_multi, test_similarity_mat, seen_masks, balanced_weight)

            test_preds.extend(probs.argmax(dim=1).cpu().numpy())
            test_targets.extend(targets.cpu().numpy())

            del inputs, masks, targets, probs
            torch.cuda.empty_cache()

    test_acc = np.mean(np.array(test_preds) == np.array(test_targets))
    return test_acc


def test(args, model):
    run_folder = os.getcwd()
    data_folder = os.path.join(run_folder, args.data_root)
    seed = args.seed
    test_type = args.test_type
    dataset = args.dataset
    text_features = args.text_features
    np.random.seed(seed)
    torch.manual_seed(seed)

    print(f'------------------ Test_type: {test_type} ------------------')
    if test_type in ['same', 'contra', 'data1', 'data2', 'total']:
        test_acc_audio = test_step(args, model, data_folder, dataset, text_features, test_type, mode='audio')
        test_acc_video = test_step(args, model, data_folder, dataset, text_features, test_type, mode='video')
        test_acc_multi = test_step(args, model, data_folder, dataset, text_features, test_type, mode='multi')
        if test_type == 'same':
            print(f'A_s acc: {test_acc_audio * 100: .2f}')
            print(f'B_s acc: {test_acc_video * 100: .2f}')
            print(f'A_s & B_s acc: {test_acc_multi * 100: .2f}')
        elif test_type == 'contra':
            print(f'A_u acc: {test_acc_audio * 100: .2f}')
            print(f'B_u acc: {test_acc_video * 100: .2f}')
            print(f'A_u & B_u acc: {test_acc_multi * 100: .2f}')
        elif test_type == 'data1':
            print(f'A_s acc: {test_acc_audio * 100: .2f}')
            print(f'B_u acc: {test_acc_video * 100: .2f}')
            print(f'A_s + B_u acc: {test_acc_multi * 100: .2f}')
        elif test_type == 'data2':
            print(f'A_u acc: {test_acc_audio * 100: .2f}')
            print(f'B_s acc: {test_acc_video * 100: .2f}')
            print(f'A_u + B_s acc: {test_acc_multi * 100: .2f}')
        else:
            print(f'A_all acc: {test_acc_audio * 100: .2f}')
            print(f'B_all acc: {test_acc_video * 100: .2f}')
            print(f'A_all + B_all acc: {test_acc_multi * 100: .2f}')
    elif test_type == 'mixture':
        test_acc_multi = test_step(args, model, data_folder, dataset, text_features, test_type, mode='multi')
        print(f'A_s & B_s & A_u & B_u & (A_all + B_all) acc: {test_acc_multi * 100: .2f}')
    elif test_type == 'all':
        metrics = {}
        for name, label, protocol, mode in ALL_METRICS:
            metrics[name] = float(test_step(args, model, data_folder, dataset, text_features,
                                            type=protocol, mode=mode) * 100)
            print(f'{label} acc: {metrics[name]: .2f}')
        return metrics
    else:
        raise ValueError(f'test_type should be one of same, contra, data1, data2, total, mixture or all')


ALL_METRICS = (
    ('same', 'A_s & B_s', 'same', 'multi'),
    ('contra', 'A_u & B_u', 'contra', 'multi'),
    ('audio_data1', 'A_s', 'data1', 'audio'),
    ('video_data1', 'B_u', 'data1', 'video'),
    ('total_data1', 'A_s + B_u', 'data1', 'multi'),
    ('audio_data2', 'A_u', 'data2', 'audio'),
    ('video_data2', 'B_s', 'data2', 'video'),
    ('total_data2', 'A_u + B_s', 'data2', 'multi'),
    ('audio', 'A_all', 'total', 'audio'),
    ('video', 'B_all', 'total', 'video'),
    ('total', 'A_all + B_all', 'total', 'multi'),
    ('mixture', 'mixture', 'mixture', 'multi'),
)


if __name__ == '__main__':
    args = get_parser()
    model = train(args)
    test(args, model)
