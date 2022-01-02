
from torch.utils.data import DataLoader

def create_dataloaders(args):
    """create dataloader"""
    if args.dataset == 'AID':
        from data.aid import AIDataset
        training_set = AIDataset(args, root_dir='E:/leisen-workspace/codelife/super-resolution/dataset/SR for remote sensing/AID_dataset',  # AID_dataset
                                 train=True)
        val_set = AIDataset(args, root_dir='E:/leisen-workspace/codelife/super-resolution/dataset/SR for remote sensing/AID_dataset',
                            train=False)
    elif args.dataset == 'UCMerced':
        from data.ucmerced import UCMercedDataset
        training_set = UCMercedDataset(args, root_dir='F:/research/dataset/SR for remote sensing/UCMerced_LandUse/slipt-train-val/train-x3',
                                 train=True)
        val_set = UCMercedDataset(args, root_dir='F:/research/dataset/SR for remote sensing/UCMerced_LandUse/slipt-train-val/val-x3',
                            train=False)
    elif args.dataset == 'DIV2K':
        from data.div2k import DIV2KDataset
        training_set = DIV2KDataset(args, root_dir='F:/research/dataset/SR for natural image/DIV2K_train',
                                    train=True)
        val_set = DIV2KDataset(args, root_dir='F:/research/dataset/SR for natural image/valid',
                               train=False)
    else:
        raise NotImplementedError(
            'Wrong dataset name %s ' % args.dataset)

    dataloaders = {'train': DataLoader(training_set, batch_size=args.batch_size,
                                 shuffle=True, num_workers=0),  # args.n_threads
                   'val': DataLoader(val_set, batch_size=args.batch_size,
                                 shuffle=True, num_workers=0)}  # args.n_threads

    return dataloaders



















