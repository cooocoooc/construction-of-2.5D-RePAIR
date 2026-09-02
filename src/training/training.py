import sys
sys.path.append('../src/data') # add self-defined script path
sys.path.append('../src/models')
from dataset_puzz import PuzzleDataset, collate_puzzle
from diffusion import PatchEncoder, FragmentAggregator, CoarseDenoiser, FineDenoiser, DiffusionScheduler
import time
import torch
import torch.nn.functional as nnFun
from torch.utils.data import DataLoader, random_split

from trainer_puzz import TrainerPuzz
import argparse


def parse_args():
    # create the argument parser
    parser = argparse.ArgumentParser(description="Training script for 2D RePAIR")
    parser.add_argument("--data_root", type=str, default="../raw_data/2D_Fragments/2D_Images/assembled_objects", help="Root directory of the dataset")
    parser.add_argument("--ground_root", type=str, default="../raw_data/2D_Fragments/2D_Ground_Truth", help="Root directory of the ground truth data")
    parser.add_argument("--patch_size", type=int, default=256, help="Size of the patches")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loading")
    parser.add_argument("--T", type=int, default=20, help="Number of diffusion steps")
    return parser.parse_args()

def main():
    args = parse_args()

    # paramaters
    data_root = args.data_root
    ground_root = args.ground_root
    patch_size = args.patch_size
    epochs = args.epochs
    batch_size = args.batch_size
    num_workers = args.num_workers
    T = args.T
    lr = args.lr
    ground_root = args.ground_root
    patch_size = args.patch_size
    expand = 8
    max_frag_size = 512

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"using device:{device}")

    # dataset
    full_dataset = PuzzleDataset(
        imgs_dir = data_root,
        ground_dir = ground_root,
        patch_size = patch_size,
        expand = expand,
        max_frag_size = max_frag_size,
        preload = True
    )

    total_len = len(full_dataset)
    train_len = int(0.7 * total_len)
    val_len = int(0.15 * total_len)
    test_len = total_len - train_len - val_len
    torch.manual_seed(42)
    train_set, val_set, test_set = random_split(full_dataset,[train_len, val_len, test_len])
    train_loader = DataLoader(train_set, batch_size = batch_size, shuffle = True,
                            collate_fn = collate_puzzle, num_workers = num_workers, pin_memory = True)
    val_loader = DataLoader(val_set, batch_size = batch_size, shuffle = False,
                            collate_fn = collate_puzzle, num_workers = num_workers, pin_memory = True)
    test_loader = DataLoader(test_set, batch_size = batch_size, shuffle = False,
                            collate_fn = collate_puzzle, num_workers = num_workers, pin_memory = True)

    encoder = PatchEncoder(in_channel = 4, embed_dim = 128).to(device)
    aggregator = FragmentAggregator(embed_dim = 128).to(device)
    model_coarse = CoarseDenoiser(feature_dim = 128, dim_model = 128).to(device)
    model_fine = FineDenoiser(feature_dim = 128, dim_model = 128).to(device)

    optimizer_coarse = torch.optim.Adam(model_coarse.parameters(), lr = lr)
    optimizer_fine = torch.optim.Adam(model_fine.parameters(), lr = lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_coarse, T_max = len(train_loader) * epochs)

    diffusion = DiffusionScheduler(T=T)

    print("\n === training start=====")
    start_time = time.time()
    trainer = TrainerPuzz(model_coarse, model_fine, encoder, aggregator, train_loader, val_loader,
                   optimizer_coarse, optimizer_fine, scheduler, diffusion, device)
    history = trainer.fit(epochs,'best_model.pth')    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"total time:{elapsed_time}")

     # test
    print("\n === testing on best model=====")
    ckpt = torch.load('best_model.pth')
    model_coarse.load_state_dict(ckpt['model_coarse_state_dict'])
    model_fine.load_state_dict(ckpt['model_fine_state_dict'])
    encoder.load_state_dict(ckpt['encoder_state_dict'])
    aggregator.load_state_dict(ckpt['aggregator_state_dict'])

    test_coarse, test_fine = trainer.evaluate(1, model_coarse, model_fine, encoder, aggregator, test_loader, diffusion, device)
    print(f"test loss: -coarse:{test_coarse:.4f}, -fine:{test_fine:.4f}")

    trainer.plt_loss_curves()


if __name__ == '__main__':
    main()