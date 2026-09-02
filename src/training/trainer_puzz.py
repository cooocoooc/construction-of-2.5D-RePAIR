
import csv
import os

import torch
from tqdm import tqdm
import torch.nn.functional as nnFun
import matplotlib.pyplot as plt

class TrainerPuzz:

    def __init__(self, model_coarse, model_fine, encoder, aggregator, train_loader,val_loader,
                   optimizer_coarse, optimizer_fine, scheduler, diffusion, device, auto_resume=True):
        
        self.model_coarse = model_coarse
        self.model_fine = model_fine
        self.encoder = encoder
        self.aggregator = aggregator
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer_coarse = optimizer_coarse
        self.optimizer_fine = optimizer_fine
        self.scheduler = scheduler
        self.diffusion = diffusion
        self.device = device

        self.start_epoch = 0
        self.best_val_loss = float('inf')
        self.history = {'epoch': [], 'train-coarse': [], 'train-fine': [], 'val-coarse': [], 'val-fine': []}  # To store training and validation loss history

        if auto_resume:
            self.auto_resume()  # Automatically resume from checkpoint if available

        # create a log file to record the training and validation loss
        if not os.path.isfile("training_log.txt"):
            with open("training_log.txt", "w", encoding='utf-8', newline='') as f:
                write_header = "epoch,train_loss_coarse,train_loss_fine,val_loss_coarse,val_loss_fine\n"
                writer = csv.writer(f)
                writer.writerow(write_header.strip().split(','))

    def auto_resume(self):
        """
        Automatically resume training from a checkpoint if it exists.
        """
        ckpt_path = "latest_model.pth"
        if os.path.isfile(ckpt_path):
            checkpoint = torch.load(ckpt_path, map_location=self.device)
            self.model_coarse.load_state_dict(checkpoint['model_coarse_state_dict'])
            self.model_fine.load_state_dict(checkpoint['model_fine_state_dict'])
            self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
            self.aggregator.load_state_dict(checkpoint['aggregator_state_dict'])
            self.optimizer_coarse.load_state_dict(checkpoint['optimizer_coarse_state_dict'])
            self.optimizer_fine.load_state_dict(checkpoint['optimizer_fine_state_dict'])
            self.start_epoch = checkpoint.get('epoch', 0) + 1
            self.history = checkpoint.get('history', [])
            self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))

            print(f"Resumed training from checkpoint: {ckpt_path}")
        else:
            print("No checkpoint found. Starting training from scratch.")

    def save_checkpoint(self, epoch, is_best = False):
        """
        Save the current state of the model and optimizer to a checkpoint.
        """
        torch.save({
            'epoch': epoch,
            'model_coarse_state_dict': self.model_coarse.state_dict(),
            'model_fine_state_dict': self.model_fine.state_dict(),
            'encoder_state_dict': self.encoder.state_dict(),
            'aggregator_state_dict': self.aggregator.state_dict(),
            'optimizer_coarse_state_dict': self.optimizer_coarse.state_dict(),
            'optimizer_fine_state_dict': self.optimizer_fine.state_dict(),
            'history': self.history,
            'best_val_loss': self.best_val_loss

        }, 'latest_model.pth')

        print(f"Checkpoint saved at epoch {epoch} to latest_model.pth")

        if is_best:
            torch.save({
                'epoch': epoch,
                'model_coarse_state_dict': self.model_coarse.state_dict(),
                'model_fine_state_dict': self.model_fine.state_dict(),
                'encoder_state_dict': self.encoder.state_dict(),
                'aggregator_state_dict': self.aggregator.state_dict(),

            }, 'best_model.pth')
            print(f"Best model saved at epoch {epoch} to best_model.pth")

    def log_to_csv(self, epoch, train_loss_coarse, train_loss_fine, val_loss_coarse, val_loss_fine):
        """
        Log the training and validation loss to a CSV file.
        """
        with open("training_log.txt", "a", encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss_coarse, train_loss_fine, val_loss_coarse, val_loss_fine])

    def train_one_epoch(self, epochs):
    
        self.model_coarse.train()
        self.model_fine.train()
        self.encoder.train()
        self.aggregator.train()

        total_loss_coarse = 0.0
        total_loss_fine = 0.0
        num_batches = 0

        process_bar = tqdm(self.train_loader, desc=f'Epoch {epochs} Training')

        for batch in process_bar:
            patches = batch['patches'].to(self.device) #[B, N, C, H, W]
            patch_centers = batch['patch_centers'].to(self.device) #[B, N, 2]
            frag_centers = batch['frag_centers'].to(self.device) #[B, M, 2]
            offset_labels = batch['offset_labels'].to(self.device) #[B, N, 2]
            patch_frag_idx = batch['patch_frag_idx'].to(self.device)#[B, N]
            mask_patch = batch['mask_patch'].to(self.device)#[B, N]
            mask_frag = batch['mask_frag'].to(self.device)#[B, M]
            num_frags = batch['num_frags'][0].item()
            max_frag_num = batch['num_frags'].max().item()
            max_patch_num = batch['num_patches'].max().item()

            B, N, C, patch_H, path_W = patches.shape
            
            # encoder patches
            patch_features = self.encoder(patches) # [B*N, D]
            patch_features = patch_features.view(B, N, -1) #[B, N, D]

            # fragment features
            frag_features = self.aggregator(patch_features, patch_frag_idx, mask_patch, max_frag_num) # [B, M, D]

            # random step
            t = torch.randint(0, self.diffusion.T, (B,), device = self.device)
            
            # coarse
            noise_coarse = torch.randn_like(frag_centers)
            noisy_frag = self.diffusion.sqrt_sample(frag_centers, t, noise_coarse)
            pre_noise_coarse, pred_angle = self.model_coarse(noisy_frag, frag_features, t)
            # angle label(assume 0)
            angle_labels = torch.zeros(B, max_frag_num, dtype = torch.long, device = self.device)
            mask_frag_2d = mask_frag.unsqueeze(-1).expand(-1, -1, 2) # [B, M, 2]
            loss_coarse = nnFun.mse_loss(pre_noise_coarse[mask_frag_2d], noise_coarse[mask_frag_2d])
            loss_angle = nnFun.cross_entropy(pred_angle.reshape(-1, 4), angle_labels.reshape(-1))
            loss_coarse = loss_coarse + 0.1 * loss_angle

            # fine
            frag_centers_expand = torch.zeros(B, N, 2, device = self.device)#[B, N, 2]
            for each_batch in range(B):
                valid_region = mask_patch[each_batch]
                if valid_region.any():
                    valid_index = patch_frag_idx[each_batch][valid_region]
                    frag_centers_expand[each_batch][valid_region] = frag_centers[each_batch][valid_index]
            noise_fine = torch.randn_like(offset_labels)
            noise_off = self.diffusion.sqrt_sample(offset_labels, t, noise_fine)
            pred_noise_fine = self.model_fine(noise_off, patch_features, frag_centers_expand, t)

            # loss of valid patch
            mask_patch_2d = mask_patch.unsqueeze(-1).expand(-1, -1, 2) # [B, N, 2]
            loss_fine = nnFun.mse_loss(pred_noise_fine[mask_patch_2d], noise_fine[mask_patch_2d])

            # total loss
            total_loss = loss_coarse + 0.3 * loss_fine

            self.optimizer_coarse.zero_grad()
            self.optimizer_fine.zero_grad()
            total_loss.backward()
            self.optimizer_coarse.step()
            self.optimizer_fine.step()

            total_loss_coarse += loss_coarse.item()
            total_loss_fine += loss_fine.item()
            num_batches += 1

            process_bar.set_postfix({
                'loss_coarse': loss_coarse.item(),
                'loss_fine': loss_fine.item()})
        self.scheduler.step()
        return total_loss_coarse / num_batches, total_loss_fine / num_batches

    @torch.no_grad()
    def evaluate(self, epochs, model_coarse, model_fine, encoder, aggregator, dataloader, diffusion, device):

        model_coarse.eval()
        model_fine.eval()
        encoder.eval()
        aggregator.eval()

        total_loss_coarse = 0.0
        total_loss_fine = 0.0
        num_batches = 0

        process_bar = tqdm(dataloader, desc=f'Epoch {epochs} Evaluating')

        for batch in process_bar:
            patches = batch['patches'].to(device) #[B, N, C, H, W]
            patch_centers = batch['patch_centers'].to(device) #[B, N, 2]
            frag_centers = batch['frag_centers'].to(device) #[B, M, 2]
            offset_labels = batch['offset_labels'].to(device) #[B, N, 2]
            patch_frag_idx = batch['patch_frag_idx'].to(device)#[B, N]
            mask_patch = batch['mask_patch'].to(device)#[B, N]
            mask_frag = batch['mask_frag'].to(device)#[B, M]
            num_frags = batch['num_frags'][0].item()
            max_frag_num = batch['num_frags'].max().item()
            max_patch_num = batch['num_patches'].max().item()

            B, N, C, patch_H, path_W = patches.shape

            
            # encoder patches
            patch_features = encoder(patches) # [B*N, D]
            patch_features = patch_features.view(B, N, -1) #[B, N, D]

            # fragment features
            frag_features = aggregator(patch_features, patch_frag_idx, mask_patch, max_frag_num) # [B, M, D]

            # random step
            t = torch.randint(0, diffusion.T, (B,), device = device)
            
            # coarse
            noise_coarse = torch.randn_like(frag_centers)
            noisy_frag = diffusion.sqrt_sample(frag_centers, t, noise_coarse)
            pre_noise_coarse, pred_angle = model_coarse(noisy_frag, frag_features, t)
            # angle label(assume 0)
            # angle_labels = torch.zeros(B, max_frag_num, dtype = torch.long, device = device)
            mask_frag_2d = mask_frag.unsqueeze(-1).expand(-1, -1, 2) # [B, M, 2]
            loss_coarse = nnFun.mse_loss(pre_noise_coarse[mask_frag_2d], noise_coarse[mask_frag_2d])
            #loss_angle = nnFun.cross_entropy(pred_angle.reshape(-1, 4), angle_labels.reshape(-1))
            #loss_coarse = loss_coarse + 0.1 * loss_angle

            # fine
            frag_centers_expand = torch.zeros(B, N, 2, device = device)#[B, N, 2]
            for each_batch in range(B):
                valid_region = mask_patch[each_batch]
                if valid_region.any():
                    valid_index = patch_frag_idx[each_batch][valid_region]
                    frag_centers_expand[each_batch][valid_region] = frag_centers[each_batch][valid_index]
            noise_fine = torch.randn_like(offset_labels)
            noise_off = diffusion.sqrt_sample(offset_labels, t, noise_fine)
            pred_noise_fine = model_fine(noise_off, patch_features, frag_centers_expand, t)

            # loss of valid patch
            mask_patch_2d = mask_patch.unsqueeze(-1).expand(-1, -1, 2) # [B, N, 2]
            loss_fine = nnFun.mse_loss(pred_noise_fine[mask_patch_2d], noise_fine[mask_patch_2d])

            # total loss
            total_loss = loss_coarse + 0.3 * loss_fine

            total_loss_coarse += loss_coarse.item()
            total_loss_fine += loss_fine.item()
            num_batches += 1

            process_bar.set_postfix({
                'loss_coarse': loss_coarse.item(),
                'loss_fine': loss_fine.item()})
        return total_loss_coarse / num_batches, total_loss_fine / num_batches
    
    def fit(self, max_epochs, save_path):
        """
        Train the model for a specified number of epochs and save the best model based on validation loss."""
        for epoch in range(self.start_epoch, max_epochs):
            train_loss_coarse, train_loss_fine = self.train_one_epoch(epoch)
            val_loss_coarse, val_loss_fine = self.evaluate(epoch,
                                                            self.model_coarse, 
                                                            self.model_fine, 
                                                            self.encoder, 
                                                            self.aggregator,
                                                            self.val_loader, 
                                                            self.diffusion, 
                                                            self.device)
            print(f"Epoch {epoch}: Train Loss Coarse: {train_loss_coarse:.4f}, Train Loss Fine: {train_loss_fine:.4f}, Val Loss Coarse: {val_loss_coarse:.4f}, Val Loss Fine: {val_loss_fine:.4f}")

            self.history['epoch'].append(epoch)
            self.history['train-coarse'].append(train_loss_coarse)
            self.history['train-fine'].append(train_loss_fine)
            self.history['val-coarse'].append(val_loss_coarse)
            self.history['val-fine'].append(val_loss_fine)
            self.log_to_csv(epoch, train_loss_coarse, train_loss_fine, val_loss_coarse, val_loss_fine)

            # Save the model if validation loss improves
            is_best = val_loss_coarse + val_loss_fine < self.best_val_loss
            if is_best:
                self.best_loss = val_loss_coarse + val_loss_fine
                print(f"Model saved at epoch {epoch} with validation loss {self.best_loss:.4f}")

            # save checkpoint for every epoch
            self.save_checkpoint(epoch, is_best)
        return self.history

    def plt_loss_curves(self, save_path = 'loss_curves.png', is_save = True):
        """
        Plot the training and validation loss curves.
        """

        epochs = self.history['epoch']
        train_loss_coarse = self.history['train-coarse']
        train_loss_fine = self.history['train-fine']
        val_loss_coarse = self.history['val-coarse']
        val_loss_fine = self.history['val-fine']

        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(epochs, train_loss_coarse, label='Train Loss Coarse')
        plt.plot(epochs, val_loss_coarse, label='Val Loss Coarse')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Coarse Denoiser Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(epochs, train_loss_fine, label='Train Loss Fine')
        plt.plot(epochs, val_loss_fine, label='Val Loss Fine')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Fine Denoiser Loss')
        plt.legend()

        plt.tight_layout()
        if is_save:
            plt.savefig(save_path, dpi=300)
        plt.show()