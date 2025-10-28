import torch
import os
from tqdm import tqdm
from torchattacks import FGSM, PGD, MultiAttack
import torchvision
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import pickle
from trainer_dif import TrainerMultiple
from pathlib import Path
import torch.nn as nn
from attack import Attack

from utils.util import OptionConfigurator, fix_randomseed
from model.cnn import build_classifier_from_opt

class PGD_Framework:
    
    def __init__(self, vit_model, dif_model, eps=8/255):
        self.model = vit_model
        self.trainer = dif_model
        self.eps = eps
        self.device = dif_model.device



    def __call__(self, images, labels):
        return self.forward(images, labels)
    
    
    def forward(self, images, labels):
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self.targeted:
            target_labels = self.get_target_label(images, labels)
        
        
        # 2. Fingerprint
        fingerprint = self.trainer.fingerprint.to(self.device)
        fingerprint = fingerprint.repeat((self.trainer.batch_size, 1, 1, 1))
        
        residuals = self.trainer.denoiser.denoise(images).float()
        
        # 3. Correlation
        corr = self.trainer.corr_fun(fingerprint, residuals)
        # 4. Contrastive loss
        corr_val = corr.mean((1, 2, 3))
        #print("corr:", corr_val)
        #print("isnan:", torch.isnan(corr_val).any(), "isinf:", torch.isinf(corr_val).any())
        dif_cost = self.trainer.loss_contrast(corr.mean((1, 2, 3)), labels) / self.trainer.m
        
        #print("loss after loss_contrast:", loss)
        #loss = loss.mean()     

        vit_loss = nn.CrossEntropyLoss()
        adv_images = images.clone().detach()

        if self.random_start:
            # Starting at a uniformly random point
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(
                -self.eps, self.eps
            )
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        for _ in range(self.steps):
            adv_images.requires_grad = True
            outputs = self.get_logits(adv_images)

            # Calculate loss
            if self.targeted:
                vit_cost = -vit_loss(outputs, target_labels)
            else:
                vit_cost = vit_loss(outputs, labels)


            cost = (vit_cost + dif_cost)/max(vit_cost + dif_cost)
            
            # Update adversarial images
            grad = torch.autograd.grad(
                cost, adv_images, retain_graph=False, create_graph=False
            )[0]

            adv_images = adv_images.detach() + self.alpha * grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        return adv_images

    '''
    def forward2(self, images, labels):
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        images.requires_grad = True

        # 1. Denoise
       
        # 2. Fingerprint
        fingerprint = self.trainer.fingerprint.to(self.device)
        fingerprint = fingerprint.repeat((self.trainer.batch_size, 1, 1, 1))
        
        residuals = self.trainer.denoiser.denoise(images).float()
        
        # 3. Correlation
        corr = self.trainer.corr_fun(fingerprint, residuals)
        # 4. Contrastive loss
        corr_val = corr.mean((1, 2, 3))
        #print("corr:", corr_val)
        #print("isnan:", torch.isnan(corr_val).any(), "isinf:", torch.isinf(corr_val).any())
        loss = self.trainer.loss_contrast(corr.mean((1, 2, 3)), labels) / self.trainer.m
        
        #print("loss after loss_contrast:", loss)
        #loss = loss.mean() 
        
        
        # 5. PGD update
        grad = torch.autograd.grad(
            loss, images, retain_graph=False, create_graph=False
        )[0]
        
        #print(grad.sign())
        
        adv_images = images + self.eps * grad.sign()
        adv_images = torch.clamp(adv_images, 0, 1).detach()
        return adv_images
    '''

def From_vit():
    fix_randomseed(42)
    opt = OptionConfigurator().parse_options()

    device = torch.device(opt.device)
    
    print('[  Current classifier : {}  ]\n'.format(opt.classifier))

    root = './results'
    load_path = os.path.join(root, opt.dst, '{}_classifier/'.format(opt.classifier))

    model = build_classifier_from_opt(opt)
    param = torch.load(os.path.join(load_path, 'model.pt'), map_location = torch.device('cpu'))
    model.load_state_dict(param)
    model = model.to(device)
    return model

def From_dif():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # TrainerMultiple 객체 준비
    #check_dir = Path(r"C:\DEMID/GitHub/konkuk_GP_CAR_Team\STIG_Testing\STIG-main\DIF_pytorch_official-master/checks\sd15_cat")
    check_dir = Path(r"C:\DEMID/GitHub/konkuk_GP_CAR_Team\STIG_Testing\STIG-main\DIF_pytorch_official-master/checks\sd15_human")
    with open(check_dir / "train_hypers.pt", 'rb') as pickle_file:
        hyper_pars = pickle.load(pickle_file)
    hyper_pars['Device'] = device
    hyper_pars['Batch Size'] = 1
    #hyper_pars['Inp. Channel'] = 3
    trainer = TrainerMultiple(hyper_pars)
    trainer.load_stats(check_dir / f"chk_10.pt")
    
    return trainer

def From_STIG():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 이미지 불러오기 및 전처리
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    from glob import glob
    from PIL import Image

    image_paths = sorted(glob(r"C:\DEMID/GitHub/konkuk_GP_CAR_Team\STIG_Testing\STIG-main\datasets/human_inference/fake/*.png"))
    images = [transform(Image.open(p).convert('RGB')) for p in image_paths]
    images = torch.stack(images).to(device)
    labels = torch.ones(len(images), dtype=torch.long).to(device)  # 필요에 따라 라벨 수정
    
    return images, labels

def PGD_Layer():
    
    vit_model = From_vit()
    dif_trainer = From_dif()
    atk = PGD_Framework(vit_model, dif_trainer, eps=8/255)

    images, labels = From_STIG()
    
    torch.cuda.empty_cache()
    batch_size = 1  # GPU 메모리에 맞게 조절

    for i in tqdm(range(0, len(images), batch_size), desc="PGD Attack"):
        batch_images = images[i:i+batch_size]
        batch_labels = labels[i:i+batch_size]
        adv_batch = atk(batch_images, batch_labels)
        for j, (orig_img, adv_img) in enumerate(zip(batch_images, adv_batch)):
            # 정규화 없이 바로 clamp 후 저장
            orig_img_cpu = torch.clamp(orig_img.detach().cpu(), 0, 1)
            adv_img_cpu = torch.clamp(adv_img.detach().cpu(), 0, 1)
            diff = (adv_img_cpu - orig_img_cpu).abs().mean().item()
            #print(f"이미지 변화량: {diff:.6f}")
            img_pil = transforms.ToPILImage()(adv_img_cpu)
    


'''
if __name__ == '__main__' :
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # TrainerMultiple 객체 준비
    #check_dir = Path(r"C:\DEMID/GitHub/konkuk_GP_CAR_Team\STIG_Testing\STIG-main\DIF_pytorch_official-master/checks\sd15_cat")
    check_dir = Path(r"C:\DEMID/GitHub/konkuk_GP_CAR_Team\STIG_Testing\STIG-main\DIF_pytorch_official-master/checks\sd15_human")
    with open(check_dir / "train_hypers.pt", 'rb') as pickle_file:
        hyper_pars = pickle.load(pickle_file)
    hyper_pars['Device'] = device
    hyper_pars['Batch Size'] = 1
    #hyper_pars['Inp. Channel'] = 3
    trainer = TrainerMultiple(hyper_pars)
    trainer.load_stats(check_dir / f"chk_10.pt")

    # 이미지 불러오기 및 전처리
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    from glob import glob
    from PIL import Image

    image_paths = sorted(glob(r"C:\DEMID/GitHub/konkuk_GP_CAR_Team\STIG_Testing\STIG-main\datasets/human_inference/fake/*.png"))
    images = [transform(Image.open(p).convert('RGB')) for p in image_paths]
    images = torch.stack(images).to(device)
    labels = torch.ones(len(images), dtype=torch.long).to(device)  # 필요에 따라 라벨 수정

    save_dir = 'FGSM_PGD/dif-human'
    os.makedirs(save_dir, exist_ok=True)
    
    torch.cuda.empty_cache()
    batch_size = 2  # GPU 메모리에 맞게 조절
    
    atk = PGD_Framework(trainer, eps=8/255)

    for i in tqdm(range(0, len(images), batch_size), desc="FGSM Attack"):
        batch_images = images[i:i+batch_size]
        batch_labels = labels[i:i+batch_size]
        adv_batch = atk(batch_images, batch_labels)
        for j, (orig_img, adv_img) in enumerate(zip(batch_images, adv_batch)):
            # 정규화 없이 바로 clamp 후 저장
            orig_img_cpu = torch.clamp(orig_img.detach().cpu(), 0, 1)
            adv_img_cpu = torch.clamp(adv_img.detach().cpu(), 0, 1)
            diff = (adv_img_cpu - orig_img_cpu).abs().mean().item()
            #print(f"이미지 변화량: {diff:.6f}")
            img_pil = transforms.ToPILImage()(adv_img_cpu)
            img_pil.save(os.path.join(save_dir, f"adv_{i+j:04d}.jpg"))
'''