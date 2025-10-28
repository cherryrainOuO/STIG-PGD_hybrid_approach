import os
import torch
from torch import nn
from torchsummary import summary
from .networks import NestedUNet, PatchSampleF, PatchDiscriminator
from .networks import get_norm_layer, init_net
from .patchnce import PatchNCELoss
from .loss import LFLoss, RectAverage, SSIM
from utils.processing import get_magnitude_fwt, img2fwt, fwt2polar, normalize, inverse_normalize, polar2img_fwt, image_normalize

from utils.util import OptionConfigurator, fix_randomseed
from model.cnn import build_classifier_from_opt
from trainer_dif import TrainerMultiple


'''
STIG model : The frequency domain method for enhancing the generated images.
Input : Real image & Generated image -> torch.Tensor
Output : Enhanced generated image -> torch.Tensor

Augument
- opt : option configurator
- device : torch.cude.device()
'''
class STIG_PGD(nn.Module) :
    
    def __init__(self, opt, device, vit_model=None, dif_trainer=None) :
        super(STIG_PGD, self).__init__()

        self.device = device
        self.lambda_PSD = None
        self.first_setting = False
        self.radial_profile = RectAverage(opt.size, self.device)

        self._load_argument_options(opt)
        self._build_modules()

        self.dc_pos = self.opt.size // 2 + 1
        self.sigma = opt.sigma
        self.in_channels = self.opt.input_nc
        self.out_channels = self.opt.output_nc
        
        self.vit_model = vit_model
        self.dif_trainer = dif_trainer
        
        # learnable radius (as fraction 0..1) + softmask sharpness and reg weight
        self.radius_param = nn.Parameter(torch.tensor(0.15))   # initial fraction (learnable)
        self.radius_gamma = 40.0                               # mask sharpness; 클수록 hard threshold 근사
        self.radius_reg_lambda = 1e-2                          # radius regularization weight
        # NOTE: include radius_param in optimizerG when building optimizers
        self._build_optimizers()
        

    def set_input(self, input, evaluation = False) :

        if not evaluation :
            self.image_A = input['noise']
            self.image_B = input['clean']

            self.image_A = self.image_A.to(self.device)
            self.image_B = self.image_B.to(self.device)

        else :
            self.image_A = input
            self.image_A = self.image_A.to(self.device)

    '''
    def to_frequency_domain(self, image) :
        fft = img2fft(image)
        magnitude, phase = fft2polar(fft)
        return magnitude, phase
    '''

    def to_frequency_domain(self, image) :
        fft = img2fwt(image)
        magnitude, phase = fwt2polar(fft)
        return magnitude, phase

    def forward(self):

        # Transformation image domain to frequency domain
        # Normalization magnitude to value of [-1.0, 1.0]
        self.real_A, self.real_A_phase = self.to_frequency_domain(self.image_A)
        self.real_B, self.real_B_phase = self.to_frequency_domain(self.image_B)

        self.real_A, self.A_vmax, self.A_vmin = normalize(self.real_A)
        self.real_B, self.B_vmax, self.B_vmin = normalize(self.real_B)

        # Forward noisy magnitude to generator network to get the clean version
        self.real = torch.cat((self.real_A, self.real_B), dim=0) if self.nce_idt else self.real_A

        self.fake = self.generator(self.real)
        self.fake_B = self.fake[:self.real_A.size(0)]
        if self.nce_idt:
            self.idt_B = self.fake[self.real_A.size(0):]
        else :
            self.idt_B = torch.ones_like(self.fake_B)

        fake_B_inverse_normed = inverse_normalize(self.fake_B, self.A_vmax, self.A_vmin)
        idt_B_inverse_normed = inverse_normalize(self.idt_B, self.B_vmax, self.B_vmin)


        self.denoised_image_A = polar2img_fwt(fake_B_inverse_normed, self.real_A_phase)
        self.identity_image_B = polar2img_fwt(idt_B_inverse_normed, self.real_B_phase)

        self.denoised_image_normed = image_normalize(self.denoised_image_A)
        self.denoised_mag = get_magnitude_fwt(self.denoised_image_normed)
        
        
        
        
        
        

        self.real_psd = self.radial_profile(self.real_B)
        self.denoised_psd = self.radial_profile(self.denoised_mag)

    def evaluation(self) :

        self.real_A, self.real_A_phase = self.to_frequency_domain(self.image_A)
        self.real_A, self.A_vmax, self.A_vmin = normalize(self.real_A)
        
        self.input_image_normed = image_normalize(self.image_A)
        self.input_mag = get_magnitude_fwt(self.input_image_normed)

        with torch.no_grad():
            self.fake_B = self.generator(self.real_A)

            fake_B_inverse_normed = inverse_normalize(self.fake_B, self.A_vmax, self.A_vmin)
            self.denoised_image_A = polar2img_fwt(fake_B_inverse_normed, self.real_A_phase)
            self.denoised_image_normed = image_normalize(self.denoised_image_A)
            self.denoised_mag = get_magnitude_fwt(self.denoised_image_normed)

    def update_optimizer(self) :
        
        self.forward()
        
        # --- apply PGD on low-frequency every training step / epoch (merged with STIG high-freq) ---
        # 조절값: eps, alpha, steps, radius 필요에 따라 변경
        try:
            labels_for_pgd = torch.ones(self.denoised_image_A.size(0), dtype=torch.long, device=self.device)
            # freeze vit params and set eval (PGD needs classifier forward/backward but shouldn't update vit weights)
            if self.vit_model is not None:
                self.vit_model.eval()
                for p in self.vit_model.parameters():
                    p.requires_grad = False

            adv = self.apply_pgd_on_lowfreq_with_ssim(
                self.denoised_image_A, self.image_B,
                classifier_fn=(lambda x: self.vit_model(x)) if self.vit_model is not None else None,
                labels=labels_for_pgd, dif_trainer=self.dif_trainer,
                eps=2/255, alpha=0.25/255, steps=2, radius=None,
                random_start=False, w_rec=1.0, w_vit=0.5, w_dif=0.5
            )
            # PGD 결과로 denoised image 교체하여 이후 loss 계산에 반영
            self.denoised_image_A = adv.detach()
            self.denoised_image_normed = image_normalize(self.denoised_image_A)
            self.denoised_mag = get_magnitude_fwt(self.denoised_image_normed)
            self.denoised_psd = self.radial_profile(self.denoised_mag)
        except Exception as e:
            # 실패 시 무시하고 진행
            print("apply_pgd_on_lowfreq_with_ssim failed:", e)
        # --- end PGD merge ---
        
        # log current learnable radius (fraction in 0..1)
        if hasattr(self, 'radius_param'):
            cur_radius_frac = torch.sigmoid(self.radius_param).item()
            # 원하는 로그 방식으로 변경 (print / logger)
            #print(f"[Radius] fraction={cur_radius_frac:.4f}, pixel_radius≈{cur_radius_frac * min(self.denoised_mag.shape[-2], self.denoised_mag.shape[-1]) / 2:.1f}")
        # --- continue training updates ---

        # update optimizerD
        self.set_requires_grad(self.discriminator, True)
        self.optimizerD.zero_grad()
        self.optimizerS.zero_grad()
        self.loss_D_total = self.compute_D_loss()
        self.loss_D_total.backward()
        self.optimizerD.step()
        self.optimizerS.step()

        # update optimizerG & optimizerM
        self.set_requires_grad(self.discriminator, False)
        self.optimizerG.zero_grad()
        self.optimizerM.zero_grad()
        self.loss_G = self.compute_G_loss()
        self.loss_G.backward()
        self.optimizerG.step()
        self.optimizerM.step()

        #print("radius grad:", self.radius_param.grad)  # None이면 optimizer에 포함되지 않았거나 detach됨
        


    def compute_G_loss(self) :

        fake = self.fake_B
        pred_fake = self.discriminator(fake)
        self.loss_G_GAN = self.criterionGAN(pred_fake, torch.ones_like(pred_fake))

        fake_psd = self.denoised_psd
        pred_s_fake = self.spectral_discriminator(fake_psd)
        self.loss_SG_GAN = self.criterionGAN(pred_s_fake, torch.ones_like(pred_s_fake))

        self.loss_identity = (1. - self.criterionSSIM(self.image_B, self.identity_image_B))
        self.loss_fake_identity = (1. - self.criterionSSIM(self.image_A, self.denoised_image_A))
        self.loss_LF = self.criterionLF(self.fake_B, self.real_A) + self.criterionLF(self.idt_B, self.real_B)

        self.loss_NCE = self.compute_NCE_loss(self.real_A, self.fake_B)

        if self.nce_idt :
            self.loss_idt_NCE = self.compute_NCE_loss(self.real_B, self.idt_B)
            self.loss_NCE_total = self.loss_NCE + self.loss_idt_NCE
        else :
            self.loss_idt_NCE = torch.tensor(0.)
            self.loss_NCE_total = self.loss_NCE

        self.loss_DC = 0.
        self.loss_PSD = 0.
        self.loss_SYM = 0.

        self.loss_G = self.loss_G_GAN * self.lambda_GAN + self.loss_SG_GAN * self.lambda_PSD + self.loss_NCE_total * self.lambda_NCE + 3.0 * self.loss_identity + 1.0 * self.loss_fake_identity + self.lambda_LF * self.loss_LF

        # optional radius regularizer: keep radius near initial value or desired prior
        radius_frac = torch.sigmoid(self.radius_param)
        self.loss_radius_reg = self.radius_reg_lambda * (radius_frac - 0.15)**2
        self.loss_G = self.loss_G + self.loss_radius_reg
        
        return self.loss_G


    def compute_D_loss(self) :

        pred_fake = self.discriminator(self.fake_B.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, torch.zeros_like(pred_fake))

        pred_real = self.discriminator(self.real_B)
        self.loss_D_real = self.criterionGAN(pred_real, torch.ones_like(pred_real))

        self.loss_D = (self.loss_D_real + self.loss_D_fake) * 0.5

        pred_s_fake = self.spectral_discriminator(self.denoised_psd.detach())
        self.loss_SD_fake = self.criterionGAN(pred_s_fake, torch.zeros_like(pred_s_fake))

        pred_s_real = self.spectral_discriminator(self.real_psd)
        self.loss_SD_real = self.criterionGAN(pred_s_real, torch.ones_like(pred_s_real))

        self.loss_SD = (self.loss_SD_real + self.loss_SD_fake) * 0.5

        return self.loss_D * 1.0 + self.loss_SD * 1.0


    def compute_NCE_loss(self, src, dst) :

        n_layers = len(self.nce_layers)

        feat_q = self.generator.encode_sample(dst)
        feat_k = self.generator.encode_sample(src)
        
        feat_k_pool, sample_ids = self.mlp(feat_k, self.n_patches, None)
        feat_q_pool, _ = self.mlp(feat_q, self.n_patches, sample_ids)

        total_nce_loss = 0.0

        for f_q, f_k, crit, nce_layer in zip(feat_q_pool, feat_k_pool, self.criterionNCE, self.nce_layers):
            loss = crit(f_q, f_k) * self.opt.lambda_NCE
            total_nce_loss += loss.mean()

        return total_nce_loss / n_layers


    def _build_modules(self) :

        self.generator = NestedUNet(self.input_nc, self.output_nc, spatial = False)
        self.discriminator = PatchDiscriminator(self.input_nc).to(self.device)
        self.mlp = PatchSampleF(use_mlp=True, nc=self.n_patches, device = self.device).to(self.device)
        
        self.generator = init_net(self.generator)
        self.discriminator = init_net(self.discriminator)
        self.mlp = init_net(self.mlp)

        self.criterionGAN = nn.MSELoss()
        self.criterionNCE = [PatchNCELoss(self.opt)]
        self.criterionIDT = nn.L1Loss()
        self.criterionLF = LFLoss(self.opt)
        self.criterionSSIM = SSIM(window_size = 35)

    def _build_optimizers(self) :

        # include radius_param so it is updated with generator
        self.optimizerG = torch.optim.Adam(list(self.generator.parameters()) + [self.radius_param], lr=self.opt.lr, betas=(self.beta1, self.beta2))
        self.optimizerD = torch.optim.Adam(self.discriminator.parameters(), lr=self.opt.lr, betas=(self.beta1, self.beta2))

    def _load_argument_options(self, opt) :

        self.opt = opt
        self.norm = get_norm_layer(self.opt.norm)
        self.n_patches = self.opt.num_patch
        self.beta1 = self.opt.beta1
        self.beta2 = self.opt.beta2
        self.input_nc = self.opt.input_nc
        self.output_nc = self.opt.output_nc
        self.nce_idt = self.opt.nce_idt
        self.lambda_GAN = self.opt.lambda_GAN
        self.lambda_NCE = self.opt.lambda_NCE
        self.lambda_PSD = self.opt.lambda_PSD
        self.lambda_LF = self.opt.lambda_LF
        self.lambda_identity = self.opt.lambda_identity
        self.nce_layers = [int(i) for i in self.opt.nce_layers.split(',')]

    def save_checkpoint(self, save_path, n, name = None) :

        checkpoint = {
            'Generator' : self.generator.state_dict(),
            'Discriminator' : self.discriminator.state_dict(),
            'MLP' : self.mlp.state_dict(),
            'PSD' : self.spectral_discriminator.state_dict()
        }
        
        # save learnable radius_param if exists
        if hasattr(self, 'radius_param'):
            checkpoint['radius_param'] = self.radius_param.detach().cpu()

        if name is None :
            file_name = 'parameters_{}_epoch.pt'.format(n)
        else :
            file_name = name

        torch.save(checkpoint, os.path.join(save_path, file_name))

    def load_checkpoint(self, load_path) :
        
        checkpoint = torch.load(load_path)
        self.generator.load_state_dict(checkpoint['Generator'])
        
        # load radius_param if present
        if 'radius_param' in checkpoint and hasattr(self, 'radius_param'):
            self.radius_param.data.copy_(checkpoint['radius_param'].to(self.device))
        print('Successfully loaded checkpoint...')
        
        print('Successfully loaded checkpoint...')

    def step_scheduler(self) :

        self.schedulerD.step()
        self.schedulerG.step()
        self.schedulerS.step()
        self.schedulerM.step()

    def data_dependent_init(self, data, n_train_samples) :

        self.set_input(data)
        self.forward()

        self.psd = self.radial_profile(self.real_A)
        self.spectral_discriminator = nn.Sequential(
            nn.Linear(self.psd.shape[1], 1),
            nn.Sigmoid()
        ).to(self.device)
        self.optimizerS = torch.optim.Adam(self.spectral_discriminator.parameters(), lr = self.opt.lr, betas = (self.beta1, self.beta2))

        self.compute_D_loss().backward()
        self.compute_G_loss().backward()

        self.schedulerG = self.get_lr_scheduler(self.optimizerG, n_train_samples * int(self.opt.epoch * 0.2), n_train_samples * int(self.opt.epoch * 0.8))
        self.schedulerD = self.get_lr_scheduler(self.optimizerD, n_train_samples * int(self.opt.epoch * 0.2), n_train_samples * int(self.opt.epoch * 0.8))
        self.schedulerS = self.get_lr_scheduler(self.optimizerS, n_train_samples * int(self.opt.epoch * 0.2), n_train_samples * int(self.opt.epoch * 0.8))
        if self.lambda_NCE > 0. :
            self.optimizerM = torch.optim.Adam(self.mlp.parameters(), lr = self.opt.lr, betas = (self.beta1, self.beta2))
            self.schedulerM = self.get_lr_scheduler(self.optimizerM, n_train_samples * int(self.opt.epoch * 0.2), n_train_samples * int(self.opt.epoch * 0.8))
        self.first_setting = True

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def get_lr_scheduler(self, optimizer, fixed_lr_step, decay_lr_step) :

        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch - fixed_lr_step) / float(decay_lr_step + 1)
            return lr_l

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
        return scheduler
    
    
    
    
    def _create_soft_circular_mask(self, h, w, device='cpu', radius_frac=None, gamma=None):
        """
        Differentiable circular mask: returns values in (0,1) via sigmoid(gamma*(radius - dist)).
        radius_frac: fraction of max radius (0..1). If None, uses self.radius_param (sigmoid-mapped).
        gamma: sharpness, if None uses self.radius_gamma.
        """
        if radius_frac is None:
            radius_frac = torch.sigmoid(self.radius_param)  # maps to (0,1)
        if gamma is None:
            gamma = self.radius_gamma

        # compute radius in pixels (max radius = min(h,w)/2)
        max_r = min(h, w) / 2.0
        radius = radius_frac * max_r

        ys = torch.arange(0, h, device=device).view(h, 1).expand(h, w)
        xs = torch.arange(0, w, device=device).view(1, w).expand(h, w)
        cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
        dist = torch.sqrt((ys - cy) ** 2 + (xs - cx) ** 2)

        # soft mask: 1 inside (low-freq) smoothly decays to 0 outside
        mask = torch.sigmoid(gamma * (radius - dist))
        return mask.unsqueeze(0).unsqueeze(0)  # shape (1,1,h,w)
    
    
    # ...existing code...
    def _create_circular_mask(self, h, w, radius=None, device='cpu'):
        if radius is None:
            radius = min(h, w) // 6
        ys = torch.arange(0, h, device=device).view(h, 1).expand(h, w)
        xs = torch.arange(0, w, device=device).view(1, w).expand(h, w)
        cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
        dist = torch.sqrt((ys - cy) ** 2 + (xs - cx) ** 2)
        mask = (dist <= radius).float()  # 1 = low-freq region
        return mask.unsqueeze(0).unsqueeze(0)  # shape (1,1,h,w)

    def apply_pgd_on_lowfreq(self, image, classifier_fn, labels,
                             eps=8/255, alpha=2/255, steps=10, radius=None,
                             random_start=True, targeted=False, target_labels=None):
        """
        PGD attack applied only to low-frequency magnitude of `image`.
        - image: [B,C,H,W] in image domain (not normalized).
        - classifier_fn: function(image_tensor)->logits (expects same device).
        - labels: ground-truth labels tensor for classifier (or target labels if targeted).
        Returns adversarial image with only low-frequency magnitude perturbed.
        """
        device = image.device
        B = image.shape[0]

        # 1) FWT -> magnitude, phase
        fwt = img2fwt(image)                          # complex tensor
        mag, phase = fwt2polar(fwt)                  # mag: log1p(abs(fwt))

        # mag shape: [B, C, h2, w2]
        _, _, h2, w2 = mag.shape
        mask = self._create_circular_mask(h2, w2, radius=radius, device=device)  # (1,1,h2,w2)
        mask = mask.expand(B, mag.size(1), -1, -1)  # broadcast to [B,C,h2,w2]

        # initialize adversarial magnitude
        adv_mag = mag.clone().detach()
        if random_start:
            adv_mag = adv_mag + (torch.empty_like(adv_mag).uniform_(-eps, eps))

        adv_mag = torch.clamp(adv_mag, min=0.0)  # magnitude is log1p(abs) so non-negative

        # PGD loop (opt-free)
        for _ in range(steps):
            adv_mag.requires_grad_(True)
            # merge low-freq candidate with original high-freq
            merged_mag = adv_mag * mask + mag * (1.0 - mask)
            # reconstruct image from merged magnitude + original phase
            # polar2img_fwt expects (magnitude, phase) in same format as fwt2polar output
            recon = polar2img_fwt(merged_mag, phase)  # returns [B,C,H,W] image domain

            # compute classifier loss
            logits = classifier_fn(recon)
            loss_fn = torch.nn.CrossEntropyLoss()
            if targeted:
                assert target_labels is not None
                clf_loss = -loss_fn(logits, target_labels)  # maximize target
            else:
                clf_loss = loss_fn(logits, labels)         # increase classification loss to cause misclass

            # compute gradient w.r.t adv_mag
            grad = torch.autograd.grad(clf_loss, adv_mag, retain_graph=False, create_graph=False)[0]
            # only apply update inside low-frequency mask
            update = alpha * grad.sign() * mask
            adv_mag = (adv_mag.detach() + update).clamp(min=0.0).detach()

            # project to eps-ball around original magnitude (Linf on mag)
            delta = torch.clamp(adv_mag - mag, min=-eps, max=eps)
            adv_mag = (mag + delta).detach()

        # final merged magnitude
        final_mag = adv_mag * mask + mag * (1.0 - mask)
        # reconstruct final adversarial image
        adv_image = polar2img_fwt(final_mag, phase)
        return adv_image
# ...existing code...

    '''
    # ...existing code...
    def apply_pgd_on_lowfreq_with_ssim(self, image, target_image,
                                        eps=2/255, alpha=0.5/255, steps=5, radius=None,
                                        random_start=True):
        """
        Low-frequency PGD attack that optimizes SSIM between reconstructed image and target_image.
        - image: [B,C,H,W] reconstructed (denoised) image in image domain
        - target_image: [B,C,H,W] clean target (image_B)
        Returns adversarially-perturbed reconstructed image (merged low-freq PGD + original high-freq).
        """
        device = image.device
        B = image.shape[0]

        # FWT -> magnitude, phase (uses utils.processing img2fwt / fwt2polar)
        fwt = img2fwt(image)
        mag, phase = fwt2polar(fwt)  # mag: log1p(abs(..))

        _, _, h2, w2 = mag.shape
        mask = self._create_circular_mask(h2, w2, radius=radius, device=device)  # (1,1,h2,w2)
        mask = mask.expand(B, mag.size(1), -1, -1)  # [B,C,h2,w2]

        adv_mag = mag.clone().detach()
        if random_start:
            adv_mag = adv_mag + (torch.empty_like(adv_mag).uniform_(-eps, eps))
        adv_mag = adv_mag.clamp(min=0.0).detach()

        loss_fn = self.criterionSSIM  # SSIM returns similarity -> use (1 - SSIM) as loss

        for _ in range(steps):
            adv_mag.requires_grad_(True)
            merged_mag = adv_mag * mask + mag * (1.0 - mask)
            recon = polar2img_fwt(merged_mag, phase)  # reconstructed image

            # compute reconstruction loss (we want to MINIMIZE 1 - SSIM => maximize SSIM)
            ssim_val = loss_fn(target_image, recon)  # returns SSIM (similarity)
            recon_loss = (1.0 - ssim_val).mean()  # scalar loss

            # gradient descent on adv_mag (minimize recon_loss)
            grad = torch.autograd.grad(recon_loss, adv_mag, retain_graph=False, create_graph=False)[0]
            update = - alpha * grad.sign() * mask  # descent step (note minus)
            adv_mag = (adv_mag.detach() + update).clamp(min=0.0).detach()

            # project to Linf-ball around original mag
            delta = torch.clamp(adv_mag - mag, min=-eps, max=eps)
            adv_mag = (mag + delta).detach()

        final_mag = adv_mag * mask + mag * (1.0 - mask)
        adv_image = polar2img_fwt(final_mag, phase)
        return adv_image
    # ...existing code...
    '''

    # ...existing code...
    def apply_pgd_on_lowfreq_with_ssim(self, image, target_image,
                                        classifier_fn=None, labels=None, dif_trainer=None,
                                        eps=2/255, alpha=0.5/255, steps=5, radius=None,
                                        random_start=True, targeted=False,
                                        w_rec=1.0, w_vit=1.0, w_dif=1.0, eps_norm=1e-8):
        """
        Low-frequency PGD that combines reconstruction loss (1-SSIM), classifier loss (CrossEntropy)
        and DIF contrastive loss into a single objective. The combined objective is:
            J = w_vit * vit_norm + w_dif * dif_norm - w_rec * rec_norm
        and we perform gradient ascent on J (so vit/dif are increased, rec is decreased).
        - image: [B,C,H,W] denoised / reconstructed image (current)
        - target_image: [B,C,H,W] clean target (image_B)
        - classifier_fn: function(image_tensor)->logits (required if w_vit>0)
        - labels: labels tensor for classifier/dif (required if corresponding weights>0)
        - dif_trainer: TrainerMultiple instance (required if w_dif>0)
        Returns adversarially-perturbed reconstructed image (merged low-freq PGD + original high-freq).
        """
        device = image.device
        B = image.shape[0]
        
        # denoised는 fake label로 고정되어 있으므로 labels가 None이면 모두 1로 설정
        if labels is None:
            labels = torch.ones(B, dtype=torch.long, device=device)
        # ...existing code...

        # FWT -> magnitude, phase
        fwt = img2fwt(image)
        mag, phase = fwt2polar(fwt)  # mag: log1p(abs(..))

        _, _, h2, w2 = mag.shape
        # use differentiable soft mask but DETACH radius during inner PGD so radius isn't updated inside PGD loop
        # radius_frac은 학습 가능한 파라미터이지만 inner PGD에서는 고정시킨다.
        radius_frac_det = torch.sigmoid(self.radius_param).detach()
        mask = self._create_soft_circular_mask(h2, w2, device=device, radius_frac=radius_frac_det, gamma=self.radius_gamma)
        mask = mask.expand(B, mag.size(1), -1, -1)  # [B,C,h2,w2]

        # initialize adversarial magnitude
        adv_mag = mag.clone().detach()
        if random_start:
            adv_mag = adv_mag + (torch.empty_like(adv_mag).uniform_(-eps, eps))
        adv_mag = adv_mag.clamp(min=0.0).detach()

        # compute baseline (normalization denominators) on original reconstruction
        with torch.no_grad():
            orig_recon = polar2img_fwt(mag, phase)
            # recon baseline
            rec_ssim = self.criterionSSIM(target_image, orig_recon).mean().detach()
            rec_baseline = (1.0 - rec_ssim).abs().detach().clamp(min=eps_norm)

            # vit baseline
            if (w_vit != 0.0) and (classifier_fn is not None) and (labels is not None):
                logits0 = classifier_fn(orig_recon)
                vit_loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
                if targeted:
                    vit0 = (-vit_loss_fn(logits0, labels)).abs().detach()  # target-case magnitude
                else:
                    vit0 = (vit_loss_fn(logits0, labels)).abs().detach()
                vit_baseline = vit0.clamp(min=eps_norm)
            else:
                vit_baseline = torch.tensor(1.0, device=device)

            # dif baseline
            if (w_dif != 0.0) and (dif_trainer is not None) and (labels is not None):
                res0 = dif_trainer.denoiser.denoise(orig_recon).float()
                fp = dif_trainer.fingerprint.to(device)
                fp_rep = fp.repeat((orig_recon.size(0), 1, 1, 1))
                corr0 = dif_trainer.corr_fun(fp_rep, res0)
                dif0 = (dif_trainer.loss_contrast(corr0.mean((1, 2, 3)), labels) / dif_trainer.m).mean().abs().detach()
                dif_baseline = dif0.clamp(min=eps_norm)
            else:
                dif_baseline = torch.tensor(1.0, device=device)

        # PGD iterations: maximize J = w_vit*V_norm + w_dif*D_norm - w_rec*R_norm
        for _ in range(steps):
            adv_mag.requires_grad_(True)
            merged_mag = adv_mag * mask + mag * (1.0 - mask)
            recon = polar2img_fwt(merged_mag, phase)  # reconstructed image

            # 1) reconstruction loss (we want to MINIMIZE 1 - SSIM -> contributes negatively to J)
            ssim_val = self.criterionSSIM(target_image, recon)  # similarity
            rec_loss = (1.0 - ssim_val).mean()

            # 2) vit loss (we want to MAXIMIZE classifier loss -> contributes positively to J)
            if (w_vit != 0.0) and (classifier_fn is not None) and (labels is not None):
                logits = classifier_fn(recon)
                vit_loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
                if targeted:
                    vit_loss = -vit_loss_fn(logits, labels)  # maximize target probability
                else:
                    vit_loss = vit_loss_fn(logits, labels)
                vit_loss_mean = vit_loss
            else:
                vit_loss_mean = torch.tensor(0.0, device=device)

            # 3) dif loss (we want to MAXIMIZE dif contrastive loss -> contributes positively)
            if (w_dif != 0.0) and (dif_trainer is not None) and (labels is not None):
                res_adv = dif_trainer.denoiser.denoise(recon).float()
                fp = dif_trainer.fingerprint.to(device)
                fp_rep = fp.repeat((recon.size(0), 1, 1, 1))
                corr_adv = dif_trainer.corr_fun(fp_rep, res_adv)
                dif_loss = (dif_trainer.loss_contrast(corr_adv.mean((1, 2, 3)), labels) / dif_trainer.m).mean()
                dif_loss_mean = dif_loss
            else:
                dif_loss_mean = torch.tensor(0.0, device=device)

            # Normalize each term by its baseline to avoid domination
            rec_norm = rec_loss / rec_baseline
            vit_norm = vit_loss_mean / vit_baseline if isinstance(vit_baseline, torch.Tensor) else vit_loss_mean
            dif_norm = dif_loss_mean / dif_baseline if isinstance(dif_baseline, torch.Tensor) else dif_loss_mean

            # combined objective (maximize J)
            combined = (w_vit * vit_norm + w_dif * dif_norm) - (w_rec * rec_norm)
            # ensure scalar
            combined_loss = combined.mean()

            # compute gradient w.r.t adv_mag
            grad = torch.autograd.grad(combined_loss, adv_mag, retain_graph=False, create_graph=False)[0]

            # ascent step (maximize combined objective); only apply inside low-frequency mask
            update = alpha * grad.sign() * mask
            adv_mag = (adv_mag.detach() + update).clamp(min=0.0).detach()

            # project to Linf-ball around original magnitude (Linf on mag)
            delta = torch.clamp(adv_mag - mag, min=-eps, max=eps)
            adv_mag = (mag + delta).detach()

        # final merged magnitude -> reconstruct final adversarial image
        final_mag = adv_mag * mask + mag * (1.0 - mask)
        adv_image = polar2img_fwt(final_mag, phase)
        return adv_image
# ...existing code...