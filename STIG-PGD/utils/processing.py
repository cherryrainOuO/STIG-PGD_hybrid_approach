import torch
import torch.nn.functional as F

def img2fft(img) :
    coord_matched_img = torch.fft.fftshift(img, dim = (-1, -2))
    fft = torch.fft.fft2(coord_matched_img)
    fft = torch.fft.fftshift(fft)
    return fft

def fft2polar(fft) :
    magnitude = torch.log1p(torch.abs(fft))
    phase = torch.angle(fft)
    return magnitude, phase

def img2fwt(img):
    """
    2D Haar-like Fast Wavelet Transform (single level).
    입력: img tensor [B, C, H, W]
    반환: complex tensor [B, C, H/2, W/2] (real=LL, imag=LH+HL+HH)
    """
    # separable Haar kernels (normalized)
    device = img.device
    k_factor = 0.5  # (1/sqrt2 * 1/sqrt2) = 1/2
    k_ll = torch.tensor([[1., 1.], [1., 1.]], device=device) * k_factor
    k_lh = torch.tensor([[1., -1.], [1., -1.]], device=device) * k_factor
    k_hl = torch.tensor([[1., 1.], [-1., -1.]], device=device) * k_factor
    k_hh = torch.tensor([[1., -1.], [-1., 1.]], device=device) * k_factor

    B, C, H, W = img.shape
    # build grouped conv weights: shape (C,1,2,2)
    weight_ll = k_ll.view(1, 1, 2, 2).repeat(C, 1, 1, 1)
    weight_lh = k_lh.view(1, 1, 2, 2).repeat(C, 1, 1, 1)
    weight_hl = k_hl.view(1, 1, 2, 2).repeat(C, 1, 1, 1)
    weight_hh = k_hh.view(1, 1, 2, 2).repeat(C, 1, 1, 1)

    # conv2d with groups=C to apply per-channel filters, stride=2 to downsample
    ll = F.conv2d(img, weight_ll, bias=None, stride=2, padding=0, groups=C)
    lh = F.conv2d(img, weight_lh, bias=None, stride=2, padding=0, groups=C)
    hl = F.conv2d(img, weight_hl, bias=None, stride=2, padding=0, groups=C)
    hh = F.conv2d(img, weight_hh, bias=None, stride=2, padding=0, groups=C)

    # construct complex-like coefficients: real part = LL, imag part = sum of detail bands
    detail = lh + hl + hh
    fwt = torch.complex(ll, detail)
    return fwt

def fwt2polar(fwt):
    """
    fwt -> magnitude, phase analogous to fft2polar
    입력: fwt complex tensor
    반환: magnitude = log1p(abs(fwt)), phase = angle(fwt)
    """
    magnitude = torch.log1p(torch.abs(fwt))
    phase = torch.angle(fwt)
    return magnitude, phase

def normalize(tensor) :
    v_max, v_min = tensor.max(), tensor.min()
    tensor_normed = (tensor - v_min) / (v_max - v_min)
    tensor_normed = tensor_normed * 2. - 1.
    return tensor_normed, v_max, v_min

def image_normalize(image) :
    v_max, v_min = image.max(), image.min()
    return (image - v_min) / (v_max - v_min)

def inverse_normalize(tensor, v_max, v_min) :
    tensor_scaled = (tensor + 1.) / 2.
    tensor_scaled = (tensor_scaled * (v_max - v_min)) + v_min
    return tensor_scaled

def polar2img(magnitude, phase) :
    fft = torch.expm1(magnitude) * torch.exp(1j * phase)
    ifft = torch.fft.ifftshift(fft)
    ifft = torch.fft.ifft2(ifft)
    img_mag = torch.abs(ifft)
    #img_mag = ifft.real
    coord_matched_img = torch.fft.fftshift(img_mag, dim = (-1, -2))
    return coord_matched_img

def polar2img_fwt(magnitude, phase) :
    # 복원할 FWT 복소 계수 재구성
        coeff = torch.expm1(magnitude) * torch.exp(1j * phase)
        ll = coeff.real    # forward에서 real=LL
        detail = coeff.imag  # forward에서 imag=LH+HL+HH (합으로 저장됨)

        # inverse Haar-like transform (근사 복원)
        # forward에서 사용한 커널과 동일하게 구성
        device = ll.device
        k_factor = 0.5
        k_ll = torch.tensor([[1., 1.], [1., 1.]], device=device) * k_factor
        k_lh = torch.tensor([[1., -1.], [1., -1.]], device=device) * k_factor
        k_hl = torch.tensor([[1., 1.], [-1., -1.]], device=device) * k_factor
        k_hh = torch.tensor([[1., -1.], [-1., 1.]], device=device) * k_factor

        B, C, h2, w2 = ll.shape
        # grouped conv_transpose weights: (C,1,2,2)
        weight_ll = k_ll.view(1, 1, 2, 2).repeat(C, 1, 1, 1)
        weight_lh = k_lh.view(1, 1, 2, 2).repeat(C, 1, 1, 1)
        weight_hl = k_hl.view(1, 1, 2, 2).repeat(C, 1, 1, 1)
        weight_hh = k_hh.view(1, 1, 2, 2).repeat(C, 1, 1, 1)

        # detail을 LH, HL, HH로 균등 분배 (forward에서 합쳤기 때문에 근사 분해)
        lh = detail / 3.0
        hl = detail / 3.0
        hh = detail / 3.0

        # 그룹별 transposed conv로 업샘플 복원
        rec_ll = F.conv_transpose2d(ll, weight_ll, bias=None, stride=2, padding=0, groups=C)
        rec_lh = F.conv_transpose2d(lh, weight_lh, bias=None, stride=2, padding=0, groups=C)
        rec_hl = F.conv_transpose2d(hl, weight_hl, bias=None, stride=2, padding=0, groups=C)
        rec_hh = F.conv_transpose2d(hh, weight_hh, bias=None, stride=2, padding=0, groups=C)

        recon = rec_ll + rec_lh + rec_hl + rec_hh
        return recon

def get_magnitude(img) :
    fft = torch.fft.fft2(img, dim = [-1, -2])
    fft = torch.fft.fftshift(fft, dim = [-1, -2])
    mag = torch.log1p(torch.abs(fft))
    v_min, v_max = torch.amin(mag, dim = (1, 2, 3)).view(-1, 1, 1, 1), torch.amax(mag, dim = (1, 2, 3)).view(-1, 1, 1, 1)
    mag_normed = (mag - v_min) / (v_max - v_min)
    return mag_normed

def get_averaged_magnitude(img) :
    fft = torch.fft.fft2(img)
    fft = torch.fft.fftshift(fft)
    mag = torch.log1p(torch.abs(fft))
    v_min, v_max = torch.amin(mag, dim = (1, 2, 3)).view(-1, 1, 1, 1), torch.amax(mag, dim = (1, 2, 3)).view(-1, 1, 1, 1)
    mag_normed = (mag - v_min) / (v_max - v_min)
    mag_normed = mag_normed * 2. - 1.
    #mag_mean = mag_normed.mean(0, keepdim = True)
    return mag_normed


def get_magnitude_fwt(img) :
    fwt = img2fwt(img)
    mag = torch.log1p(torch.abs(fwt))
    v_min, v_max = torch.amin(mag, dim = (1, 2, 3)).view(-1, 1, 1, 1), torch.amax(mag, dim = (1, 2, 3)).view(-1, 1, 1, 1)
    mag_normed = (mag - v_min) / (v_max - v_min)
    return mag_normed

def get_averaged_magnitude_fwt(img) :
    fwt = img2fwt(img)
    mag = torch.log1p(torch.abs(fwt))
    v_min, v_max = torch.amin(mag, dim = (1, 2, 3)).view(-1, 1, 1, 1), torch.amax(mag, dim = (1, 2, 3)).view(-1, 1, 1, 1)
    mag_normed = (mag - v_min) / (v_max - v_min)
    mag_normed = mag_normed * 2. - 1.
    #mag_mean = mag_normed.mean(0, keepdim = True)
    return mag_normed