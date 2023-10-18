import torch
from math import pi


def angular_spectrum(wavefront, distance, wavelength, pixel_size):
    """
    :param wavefront: [N, H, W], where N means the number of pattern, H and W is the height and weight of wavefront
    :param distance: diffraction distance
    :param wavelength: wavelength of illumination
    :param pixel_size: pixel size of detector
    :return: discrete propagation wavefront
    """
    N, H, W = wavefront.shape
    hh = torch.arange((-H / 2), (H / 2), 1)
    ww = torch.arange((-W / 2), (W / 2), 1)
    h, w = torch.meshgrid(hh, ww)
    U = 1 - (wavelength ** 2) * ((h / (pixel_size * H)) ** 2 + (w / (pixel_size * W)) ** 2).to('cuda:0')
    TF = torch.exp(1j * 2 * pi / wavelength * distance * torch.sqrt(U))
    TF[U < 0] = 0
    prop_wave = torch.fft.ifft2(torch.fft.ifftshift(TF.unsqueeze(dim=0) * torch.fft.fftshift(torch.fft.fft2(wavefront))))
    return prop_wave
