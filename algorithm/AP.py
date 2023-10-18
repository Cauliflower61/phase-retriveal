import torch
from utils.AngularSpectrum import angular_spectrum
from tqdm import tqdm


def AP_2plane(y, masks, distance, wavelength, pixel_size):
    iteration = 200
    # step1: initialize the wavefront
    x = torch.mean(angular_spectrum(y, -distance, wavelength, pixel_size) * torch.conj(masks), dim=0).unsqueeze(dim=0)
    loop = tqdm(range(iteration), desc='iteration')
    for iter1 in loop:
        # step2: forward propagate
        y_hat = angular_spectrum(x * masks, distance, wavelength, pixel_size)
        # step3: amplitude constraint
        y_hat = torch.abs(y) * torch.exp(1j * torch.angle(y_hat))
        # step4: backward propagation
        x = torch.mean(angular_spectrum(y_hat, -distance, wavelength, pixel_size)
                       * torch.conj(masks), dim=0).unsqueeze(dim=0)
        loop.set_description(f'iter [{iter1 + 1}/{iteration}]')
    return x


def AP_3plane(y, masks, d1, d2, wavelength, pixel_size):
    iteration = 200
    # step1: initialize the wavefront
    x = torch.mean(angular_spectrum(angular_spectrum(y, -d2, wavelength, pixel_size) *
                                    torch.conj(masks), -d1, wavelength, pixel_size), dim=0).unsqueeze(dim=0)
    loop = tqdm(range(iteration), desc='iteration')
    for iter1 in loop:
        # step2: forward propagate
        y_hat = angular_spectrum(angular_spectrum(x, d1, wavelength, pixel_size) * masks, d2, wavelength, pixel_size)
        # step3: amplitude constraint
        y_hat = torch.abs(y) * torch.exp(1j * torch.angle(y_hat))
        # step4: backward propagate
        x = torch.mean(angular_spectrum(angular_spectrum(y_hat, -d2, wavelength, pixel_size) *
                                        torch.conj(masks), -d1, wavelength, pixel_size), dim=0).unsqueeze(dim=0)
        loop.set_description(f'iter [{iter1 + 1}/{iteration}]')
    return x
