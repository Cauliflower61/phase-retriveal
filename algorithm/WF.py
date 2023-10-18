import torch
from utils.AngularSpectrum import angular_spectrum
from tqdm import tqdm


def WF_global_2plane(y, masks, distance, wavelength, pixel_size):
    step = 0.2
    iteration = 40
    # step1: initialize the wavefront
    x = torch.mean(angular_spectrum(y, -distance, wavelength, pixel_size) * torch.conj(masks), dim=0).unsqueeze(dim=0)
    # step2: iteration by Wirtinger flow
    loop = tqdm(range(iteration), desc='iteration')
    for iter1 in loop:
        u = angular_spectrum(x * masks, distance, wavelength, pixel_size)
        a = torch.abs(u)
        e = a - y
        dF = angular_spectrum(u * 1/a * e, -distance, wavelength, pixel_size) * torch.conj(masks)
        df = 1/2 * torch.mean(dF, dim=0)
        x -= step * df
        loop.set_description(f'iter [{iter1 + 1}/{iteration}]')
        loop.set_postfix(error=torch.mean(torch.abs(e)).data.item())
    return x


def WF_global_3plane(y, masks, d1, d2, wavelength, pixel_size):
    step = 0.2
    iteration = 40
    # step1: initialize the wavefront
    x = torch.mean(angular_spectrum(angular_spectrum(y, -d2, wavelength, pixel_size) *
                                    torch.conj(masks), -d1, wavelength, pixel_size), dim=0).unsqueeze(dim=0)
    # step2: iteration by Wirtinger flow
    loop = tqdm(range(iteration), desc='iteration')
    for iter1 in loop:
        u = angular_spectrum(angular_spectrum(x, d1, wavelength, pixel_size) * masks, d2, wavelength, pixel_size)
        a = torch.abs(u)
        e = a - y
        dF = angular_spectrum(angular_spectrum(u * 1/a * e, -d2, wavelength, pixel_size)
                              * torch.conj(masks), -d1, wavelength, pixel_size)
        df = 1/2 * torch.mean(dF, dim=0)
        x -= step * df
        loop.set_description(f'iter [{iter1 + 1}/{iteration}]')
        loop.set_postfix(error=torch.mean(torch.abs(e)).data.item())
    return x
