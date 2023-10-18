import torch
import numpy as np
from math import pi
import cv2
from utils.AngularSpectrum import angular_spectrum
from algorithm import AP, WF
import matplotlib.pyplot as plt


def rec_AP_2plane():
    # set parameters
    num = 64
    wavelength = 532e-9
    pixel_size = 3.8e-6
    distance = 50e-3
    device = 'cuda:0'
    # generate simulation objection
    amplitude = cv2.imread('./data/sim/barbara.png', cv2.IMREAD_UNCHANGED).astype(np.float32)
    amplitude = amplitude / amplitude.max()
    phase = cv2.imread('./data/sim/peppers.png', cv2.IMREAD_UNCHANGED).astype(np.float32)
    phase = phase / phase.max()
    obj = amplitude * np.exp(1j * pi/2 * phase)
    # generate masks
    torch.manual_seed(10)
    masks = torch.exp(1j * torch.randn(num, int(obj.shape[0]), int(obj.shape[1]))).to(device)
    # generate intensity
    obj = torch.from_numpy(obj).unsqueeze(dim=0).to(device)
    y = torch.abs(angular_spectrum(obj * masks, distance, wavelength, pixel_size))
    # reconstruction
    obj_hat = AP.AP_2plane(y, masks, distance, wavelength, pixel_size)
    # show the image
    obj = obj[0, :, :].detach().cpu().numpy()
    obj_hat = obj_hat[0, :, :].detach().cpu().numpy()
    plt.figure()
    plt.subplot(221), plt.imshow(np.abs(obj), cmap='gray')
    plt.subplot(222), plt.imshow(np.angle(obj), cmap='gray')
    plt.subplot(223), plt.imshow(np.abs(obj_hat), cmap='gray')
    plt.subplot(224), plt.imshow(np.angle(obj_hat), cmap='gray')
    plt.show()


def rec_AP_3plane():
    # set parameters
    num = 64
    wavelength = 532e-9
    pixel_size = 3.8e-6
    d1 = 50e-3
    d2 = 50e-3
    device = 'cuda:0'
    # generate simulation objection
    amplitude = cv2.imread('./data/sim/barbara.png', cv2.IMREAD_UNCHANGED).astype(np.float32)
    amplitude = amplitude / amplitude.max()
    phase = cv2.imread('./data/sim/peppers.png', cv2.IMREAD_UNCHANGED).astype(np.float32)
    phase = phase / phase.max()
    obj = amplitude * np.exp(1j * pi / 2 * phase)
    # generate masks
    torch.manual_seed(10)
    masks = torch.exp(1j * torch.randn(num, int(obj.shape[0]), int(obj.shape[1]))).to(device)
    # generate intensity
    obj = torch.from_numpy(obj).unsqueeze(dim=0).to(device)
    y = torch.abs(angular_spectrum(angular_spectrum(obj, d1, wavelength, pixel_size)
                                   * masks, d2, wavelength, pixel_size))
    # reconstruction
    obj_hat = AP.AP_3plane(y, masks, d1, d2, wavelength, pixel_size)
    # show the image
    obj = obj[0, :, :].detach().cpu().numpy()
    obj_hat = obj_hat[0, :, :].detach().cpu().numpy()
    plt.figure()
    plt.subplot(221), plt.imshow(np.abs(obj), cmap='gray')
    plt.subplot(222), plt.imshow(np.angle(obj), cmap='gray')
    plt.subplot(223), plt.imshow(np.abs(obj_hat), cmap='gray')
    plt.subplot(224), plt.imshow(np.angle(obj_hat), cmap='gray')
    plt.show()


def rec_WF_2plane():
    # set parameters
    num = 64
    wavelength = 532e-9
    pixel_size = 3.8e-6
    distance = 50e-3
    device = 'cuda:0'
    # generate simulation objection
    amplitude = cv2.imread('./data/sim/barbara.png', cv2.IMREAD_UNCHANGED).astype(np.float32)
    amplitude = amplitude / amplitude.max()
    phase = cv2.imread('./data/sim/peppers.png', cv2.IMREAD_UNCHANGED).astype(np.float32)
    phase = phase / phase.max()
    obj = amplitude * np.exp(1j * pi / 2 * phase)
    # generate masks
    torch.manual_seed(10)
    masks = torch.exp(1j * torch.randn(num, int(obj.shape[0]), int(obj.shape[1]))).to(device)
    # generate intensity
    obj = torch.from_numpy(obj).unsqueeze(dim=0).to(device)
    y = torch.abs(angular_spectrum(obj * masks, distance, wavelength, pixel_size))
    # reconstruction
    obj_hat = WF.WF_global_2plane(y, masks, distance, wavelength, pixel_size)
    # show the image
    obj = obj[0, :, :].detach().cpu().numpy()
    obj_hat = obj_hat[0, :, :].detach().cpu().numpy()
    plt.figure()
    plt.subplot(221), plt.imshow(np.abs(obj), cmap='gray')
    plt.subplot(222), plt.imshow(np.angle(obj), cmap='gray')
    plt.subplot(223), plt.imshow(np.abs(obj_hat), cmap='gray')
    plt.subplot(224), plt.imshow(np.angle(obj_hat), cmap='gray')
    plt.show()


def rec_WF_3plane():
    # set parameters
    num = 64
    wavelength = 532e-9
    pixel_size = 3.8e-6
    d1 = 50e-3
    d2 = 50e-3
    device = 'cuda:0'
    # generate simulation objection
    amplitude = cv2.imread('./data/sim/barbara.png', cv2.IMREAD_UNCHANGED).astype(np.float32)
    amplitude = amplitude / amplitude.max()
    phase = cv2.imread('./data/sim/peppers.png', cv2.IMREAD_UNCHANGED).astype(np.float32)
    phase = phase / phase.max()
    obj = amplitude * np.exp(1j * pi / 2 * phase)
    # generate masks
    torch.manual_seed(10)
    masks = torch.exp(1j * torch.randn(num, int(obj.shape[0]), int(obj.shape[1]))).to(device)
    # generate intensity
    obj = torch.from_numpy(obj).unsqueeze(dim=0).to(device)
    y = torch.abs(angular_spectrum(angular_spectrum(obj, d1, wavelength, pixel_size)
                                   * masks, d2, wavelength, pixel_size))
    # reconstruction
    obj_hat = WF.WF_global_3plane(y, masks, d1, d2, wavelength, pixel_size)
    # show the image
    obj = obj[0, :, :].detach().cpu().numpy()
    obj_hat = obj_hat[0, :, :].detach().cpu().numpy()
    plt.figure()
    plt.subplot(221), plt.imshow(np.abs(obj), cmap='gray')
    plt.subplot(222), plt.imshow(np.angle(obj), cmap='gray')
    plt.subplot(223), plt.imshow(np.abs(obj_hat), cmap='gray')
    plt.subplot(224), plt.imshow(np.angle(obj_hat), cmap='gray')
    plt.show()


if __name__ == '__main__':
    rec_WF_2plane()
    rec_AP_2plane()
    rec_WF_3plane()
    rec_AP_3plane()