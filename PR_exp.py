import os, cv2, torch
import numpy as np
from algorithm import WF, AP
from utils.AngularSpectrum import angular_spectrum
import matplotlib.pyplot as plt


def read_image(dir, ROI=[1110,1812,1096,1096], isMasks=False):
    image_list = os.listdir(dir)
    x, y, h, w = ROI
    out = torch.zeros(int(len(image_list)), h, w)
    for iter1, name in enumerate(image_list):
        image_path = os.path.join(dir, name)
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
        image = (2.5907 * cv2.normalize(-image, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)) if isMasks \
            else (image / np.max(image))
        out[iter1, :, :] = torch.from_numpy(image[x:x+h, y:y+w])
    return out


# load data
device = 'cuda:0'
masks_dir = './data/exp/Mask_rec_20dot20_max127/'
intensity_dir = './data/exp/GT/'
masks = read_image(masks_dir, isMasks=True).to(device)
print('---------------------loading masks finished---------------------------------------')
y = read_image(intensity_dir).to(device)
print('------------------loading intensity finished--------------------------------------')
# reconstruction
y_hat = WF.WF_global_3plane(y, torch.exp(-1j * masks), 108.9e-3, 50e-3, 532e-9, 3.8e-6)\
    .squeeze(dim=0).detach().cpu().numpy()
# show
plt.figure()
plt.subplot(211), plt.imshow(np.abs(y_hat), cmap='gray')
plt.subplot(212), plt.imshow(np.angle(y_hat), cmap='gray')
plt.show()
