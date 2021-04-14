import yaml
import torch
import argparse
import timeit
import time
import os
import numpy as np
import scipy.misc as misc

from torch.utils import data
from torchstat import stat
from pytorch_bn_fusion.bn_fusion import fuse_bn_recursively

from ptsemseg.models import get_model
from ptsemseg.loader import get_loader
from ptsemseg.metrics import runningScore
from ptsemseg.utils import convert_state_dict

#from SemanticAdv.attacks import semantic_attack
import torch.nn.functional as F
from ptsemseg.loss import get_loss_function
import cv2
#import sys
#sys.path.append("..")
from Pytorch_AdversarialAttacks import *
from Pytorch_AdversarialAttacks.adversarials import Rician
#from Rician import generate_rician
import random
from scipy.stats import rice
from skimage.measure import compare_ssim as ssim

torch.backends.cudnn.benchmark = True

'''def generate_rician(image):
    
    ssim_noise = 0

    if torch.is_tensor(image):
        image = image.numpy()

    rician_image = np.zeros_like(image)

    iteration = 0
    while ssim_noise <= 0.97 or ssim_noise >= 0.99:
        b = random.uniform(0, 0.3)
        rv = rice(b)
        rician_image = rv.pdf(image)
        #print(image[0].transpose(1, 2, 0).shape)
        ssim_noise =  ssim(image[0].transpose(1, 2, 0), rician_image[0].transpose(1, 2, 0), data_range=rician_image[0].transpose(1, 2 ,0).max() - rician_image[0].transpose(1, 2, 0).min(), multichannel=True)
        iteration += 1
        if iteration > 2:
            break

    #print('ssim : {:.2f}'.format(ssim_noise))
    return rician_image'''

# FGSM attack code
def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon*sign_data_grad
    # Adding clipping to maintain [0,1] range
    #perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    perturbed_image1 = torch.clamp(perturbed_image[:, 0:1, :, :], (0-0.406)/0.225, (1-0.406)/0.225)
    perturbed_image2 = torch.clamp(perturbed_image[:, 1:2, :, :], (0-0.456)/0.224, (1-0.456)/0.224)
    perturbed_image3 = torch.clamp(perturbed_image[:, 2:3, :, :], (0-0.485)/0.229, (1-0.485)/0.229)
    perturbed_image = torch.cat([perturbed_image1, perturbed_image2], dim=1)
    perturbed_image = torch.cat([perturbed_image, perturbed_image3], dim=1)
    return perturbed_image

def reset_batchnorm(m):
    if isinstance(m, torch.nn.BatchNorm2d):
      m.reset_running_stats()
      m.momentum = None

def validate(cfg, args):
    print(args.model_path)
    #device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    # Setup Dataloader
    data_loader = get_loader(cfg["data"]["dataset"])
    data_path = cfg["data"]["path"]

    loader = data_loader(
        data_path,
        split=cfg["data"]["val_split"],
        is_transform=True,
        img_size=(1024,2048),
    )

    n_classes = loader.n_classes

    valloader = data.DataLoader(loader, batch_size=1, num_workers=1)
    running_metrics = runningScore(n_classes)

    # Setup Model

    model = get_model(cfg["model"], n_classes).to(device)
    state = convert_state_dict(torch.load(args.model_path)["model_state"])
    model.load_state_dict(state)
    
    if args.bn_fusion:
      model = fuse_bn_recursively(model)
    
    #Transform model into v2. Please set trt=True when converting to TensorRT model
    #model.v2_transform(trt=False) 
    #print(model)
    
    if args.update_bn:
      print("Reset BatchNorm and recalculate mean/var")
      model.apply(reset_batchnorm)
      model.train()
    else:
      model.eval()
    model = model.to(device)
    total_time = 0
    
    total_params = sum(p.numel() for p in model.parameters())
    print('Parameters: ', total_params )
    
    #stat(model, (3, 1024, 2048))
    torch.backends.cudnn.benchmark=True

    #adversary = semantic_attack.FP_CW_TV() 
    loss_fn = get_loss_function(cfg)
    print("Using loss {}".format(loss_fn))
    #loss_fn = loss_fn.cuda()
    
    for i, (images, labels, fname) in enumerate(valloader):
        start_time = timeit.default_timer()
        #images = generate_rician(images)
        #images = torch.tensor(images, dtype=torch.float)
        images = images.to(device)
        labels = labels.to(device)
        # Set requires_grad attribute of tensor. Important for Attack
        #images = images.detach()
        images.requires_grad = True
        #images.retain_grad()
        #print(images.requires_grad)
        #print(images.grad)
        outputs = model(images)
        #print(images.grad)
        loss = loss_fn(input=outputs, target=labels)
        #print(loss)
        # Zero all existing gradients
        model.zero_grad()
        # Calculate gradients of model in backward pass
        loss.backward()
        #print(images.grad)
        #print(images)
        data_grad = images.grad
        images = fgsm_attack(images, 0.12, data_grad)
        
        if args.eval_flip:
            outputs = model(images)

            # Flip images in numpy (not support in tensor)
            outputs = outputs.data.cpu().numpy()
            flipped_images = np.copy(images.data.cpu().numpy()[:, :, :, ::-1])
            flipped_images = torch.from_numpy(flipped_images).float().to(device)
            outputs_flipped = model(flipped_images)
            outputs_flipped = outputs_flipped.data.cpu().numpy()
            outputs = (outputs + outputs_flipped[:, :, :, ::-1]) / 2.0

            pred = np.argmax(outputs, axis=1)
        else:
            torch.cuda.synchronize()
            start_time = time.perf_counter()

            #with torch.no_grad():
            outputs = model(images)

            torch.cuda.synchronize()
            elapsed_time = time.perf_counter() - start_time
            
            if args.save_image:
                pred = np.squeeze(outputs.data.max(1)[1].cpu().numpy(), axis=0)
                save_rgb = True
                
                decoded = loader.decode_segmap_id(pred)
                dir = "./out_predID/"
                if not os.path.exists(dir):
                  os.mkdir(dir)
                #misc.imsave(dir+fname[0], decoded)
                '''decoded_gray = decoded
                decoded = cv2.cvtColor(decoded, cv2.COLOR_GRAY2BGR)
                
                B = decoded[:,:,0]
                G = decoded[:,:,1]
                R = decoded[:,:,2]
                # 灰度g=p*R+q*G+t*B（其中p=0.2989,q=0.5870,t=0.1140），于是B=(g-p*R-q*G)/t。于是我们只要保留R和G两个颜色分量，再加上灰度图g，就可以回复原来的RGB图像。
                g = decoded_gray[:]
                p = 0.2989; q = 0.5870; t = 0.1140
                B_new = (g-p*R-q*G)/t
                B_new = np.uint8(B_new)
                decoded = np.zeros((decoded.shape)).astype("uint8")
                decoded[:,:,0] = B_new
                decoded[:,:,1] = G
                decoded[:,:,2] = R
                #print(decoded.shape)'''

                cv2.imwrite(dir+fname[0], decoded)

                if save_rgb:
                    decoded = loader.decode_segmap(pred)
                    img_input = np.squeeze(images.cpu().detach().numpy(),axis=0)
                    img_input = img_input.transpose(1, 2, 0)
                    #blend = img_input * 0.2 + decoded * 0.8
                    blend = img_input * 1 + decoded * 0

                    value_scale = 255
                    mean = [-0.406/0.225, -0.456/0.224, -0.485/0.229]
                    #mean = [item * value_scale for item in mean]
                    std = [1/0.225, 1/0.224, 1/0.229]
                    #std = [item * value_scale for item in std]
                    blend = (blend - mean) / std
                    blend *= value_scale

                    fname_new = fname[0]
                    fname_new = fname_new[:-4]
                    fname_new += '.jpg'
                    dir = "./out_rgb/"
                    if not os.path.exists(dir):
                      os.mkdir(dir)
                    #misc.imsave(dir+fname_new, blend)
                    cv2.imwrite(dir+fname_new, blend)
                    #print(blend.shape)
                    #print(blend)

                
            pred = outputs.data.max(1)[1].cpu().numpy()

        labels = labels.cpu()
        gt = labels.numpy()
        s = np.sum(gt==pred) / (1024*2048)

        if args.measure_time:
            total_time += elapsed_time
            print(
                "Inference time \
                  (iter {0:5d}): {1:4f}, {2:3.5f} fps".format(
                    i + 1, s,1 / elapsed_time
                )
            )
        
        running_metrics.update(gt, pred)
        

    score, class_iou = running_metrics.get_scores()
    print("Total Frame Rate = %.2f fps" %(500/total_time ))

    if args.update_bn:
      model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
      state2 = {"model_state": model.state_dict()}
      torch.save(state2, 'hardnet_cityscapes_mod.pth')

    for k, v in score.items():
        print(k, v)

    for i in range(n_classes):
        print(i, class_iou[i])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperparams")
    parser.add_argument(
        "--config",
        nargs="?",
        type=str,
        default="configs/hardnet.yml",
        help="Config file to be used",
    )
    parser.add_argument(
        "--model_path",
        nargs="?",
        type=str,
        default="hardnet_cityscapes_best_model.pkl",
        help="Path to the saved model",
    )
    parser.add_argument(
        "--eval_flip",
        dest="eval_flip",
        action="store_true",
        help="Enable evaluation with flipped image |\
                              False by default",
    )
    parser.add_argument(
        "--no-eval_flip",
        dest="eval_flip",
        action="store_false",
        help="Disable evaluation with flipped image",
    )
    parser.set_defaults(eval_flip=False)

    parser.add_argument(
        "--measure_time",
        dest="measure_time",
        action="store_true",
        help="Enable evaluation with time (fps) measurement |\
                              True by default",
    )
    parser.add_argument(
        "--no-measure_time",
        dest="measure_time",
        action="store_false",
        help="Disable evaluation with time (fps) measurement",
    )
    parser.set_defaults(measure_time=True)

    parser.add_argument(
        "--save_image",
        dest="save_image",
        action="store_true",
        help="Enable saving inference result image into out_img/ |\
                              False by default",
    )
    parser.set_defaults(save_image=False)
    
    parser.add_argument(
        "--update_bn",
        dest="update_bn",
        action="store_true",
        help="Reset and update BatchNorm running mean/var with entire dataset |\
              False by default",
    )
    parser.set_defaults(update_bn=False)
    
    parser.add_argument(
        "--no-bn_fusion",
        dest="bn_fusion",
        action="store_false",
        help="Disable performing batch norm fusion with convolutional layers |\
              bn_fusion is enabled by default",
    )
    parser.set_defaults(bn_fusion=True)   

    args = parser.parse_args()

    with open(args.config) as fp:
        cfg = yaml.load(fp)

    validate(cfg, args)
