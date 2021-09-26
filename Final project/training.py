# **********************************
# *      INFO8010 - Project        *
# *  CycleGAN for style transfer   *
# *             ---                *
# *  Antoine DEBOR & Pierre NAVEZ  *
# *       ULiège, May 2021         *
# **********************************

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm # progress bar
from torch.autograd import grad
from torch.nn import functional as F
from torchvision.utils import save_image
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import pickle
from generator import Generator
from discriminator import Discriminator
import config
from data import make_loader


def training_fun(gX, gY, dX, dY, d_scaler, g_scaler, dOptim, gOptim, Y_loader, X_loader, L1, MSE, gen_loss_coeff):

    adv_coeff, cycle_coeff, identity_coeff = gen_loss_coeff
    mean_d_loss = 0
    mean_g_loss = 0
    mean_dx_rl = 0
    mean_dy_rl = 0
    mean_dx_fk = 0
    mean_dy_fk = 0
    mean_gx_adv  = 0
    mean_gy_adv  = 0
    mean_gx_cycle = 0
    mean_gy_cycle  = 0
    mean_gx_id = 0
    mean_gy_id = 0

    L1_norm_fctr = config.L1_NORM_FACTOR

    for idx, (X, Y) in enumerate(tqdm(zip(X_loader, Y_loader), leave=True)):
        # Transfer data to GPU
        X = X.to(config.DEVICE) # Faces
        Y = Y.to(config.DEVICE) # Paintings/Simpsons

        X.requires_grad = True
        Y.requires_grad = True

        # -- Discriminators training --

        # Computing the loss
        with torch.cuda.amp.autocast(): # Automatic mixed precision handling
            # Generating fake images (X->Y and Y->X)
            X_fk = gX(Y)
            Y_fk = gY(X)

            # Discriminating
            # Real data
            dX_rl = dX(X)
            dY_rl = dY(Y)
            # Fake data
            # Note : we need to detach the generated data from its computational graph
            dX_fk = dX(X_fk.detach())
            dY_fk = dY(Y_fk.detach())

            # MSE loss
            # Between real data and True
            dX_rl_loss = MSE(dX_rl, torch.ones_like(dX_rl))
            dY_rl_loss = MSE(dY_rl, torch.ones_like(dY_rl))
            # Between fake data and False
            dX_fk_loss = MSE(dX_fk, torch.zeros_like(dX_fk))
            dY_fk_loss = MSE(dY_fk, torch.zeros_like(dY_fk))

            # Discriminators total loss
            d_loss = (dX_rl_loss + dY_rl_loss + dX_fk_loss + dY_fk_loss) / 2 #+ R1_penalty # "/2" maybe not mandatory, we could try without it later on
            print("\nDiscriminators total loss : ", d_loss.item())
            mean_d_loss += d_loss.item()
            mean_dx_rl += dX_rl_loss.item()
            mean_dy_rl += dY_rl_loss.item()
            mean_dx_fk += dX_fk_loss.item()
            mean_dy_fk += dY_fk_loss.item()

        # Optimizing
        dOptim.zero_grad()
        d_scaler.scale(d_loss).backward()
        d_scaler.step(dOptim)
        d_scaler.update()

        # -- Generators training --

        # Computing the loss
        with torch.cuda.amp.autocast(): # Automatic mixed precision handling
            # I) Adversial loss (MSE)
            # Discriminating
            dX_fk = dX(X_fk) # Note : no need to detach anymore !
            dY_fk = dY(Y_fk)
            # MSE loss between fake data and True
            gX_adv_loss = MSE(dX_fk, torch.ones_like(dX_fk))
            gY_adv_loss = MSE(dY_fk, torch.ones_like(dY_fk))

            # II) Cycle consistency loss (L1)
            # Generating fake images (cycle fakes) from fake images (X->Y->X and Y->X->Y)
            X_fk_cycle = gX(Y_fk)
            Y_fk_cycle = gY(X_fk)
            # L1 loss between cycle fakes and ground truth
            gX_cycle_loss = L1(X_fk_cycle, X) / L1_norm_fctr
            gY_cycle_loss = L1(Y_fk_cycle, Y) / L1_norm_fctr

            # III) Identity mapping loss (L1) (used for paintings)
            # Generating fake 'self' images (X->X and Y->Y)
            X_fk_self = gX(X)
            Y_fk_self = gY(Y)
            # L1 loss between self fakes and ground truth
            gX_identity_loss = L1(X_fk_self, X) / L1_norm_fctr
            gY_identity_loss = L1(Y_fk_self, Y) / L1_norm_fctr

            # Generators total loss
            g_loss = (adv_coeff * (gX_adv_loss + gY_adv_loss)
                    + cycle_coeff * (gX_cycle_loss + gY_cycle_loss)
                    + identity_coeff * (gX_identity_loss * cycle_coeff + gY_identity_loss * cycle_coeff))
            print("\nGenerators total loss : ", g_loss.item())
            mean_g_loss += g_loss.item()
            mean_gx_adv  += gX_adv_loss.item()
            mean_gy_adv  += gY_adv_loss.item()
            mean_gx_cycle += gX_cycle_loss.item()
            mean_gy_cycle  += gY_cycle_loss.item()
            mean_gx_id += gX_identity_loss.item()
            mean_gy_id += gY_identity_loss.item()

        # Optimizing
        gOptim.zero_grad()
        g_scaler.scale(g_loss).backward()
        g_scaler.step(gOptim)
        g_scaler.update()

        # -- Image saving --
        if idx % 20 == 0:
          save_image(X_fk*0.5 + 0.5, config.SAVE_PATH + f"{idx}_X_fake_"+config.MODE+".png")
          save_image(Y_fk*0.5 + 0.5, config.SAVE_PATH + f"{idx}_Y_fake_"+config.MODE+".png")

          save_image(X_fk_cycle*0.5 + 0.5, config.SAVE_PATH + f"{idx}_X_cycle_"+config.MODE+".png")
          save_image(Y_fk_cycle*0.5 + 0.5, config.SAVE_PATH + f"{idx}_Y_cycle_"+config.MODE+".png")

          save_image(X*0.5 + 0.5, config.SAVE_PATH + f"{idx}_X_original_"+config.MODE+".png")
          save_image(Y*0.5 + 0.5, config.SAVE_PATH + f"{idx}_Y_original_"+config.MODE+".png")

          save_image(X_fk_self*0.5 + 0.5, config.SAVE_PATH + f"{idx}_X_identity_"+config.MODE+".png")
          save_image(Y_fk_self*0.5 + 0.5, config.SAVE_PATH + f"{idx}_Y_identity_"+config.MODE+".png")

    mean_d_loss /= float(idx+1)
    mean_g_loss /= float(idx+1)
    mean_dx_rl /= float(idx+1)
    mean_dy_rl /= float(idx+1)
    mean_dx_fk /= float(idx+1)
    mean_dy_fk /= float(idx+1)
    mean_gx_adv  /= float(idx+1)
    mean_gy_adv  /= float(idx+1)
    mean_gx_cycle /= float(idx+1)
    mean_gy_cycle  /= float(idx+1)
    mean_gx_id /= float(idx+1)
    mean_gy_id /= float(idx+1)

    return mean_d_loss, mean_g_loss, mean_dx_rl , mean_dy_rl , mean_dx_fk , mean_dy_fk , mean_gx_adv  , mean_gy_adv  , mean_gx_cycle , mean_gy_cycle  , mean_gx_id , mean_gy_id

def show_images(img):
    img = img
    npimg = img.numpy() * .5 + .5
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def training_process(X_path, Y_path):
    # DATALOADERS
    #############
    X_trainloader, Y_trainloader = make_loader(X_path, Y_path, train=True)
    Y_dataiter = iter(Y_trainloader)
    X_dataiter = iter(X_trainloader)
    Ys = Y_dataiter.next()
    Xs = X_dataiter.next()
    show_images(torchvision.utils.make_grid(Ys, nrow=5))
    show_images(torchvision.utils.make_grid(Xs, nrow=5))

    # MODULES
    #########
    n_residuals = 9 if config.IMG_DIM >= 256 else 6
    gX = Generator(n_residuals=n_residuals).to(config.DEVICE)
    gY = Generator(n_residuals=n_residuals).to(config.DEVICE)
    dX = Discriminator(in_channels=3).to(config.DEVICE)
    dY = Discriminator(in_channels=3).to(config.DEVICE)

    # OPTIMIZERS
    ############
    gOptim = optim.Adam(params=list(gX.parameters()) + list(gY.parameters()), lr=config.LR, betas=config.BETAS)
    dOptim = optim.Adam(params=list(dX.parameters()) + list(dY.parameters()), lr=config.LR, betas=config.BETAS)

    # LOSSES
    ########
    L1_loss = nn.L1Loss()
    mse_loss = nn.MSELoss()

    # PLOT VAR
    ##########
    d_loss_epochs = []
    dX_rl_loss_epochs = []
    dY_rl_loss_epochs = []
    dX_fk_loss_epochs = []
    dY_fk_loss_epochs = []
    g_loss_epochs = []
    gX_adv_loss_epochs = []
    gY_adv_loss_epochs = []
    gX_cycle_loss_epochs = []
    gY_cycle_loss_epochs = []
    gX_identity_loss_epochs = []
    gY_identity_loss_epochs = []

    # SCALERS
    #########
    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()

    # EPOCHS
    ########
    for epoch in range(config.N_EPOCHS):
        print("\nepoch ", epoch)

        d_loss_epoch, g_loss_epoch, dX_rl_loss_epoch, dY_rl_loss_epoch, dX_fk_loss_epoch, dY_fk_loss_epoch, gX_adv_loss_epoch, gY_adv_loss_epoch, gX_cycle_loss_epoch, gY_cycle_loss_epoch, gX_identity_loss_epoch, gY_identity_loss_epoch = training_fun(gX, gY, dX, dY, d_scaler, g_scaler, dOptim, gOptim, Y_trainloader, X_trainloader, L1_loss, mse_loss, config.GEN_LOSS_COEFF)

        d_loss_epochs.append(d_loss_epoch)
        g_loss_epochs.append(g_loss_epoch)
        dX_rl_loss_epochs.append(dX_rl_loss_epoch)
        dY_rl_loss_epochs.append(dY_rl_loss_epoch)
        dX_fk_loss_epochs.append(dX_fk_loss_epoch)
        dY_fk_loss_epochs.append(dY_fk_loss_epoch)
        gX_adv_loss_epochs.append(gX_adv_loss_epoch)
        gY_adv_loss_epochs.append(gY_adv_loss_epoch)
        gX_cycle_loss_epochs.append(gX_cycle_loss_epoch)
        gY_cycle_loss_epochs.append(gY_cycle_loss_epoch)
        gX_identity_loss_epochs.append(gX_identity_loss_epoch)
        gY_identity_loss_epochs.append(gY_identity_loss_epoch)

    # SAVE MODELS AND OPTIMIZERS
    ############################
    torch.save({
            'gX_state_dict_'+config.MODE: gX.state_dict(),
            'gY_state_dict_'+config.MODE: gY.state_dict(),
            'dX_state_dict_'+config.MODE: dX.state_dict(),
            'dY_state_dict_'+config.MODE: dY.state_dict(),
            'g_optimizer_state_dict_'+config.MODE: gOptim.state_dict(),
            'd_optimizer_state_dict_'+config.MODE: dOptim.state_dict()
            }, config.SAVE_MODEL_PATH)

    return (d_loss_epochs, g_loss_epochs, dX_rl_loss_epochs, dY_rl_loss_epochs, dX_fk_loss_epochs, dY_fk_loss_epochs, gX_adv_loss_epochs, gY_adv_loss_epochs, gX_cycle_loss_epochs, gY_cycle_loss_epochs, gX_identity_loss_epochs, gY_identity_loss_epochs)
