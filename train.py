import argparse
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import os
import cv2
import numpy as np
from data import dataset
from model import network
import torch.nn.functional as F


def get_args():
    # Training settings
    parser = argparse.ArgumentParser(description='Semantic Human Matting !')
    parser.add_argument('--dataDir', default='./data/', help='dataset directory')
    parser.add_argument('--fgLists', type=list, default=[], required=True, help="training fore-ground images lists")
    parser.add_argument('--bg_list', type=str, required=True, help='train back-ground images list, one file')
    parser.add_argument('--dataRatio', type=list, default=[], required=True, help="train bg:fg raio, eg. [100]")
    parser.add_argument('--saveDir', default='./ckpt', help='model save dir')
    parser.add_argument('--trainData', default='human_matting_data', help='train dataset name')

    parser.add_argument('--continue_train', action='store_true', default=False, help='continue training the training')
    parser.add_argument('--pretrain', action='store_true', help='load pretrained model from t_net & m_net ')
    parser.add_argument('--without_gpu', action='store_true', default=False, help='no use gpu')

    parser.add_argument('--nThreads', type=int, default=4, help='number of threads for data loading')
    parser.add_argument('--train_batch', type=int, default=8, help='input batch size for train')
    parser.add_argument('--patch_size', type=int, default=400, help='patch size for train')

    parser.add_argument('--lr', type=float, default=1e-5, help='learning rate')
    parser.add_argument('--lrDecay', type=int, default=100)
    parser.add_argument('--lrdecayType', default='keep')
    parser.add_argument('--nEpochs', type=int, default=300, help='number of epochs to train')
    parser.add_argument('--save_epoch', type=int, default=5, help='number of epochs to save model')
    parser.add_argument('--print_iter', type=int, default=1000, help='pring loss and save image')

    parser.add_argument('--train_phase', default= 'end_to_end', help='train phase')
    parser.add_argument('--debug', action='store_true', default=False, help='debug mode')

    args = parser.parse_args()
    return args

def torch2numpy(tensor, without_gpu=False):
    if without_gpu:
        array = tensor.data.numpy()
    else:
        array = tensor.cpu().data.numpy()

    array = array * 255.0
    return array.astype(np.uint8)

def save_img(args, all_img, epoch, i=0):
    img_dir = os.path.join(args.saveDir, args.train_phase, 'save_img')
    if not os.path.isdir(img_dir):
        os.mkdir(img_dir)

    if args.train_phase == 'pre_train_t_net':
        img, trimap_pre, trimap_gt = all_img
        trimap_fake = torch.argmax(trimap_pre[0], dim=0)
        trimap_cat = torch.cat((trimap_gt[0][0], trimap_fake.float()), dim=-1)  # horizontal concate
        trimap_cat = torch.stack((trimap_cat,) * 3, dim=0)
        img_cat = torch.cat((img[0] * 255.0, trimap_cat * 127.5), dim=-1)
        if args.without_gpu:
            img_cat = img_cat.data.numpy()
        else:
            img_cat = img_cat.cpu().data.numpy()
        cv2.imwrite(img_dir + '/trimap_{}_{}.png'.format(str(epoch), str(i)),
                    img_cat.transpose((1, 2, 0)).astype(np.uint8))

    if args.train_phase == 'pre_train_m_net':
        img, alpha_pre, alpha_gt = all_img
        img = img[0] * 255.0
        alpha_pre = torch.stack((alpha_pre[0][0] * 255.0,) * 3, dim=0)
        alpha_gt = torch.stack((alpha_gt[0][0] * 255.0,) * 3, dim=0)
        img_cat = torch.cat((img, alpha_gt, alpha_pre), dim=-1)

        if args.without_gpu:
            img_cat = img_cat.data.numpy()
        else:
            img_cat = img_cat.cpu().data.numpy()
        cv2.imwrite(img_dir + '/alpha_{}_{}.png'.format(str(epoch), str(i)),
                    img_cat.transpose((1, 2, 0)).astype(np.uint8))

    if args.train_phase == 'end_to_end':
        img, trimap_gt, alpha_gt, trimap_pre, alpha_pre = all_img
        img = img[0] * 255.0
        trimap_pre = torch.argmax(trimap_pre[0], dim=0)
        trimap_pre = torch.stack((trimap_pre.float(),)*3, dim=0) * 127.5
        trimap_gt = torch.stack((trimap_gt[0][0],)*3, dim=0) * 127.5
        alpha_pre = torch.stack((alpha_pre[0][0] * 255.0,) * 3, dim=0)
        alpha_gt = torch.stack((alpha_gt[0][0] * 255.0,) * 3, dim=0)

        img_cat = torch.cat((img, trimap_gt, trimap_pre, alpha_gt, alpha_pre), dim=-1)
        #alpha_cat = torch.cat((alpha_gt, alpha_pre), dim=-1)
        if args.without_gpu:
            img_cat = img_cat.data.numpy()
        else:
            img_cat = img_cat.cpu().data.numpy()

        img_cat = img_cat.transpose((1,2,0)).astype(np.uint8)

        # save image
        cv2.imwrite(img_dir + '/e2e_{}_{}.png'.format(str(epoch), str(i)),img_cat)

def set_lr(args, epoch, optimizer):

    lrDecay = args.lrDecay
    decayType = args.lrdecayType
    if decayType == 'keep':
        lr = args.lr
    elif decayType == 'step':
        epoch_iter = (epoch + 1) // lrDecay
        lr = args.lr / 2**epoch_iter
    elif decayType == 'exp':
        k = math.log(2) / lrDecay
        lr = args.lr * math.exp(-k * epoch)
    elif decayType == 'poly':
        lr = args.lr * math.pow((1 - epoch / args.nEpochs), 0.9)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr

  

class Train_Log():
    def __init__(self, args):
        self.args = args

        self.save_dir = os.path.join(args.saveDir, args.train_phase)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        self.save_dir_model = os.path.join(self.save_dir, 'model')
        if not os.path.exists(self.save_dir_model):
            os.makedirs(self.save_dir_model)

        if os.path.exists(self.save_dir + '/log.txt'):
            self.logFile = open(self.save_dir + '/log.txt', 'a')
        else:
            self.logFile = open(self.save_dir + '/log.txt', 'w')

        # in case pretrained weights need to be loaded
        if self.args.pretrain:
            self.t_path = os.path.join(args.saveDir, 'pre_train_t_net', 'model', 'ckpt_lastest.pth')
            self.m_path = os.path.join(args.saveDir, 'pre_train_m_net', 'model', 'ckpt_lastest.pth')
            assert os.path.isfile(self.t_path) and os.path.isfile(self.m_path), \
                'Wrong dir for pretrained models:\n{},{}'.format(self.t_path, self.m_path)

            
    def save_model(self, model, epoch, save_as=False):
        if save_as:   # for args.save_epoch
            lastest_out_path = "{}/ckpt_{}.pth".format(self.save_dir_model, epoch)
            model_out_path = "{}/model_obj.pth".format(self.save_dir_model)
            torch.save(
                model,
                model_out_path)
        else:   # for regular save
            lastest_out_path = "{}/ckpt_lastest.pth".format(self.save_dir_model)

        torch.save({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            }, lastest_out_path)

    def load_pretrain(self, model):
        t_ckpt = torch.load(self.t_path)
        model.load_state_dict(t_ckpt['state_dict'], strict=False)
        m_ckpt = torch.load(self.m_path)
        model.load_state_dict(m_ckpt['state_dict'], strict=False)
        print('=> loaded pretrained t_net & m_net pretrained models !')

        return model

    def load_model(self, model):
        lastest_out_path = "{}/ckpt_lastest.pth".format(self.save_dir_model)
        ckpt = torch.load(lastest_out_path)
        start_epoch = ckpt['epoch'] + 1
        model.load_state_dict(ckpt['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})".format(lastest_out_path, ckpt['epoch']))

        return start_epoch, model    

    def save_log(self, log):
        self.logFile.write(log + '\n')

# initialise conv2d weights
def weight_init(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_normal_(m.weight.data)
        #n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #m.weight.data.normal_(0, math.sqrt(2. / n))
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()

def loss_f_T(trimap_pre, trimap_gt):
    criterion = nn.CrossEntropyLoss()
    L_t = criterion(trimap_pre, trimap_gt[:, 0, :, :].long())

    return L_t


def loss_f_M(img, alpha_pre, alpha_gt, bg, fg, trimap):
    # -------------------------------------
    # prediction loss L_p
    # ------------------------
    eps = 1e-6
    # l_alpha
    L_alpha = torch.sqrt(torch.pow(alpha_pre - alpha_gt, 2.) + eps).mean()

    comp_pred = alpha_pre * fg + (1 - alpha_pre) * bg

    # be careful about here: if img's range is [0,1] then eps should divede 255
    L_composition = torch.sqrt(torch.pow(img - comp_pred, 2.) + eps).mean()

    L_p = 0.5 * L_alpha + 0.5 * L_composition

    return L_p, L_alpha, L_composition

def loss_function(img, trimap_pre, trimap_gt, alpha_pre, alpha_gt, bg, fg):

    # -------------------------------------
    # classification loss L_t
    # ------------------------
    criterion = nn.CrossEntropyLoss()
    L_t = criterion(trimap_pre, trimap_gt[:,0,:,:].long())

    # -------------------------------------
    # prediction loss L_p
    # ------------------------
    eps = 1e-6
    # l_alpha
    L_alpha = torch.sqrt(torch.pow(alpha_pre - alpha_gt, 2.) + eps).mean()

    comp_pred = alpha_pre * fg + (1 - alpha_pre) * bg

    # be careful about here: if img's range is [0,1] then eps should divede 255
    L_composition = torch.sqrt(torch.pow(img - comp_pred, 2.) + eps).mean()
    L_p = 0.5 * L_alpha + 0.5 * L_composition

    # train_phase
    loss = L_p + 0.01*L_t
        
    return loss, L_alpha, L_composition, L_t


def main():
    args = get_args()

    if args.without_gpu:
        print("use CPU !")
        device = torch.device('cpu')
    else:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            print("No GPU is is available !")

    print("============> Building model ...")
    if args.train_phase == 'pre_train_t_net':
        model = network.net_T()
    elif args.train_phase == 'pre_train_m_net':
        model = network.net_M()
        model.apply(weight_init)
    elif args.train_phase == 'end_to_end':
        model = network.net_F()
        if args.pretrain:
            model = Train_Log.load_pretrain(model)
    else:
        raise ValueError('Wrong train phase request!')
    train_data = dataset.human_matting_data(args)
    model.to(device)

    # debug setting
    save_latest_freq = int(len(train_data)//args.train_batch*0.55)
    if args.debug:
        args.save_epoch = 1
        args.train_batch = 1  # defualt debug: 1
        args.nEpochs = 1
        args.print_iter = 1
        save_latest_freq = 10

    print(args)
    print("============> Loading datasets ...")

    trainloader = DataLoader(train_data,
                             batch_size=args.train_batch, 
                             drop_last=True, 
                             shuffle=True, 
                             num_workers=args.nThreads, 
                             pin_memory=True)

    print("============> Set optimizer ...")
    lr = args.lr
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), \
                                   lr=lr, betas=(0.9, 0.999), 
                                   weight_decay=0.0005)    

    print("============> Start Train ! ...")
    start_epoch = 1
    trainlog = Train_Log(args)
    if args.continue_train:
        start_epoch, model = trainlog.load_model(model)

    model.train() 
    for epoch in range(start_epoch, args.nEpochs+1):

        loss_ = 0
        L_alpha_ = 0
        L_composition_ = 0
        L_cross_ = 0
        if args.lrdecayType != 'keep':
            lr = set_lr(args, epoch, optimizer)

        t0 = time.time()
        for i, sample_batched in enumerate(trainloader):

            optimizer.zero_grad()

            if args.train_phase == 'pre_train_t_net':
                img, trimap_gt = sample_batched['image'], sample_batched['trimap']
                img, trimap_gt = img.to(device), trimap_gt.to(device)

                trimap_pre = model(img)
                if args.debug:  #debug only
                    assert tuple(trimap_pre.shape) == (args.train_batch, 3, args.patch_size, args.patch_size)
                    assert tuple(trimap_gt.shape) == (args.train_batch, 1, args.patch_size, args.patch_size)

                loss = loss_f_T(trimap_pre, trimap_gt)

                loss_ += loss.item()

                if i!=0 and i % args.print_iter == 0:
                    save_img(args, (img, trimap_pre, trimap_gt), epoch, i)
                    print("[epoch:{} iter:{}]  \tloss: {:.5f}".format(epoch, i, loss))
                if i!=0 and i % save_latest_freq == 0:
                    print("average loss: {:.5f}\nsaving model ....".format(loss_ / (i+1)))
                    trainlog.save_model(model, epoch)

            elif args.train_phase == 'pre_train_m_net':
                img, trimap_gt, alpha_gt, bg, fg = sample_batched['image'], sample_batched['trimap'], sample_batched['alpha'], sample_batched['bg'], sample_batched['fg']
                img, trimap_gt, alpha_gt, bg, fg = img.to(device), trimap_gt.to(device), alpha_gt.to(device), bg.to(device), fg.to(device)

                alpha_pre = model(img, trimap_gt)
                if args.debug:
                    assert tuple(alpha_pre.shape) == (args.train_batch, 1, args.patch_size, args.patch_size)
                    img_dir = os.path.join(args.saveDir, args.train_phase, 'save_img')
                    img_fg_bg = np.concatenate((torch2numpy(trimap_gt[0]), torch2numpy(fg[0]), torch2numpy(bg[0])), axis=-1)
                    img_fg_bg = np.transpose(img_fg_bg, (1,2,0))
                    cv2.imwrite(img_dir + '/fgbg_{}_{}.png'.format(str(epoch), str(i)), img_fg_bg)

                loss, L_alpha, L_composition = loss_f_M(img, alpha_pre, alpha_gt, bg, fg, trimap_gt)

                loss_ += loss.item()
                L_alpha_ += L_alpha.item()
                L_composition_ += L_composition.item()

                if i!=0 and i % args.print_iter == 0:
                    save_img(args, (img, alpha_pre, alpha_gt), epoch, i)
                    print("[epoch:{} iter:{}]  loss: {:.5f}  loss_a: {:.5f}  loss_c: {:.5f}"\
                          .format(epoch, i, loss, L_alpha, L_composition))
                if i!=0 and i % save_latest_freq == 0:
                    print("average loss: {:.5f}\nsaving model ....".format(loss_ / (i + 1)))
                    trainlog.save_model(model, epoch)

            elif args.train_phase == 'end_to_end':
                img, trimap_gt, alpha_gt, bg, fg = sample_batched['image'], sample_batched['trimap'], sample_batched['alpha'], sample_batched['bg'], sample_batched['fg']
                img, trimap_gt, alpha_gt, bg, fg = img.to(device), trimap_gt.to(device), alpha_gt.to(device), bg.to(device), fg.to(device)

                trimap_pre, alpha_pre = model(img)
                loss, L_alpha, L_composition, L_cross = loss_function(img, trimap_pre, trimap_gt, alpha_pre, alpha_gt, bg)

                loss_ += loss.item()
                L_alpha_ += L_alpha.item()
                L_composition_ += L_composition.item()
                L_cross_ += L_cross.item()

            loss.backward()
            optimizer.step()


        # shuffle data after each epoch to recreate the dataset
        print('epoch end, shuffle datasets again ...')
        train_data.shuffle_data()
        #trainloader.dataset.shuffle_data()

        t1 = time.time()

        if args.train_phase == 'pre_train_t_net':
            loss_ = loss_ / (i+1)
            log = "[{} / {}] \tloss: {:.5f}\ttime: {:.0f}".format(epoch, args.nEpochs, loss_, t1-t0)

        elif args.train_phase == 'pre_train_m_net':
            loss_ = loss_ / (i + 1)
            L_alpha_ = L_alpha_ / (i + 1)
            L_composition_ = L_composition_ / (i + 1)
            log = "[{} / {}]  loss: {:.5f}  loss_a: {:.5f}  loss_c: {:.5f}  time: {:.0f}"\
                .format(epoch, args.nEpochs,
                        loss_,
                        L_alpha_,
                        L_composition_,
                        t1 - t0)

        elif args.train_phase == 'end_to_end':
            loss_ = loss_ / (i+1)
            L_alpha_ = L_alpha_ / (i+1)
            L_composition_ = L_composition_ / (i+1)
            L_cross_ = L_cross_ / (i+1)

            log = "[{} / {}]  loss: {:.5f}  loss_a: {:.5f}  loss_c: {:.5f}  loss_t: {:.5f}  time: {:.0f}"\
                     .format(epoch, args.nEpochs,
                            loss_,
                            L_alpha_,
                            L_composition_,
                            L_cross_,
                            t1 - t0)
        print(log)
        trainlog.save_log(log)
        trainlog.save_model(model, epoch)

        if epoch % args.save_epoch == 0:
            trainlog.save_model(model, epoch, save_as=True)



if __name__ == "__main__":
    main()
