
import time
import cv2
import torch 
import argparse
import numpy as np
import os 
import torch.nn.functional as F

parser = argparse.ArgumentParser(description='human matting')
parser.add_argument('--model', default='./ckpt', help='root dir of preTrained model')
parser.add_argument('--size', type=int, default=400, help='input size')
parser.add_argument('--without_gpu', action='store_true', default=False, help='no use gpu')
parser.add_argument('--train_phase', default='end_to_end',help='which phase of the model')

args = parser.parse_args()

torch.set_grad_enabled(False)

    
#################################
#----------------
if args.without_gpu:
    print("use CPU !")
    device = torch.device('cpu')
else:
    if torch.cuda.is_available():
        n_gpu = torch.cuda.device_count()
        print("use GPU")
        device = torch.device('cuda')

#################################
#---------------
def load_model(args):
    model_path = os.path.join(args.model, args.train_phase, 'model/model_obj.pth')
    assert os.path.isfile(model_path), 'Wrong model path: {}'.format(model_path)
    print('Loading model from {}...'.format(model_path))

    if args.without_gpu:
        myModel = torch.load(model_path, map_location=lambda storage, loc: storage)
    else:
        myModel = torch.load(model_path)

    myModel.eval()
    myModel.to(device)
    
    return myModel

def seg_process(args, inputs, net):

    if args.train_phase == 'pre_train_t_net':
        trimap = net(inputs)
        trimap = torch.argmax(trimap[0], dim=0)
        trimap[trimap == 0] = 0
        trimap[trimap == 1] = 128
        trimap[trimap == 2] = 255

        if args.without_gpu:
            trimap_np = trimap.data.numpy()
        else:
            trimap_np = trimap.cpu().data.numpy()

        trimap_np = trimap_np.astype(np.uint8)
        return trimap_np

    elif args.train_phase == 'pre_train_m_net':
        alpha = net(inputs[0], inputs[1])

        if args.without_gpu:
            alpha = alpha.data.numpy()
        else:
            alpha = alpha.cpu().data.numpy()

        alpha = alpha[0][0] * 255.0
        alpha = alpha.astype(np.uint8)

        return alpha

    else:
        alpha = net(inputs)

        if args.without_gpu:
            alpha = alpha.data.numpy()
        else:
            alpha = alpha.cpu().data.numpy()

        alpha = alpha[0][0] * 255.0
        alpha = alpha.astype(np.uint8)
        return alpha

def test(args, net):

    t0 = time.time()
    out_dir = 'result/' + args.train_phase + '/'
    os.makedirs(out_dir, exist_ok=True)

    # get a frame
    imgList = os.listdir('result/test')
    if imgList==[]:
        raise ValueError('Empty dir at: ./result/test')
    for imgname in imgList:
        img = cv2.imread('result/test/'+imgname)

        if args.train_phase == 'pre_train_t_net':
            img = img / 255.0

            tensor_img = torch.from_numpy(img.astype(np.float32)[np.newaxis, :, :, :]).permute(0, 3, 1, 2)

            tensor_img = tensor_img.to(device)

            frame_seg = seg_process(args, tensor_img, net)

        elif args.train_phase == 'pre_train_m_net':
            # No resize
            h, w, _ = img.shape
            img = img / 255.0

            tri_path = 'result/trimap/'+imgname
            assert os.path.isfile(tri_path), 'wrong trimap path: {}'.format(tri_path)

            trimap_src = cv2.imread(tri_path, 0)

            trimap = trimap_src.copy()
            trimap[trimap == 0] = 0
            trimap[trimap == 128] = 1
            trimap[trimap == 255] = 2
            trimap = np.eye(3)[trimap.reshape(-1)].reshape(list(trimap.shape) + [3])

            tensor_img = torch.from_numpy(img.astype(np.float32)[np.newaxis, :, :, :]).permute(0, 3, 1, 2)
            tensor_tri = torch.from_numpy(trimap.astype(np.float32)[np.newaxis, :, :, :]).permute(0, 3, 1, 2)
            tensor_img = tensor_img.to(device)
            tensor_tri = tensor_tri.to(device)

            frame_seg = seg_process(args, (tensor_img, tensor_tri), net, trimap=trimap_src)

        else:
            img = img / 255.0
            tensor_img = torch.from_numpy(img.astype(np.float32)[np.newaxis, :, :, :]).permute(0, 3, 1, 2)
            tensor_img = tensor_img.to(device)
            frame_seg = seg_process(args, tensor_img, net)

        # show a frame
        cv2.imwrite(out_dir+imgname, frame_seg)

    print('Average time cost: {:.0f} s/image'.format((time.time() - t0) / len(imgList)))
    print('output images were saved at: ', out_dir)

def main(args):

    myModel = load_model(args)
    test(args, myModel)


if __name__ == "__main__":
    main(args)


