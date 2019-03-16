
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
        print("----------------------------------------------------------")
        print("|       use GPU !      ||   Available GPU number is {} !  |".format(n_gpu))
        print("----------------------------------------------------------")

        device = torch.device('cuda:0,1')

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

def seg_process(args, inputs, net, trimap=None):

    if args.train_phase == 'pre_train_t_net':
        #origin_w, origin_h, _ = inputs.shape
        trimap = net(inputs)
        trimap = torch.argmax(trimap[0], dim=0)
        trimap = trimap * 127.5
        # print((time.time() - t0))

        if args.without_gpu:
            trimap_np = trimap.data.numpy()
        else:
            trimap_np = trimap.cpu().data.numpy()

        trimap_np = trimap_np.astype(np.uint8)
        #trimap_out = cv2.resize(trimap_np, (origin_w, origin_h), interpolation=cv2.INTER_CUBIC)

        return trimap_np

    elif args.train_phase == 'pre_train_m_net':
        assert isinstance(trimap, np.ndarray), 'trimap is None !'
        alpha = net(inputs[0], inputs[1])

        if args.without_gpu:
            alpha = alpha.data.numpy()
        else:
            alpha = alpha.cpu().data.numpy()

        alpha = alpha[0][0] * 255.0
        alpha[trimap==255] = 255
        alpha[trimap== 0 ] = 0
        alpha = alpha.astype(np.uint8)

        return alpha

    else:
        # TODO: to be implemented
        pass
        """
        fg = np.multiply(alpha_np[..., np.newaxis], image)
        bg = image
        bg_gray = np.multiply(1-alpha_np[..., np.newaxis], image)
        bg_gray = cv2.cvtColor(bg_gray, cv2.COLOR_BGR2GRAY)
    
        bg[:,:,0] = bg_gray
        bg[:,:,1] = bg_gray
        bg[:,:,2] = bg_gray
    
        # fg[fg<=0] = 0
        # fg[fg>255] = 255
        # fg = fg.astype(np.uint8)
        # out = cv2.addWeighted(fg, 0.7, bg, 0.3, 0)
        out = fg + bg
        out[out<0] = 0
        out[out>255] = 255
        out = out.astype(np.uint8)
        """


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
            # TODO: end2end phase
            pass

        # show a frame
        cv2.imwrite(out_dir+imgname, frame_seg)

    print('Average time cost: {:.0f} s/image'.format((time.time() - t0) / len(imgList)))
    print('output images were saved at: ', out_dir)

def main(args):

    myModel = load_model(args)
    test(args, myModel)


if __name__ == "__main__":
    main(args)


