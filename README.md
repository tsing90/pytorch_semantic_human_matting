# pytorch_semantic_human_matting
This is an unofficial implementation of the paper "Semantic human matting". 

# testing environment:
Ubuntu 16.04 \n
Pytorch 0.4.1


# pre_trian T_net
python train.py --patch_size=400 --nEpochs=500 --save_epoch=5 --train_batch=8 --train_phase=pre_train_t_net

optional: --continue_train

# pre_train M_net
python train.py --patch_size=400 --nEpochs=500 --save_epoch=1 --train_batch=8 --train_phase=pre_train_m_net

optional: --continue_train

# end to end
python train.py
--patch_size=800
--nEpochs=500
--lr=1e-5
--save_epoch=10
--train_batch=8
--continue_train
--train_phase=end_to_end


# testing
python test.py --train_phase=end_to_end
