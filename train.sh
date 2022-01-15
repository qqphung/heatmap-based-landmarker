CUDA_VISIBLE_DEVICES=0 python3 train.py --snapshot ckpt_entropy_weight_gaussian\
                                        --dataroot /home/ubuntu/A-Practical-Facial-Landmark-Detector/data/train_data/imgs\
                                        --val_dataroot /home/ubuntu/A-Practical-Facial-Landmark-Detector/data/test_data/imgs\
                                        --get_topk_in_pred_heats_training 0\
                                        --lr 0.001\
                                        --step_size 20\
                                        --gamma 0.5\
                                        --random_round_with_gaussian 1\
                                        --mode train
