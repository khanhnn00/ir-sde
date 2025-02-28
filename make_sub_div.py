# import os

# gt_train_src_dir = '/data1/nlp_hcm/nguyennnt/data/dataset/DIV2K/DIV2K_train_HR'
# lq_train_src_dir = '/data1/nlp_hcm/nguyennnt/data/dataset/DIV2K/DIV2K_train_LR_bicubic/X4'

# gt_train_dest_dir = '/data1/nlp_hcm/nguyennnt/data/dataset/DIV2K_sub/DIV2K_train_HR'
# lq_train_dest_dir = '/data1/nlp_hcm/nguyennnt/data/dataset/DIV2K_sub/DIV2K_train_LR_bicubic/X4'


# dest_dirs = [gt_train_dest_dir, lq_train_dest_dir]

# for i in dest_dirs:
#     if not os.path.exists(i):
#         os.makedirs(i)
        
# n_samples = 100

# fnames = os.listdir(gt_train_src_dir)
# sub, keep = fnames[:n_samples], fnames[n_samples:]

# for i in sub:
#     old_gt_name = os.path.join(gt_train_src_dir, i)
#     pref = i.split('.')[0]
#     old_lq_name = os.path.join(lq_train_src_dir, f'{pref}x4.png')
    
#     new_gt_name = os.path.join(gt_train_dest_dir, i)
#     new_lq_name = os.path.join(lq_train_dest_dir, f'{pref}x4.png')
    
#     os.rename(old_gt_name, new_gt_name)
#     os.rename(old_lq_name, new_lq_name)