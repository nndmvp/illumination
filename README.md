首先在 train_ddpm.py 配置 ir_path、vi_path、source_path，全部配置同一文件夹即可，要求是异常曝光的图片数据集。   

预训练完成后，从 experients 找到你训练好的 ddpm 文件夹，找到 checkpoint，配置到 config 的 fusion_train.json 里面，将 path 的 resume_state 配置成 checkpoint 的路径，注意最后的文件只截取前面两项，如 I70000_E1459。  

然后配置 n2.py 以下内容：  

gt_dir = r"../high"  
input_dir = r"../low"  
output_dir = r"./lol"  
test_dir = r"../test"  
test_gt_dir = r"../test"  
valid_input_dir = r"../val"  
valid_gt_dir = r"../val_true"  

然后训练即可。
