### Boundary Detection training

For training either on the ABC boundary dataset or PartNet boundary dataset, using LocalEdgeConv w/normals, you can use the ```train_multi_gpu.py``` 
script as follows, 

- ABC dataset training
```bash
python train_multi_gpu.py --dataset <abc_dataset_base_dir> --abc_dataset --align_pointclouds --output_dir <output-path> --batch_norm --num_gpu <k>

            # <abc_dataset_base_dir> = e.g. ../datasets/ABC_dataset/h5/boundary_seg_poisson_10K_h5
            # --abc_dataset: to use ABC dataset
            # --align_pointclouds: use spatial transformer module
            # --batch_norm: enable batch normalization 
            # --num_gpu: number of GPUs used for training
```

- PartNet dataset training
```bash
python train_multi_gpu.py --dataset <partnet_boundary_dataset_base_dir> --category <cat_name> --output_dir <output-path> --batch_norm 

            # <partnet_boundary_dataset_base_dir> = e.g. ../datasets/partnet_dataset/h5/boundary_seg_poisson_h5
            # <cat_name> = Bag
            # --batch_norm: enable batch normalization
            # --num_gpu: number of GPUs used for training
            # no spatial transformer was used in our PartNet experiments
```

For ```batch_size=8``` you will need 2 GPUs with 12GB memory each.
