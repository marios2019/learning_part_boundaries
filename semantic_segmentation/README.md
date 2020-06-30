### Semantic Segmentation

In here we incorporate a graph cuts formulation with per-point part probability used as a unary term and the boundary probabilities
and/or normal angle differences. For the graph-cuts implementation we are using the [GCoptimization Library Version 2.3](https://cs.uwaterloo.ca/~oveksler/OldCode.html)
along with [MATLAB Engine API for Python](https://www.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html).
You can use the ```semantic_segmentation.py``` script as follows
```bash
python semantic_segmentation.py --dataset <partnet_semantic_dataset_base_dir> --category <cat_name> --semantic_model_path <sem_model_path> \
                                --boundary_model_path <boundary_model_path> --pairwise_features <pt_feat> 
            # <partnet_semantic_dataset_base_dir> = e.g. ../datasets/partnet_dataset/h5/sem_seg_poisson_h5_release
            # <cat_name> = e.g Bag
            # <sem_model_path> = e.g. ../trained_models/partnet_sem_seg_models/Bag/model.ckpt
            # <boundary_model_path> = e.g. ../trained_models/partnet_boundary_detection_models/Bag/model.ckpt
            # <pt_feat> = (1) normal_angles, (2) boundary_confidence, (3) combine (weighted combination of (1) and (2)
```


