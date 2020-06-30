### Evaluation

For evaluating the PB-DGCNN using LocalEdgeConv w/normals on the ABC boundary dataset, you can use the ```evaluate_boundary_detection.py``` 
script as follows,

```bash
python evaluate_boundary_detection.py --model_path <pre_trained_model_path> --dataset <abc_evaluation_dataset_base_dir>

            # <pre_trained_model_path> = e.g. ../trained_models/abc_local_edge_conv_with_normals/model.ckpt
            # <abc_evaluation_dataset_base_dir> = e.g. ../datasets/ABC_dataset/h5/boundary_seg_poisson_10K_evaluation_h5

```
