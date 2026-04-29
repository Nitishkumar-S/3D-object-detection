Create a virtual environment and activate it
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

Install requirements inside the environment
pip install --no-cache -r requirements.txt

Then run below command to download mmcv and mmdet3d aligning with Cuda 12.1 compatibility
mim install mmengine "mmcv==2.1.0" "mmdet>=3.0.0" mmdet3d

Download the trainval data in root/mmdetection3d/data/nuscenes folder
from inside the data/nuscenes run below commands
tar -xf v1.0-trainval_meta.tgz
tar -xf v1.0-trainval_01_blobs.tgz

Generate the '.pkl' Database for Centerpoint
Make sure there is lot of RAM (32GB?)
python tools/create_data.py nuscenes \
    --root-path ./data/nuscenes \
    --out-dir ./data/nuscenes \
    --extra-tag nuscenes \
    --version v1.0

After generating data, prune nuscenes_info_train.pkl and nuscenes_info_val.pkl as it has metadata of missing files.
python prune_nuscenes.py

Train
python tools/train.py checkpoints/centerpoint_voxel01_second_secfpn_head-circlenms_8xb4-cyclic-20e_nus-3d.py

Evaluate
python tools/test.py \
    checkpoints/centerpoint_voxel01_second_secfpn_head-circlenms_8xb4-cyclic-20e_nus-3d.py \
    checkpoints/centerpoint_01voxel_second_secfpn_circlenms_4x8_cyclic_20e_nus_20220810_030004-9061688e.pth \
    --work-dir work_dirs/val_report

## Some hacks for evaluating only on the 1st blob of trainval
1. In /work/dlclarge1/solpuren-object_detection/3D-object-detection/.venv/lib/python3.10/site-packages/nuscenes/eval/detection/evaluate.py

# assert set(self.pred_boxes.sample_tokens) == set(self.gt_boxes.sample_tokens), \

        #     "Samples in split doesn't match samples in predictions."



        # --- CUSTOM FIX FOR PRUNED DATASET ---

        # Only evaluate the ground truth boxes that we actually have predictions for

        valid_tokens = set(self.pred_boxes.sample_tokens)

        self.gt_boxes.boxes = {k: v for k, v in self.gt_boxes.boxes.items() if k in valid_tokens}

        # -------------------------------------



2. test_dataloader, val_dataloader, test_evaluator, val_evaluator, train_dataloader, train_pipeline to pruned.pkl in centerpoint config python file



3. Comment out mmengine.check_file_exist(lidar_path) in nuscenes_converter.py

4. Added below code chunk in two places in create_gt_database.py to skip checking for non-existing trainval blobs
# --- Partial train FIX START ---
        try:
            example = self.pipeline(input_dict)
        except FileNotFoundError:
            return single_db_infos  # Return empty dict if file is missing

5. Updated centerpoint config to load pre-trained model
load_from = 'checkpoints/centerpoint_01voxel_second_secfpn_circlenms_4x8_cyclic_20e_nus_20220810_030004-9061688e.pth'

6. updated mmdetection3d/mmdet3d/datasets/transforms/dbsampler.py
Changed np.long to int as np.long is deprecated in newer versions of numpy

For Idea2: CBAM
1. modified '/work/dlclarge1/solpuren-object_detection/3D-object-detection/mmdetection3d/mmdet3d/models/necks/second_fpn.py' to add new class

2. Modified config to use CBAMSECONDFPN in pts_neck

For generating final results on test data I had to change config.py as
1. test dataloader - pointed to test.pkl
2. test evaluator - pointed to test.pkl and format_only=True and jsonfile_prefix='work_dirs/official_submission'
3. And command
python tools/test.py \
    checkpoints/centerpoint_voxel01_second_secfpn_head-circlenms_8xb4-cyclic-20e_nus-3d.py \
    work_dirs/centerpoint_voxel01_second_secfpn_head-circlenms_8xb4-cyclic-20e_nus-3d/epoch_20.pth