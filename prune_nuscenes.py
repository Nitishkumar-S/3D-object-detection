import pickle
import os
from tqdm import tqdm

def prune_infos(input_pkl, output_pkl, dataset_root):
    print(f"Loading {input_pkl}...")
    with open(input_pkl, 'rb') as f:
        data = pickle.load(f)
    
    if isinstance(data, dict) and 'data_list' in data:
        info_list = data['data_list']
        format_type = 'new'
    elif isinstance(data, dict) and 'infos' in data:
        info_list = data['infos']
        format_type = 'legacy'
    else:
        info_list = data
        format_type = 'list'

    pruned_infos = []

    print("Filtering missing samples...")
    for info in tqdm(info_list):
        lidar_path = None
        
        if 'lidar_points' in info and 'lidar_path' in info['lidar_points']:
            lidar_path = info['lidar_points']['lidar_path']
        elif 'lidar_path' in info:
            lidar_path = info['lidar_path']
            
        if lidar_path:
            # Manually add the 'samples/LIDAR_TOP' folders to the path
            if not lidar_path.startswith('samples'):
                full_path = os.path.join(dataset_root, 'samples', 'LIDAR_TOP', lidar_path)
            else:
                full_path = os.path.join(dataset_root, lidar_path)
            
            # Only keep the sample if the file physically exists
            if os.path.exists(full_path):
                pruned_infos.append(info)

    # Reconstruct the dictionary based on its original format
    if format_type == 'new':
        data['data_list'] = pruned_infos
    elif format_type == 'legacy':
        data['infos'] = pruned_infos
    else:
        data = pruned_infos

    print(f"Saving pruned data to {output_pkl}...")
    with open(output_pkl, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"Original samples: {len(info_list)} | Pruned samples kept: {len(pruned_infos)}\n")

# Run the pruning
root_dir = "./mmdetection3d/data/nuscenes" 
prune_infos(f'{root_dir}/nuscenes_infos_train.pkl', f'{root_dir}/nuscenes_infos_train_pruned.pkl', root_dir)
prune_infos(f'{root_dir}/nuscenes_infos_val.pkl', f'{root_dir}/nuscenes_infos_val_pruned.pkl', root_dir)