
import json
from ultralytics import YOLO
import os
import motmetrics as mm
import pandas as pd
import numpy as np

# Directory containing the JSON files
roboflow_images_directory =  "/scratch2/devashree/yolov8/images/"
roboflow_label_directory = '/scratch2/devashree/yolov8/labels'

model = YOLO("/scratch2/devashree/yolov8/datasets/runs/detect/yolov8m/weights/best.pt")
model_name = "yolov8"

results =  model.track(source = roboflow_images_directory, tracker = "bytetrack.yaml")

# Save results in MOTChallenge format (frame, id, bbox, conf)
with open(f'tracker_results_{model_name}.txt', 'w') as f:
    for frame_id, result in enumerate(results):
        for box in result.boxes:
            if box.id:
                bbox = box.xyxyn[0].tolist()  # Convert from tensor to list
                track_id = box.id.item()  # Get track id
                conf = box.conf.item()  # Get confidence score
                f.write(f'{frame_id+1},{track_id},{bbox[0]},{bbox[1]},{bbox[2]-bbox[0]},{bbox[3]-bbox[1]},-1,-1,{conf}\n')
# Iterate over every file in the directory




# Directory containing the text files

# Output file where all contents will be combined
output_file = 'GT_Combined.txt'

# List all files in the directory
files = [f for f in os.listdir(roboflow_label_directory) if f.endswith('.txt')]

# Sort files by name (optional)
files.sort()

# Open the output file
with open(output_file, 'w') as outfile:
    # Process each file
    for i, filename in enumerate(files, start=1):
        # Create the full path to the file
        filepath = os.path.join(roboflow_label_directory, filename)
        # Open the text file
        with open(filepath, 'r') as infile:
            # Read the content of the file
            content = infile.read()
            content = [f"{i},"+",".join(x.split(" ")) for x in  content.split("\n")]
            content = "\n".join(content)
            # Write the content to the output file, including the file number
            outfile.write(f"{content}\n")
            # break

print("Files have been combined into", output_file)
def read_data(file_path):
    df = pd.read_csv(file_path, header=None)
    df = df.iloc[:,:6]
    df.columns = ['frame', 'id', 'x_min', 'y_min', 'x_max', 'y_max']
    df['bbox'] = df.apply(lambda row: (row['x_min'], row['y_min'], row['x_max'] - row['x_min'], row['y_max'] - row['y_min']), axis=1)

    return df[['frame', 'id', 'bbox']]

gt_data = read_data('GT_Combined.txt')
tracker_data = read_data(f'tracker_results_{model_name}.txt')

def calculate_iou_shapely(box_1, box2):
    # Shapely expects (minx, miny, maxx, maxy)
    from shapely.geometry import box

    poly1 = box(*box_1)
    poly2 = box(*box2)
    intersection = poly1.intersection(poly2).area
    union = poly1.union(poly2).area
    return intersection / union if union != 0 else 0



acc = mm.MOTAccumulator(auto_id=True)

frames = sorted(set(gt_data['frame'].unique()).union(set(tracker_data['frame'].unique())))

for frame in frames:
    g = gt_data[gt_data['frame'] == frame]
    t = tracker_data[tracker_data['frame'] == frame]

    gt_ids = g['id'].values
    tr_ids = t['id'].values
    gt_boxes = g['bbox'].values
    tr_boxes = t['bbox'].values

    # Create an empty distance matrix
    distances = np.full((len(gt_ids), len(tr_ids)), np.inf)
    # Fill the distance matrix using the custom IoU function
    for i, gt_box in enumerate(gt_boxes):
        for j, tr_box in enumerate(tr_boxes):
            iou = calculate_iou_shapely(gt_box, tr_box)
            if iou > 0.5:  # Assuming IoU > 0 means there is an overlap
                distances[i, j] = 1 - iou  # Convert IoU to cost for motmetrics

    acc.update(
        gt_ids,  # Ground truth objects in this frame
        tr_ids,  # Tracker objects in this frame
        distances  # The distance matrix
    )

mh = mm.metrics.create()
summary = mh.compute(acc, metrics=['num_frames', 'idf1', 'idp', 'idr', \
                                     'recall', 'precision', 'num_objects', \
                                     'mostly_tracked', 'partially_tracked', \
                                     'mostly_lost', 'num_false_positives', \
                                     'num_misses', 'num_switches', \
                                     'num_fragmentations', 'mota', 'motp', "id_global_assignment","obj_frequencies" \
                                    ], \
                      name='acc')
print(summary)
summary.to_csv(f"MOTA_evaluation_{model_name}.csv")
