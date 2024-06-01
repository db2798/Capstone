import json
import os

raw_jsons_directory = '/scratch2/devashree/yolov8/jsons'

for filename in os.listdir(raw_jsons_directory):
    # Check if the file is a JSON file
    if filename.endswith('.json'):
        filepath = os.path.join(raw_jsons_directory, filename)
        
        # Read the JSON data from the file
        with open(filepath, 'r') as file:
            data = json.load(file)
        
        # Modify the 'label' in each shape based on 'group_id'
        data["shapes"] = [json for json in data["shapes"] if len(json["points"])==4]
        for shape in data['shapes']:
            if shape['group_id']:
                shape['label'] = f"Dolphin_{shape['group_id']}"

        
        # Write the modified data back to the file
        with open(filepath, 'w') as file:
            json.dump(data, file, indent=4)