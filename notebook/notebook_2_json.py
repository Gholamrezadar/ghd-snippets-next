import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", type=str,
                    help="the input notebook to be converted to json", required=True)
args = parser.parse_args()
notebook_path = args.input

processed_data = "["

with open(notebook_path, "r") as f:
    json_file = json.load(f)
    idx = 0
    for cell in json_file["cells"]:
        if cell["cell_type"] == "code":
            idx += 1
            if cell["source"][0].startswith("##") and cell["source"][1].startswith("##") and cell["source"][2].startswith("##"):
                # Extract data from cell
                title = cell["source"][0][3:]
                subtitle = cell["source"][1][3:]
                tags = cell["source"][2][3:]
                content = cell["source"][3:]
                
                # Write in the json format
                processed_data += "{\n"
                processed_data += f"id: {idx},\n"
                processed_data += f"title: \"{title[:-1]}\",\n"
                processed_data += f"subtitle: \"{subtitle[:-1]}\",\n"
                processed_data += f"tags: {list(map(str.lstrip, tags[:-1].split(',')))},\n"
                processed_data += f"content: `{''.join(content)}`"
                processed_data += "},\n"

processed_data += "]"
with open("temp_json.json", "w+") as f:
    f.write(processed_data)