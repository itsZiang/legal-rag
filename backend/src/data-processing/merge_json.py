import json
import os
import shutil

# Function to read a single JSON file
def read_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

# Function to write JSON file
def write_json_file(file_path, data):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=2)

# Process data from a specific file in folder1 using data from folder2
def process_file_data(file1_data, folder2_data):
    result = []
    
    # Create a dictionary to store all items from folder2 by title
    folder2_by_title = {}
    for item in folder2_data:
        title = item['title']
        if title not in folder2_by_title:
            folder2_by_title[title] = []
        folder2_by_title[title].append(item)
    
    # Process items from file1
    for item in file1_data:
        title = item['title']
        
        # If title exists in folder2, add all items from folder2 with this title
        if title in folder2_by_title:
            result.extend(folder2_by_title[title])
            # We don't delete from folder2_by_title here because we process each file independently
        else:
            # If title doesn't exist in folder2, keep the original item
            result.append(item)
    
    return result

# Main function
def main():
    # Define folder paths
    folder1_path = "../output"  # Path to folder1
    folder2_path = "../output_llm_chunking_1024"  # Path to folder2
    output_folder_path = "../final_chunk"  # Path to output folder
    
    # Create output folder if it doesn't exist
    if os.path.exists(output_folder_path):
        shutil.rmtree(output_folder_path)  # Remove old output folder if exists
    os.makedirs(output_folder_path)
    
    # Read all data from folder2 (needed for each file in folder1)
    folder2_data = []
    for filename in os.listdir(folder2_path):
        if filename.endswith('.json'):
            file_path = os.path.join(folder2_path, filename)
            try:
                data = read_json_file(file_path)
                if isinstance(data, list):
                    folder2_data.extend(data)
                else:
                    folder2_data.append(data)
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
    
    # Process each file in folder1
    file_count = 0
    for filename in os.listdir(folder1_path):
        if filename.endswith('.json'):
            file_count += 1
            file_path = os.path.join(folder1_path, filename)
            try:
                # Read data from file1
                file1_data = read_json_file(file_path)
                
                # Process data
                if isinstance(file1_data, list):
                    output_data = process_file_data(file1_data, folder2_data)
                else:
                    output_data = process_file_data([file1_data], folder2_data)
                    output_data = output_data[0] if len(output_data) == 1 else output_data
                
                # Write output to the same filename in output folder
                output_file_path = os.path.join(output_folder_path, filename)
                write_json_file(output_file_path, output_data)
                print(f"Processed {filename} -> {output_file_path}")
                
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
    
    print(f"Processing complete! {file_count} files processed.")
    
if __name__ == "__main__":
    main()