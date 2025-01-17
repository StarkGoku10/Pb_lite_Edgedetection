import os

def base_folder(base_dir="results"):
    """
    Creates the base folder where all results will be stored.
    Parameters:
        base_dir (str): The base directory to create. Defaults to 'results'.
    """
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
        print(f"Created base directory: {base_dir}")
    else:
        print(f"Base directory already exists: {base_dir}")

    return base_dir

def filterbanks_folder(base_dir="results"):
    """
    Creates the 'filterbanks' folder inside the base directory.
    Parameters:
        base_dir (str): The base directory. Defaults to 'results'.
    """
    folder_path = os.path.join(base_dir, "filterbanks")
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Created folder: {folder_path}")
    else:
        print(f"Folder already exists: {folder_path}")

    return folder_path

def maps_folder(base_dir="results"):
    """
    Creates the 'maps' folder inside the base directory.
    Parameters:
        base_dir (str): The base directory. Defaults to 'results'.
    """
    folder_path = os.path.join(base_dir, "maps")
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Created folder: {folder_path}")
    else:
        print(f"Folder already exists: {folder_path}")

    return folder_path

def edges_folder(base_dir="results"):
    """
    Creates the 'edges' folder inside the base directory.
    Parameters:
        base_dir (str): The base directory. Defaults to 'results'.
    """
    folder_path = os.path.join(base_dir, "edges")
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Created folder: {folder_path}")
    else:
        print(f"Folder already exists: {folder_path}")

    for i in range(1, 11):
        folder_name = f"img{i}"
        img_folder_path = os.path.join(folder_path, folder_name)
        if not os.path.exists(img_folder_path):
            os.makedirs(img_folder_path)
            print(f"Created folder: {img_folder_path}")
        else:
            print(f"Folder already exists: {img_folder_path}")
    
    return folder_path

def create_all_folders(base_dir="results"):
    """
    Creates the entire folder structure for the project.
    
    Parameters:
        base_dir (str): The base directory. Defaults to 'results'.
    """
    base_folder(base_dir)
    filterbanks_folder(base_dir)
    maps_folder(base_dir)
    edges_folder(base_dir)
    print("All required folders have been created successfully!")

# Run the folder creation process
if __name__ == "__main__":
    create_all_folders()
