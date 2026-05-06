import os

def remove_dot_underscore_files(root_dir):
    count = 0

    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.startswith("._"):
                file_path = os.path.join(dirpath, filename)
                try:
                    os.remove(file_path)
                    print(f"Deleted: {file_path}")
                    count += 1
                except Exception as e:
                    print(f"Failed to delete {file_path}: {e}")

    print(f"\nTotal deleted: {count}")

if __name__ == "__main__":
    root_dir = "/home/abc/yg/LHM_Track/bs2flame_mapping/LAM_data/export"  # 改成你的路径
    remove_dot_underscore_files(root_dir)