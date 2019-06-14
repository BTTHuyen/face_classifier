import os
import shutil

import glob

file_name = os.listdir("data")
path = glob.glob("data/*.jpg")
print(file_name)

for i, name in zip(path, file_name):
    try:
        # Create target Directory
        os.mkdir(name[:3])
        print("Directory ", name[:3], " Created ")
    except FileExistsError:
        print("Directory ", name[:3], " already exists")
    print(i)
    shutil.copy(i, name[:3]+"/"+name)
