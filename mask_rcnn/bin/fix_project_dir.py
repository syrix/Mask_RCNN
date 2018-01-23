import os
import sys

def fix_project_dir():
    project_dir = os.getcwd()
    if project_dir.endswith('/mask_rcnn/bin'):
        project_dir = os.path.split(os.path.split(os.getcwd())[0])[0]
        os.chdir(project_dir)
        if project_dir not in sys.path:
            sys.path.append(project_dir)
        print(f'INFO: switched to project_dir: {project_dir}')