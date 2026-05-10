import os
import sys
import pickle
import click
import pandas as pd
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

@click.command()
@click.option('-d', '--tag_path', required=True)
@click.option('-c', '--trajectory_path', required=True)
def main(tag_path, trajectory_path):
    slam_results = pd.read_csv(trajectory_path)
    tag_detection_results = pickle.load(open(tag_path, 'rb'))
    
