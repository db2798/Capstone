import click
import cv2
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.visualizer import Visualizer
from utils import get_labelme_dataset_function
import os


@click.command()
@click.argument('directory', nargs=1, default = "/scratch/subramav/blog/detectron2/jsons")
def main(directory):
    class_labels = ["rectangle"]
    dataset_name = "/scratch/subramav/blog/detectron2/data"
    dataset_function = get_labelme_dataset_function(directory, class_labels)
    MetadataCatalog.get(dataset_name).thing_classes = class_labels
    DatasetCatalog.register(dataset_name, dataset_function)
    metadata = MetadataCatalog.get(dataset_name)
    for d in dataset_function():
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=metadata, scale=1.0)
        out = visualizer.draw_dataset_dict(d)
        cv2.imshow("Ground truth", out.get_image()[:, :, ::-1])
        cv2.waitKey(1000)


if __name__ == "__main__":
    main()
