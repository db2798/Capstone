import click
import cv2
import os

from utils import get_labelme_dataset_function

from detectron2 import model_zoo
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg


@click.command()
@click.argument('directory', nargs=1, default = "//scratch2/devashree/detectron2/Frames - Random")
@click.option('--weights', default="/scratch2/devashree/detectron2/model_0008999.pth", help='Path to the model to use')
def main(directory, weights):
    class_labels = ["dolphin"]
    dataset_name = "ship_dataset"

    dataset_function = get_labelme_dataset_function(directory, class_labels)
    MetadataCatalog.get("ship_dataset").thing_classes = class_labels
    DatasetCatalog.register("ship_dataset", dataset_function)
    metadata = MetadataCatalog.get("ship_dataset")

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(
        "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))  # Base model
    # Rotated bbox specific config in the same directory as this file
    cfg.merge_from_file(os.path.join(os.path.dirname(
        os.path.abspath(__file__)), "rotated_bbox_config.yaml"))
    cfg.DATASETS.TRAIN = (dataset_name,)
    cfg.DATASETS.TEST = (dataset_name,)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(class_labels)

    cfg.MODEL.WEIGHTS = weights
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.9  # Preiction confidence threshold,

    predictor = DefaultPredictor(cfg)

    for n,d in enumerate(dataset_function()):
        img = cv2.imread(d["file_name"])
        outputs = predictor(img)
        v = Visualizer(img[:, :, ::-1], metadata, scale=1.0)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

        # cv2.imshow("Predictions", out.get_image()[:, :, ::-1])
        print(d["file_name"],outputs)
        cv2.imwrite(f"/scratch2/devashree/detectron2/predictions/cohort/{n}_predictions_val.png", out.get_image()[:, :, ::-1])
        # cv2.waitKey(1000)
        


if __name__ == "__main__":
    main()
