import click
import numpy as np
import os
import torch
from utils import get_labelme_dataset_function
from detectron2 import model_zoo
from detectron2.data import DatasetCatalog, detection_utils as utils, transforms as T, build_detection_train_loader
from detectron2.engine import launch, DefaultTrainer, hooks
from detectron2.structures import BoxMode
from detectron2.evaluation import RotatedCOCOEvaluator, DatasetEvaluators
from detectron2.config import get_cfg
import wandb

def rotate_bbox(annotation, transforms):
    annotation["bbox"] = transforms.apply_rotated_box(np.asarray([annotation['bbox']]))[0]
    annotation["bbox_mode"] = BoxMode.XYXY_ABS
    return annotation

def get_shape_augmentations():
    # Optional shape augmentations
    return [
        T.RandomFlip(),
        T.ResizeShortestEdge(short_edge_length=(640, 672, 704, 736, 768, 800), max_size=1333, sample_style='choice'),
        T.RandomFlip()
    ]

def get_color_augmentations():
    # Optional color augmentations
    return T.AugmentationList([
        T.RandomBrightness(0.9, 1.1),
        T.RandomSaturation(intensity_min=0.75, intensity_max=1.25),
        T.RandomContrast(intensity_min=0.76, intensity_max=1.25)
    ])

def dataset_mapper(dataset_dict):
    image = utils.read_image(dataset_dict["file_name"], format="BGR")
    color_aug_input = T.AugInput(image)
    get_color_augmentations()(color_aug_input)
    image = color_aug_input.image
    image, image_transforms = T.apply_transform_gens(get_shape_augmentations(), image)
    dataset_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))

    annotations = [
        rotate_bbox(obj, image_transforms)
        for obj in dataset_dict.pop("annotations")
        if obj.get("iscrowd", 0) == 0
    ]
    instances = utils.annotations_to_instances_rotated(annotations, image.shape[:2])
    dataset_dict["instances"] = utils.filter_empty_instances(instances)
    return dataset_dict

class RotatedBoundingBoxTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluators = [RotatedCOCOEvaluator(dataset_name, cfg, True, output_folder)]
        return DatasetEvaluators(evaluators)

    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, mapper=dataset_mapper)

    def build_hooks(self):
        hooks = super().build_hooks()
        hooks.append(WandbHook())
        return hooks

class WandbHook(hooks.HookBase):
    def after_step(self):
        # Ensure we're only logging if it's time to log
        if self.trainer.iter % self.trainer.cfg.SOLVER.LOG_PERIOD == 0:
            metrics = self.trainer.storage.latest()
            wandb.log(metrics)

def train_detectron(flags):
    wandb.init(project="detectron2_project", name="rotated_bounding_box_training", reinit=True)
    class_labels = ["rectangle"]
    dataset_function = get_labelme_dataset_function(flags["directory"], class_labels)

    dataset_name = "/scratch/subramav/blog/detectron2/data"
    train_name = "/scratch/subramav/blog/detectron2/data/train"
    test_name = "/scratch/subramav/blog/detectron2/data/test"
    train_dataset_function = get_labelme_dataset_function(train_name, class_labels)
    test_dataset_function = get_labelme_dataset_function(test_name, class_labels)
    DatasetCatalog.register(dataset_name, dataset_function)
    DatasetCatalog.register(train_name, train_dataset_function)
    DatasetCatalog.register(test_name, test_dataset_function)

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))  # Base model
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")  # Weights
    cfg.merge_from_file(os.path.join(os.path.dirname(os.path.abspath(__file__)), "rotated_bbox_config.yaml"))
    cfg.DATASETS.TRAIN = (train_name,)
    cfg.DATASETS.TEST = (test_name,)
    cfg.OUTPUT_DIR = "."
    cfg.SOLVER.LOG_PERIOD = 20  # Adjust this value as needed
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(class_labels)

    trainer = RotatedBoundingBoxTrainer(cfg)
    trainer.register_hooks([WandbHook()])
    hooks = trainer.build_hooks()
    trainer.train()

@click.command()
@click.argument('directory', nargs=1, default = "/scratch/subramav/blog/detectron2/jsons")
@click.option('--num-gpus', default=0, help='Number of GPUs to use, default none')
def main(**flags):
    launch(
        train_detectron,
        flags["num_gpus"],
        args=(flags,),
    )

if __name__ == "__main__":
    main()
