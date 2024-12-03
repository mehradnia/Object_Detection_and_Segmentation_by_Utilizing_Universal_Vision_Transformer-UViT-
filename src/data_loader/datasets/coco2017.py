from pycocotools.coco import COCO
import os
import tensorflow as tf

from data_loader.data_loader import DataLoader


class COCO2017(DataLoader):

    def __init__(self, data_dir: str, batch_size: int, image_size: tuple) -> None:
        super().__init__(data_dir, batch_size, image_size)

    def _load_dataset(self, split):
        ann_file = os.path.join(
            f'data/coco2017/annotations/instances_{split}2017.json')
        coco = COCO(ann_file)
        image_ids = list(coco.imgs.keys())

        images, bboxes, labels = [], [], []
        for img_id in image_ids:
            img_info = coco.loadImgs(img_id)[0]
            img_path = os.path.join(
                self.data_dir, split, img_info['file_name'])

            image = tf.io.read_file(img_path)
            image = tf.image.decode_jpeg(image, channels=3)
            image = tf.image.resize(image, [self.image_size, self.image_size])

            ann_ids = coco.getAnnIds(imgIds=img_id)
            annotations = coco.loadAnns(ann_ids)
            img_bboxes, img_labels = [], []
            for ann in annotations:
                bbox = ann['bbox']
                category_id = ann['category_id']
                img_bboxes.append(bbox)
                img_labels.append(category_id)

            images.append(image)
            bboxes.append(img_bboxes)
            labels.append(img_labels)

        return tf.data.Dataset.from_tensor_slices((images, bboxes, labels)).batch(self.batch_size)
