"""ImageNet input pipeline."""
import math
import numpy as np
from functools import partial
from typing import Tuple

import tensorflow as tf
import tensorflow_datasets as tfds
tf.config.set_visible_devices([], 'GPU')

import jax


def _distorted_bounding_box_crop( # pylint: disable=too-many-arguments
        image, bbox, min_object_covered,
        aspect_ratio_range, area_range, max_attempts):
    shape = tf.shape(image)
    sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(
        shape, bounding_boxes=bbox, min_object_covered=min_object_covered,
        aspect_ratio_range=aspect_ratio_range, area_range=area_range,
        max_attempts=max_attempts, use_image_if_no_bounding_boxes=True)
    bbox_begin, bbox_size, _ = sample_distorted_bounding_box
    image = tf.slice(image, bbox_begin, bbox_size)
    return image


def _at_least_x_are_equal(a, b, x):
    match = tf.equal(a, b)
    match = tf.cast(match, tf.int32)
    return tf.greater_equal(tf.reduce_sum(match), x)


def _resize(image, image_size):
    return tf.image.resize(
        [image], [image_size, image_size],
        method=tf.image.ResizeMethod.BICUBIC)[0]


def _random_flip(image):
    return tf.image.random_flip_left_right(image)


def _random_crop(image, image_size):
    original_shape = tf.shape(image)
    bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])
    cropped_image = _distorted_bounding_box_crop(
        image, bbox, min_object_covered=0.1,
        aspect_ratio_range=(3.0/4.0, 4.0/3.0), area_range=(0.08, 1.0),
        max_attempts=10)
    bad = _at_least_x_are_equal(original_shape, tf.shape(cropped_image), 3)
    return tf.cond(
        bad,
        lambda: _center_crop(image, image_size),
        lambda: _resize(cropped_image, image_size))


def _center_crop(image, image_size):
    shape = tf.shape(image)
    image_h = shape[0]
    image_w = shape[1]
    padded_center_crop_size = tf.cast((
        (image_size / (image_size + 32)) *
            tf.cast(tf.minimum(image_h, image_w), tf.float32)), tf.int32)
    offset_h = ((image_h - padded_center_crop_size) + 1) // 2
    offset_w = ((image_w - padded_center_crop_size) + 1) // 2
    bbox_begin = tf.stack([
        offset_h,
        offset_w,
        tf.constant(0, dtype=tf.int32)])
    bbox_size = tf.stack([
        padded_center_crop_size,
        padded_center_crop_size,
        tf.constant(-1, dtype=tf.int32)])
    cropped_image = tf.slice(image, bbox_begin, bbox_size)
    return _resize(cropped_image, image_size)


def _prepare_tf_data(batch, batch_size, shard_shape):
    batch['images'] = batch['images'].numpy()
    batch['labels'] = batch['labels'].numpy()
    if batch['images'].shape[0] < batch_size:
        batch['images'] = np.concatenate([
            batch['images'], np.zeros([
                batch_size - batch['images'].shape[0],
                *batch['images'].shape[1:]
            ], batch['images'].dtype)])
        batch['labels'] = np.concatenate([
            batch['labels'], np.zeros([
                batch_size - batch['labels'].shape[0],
                *batch['labels'].shape[1:]
            ], batch['labels'].dtype)])
    def _prepare(x):
        return x.reshape(shard_shape + x.shape[1:])
    return jax.tree_util.tree_map(_prepare, batch)


def create_trn_iter( # pylint: disable=too-many-arguments
        data_builder: tfds.core.DatasetBuilder,
        batch_size: int,
        shard_shape: Tuple[int],
        split: str = 'train',
        dtype: tf.DType = tf.float32,
        image_size: int = 224,
        cache: bool = True,
    ):
    """
    Create a generator that produces sharded mini-batches, ensuring consistent
    sizes by setting `drop_remainder` to `True`.
    """
    image_decoder = data_builder.info.features['image'].decode_example

    def decode_example(example):
        image = image_decoder(example['image'])
        image = _random_crop(image, image_size)
        image = _random_flip(image)
        image = tf.reshape(image, [image_size, image_size, 3])
        image = tf.cast(image, dtype=dtype)
        _dict = {'images': image, 'labels': example['label']}
        return _dict

    data = data_builder.as_dataset(
        split=split, shuffle_files=True,
        decoders={'image': tfds.decode.SkipDecoding()})
    if cache:
        data = data.cache()

    data = data.repeat()
    data = data.shuffle(
        min(16*batch_size, data_builder.info.splits[split].num_examples))
    data = data.map(decode_example, num_parallel_calls=tf.data.AUTOTUNE)
    data = data.batch(batch_size, drop_remainder=True)
    data = data.prefetch(tf.data.AUTOTUNE)

    data = map(partial(
        _prepare_tf_data,
        batch_size=batch_size,
        shard_shape=shard_shape), data)
    dataset_size = data_builder.info.splits[split].num_examples
    steps_per_epoch = math.ceil(dataset_size / batch_size)

    return data, dataset_size, steps_per_epoch


def create_val_iter( # pylint: disable=too-many-arguments
        data_builder: tfds.core.DatasetBuilder,
        batch_size: int,
        shard_shape: Tuple[int],
        split: str = 'validation',
        dtype: tf.DType = tf.float32,
        image_size: int = 224,
        cache: bool = True,
    ):
    """
    Create a sharded mini-batch generator where all data points are processed
    by setting `drop_remainder` to `False`. Note that the last mini-batch is
    padded with zeros in `_prepare_tf_data` to maintain a consistent size.
    """
    image_decoder = data_builder.info.features['image'].decode_example

    def decode_example(example):
        image = image_decoder(example['image'])
        image = _center_crop(image, image_size)
        image = tf.reshape(image, [image_size, image_size, 3])
        image = tf.cast(image, dtype=dtype)
        _dict = {'images': image, 'labels': example['label']}
        return _dict

    data = data_builder.as_dataset(
        split=split, shuffle_files=False,
        decoders={'image': tfds.decode.SkipDecoding()})
    if cache:
        data = data.cache()

    data = data.map(decode_example, num_parallel_calls=tf.data.AUTOTUNE)
    data = data.batch(batch_size, drop_remainder=False)
    data = data.repeat()
    data = data.prefetch(tf.data.AUTOTUNE)

    data = map(partial(
        _prepare_tf_data,
        batch_size=batch_size,
        shard_shape=shard_shape), data)
    dataset_size = data_builder.info.splits[split].num_examples
    steps_per_epoch = math.ceil(dataset_size / batch_size)

    return data, dataset_size, steps_per_epoch
