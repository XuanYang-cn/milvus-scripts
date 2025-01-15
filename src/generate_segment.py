"""Give SegmentDistribution(segment size, collection, and segments count), generate segments.

Notes:
    Before running the scripts, The server MUST have:
        1. started
        2. created collection with zero num_rows

    If the segment size is larger than 100MB, its recommended to:
        1. set the server config: dataCoord.segment.sealProportion = 1
"""

import logging
import math
from typing import Union

import pymilvus
from pymilvus import Collection, Partition, connections, utility
from tqdm import tqdm

from .common_func import estimate_count_by_size
from .data_utils import gen_coloumn_data, gen_rows
from .segment_distribution import SegmentDistribution

logger = logging.getLogger("pymilvus")
logger.setLevel(logging.INFO)


def generate_segment_by_size(
    size: int, schema: pymilvus.CollectionSchema, partition_key: int | None = None
) -> list[dict]:
    max_size = 5 * 1024 * 1024  # 5MB
    total_count = 0

    max_count = estimate_count_by_size(max_size, schema)
    if size > max_size:
        batch = size // max_size
        tail = size - batch * max_size

        for _ in range(batch):
            data = gen_rows(schema, max_count, total_count, partition_key)
            total_count += max_count
            yield data
    else:
        tail = size
    if tail > 0:
        count = estimate_count_by_size(tail, schema)
        data = gen_rows(schema, count, total_count, partition_key)
        yield data


# TODO: remove
def generate_segments(dist: SegmentDistribution) -> list[int | str]:
    if not utility.has_collection(dist.collection_name):
        msg = f"Collection {dist.collection_name} does not exist"
        raise ValueError(msg)

    c = Collection(dist.collection_name)
    p = c.partition(dist.partition_name)

    pks = []
    for size in dist.size_dist:
        pks.append(generate_one_segment(p, c.schema, dist.as_bytes(size)))

    return pks


def generate_one_segment(
    c: Union[Collection, Partition], schema: pymilvus.CollectionSchema, size: int
) -> list:
    max_size = 5 * 1024 * 1024
    total_count = 0
    pks = []

    if size > max_size:
        batch = size // (5 * 1024 * 1024)
        tail = size - batch * max_size

        for i in range(batch):
            count = estimate_count_by_size(max_size, schema)
            data = gen_coloumn_data(schema, count)
            rt = c.insert(data)
            logger.info(
                f"inserted {max_size * (i + 1)}/{size}Bytes entities in batch 5MB, nun rows: {count}"
            )
            pks.extend(rt.primary_keys)
            total_count += count
    else:
        tail = size

    if tail > 0:
        count = estimate_count_by_size(tail, schema)
        data = gen_coloumn_data(schema, count)
        c.insert(data)
        logger.info(
            f"inserted entities size: {tail}Bytes, {tail / 1024 / 1024}MB, nun rows: {count}"
        )
        pks.extend(rt.primary_keys)
        total_count += count

    c.flush()
    logger.info(
        f"One segment num rows: {c.num_entities}, size: {size}Bytes, {size / 1024 / 1024}MB"
    )
    return pks


def stream_insert(
    c: Union[Collection, Partition], schema: pymilvus.CollectionSchema, size: int
) -> list[list]:
    max_size = 5 * 1024 * 1024
    total_count = 0
    pks = []

    logger.info(f"Try to load {size / 1024 / 1024:.2f}MB data in batch 5MB")
    if size > max_size:
        batch = math.ceil(size / (5 * 1024 * 1024))
        tail = size - (batch - 1) * max_size
    else:
        batch = 1
        tail = size

    for i in tqdm(range(batch)):
        batch_size = max_size if i < batch - 1 else tail

        count = estimate_count_by_size(batch_size, schema)
        data = gen_coloumn_data(schema, count)
        rt = c.insert(data)
        pks.append(rt.primary_keys)
        total_count += count

    logger.info(f"Loaded num rows: {total_count}, size: {size:.2f}B, {size / 1024 / 1024:.2f}MB")
    return pks


if __name__ == "__main__":
    dist = [32 * 1024 * 1024, 32 * 1024 * 1024]
    connections.connect()
    pks = generate_segments(
        SegmentDistribution(collection_name="test1", size_dist=(16 * 1024 * 1024, 32 * 1024 * 1024))
    )
