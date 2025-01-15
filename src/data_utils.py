import logging
import uuid

import numpy as np
import pymilvus
from pymilvus import DataType

logger = logging.getLogger("pymilvus")
logger.setLevel(logging.INFO)

pre_sur = "{} Vector databases are specialized systems designed for managing and retrieving unstructured data through vector embeddings and numerical representations that capture the essence of data items like images, audio, videos"


def gen_coloumn_data(schema: pymilvus.CollectionSchema, count: int) -> list[list]:
    rng = np.random.default_rng()
    data = []
    for fs in schema.fields:
        if fs.dtype == DataType.INT64:
            if fs.is_primary and not fs.auto_id:
                data.append([uuid.uuid1().int >> 65 for _ in range(count)])
            else:
                data.append(list(range(count)))

        elif fs.dtype == DataType.VARCHAR:
            if fs.is_primary and not fs.auto_id:
                data.append([str(uuid.uuid4()) for _ in range(count)])
            else:
                data.append([pre_sur.format(str(uuid.uuid4())) for i in range(count)])

        elif fs.dtype == DataType.FLOAT_VECTOR:
            data.append(rng.random((count, fs.dim)))

        elif fs.dtype == DataType.DOUBLE:
            data.append(rng.random(count))

        else:
            msg = f"Unsupported data type: {fs.dtype.name}, please impl in generate_segment.py yourself"
            raise ValueError(msg)
    return data


def gen_one_row(
    schema: pymilvus.CollectionSchema, row_id: int, partition_key: int | None = None
) -> dict:
    rng = np.random.default_rng()
    data = {}
    for fs in schema.fields:
        if fs.dtype == DataType.INT64:
            if fs.is_primary and not fs.auto_id:
                data[fs.name] = uuid.uuid1().int >> 65
            if fs.is_partition_key:
                data[fs.name] = row_id if partition_key is None else partition_key

        elif fs.dtype == DataType.VARCHAR:
            if fs.is_primary and not fs.auto_id:
                data[fs.name] = str(uuid.uuid4())
            else:
                data[fs.name] = pre_sur.format(str(uuid.uuid4()))

        elif fs.dtype == DataType.FLOAT_VECTOR:
            data[fs.name] = rng.random((1, fs.dim))[0]

        elif fs.dtype == DataType.DOUBLE:
            data[fs.name] = rng.random(row_id)

        else:
            msg = f"Unsupported data type: {fs.dtype.name}, please impl in generate_segment.py yourself"
            raise ValueError(msg)
    return data


def gen_rows(
    schema: pymilvus.CollectionSchema, count: int, start_id: int, partition_key: int | None = None
) -> list[dict]:
    return [gen_one_row(schema, start_id + i, partition_key) for i in range(count)]
