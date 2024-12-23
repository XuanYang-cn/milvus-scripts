from enum import Enum

import pymilvus
from pymilvus import DataType


class Unit(str, Enum):
    B = "Bytes"
    KB = "Kilobytes"
    MB = "Megabytes"
    GB = "Gigabytes"


def estimate_count_by_size(size: int, schema: pymilvus.CollectionSchema) -> int:
    size_per_row = 0
    for fs in schema.fields:
        if fs.dtype == DataType.INT64:
            size_per_row += 8
        elif fs.dtype == DataType.VARCHAR:
            size_per_row += fs.max_length
        elif fs.dtype == DataType.FLOAT_VECTOR:
            size_per_row += fs.dim * 4
        elif fs.dtype == DataType.DOUBLE:
            size_per_row += 8
        else:
            msg = f"Unsupported data type: {fs.dtype.name}, please impl in generate_segment.py yourself"
            raise ValueError(msg)

    return int(size / size_per_row)


def estimate_size_by_count(count: int, schema: pymilvus.CollectionSchema) -> int:
    size_per_row = 0
    for fs in schema.fields:
        if fs.dtype == DataType.INT64:
            size_per_row += 8
        elif fs.dtype == DataType.VARCHAR:
            size_per_row += fs.max_length
        elif fs.dtype == DataType.FLOAT_VECTOR:
            size_per_row += fs.dim * 4
        elif fs.dtype == DataType.DOUBLE:
            size_per_row += 8
        else:
            msg = f"Unsupported data type: {fs.dtype.name}, please impl in generate_segment.py yourself"
            raise ValueError(msg)
    return int(count * size_per_row)
