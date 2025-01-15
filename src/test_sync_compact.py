"""
python test_sync_compact.py -h

options:
  -h, --help            show this help message and exit
  --uri URI             uri to connect
  -c COLLECTION, --collection COLLECTION
                        collection name
  -d DELETE_PROPORTION, --delete_proportion DELETE_PROPORTION
                        delete proportion
  -n NUM_ROWS, --num_rows NUM_ROWS
                        total num rows inserted
"""

import argparse

from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, connections

from generate_segment import estimate_size_by_count, stream_insert

# local
from load_data import prepare_collection
from test_compact_n_segments import delete_n_percent


def load_by_count_delete_n_per(
    name: str,
    count: int = 10_000_000,
    num_partitions: int = 1024,
    delete_proportion: int = 20,
    **kwargs,
):
    dim = 768

    fields = [
        FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=256, is_partition_key=True),
        FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=dim),
    ]
    schema = CollectionSchema(fields)

    prepare_collection(name, dim, True, schema=schema, num_partitions=num_partitions, **kwargs)
    c = Collection(name)
    if not c.has_index():
        c.create_index("embeddings", {"index_type": "FLAT", "params": {"metric_type": "L2"}})

    actual_count, actual_deleted_count = 0, 0

    batch = 100
    size = estimate_size_by_count(count, schema)
    print(
        f"Try to load {count} num rows in dim={dim} in batch {batch}, size ~= {size / 1024 / 1024 / 1024:.2f}GB"
    )
    for i in range(batch):
        print(f"------------------------------ batch {i + 1} -------------------------------")
        batch_count = count // batch
        batch_size = estimate_size_by_count(batch_count, schema)
        batch_pks = stream_insert(c, schema, batch_size)
        batch_deleted = delete_n_percent(name, batch_pks, n=delete_proportion, flush=False)

        actual_count += sum(len(pks) for pks in batch_pks)
        actual_deleted_count += batch_deleted

    print("=============")
    print(f"Actual loaded {count} num rows data, deleted count {actual_deleted_count}")


def test_load_by_count_multiprocess(name: str, count: int = 10_000_000, **kwargs):
    dim = 768

    fields = [
        FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True),
        FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=dim),
    ]
    schema = CollectionSchema(fields)

    prepare_collection(name, dim, True, schema=schema, **kwargs)
    c = Collection(name)
    if not c.has_index():
        c.create_index("embeddings", {"index_type": "FLAT", "params": {"metric_type": "L2"}})

    actual_count, actual_deleted_count = 0, 0

    batch = 100
    size = estimate_size_by_count(count, schema)
    print(
        f"Try to load {count} num rows in dim={dim} in batch {batch}, size ~= {size / 1024 / 1024 / 1024:.2f}GB"
    )
    for i in range(batch):
        print(f"------------------------------ batch {i + 1} -------------------------------")
        batch_count = count // batch
        batch_size = estimate_size_by_count(batch_count, schema)
        batch_pks = stream_insert(c, schema, batch_size)
        batch_deleted = delete_n_percent(name, batch_pks, n=delete_proportion, flush=False)

        actual_count += sum(len(pks) for pks in batch_pks)
        actual_deleted_count += batch_deleted

    print("=============")
    print(f"Actual loaded {count} num rows data, deleted count {actual_deleted_count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--uri", type=str, default="http://localhost:19530", help="uri to connect")
    parser.add_argument(
        "-c", "--collection", type=str, default="test_sync_compact", help="collection name"
    )
    parser.add_argument(
        "-d", "--delete_proportion", type=int, default="20", help="delete proportion"
    )
    parser.add_argument(
        "-n", "--num_rows", type=int, default="10_000_000", help="total num rows inserted"
    )
    parser.add_argument(
        "-np", "--num_partitions", type=int, default="1024", help="total num partitions"
    )

    flags = parser.parse_args()

    connections.connect(uri=flags.uri)
    load_by_count_delete_n_per(
        name=flags.collection,
        count=flags.num_rows,
        num_partitions=flags.num_partitions,
        delete_proportion=flags.delete_proportion,
    )
