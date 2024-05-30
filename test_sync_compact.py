from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType
from  pathlib import Path
import argparse

# local
from load_data import prepare_collection
from generate_segment import stream_insert, SegmentDistribution, Unit, estimate_size_by_count


def load_by_count(name: str, count: int = 10_000_000):
    dim = 768

    fields = [
        FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=256, is_partition_key=True),
        FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=dim)
    ]
    schema = CollectionSchema(fields)

    prepare_collection(name, dim, False, schema=schema)
    c = Collection(name)
    if not c.has_index():
        c.create_index("embeddings", {"index_type": "FLAT", "params": {"metric_type": "L2"}})

    total_size = estimate_size_by_count(count, schema)
    print(f"Try to load {count} num rows data in dim={dim}, approximate size = {total_size / 1024 / 1024}MB")

    return stream_insert(c, schema, total_size)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--uri", type=str, default="http://localhost:19530", help="uri to connect")
    parser.add_argument("-c", "--collection", type=str, default="test_sync_compact", help="collection name")

    flags = parser.parse_args()

    connections.connect(uri=flags.uri)
    load_by_count(flags.collection)

