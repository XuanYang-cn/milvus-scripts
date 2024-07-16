""" python load_data.py -c test1 """

import concurrent
import threading
import argparse
import time
from multiprocessing import get_context
import multiprocessing as mp

import numpy as np
from pymilvus import (
    Collection,
    CollectionSchema,
    #  connections,
    DataType,
    FieldSchema,
    utility,
)

def delete(name: str, expr: str):
    from pymilvus import connections
    connections.connect()
    c = Collection(name)

    print(f"delete {expr}")
    c.delete(expr)
    c.flush()

    c.delete(expr)
    c.flush()

    c.delete(expr)
    c.flush


def prepare_collection(name: str, dim: int, recreate_if_exist: bool=False, schema: CollectionSchema=None, **kwargs):
    from pymilvus import connections
    connections.connect(**kwargs)

    def create():
        fields = [
            FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True),
            FieldSchema(name="random", dtype=DataType.DOUBLE),
            FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=dim)
        ]

        local_schema = CollectionSchema(fields) if schema is None else schema
        Collection(name, local_schema, **kwargs)

    if not utility.has_collection(name):
        create()

    elif recreate_if_exist is True:
        utility.drop_collection(name)
        create()
    connections.disconnect("default")

class MilvusMultiThreadingInsert:
    def __init__(self, collection_name: str, total_count: int, num_per_batch: int, dim: int):

        batch_count = int(total_count / num_per_batch)

        self.thread_local = threading.local()
        self.collection_name = collection_name
        self.dim = dim
        self.total_count = total_count
        self.num_per_batch = num_per_batch
        self.batchs = list(range(batch_count))

    def connect(self, uri: str):
        from pymilvus import connections
        connections.connect(uri=uri)

    def get_thread_local_collection(self):
        if not hasattr(self.thread_local, "collection"):
            self.thread_local.collection = Collection(self.collection_name)
        return self.thread_local.collection

    def insert_work(self, number: int):
        print(f"No.{number:2}: Start inserting entities")
        rng = np.random.default_rng(seed=number)
        entities = [
            list(range(self.num_per_batch*number, self.num_per_batch*(number+1))),
            rng.random(self.num_per_batch).tolist(),
            rng.random((self.num_per_batch, self.dim)),
        ]

        insert_result = self.get_thread_local_collection().insert(entities)
        assert len(insert_result.primary_keys) == self.num_per_batch
        print(f"No.{number:2}: Finish inserting entities")

    def _insert_all_batches(self):
        with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:
            executor.map(self.insert_work, self.batchs)

    def run(self):
        start_time = time.time()
        self._insert_all_batches()
        duration = time.time() - start_time
        print(f'Inserted {len(self.batchs)} batches of entities in {duration} seconds')
        self.get_thread_local_collection().flush()
        print(f"Inserted num_entities: {self.total_count}. \
                Actual num_entites: {self.get_thread_local_collection().num_entities}")

class MilvusUploader:
    client = None
    upload_params = {}
    collection: Collection = None
    distance: str = None

    @classmethod
    def get_mp_start_method(cls):
        return "spawn"
        #  return "forkserver" if "forkserver" in mp.get_all_start_methods() else "spawn"

    @classmethod
    def init_client(cls, kwargs:dict):
        from pymilvus import connections
        cls.client = connections.connect(**kwargs)
        cls.collection = Collection("bench")
        print("connected")

    @classmethod
    def upload_batch(cls, number: int):
        rng = np.random.default_rng(seed=number)
        num_per_batch = 5000
        entities = [
            list(range(num_per_batch*number, num_per_batch*(number+1))),
            rng.random(num_per_batch).tolist(),
            rng.random((num_per_batch, 768)),
        ]

        print(f"No.{number:2}: Start inserting entities")
        try:
            ret = cls.collection.insert(entities)
        except exception as e:
            print("error", e)

        print(f"Inserted {ret.insert_count} records")


class MilvusMultiProcessing(MilvusUploader):
    def __init__(self, **connection_params):
        self.connection_params = connection_params

    def upload(self):
        self.init_client(self.connection_params)
        print("connected")

        ctx = get_context(self.__class__.get_mp_start_method())
        with ctx.Pool(
            processes=1,
            initializer=self.__class__.init_client,
            initargs=(self.connection_params,),
        ) as pool:
            for res in pool.imap(self.__class__._upload_batch, range(1000000//5000)):
                print("OK")


    @classmethod
    def _upload_batch(cls, number):
        return cls.upload_batch(number)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--collection", type=str, required=True, help="collection name")
    parser.add_argument("-d", "--dim", type=int, default=768, help="dimension of the vectors")
    parser.add_argument("-n", "--new", action="store_true", help="Whether to create a new collection or use the existing one")

    flags = parser.parse_args()
    #  TODO
