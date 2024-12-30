"""Giving a rate and target segment size, generating segment forever"""

import logging

import pymilvus
from pydantic import BaseModel, ConfigDict
from pymilvus import DataType, MilvusClient

from common_func import Unit
from generate_segment import generate_segment_by_partition_key
from segment_distribution import Size

logger = logging.getLogger("pymilvus")
logger.setLevel(logging.INFO)


class BuildRowsByRate(BaseModel):
    collection_name: str = "test_segment_rate_coll"
    partition_key: bool = True
    num_partitions: int = 16
    connection_config: dict = {"uri": "http://localhost:19530"}
    cschema: pymilvus.CollectionSchema | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.connection_config = kwargs

    def run(self, size: Size, drop_old: bool = True):
        self.prep_collection(drop_old)
        self.insert_work(size)

    def load_one_segment_for_all_partitionkey(self, size: Size):
        c = MilvusClient(**self.connection_config)
        for part_key_id in range(self.num_partitions):
            for data in generate_segment_by_partition_key(
                size.as_bytes(), self.cschema, part_key_id
            ):
                logger.info(f"Inserting {len(data)} rows for partition_key={part_key_id}")
                c.insert(self.collection_name, data)
            logger.info(f"Flush {self.collection_name} for partition_key={part_key_id}")
            c.flush(self.collection_name)

    def insert_work(self, size: Size):
        self.load_one_segment_for_all_partitionkey(size)
        self.load_one_segment_for_all_partitionkey(size)

    def delete_by_partition_key(self, partition_key: int):
        c = MilvusClient(**self.connection_config)
        expr = f"session_id == {partition_key}"

        logger.info(f"Deleting {expr}")
        c.delete(self.collection_name, filter=expr)
        c.flush(self.collection_name)

    def prep_collection(self, drp_old: bool):
        c = MilvusClient(**self.connection_config)
        if c.has_collection(self.collection_name):
            c.drop_collection(self.collection_name)

        schema = c.create_schema(
            auto_id=False,
            enable_dynamic_field=True,
            partition_key_field="session_id",
            num_partitions=self.num_partitions,
        )

        schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
        schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=128)
        schema.add_field(field_name="session_id", datatype=DataType.INT64)
        self.cschema = schema

        index_params = MilvusClient.prepare_index_params()
        index_params.add_index(field_name="id", index_type="STL_SORT")

        index_params.add_index(
            field_name="vector", index_type="IVF_FLAT", metric_type="L2", params={"nlist": 1024}
        )
        c.create_collection(
            collection_name=self.collection_name, schema=schema, index_params=index_params
        )


if __name__ == "__main__":
    runner = BuildRowsByRate()
    size = Size(count=100, unit=Unit.MB)
    runner.run(size, False)
    #  runner.delete_by_partition_key(1)
