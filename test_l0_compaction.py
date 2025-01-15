"""Giving a rate and target segment size, generating segment forever"""

import logging

import pymilvus
from pydantic import BaseModel, ConfigDict
from pymilvus import DataType, MilvusClient

from src.common_func import Unit
from src.generate_segment import generate_segment_by_size
from src.segment_distribution import Size

logger = logging.getLogger("pymilvus")
logger.setLevel(logging.INFO)


class TestCompactionOrder(BaseModel):
    """Test case:
    Pre: [Server] change trigger L0, when meet 2 L0, select the later one
    1. [Server]: disable auto compaction
        - Generate 1 sealed segment
        - Delete 50% of the data, generate 1 L0 segment
        - Delete 30% of the data, generate 1 L0 segment
        - assert Count(*) == 20% * num_rows
    2. [Server]
        - dataCoord.compaction.levelzero.triggerInterval = 1
        - dataCoord.compaction.mix.triggerInterval = 10000
        - enable auto compaction
        - wait for L0 compaction done
        - disable auto compaction
    3. release load
        - assert count(*) == 20% * num_rows
    """

    collection_name: str = "test_compaction_order"
    connection_config: dict = {"uri": "http://localhost:19530"}
    cschema: pymilvus.CollectionSchema | None = None
    pks: list[int] = []

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.connection_config = kwargs

    def run(self, size: Size, drop_old: bool = True):
        self.prep_collection(drop_old)
        self.pks.extend(self.load_one_segment(size))

        pct50, pct80 = int(len(self.pks) * 0.5), int(len(self.pks) * 0.8)
        first_del, second_del = self.pks[:pct50], self.pks[pct50:pct80]
        self.delete_by_pk(first_del)
        self.delete_by_pk(second_del)
        logger.info("Finish deletes, start tests")

        c = MilvusClient(**self.connection_config)
        left_count = int(0.2 * len(self.pks)) if len(self.pks) > 0 else int(201640 * 0.2)
        while True:
            got = input("Do you want to continue? (y/n)")
            if got == "n":
                return

            c.release_collection(self.collection_name)
            c.load_collection(self.collection_name)
            count = c.query(self.collection_name, output_fields=["count(*)"])[0]["count(*)"]
            logger.info(f"query count ={count}, left_count = {left_count}")
            assert count == left_count

    def load_one_segment(self, size: Size) -> list[int | str]:
        c = MilvusClient(**self.connection_config)
        pks = []
        for data in generate_segment_by_size(size.as_bytes(), self.cschema):
            logger.info(f"Inserting {len(data)} rows")
            pks.extend(c.insert(self.collection_name, data).get("ids"))
        logger.info(f"Flush {self.collection_name}")
        c.flush(self.collection_name)
        return pks

    def delete_by_pk(self, pks: list[int | str]):
        c = MilvusClient(**self.connection_config)
        expr = f"id in {pks}"

        count = c.delete(self.collection_name, filter=expr)["delete_count"]
        logger.info(f"Delete count {count}")
        c.flush(self.collection_name)

    def prep_collection(self, drp_old: bool):
        c = MilvusClient(**self.connection_config)
        if c.has_collection(self.collection_name):
            c.drop_collection(self.collection_name)

        schema = c.create_schema(auto_id=False)

        schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
        schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=128)
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
    runner = TestCompactionOrder()
    size = Size(count=100, unit=Unit.MB)
    runner.run(size, False)
