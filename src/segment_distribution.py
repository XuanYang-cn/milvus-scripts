from pydantic import BaseModel

from .common_func import Unit


class Size(BaseModel):
    count: int
    unit: Unit = Unit.B

    def as_bytes(self) -> int:
        factor = 1
        if self.unit == Unit.B:
            factor = 1
        elif self.unit == Unit.KB:
            factor = 1024
        elif self.unit == Unit.MB:
            factor = 1024 * 1024
        elif self.unit == Unit.GB:
            factor = 1024 * 1024 * 1024
        return self.count * factor

    def __repr__(self):
        return f"{self.count} {self.unit}"

    __str__ = __repr__


class SegmentDistribution(BaseModel):
    collection_name: str
    size_dist: tuple[Size]
    partition_name: str = "_default"
