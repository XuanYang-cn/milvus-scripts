from pathlib import Path

from pymilvus import Collection, connections

from generate_segment import SegmentDistribution, Size, Unit, generate_segments

# local
from load_data import prepare_collection


def generate_n_segments(name: str, n: int = 20):
    prepare_collection(name, 768, False)
    connections.connect()
    c = Collection(name)
    if not c.has_index():
        c.create_index("embeddings", {"index_type": "FLAT", "params": {"metric_type": "L2"}})

    ten_segs = [Size(123, Unit.MB) for i in range(n)]
    dist = SegmentDistribution(
        collection_name=name,
        size_dist=ten_segs,
    )
    return generate_segments(dist)


def delete_n_percent(name: str, all_pks: list[list] | None = None, n: int = 20, flush: bool = True):
    if n == 0:
        print("No deletion, return...")
        return 0

    import numpy as np

    c = Collection(name)
    c.load()
    delete_count = 0

    if not isinstance(all_pks, list):
        raise TypeError(f"pks should be a list, but got {type(all_pks)}")

    for i, pks in enumerate(all_pks):
        sample_pks = np.random.choice(pks, size=int(n * 0.01 * len(pks)), replace=False)
        expr = f"pk in {sample_pks.tolist()}"
        rt = c.delete(expr)
        delete_count += rt.delete_count

    print(
        f"PK count = {sum(len(pks) for pks in all_pks)}, Delete percent = {n}%, Delete count = {delete_count}"
    )
    if flush is True:
        print("Delete done and flush done")
        c.flush()
    return delete_count


def delete_n_percent_to_files(name: str, all_pks: list[list] = None, n: int = 20):
    import numpy as np

    c = Collection(name)
    c.load()

    if not isinstance(all_pks, list) or not isinstance(all_pks[0], list):
        raise TypeError(f"pks should be a list, but got {type(all_pks)}")

    for i, pks in enumerate(all_pks):
        sample_pks = np.random.choice(pks, size=int(n * 0.01 * len(pks)), replace=False)
        with Path(f"pks_{i}.txt").open("w") as f:
            for pk in sample_pks:
                f.write(f"{pk}\n")


def delete_all(name):
    c = Collection(name)
    c.load()
    ret = c.delete("pk > 1")
    c.flush()
    print(f"delete counts: {ret.delete_count}")


def delete_by_files(name: str):
    c = Collection(name)
    c.load()
    del_count = 0

    for i in range(20):
        with Path(f"pks_{i}.txt").open("r") as f:
            pks = [int(line.strip()) for line in f.readlines()]
        expr = f"pk in {pks}"
        ret = c.delete(expr)
        print(f"sampled pk counts: {len(pks)} and delete done")
        del_count += ret.delete_count
        c.flush()

    print(f"delete counts: {del_count}")


def test_case_generate_20_segments_del_20perc():
    name = "test_l0_compact_20_seg"
    pks = generate_n_segments(name, 20)
    print(f"generated 20 segments, total row counts: {len(pks)}")
    delete_n_percent(name, pks, 20)


def test_case_generate_20_segments_del_20perc_to_files():
    name = "test_l0_compact_20_seg"
    pks = generate_n_segments(name, 20)
    delete_n_percent_to_files(name, pks, 20)


def test_case_generate_20_segments_del_all():
    name = "test_l0_compact_20_seg_clean_all"
    generate_n_segments(name, 20)
    delete_all(name)


def test_case_generate_20_segments_no_del():
    name = "test_l0_compact_20_seg"
    generate_n_segments(name, 20)


if __name__ == "__main__":
    connections.connect()
    #  test_case_generate_20_segments_no_del()
    test_case_generate_20_segments_del_all()
