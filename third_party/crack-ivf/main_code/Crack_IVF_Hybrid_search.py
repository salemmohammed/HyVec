import numpy as np
from Crack_IVF_hybrid_search_version import CrackIVFHybridSearch


def main():
    # Create the index
    index = CrackIVFHybridSearch(
        nlist=2,
        default_nprobe=2,
        niter=1,
        max_pts=32,
        metric="euclidean",
        dbname="SIFT1M",
        nthreads=6
    )

    # ------------------------------------------------------------------
    # Create a tiny random dataset (100 vectors of dimension 8)
    # ------------------------------------------------------------------
    xb = np.random.rand(100, 8).astype(np.float32)

    # Build the index
    index.add(xb)

    # ------------------------------------------------------------------
    # Assign ONE attribute to every database vector.
    # Attribute values are integers in [0, 4].
    # ------------------------------------------------------------------
    attributes_dataset = np.random.randint(0, 3, size=len(xb)).tolist()

    index.populate_attribute_list(attributes_dataset)

    # ------------------------------------------------------------------
    # Create 5 random queries
    # ------------------------------------------------------------------
    xq = np.random.rand(5, 8).astype(np.float32)

    # Assign ONE attribute to every query
    attributes_queries = np.random.randint(0, 3, size=len(xq)).tolist()

    # print("Database attributes:")
    # print(attributes_dataset)

    # print("\nQuery attributes:")
    # print(attributes_queries)

    # ------------------------------------------------------------------
    # Search using vector + attribute predicate
    # ------------------------------------------------------------------
    D, I, timings = index.search(
        xq,
        attributes_queries,
        k=10
    )

    # print("\nDistances:")
    # print(D)

    # print("\nIndices:")
    # print(I)

    # # Print attributes of returned neighbors
    # print("\nReturned attributes:")
    
    
    # for qid in range(len(xq)):
    #     print(f"\nQuery {qid} (attribute={attributes_queries[qid]}):")
    #     for rank, idx in enumerate(I[qid]):
    #         if idx == -1:
    #             print(f"  Rank {rank}: FILTERED")
    #         else:
    #             print(
    #                 f"  Rank {rank}: "
    #                 f"Index={idx}, "
    #                 f"Distance={D[qid, rank]:.4f}, "
    #                 f"Attribute={attributes_dataset[idx]}"
    #             )

    # print("\nTimings:")
    # print(timings)


if __name__ == "__main__":
    main()