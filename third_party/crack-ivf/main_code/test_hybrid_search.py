import numpy as np
import pandas as pd

from Crack_IVF_hybrid_search_version import CrackIVFHybridSearch


def load_dataset(path):
    """
    Load vectors from a binary file formatted as:

        int32: number of vectors
        int32: vector dimension
        float32: vector values
    """
    with open(path, "rb") as f:
        header = np.fromfile(f, dtype=np.int32, count=2)

        if header.size != 2:
            raise ValueError(
                f"Could not read the dataset header from: {path}"
            )

        n, d = header # number and dimension of vectors
        vectors = np.fromfile(f, dtype=np.float32)

    n = int(n)
    d = int(d)
    expected = n * d

    if vectors.size != expected:
        raise ValueError(
            f"Expected {expected} float values, "
            f"but found {vectors.size}"
        )

    return vectors.reshape(n, d)


def load_queries(path):
    """
    Load query vectors and query attributes from a CSV file.

    Expected columns:
        embedding
        rand_int

    The file uses semicolons between columns and commas inside
    each embedding.
    """
    df = pd.read_csv(path, sep=";")

    required_columns = {"embedding", "rand_int"}

    if not required_columns.issubset(df.columns):
        raise ValueError(
            f"Query file must contain columns {required_columns}. "
            f"Found columns: {list(df.columns)}"
        )

    query_vectors = [
        np.fromstring(
            embedding,
            sep=",",
            dtype=np.float32
        )
        for embedding in df["embedding"]
    ]

    if not query_vectors:
        raise ValueError("No queries were found in the query file.")

    dimensions = {vector.size for vector in query_vectors}

    if len(dimensions) != 1:
        raise ValueError(
            f"Query embeddings have different dimensions: {dimensions}"
        )

    xq = np.stack(query_vectors)

    attributes_queries = df["rand_int"].to_numpy(
        dtype=np.int32
    )

    return xq, attributes_queries


def main():
    dataset_directory = (
        "/home/salemmohammed/Projects/research-projects/"
        "HyVec/data/hnsw/sift"
    )

    sift_path = f"{dataset_directory}/sift_LNG.bin"
    metadata_path = f"{dataset_directory}/meta_data.csv"
    queries_path = f"{dataset_directory}/sift_queries.csv"

    # ---------------------------------------------------------
    # 1. Load the SIFT database vectors
    # ---------------------------------------------------------
    print("Loading SIFT database...")

    xb = load_dataset(sift_path)

    print("Database shape:", xb.shape)
    print("Database dtype:", xb.dtype)

    # ---------------------------------------------------------
    # 2. Create the CrackIVF index
    # ---------------------------------------------------------
    index = CrackIVFHybridSearch(
        nlist=1000,
        default_nprobe=10,
        niter=10,
        max_pts=100000,
        metric="euclidean",
        dbname="SIFT1M",
        nthreads=6
    )

    # ---------------------------------------------------------
    # 3. Build the CrackIVF index
    # ---------------------------------------------------------
    print("Building CrackIVF index...")

    index.add(xb)

    print("CrackIVF index built successfully.")

    # ---------------------------------------------------------
    # 4. Load database metadata
    # ---------------------------------------------------------
    print("Loading database metadata...")

    attributes_dataset = np.loadtxt(
        metadata_path,
        dtype=np.int32
    )

    # Ensure the metadata is a one-dimensional array.
    attributes_dataset = np.asarray(
        attributes_dataset,
        dtype=np.int32
    ).reshape(-1)

    print("Metadata shape:", attributes_dataset.shape)
    print("First metadata values:", attributes_dataset[:5])

    if attributes_dataset.shape[0] != xb.shape[0]:
        raise ValueError(
            f"Database contains {xb.shape[0]} vectors, "
            f"but metadata contains "
            f"{attributes_dataset.shape[0]} values."
        )

    # Give the metadata to CrackIVF.
    index.populate_attribute_list(attributes_dataset)

    print("Metadata loaded successfully.")

    # ---------------------------------------------------------
    # 5. Load query vectors and query attributes
    # ---------------------------------------------------------
    print("Loading SIFT queries...")

    xq, attributes_queries = load_queries(queries_path)

    print("Query shape:", xq.shape)
    print("Query dtype:", xq.dtype)
    print("Query attributes shape:", attributes_queries.shape)
    print("First query attribute:", attributes_queries[0])

    if xq.shape[1] != xb.shape[1]:
        raise ValueError(
            f"Database dimension is {xb.shape[1]}, "
            f"but query dimension is {xq.shape[1]}."
        )

    if xq.shape[0] != attributes_queries.shape[0]:
        raise ValueError(
            f"There are {xq.shape[0]} query vectors, "
            f"but {attributes_queries.shape[0]} "
            f"query attributes."
        )

    # ---------------------------------------------------------
    # 6. Run a small hybrid-search test
    # ---------------------------------------------------------
    number_of_test_queries = 10
    k = 10

    xq_test = np.ascontiguousarray(
        xq[:number_of_test_queries],
        dtype=np.float32
    )

    attributes_queries_test = np.ascontiguousarray(
        attributes_queries[:number_of_test_queries],
        dtype=np.int32
    )

    print(
        f"Running CrackIVF hybrid search on "
        f"{number_of_test_queries} queries..."
    )

    distances, indices, timings = index.search(
        xq_test,
        attributes_queries_test,
        k=k
    )

    # ---------------------------------------------------------
    # 7. Debug the hybrid-search results
    # ---------------------------------------------------------
    
    print("\n" + "=" * 70)
    print("DEBUGGING HYBRID SEARCH")
    print("=" * 70)

    total_valid_results = 0
    total_predicate_correct = 0

    for query_id in range(number_of_test_queries):
        query_attribute = attributes_queries_test[query_id]

        matching_database_ids = np.flatnonzero(
            attributes_dataset == query_attribute
        )

        returned_ids = indices[query_id]
        returned_distances = distances[query_id]

        valid_mask = (
            (returned_ids >= 0)
            & (returned_ids < attributes_dataset.shape[0])
        )

        valid_ids = returned_ids[valid_mask]
        valid_distances = returned_distances[valid_mask]

        total_valid_results += valid_ids.size

        print(f"\nQuery {query_id}")
        print("Query attribute:", query_attribute)

        print(
            "Number of matching vectors in full database:",
            matching_database_ids.size
        )

        print("Returned IDs:", returned_ids)
        print("Returned distances:", returned_distances)
        print("Number of valid returned IDs:", valid_ids.size)

        if valid_ids.size > 0:
            returned_metadata = attributes_dataset[valid_ids]

            predicate_mask = (
                returned_metadata == query_attribute
            )

            correct_count = int(np.sum(predicate_mask))
            total_predicate_correct += correct_count

            print("Valid IDs:", valid_ids)
            print("Valid distances:", valid_distances)
            print("Metadata of valid returned IDs:", returned_metadata)

            print(
                "Number satisfying predicate:",
                correct_count
            )

            print(
                "All valid results satisfy predicate:",
                bool(np.all(predicate_mask))
            )
        else:
            print("No valid neighbors were returned.")
    
    # ---------------------------------------------------------
    # 8. Overall search summary
    # ---------------------------------------------------------
    total_results = indices.size
    valid_results = total_valid_results
    invalid_results = total_results - valid_results

    completion_rate = (
        valid_results / total_results
        if total_results > 0
        else 0.0
    )

    predicate_accuracy = (
        total_predicate_correct / valid_results
        if valid_results > 0
        else 0.0
    )

    print("\n" + "=" * 70)
    print("OVERALL SEARCH SUMMARY")
    print("=" * 70)

    print(f"Queries:                  {number_of_test_queries}")
    print(f"Requested neighbors:      {k}")
    print(f"Total result positions:   {total_results}")
    print(f"Valid results:            {valid_results}")
    print(f"Invalid results:          {invalid_results}")
    print(f"Completion rate:          {completion_rate:.2%}")
    print(f"Predicate accuracy:       {predicate_accuracy:.2%}")

    # ---------------------------------------------------------
    # 9. First-query example
    # ---------------------------------------------------------
    first_query_attribute = attributes_queries_test[0]
    first_query_ids = indices[0]
    first_query_distances = distances[0]

    print("\n" + "=" * 70)
    print("FIRST QUERY RESULT")
    print("=" * 70)

    print(f"Query attribute: {first_query_attribute}")
    print("Result IDs:     ", first_query_ids)
    print("Distances:      ", first_query_distances)

if __name__ == "__main__":
    main()