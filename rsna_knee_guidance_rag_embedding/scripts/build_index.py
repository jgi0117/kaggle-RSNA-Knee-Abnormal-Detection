import argparse

from knee_rag.indexer import build_index


def main():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--guidance-dir",
        required=True,
    )

    p.add_argument(
        "--output-dir",
        required=True,
    )

    p.add_argument(
        "--embedding-model",
        default="intfloat/multilingual-e5-small",
    )

    args = p.parse_args()

    n = build_index(
        args.guidance_dir,
        args.output_dir,
        embedding_model=args.embedding_model,
    )

    print(
        f"Built dense embedding index "
        f"with {n} chunks -> {args.output_dir}"
    )


if __name__ == "__main__":
    main()
