import argparse
import pandas as pd

from knee_rag.retriever import EmbeddingGuidanceRetriever
from knee_rag.constants import LABEL_COLS


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--index-dir", required=True)
    p.add_argument("--row", type=int, default=0)
    p.add_argument("--top-k", type=int, default=2)
    args = p.parse_args()

    df = pd.read_csv(args.csv)
    report = str(df.iloc[args.row]["Report"])
    r = EmbeddingGuidanceRetriever(args.index_dir)

    print("=" * 100)
    print("REPORT")
    print("=" * 100)
    print(report[:4000])

    for target in LABEL_COLS:
        print("\n" + "=" * 100)
        print(target)
        for hit in r.retrieve(target, report, top_k=args.top_k):
            print(f"\nscore={hit['score']:.4f} source={hit['source']} chunk={hit['chunk_id']}")
            print(hit["text"][:1800])


if __name__ == "__main__":
    main()
