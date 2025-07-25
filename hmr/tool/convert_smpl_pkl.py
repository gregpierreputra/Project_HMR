import pickle
from argparse import ArgumentParser
from pathlib import Path

import dill


def convert_pkl(old_pkl: str, new_pkl: str):
    """
    Convert a Python 2 pickle to Python 3
    """

    # Convert Python 2 "ObjectType" to Python 3 object
    dill._dill._reverse_typemap["ObjectType"] = object

    # Open the pickle using latin1 encoding
    with open(old_pkl, "rb") as f:
        loaded = pickle.load(f, encoding="latin1")

    # Re-save as Python 3 pickle
    with open(new_pkl, "wb") as outfile:
        pickle.dump(loaded, outfile)


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Convert SMPL pkl file to be compatible with Python 3"
    )

    parser.add_argument(
        "--old_pkl",
        type=Path,
        help="Path to old pickle file (e.g. basicModel_neutral_lbs_10_207_0_v1.0.0.pkl).",
        required=True,
    )
    parser.add_argument(
        "--new_pkl", type=Path, help="Save path for the new pickle file.", required=True
    )
    args = parser.parse_args()

    if not args.old_pkl.exists():
        raise FileNotFoundError(args.old_pkl)
    if not args.new_pkl.parent.exists():
        raise FileNotFoundError(f"Parent folder does not exist: {args.new_pkl}")

    convert_pkl(str(args.old_pkl), str(args.new_pkl))
