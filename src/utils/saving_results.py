import os
import pickle
import logging


def save_to_pickle(data, filename):
    os.makedirs("simulation_results", exist_ok=True)
    filepath = os.path.join("simulation_results", filename)
    with open(filepath, "wb") as f:
        pickle.dump(data, f)
    logging.info(f"Saved: {filepath}")
    return filepath
