import json
import os


def save_experiment(config, results, path="experiments.json"):
    entry = {
        "config": config,
        "results": results
    }

    if os.path.exists(path):
        with open(path, "r") as f:
            data = json.load(f)
    else:
        data = []

    data.append(entry)

    with open(path, "w") as f:
        json.dump(data, f, indent=4)