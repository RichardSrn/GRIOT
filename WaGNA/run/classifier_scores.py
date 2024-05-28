from ..model_classifier.GNN import train_test_gnn
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from ..config import cfg

tqdm.pandas(leave=False)

def classifiers_scores(path_json, final_only=True):
    def get_scores(x):
        # gr = next((item for item in cfg.graphs_parameters if item["name"] == x["graph"]), None)
        # cfg.graph_parameters is now a single element
        gr = cfg.graph_parameters
        proximity_matrix = np.load(gr["proximity_matrix"])
        scores = None
        for i, step_path in enumerate(x["save_steps_path"]):
            if final_only and not step_path.endswith("FINAL_.npy"):
                continue
            F_set = np.load(step_path)
            scores_gnn = train_test_gnn(proximity_matrix,
                                        F_set.copy(),
                                        seed=cfg.seeds[x["n_rep_idx"]],
                                        save_path="/".join(step_path.split("/")[:-1]),
                                        save_name=step_path.split("/")[-1].replace(".npy", ""),
                                        device=cfg.device)
            if scores is None:
                scores = {k: [v] for k, v in scores_gnn.items()}
            else:
                for k, v in scores_gnn.items():
                    scores[k].append(v)
        return scores

    # load json file
    df = pd.read_json(path_json)
    df = pd.concat([df, df[["graph",
                            "save_steps_path",
                            "n_rep_idx"]].progress_apply(get_scores,
                                                         axis=1,
                                                         result_type="expand")
                    ], axis=1)
    path_save = path_json.replace(".json", "_classifiers_scores.json")
    df.to_json(path_save,
               orient="records",
               indent=4)
    return path_save
