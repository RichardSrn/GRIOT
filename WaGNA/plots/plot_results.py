import os
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch


def get_value_i(df_row, col, i):
    df_row[col] = df_row[col][i]
    return df_row


def errorbar_func(vec: torch.Tensor, d: float = 1):
    m = vec.mean()
    margin = vec.std() * d
    return m + margin, m - margin


def plot_results(path_save=".",
                 path_results=".",
                 path_baselines=None,
                 path_baseline_muzellec=None,
                 show=False, save=True):
    path_save = os.path.join(path_save, "plots")
    os.makedirs(path_save, exist_ok=True)
    # # list dir in path
    # files = os.listdir(path_results)
    # if path_baselines is not None:
    #     file_baselines = os.listdir(path_baselines)
    #     file_baselines = [file for file in file_baselines if file.endswith(".json")]
    # else:
    #     file_baselines = None
    # if path_baseline_muzellec is not None:
    #     file_baseline_muzellec = os.listdir(path_baseline_muzellec)
    #     file_baseline_muzellec = [file for file in file_baseline_muzellec if file.endswith(".json")]
    #     df_muzellec = pd.read_json(os.path.join(path_baseline_muzellec, file_baseline_muzellec[0]))
    # else:
    #     file_baseline_muzellec = None
    # # load json files
    # json_files = [file for file in files if file.endswith(".json")]
    # # get the files that starts with "baselines"
    # # get the files that starts with "data"
    # file_path_data = [file for file in json_files if file.startswith("data") and file.endswith(".json")][0]
    # df_data = pd.read_json(f"{path_results}/{file_path_data}")
    # # round alpha columns to 2 decimals
    # # df_data["alpha"] = df_data["alpha"].round(2)
    # # keep only 1st element of each loss item
    df_data = pd.read_json(path_results)
    df_baselines = pd.read_json(path_baselines) if not path_baselines is None else None
    df_muzellec = pd.read_json(path_baseline_muzellec) if not path_baseline_muzellec is None else None
    for col in [
        "mae", "rmse", "loss",
        "svm_accuracy", "svm_roc_auc_score",
        "svm_precision_micro", "svm_recall_micro", "svm_f1_micro",
        "svm_precision_macro", "svm_recall_macro", "svm_f1_macro",
        "svm_precision_weighted", "svm_recall_weighted", "svm_f1_weighted",
        "gnn_accuracy", "gnn_roc_auc_score",
        "gnn_precision_micro", "gnn_recall_micro", "gnn_f1_micro",
        "gnn_precision_macro", "gnn_recall_macro", "gnn_f1_macro",
        "gnn_precision_weighted", "gnn_recall_weighted", "gnn_f1_weighted",
    ]:
        df_data[col] = df_data[col].apply(lambda x: x[-1])
        df_muzellec[col] = df_muzellec[col].apply(lambda x: x[-1])
    df_data["loss"] = df_data["loss"].apply(lambda x: x["MultiW"] if "MultiW" in x.keys() else x["FGW"])

    colors_baselines = ["cyan", "blue", "midnightblue"]
    labels_baselines = ["true", "average", "average by\ncommunity"]
    linestyles_baselines = [(0, (5, 1, 1, 3)), (0, (5, 1, 1, 1, 1, 3)), (0, (5, 1, 1, 1, 1, 1, 1, 3))]

    for d in [0,1]:
        errorbar_func_d = lambda x: errorbar_func(x, d)
        for g in tqdm(df_data["graph"].unique(),
                      desc="graph".ljust(25),
                      leave=False,
                      position=1,
                      colour="red"):
            # if path_baselines is not None:
            #     current_path_baseline = [file for file in file_baselines if g.lower() in file.lower()]
            #     if len(current_path_baseline) == 0:
            #         df_baselines = None
            #     else:
            #         if len(current_path_baseline) > 1:
            #             if logging:
            #                 logging.warning(f"More than one baseline file found for {g}, using {current_path_baseline[0]}")
            #             else:
            #                 print(f"WARNING : More than one baseline file found for {g}, using {current_path_baseline[0]}")
            #         current_baseline_file = os.path.join(path_baselines, current_path_baseline[0])
            #         df_baselines = pd.read_json(current_baseline_file)
            # else:
            #     df_baselines = None
            if df_muzellec is not None:
                df_muzellec_tmp = df_muzellec[df_muzellec["graph"] == g.upper()]
            else:
                df_muzellec_tmp = None
            for fl in tqdm(df_data[df_data['graph'] == g]["features_label"].unique(),
                           desc="features_label".ljust(25),
                           leave=False,
                           position=2,
                           colour="blue"):
                if df_baselines is not None:
                    df_baselines_tmp = df_baselines[(df_baselines["graph"] == g.upper()) &
                                                    (df_baselines["features_label"] == fl)]
                else:
                    df_baselines_tmp = None
                for lfn in tqdm(df_data[(df_data['graph'] == g) & (df_data['features_label'] == fl)]["lossfn"].unique(),
                                desc="lossfn".ljust(25),
                                leave=False,
                                position=3,
                                colour="yellow"):
                    df_data_tmp = df_data[(df_data["graph"] == g) &
                                          (df_data["features_label"] == fl) &
                                          (df_data["lossfn"] == lfn)]

                    for m in ["mae", "rmse"]:
                        plt.figure(figsize=(15, 7))
                        # data
                        sns.lineplot(data=df_data_tmp,
                                     x="p_miss",
                                     y=m,
                                     hue="label",
                                     alpha=0.75,
                                     errorbar=errorbar_func_d,
                                     markers=True,
                                     style="label",
                                     palette="flare")
                        # simple baseline
                        if path_baselines is not None and df_baselines is not None:
                            sns.lineplot(data=df_baselines_tmp,
                                         x="p_miss",
                                         y=f"{m}_avg",
                                         label=f"average",
                                         color="magenta",
                                         linestyle=(0, (1, 3)),
                                         errorbar=errorbar_func_d,
                                         markers=True)
                            # sns.lineplot(data=df_baselines_tmp,
                            #              x="p_miss",
                            #              y=f"{m}_avg_by_community",
                            #              label=f"average by\ncommunity",
                            #              color="lime",
                            #              linestyle=(0, (5, 3)),
                            #              errorbar=errorbar_func_d,
                            #              markers=True)
                        # muzellec et al. baseline
                        if df_muzellec is not None:
                            sns.lineplot(data=df_muzellec_tmp,
                                         x="p_miss",
                                         y=m,
                                         label=f"muzellec",
                                         color="black",
                                         linestyle=(0, (3, 5, 3, 3)),
                                         alpha=0.75,
                                         errorbar=errorbar_func_d,
                                         markers=True)
                        plt.legend(title="alpha")
                        plt.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.)
                        title = f"{m.upper()} - {g} - {fl} - {lfn} - {d}"
                        plt.title(title)
                        # plt.tight_layout()
                        if show:
                            plt.show()
                        if save:
                            plt.savefig(f"{path_save}/{title.replace(' ', '_')}.jpg")
                        plt.close()

                    if "loss" in df_data_tmp.columns:
                        plt.figure(figsize=(15, 7))
                        sns.lineplot(data=df_data_tmp,
                                     x="p_miss",
                                     y="loss",
                                     hue="label",
                                     alpha=0.75,
                                     errorbar=errorbar_func_d,
                                     markers=True,
                                     style="label",
                                     palette="flare")
                        plt.legend(title="alpha")
                        plt.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.)
                        title = f"LOSS - {g} - {fl} - {lfn} - {d}"
                        plt.title(title)
                        # plt.tight_layout()
                        if show:
                            plt.show()
                        if save:
                            plt.savefig(f"{path_save}/{title.replace(' ', '_')}.jpg")
                        plt.close()

                    for l in ["svm", "gnn"]:
                        for m in ["accuracy", "precision", "recall", "f1"]:
                            for n in ["_micro,", "_macro", "_weighted"] if m != "accuracy" else [""]:
                                if f"{l}_{m}{n}" in df_data_tmp.columns:
                                    plt.figure(figsize=(15, 7))
                                    # data
                                    sns.lineplot(data=df_data_tmp,
                                                 x="p_miss",
                                                 y=f"{l}_{m}{n}",
                                                 hue="label",
                                                 alpha=0.75,
                                                 errorbar=errorbar_func_d,
                                                 markers=True,
                                                 style="label",
                                                 palette="flare")
                                    # simple baseline
                                    if df_baselines_tmp is not None:
                                        for i in range(len(labels_baselines)):
                                            sns.lineplot(data=df_baselines_tmp.apply(lambda x: get_value_i(x,
                                                                                                           f"{l}_{m}{n}",
                                                                                                           i),
                                                                                     axis=1),
                                                         x="p_miss",
                                                         y=f"{l}_{m}{n}",
                                                         label=labels_baselines[i],
                                                         color=colors_baselines[i],
                                                         alpha=0.75,
                                                         linestyle=linestyles_baselines[i],
                                                         errorbar=errorbar_func_d,
                                                         markers=True)
                                    # muzellec et al. baseline
                                    if df_muzellec is not None:
                                        sns.lineplot(data=df_muzellec_tmp,
                                                     x="p_miss",
                                                     y=f"{l}_{m}{n}",
                                                     label=f"muzellec",
                                                     color="black",
                                                     linestyle=(0, (3, 5, 3, 3)),
                                                     errorbar=errorbar_func_d,
                                                     alpha=0.75,
                                                     markers=True)
                                    plt.legend(title="alpha")
                                    plt.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.)
                                    title = f"{l.upper()} {m.upper()}{n.replace('_', ' ')} - {g} - {fl} - {lfn} - {d}"
                                    plt.title(title)
                                    # plt.tight_layout()
                                    if show:
                                        plt.show()
                                    if save:
                                        plt.savefig(f"{path_save}/{title.replace(' ', '_')}.jpg")
                                    plt.close()

                        for m in ["ari", "ami", "nmi"]:
                            if f"{l}_{m}" in df_data_tmp.columns:
                                plt.figure(figsize=(15, 7))
                                # data
                                sns.lineplot(data=df_data_tmp,
                                             x="p_miss",
                                             y=f"{l}_{m}",
                                             hue="label",
                                             alpha=0.75,
                                             errorbar=errorbar_func_d,
                                             markers=True,
                                             style="label",
                                             palette="flare")
                                # simple baseline
                                if df_baselines_tmp is not None:
                                    for i in range(len(labels_baselines)):
                                        sns.lineplot(data=df_baselines_tmp.apply(lambda x: get_value_i(x,
                                                                                                       f"{l}_{m}",
                                                                                                       i),
                                                                                 axis=1),
                                                     x="p_miss",
                                                     y=f"{l}_{m}",
                                                     label=labels_baselines[i],
                                                     color=colors_baselines[i],
                                                     alpha=0.75,
                                                     linestyle=linestyles_baselines[i],
                                                     errorbar=errorbar_func_d,
                                                     markers=True)
                                # muzellec et al. baseline
                                if df_muzellec is not None:
                                    sns.lineplot(data=df_muzellec_tmp,
                                                 x="p_miss",
                                                 y=f"{l}_{m}",
                                                 label=f"muzellec",
                                                 color="black",
                                                 linestyle=(0, (3, 5, 3, 3)),
                                                 errorbar=errorbar_func_d,
                                                 alpha=0.75,
                                                 markers=True)
                                plt.legend(title="alpha")
                                plt.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.)
                                title = f"{l.upper()} {m.upper()} - {g} - {fl} - {lfn} - {d}"
                                plt.title(title)
                                # plt.tight_layout()
                                if show:
                                    plt.show()
                                if save:
                                    plt.savefig(f"{path_save}/{title.replace(' ', '_')}.jpg")
                                plt.close()


def table_results(path_save=".",
                  path_results=".",
                  path_baselines=None,
                  path_baseline_muzellec=None,
                  show=False, save=True):
    def make_latex_table(table, path_save, title, show=False, save=True):
        latex_table = pd.DataFrame(table).T.to_latex()
        latex_table = latex_table.replace("+-", "$\\pm$")
        latex_table = latex_table.replace("toprule\n{}", "toprule\np\\_miss")
        latex_table = "\\begin{table}[h!]\n\\centering\n" + latex_table
        latex_table += "\n\\caption{{{}}}".format(title.replace('_', '\\_'))
        latex_table += f"\n\\label{{tab:{title.replace('- ', '').replace(' ', '_')}}}"
        latex_table += "\n\\end{table}"
        latex_table = latex_table.replace("Muzellec", "Muzellec et al.")
        latex_table = latex_table.replace("toprule", "hline\\hline")
        latex_table = latex_table.replace("midrule", "hline\\hline")
        latex_table = latex_table.replace("bottomrule", "hline\\hline")
        latex_table = latex_table.replace("α", r"\alpha")
        if show:
            print(title, "\n", latex_table)
        if save:
            with open(f"{path_save}/{title.replace(' ', '_')}.txt", "w") as f:
                f.write(title + "\n\n")
                f.write(latex_table)

    path_save = os.path.join(path_save, "tables")
    os.makedirs(path_save, exist_ok=True)

    df_data = pd.read_json(path_results)
    df_baselines = pd.read_json(path_baselines) if not path_baselines is None else None
    df_muzellec = pd.read_json(path_baseline_muzellec) if not path_baseline_muzellec is None else None
    for col in [
        "mae", "rmse", "loss",
        "svm_accuracy",
        "svm_precision_micro", "svm_recall_micro", "svm_f1_micro",
        "svm_precision_macro", "svm_recall_macro", "svm_f1_macro",
        "svm_precision_weighted", "svm_recall_weighted", "svm_f1_weighted",
        "gnn_accuracy",
        "gnn_precision_micro", "gnn_recall_micro", "gnn_f1_micro",
        "gnn_precision_macro", "gnn_recall_macro", "gnn_f1_macro",
        "gnn_precision_weighted", "gnn_recall_weighted", "gnn_f1_weighted",
        "svm_ari", "svm_ami", "svm_nmi", "gnn_ari", "gnn_ami", "gnn_nmi"
    ]:
        df_data[col] = df_data[col].apply(lambda x: x[-1])
        df_muzellec[col] = df_muzellec[col].apply(lambda x: x[-1])
    df_data["loss"] = df_data["loss"].apply(lambda x: x["MultiW"] if "MultiW" in x.keys() else x["FGW"])

    labels_baselines = ["true", "average", "average by community"]

    for g in tqdm(df_data["graph"].unique(),
                  desc="graph".ljust(25),
                  leave=False,
                  position=1,
                  colour="red"):
        df_muzellec_tmp = df_muzellec[df_muzellec["graph"] == g.upper()]
        for fl in tqdm(df_data[df_data['graph'] == g]["features_label"].unique(),
                       desc="features_label".ljust(25),
                       leave=False,
                       position=2,
                       colour="blue"):
            if path_baselines is not None and df_baselines is not None:
                df_baselines_tmp = df_baselines[(df_baselines["graph"] == g.upper()) &
                                                (df_baselines["features_label"] == fl)]
            else:
                df_baselines_tmp = None
            for lfn in tqdm(df_data[(df_data['graph'] == g) & (df_data['features_label'] == fl)]["lossfn"].unique(),
                            desc="lossfn".ljust(25),
                            leave=False,
                            position=3,
                            colour="yellow"):
                df_data_tmp = df_data[(df_data["graph"] == g) &
                                      (df_data["features_label"] == fl) &
                                      (df_data["lossfn"] == lfn)]

                for m in ["mae", "rmse"]:
                    if m not in df_data_tmp.columns:
                        continue
                    title = f"{m.upper()} - {g} - {fl} - {lfn}"
                    table = dict()

                    # data
                    for label in df_data_tmp["label"].unique():
                        table[label] = dict()
                        for pm in df_data_tmp["p_miss"].unique():
                            avg = df_data_tmp[(df_data_tmp["p_miss"] == pm) &
                                              (df_data_tmp["label"] == label)][m].values.mean()
                            std = df_data_tmp[(df_data_tmp["p_miss"] == pm) &
                                              (df_data_tmp["label"] == label)][m].values.std()
                            table[label][pm] = f'{avg:.3f}+-{std:.3f}'

                    # simple baseline
                    if path_baselines is not None and df_baselines is not None:
                        table["average"] = dict()
                        table["average by community"] = dict()
                        for pm in df_baselines_tmp["p_miss"].unique():
                            avg = df_baselines_tmp[df_baselines_tmp["p_miss"] == pm][f"{m}_avg"].values.mean()
                            std = df_baselines_tmp[df_baselines_tmp["p_miss"] == pm][f"{m}_avg"].values.std()
                            table["average"][pm] = f'{avg:.3f}+-{std:.3f}'
                            avg = df_baselines_tmp[df_baselines_tmp["p_miss"] == pm][
                                f"{m}_avg_by_community"].values.mean()
                            std = df_baselines_tmp[df_baselines_tmp["p_miss"] == pm][
                                f"{m}_avg_by_community"].values.std()
                            table["average by community"][pm] = f'{avg:.3f}+-{std:.3f}'

                    # muzellec et al. baseline
                    table["Muzellec"] = dict()
                    for pm in df_muzellec_tmp["p_miss"].unique():
                        avg = df_muzellec_tmp[df_muzellec_tmp["p_miss"] == pm][m].values.mean()
                        std = df_muzellec_tmp[df_muzellec_tmp["p_miss"] == pm][m].values.std()
                        table["Muzellec"][pm] = f'{avg:.3f}+-{std:.3f}'

                    make_latex_table(table, path_save, title, save=save, show=show)

                if "loss" in df_data_tmp.columns:
                    title = f"LOSS - {g} - {fl} - {lfn}"
                    table = dict()

                    # data
                    for label in df_data_tmp["label"].unique():
                        table[label] = dict()
                        for pm in df_data_tmp["p_miss"].unique():
                            avg = df_data_tmp[(df_data_tmp["p_miss"] == pm) &
                                              (df_data_tmp["label"] == label)]["loss"].values.mean()
                            std = df_data_tmp[(df_data_tmp["p_miss"] == pm) &
                                              (df_data_tmp["label"] == label)]["loss"].values.std()
                            table[label][pm] = f'{avg:.3f}+-{std:.3f}'

                    make_latex_table(table, path_save, title, save=save, show=show)

                for l in ["svm", "gnn"]:
                    for m in ["accuracy", "precision", "recall", "f1"]:
                        for n in ["_micro,", "_macro", "_weighted"] if m != "accuracy" else [""]:
                            if f"{l}_{m}{n}" in df_data_tmp.columns:
                                title = f"{l.upper()} {m.upper()}{n.replace('_', ' ')} - {g} - {fl} - {lfn}"
                                table = dict()
                                # data
                                for label in df_data_tmp["label"].unique():
                                    table[label] = dict()
                                    for pm in df_data_tmp["p_miss"].unique():
                                        avg = df_data_tmp[(df_data_tmp["p_miss"] == pm) &
                                                          (df_data_tmp["label"] == label)][f"{l}_{m}{n}"].values.mean()
                                        std = df_data_tmp[(df_data_tmp["p_miss"] == pm) &
                                                          (df_data_tmp["label"] == label)][f"{l}_{m}{n}"].values.std()
                                        table[label][pm] = f'{avg:.3f}+-{std:.3f}'
                                # simple baseline
                                if df_baselines_tmp is not None:
                                    for i in range(3):
                                        table[labels_baselines[i]] = dict()
                                        for pm in df_baselines_tmp["p_miss"].unique():
                                            avg = df_baselines_tmp.apply(lambda x: get_value_i(x,
                                                                                               f"{l}_{m}{n}",
                                                                                               i),
                                                                         axis=1)[(df_baselines_tmp["p_miss"] == pm)][
                                                f"{l}_{m}{n}"].values.mean()
                                            std = df_baselines_tmp.apply(lambda x: get_value_i(x,
                                                                                               f"{l}_{m}{n}",
                                                                                               i),
                                                                         axis=1)[(df_baselines_tmp["p_miss"] == pm)][
                                                f"{l}_{m}{n}"].values.std()
                                            table[labels_baselines[i]][pm] = f'{avg:.3f}+-{std:.3f}'
                                # muzellec et al. baseline
                                table["Muzellec"] = dict()
                                for pm in df_muzellec_tmp["p_miss"].unique():
                                    avg = df_muzellec_tmp[df_muzellec_tmp["p_miss"] == pm][f"{l}_{m}{n}"].values.mean()
                                    std = df_muzellec_tmp[df_muzellec_tmp["p_miss"] == pm][f"{l}_{m}{n}"].values.std()
                                    table["Muzellec"][pm] = f'{avg:.3f}+-{std:.3f}'

                                make_latex_table(table, path_save, title, save=save, show=show)

                    for m in ["ari", "ami", "nmi"]:
                        if f"{l}_{m}" in df_data_tmp.columns:
                            title = f"{l.upper()} {m.upper()} - {g} - {fl} - {lfn}"
                            table = dict()
                            # data
                            for label in df_data_tmp["label"].unique():
                                table[label] = dict()
                                for pm in df_data_tmp["p_miss"].unique():
                                    avg = df_data_tmp[(df_data_tmp["p_miss"] == pm) &
                                                      (df_data_tmp["label"] == label)][f"{l}_{m}"].values.mean()
                                    std = df_data_tmp[(df_data_tmp["p_miss"] == pm) &
                                                      (df_data_tmp["label"] == label)][f"{l}_{m}"].values.std()
                                    table[label][pm] = f'{avg:.3f}+-{std:.3f}'
                            # simple baseline
                            if df_baselines_tmp is not None:
                                for i in range(3):
                                    table[labels_baselines[i]] = dict()
                                    for pm in df_baselines_tmp["p_miss"].unique():
                                        avg = df_baselines_tmp.apply(lambda x: get_value_i(x,
                                                                                           f"{l}_{m}",
                                                                                           i),
                                                                     axis=1)[(df_baselines_tmp["p_miss"] == pm)][
                                            f"{l}_{m}"].values.mean()
                                        std = df_baselines_tmp.apply(lambda x: get_value_i(x,
                                                                                           f"{l}_{m}",
                                                                                           i),
                                                                     axis=1)[(df_baselines_tmp["p_miss"] == pm)][
                                            f"{l}_{m}"].values.std()
                                        table[labels_baselines[i]][pm] = f'{avg:.3f}+-{std:.3f}'
                            # muzellec et al. baseline
                            table["Muzellec"] = dict()
                            for pm in df_muzellec_tmp["p_miss"].unique():
                                avg = df_muzellec_tmp[df_muzellec_tmp["p_miss"] == pm][f"{l}_{m}"].values.mean()
                                std = df_muzellec_tmp[df_muzellec_tmp["p_miss"] == pm][f"{l}_{m}"].values.std()
                                table["Muzellec"][pm] = f'{avg:.3f}+-{std:.3}'

                            make_latex_table(table, path_save, title, save=save, show=show)
