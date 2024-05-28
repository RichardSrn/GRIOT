import datetime
import os
import torch
import warnings

from tqdm import tqdm

from ..config import cfg

from typing import Union

from ..baseline_OTtab.baseline_muzellec import baseline_muzellec
from ..baseline_FP.baseline_rossi import baseline_rossi
from ..baselines_weak.baselines import get_baselines
from ..baseline_PaGNN.baseline_pagnn import baseline_pagnn, pagnn_scores
from ..data_path import data_path
from ..plots.plot_results import plot_results, table_results
from ..run.classifier_scores import classifiers_scores
from ..run.train_test_models import train_test
from ..run.train_test_models_v2 import train_test2
from ..run.train_test_multi_models import train_test_multi
from ..run.train_test_multi_models_v2 import train_test2_multi
from ..tools.timer import timer

warnings.filterwarnings("ignore", category=FutureWarning)

tqdm.pandas()


class RunTest:
    def __init__(self):
        cfg.seeds = cfg.rng.randint(100, 9999, cfg.nr)
        cfg.graph_parameters = data_path(cfg.graph_name)
        if cfg.path_proximity is not None:
            cfg.graph_parameters["proximity_matrix"] = cfg.path_proximity
        cfg.paths_save = {}
        cfg.sumup(show=True)

    def start(self):
        if not "skip" in cfg.path_save_weak:
            if cfg.path_save_weak.endswith(".json"):
                cfg.paths_save["path_save_weak"] = cfg.path_save_weak
            else:
                self.impute(which="weak")
            if not cfg.path_save_weak.endswith("classifiers_scores.json") and not cfg.skip_classification:
                self.classify(which="weak")

        elif not "skip" in cfg.path_save_muzellec:
            if cfg.path_save_muzellec.endswith(".json"):
                cfg.paths_save["path_save_muzellec"] = cfg.path_save_muzellec
            else:
                self.impute(which="muzellec")
            if not cfg.path_save_muzellec.endswith("classifiers_scores.json") and not cfg.skip_classification:
                self.classify(which="muzellec")

        elif not "skip" in cfg.path_save_rossi:
            if cfg.path_save_rossi.endswith(".json"):
                cfg.paths_save["path_save_rossi"] = cfg.path_save_rossi
            else:
                self.impute(which="rossi")
            if not cfg.path_save_rossi.endswith("classifiers_scores.json") and not cfg.skip_classification:
                self.classify(which="rossi")

        elif not "skip" in cfg.path_save_pagnn:
            if cfg.path_save_pagnn.endswith(".json"):
                cfg.paths_save["path_save_pagnn"] = cfg.path_save_pagnn
            else:
                self.impute(which="pagnn")
            if not cfg.path_save_pagnn.endswith("classifiers_scores.json") and not cfg.skip_classification:
                self.classify(which="pagnn")

        elif not "skip" in cfg.path_save_wagna:
            if cfg.path_save_wagna.endswith(".json"):
                cfg.paths_save["path_save_wagna"] = cfg.path_save_wagna
            else:
                self.impute(which="wagna")
            if not cfg.path_save_wagna.endswith("classifiers_scores.json") and not cfg.skip_classification:
                self.classify(which="wagna")
            else:
                cfg.paths_save["path_save_wagna_classifier"] = cfg.path_save_wagna

        elif not "skip" in cfg.path_save_wagnamulti:
            if cfg.path_save_wagnamulti.endswith(".json"):
                cfg.paths_save["path_save_wagnamulti"] = cfg.path_save_wagnamulti
            else:
                self.impute(which="wagnamulti")
            if not cfg.path_save_wagnamulti.endswith("classifiers_scores.json") and not cfg.skip_classification:
                self.classify(which="wagnamulti")
            else:
                cfg.paths_save["path_save_wagnamulti_classifier"] = cfg.path_save_wagnamulti

        elif not "skip" in cfg.path_save_wagna2:
            if cfg.path_save_wagna2.endswith(".json"):
                cfg.paths_save["path_save_wagna2"] = cfg.path_save_wagna2
            else:
                self.impute(which="wagna2")
            if not cfg.path_save_wagna2.endswith("classifiers_scores.json") and not cfg.skip_classification:
                self.classify(which="wagna2")
            else:
                cfg.paths_save["path_save_wagna2_classifier"] = cfg.path_save_wagna2

        else:
            raise ValueError("No path to save results")

        if cfg.plot and False:
            self.plot_table()

    @timer
    def plot_table(self, path_save=""):
        cfg.logger.info("plots and tables")
        plot_results(path_save=path_save if path_save else cfg.UNIQUE,
                     path_results=cfg.paths_save["path_save_wagna_classifier"],
                     path_baselines=cfg.paths_save["path_save_weak"],
                     path_baseline_muzellec=cfg.paths_save["path_save_muzellec"])
        table_results(path_save=path_save if path_save else cfg.UNIQUE,
                      path_results=cfg.paths_save["path_save_wagna_classifier"],
                      path_baselines=cfg.paths_save["path_save_weak"],
                      path_baseline_muzellec=cfg.paths_save["path_save_muzellec"])

    @timer
    def impute(self, which=""):
        """
        :param which: \in \{"weak", "muzellec", "wagna", "wagnamulti", "wagna2", "pagnn"\}
        """
        if which.lower() == "weak":
            cfg.logger.info('BASELINES WEAK')
            cfg.paths_save["path_save_weak"] = get_baselines()
        elif which.lower() == "muzellec":
            cfg.logger.info('BASELINE MUZELLEC et al.')
            assert cfg.model in {"basic", "rr"}, f"model should be in {{basic, rr}} but is {cfg.model}"
            cfg.paths_save["path_save_muzellec"] = baseline_muzellec()
        elif which.lower() == "rossi":
            cfg.logger.info('BASELINE ROSSI et al.')
            assert cfg.model == "fp", f"model should be 'fp' but is {cfg.model}"
            cfg.paths_save["path_save_rossi"] = baseline_rossi()
        elif which.lower() == "pagnn":
            cfg.logger.info('BASELINE PaGNN')
            assert cfg.model in ["rand", "fp", "gcn"], f"model should be in rand, or fp, or gcn but is {cfg.model}"
            cfg.paths_save["path_save_pagnn"] = baseline_pagnn()
        elif which.lower() == "wagna":
            assert cfg.model in {"basic", "rr", "gcn", "gat", "gct"}, \
                f"model should be in {{basic, rr, gcn, gat, gct}} but is {cfg.model}"
            cfg.logger.info('TESTS model ' + cfg.model.upper())
            cfg.paths_save["path_save_wagna"] = train_test()
        elif which.lower() == "wagnamulti":
            assert cfg.model in {"basic", "rr", "gcn", "gat", "gct"}, \
                f"model should be in {{basic, rr, gcn, gat, gct}} but is {cfg.model}"
            cfg.logger.info('TESTS MULTI model ' + cfg.model.upper())
            cfg.paths_save["path_save_wagnamulti"] = train_test_multi()
        elif which.lower() == "wagna2":
            assert self.model in {"basic", "rr", "gcn", "gat", "gct"}, \
                f"model should be in {{basic, rr, gcn, gat, gct}} but is {cfg.model}"
            cfg.logger.info('TESTS model ' + cfg.model.upper())
            cfg.paths_save["path_save_wagna2"] = train_test2()
        else:
            raise ValueError(
                f"which should be in {{weak, muzellec, rossi, pagnn, wagna, wagnamulti, wagna2}} but is {which}")

    @timer
    def classify(self, which=""):
        """
        :param which: \in \{"weak", "muzellec", "wagna"\}
        :return:
        """
        if which.lower() == "muzellec":
            cfg.logger.info('CLASSIFIER MUZELLEC')
            cfg.paths_save["path_save_muzellec"] = classifiers_scores(path_json=cfg.paths_save["path_save_muzellec"],
                                                                      final_only=True, )
        elif which.lower() == "rossi":
            cfg.logger.info('CLASSIFIER ROSSI')
            cfg.paths_save["path_save_rossi"] = classifiers_scores(path_json=cfg.paths_save["path_save_rossi"],
                                                                   final_only=True, )
        elif which.lower() == "pagnn":
            cfg.logger.info('CLASSIFIER PaGNN')
            cfg.paths_save["path_save_pagnn"] = pagnn_scores(path_json=cfg.paths_save["path_save_pagnn"],
                                                             final_only=True, )
        elif which.lower() == "weak":
            cfg.logger.info('CLASSIFIER WEAK')
            cfg.paths_save["path_save_weak"] = classifiers_scores(path_json=cfg.paths_save["path_save_weak"],
                                                                  final_only=False, )
        elif which.lower() == "wagna":
            cfg.logger.info('CLASSIFIER TEST')
            cfg.paths_save[
                "path_save_wagna_classifier"
            ] = classifiers_scores(path_json=cfg.paths_save["path_save_wagna"],
                                   final_only=True, )
        elif which.lower() == "wagnamulti":
            cfg.logger.info('CLASSIFIER TEST MULTI')
            cfg.paths_save[
                "path_save_wagnamulti_classifier"
            ] = classifiers_scores(path_json=cfg.paths_save["path_save_wagnamulti"],
                                   final_only=True, )
        elif which.lower() == "wagna2":
            cfg.logger.info('CLASSIFIER TEST')
            cfg.paths_save[
                "path_save_wagna2_classifier"
            ] = classifiers_scores(path_json=cfg.paths_save["path_save_wagna2"],
                                   final_only=True, )

        else:
            raise ValueError(
                f"which should be in {{weak, muzellec, rossi, pagnn, wagna, wagnamulti, wagna2}} but is {which}")


@timer
def main():
    assert sum(
        (cfg.file_weak == "skip",
         cfg.file_muzellec == "skip",
         cfg.file_wagna == "skip",
         cfg.file_wagna2 == "skip",
         cfg.file_rossi == "skip",
         cfg.file_pagnn == "skip",
         cfg.file_wagnamulti == "skip")) >= 6, 'At most one of the following can be different from skip : --file_weak,' \
                                               ' --file_muzellec, --file_rossi, --file_pagnn, --file_wagna, --file_wagnamulti.' \
                                               ' Here --file_weak = {}, --file_muzellec = {}, --file_rossi = {}, ' \
                                               '--file_pagnn = {}, --file_wagna = {}, --file_wagna_multi = {}, ' \
                                               '--file_wagna2 = {}.'.format(
        cfg.file_weak,
        cfg.file_muzellec,
        cfg.file_rossi,
        cfg.file_pagnn,
        cfg.file_wagna,
        cfg.file_wagnamulti,
        cfg.file_wagna2,
    )

    if sum((cfg.file_weak == "skip",
            cfg.file_muzellec == "skip",
            cfg.file_wagna == "skip",
            cfg.file_wagnamulti == "skip",
            cfg.file_wagna2 == "skip",
            cfg.file_rossi == "skip",
            cfg.file_pagnn == "skip")) == 7:
        assert cfg.which != None, "if all files are skipped, --which must be specified"

        if cfg.which == "weak":
            cfg.file_weak = "None"
        elif cfg.which == "muzellec":
            cfg.file_muzellec = "None"
        elif cfg.which == "wagna" or cfg.which == "ours":
            cfg.file_wagna = "None"
        elif cfg.which == "wagnamulti" or cfg.which == "oursmulti":
            cfg.file_wagnamulti = "None"
        elif cfg.which == "wagna2":
            cfg.file_wagna2 = "None"
        elif cfg.which == "rossi":
            cfg.file_rossi = "None"
        elif cfg.which == "pagnn":
            cfg.file_pagnn = "None"
        else:
            raise ValueError(f"which must be in {{weak, muzellec, rossi, pagnn, wagna, wagnamulti, wagna2}}"
                             f" but is f{cfg.which}")

    if cfg.model == "rr":
        assert cfg.graph_name != "real", "RR takes too much time for running on real graphs"
        assert cfg.graph_name != "wkb", "RR takes too much time for running on wkb graphs"
        assert cfg.graph_name != "planetoid", "RR takes too much time for running on planetoid graphs"
        for g in ["citeseer", "cora", "pubmed"]:#, "texa", "cornell", "wisconsin"]:
            assert g not in cfg.graph_name, "RR takes too much time for running on {}".format(g)

    assert (cfg.mecha == "MCAR" and cfg.opt == "None") or \
           (cfg.mecha == "MNAR" and cfg.opt in ["logistic", "quantile", "selfmasked"]), \
        "opt must be None for MCAR, and in {logistic, quantile, selfmasked} for MNAR"

    assert ((cfg.ni is None and cfg.mi is None) or (cfg.epochs is None or cfg.epochs == "None")) and \
           (cfg.ni is not None or (cfg.epochs is not None and cfg.epochs != "None")), \
        "Either ni or epochs must be specified, but not both."

    assert cfg.model is not None or cfg.file_weak != "skip", "Model must be specified if not running weak baselines."

    if cfg.epochs is not None and cfg.epochs != "None":
        if cfg.report_interval is None or cfg.report_interval == "None":
            cfg.report_interval = 25
        cfg.mi = cfg.epochs // cfg.report_interval + cfg.epochs % cfg.report_interval
        cfg.ni = cfg.report_interval

    if cfg.ni is not None:
        if cfg.model == "basic" and cfg.ni == 25:
            cfg.ni = 250
        if cfg.model == "basic":
            cfg.mi = None

    cfg.path_save_weak = os.path.join(cfg.dir_weak, cfg.file_weak)
    cfg.path_save_muzellec = os.path.join(cfg.dir_muzellec, cfg.file_muzellec)
    cfg.path_save_wagna = os.path.join(cfg.dir_wagna, cfg.file_wagna)
    cfg.path_save_wagnamulti = os.path.join(cfg.dir_wagnamulti, cfg.file_wagnamulti)
    cfg.path_save_wagna2 = os.path.join(cfg.dir_wagna2, cfg.file_wagna2)
    cfg.path_save_rossi = os.path.join(cfg.dir_rossi, cfg.file_rossi)

    cfg.path_save_pagnn = os.path.join(cfg.dir_pagnn, cfg.file_pagnn)

    # if cfg.graph_name == "artificial":
    #     cfg.graph_name = ("sbm",)  # , "dancer", "dancer_messy")
    # elif cfg.graph_name == "artificial_fit01":
    #     cfg.graph_name = ("sbm_fit01",)  # , "dancer_fit01", "dancer_messy_fit01")
    # elif cfg.graph_name == "artificial_binary" or cfg.graph_name == "artificial_binarised":
    #     cfg.graph_name = ("sbm_binary",)  # , "dancer_binary", "dancer_messy_binary")
    # elif cfg.graph_name == "real":
    #     cfg.graph_name = ("citeseer", "cora", "pubmed", "texas", "cornell", "wisconsin")
    # elif cfg.graph_name == "wkb" or cfg.graph_name == "webkb":
    #     cfg.graph_name = ("texas", "cornell", "wisconsin")
    # elif cfg.graph_name == "planetoid":
    #     cfg.graph_name = ("citeseer", "cora", "pubmed")
    # else:
    #     cfg.graph_name = cfg.graph_name.split(",")

    cfg.UNIQUE = f"./output/"
    if cfg.file_wagna != "skip":
        cfg.UNIQUE = os.path.join(cfg.UNIQUE, "TEST")
    elif cfg.file_wagnamulti != "skip":
        cfg.UNIQUE = os.path.join(cfg.UNIQUE, "TESM")
    elif cfg.file_wagna2 != "skip":
        cfg.UNIQUE = os.path.join(cfg.UNIQUE, "TES2")
    elif cfg.file_muzellec != "skip":
        cfg.UNIQUE = os.path.join(cfg.UNIQUE, "MUZE")
        cfg.lossfn, cfg.alphas, cfg.epsilons, cfg.p = None, None, None, None
    elif cfg.file_rossi != "skip":
        cfg.UNIQUE = os.path.join(cfg.UNIQUE, "ROSS")
        cfg.mi, cfg.np, cfg.reg, cfg.batch_size = None, None, None, None
        cfg.lossfn, cfg.alphas, cfg.epsilons, cfg.p = None, None, None, None
    elif cfg.file_pagnn != "skip":
        cfg.UNIQUE = os.path.join(cfg.UNIQUE, "PAGN")
        if cfg.model != "gcn":
            cfg.mi, cfg.np, cfg.reg, cfg.batch_size = None, None, None, None
            cfg.lossfn, cfg.alphas, cfg.epsilons, cfg.p = None, None, None, None
            if cfg.model == "rand":
                cfg.ni = None
    elif cfg.file_weak != "skip":
        cfg.UNIQUE = os.path.join(cfg.UNIQUE, "SIMP")
        cfg.mi, cfg.np, cfg.ni, cfg.model, cfg.reg, cfg.batch_size = None, None, None, None, None, None
        cfg.lossfn, cfg.alphas, cfg.epsilons, cfg.p = None, None, None, None
    else:
        raise ValueError("No file to save results")

    if cfg.ni is None and cfg.mi is not None and cfg.epochs is not None:
        cfg.mi = cfg.epochs
    elif cfg.mi is None and cfg.ni is not None and cfg.epochs is not None:
        cfg.ni = cfg.epochs
    elif cfg.mi is None and cfg.ni is None:
        pass  # do nothing in this case

    if cfg.model != "gat" and cfg.model != "gct":
        cfg.num_heads = "None"
        cfg.concat = "None"

    if cfg.lossfn == "FGW":
        cfg.epsilons, cfg.p = None, None

    os.makedirs(cfg.UNIQUE, exist_ok=True)

    cfg.UNIQUE = os.path.join(
        cfg.UNIQUE,
        f"{cfg.prefix}_{cfg.graph_name}"
    )

    if cfg.restructure == "True":
        cfg.restructure = True
    elif cfg.restructure == "False":
        cfg.restructure = False
        cfg.restruct_iters = None
        cfg.restruct_thld = None
        cfg.restruct_thld_low = None
        cfg.restruct_method = None
        cfg.restruct_integer = None
        cfg.restruct_for_loss = None
    else :
        raise NotImplementedError(f"restructure={cfg.restructure} has not been implemented yet.")

    for s in [f"_{cfg.model}",
              f"_{cfg.num_heads}",
              f"_{cfg.concat}"
              f"_{cfg.dropout}",
              f"_{cfg.normalization}",
              f"_{cfg.mecha}",
              f"_{cfg.opt}",
              f"_{cfg.lossfn}",
              f"_nr{cfg.nr}",
              f"_np{cfg.np}",
              f"_ni{cfg.ni}",
              f"_mi{cfg.mi}",
              f"_bs{cfg.batch_size}",
              f"_p{cfg.p}",
              f"_pmiss{cfg.pmiss}",
              f"_α{cfg.alphas}",
              f"_ε{cfg.epsilons}",
              f"_CE{cfg.use_ce}",
              f"_GEOM{cfg.use_geomloss}",
              f"_R{cfg.restructure}"]:
        if not s.endswith("None"):
            cfg.UNIQUE += s
    if cfg.restructure == True:
        for s in [f"_Rloss{cfg.restruct_for_loss}",
                  f"_Ri{cfg.restruct_iters}",
                  f"_Rr{cfg.restruct_integer}",
                  f"_Rthl{cfg.restruct_thld}",
                  f"_Rmtd{cfg.restruct_method}"]:
            cfg.UNIQUE += s
    if cfg.restruct_method == "increment":
        cfg.UNIQUE += f"_RthlL{cfg.restruct_thld_low}"

    if cfg.plot:
        cfg.UNIQUE += "_plot"

    if cfg.unique == "auto" :
        cfg.UNIQUE += f"_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    elif cfg.unique != "False" :
        cfg.UNIQUE += f"_{cfg.unique}"

    os.makedirs(cfg.UNIQUE, exist_ok=True)

    cfg.p_miss_s = [float(x) for x in cfg.pmiss.split(",")]

    if cfg.alphas is not None:
        if cfg.file_wagna != "skip" or (cfg.file_pagnn != "skip" and cfg.model == "gcn") or cfg.file_wagna2 != 2:
            if ',' in cfg.alphas:
                cfg.alphas = [float(x) for x in cfg.alphas.split(",")]
            else:
                cfg.alphas = [float(cfg.alphas)]
        elif cfg.file_wagnamulti != "skip":
            cfg.alphas = [a for a in cfg.alphas.split("_")]
            cfg.alphas = [tuple(float(a) for a in alpha.split(',')) for alpha in cfg.alphas]
            # run assertion check: the sum of each tuple should be equal to 1
            for alpha in cfg.alphas:
                assert sum(alpha) == 1, f"sum of alpha should be 1, but is sum({alpha})={sum(alpha)}."
        else:
            raise ValueError("alphas should be None with these parameters.")
    else:
        cfg.alphas = [cfg.alphas]

    if cfg.epsilons is not None:
        if ',' in cfg.epsilons:
            cfg.epsilons = [float(x) for x in cfg.epsilons.split(",")]
        else:
            cfg.epsilons = [float(cfg.epsilons)]
    else:
        cfg.epsilons = [cfg.epsilons]

    if cfg.p is not None:
        if cfg.p == "inf":
            cfg.p = torch.inf
        else:
            cfg.p = float(cfg.p)

    if cfg.device is None:
        cfg.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        assert torch.cuda.is_available() or cfg.device == "cpu", "No GPU available, use CPU instead."
        cfg.device = torch.device(cfg.device)

    run_test = RunTest()

    run_test.start()