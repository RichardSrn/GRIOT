import torch
import logging
import warnings
import click

warnings.filterwarnings("ignore", category=FutureWarning)

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logging.debug("test")

torch.set_default_dtype(torch.double)


# from .model_imputers import imputer_GRIT
#
# exit()


@click.command()
@click.option('--graph-name', '-G',
              required=False,
              help="""\b
              graphs name, belongs to :
              {sbm, cora, texas, cornell, wisconsin}
              or
              {artificial, real (=WKB+PLANETOID), wkb, planetoid}.
              It can be multiple (seperated by a comma, no space).
              >
              """,
              type=str,
              default="texas",
              show_default=True)
@click.option('--model', '-M',
              default=None,
              help="""\b
              model, belongs to : 
              {
              basic,  # Basic model adapted from Muzellec et al.
              rr,  # Algorithm3 adapted from Muzellec et al. (does not run with real graphs)
              gcn,  # GCN imputer (for GRIOT)
              gat,  # GATv2 imputer (for GRIOT)
              gct,  # GCT (Graph Convolution Transformer) imputer (for GRIOT)
              fp, # Feature Propagation from Rossi et al.
              }.
              Required with -W \in \{griot, griotmulti, griot2, pagnn, muzellec\}.
              Only basic and rr are available with -W muzellec.
              >
              """,
              type=str,
              show_default=True)
@click.option('--num-heads', '-H',
              default=1,
              help="""\b
              Number of heads for GATv2 and GCT.
              """,
              type=int,
              show_default=True)
@click.option('--concat', '-c',
                default=True,
                help="""\b
                Whether to concatenate heads output or to average them. 
                Used only for GATv2 and GCT, when --num-heads > 1.
                """,
                type=bool,
                show_default=True)
@click.option('--dropout', '-D',
              default=0.5,
              help="""\b
              Dropout rate for GATv2 and GCT.
              """,
              type=float,
              show_default=True)
@click.option('--plot',
              default=False,
              help="""\b
              plot, True or False
              >
              """,
              type=bool,
              show_default=True)
@click.option('--which', "-W",
              default=None,
              help="""\b
              Which part of the code to run, belongs to :
              {
              weak, # run baselines weak (zeros, random, average, average neighbors)
              muzellec, # run baseline Muzellec et al.
              rossi, # run baseline features propagation (Rossi et al.)
              pagnn, # run PaGNN
              griot, # run GRIOT
              griotmulti, # run GRIOT multiple views (3 views for now : structure, attributes, spectral)
              griot2, # run GRIOTv2
              }
              """,
              type=str,
              show_default=True)
@click.option('--dir-weak',
              default="",
              help="""\b
              Directory path to baselines weaks.
              >
              """,
              type=str,
              show_default=True)
@click.option('--file-weak',
              default="skip",
              help="""\b
              File name of baselines weaks.
              >
              """,
              type=str,
              show_default=True)
@click.option('--dir-muzellec',
              default="",
              help="""\b
              Directory path to baseline muzellec.
              >
              """,
              type=str,
              show_default=True)
@click.option('--file-muzellec',
              default="skip",
              help="""\b
              File name of baseline muzellec.
              >
              """,
              type=str,
              show_default=True)
@click.option('--dir-rossi',
              default="",
              help="""\b
              Directory path to baseline rossi (feature propagation).
              >
              """,
              type=str,
              show_default=True)
@click.option('--file-rossi',
              default="skip",
              help="""\b
              File name of baseline rossi (feature propagation).
              >
              """,
              type=str,
              show_default=True)
@click.option('--dir-pagnn',
              default="",
              help="""\b
              Directory path to baseline PaGNN (Jiang et al.).
              >
              """,
              type=str,
              show_default=True)
@click.option('--file-pagnn',
              default="skip",
              help="""\b
              File name of baseline PaGNN (Jiang et al.).
              >
              """,
              type=str,
              show_default=True)
@click.option('--dir-griot',
              default="",
              help="""\b
              Directory path to griot file.
              >
              """,
              type=str,
              show_default=True)
@click.option('--file-griot',
              default="skip",
              help="""\b
              File name of griot file.
              >
              """,
              type=str,
              show_default=True)
@click.option('--dir-griotmulti',
              default="",
              help="""\b
              Directory path to griot_multi file.
              >
              """,
              type=str,
              show_default=True)
@click.option('--file-griotmulti',
              default="skip",
              help="""\b
              File name of griot_multi file.
              >
              """,
              type=str,
              show_default=True)
@click.option('--dir-griot2',
              default="",
              help="""\b
              Directory path to griot2 file.
              >
              """,
              type=str,
              show_default=True)
@click.option('--file-griot2',
              default="skip",
              help="""\b
              File name of griot2 file.
              >
              """,
              type=str,
              show_default=True)
@click.option('--normalization', '-n',
              default="unn",
              help="""\b
              normalization parameters, belongs to :
              {
              unn,  # p is uniform, F is not normalized, M_F and M_C are not normalized
              wnn,  # p is weighted, F is not normalized, M_F and M_C are not normalized
              urn,  # p is uniform, F is normalized, M_F and M_C are not normalized
              unr  # p is uniform, F is not normalized, M_F and M_C are normalized
              }.
              >
              """,
              type=str,
              show_default=True)
@click.option('--mecha', '-m',
              required=False,
              help="""\b
              Missing features mechanism, belongs to : 
              {
              MNAR,  # Missing Not At Random
              MAR,  # Missing At Random - **NOT IMPLEMENTED YET**
              MCAR  # Missing Completely At Random
              }.
              >
              """,
              type=str,
              default="MCAR",
              show_default=True)
@click.option('--opt', '-o',
              default="None",
              required=False,
              help="""\b
              For mecha = "MNAR", it indicates how the missing-data mechanism is generated, belongs to :
              {
              logistic,  # using a logistic regression
              quantile,  # quantile censorship
              selfmasked  # logistic regression for generating a self-masked MNAR mechanism
              }.
              >
              """,
              type=str,
              show_default=True)
@click.option('--unique',
              default="auto",
              help="""\b
              If True, add time and date to saved files.
              >
              """,
              type=str,
              show_default=True)
@click.option('--prefix',
              default="",
              help="""̄\b
              Prefix to place at the beginning of save folder path. 
              """,
              type=str,
              show_default=True)
@click.option('--lossfn', '-l',
              default="MultiW",
              help="""̄\b
              Loss function to account for training, belong to :
              {
              FGW,  # Fused Gromow Wasserstein
              MultiW,  # Multi-view Wasserstein (ours)
              }.
              """,
              type=str,
              show_default=True)
@click.option('--tildec', '-C',
              default=1,
              help="""̄\b
              tildec is used to know how to sub-sample the proximity matrix
              {
              <0, # take the square sub-matrix: C1 = C[idx1, :][:, idx1]
              0, # take the rectangular idx1, idx1 U idx2 sub-matrix: C1 = C[idx1, :][:, idx1 | idx2]
              >0, # take the rectangular full sub-matrix: C1 = C[idx1]
              }.
              """,
              type=int,
              show_default=True)
@click.option('--alphas', '-a',
              default="0.0,0.25,0.5,0.75",
              help="""̄\b
              Alpha parameter for MultiW loss function.
              All values must be between 0 and 1.
              Separated by commas.
              """,
              type=str,
              show_default=True)
@click.option("--epsilons", "-e",
              default="0.01",
              help="""̄\b
              Epsilon parameter for MultiW loss function.
              All values must be between 0 and 1.
              Separated by commas.
              """,
              type=str,
              show_default=True)
@click.option("--lr",
              default=0.001,
              help="""̄\b
                Learning rate for the optimizer.
                """,
              type=float,
              show_default=True)
@click.option("--p", "-p",
              default="2",
              help="""̄\b
              p is the parameter to set the lp-norm in the computation of the cost (distance) matrix.
              """,
              type=str,
              show_default=2)
@click.option('--nr',
              default=5,
              help="""̄\b
              Number or repetitions.
              Used fr computing average and error.
              """,
              type=int,
              show_default=True)
@click.option('--np',
              default=8,
              help="""̄\b
              Number of pairs.
              Number of time 2 sub-graphs are selected to computer the loss.
              """,
              type=int,
              show_default=True)
@click.option('--ni',
              default=None,
              help="""̄\b
              Number of iterations.
              Number of time the loss is back-propagated.
              Default is 250 for 'basic' algorithm, 25 for RR and GCN.
              """,
              type=int,
              show_default=True)
@click.option('--mi',
              default=None,
              help="""\b
              Max Iterations.
              Number of time model checks for convergence, compute the loss on the validation set, etc.
              Only used with RR and GCN.
              """,
              type=int,
              show_default=True)
@click.option('--epochs',
              default=375,
              help="""̄\b
              Number of epochs to train the model for.
              """,
              type=int,
              show_default=True)
@click.option('--report-interval',
              default=25,
              help="""\b
              Number of epochs between each report of MAE and RMSE.
              """,
              type=int,
              show_default=True)
@click.option('--batch-size', '-b',
              default=16,
              help="""\b
              Size of the sub-graphs.
              """,
              type=int,
              show_default=True)
@click.option('--pmiss',
              default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9",
              help="""\b
                Proportion of missing data.
                All values must be between 0 and 1.
                Separated by commas.
                """,
              type=str,
              show_default=True)
@click.option('--restructure', "-R",
              default="auto",
              help="""\b
              Restructure the graph to help homophily.
              auto : restructure if the graph is detected as non homophilic based on training data,
              True : force restructure (might degrade performance if the graph is already homophilic),
              False : do not restructure.
              """,
              type=str,
              show_default=True)
@click.option('--restruct-iters',
              default=25,
              help="""\b
              Number of iterations for the restructure algorithm.
              """,
              type=int,
              show_default=True)
@click.option('--restruct-thld',
              default=0.15,
              help="""\b
              Theshold to account for in the restructure algorithm.
              """,
              type=float,
              show_default=True)
@click.option('--restruct-thld-low',
              default=0.3,
              help="""\b
              Lower theshold to account for in the restructure algorithm.
              Only used with '--restruct-method increment'. 
              """,
              type=float,
              show_default=True)
@click.option('--restruct-method',
              default="increment",
              help="""\b
              Method to use in the restructure algorithm.
              "increment" : add links to the already existing structure.
              "remake" : make a new structure, discard the already existing one.  
              """,
              type=str,
              show_default=True)
@click.option('--restruct-integer',
              default=True,
              help="""\b
              Whether to round the new proximity matrix to integers or to keep floating numbers.
              If memory issues, better keep this to True.  
              """,
              type=bool,
              show_default=True)
@click.option('--restruct-for-loss',
              default=True,
              help="""\b
              Whether to use the new structure for computing the loss or to keep the original one.
              True : use the new structure in loss
              False : use the original structure in loss 
              """,
              type=bool,
              show_default=True)
@click.option('--use-ce',
              default=False,
              help="""\b
              Whether to use the CrossEntropy instead of the l2 norm for computing transport cost matrix of binary features.
              """,
              type=bool,
              show_default=True)
@click.option('--use-geomloss',
              default=True,
              help="""\b
              Whether to use the geomloss library instead of the POT library for computing transport cost in the loss function.
              """,
              type=bool,
              show_default=True)
@click.option('--seed',
              default=42,
              help="""\b
                Seed for the whole framework.
                """,
              type=int,
              show_default=True)
@click.option('--device',
              default=None,
              help="""\b
                Device to use for computation.
                """,
              type=str,
              show_default=True)
@click.option("--path-proximity",
              default=None,
              help="""\b
              /!\\ DEBUG ONLY /!\\
              Path to the proximity matrix.
              Overrides the proximity matrix path found in data_path.
              Use only with 1 graph at a time.
              """,
              type=str,
              show_default=True)
@click.option("--skip-classification", is_flag=True, help="If specified, only data imputation is performed.")
@click.option("--save-pred", is_flag=True, help="If specified, save the classifier's predictions.")
@click.option("--make-data", is_flag=True, help="Run preprocessing")
def main_(
        graph_name=None,
        model=None,
        num_heads=None,
        concat=None,
        dropout=None,
        plot=None,
        which=None,
        dir_weak=None,
        file_weak=None,
        dir_muzellec=None,
        file_muzellec=None,
        dir_rossi=None,
        file_rossi=None,
        dir_pagnn=None,
        file_pagnn=None,
        dir_griot=None,
        file_griot=None,
        dir_griotmulti=None,
        file_griotmulti=None,
        dir_griot2=None,
        file_griot2=None,
        normalization=None,
        mecha=None,
        opt=None,
        unique=False,
        prefix=None,
        lossfn=None,
        tildec=None,
        alphas=None,
        epsilons=None,
        lr=None,  # learn-rate
        p=None,  # lp-norm, default p=2
        nr=None,
        np=None,
        ni=None,
        mi=None,
        epochs=None,
        report_interval=None,
        batch_size=None,
        pmiss=None,
        seed=None,
        restructure=None,
        restruct_iters=None,
        restruct_thld=None,
        restruct_thld_low=None,
        restruct_method=None,
        restruct_integer=None,
        restruct_for_loss=None,
        use_ce=None,
        use_geomloss=None,
        device=None,
        path_proximity=None,
        skip_classification=None,
        save_pred=None,
        make_data=False,
):
    from .config import cfg

    # add all the parameters of main() to the config with a for loop
    for k, v in locals().items():
        if k != "cfg":
            cfg[k] = v
    cfg.seed_all()
    cfg.logger.info("\n\n\n\n\n-----INITIALIZATION-----\n")
    cfg.logger.info(cfg)
    cfg.logger.info("\n"+"-"*50+"\n\n\n\n")

    if cfg.make_data:
        from .make_datasets import preprocess
        preprocess.main()
    else:
        from .run.main import main
        try:
            main()
        except ValueError:
            raise
        finally:
            try:
                cfg.sumup()
            except FileNotFoundError:
                cfg.logger.info("\n"*10 + "config : " + str(cfg) + "\n"*10)
            cfg.logger.info("FINALLY - DONE.")


main_()
