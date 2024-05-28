# Reconstructing the Unseen: Attributed Graph Imputation with Optimal Transport

This code is the implementation of the paper **Reconstructing the Unseen: Attributed Graph Imputation with Optimal Transport**.

The paper is under review for AISTATS 2024.

The implementation here features our proposed framework multi-
**W**assertsein for imput**A**tion of **G**raph with **N**ode
**A**ttributes : <tt>WaGNA</tt>.

# General Information

- Track Name: AISTATS2024
- Paper ID: 498
- Paper Title: Reconstructing the Unseen: Attributed Graph Imputation with Optimal Transport
- Primary Subject Area: General Machine Learning - Learning on Graphs
- Secondary Subject Areas:  Applications - Network Analysis

# Abstract:

In recent years, there has been a significant surge in machine learning techniques, particularly in the domain of deep learning, tailored for handling attributed graphs. Nevertheless, in order to work, these methods assume that the attributes are fully known, which is not realistic in numerous real-world applications. This paper addresses the challenge of missing attributes on graphs' nodes leveraging the Optimal Transport (OT) theory. To proceed, we first design a new multi-view OT loss function that can encompass both node feature data and the underlying topological structure of the graph by utilizing multiple graph representations. We then utilize this novel loss to train efficiently a Graph Convolutional Neural Network (GCN) architecture capable of imputing all missing values over the graph at once. We evaluate the interest of our approach with experiments both on synthetic data and real-world graphs, including different missingness mechanisms and a wide range of missing data. These experiments demonstrate that our method is competitive with the state-of-the-art in all cases and of particular interest on weakly homophilic graphs.

# Code Specific

Code usage: `python -m WaGNA [OPTIONS]`

## To create the datasets
To download or synthesize the datasets: `python3 -m WaGNA --make-data`.

## Main explored hyperparameters
- `--which ours --model {basic,rr,gcn} --pmiss 0.2,0.5,0.8 --mecha {MCAR,MNAR} --opt {None,selfmasked} --graphs-name {sbm,cornell,texas,wisconsin,cora,citeseer,pubmed} --nr 5 --batch-size 16 --np 8`  
- `--which weak --pmiss 0.2,0.5,0.8 --mecha {MCAR,MNAR} --opt {None,selfmasked} --graphs-name {sbm,cornell,texas,wisconsin,cora,citeseer,pubmed} --nr 5 --batch-size 16 --np 8`
- `--which rossi --model fp --pmiss 0.2,0.5,0.8 --mecha {MCAR,MNAR} --opt {None,selfmasked} --graphs-name {sbm,cornell,texas,wisconsin,cora,citeseer,pubmed} --nr 5 --batch-size 16 --np 8`
- `--which muzellec --model {basic,fp} --pmiss 0.2,0.5,0.8 --mecha {MCAR,MNAR} --opt {None,selfmasked} --graphs-name {sbm,cornell,texas,wisconsin,cora,citeseer,pubmed} --nr 5 --batch-size 16 --np 8`
- `--which pagnn --model {rand,fp,gcn} --pmiss 0.2,0.5,0.8 --mecha {MCAR,MNAR} --opt {None,selfmasked} --graphs-name {sbm,cornell,texas,wisconsin,cora,citeseer,pubmed} --nr 5 --batch-size 16 --np 8`

## Examples of use with different scenarios :
- our model with 20%, 50%, and 80% missing data, MCAR mechanism, Cora, and CiteSeer datasets, on 375 epochs with a report of MAE and RMSE every 25 epochs, batch_size=16, n_pairs=8, and 5 runs (for averaging the results):

`python3 -m WaGNA --which ours --model gcn --pmiss 0.2,0.5,0.8 --mecha MCAR --graphs-name cora,citeseer --epochs 375 --report-interval 25 --nr 5 --batch-size 16 --np 8`

- weak baselines with 20% missing data, MNAR (selfmasked) mechanism, SBM dataset, on 375 epochs with a report of MAE and RMSE every 25 epochs, and 5 runs (for averaging the results):

`python3 -m WaGNA --which weak --pmiss 0.2 --mecha MNAR --opt selfmasked --graphs-name sbm --epochs 375 --report-interval 25 --nr 5`

- Rossi et al. FP baseline with 10% to 90% for a step of 20% missing data, MCAR mechanism, Texas, Cornell, and Wisconsin (WebKB) dataset, on 375 epochs with a report of MAE and RMSE every 25 epochs, and 5 runs (for averaging the results):

`python3 -m WaGNA --which rossi --model fp --pmiss 0.1,0.3,0.5,0.7,0.9 --mecha MCAR --graphs-name wkb --epochs 375 --report-interval 25 --nr 5`

- Muzellec et al. FP baseline with 50% missing data, MCAR mechanism, Cora, CiteSeer, and PubMed (Planetoid) dataset, on 375 epochs with a report of MAE and RMSE every 25 epochs, and 5 runs (for averaging the results):

`python3 -m WaGNA --which muzellec --model basic --pmiss 0.5 --mecha MCAR --graphs-name planetoid --epochs 375 --report-interval 25 --nr 5`

## Definition of the main parameters
One can use `python3 -m WaGNA --help` for an exhaustive description of the parameters.

```python
--make-data # to download or synthesize the datasets
-G, --graphs-name # name of graph(s) to run the algorithm on
-W, --which # which part of the code to run (ours, Rossi et al., Muzellec et al., weak baselines)
-M, --model # name of the model to use (do not specify for weak baselines)
--pmiss # amount of missing data to generate
-m, --mecha # missing mechanism to use (MCAR, MNAR)
-o, --opt # mask to use with MNAR, only selfmasked is implemented for now
--epochs # number of epochs
--report-interval # number of epochs between two reports of MAE and RMSE 
-b, --batch-size # number of nodes sampled for each batch (default: 16)
--np # number of pairs of subgraphs to estimate the distance from, see Algorithm 1 (default: 8)
```

# Datasets source

- SBM is synthesized using the Stochastic-Block-Model algorithm: https://networkx.org/documentation/stable/reference/generated/networkx.generators.community.stochastic_block_model.html
- WebKB (Texas, Cornell, Wisconsin): "Geom-GCN: Geometric Graph Convolutional Networks" - Pei et al. - 2020 - https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.WebKB.html
- Planetoid --citation networks-- (Cora, CiteSeer, PubMed): "Revisiting Semi-Supervised Learning with Graph Embeddings" - Yang et al. - 2016 - https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.Planetoid.html#torch_geometric.datasets.Planetoid
