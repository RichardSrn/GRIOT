def data_path(graph_name: str) -> dict:
    sbm = {
        "name": "SBM",
        "proximity_matrix": "./data/artificial/SBM_2/proximity_matrix.npy",
        "features": {
            "F_true": "./data/artificial/SBM_2/F_true_fit01.npy"
        }
    }

    sbm0 = {
        "name": "SBM3",
        "proximity_matrix": "./data/artificial/SBM_0/proximity_matrix.npy",
        "features": {
            "F_true": "./data/artificial/SBM_0/F_true_fit01.npy"
        }
    }

    sbm1 = {
        "name": "SBM1",
        "proximity_matrix": "./data/artificial/SBM_1/proximity_matrix.npy",
        "features": {
            "F_true": "./data/artificial/SBM_1/F_true_fit01.npy"
        }
    }

    sbm2 = {
        "name": "SBM2",
        "proximity_matrix": "./data/artificial/SBM_2/proximity_matrix.npy",
        "features": {
            "F_true": "./data/artificial/SBM_2/F_true_fit01.npy"
        }
    }

    texas = {
        "name": "TEXAS",
        "proximity_matrix": "./data/real/WKB/texas/proximity_matrix.npy",
        "features": {
            "F_true": "./data/real/WKB/texas/F_true.npy"
        }
    }

    cornell = {
        "name": "CORNELL",
        "proximity_matrix": "./data/real/WKB/cornell/proximity_matrix.npy",
        "features": {
            "F_true": "./data/real/WKB/cornell/F_true.npy"
        }
    }

    wisconsin = {
        "name": "WISCONSIN",
        "proximity_matrix": "./data/real/WKB/wisconsin/proximity_matrix.npy",
        "features": {
            "F_true": "./data/real/WKB/wisconsin/F_true.npy"
        }
    }

    cora = {
        "name": "CORA",
        "proximity_matrix": "./data/real/PLANETOID/Cora/proximity_matrix.npy",
        "features": {
            "F_true": "./data/real/PLANETOID/Cora/F_true.npy"
        }
    }

    citeseer = {
        "name": "CITESEER",
        "proximity_matrix": "./data/real/PLANETOID/CiteSeer/proximity_matrix.npy",
        "features": {
            "F_true": "./data/real/PLANETOID/CiteSeer/F_true.npy"
        }
    }

    pubmed = {
        "name": "PUBMED",
        "proximity_matrix": "./data/real/PLANETOID/PubMed/proximity_matrix.npy",
        "features": {
            "F_true": "./data/real/PLANETOID/PubMed/F_true.npy"
        }
    }




    data = {
        'sbm0': sbm0,
        'sbm': sbm,
        'sbm1': sbm1,
        'sbm2': sbm2,
        'texas': texas,
        'cornell': cornell,
        'wisconsin': wisconsin,
        'cora': cora,
        'citeseer': citeseer,
        'pubmed': pubmed
    }

    return data[graph_name.lower()]
