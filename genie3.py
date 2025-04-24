import os
import argparse
import pandas as pd
import numpy as np
import scanpy as sc
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from joblib import Parallel, delayed, parallel_backend
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_gene_list(gene_list_path):
    df = pd.read_csv(gene_list_path, header=None)
    gene_list = df.iloc[:, 0].dropna().astype(str).tolist()
    logging.info(f"Loaded {len(gene_list)} TF genes from {gene_list_path}")
    return gene_list

def compute_importances(target_gene, predictors, target, tree_model, max_features, n_trees):
    try:
        model = tree_model(n_estimators=n_trees, max_features=max_features, random_state=42)
        model.fit(predictors, target)
        importances = model.feature_importances_
        return {predictors.columns[i]: importances[i] for i in range(len(predictors.columns))}
    except Exception as e:
        logging.error(f"Error computing importances for {target_gene}: {e}")
        return {}

def genie3_weight_matrix(expr_data, tree_method="GB", K="sqrt", n_trees=100, n_threads=8, regulators=None, targets=None):
    genes = expr_data.columns.tolist()
    regulators = regulators if regulators is not None else genes
    targets = targets if targets is not None else genes

    weight_matrix = pd.DataFrame(
        np.zeros((len(regulators), len(targets)), dtype=np.float32),
        index=regulators,
        columns=targets
    )

    tree_model = {
        "RF": RandomForestRegressor,
        "ET": ExtraTreesRegressor,
        "GB": GradientBoostingRegressor
    }.get(tree_method.upper(), None)

    if not tree_model:
        raise ValueError("Invalid tree method. Choose 'RF', 'ET', or 'GB'.")

    max_features = K if isinstance(K, int) else ("sqrt" if K == "sqrt" else None)

    with parallel_backend("threading"):
        results = Parallel(n_jobs=n_threads)(
            delayed(compute_importances)(
                target_gene, expr_data[regulators], expr_data[target_gene], tree_model, max_features, n_trees
            )
            for target_gene in targets
        )

    for target_gene, result in zip(targets, results):
        for regulator, importance in result.items():
            weight_matrix.loc[regulator, target_gene] = np.float32(importance)

    for gene in weight_matrix.index.intersection(weight_matrix.columns):
        weight_matrix.loc[gene, gene] = 0

    return weight_matrix

def get_sorted_links(weight_matrix, max_links=None):
    links = weight_matrix.stack().reset_index()
    links.columns = ["RegulatoryGene", "TargetGene", "Weight"]
    links = links[links["Weight"] > 0].sort_values(by="Weight", ascending=False)
    if max_links:
        links = links.head(max_links)
    return links

def save_results(weight_matrix, sorted_links, output_dir, file_prefix):
    os.makedirs(output_dir, exist_ok=True)
    weight_path = os.path.join(output_dir, f"genie3_weight_matrix_{file_prefix}.csv")
    link_path = os.path.join(output_dir, f"genie3_sorted_links_{file_prefix}.csv")
    weight_matrix.to_csv(weight_path)
    sorted_links.to_csv(link_path, index=False)
    logging.info(f"Saved:\n - {weight_path}\n - {link_path}")

def main():
    parser = argparse.ArgumentParser(description="GENIE3-based gene regulatory network inference")
    parser.add_argument("--input", required=True, help="Path to input .h5ad file")
    parser.add_argument("--tf-list", required=True, help="Path to CSV file containing TF gene list")
    parser.add_argument("--output-dir", required=True, help="Directory to save output CSVs")
    parser.add_argument("--normalize", action="store_true", help="Apply normalization and log1p before HVG selection")
    parser.add_argument("--tree-method", default="GB", choices=["RF", "ET", "GB"], help="Tree model to use (default: GB)")
    parser.add_argument("--max-links", type=int, default=None, help="Maximum number of top links to output")
    parser.add_argument("--n-hvg", type=int, default=2000, help="Number of highly variable genes to use")

    args = parser.parse_args()

    # 파일 이름 접두어 추출
    file_prefix = os.path.splitext(os.path.basename(args.input))[0]

    # 데이터 로딩
    logging.info(f"Reading {args.input}")
    adata = sc.read_h5ad(args.input)

    # 정규화 및 HVG 추출
    if args.normalize:
        logging.info("Applying normalization and log1p before HVG selection...")
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
    else:
        logging.info("Skipping normalization...")

    sc.pp.highly_variable_genes(adata, n_top_genes=args.n_hvg, flavor="seurat")
    hvg_genes = adata.var[adata.var.highly_variable].index.tolist()
    expr_data = adata.to_df()[hvg_genes]

    # TF 리스트 로딩 및 유효 TF 필터링
    tf_genes = load_gene_list(args.tf_list)
    valid_tfs = list(set(tf_genes) & set(hvg_genes))
    if not valid_tfs:
        raise ValueError("No valid TFs found among HVGs.")

    logging.info(f"Using {len(valid_tfs)} TFs (regulators) and {len(hvg_genes)} HVGs (targets)")

    # GENIE3 실행
    weight_matrix = genie3_weight_matrix(
        expr_data,
        tree_method=args.tree_method,
        K="sqrt",
        n_trees=100,
        n_threads=8,
        regulators=valid_tfs,
        targets=hvg_genes
    )

    sorted_links = get_sorted_links(weight_matrix, max_links=args.max_links)

    # 결과 저장
    save_results(weight_matrix, sorted_links, args.output_dir, file_prefix)

if __name__ == "__main__":
    main()
