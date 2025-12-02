from scripts.data_collection import (get_negative_dataset,
                                    get_positive_dataset,
                                     filter_function,
                                     extract_function_for_positives,
                                     extract_function_for_negatives)

from scripts.data_preparation import exec_mmseqs_easy_cluster
from scripts.pswm import compute_pswm

# TODO config from .yaml?
output_dir = "./data"
# TODO check URLS
negative_URL = "https://rest.uniprot.org/uniprotkb/search?format=json&query=%28reviewed%3Atrue%29%20AND%20%28fragment%3Afalse%29%20AND%20%28taxonomy_id%3A2759%29%20AND%20%28length%3A%5B40%20TO%20%2A%5D%29%20AND%20%28cc_scl_term_exp%3ASL-0091%20OR%20cc_scl_term_exp%3ASL-0039%20OR%20cc_scl_term_exp%3ASL-0173%20OR%20cc_scl_term_exp%3ASL-0209%20OR%20cc_scl_term_exp%3ASL-0204%20OR%20cc_scl_term_exp%3ASL-0191%29%20NOT%20%28ft_signal%3A%2A%29%20AND%20existence%3A1&size=100"
negative_prefix = "negative"
positive_URL = "https://rest.uniprot.org/uniprotkb/search?format=json&query=%28existence%3A1%29%20AND%20%28reviewed%3Atrue%29%20AND%20%28fragment%3Afalse%29%20AND%20%28length%3A%5B40%20TO%20%2A%5D%29%20AND%20%28ft_signal_exp%3A%2A%29%20AND%20%28taxonomy_id%3A2759%29&size=100"
positive_prefix = "positive"

coverage_threshold = 0.4
min_seq_id = 0.3

if __name__ == '__main__':
    # get_negative_dataset(negative_URL, extract_function_for_negatives, output_dir, negative_prefix)
    # get_positive_dataset(positive_URL, filter_function, extract_function_for_positives, output_dir, positive_prefix)

    # Cluster using mmseqs2
    # exec_mmseqs_easy_cluster(
    #     output_dir,
    #     negative_prefix,
    #     min_seq_id,
    #     coverage_threshold
    # )
    #
    # exec_mmseqs_easy_cluster(
    #     output_dir,
    #     positive_prefix,
    #     min_seq_id,
    #     coverage_threshold
    # )

    # Example usage:
    pswm_df = compute_pswm(
        output_dir,
        positive_prefix,
    )
    print(pswm_df.round(2))
    pswm_df.to_csv(f"{output_dir}/PSWM_signal_peptides.tsv", sep="\t")