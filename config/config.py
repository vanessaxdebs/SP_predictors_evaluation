# We use config hash in order to maintain all parameters in one place
# TODO: read config from .yaml file or .env
config = {
    "output_root_dir": "./data",
    "data_collection_dir_inside_output": "../../data_collection",
    "data_collection_dir": "./data/data_collection",
    "data_preparation_dir": "./data/data_preparation",
    "data_analysis_dir": "./data/data_analysis",
    "pswm_dir": "./data/pswm",
    "svm_dir": "./data/svm",
    "ffnn_dir": "./data/ffnn",
    "results_analysis_dir": "./data/results_analysis",
    "negative_URL": "https://rest.uniprot.org/uniprotkb/search?format=json&query=%28reviewed%3Atrue%29%20AND%20%28fragment%3Afalse%29%20AND%20%28taxonomy_id%3A2759%29%20AND%20%28length%3A%5B40%20TO%20%2A%5D%29%20AND%20%28cc_scl_term_exp%3ASL-0091%20OR%20cc_scl_term_exp%3ASL-0039%20OR%20cc_scl_term_exp%3ASL-0173%20OR%20cc_scl_term_exp%3ASL-0209%20OR%20cc_scl_term_exp%3ASL-0204%20OR%20cc_scl_term_exp%3ASL-0191%29%20NOT%20%28ft_signal%3A%2A%29%20AND%20existence%3A1&size=100",
    "negative_prefix": "negative",
    "positive_URL": "https://rest.uniprot.org/uniprotkb/search?format=json&query=%28existence%3A1%29%20AND%20%28reviewed%3Atrue%29%20AND%20%28fragment%3Afalse%29%20AND%20%28length%3A%5B40%20TO%20%2A%5D%29%20AND%20%28ft_signal_exp%3A%2A%29%20AND%20%28taxonomy_id%3A2759%29&size=100",
    "positive_prefix": "positive",
    "coverage_threshold": 0.4,
    "min_seq_id": 0.3,
}