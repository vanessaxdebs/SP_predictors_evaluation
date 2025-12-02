import os
import subprocess
from config import config

def exec_mmseqs_easy_cluster(
        input_dir,
        output_dir,
        prefix,
        min_seq_id,
        coverage_threshold,
):
    subprocess.run(f"rm -rf {output_dir}/{prefix} \
                    && mkdir {output_dir}/{prefix} \
                    && cd {output_dir}/{prefix} \
                    && mmseqs easy-cluster {input_dir}/{prefix}.fasta cluster-results output --min-seq-id {min_seq_id} -c {coverage_threshold} --cov-mode 0 --cluster-mode 1 \
                    && cd ..",
                   shell=True,
                   text=True)

if __name__ == "__main__":
    os.makedirs(config.config["data_preparation_dir"], exist_ok=True)

    exec_mmseqs_easy_cluster(
        config.config["data_collection_dir_inside_output"],
        config.config["data_preparation_dir"],
        config.config["positive_prefix"],
        config.config["min_seq_id"],
        config.config["coverage_threshold"],
    )

    exec_mmseqs_easy_cluster(
        config.config["data_collection_dir_inside_output"],
        config.config["data_preparation_dir"],
        config.config["negative_prefix"],
        config.config["min_seq_id"],
        config.config["coverage_threshold"],
    )