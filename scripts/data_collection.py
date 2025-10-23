import requests
from requests.adapters import HTTPAdapter, Retry
import json
import re

# Setup session with retry
retries = Retry(total=5, backoff_factor=0.25, status_forcelist=[500, 502, 503, 504])
session = requests.Session()
session.mount("https://", HTTPAdapter(max_retries=retries))

# Pagination handler
def get_next_link(headers):
    if "Link" in headers:
        match = re.search(r'<(.+)>; rel="next"', headers["Link"])
        if match:
            return match.group(1)

def get_batch(batch_url):
    while batch_url:
        response = session.get(batch_url)
        response.raise_for_status()
        total = response.headers.get("x-total-results", "?")
        yield response, total
        batch_url = get_next_link(response.headers)

# Extract required fields for TSV output
def extract_function_for_negatives(entry):
    accession = entry.get("primaryAccession", "Unknown")
    org_name = entry.get("organism", {}).get("scientificName", "Unknown")
    lineage = entry.get("organism", {}).get("lineage", [])
    tmp_kingdom = lineage[1] if len(lineage) > 1 else "Other"
    if tmp_kingdom in ['Metazoa', 'Fungi', 'Viridiplantae']:
        kingdom = tmp_kingdom
    else :
        kingdom = 'other'
    length = entry.get("sequence", {}).get("length", "?")

    # Check for transmembrane helix in first 90 residues
    has_tm = False
    for feature in entry["features"]:
        if feature["type"] == "Transmembrane":
            start = feature["location"]["start"]["value"]
            if isinstance(start, int) and start <= 90:
                has_tm = True
                break

    return accession, org_name, kingdom, length, str(has_tm)

# Main dataset builder (no filter function needed)
def get_negative_dataset(search_url, extractor, output_dir, prefix):
    filtered_json = []
    n_total = 0
    for batch, total in get_batch(search_url):
        batch_json = json.loads(batch.text)
        for entry in batch_json.get("results", []):
            n_total += 1
            filtered_json.append(entry)
    print(f"Total entries retrieved: {n_total}")
    with open(f"{output_dir}/{prefix}.tsv", "w") as ofs:
        for entry in filtered_json:
            fields = extractor(entry)
            print(*fields, sep="\t", file=ofs)

    write_fasta(filtered_json, f"{output_dir}/{prefix}.fasta")


def write_fasta(entries, output_fasta_file):
    with open(output_fasta_file, "w") as f:
        for entry in entries:
            accession = entry.get("primaryAccession", "Unknown")
            org_name = entry.get("organism", {}).get("scientificName", "Unknown")
            sequence = entry.get("sequence", {}).get("value", "")
            if sequence:
                header = f">{accession} {org_name}"
                f.write(header + "\n")
                for i in range(0, len(sequence), 60):
                    f.write(sequence[i:i+60] + "\n")


# Filtering based on SP-specific criteria
def filter_function(entry):
    for feature in entry.get("features", []):
        if feature["type"] in ["Signal", "Signal peptide"]:
            begin = feature.get("location", {}).get("start", {}).get("value")
            end = feature.get("location", {}).get("end", {}).get("value")

            if (not end) or (not isinstance(end, int)):
                return False

            # Check Single Peptide length
            if (end - begin) < 13:
                continue

            for ev in feature.get("evidences", []):
                if ev.get("evidenceCode") == "ECO:0000269":
                    return True
    return False


# Extract required fields for TSV
def extract_function_for_positives(entry):
    accession = entry.get("primaryAccession", 'Unknown')
    org_name = entry.get("organism", {}).get("scientificName", "Unknown")

    kingdom_list = entry.get("organism", {}).get("lineage", [])
    tmp_kingdom = kingdom_list[1]
    if tmp_kingdom in ['Metazoa', 'Fungi', 'Viridiplantae']:
        kingdom = tmp_kingdom
    else:
        kingdom = 'other'

    length = entry.get("sequence", {}).get("length", "?")

    cleavage = "?"
    for feature in entry["features"]:
        if feature["type"] in ["Signal", "Signal peptide"]:
            cleavage = feature.get("location", {}).get("end", "?").get("value", "?")
            break
    return (accession, org_name, kingdom, length, cleavage)


# Main dataset builder
def get_positive_dataset(search_url, filter_function, extract_function, output_dir, prefix):
    filtered_json = []
    n_total, n_filtered = 0, 0
    for batch, total in get_batch(search_url):
        batch_json = json.loads(batch.text)
        for entry in batch_json.get("results", []):
            n_total += 1
            if filter_function(entry):
                n_filtered += 1
                filtered_json.append(entry)
    print(f"Total entries: {n_total}, Filtered: {n_filtered}")
    with open(f"{output_dir}/{prefix}.tsv", "w") as ofs:
        for entry in filtered_json:
            fields = extract_function(entry)
            print(*fields, sep="\t", file=ofs)

    write_fasta(filtered_json, f"{output_dir}/{prefix}.fasta")



