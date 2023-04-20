# Readily-interpretable deep learning translation of GWAS and multi-omics findings to understand pathobiology and drug repurposing in Alzheimer's disease

Human genome sequencing studies have identified numerous loci associated with complex diseases, including Alzheimer’s disease (AD). However, translating human genetic findings (i.e., genome-wide association studies [GWAS]) to pathobiology and therapeutic discovery remains a major challenge. We present a network topology-based deep learning framework to identify disease-associated genes (NETTAG). NETTAG integrates multi-genomic data and the protein-protein interactome to infer putative risk genes and drug targets implicated by GWAS loci. We leverage non-coding GWAS loci effects on quantitative trait loci, enhancers and CpG islands, promoter regions, open chromatin, and promoter flanking regions. By applying NETTAG to the latest AD GWAS data, we identified 156 potential AD-risk genes that were: 1) significantly enriched in AD-related pathobiological pathways, 2) differentially expressed with respect to the brain transcriptome and proteome, and 3) enriched in druggable targets with approved medicines. Combining network-based prediction and retrospective case-control observations with 10 million individuals, we identified that usage of four drugs (ibuprofen, gemfibrozil, cholecalciferol, and ceftriaxone) is associated with reduced likelihood of AD incidence, after adjusting for various confounding factors. Importantly, gemfibrozil (an approved lipid regulator) is significantly associated with 43% reduced risk of AD compared to simvastatin (another approved anti-lipid medicine under Phase II AD trials), using an active user design (95% confidence interval 0.51-0.63, P<0.0001). In summary, NETTAG offers a deep learning methodology that utilizes GWAS and multi-genomic findings to identify pathobiology and drug repurposing in AD.

## Requirments:
* Python 3.7
* pytorch=1.11.0
* scipy=1.6.3
* numpy=1.20.2
* networkx=2.5.1
* sklearn=0.24.2
* pyhocon=0.3.59


## Installation

* conda create -n nettag python=3.7

* conda activate nettag

* pip install torch==1.11.0+cu102 torchvision==0.12.0+cu102 torchaudio==0.11.0+cu102 -f https://download.pytorch.org/whl/cu102/torch_stable.html

* pip install networkx

* pip install spicy

* pip install -U scikit-learn

* pip install pandas

* pip install pyhocon

## Usage

cd ./code/

./human_350k_run_dev.sh

Default parameters in the human_350k_run_dev.sh and main.py file are the ones used in the manuscript.

In the current ./human_350k_run_dev.sh file, in order to improve prediction stability, we ensemble results with 10 different seed.

For the sake of convenience, users can provide your own directories of functional genomics data (e.g., regulatory element or xQTL) into config file (see our example config file 'functional_genomics.conf' in data folder).
