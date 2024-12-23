# **Code and Data for "How Natural Sequence Variation Modulates Protein Folding Dynamics"**

## **Description**
This repository contains code and data associated with the article:  
**"How Natural Sequence Variation Modulates Protein Folding Dynamics"**  
**Authors**: Ezequiel A. Galpern, Ernesto A. Roman, Diego U. Ferreiro  
**DOI**: [10.48550/arXiv.2412.14341](https://doi.org/10.48550/arXiv.2412.14341)

---

## **Notebooks**

A demonstration Jupyter Notebook is available for use in Google Colab. This notebook allows users to simulate the folding Ising model for any sequence in the Multiple Sequence Alignment or mutants and visualize the results for the 15 studied protein families.

[![Open in Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/eagalpern/folding-ising-globular/blob/master/notebooks/simulation_colab.ipynb)

---

## **Source Code**

The repository includes Python scripts to:  
- Run folding simulations.  
- Visualize and analyze results.  

---

## **Data**

The following data is available for 15 protein families:  
- **PDB files**: Tertiary structure files provided as `.pdb` files.  
- **DSSP files**: Secondary structure detail tables provided as `.csv` files.  
- **Table S1**: Summary of protein family details.  
- **Multi results**: Simulation results for 500 sequences for each family.

---

## **Extended Data**

Additional data is hosted in the linked Zenodo repository:  
- **/simplified_rbm_and_msa/**: Contains a folder for each of the 15 protein families. Each folder includes:  
  - A **.fasta** file for the multiple sequence alignment (MSA) of the protein family.  
  - A **.npz** file containing the Potts model, with local fields (`h`) and couplings (`J`), saved in NumPy format.  

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14547875.svg)](https://doi.org/10.5281/zenodo.14547875)

---

### **How to Use This Repository**
1. Clone the repository locally or open the Colab notebook directly using the badge above.  
2. Access the extended data from Zenodo for detailed MSA and Potts model files.  

Feel free to contribute or report any issues in the repository's **Issues** section.

