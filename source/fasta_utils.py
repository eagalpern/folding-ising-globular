import numpy as np
import os
from Bio import SeqIO


def get_fasta_file(directory):
    # Iterate over files in the directory
    for filename in os.listdir(directory):
        # Check if the file has a .fasta extension
        if filename.endswith(".fasta"):
            return filename  # Return the first .fasta file found
    return None  # Return None if no .fasta file is found


def extract_sequence_from_fasta(uniprot_id, fasta_file):
        found_sequence = None
        with open(fasta_file, "r") as handle:
            for record in SeqIO.parse(handle, "fasta"):
                if uniprot_id in record.description:
                    found_sequence = record.seq
                    break  # Stop searching after finding the first match (if there are multiple)

        return found_sequence


def load_fasta(fasta_file):
        fasta_sequences = list(SeqIO.parse(fasta_file,'fasta'))
        MSA=np.empty([len(fasta_sequences),len(fasta_sequences[0].seq)],dtype='<U1')
        names=[]
        w_list=[]
        for j in range(len(fasta_sequences)):
            MSA[j,:]=[i for i in fasta_sequences[j].seq]
            names.append(fasta_sequences[j].id)
            w_list.append(float(fasta_sequences[j].description.split(' ')[1]))
        name_list=[names[i].split('.')[0] for i in range(len(names))]
        ini=[int(names[i].split('/')[1].split('-')[0]) for i in range(len(names))]
        fin=[int(names[i].split('/')[1].split('-')[1]) for i in range(len(names))]
        weights=np.array(w_list)
        return  MSA, weights,name_list
    

def load_simple_fasta(fasta_file):
        fasta_sequences = list(SeqIO.parse(fasta_file,'fasta'))
        MSA=np.empty([len(fasta_sequences),len(fasta_sequences[0].seq)],dtype='<U1')
        names=[]
        for j in range(len(fasta_sequences)):
            MSA[j,:]=[i for i in fasta_sequences[j].seq]
            names.append(fasta_sequences[j].id)
        return  MSA, names
    
    
def freq(ali,npos,Naa,w):
    fi=np.zeros((npos,Naa))
    for i in range(npos):
        fi[i,:]=np.histogram(ali[:,i],bins=np.arange(-0.5,Naa+0.5),weights=w)[0]
    fi=fi/sum(w)
    return fi

# def str_to_save(x):
#     xstr=''
#     for i in range(len(x)):
#         xstr=xstr+str(x[i])+ '\t'
#     return xstr
    
    




