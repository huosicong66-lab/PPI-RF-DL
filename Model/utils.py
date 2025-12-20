from Bio import SeqIO

def load_fasta_sequences(fasta_path):
    from Bio import SeqIO
    seq_dict = {}
    for record in SeqIO.parse(fasta_path, "fasta"):
        seq_id = record.id.split('.')[0]
        seq_dict[seq_id] = str(record.seq)
    return seq_dict

