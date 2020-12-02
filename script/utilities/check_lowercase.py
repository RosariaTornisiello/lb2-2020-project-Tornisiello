import sys
from Bio import SeqIO
f = open("output_lowercase.txt", "w")
for seq_record in SeqIO.parse(sys.argv[1], "fasta"):
    sequence = str(seq_record.seq)
    if sequence.islower() == True:
        print(sequence, file = f)
