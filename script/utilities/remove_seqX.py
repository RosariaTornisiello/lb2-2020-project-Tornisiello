import sys
from Bio import SeqIO
f = open("clean_woX.fasta", "w")
for record in SeqIO.parse(sys.argv[1], "fasta"):
    if record.seq.count('X') == 0:
        print(record.format("fasta"), file = f)
