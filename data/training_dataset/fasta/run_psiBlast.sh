#!/bin/bash
for file in $(ls /home/rosaria/Desktop/LAB2/LAB2_project/fasta/);
do
	string=$file
	id=${string:0:-6}	
	psiblast -query $file -db ../uniprot_sprot.fasta -evalue 0.01 -num_iterations 3 -out_ascii_pssm ../psiBlast_outputs/$id".pssm" -num_descriptions 10000 -num_alignments 10000 -out ../psiBlast_outputs/$id".alns.blast"
done
