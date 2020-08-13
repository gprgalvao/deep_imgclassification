#!/bin/bash

#______________________________________________________________________________
#
#	Executar com: bash instalador_de_bibliotecas.sh
#______________________________________________________________________________

echo "Qual o nome do seu novo ambiente? "
read nome_do_ambiente;

#Criando novo ambiente para processamento:
module load anaconda3
conda create -n $nome_do_ambiente python=3.7
source activate $nome_do_ambiente

#Bibliotecas via conda (https://anaconda.org/conda-forge/repo), (-y confirma etapas):
conda install -c conda-forge keras -y
conda install -c conda-forge scikit-learn -y
conda install -c conda-forge pandas -y
conda install -c conda-forge seaborn -y

conda install -c conda-forge scikit-image -y
conda install -c conda-forge opencv -y
conda install -c conda-forge scipy==1.1.0 -y

echo "\n----------Instalação concluída!----------"
echo "Use sempre: \n  module load anaconda3 \n  source activate $nome_do_ambiente \npara acessar seu ambiente de programação em python"
