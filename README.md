#Modo de separação dos dados

▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
▒▒▒▒▒▒▒▒▒BANCO DE DADOS▒▒▒▒▒▒▒▒▒▒
▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
▒▒░░░░░░░░░░░░░░░░░░░░▒▒▓▓▓▓▓▓▓▒▒
▒▒░TREINO + VALIDAÇÃO░▒▒▓TESTE▓▒▒
▒▒░░░░░░░░░░░░░░░░░░░░▒▒▓▓▓▓▓▓▓▒▒
▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒

(TREINO + VALIDAÇÃO) + TESTE = 100%

#Criando ambiente para processamento:
module load anaconda3
conda create -n nome_do_ambiente python=3.7
source activate nome_do_ambiente

#Bibliotecas instaladas via conda:
conda install -c conda-forge keras
conda install -c conda-forge scikit-learn
conda install -c conda-forge pandas
conda install -c conda-forge seaborn

conda install -c conda-forge scikit-image
conda install -c conda-forge opencv
conda install -c conda-forge scipy==1.1.0

#Tarefas (scripts) para execusão (usar uma vez e aguardar o término do processo):
bash super_tarefas.sh
sbatch super_tarefas.sh	#apenas para cluster

#Execusao em segundo plano (usar uma vez e aguardar o término do processo):
nohup bash super_tarefas.sh &
# deep_imgclassification
