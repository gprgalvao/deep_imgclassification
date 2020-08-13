# Modo de separação dos dados

(TREINO + VALIDAÇÃO) + TESTE = 100%

# Criando ambiente para processamento:
```
module load anaconda3
conda create -n nome_do_ambiente python=3.7
source activate nome_do_ambiente
```

# Bibliotecas instaladas via conda:
```
conda install -c conda-forge keras
conda install -c conda-forge scikit-learn
conda install -c conda-forge pandas
conda install -c conda-forge seaborn
```

### Extras
```
conda install -c conda-forge scikit-image
conda install -c conda-forge opencv
conda install -c conda-forge scipy==1.1.0
```

# Tarefas (scripts) para execusão (usar uma vez e aguardar o término do processo):
```
bash super_tarefas.sh
sbatch super_tarefas.sh	#apenas para cluster
```

# Execusao em segundo plano (usar uma vez e aguardar o término do processo):
```
nohup bash super_tarefas.sh &
```

# Desenvolvedores
| [<img src="https://media-exp1.licdn.com/dms/image/C4D03AQHMAHymw99pVw/profile-displayphoto-shrink_100_100/0?e=1602720000&v=beta&t=oHIYd5wNz2GnDnaHCQO79qzJRB-0SpbNzDTFRCRKPQg" width=115><br><sub>Gustavo Ubeda</sub>](https://github.com/gustavo-ubeda) |  [<img src="https://media-exp1.licdn.com/dms/image/C4D03AQFW4Ti9QSuzdw/profile-displayphoto-shrink_100_100/0?e=1602720000&v=beta&t=detY-YmffbkPf_lWEoYzTOXwSJF2cTDxwJyroLwwi-E" width=115><br><sub>André Rossi</sub>](https://github.com/andrerossidc) |
| :---: | :---: 
