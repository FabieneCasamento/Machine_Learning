#####################################################################
#                         INSTALAÇÃO DO GIT                         #
#####################################################################

#instala o git
sudo apt install git

#configuração de usuário => email cadastrado no github
git config --global user.name "Fulano de Tal"
git config --global user.email fulanodetal@exemplo.br

#listar configs
git config --list

############################################################
                    ### GITHUB ###
############################################################                
# CONECTAR COM GITHUB

#ADICIONAR REPOSITÓRIO PARA INTALAR CLI GITHUB
(type -p wget >/dev/null || (sudo apt update && sudo apt-get install wget -y)) \
&& sudo mkdir -p -m 755 /etc/apt/keyrings \
&& wget -qO- https://cli.github.com/packages/githubcli-archive-keyring.gpg | sudo tee /etc/apt/keyrings/githubcli-archive-keyring.gpg > /dev/null \
&& sudo chmod go+r /etc/apt/keyrings/githubcli-archive-keyring.gpg \
&& echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null \
&& sudo apt update \
&& sudo apt install gh -y

#UPDATE E INSTALL DA CLI
sudo apt update
sudo apt install gh

#PARA LOGAR VIA HTTPS OU SSH
gh auth login

#############################################################
#aperece um código para validar o github



# no meu caso
#CUIDADO – NUNCA DAR git add .
# pode deletar o que já está no git e não está no computador

#Terminal

#$ git clone https://github.com/FabieneCasamento/Machine_Learning.git
#$ git branch -a
# * main
#   remotes/origin/HEAD -> origin/main
#   remotes/origin/fix2
#   remotes/origin/main
#   remotes/origin/new_branch_fix2

# $ git checkout new_branch_fix2
# Branch 'new_branch_fix2' set up to track remote branch 'new_branch_fix2' from 'origin'.
# Switched to a new branch 'new_branch_fix2'

# $ git branch
#   main
# * new_branch_fix2



# no terminal colocar novamente sue usuário do github
# $ git config --global user.email "nome@ygmail.com"
#$ git config --global user.name "nomenologin"



# $ git add Machine_Learning_AZ/Exer1_processo/M3_Exercicio_classificacao_pre_processamento_dos_dados.ipynb 
#$ git commit -m 'exercicio processo ML'
# $ git push origin new_branch_fix2

# Apenas para este projeto de Machine Learning:
#$ git config http.postBuffer 300000000

# Para todos os seus projetos no computador (Recomendado se usa muitos datasets):
#git config --global http.postBuffer 300000000

