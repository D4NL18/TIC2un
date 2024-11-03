# Projeto Tópicos Avançados em Inteligência Computacional - 2° Unidade
## Centro Universitário SENAI CIMATEC
### Daniel André Marinho, Felipe Ribeiro, Natália Ohana


## Requisitos

A ideia original da atividade é desenvolver códigos de algoritmos de classificação. Cada equipe apresentará um tipo de classificação, além de criar um roteiro de como utilizar o algoritmo apresentado. Com isso, cada equipe deve seguir o roteiro para aplicar os conhecimentos obtidos nas aulas.

## Solução

De acordo com o problema apresentado acima, foram desenvolvidas 6 soluções em Python, referentes a cada algoritmo apresentado em sala. Como adicional, a equipe desenvolveu um ambiente de execução utilizando Flask, com o objetivo de enviar os resultados obtidos em cada código para um frontend, desenvolvido com o Framework Angular. O repositório contém tanto o frontend quanto o backend, separados em suas respectivas pastas dentro da root principal do projeto. Para visualizar cada algoritmo individualmente, acesse a pasta do backend e busque pelo algoritmo desejado seguindo o nome dos arquivos

## Algoritmos

### SVM

O primeiro algoritmo solicitado foi o SVM (Máquinas de Vetores-Suporte). A classificação ocorreu na base de dados Iris, da biblioteca Sklearn, e aplicou a função SVC da mesma biblioteca para realizar a classificação, após o tratamento adequado dos dados. O algoritmo completo será executado dentro de um Post, que receberá o comando do frontend para execução. Os resultados serão enviados para o frontend através de duas requisições Get, sendo uma para a imagem da matriz confusão e outra para os resultados de precisão do algoritmo.

![Demo](backend/results/svm.gif)

### Deep Learning

O segundo método apresentado foi o Deep Learning, no qual foram solicitadas 2 aplicações diferentes, sendo uma delas através da biblioteca PyTorch e outra utilizando TensorFlow. A partir disso, foi realizado o treinamento de ambos os algoritmos para a classificação da biblioteca Iris, e apresentado na tela do sistema a matriz confusão e a accuracy de ambos, a fins comparativos.

![Demo](backend/results/DL.gif)

### CNN

O terceiro roteiro solicitava um algoritmo de CNN, que deveria incluir 2 arquivos, sendo um deles responsável pelo treinamento da rede e armazenamento dos pesos obtidos, e o segundo deve ler estes pesos e executar o algoritmo com base nisso. O primeiro arquivo algoritmo pode ser visto no repositório em backend/CNN/CNN_treino.py, e o segundo, em backend/CNN/CNN_teste.py. A partir do modelo de pesos obtido com a execução do primeiro arquivo, foi desenvolvida uma segunda versão do arquivo CNN_teste.py, inserindo flask para que haja a integração com o frontend. Esta versão foi inserida no arquivo backend.py, que representa o backend geral da aplicação. Com isso, foi criada uma requisição post, que solicita o teste utilizando os pesos, e duas requisições get, que enviam a matriz confusão e as métricas do algoritmo para o frontend.

![Demo](backend/results/CNN.gif)

### Fuzzy Sistems

TBD

### Aprendizagem não Supervisionada

TBD

### SOM

