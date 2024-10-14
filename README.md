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

### Deep Learn

TBD

### Redes Convolucionais

TBD

### Fuzzy Sistems

TBD

### Aprendizagem não Supervisionada

TBD

### SOM

