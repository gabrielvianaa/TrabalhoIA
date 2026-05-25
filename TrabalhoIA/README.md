
TRABALHO DE INTELIGÊNCIA ARTIFICIAL — PREDIÇÃO DE DOENÇA CARDÍACA
Comparação de Naive Bayes, Árvore de Decisão e Random Forest em R e Python

INTEGRANTES DO GRUPO
--------------------
- Gabriel Viana de Carvalho
- Frederico Augusto

Engenharia de Software / Inteligência Artificial

-----------------------------

BASE DE DADOS
-------------
Heart Disease (UCI - Cleveland), versão processada do Kaggle.
Versão utilizada: https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset
Arquivo: dados/heart.csv (303 registros únicos, 14 atributos)

COMO EXECUTAR
-------------

>> PYTHON (versão 3.12.10)

1. Instalar dependências:
   pip install pandas numpy scikit-learn matplotlib seaborn

2. Entrar na pasta codigo/

3. Garantir que o arquivo heart.csv esteja na mesma pasta:
   cp ../dados/heart.csv .

4. Executar:
   python trabalho_heart_disease.py

Saída: resultados_python.csv e arquivos .png na pasta atual.

>> R (versão 4.0 ou superior, recomendado usar RStudio)

1. Instalar pacotes (apenas na primeira execução):
   install.packages(c("caret", "e1071", "rpart", "rpart.plot",
                      "randomForest", "pROC", "ggplot2", "dplyr"))

2. Abrir trabalho_heart_disease.R no RStudio

3. Garantir que heart.csv esteja na mesma pasta do script

4. Executar:
   Menu Session > Set Working Directory > To Source File Location
   Em seguida, clicar em "Source" (ou Ctrl+Shift+S)

Saída: resultados_r.csv e arquivos .png na pasta atual.
