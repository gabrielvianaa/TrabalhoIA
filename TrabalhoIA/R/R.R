
install.packages(c("caret", "e1071", "rpart", "rpart.plot",
                    "randomForest", "pROC", "ggplot2", "dplyr"))

library(caret)        
library(e1071)         
library(rpart)         
library(rpart.plot)    
library(randomForest)  
library(pROC)          
library(ggplot2)       
library(dplyr)         

SEED <- 42
set.seed(SEED)

# DICIONÁRIO DE VARIÁVEIS:
#   age        - idade em anos
#   sex        - sexo (1 = masculino, 0 = feminino)
#   cp         - tipo de dor no peito (0-3)
#   trestbps   - pressão arterial em repouso (mm Hg)
#   chol       - colesterol sérico (mg/dl)
#   fbs        - glicemia em jejum > 120 mg/dl (1 = verdadeiro, 0 = falso)
#   restecg    - eletrocardiograma em repouso (0-2)
#   thalach    - frequência cardíaca máxima atingida
#   exang      - angina induzida por exercício (1 = sim, 0 = não)
#   oldpeak    - depressão do segmento ST induzida pelo exercício
#   slope      - inclinação do segmento ST no pico do exercício (0-2)
#   ca         - número de vasos principais coloridos por fluoroscopia (0-4)
#   thal       - resultado do teste de tálio (0-3)
#   target     - presença de doença cardíaca (1 = sim, 0 = não)  *** ALVO ***

df <- read.csv("heart.csv", stringsAsFactors = FALSE)

cat("Dimensões do dataset:", dim(df), "\n")
cat("\nPrimeiras linhas:\n")
print(head(df))
cat("\nEstrutura:\n")
str(df)
cat("\nEstatísticas descritivas:\n")
print(summary(df))
cat("\nDistribuição da variável alvo:\n")
print(table(df$target) / nrow(df))


cat("\nValores ausentes por coluna:\n")
print(colSums(is.na(df)))

cat("\nLinhas antes de remover duplicatas:", nrow(df), "\n")
df <- df[!duplicated(df), ]
cat("Linhas depois de remover duplicatas:", nrow(df), "\n")

df$target <- factor(df$target, levels = c(0, 1),
                    labels = c("Sem_doenca", "Com_doenca"))

cat("\nDistribuição após conversão em factor:\n")
print(table(df$target))

set.seed(SEED)
indice_treino <- createDataPartition(df$target, p = 0.7, list = FALSE)
treino <- df[indice_treino, ]
teste  <- df[-indice_treino, ]

cat("\nTreino:", nrow(treino), "registros\n")
cat("Teste:",  nrow(teste),  "registros\n")
cat("Distribuição treino:\n"); print(table(treino$target))
cat("Distribuição teste:\n");  print(table(teste$target))

treino_features <- treino[, -which(names(treino) == "target")]
treino_target   <- treino$target
teste_features  <- teste[, -which(names(teste) == "target")]
teste_target    <- teste$target

cat("\nVerificação de tamanhos:\n")
cat("  treino_features:", nrow(treino_features), "x", ncol(treino_features), "\n")
cat("  teste_features: ", nrow(teste_features),  "x", ncol(teste_features),  "\n")
cat("  treino_target:  ", length(treino_target),  "valores\n")
cat("  teste_target:   ", length(teste_target),   "valores\n")

preProc <- preProcess(treino_features, method = c("center", "scale"))

treino_scaled_features <- predict(preProc, treino_features)
teste_scaled_features  <- predict(preProc, teste_features)

cat("\nPadronização aplicada. Exemplo — média de 'age' após padronização:\n")
cat("  Treino:", round(mean(treino_scaled_features$age), 4),
    "| Teste:", round(mean(teste_scaled_features$age),  4), "\n")

avaliar_modelo <- function(nome, modelo, features_teste, target_teste,
                           tempo_treino, probas_positivas) {
  
  if (inherits(modelo, "rpart")) {
    y_pred <- predict(modelo, features_teste, type = "class")
  } else {
    y_pred <- predict(modelo, features_teste)
  }
  
  y_pred <- factor(y_pred, levels = levels(target_teste))
  
  if (length(y_pred) != length(target_teste)) {
    stop(paste("ERRO: y_pred tem", length(y_pred),
               "elementos mas target_teste tem", length(target_teste)))
  }
  

  cm <- confusionMatrix(y_pred, target_teste, positive = "Com_doenca")
  
  # AUC
  auc_valor <- as.numeric(auc(roc(target_teste, probas_positivas, quiet = TRUE)))
  
  acc  <- cm$overall["Accuracy"]
  prec <- cm$byClass["Precision"]
  rec  <- cm$byClass["Recall"]
  f1   <- cm$byClass["F1"]
  
  cat("\n", strrep("=", 60), "\n", sep = "")
  cat("Modelo:", nome, "\n")
  cat(strrep("=", 60), "\n")
  cat("Tempo de treinamento:", round(tempo_treino, 4), "segundos\n")
  cat("Acurácia:    ", round(acc, 4),  "\n")
  cat("Precisão:    ", round(prec, 4), "\n")
  cat("Revocação:   ", round(rec, 4),  "\n")
  cat("F1-score:    ", round(f1, 4),   "\n")
  cat("AUC:         ", round(auc_valor, 4), "\n")
  cat("Matriz de Confusão:\n")
  print(cm$table)
  
  list(
    modelo          = nome,
    acuracia        = as.numeric(acc),
    precisao        = as.numeric(prec),
    revocacao       = as.numeric(rec),
    f1_score        = as.numeric(f1),
    auc             = auc_valor,
    tempo_treino    = tempo_treino,
    matriz_confusao = cm$table,
    y_pred          = y_pred,
    y_proba         = probas_positivas
  )
}

resultados <- list()

set.seed(SEED)
t_inicio <- Sys.time()
modelo_nb <- naiveBayes(target ~ ., data = cbind(treino_scaled_features,
                                                  target = treino_target))
t_fim    <- Sys.time()
tempo_nb <- as.numeric(difftime(t_fim, t_inicio, units = "secs"))

probas_nb <- predict(modelo_nb, teste_scaled_features, type = "raw")[, "Com_doenca"]

resultados$nb <- avaliar_modelo(
  nome             = "Naive Bayes (R)",
  modelo           = modelo_nb,
  features_teste   = teste_scaled_features,
  target_teste     = teste_target,
  tempo_treino     = tempo_nb,
  probas_positivas = probas_nb
)

set.seed(SEED)
t_inicio  <- Sys.time()
modelo_arvore <- rpart(target ~ .,
                       data    = cbind(treino_features, target = treino_target),
                       method  = "class",
                       control = rpart.control(maxdepth  = 5,
                                               minsplit  = 10,
                                               cp        = 0.001))
t_fim     <- Sys.time()
tempo_arv <- as.numeric(difftime(t_fim, t_inicio, units = "secs"))

probas_arv <- predict(modelo_arvore, teste_features, type = "prob")[, "Com_doenca"]

resultados$arv <- avaliar_modelo(
  nome             = "Árvore de Decisão (R)",
  modelo           = modelo_arvore,
  features_teste   = teste_features,
  target_teste     = teste_target,
  tempo_treino     = tempo_arv,
  probas_positivas = probas_arv
)

png("arvore_decisao_r.png", width = 1400, height = 900, res = 120)
rpart.plot(modelo_arvore, type = 3, extra = 104, fallen.leaves = TRUE,
           main = "Árvore de Decisão - Heart Disease (R)")
dev.off()
cat("Gráfico salvo: arvore_decisao_r.png\n")

set.seed(SEED)
t_inicio <- Sys.time()
modelo_rf <- randomForest(x           = treino_features,
                           y           = treino_target,
                           ntree       = 100,
                           maxnodes    = 30,
                           importance  = TRUE)
t_fim    <- Sys.time()
tempo_rf <- as.numeric(difftime(t_fim, t_inicio, units = "secs"))

probas_rf <- predict(modelo_rf, teste_features, type = "prob")[, "Com_doenca"]

resultados$rf <- avaliar_modelo(
  nome             = "Random Forest (R)",
  modelo           = modelo_rf,
  features_teste   = teste_features,
  target_teste     = teste_target,
  tempo_treino     = tempo_rf,
  probas_positivas = probas_rf
)

cat("\n", strrep("=", 60), "\n", sep = "")
cat("AJUSTE DE HIPERPARÂMETROS - Random Forest\n")
cat(strrep("=", 60), "\n")

set.seed(SEED)
controle <- trainControl(method          = "cv",
                         number          = 5,
                         classProbs      = TRUE,
                         summaryFunction = twoClassSummary)

grid_rf <- expand.grid(mtry = c(2, 3, 4, 6, 8))

treino_completo <- cbind(treino_features, target = treino_target)

rf_grid <- train(target ~ .,
                 data      = treino_completo,
                 method    = "rf",
                 trControl = controle,
                 tuneGrid  = grid_rf,
                 ntree     = 200,
                 metric    = "ROC")

cat("Melhor mtry:", rf_grid$bestTune$mtry, "\n")
print(rf_grid)

t_inicio       <- Sys.time()
probas_rf_otim <- predict(rf_grid, teste_features, type = "prob")[, "Com_doenca"]
t_fim          <- Sys.time()
tempo_rf_otim  <- as.numeric(difftime(t_fim, t_inicio, units = "secs"))

resultados$rf_otim <- avaliar_modelo(
  nome             = "Random Forest Otimizado (R)",
  modelo           = rf_grid,
  features_teste   = teste_features,
  target_teste     = teste_target,
  tempo_treino     = tempo_rf_otim,
  probas_positivas = probas_rf_otim
)

cat("\nImportância das variáveis (Random Forest):\n")
print(importance(modelo_rf))

png("importancia_variaveis_r.png", width = 900, height = 600, res = 120)
varImpPlot(modelo_rf, main = "Importância das Variáveis - Random Forest (R)")
dev.off()
cat("Gráfico salvo: importancia_variaveis_r.png\n")

df_resultados <- data.frame(
  Modelo    = sapply(resultados, function(r) r$modelo),
  Acuracia  = sapply(resultados, function(r) round(r$acuracia,  4)),
  Precisao  = sapply(resultados, function(r) round(r$precisao,  4)),
  Revocacao = sapply(resultados, function(r) round(r$revocacao, 4)),
  F1_Score  = sapply(resultados, function(r) round(r$f1_score,  4)),
  AUC       = sapply(resultados, function(r) round(r$auc,       4)),
  Tempo_s   = sapply(resultados, function(r) round(r$tempo_treino, 4)),
  row.names = NULL
)

cat("\n", strrep("=", 60), "\n", sep = "")
cat("TABELA COMPARATIVA FINAL (R)\n")
cat(strrep("=", 60), "\n")
print(df_resultados, row.names = FALSE)

write.csv(df_resultados, "resultados_r.csv", row.names = FALSE)
cat("\nResultados salvos em: resultados_r.csv\n")

png("curvas_roc_r.png", width = 900, height = 700, res = 120)
plot(roc(teste_target, probas_nb,  quiet = TRUE),
     col = "blue",      lwd = 2,
     main = "Curvas ROC - Heart Disease (R)")
plot(roc(teste_target, probas_arv, quiet = TRUE),
     col = "darkgreen", lwd = 2, add = TRUE)
plot(roc(teste_target, probas_rf,  quiet = TRUE),
     col = "red",       lwd = 2, add = TRUE)
legend("bottomright",
       legend = c(
         paste0("Naive Bayes (AUC = ",    round(resultados$nb$auc,  3), ")"),
         paste0("Árvore (AUC = ",         round(resultados$arv$auc, 3), ")"),
         paste0("Random Forest (AUC = ",  round(resultados$rf$auc,  3), ")")
       ),
       col = c("blue", "darkgreen", "red"), lwd = 2)
dev.off()
cat("Gráfico salvo: curvas_roc_r.png\n")

png("comparacao_metricas_r.png", width = 1000, height = 600, res = 120)
df_long <- reshape(
  df_resultados[, c("Modelo", "Acuracia", "Precisao", "Revocacao", "F1_Score", "AUC")],
  varying   = c("Acuracia", "Precisao", "Revocacao", "F1_Score", "AUC"),
  v.names   = "Valor",
  timevar   = "Metrica",
  times     = c("Acurácia", "Precisão", "Revocação", "F1-Score", "AUC"),
  direction = "long"
)
print(
  ggplot(df_long, aes(x = Modelo, y = Valor, fill = Metrica)) +
    geom_bar(stat = "identity", position = "dodge") +
    theme_minimal() +
    labs(title = "Comparação de Métricas - Heart Disease (R)",
         y = "Valor", x = "") +
    theme(axis.text.x = element_text(angle = 20, hjust = 1)) +
    scale_fill_brewer(palette = "Set2")
)
dev.off()
cat("Gráfico salvo: comparacao_metricas_r.png\n")

cat("\n", strrep("=", 60), "\n", sep = "")
cat("Execução em R concluída com sucesso!\n")
cat(strrep("=", 60), "\n")
