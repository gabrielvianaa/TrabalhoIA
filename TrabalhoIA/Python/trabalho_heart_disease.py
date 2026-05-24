
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)

SEED = 42
np.random.seed(SEED)

df = pd.read_csv('heart.csv')
col_description = {
    'age': 'idade em anos',
    'sex': 'sexo (1 = masculino, 0 = feminino)',
    'cp': 'tipo de dor no peito (0-3)',
    'trestbps': 'pressão arterial em repouso (mm Hg)',
    'chol': 'colesterol sérico (mg/dl)',
    'fbs': 'glicemia em jejum > 120 mg/dl (1 = verdadeiro, 0 = falso)',
    'restecg': 'resultado do eletrocardiograma em repouso (0-2)',
    'thalach': 'frequência cardíaca máxima atingida',
    'exang': 'angina induzida por exercício (1 = sim, 0 = não)',
    'oldpeak': 'depressão do segmento ST induzida pelo exercício',
    'slope': 'inclinação do segmento ST no pico do exercício (0-2)',
    'ca': 'número de vasos principais coloridos por fluoroscopia (0-4)',
    'thal': 'resultado do teste de tálio (0-3)',
    'target': 'presença de doença cardíaca (1 = sim, 0 = não)'
}
print("Glossário:")
for col in df.columns:
    descricao = col_description.get(col, 'Descrição não encontrada')
    print(f"  {col}: {descricao}")
print("Dimensões do dataset:", df.shape)
print("\nPrimeiras linhas:")
print(df.head())
print("\nTipos de dados:")
print("\nEstatísticas descritivas:")
print(df.describe())
print("\nDistribuição da variável alvo:")
print(df['target'].value_counts(normalize=True))

print("\nValores ausentes por coluna:")
print(df.isnull().sum())

print(f"\nLinhas antes de remover duplicatas: {df.shape[0]}")
df = df.drop_duplicates().reset_index(drop=True)
print(f"Linhas depois de remover duplicatas: {df.shape[0]}")

X = df.drop('target', axis=1)
y = df['target']

print("\nFormato de X:", X.shape)
print("Formato de y:", y.shape)
print("Distribuição da classe alvo:", np.bincount(y))

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,        
    random_state=SEED,    
    stratify=y            
)

print(f"\nTreino: {X_train.shape}, Teste: {X_test.shape}")
print(f"Distribuição treino: {np.bincount(y_train)}")
print(f"Distribuição teste:  {np.bincount(y_test)}")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

def avaliar_modelo(nome, modelo, X_tr, y_tr, X_te, y_te):

    inicio = time.time()
    modelo.fit(X_tr, y_tr)
    tempo_treino = time.time() - inicio

    y_pred = modelo.predict(X_te)
    y_proba = modelo.predict_proba(X_te)[:, 1]

    acc = accuracy_score(y_te, y_pred)
    prec = precision_score(y_te, y_pred)
    rec = recall_score(y_te, y_pred)
    f1 = f1_score(y_te, y_pred)
    auc = roc_auc_score(y_te, y_proba)
    cm = confusion_matrix(y_te, y_pred)

    print(f"\n{'='*60}")
    print(f"Modelo: {nome}")
    print(f"{'='*60}")
    print(f"Tempo de treinamento: {tempo_treino:.4f} segundos")
    print(f"Acurácia:    {acc:.4f}")
    print(f"Precisão:    {prec:.4f}")
    print(f"Revocação:   {rec:.4f}")
    print(f"F1-score:    {f1:.4f}")
    print(f"AUC:         {auc:.4f}")
    print(f"Matriz de Confusão:\n{cm}")
    print(f"\nRelatório completo:\n{classification_report(y_te, y_pred, target_names=['Sem doença', 'Com doença'])}")

    return {
        'modelo': nome,
        'acuracia': acc,
        'precisao': prec,
        'revocacao': rec,
        'f1_score': f1,
        'auc': auc,
        'tempo_treino': tempo_treino,
        'matriz_confusao': cm,
        'y_pred': y_pred,
        'y_proba': y_proba
    }

resultados = []

nb = GaussianNB()
resultados.append(avaliar_modelo("Naive Bayes (Python)", nb,
                                   X_train_scaled, y_train,
                                   X_test_scaled, y_test))

arvore = DecisionTreeClassifier(random_state=SEED, max_depth=5,
                                 min_samples_split=10)
resultados.append(avaliar_modelo("Árvore de Decisão (Python)", arvore,
                                   X_train, y_train, X_test, y_test))

plt.figure(figsize=(20, 10))
plot_tree(arvore, feature_names=X.columns.tolist(),
          class_names=['Sem doença', 'Com doença'],
          filled=True, rounded=True, fontsize=9)
plt.title('Árvore de Decisão - Heart Disease (Python)')
plt.savefig('arvore_decisao_python.png', dpi=150, bbox_inches='tight')
plt.close()

rf = RandomForestClassifier(n_estimators=100, random_state=SEED,
                            max_depth=10, n_jobs=-1)
resultados.append(avaliar_modelo("Random Forest (Python)", rf,
                            X_train, y_train, X_test, y_test))

print("\n" + "="*60)
print("AJUSTE DE HIPERPARÂMETROS - Random Forest")
print("="*60)

param_grid = {
    'n_estimators': [50, 100, 200, 300],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid = GridSearchCV(
    RandomForestClassifier(random_state=SEED, n_jobs=-1),
    param_grid,
    cv=5,
    scoring='f1',
    n_jobs=-1
)
grid.fit(X_train, y_train)
print(f"Melhores parâmetros: {grid.best_params_}")
print(f"Melhor F1 (validação cruzada): {grid.best_score_:.4f}")

rf_otimizado = grid.best_estimator_
resultado_otimizado = avaliar_modelo("Random Forest Otimizado (Python)",
                            rf_otimizado, X_train, y_train,
                            X_test, y_test)
resultados.append(resultado_otimizado)

importancias = pd.DataFrame({
    'variavel': X.columns,
    'importancia': rf.feature_importances_
}).sort_values('importancia', ascending=False)

print("\nImportância das variáveis (Random Forest):")
print(importancias.to_string(index=False))

plt.figure(figsize=(10, 6))
sns.barplot(data=importancias, x='importancia', y='variavel',
            hue='variavel', palette='viridis', legend=False)
plt.title('Importância das Variáveis - Random Forest (Python)')
plt.xlabel('Importância')
plt.ylabel('Variável')
plt.tight_layout()
plt.savefig('importancia_variaveis_python.png', dpi=150)
plt.close()

df_resultados = pd.DataFrame([
    {
        'Modelo': r['modelo'],
        'Acurácia': r['acuracia'],
        'Precisão': r['precisao'],
        'Revocação': r['revocacao'],
        'F1-Score': r['f1_score'],
        'AUC': r['auc'],
        'Tempo (s)': r['tempo_treino']
    }
    for r in resultados
])

print("\n" + "="*60)
print("TABELA COMPARATIVA FINAL (Python)")
print("="*60)
print(df_resultados.to_string(index=False))

df_resultados.to_csv('resultados_python.csv', index=False)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, r in zip(axes, resultados[:3]):
    sns.heatmap(r['matriz_confusao'], annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Sem doença', 'Com doença'],
                yticklabels=['Sem doença', 'Com doença'])
    ax.set_title(r['modelo'])
    ax.set_xlabel('Previsto')
    ax.set_ylabel('Real')
plt.tight_layout()
plt.savefig('matrizes_confusao_python.png', dpi=150)
plt.close()

plt.figure(figsize=(8, 6))
for r in resultados[:3]:
    fpr, tpr, _ = roc_curve(y_test, r['y_proba'])
    plt.plot(fpr, tpr, label=f"{r['modelo']} (AUC={r['auc']:.3f})")
plt.plot([0, 1], [0, 1], 'k--', label='Aleatório')
plt.xlabel('Taxa de Falso Positivo')
plt.ylabel('Taxa de Verdadeiro Positivo')
plt.title('Curvas ROC - Heart Disease (Python)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('curvas_roc_python.png', dpi=150)
plt.close()

metricas = ['Acurácia', 'Precisão', 'Revocação', 'F1-Score', 'AUC']
df_plot = df_resultados.set_index('Modelo')[metricas]
df_plot.plot(kind='bar', figsize=(11, 6))
plt.title('Comparação de Métricas - Heart Disease (Python)')
plt.ylabel('Valor')
plt.xticks(rotation=20, ha='right')
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('comparacao_metricas_python.png', dpi=150)
plt.close()

print("\nGráficos salvos:")
print("  - matrizes_confusao_python.png")
print("  - curvas_roc_python.png")
print("  - comparacao_metricas_python.png")
print("  - arvore_decisao_python.png")
print("  - importancia_variaveis_python.png")
print("\nResultados salvos em: resultados_python.csv")
print("\nExecução em Python concluída!")
