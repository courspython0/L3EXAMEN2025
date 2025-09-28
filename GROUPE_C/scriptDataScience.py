import pandas as pd  # manipulation et analyse de données
from sklearn.model_selection import train_test_split  # séparation train/test
from sklearn.preprocessing import StandardScaler  # normalisation des variables
from sklearn.linear_model import LogisticRegression  # modèle de régression logistique
import matplotlib.pyplot as plt  # visualisation graphique
import seaborn as sns  # visualisation avancée avec heatmaps, boxplots, etc.
from sklearn.ensemble import RandomForestClassifier  # modèle Random Forest
from sklearn.tree import DecisionTreeClassifier  # modèle arbre de décision
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, roc_auc_score  # métriques d'évaluation


df = pd.read_csv("SMSSpamCollection", sep="\t", header=None, names=["étiquette", "message"])

print(df.head())

# variables clés
df["longueur_message"] = df["message"].apply(len)
df["nombre_mots"] = df["message"].apply(lambda x: len(x.split()))
df["nombre_chiffres"] = df["message"].apply(lambda x: sum(c.isdigit() for c in x))
df["nombre_majuscules"] = df["message"].apply(lambda x: sum(1 for c in x if c.isupper()))
mots_cles = ["free", "win", "urgent", "call", "prize","money"]
for mot in mots_cles:
    df[f"a_{mot}"] = df["message"].apply(lambda x: int(mot in x.lower()))

# compter les caractères de chaque ligne
# print('----------------------------------------------------')
print(df["longueur_message"])

# vérification des valeurs manquantes
print('valeurs manquantes')
print(df.isnull().sum())

# NETTOYAGE DES DONNÉES

# supprimer les lignes vides
df.dropna(inplace=True)

# supprimer les espaces inutiles
df["message"] = df["message"].str.strip()

# nettoyer la cible
df["étiquette"] = df["étiquette"].str.strip().str.lower()
print(df["étiquette"].value_counts())

# print('------------------------------------------------')
# print(df["message"][8])

# normalisation
scaler = StandardScaler()
colonnes_a_normaliser = ["longueur_message", "nombre_mots", "nombre_chiffres", "nombre_majuscules"]
df[colonnes_a_normaliser] = scaler.fit_transform(df[colonnes_a_normaliser])

# encoder les variables 
df["étiquette_num"] = df["étiquette"].map({"ham": 0, "spam": 1})

# séparation des textes et des trains 
# X = variables explicatives (features)
# y = variable cible (label)
X = df[["longueur_message", "nombre_mots", "nombre_chiffres", "nombre_majuscules"]]
y = df["étiquette_num"]

# Séparation train/test : 80% entraînement, 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------------------exploration et visualisation--------------------------------

# Taille du dataset et types des colonnes
print(df.shape)
print(df.info())

print(df.head())

# statistiques des colonnes
print(df.describe())

print(df["étiquette"].value_counts())

# sous forme de pourcentage
print(df["étiquette"].value_counts(normalize=True) * 100)

# visualisation avec histplot
sns.histplot(df["longueur_message"], bins=30)
plt.show()

# visualisation avec boxplot
sns.boxplot(x="étiquette", y="longueur_message", data=df)
plt.title("Distribution de la longueur des messages par étiquette")
plt.show()

# visualisation avec heatmaps
corr = df[["longueur_message", "nombre_mots", "nombre_chiffres", "nombre_majuscules"]].corr()

# Heatmap
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Corrélation entre les variables numériques")
plt.show()

# CORRÉLATION
colonnes_numeriques = ["longueur_message", "nombre_mots", "nombre_chiffres", "nombre_majuscules"]
df_numerique = df[colonnes_numeriques]

# Corrélation entre toutes les colonnes numériques
corr_matrix = df_numerique.corr()
print(corr_matrix)

# -----------------------------------------Modélisation-------------------------

# création du modèle pour la régression logistique
modele_rl = LogisticRegression(max_iter=1000)

# Entraînement sur les données d'entraînement
modele_rl.fit(X_train, y_train)

# Entraînement sur les données test
y_pred = modele_rl.predict(X_test)

# visualiser la prédiction
# Accuracy
print("Précision:", accuracy_score(y_test, y_pred))

# Rapport détaillé
print(classification_report(y_test, y_pred))

# -----------------------------------------arbre de décisions-------------------------

# Création du modèle
modele_ad = DecisionTreeClassifier(random_state=42)

# Entraînement sur le jeu d'entraînement
modele_ad.fit(X_train, y_train)

# Prédictions
y_pred = modele_ad.predict(X_test)
print('---------------------------------------affichage arbre------------------------------------')
# Accuracy
print("Précision:", accuracy_score(y_test, y_pred))

# Rapport de classification
print(classification_report(y_test, y_pred))

print('---------------------------------------random forest-----------------------------------')

# Création du modèle
modele_rf = RandomForestClassifier(n_estimators=100, random_state=42)

# Entraînement
modele_rf.fit(X_train, y_train)

# Prédictions
y_pred = modele_rf.predict(X_test)

# Accuracy
print("Précision:", accuracy_score(y_test, y_pred))

# Rapport détaillé
print(classification_report(y_test, y_pred))

print('---------------------------------------matrice de confusion-----------------------------------')

# Création de la matrice de confusion
matrice_confusion = confusion_matrix(y_test, y_pred)

# Visualisation avec seaborn
plt.figure(figsize=(6, 4))
sns.heatmap(matrice_confusion, annot=True, fmt="d", cmap="Blues", xticklabels=["Classe 0", "Classe 1"], yticklabels=["Classe 0", "Classe 1"])
plt.xlabel("Prédictions")
plt.ylabel("Valeurs réelles")
plt.title("Matrice de confusion - Random Forest")
plt.show()

#  Rapport détaillé : précision, rappel, f1-score
print("Rapport de classification :")
print(classification_report(y_test, y_pred))
print('---------------------------------------courbe roc-----------------------------------')

# Exemple avec Random Forest (tu peux remplacer par modele_rl ou modele_ad)
y_pred_proba = modele_rf.predict_proba(X_test)[:, 1]

# Calcul des points ROC
fpr, tpr, seuils = roc_curve(y_test, y_pred_proba)

# Aire sous la courbe (AUC)
roc_auc = auc(fpr, tpr)

# Tracé de la courbe ROC
plt.figure(figsize=(6,6))
plt.plot(fpr, tpr, color="blue", lw=2, label=f"Courbe ROC (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], color="red", linestyle="--")  # diagonale hasard
plt.xlabel("Taux de faux positifs (FPR)")
plt.ylabel("Taux de vrais positifs (TPR)")
plt.title("Courbe ROC - Détection Spam/Ham")
plt.legend(loc="lower right")
plt.show()  

print('---------------------------------------comparaison des modèles-----------------------------------')

# Dictionnaire pour stocker les résultats
resultats = {}

# Liste des modèles et noms
modeles = [modele_rl, modele_ad, modele_rf]
noms_modeles = ["Régression Logistique", "Arbre de décision", "Random Forest"]

for nom, modele in zip(noms_modeles, modeles):
    y_pred = modele.predict(X_test)
    y_pred_proba = modele.predict_proba(X_test)[:,1]  # Probabilité pour AUC
    precision = accuracy_score(y_test, y_pred)
    score_auc = roc_auc_score(y_test, y_pred_proba)
    rapport = classification_report(y_test, y_pred, output_dict=True)
    f1 = rapport["weighted avg"]["f1-score"]
    
    resultats[nom] = {"Précision": precision, "F1-score": f1, "AUC": score_auc}

# Afficher les résultats
df_resultats = pd.DataFrame(resultats).T
print(df_resultats)

print('------------------------------------modèle AD et RF-------------------------------------')

# Exemple avec Random Forest
importance_variables = pd.Series(modele_rf.feature_importances_, index=X.columns)

# Trier par importance décroissante
importance_variables = importance_variables.sort_values(ascending=False)
print(importance_variables)

# Visualisation
plt.figure(figsize=(8,5))
sns.barplot(x=importance_variables, y=importance_variables.index)
plt.title("Importance des variables - Random Forest")
plt.show()

print('------------------------------------modèle RL------------------------------------')

coef = pd.Series(modele_rl.coef_[0], index=X.columns)
coef = coef.sort_values(key=abs, ascending=False)  # trier par impact absolu
print(coef)

# Création des fichiers csv à partir de mon dataframe
df[["étiquette","message"]].to_csv("features_spam.csv", index=False , sep="\t")
