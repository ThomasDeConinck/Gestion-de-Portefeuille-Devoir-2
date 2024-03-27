import csv
import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import gurobipy as gp
from gurobipy import GRB
from scipy.optimize import minimize


def importClean_10ind(csv_file_path, desired_returns):
    """
    Cette fonction importe des données à partir d'un fichier CSV spécifié, effectue un nettoyage de données,
    et retourne un DataFrame contenant des données filtrées sans tenir compte de la variable T (nombre d'années).

    Args:
        csv_file_path(str): Chemin vers le fichier CSV avec les données brutes.

    Returns:
        DataFrame: DataFrame contenant les données historiques filtrées pour les 10 industries.
    """


    # Importation
    with open(csv_file_path, 'r', newline='') as file:
        reader = csv.reader(file)
        
        # Sauter les 6 premières lignes
        for _ in range(10):
            next(reader)

        # Lire le reste du fichier CSV dans un DataFrame
        df = pd.DataFrame(reader)
        df.columns = df.iloc[1]
        df = df.drop(df.index[1])
        df = df.reset_index(drop=True)

    # Nettoyage
    df = df.rename(columns={df.columns[0]: 'Date'})
    first_row = df[df.iloc[:, 0].str.strip() == desired_returns].index
    
    # Si 'desired_returns' est 'Number of Firms in Portfolios' ou 'Average Firm Size', 
    # alors sauter une ligne supplémentaire pour accéder aux données dans le bon format
    if desired_returns in ['Number of Firms in Portfolios', 'Average Firm Size']:
        df_AEWR_monthly = df.iloc[first_row.values[0] + 2:].reset_index(drop=True)
    else:
        df_AEWR_monthly = df.iloc[first_row.values[0] + 1:].reset_index(drop=True)
        
    last_row = (df_AEWR_monthly['Date'].str.len() != 6).idxmax()
    df_AEWR_monthly = df_AEWR_monthly.iloc[:last_row]
    df_AEWR_monthly['Date'] = pd.to_datetime(df_AEWR_monthly['Date'], format='%Y%m')
    df_AEWR_monthly.iloc[:, 1:] = df_AEWR_monthly.iloc[:, 1:].astype(float)

    # Retourner le DataFrame et fixer la 'Date' comme index  
    df = df_AEWR_monthly.set_index('Date', drop=True)
    
    return (df)


def importClean_rf(csv_file_path):
    """
    Cette fonction importe des données à partir du fichier CSV des 3 facteurs Fama-French, puis effectue un nettoyage de données
    et retourne un DataFrame contenant des données filtrées spécifiquement pour les taux sans risque dans la colonne 'RF'.

    Args:
        csv_file_path (str): Chemin vers le fichier CSV avec les données brutes.

    Returns:
        DataFrame: DataFrame contenant les données nettoyées pour les taux sans risque.
    """
    
    
    # Importation des données en sautant les 3 premières lignes
    df = pd.read_csv(csv_file_path, skiprows=3)  

    # Renommage des colonnes pour faciliter la manipulation
    df.columns = ['Date', 'Mkt-RF', 'SMB', 'HML', 'RF']

    # Suppression des lignes dont la date ne correspond pas au format '%Y%m'
    df = df[df['Date'].str.match(r'^\d{6}$', na=False)]

    # Conversion de 'Date' en datetime
    df['Date'] = pd.to_datetime(df['Date'], format='%Y%m')

    # Conversion de 'RF' en float
    df['RF'] = df['RF'].astype(float)
    
    # Filtrage des colonnes 
    df = df[['Date', 'RF']] 
    
    # Mise en index de la colonne 'Date'
    df.set_index('Date', inplace=True)
    
    return df


def max_sharpe(z_bar, Sigma, Rf, short_allowed = False):
    """
    Optimise les poids des actifs dans le portefeuille qui maximise le ratio de Sharpe avec et sans contrainte de vente à découvert.

    Parameters:
    - z_bar (Series): Les rendements attendus de chacun des actifs.
    - Sigma (DataFrame): La matrice de covariance des rendements des actifs. 
    - Rf (float): Le taux de rendement sans risque mensuel.
    - short_allowed (bool): Autorise ou non les positions courtes dans le portefeuille.

    Returns:
    - weights_df (pd.DataFrame): Un dataFrame contenant les poids optimaux de chaque actif dans le portefeuille qui maximise le ratio de sharpe.
    """
    
    
    # Identifier les actifs et leur nombre
    assets = z_bar.index
    number_of_assets = len(assets)
    
    # Initialiser le modèle d'optimisation
    model = gp.Model("max_sharpe")
    
    # Définir les variables de décision: les poids y et la variable auxiliaire kappa
    if short_allowed : 
        y = model.addVars(assets, lb=-100.0, name='weights')
    else: 
        y = model.addVars(assets, lb=0, name='weights')
    kappa = model.addVar(lb=0.0, name='kappa') 
    
    # Construire le vecteur des poids y
    y_vector = [y[i] for i in assets]

    # Calculer la variance du portefeuille (sigma_p) en fonction des poids y et de la matrice de covariance Sigma 
    sigma_p = sum(y_vector[i] * y_vector[j] * Sigma.loc[assets[i], assets[j]]
                    for i in range(number_of_assets) for j in range(number_of_assets))
    
    # Ajouter des contraintes au modèle d'optimisation 
    model.addConstr(sum((z_bar[i] - Rf) * y[i] for i in assets) == 1, "Rendement ajusté") # Contrainte de rendement ajusté normalisé selon k 
    model.addConstr(sum(y[i] for i in assets) == kappa) # Constrainte nécessaire pour que kappa ne soit pas nul 
    
    # Définir l'objectif: Minimiser la variance du portefeuille (Maximiser le ratio de Sharpe)
    model.setObjective(sigma_p, GRB.MINIMIZE)
    
    # Exécuter l'optimisation (sans affichage de console)
    model.Params.LogToConsole = 0
    model.optimize()
    
    # Vérifier si la solution optimale a été trouvée
    if model.status == GRB.OPTIMAL:
        y_opt = model.getAttr('x', y) # Poids optimaux y
        kappa_opt = kappa.X # Valeur optimale de kappa
        
        # Calculer et ajuster les poids du portefeuille original x à partir de y et kappa 
        x_opt = {i: y_opt[i] / (kappa_opt) for i in assets}
        
        # Convertir les poids en DataFrame 
        weights_df = pd.DataFrame([x_opt], columns=assets)
        
        return weights_df
    else:
        print("Optimization non réussie. Impossible de retrouver la solution.")
        return None


def Inverse_Variance_Portfolio(Sigma):
    """
    Calcule le portefeuille où les poids des actifs sont inversement proportionnels à la variance de l'actif.
    
    Parameters:
    - Sigma (Dataframe): La matrice de covariance des rendements des actifs. 
    
    Returns:
    - normalized_weights (DataFrame): Un dataFrame contenant les poids optimaux (inversement reliés à la variance) de chaque actif dans le portefeuille.
    """
    
    
    # Extraire les variances de la diagonale de la matrice de covariance Sigma 
    variances = Sigma.values.diagonal()
    
    # Calculer l'inverse de chaque variance
    # Objectif : donner plus de poids aux actifs ayant une plus faible variance
    inverse_variance_weights = 1 / variances
    
    # Normaliser les poids inverses de la variance pour qu'ils somment à 1
    normalized_weights_var = inverse_variance_weights / inverse_variance_weights.sum()
    
    # Convertir le tableau numpy de poids normalisés en un DataFrame pandas
    normalized_weights_var = pd.DataFrame([normalized_weights_var], columns=Sigma.index)
    
    # Retourner le DataFrame des poids normalisés 
    return normalized_weights_var


def Inverse_Volatility_Portfolio(Sigma):
    """
    Calcule les poids du portefeuille qui sont inversement proportionnels à la volatilité de chaque actif.
    
    Parameters:
    - Sigma (Dataframe):  DataFrame Pandas représentant la matrice de covariance des rendements des actifs. 
    
    Returns:
    - normalized_weights_std (Dataframe): Un dataFrame contenant les poids optimaux (inversement reliés à la volatilité) de chaque actif dans le portefeuille.
    """
    
    
    # Calcule la volatilité (écart type de la variance) pour chaque actif
    volatilites = np.sqrt(Sigma.values.diagonal())

    # Calcule l'inverse de la volatilité pour chaque actif
    # Objectif : donner plus de poids aux actifs moins volatils (c'est-à-dire ceux ayant une faible volatilité)
    inverse_volatility_weights = 1 / volatilites

    # Normaliser les poids inverses de la volatilité pour qu'ils somment à 1
    normalized_weights_std = inverse_volatility_weights / inverse_volatility_weights.sum()

    # Convertir le tableau numpy de poids normalisés en un DataFrame pandas
    normalized_weights_std = pd.DataFrame([normalized_weights_std], columns=Sigma.index)

    # Retourner le DataFrame des poids normalisés
    return normalized_weights_std


def Equally_Weighted_Portfolio(assets):
    """
    Calcule les poids équipondérés (1/N) pour chaque actif dans le portefeuille de 10 industries.
    
    Parameters:
    - Assets: DataFrame Pandas représentant le nombre d'actifs dans le portefeuille.
    
    Returns:
    - portfolio_weights (Dataframe) : Un dataFrame contenant des poids égaux pour chaque actif dans le portefeuille 1/N.
    """
    
    
    # Nombre d'actifs dans le portefeuille d'industries
    n_assets = len(assets) 
    
    # Poids équipondérés dans le portefeuille (1/N pour chaque actif)
    equal_weight = 1 / n_assets 
    
    # Créer un dictionnaire de poids pour chaque actif
    EW_portfolio_weights = {asset: equal_weight for asset in assets}
    
    # Convertir le dictionnaire de poids en un DataFrame pandas
    EW_portfolio_weights = pd.DataFrame([EW_portfolio_weights], columns=assets)
    
    return EW_portfolio_weights


def Market_Cap_Weighted_Portfolio(df_average_firm_size, df_number_firm):
    """
    Cette fonction calcule les poids proportionnels à la capitalisation boursière pour chaque actifs dans le portefeuille d'industries,
    étant donné la taille moyenne des entreprises et le nombre d'entreprises.
    
    Parameters:
    - df_average_firm_size: DataFrame représentant la taille moyenne de l'entreprise pour chaque actif dans le portefeuille au fil du temps.
    - df_number_firm: DataFrame représentant le nombre d'entreprises pour chaque actif dans le portefeuille au fil du temps.
    
    Returns:
    - weights: DataFrame contenant les poids selon la capitalisation boursière pour chaque actif dans le portefeuille.
    """
    
    
    # Calcule la capitalisation boursière pour chaque actif dans le portefeuille au fil du temps
    market_caps = df_average_firm_size.multiply(df_number_firm)
    
    # Calcule la capitalisation boursière totale pour tous les actifs pour chaque date
    total_market_cap = market_caps.sum(axis=1)
    
    # Calcule les poids pour chaque actif en divisant la capitalisation boursière de chaque actif par la capitalisation boursière totale
    VW_portfolio_weights = market_caps.div(total_market_cap, axis='index')
    
    # Retourne les poids du portefeuille optimal dans un DataFrame 
    VW_portfolio_weights = pd.DataFrame(VW_portfolio_weights, columns=df_average_firm_size.columns)
    
    return VW_portfolio_weights


def MV_optimize_portfolio(z_bar, Sigma, short_allowed=False):
    """
    Optimise le portefeuille qui minimise la variance du portefeuille avec une contrainte de retour cible (mu_target) en utilisant la programmation linéaire.
    
    Parameters:
    - z_bar (Series): Série pandas contenant les rendements attendus pour chaque actif du portefeuille.
    - Sigma (DataFrame): DataFrame pandas représentant la matrice de covariance des rendements des actifs du portefeuille. 
    - short_allowed (bool): Booléen indiquant si les positions de ventes à découvert sont autorisées (True) ou non (False).

    Returns:
    - df_results (pd.DataFrame): Un DataFrame pandas contenant les poids optimaux pour chaque actif (optimal_w) si l'optimisation a convergé vers une solution.
    """
    
    
    # Récupérer les actifs et le retour cible
    assets = z_bar.index
    # mu_target = 0.02  

    # Création d'un nouveau modèle d'optimisation linéaire
    m = gp.Model("portfolio")

    # Création de variables pour les poids de chaque actif dans le portefeuille
    if short_allowed:
        weights = m.addVars(assets, lb=-GRB.INFINITY, name="weights")
    else:
        weights = m.addVars(assets, lb=0, ub=1, name="weights")

    # Variance du portefeuille (objectif à minimiser) : w^T * Sigma * w 
    portfolio_variance = sum(weights[i] * Sigma.loc[i, j] * weights[j] for i in assets for j in assets)

    # Définir l'objectif pour minimiser la variance du portefeuille 
    m.setObjective(portfolio_variance, GRB.MINIMIZE)

    # Ajouter des contraintes pour le budget et le retour cible
    m.addConstr(sum(weights[asset] for asset in assets) == 1, "budget")
    # m.addConstr(sum(weights[asset] * z_bar[asset] for asset in assets) >= mu_target, "target_return") # Contrainte optionnelle pour le retour cible 

    # Optimiser le modèle 
    m.Params.LogToConsole = 0
    m.optimize()

    # Vérifier si l'optimisation a été réussie 
    if m.status == GRB.OPTIMAL:
        
        # Obtenir les poids optimaux et la variance si l'optimisation est réussie 
        optimal_w = {asset: weights[asset].X for asset in assets}

        # Créer un DataFrame pour les résultats optimaux
        df_results = pd.DataFrame(optimal_w, index=[0])
    else:
        print("Optimization was unsuccessful. Unable to retrieve the solution.")
        return None
    
    return df_results