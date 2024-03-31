import csv
import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import gurobipy as gp
from gurobipy import GRB
from scipy.optimize import minimize
import matplotlib.dates as mdates
from datetime import datetime
import os


# Partie A 
def importClean_Ind(csv_file_path, desired_returns):
    """
    Cette fonction importe des données à partir d'un fichier CSV spécifié, effectue un nettoyage de données,
    et retourne un DataFrame contenant des données filtrées.

    Args:
        csv_file_path(str): Chemin vers le fichier CSV avec les données brutes.

    Return:
        DataFrame: DataFrame contenant les données historiques filtrées pour les 10 industries, 
        pour les 48 industries et certains facteurs de Fama-French.
    """


    # Importation
    with open(csv_file_path, 'r', newline='') as file:
        reader = csv.reader(file)
        
        # Sauter les 10 premières lignes
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
    if desired_returns in ['Number of Firms in Portfolios', 'Average Firm Size', 'Sum of BE / Sum of ME']:
        df_AEWR_monthly = df.iloc[first_row.values[0] + 2:].reset_index(drop=True)
    else:
        df_AEWR_monthly = df.iloc[first_row.values[0] + 1:].reset_index(drop=True)
        
    last_row = (df_AEWR_monthly['Date'].str.len() != 6).idxmax()
    df_AEWR_monthly = df_AEWR_monthly.iloc[:last_row]
    
    # Adapter le format de la date en fonction de 'desired_returns'
    # Condition spécifique pour le facteur BtoM (Book-to-Market)
    if desired_returns in ['Sum of BE / Sum of ME']:  
        df_AEWR_monthly['Date'] = df_AEWR_monthly['Date'].str.strip()
        df_AEWR_monthly['Date'] = pd.to_datetime(df_AEWR_monthly['Date'], format='%Y')
    else:
        df_AEWR_monthly['Date'] = pd.to_datetime(df_AEWR_monthly['Date'], format='%Y%m')
    
    df_AEWR_monthly.iloc[:, 1:] = df_AEWR_monthly.iloc[:, 1:].astype(float)

    # Vérifier si le nom du fichier contient '48_Industry_Portfolios.CSV',
    # pour appliquer un nettoyage supplémentaire pour les NaN values
    if '48_Industry_Portfolios.CSV' in os.path.basename(csv_file_path):
        df_AEWR_monthly.replace(-99.99, np.nan, inplace=True)
        df_AEWR_monthly.replace(-999, np.nan, inplace=True)
        df_AEWR_monthly.dropna(inplace=True)
        
    # Retourner le DataFrame et fixer la 'Date' comme index  
    df = df_AEWR_monthly.set_index('Date', drop=True)
    
    return df


def importClean_rf(csv_file_path):
    """
    Cette fonction importe des données à partir du fichier CSV des 3 facteurs Fama-French, puis effectue un nettoyage de données
    et retourne un DataFrame contenant des données filtrées spécifiquement pour les taux sans risque dans la colonne 'RF'.

    Args:
        csv_file_path (str): Chemin vers le fichier CSV avec les données brutes.

    Return:
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
    Calcul les poids optimisés des actifs dans le portefeuille qui maximisent le ratio de Sharpe avec et sans contrainte de vente à découvert,
    en utilisant le solveur de programmation linéaire GUROBI dans le cas avec contrainte sur vente à découverte et analytiquement pour le cas 
    sans contrainte de vente à découvert.

    Parameters:
    - z_bar (Series): Les rendements attendus de chacun des actifs.
    - Sigma (DataFrame): La matrice de covariance des rendements des actifs.
    - Rf (float): Le taux de rendement sans risque mensuel.
    - short_allowed (bool): Autorise ou non les positions courtes dans le portefeuille.

    Return:
    - weights_df (pd.DataFrame): Un dataFrame contenant les poids optimaux de chaque actif dans le portefeuille qui maximise le ratio de sharpe.
    """
    
    
    # Identifier les actifs et leur nombre
    assets = z_bar.index
    number_of_assets = len(assets)

    if short_allowed == False:
        # Initialiser le modèle d'optimisation
        model = gp.Model("max_sharpe")
        
        # Désactiver l'affichage de la console
        model.Params.LogToConsole = 0
        model.Params.OutputFlag = 0

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
        
        # Exécuter l'optimisation
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
            print("Optimization non réussie. Impossible de retrouver la solution à cette date précise.")
            return None
    else:
        # Utiliser la formule analytique pour les poids du portefeuille tangent (qui maximise le ratio de Sharpe)
        one = np.ones(number_of_assets)
        A = np.dot(one, np.dot(np.linalg.inv(Sigma), one.T))
        B = np.dot(one, np.dot(np.linalg.inv(Sigma), z_bar))
        w = (np.dot(np.linalg.inv(Sigma), z_bar - Rf*np.ones(number_of_assets))/(B - A*Rf)).tolist()

        # Convertir les poids en DataFrame
        weights = {n: w[i] for i, n in enumerate(assets)}
        weights_df = pd.DataFrame([weights], columns=assets)

        return weights_df


def Inverse_Variance_Portfolio(Sigma):
    """
    Calcule le portefeuille où les poids des actifs sont inversement proportionnels à la variance de l'actif.
    
    Parameters:
    - Sigma (Dataframe): La matrice de covariance des rendements des actifs. 
    
    Return:
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
    
    Return:
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
    
    Return:
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
    Calcul les poids optimaux qui minimisent la variance du portefeuille avec une contrainte de budget dans les deux cas avec et sans contrainte de vente à découvert,
    en utilisant le solveur de programmation linéaire GUROBI.
    
    Parameters:
    - z_bar (Series): Série pandas contenant les rendements attendus pour chaque actif du portefeuille d'industrie.
    - Sigma (DataFrame): DataFrame pandas représentant la matrice de covariance des rendements des actifs du portefeuille. 
    - short_allowed (bool): Booléen indiquant si les positions de ventes à découvert sont autorisées (True) ou non autorisées (False).

    Return:
    - df_results (pd.DataFrame): Un DataFrame pandas contenant les poids optimaux pour chaque actif (optimal_w) si l'optimisation a convergé vers une solution.
    """
    
    
    # Récupérer les actifs et le retour cible
    assets = z_bar.index
    # mu_target = 0.02  

    # Création d'un nouveau modèle d'optimisation linéaire
    m = gp.Model("MV_portfolio")

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


def rolling_window_optimization(df,df_rf, df_average_firm_size, df_number_firm, window_size, optimization_type):
    """
    Calculate portfolio weights for each window using a rolling window based on the specified optimization type.

    Parameters:
    - df (DataFrame): DataFrame containing asset returns for each month with dates as index.
    - df_rf (DataFrame): DataFrame containing risk-free rate for each month with dates as index.
    - df_average_firm_size (DataFrame): DataFrame containing average firm size for each month with dates as index.
    - df_number_firm (DataFrame): DataFrame containing number of firms for each month with dates as index.
    - window_size (int): Size of the rolling window in months.
    - optimization_type (str): Type of portfolio optimization to perform ('min_variance' for minimum variance portfolio, or other types as needed).

    Return:
    - results_df (DataFrame): DataFrame containing portfolio weights (weights_df) for each rolling window with associated dates as the index (In-sample weights).
    """   

    
    results = []
    dates = []
    weights_df = None
    
    
    # Itérer sur chaque fenêtre de la taille spécifiée
    for start_idx in range(len(df) - window_size + 1): 
        window_data = df.iloc[start_idx:start_idx + window_size] # Récupère les rendements pour la fenêtre actuelle
        end = start_idx + window_size # Calcule l'indice de fin de la fenêtre roulante en cours

        # Récupère le taux sans risque moyen pour la fenêtre roulante en cours 
        window_rf = df_rf.iloc[start_idx:end]['RF'].mean()

        window_average_firm_size = df_average_firm_size.iloc[start_idx:start_idx + window_size] # Récupère les tailles moyennes des entreprises pour la fenêtre actuelle
        window_number_firm = df_number_firm.iloc[start_idx:start_idx + window_size] # Récupère le nombre d'entreprises pour la fenêtre actuelle 
        
        # Calcul de la matrice de covariance et du rendement attendu pour la fenêtre actuelle 
        Sigma = window_data.cov()
        z_bar = window_data.mean()

        # Appel des sept fonctions d'optimisation de portefeuille
        # Call the minimum variance portfolio optimization function with short selling allowed
        if optimization_type == 'min_variance_short_allowed':
            weights_df = MV_optimize_portfolio(z_bar, Sigma, short_allowed= True)
            
        # Call the maximum sharpe ratio portfolio optimization function with short selling not allowed
        elif optimization_type == 'max_sharpe_no_short':
            weights_df = max_sharpe(z_bar, Sigma, window_rf, short_allowed= False)
            
        # Call the maximum sharpe ratio portfolio optimization function with short selling allowed
        elif optimization_type == 'max_sharpe_short_allowed':
            weights_df = max_sharpe(z_bar, Sigma, window_rf, short_allowed= True)
            
        # Call the inverse variance portfolio optimization function
        elif optimization_type == 'inv_variance_weights':
            weights_df = Inverse_Variance_Portfolio(Sigma)
        
        # Call the inverse volatility portfolio optimization function
        elif optimization_type == 'inv_volatility_weights':
            weights_df = Inverse_Volatility_Portfolio(Sigma)
        
        # Call the equally weighted portfolio optimization function
        elif optimization_type == 'equal_weights':
            weights_df = Equally_Weighted_Portfolio(df.columns)
        
        # Call the market cap weighted portfolio optimization function    
        elif optimization_type == 'market_cap_weights':
            weights_df = Market_Cap_Weighted_Portfolio(window_average_firm_size, window_number_firm)
            
        # Vérifie si l'optimisation a retourné des résultats 
        if weights_df is not None:  
            
            # Récupération des poids pour l'optimisation market_cap_weights puisque les poids sont stockés dans un DataFrame 
            if isinstance(weights_df.index, pd.DatetimeIndex):
                weights = weights_df.iloc[-1, :].tolist()
            else:
                weights = weights_df.loc[0, df.columns].tolist() # Récupère les poids optimaux pour la fenêtre actuelle
            results.append(weights) 

            dates.append(window_data.index[-1]) # Utilisation de l'index pour la date de fin de la fenêtre
        # Si l'optimisation a échoué, affiche un message d'erreur
        else:
            print(f"Optimization failed for the window ending on {window_data.index[-1]}")

            
    # Création du DataFrame des résultats
    results_df = pd.DataFrame(results, index=dates, columns=df.columns)
    return results_df


def Out_of_sample_portfolio_returns(results_df, df):
    """
    Calculate the monthly out-of-sample returns of the portfolios using the optimized in-sample weights.
    
    Parameters:
    - results_df (DataFrame): DataFrame containing portfolio weights (weights_df) for each rolling window with associated dates as the index (In-sample weights).
    - df (DataFrame): DataFrame containing asset returns for each month with dates as the index (used to calculate the portfolio returns).
    
    Return:
    - DataFrame: The same 'results_df' DataFrame containing optimized in-sample weights with an additional column 'Portfolio Monthly Return' containing the calculated returns for the optimized portfolios (out-of-sample returns).
    """
    
    
    portfolio_monthly_returns = []
    
    # Itére sur chaque ligne de 'results_df' pour accéder aux poids optimisés de chaque fenêtre
    for index, row in results_df.iterrows():
        next_month = index + pd.DateOffset(months=1) # Calcule la date correspondant au mois suivant la fin de la fenêtre roulante
        if next_month in df.index:
            next_month_returns = df.loc[next_month]  # Extrait les rendements des actifs pour le mois suivant
            weights = row.values # Extrait les poids optimisés du portefeuille pour la fenêtre actuelle
            portfolio_return = np.dot(weights, next_month_returns) # Calcule le rendement du portefeuille pour le mois suivant
            portfolio_monthly_returns.append(portfolio_return) # Ajouter le rendement calculé du portefeuille à la liste des rendements mensuels
        else:
            portfolio_monthly_returns.append(None)
    
    # Ajouter la liste des rendements mensuels calculés à 'results_df' comme nouvelle colonne
    results_df['Portfolio Monthly Return'] = portfolio_monthly_returns
    return results_df


def plot_cumulative_returns(df, strategies, df_rf, df_average_firm_size, df_number_firm):
    """
    Cette fonction plot les rendements cumulatifs pour les 7 stratégies de portefeuilles d'industries. 

    Parameters:
    - df: DataFrame contenant les données de rendement.
    - strategies: Liste des stratégies à utiliser dans le tracé des rendements cumulatifs.
    
    Return : 
    - Plot des rendements cumulatifs pour les 7 stratégies de portefeuilles d'industries.
    """
    
    
    def cumulative_returns_calculation(mensual_returns, is_decimal=False):
        """
        Cette fonction calcule les retours cumulatifs à partir des séries temporelles de rendements mensuels pour les 7 stratégies de portefeuille.

        Parameters:
        - mensual_returns: Séries pandas contenant les retours mensuels du portefeuille sélectionné en décimal.
        
        Returns : 
        - cumulative_returns: Séries pandas contenant les retours cumulatifs du portefeuille sélectionné en décimal.
        """
        
        
        # Convertir les pourcentages en décimales et ajouter 1
        adjusted_returns = mensual_returns / 100 + 1 if not is_decimal else mensual_returns + 1

        # Calculer les retours cumulatifs avec la fonction cumprod() de pandas qui calcule le produit cumulatif des retours mensuels
        cumulative_returns = adjusted_returns.cumprod()

        return cumulative_returns

    # Initialiser le dictionnaire pour stocker les résultats de chaque stratégie de portefeuille 
    results_dict = {}

    # Boucle sur chaque stratégie de portefeuille 
    for strategy in strategies:
        # Appeler la fonction d'optimisation pour chaque stratégie et calculer les rendements du portefeuille sur la bonne fenêtre roulante d'optimisation
        results_df = Out_of_sample_portfolio_returns(rolling_window_optimization(df, df_rf, df_average_firm_size, df_number_firm, 60, optimization_type=strategy), df)
        
        # Stocker les résultats dans le dictionnaire 
        results_dict[strategy] = results_df

    # Définir la taille de la figure à plotter
    plt.figure(figsize=(9,6))

    # Boucle sur chaque stratégie de portefeuille pour tracer les rendements cumulatifs 
    for strategy in strategies:
        # Récupérer les rendements de la stratégie
        results_with_returns = results_dict[strategy]
        
        # Supprimer les valeurs manquantes dans les rendements mensuels pour la dernière date 2024-01-01
        results_with_returns['Portfolio Monthly Return'].dropna(inplace=True)
        
        # Calculer les rendements cumulatifs
        results_with_returns['Cumulative Return'] = cumulative_returns_calculation(results_with_returns['Portfolio Monthly Return'])

        # Tracer les rendements cumulatifs pour chaque stratégie
        plt.plot(results_with_returns.index, results_with_returns['Cumulative Return'], label=strategy)

    plt.title('Rendements cumulatifs des stratégies de portefeuilles (hors échantillon)')
    plt.xlabel('Date')
    plt.ylabel('Rendements cumulatifs (%)')  
    plt.legend()
    plt.show()
    
    
def format_value(val):
    """
    Cette fonction formate les valeurs numériques pour l'affichage, en utilisant un format de chaîne de caractères spécifié.
    
    Paramètres :
    - val (float) : Valeur numérique à formater.
    """
    
    
    return "{:.4f}".format(val)


def annualized_statistics_and_sharpe_ratios(strategies_results, periods, df_rf, is_decimal=False):
    """
    Calcule les statistiques annualisées pour les rendements hors échantillon de différentes stratégies de portefeuille sur des périodes spécifiées,
    et détermine les ratios de Sharpe annuels pour ces stratégies.

    Parameters:
    - strategies_results (dict) : Dictionnaire de DataFrames, où chaque clé représente le nom d'une stratégie et chaque valeur est
    un DataFrame contenant les résultats pour les poids optimaux de cette stratégie, y compris les 'Rendements Mensuels du Portefeuille'.
    - periods (liste de tuples) : Liste de tuples contenant la date de début et la date de fin d'une période à analyser,
    au format ('AAAA-MM-JJ', 'AAAA-MM-JJ').
    - df_rf (DataFrame) : DataFrame contenant les taux sans risque mensuels.
    - is_decimal (bool) : Booléen indiquant si les rendements sont en pourcentage (False) ou en décimal (True) pour le calcul.

    Return:
    - DataFrame contenant la moyenne annualisée et l'écart-type annualisé des rendements mensuels du portefeuille, ainsi que les ratios de Sharpe
    pour chaque stratégie et chaque période spécifiée.
    """
    
    
    results = []

    # Calculer le taux sans risque mensuel moyen et l'annualiser selon la formule du taux de rendement annuel composé (CAGR), 
    # pour obtenir le taux sans risque annuel moyen
    mean_rf_monthly = df_rf['RF'].mean()
    mean_rf_annual = ((1 + mean_rf_monthly/100) ** 12 - 1) 
    
    # Convertir en pourcentage pour faciliter les manipulations par la suite
    mean_rf_annual *= 100

    for strategy_name, strategy_results in strategies_results.items():
        for start_date, end_date in periods:
            # Filtrer les résultats de la stratégie pour la période spécifiée
            period_returns = strategy_results['Portfolio Monthly Return'].loc[start_date:end_date]

            # Calculer la moyenne et l'écart-type des rendements mensuels du portefeuille pour la période indiquée
            mean_return_monthly = period_returns.mean()
            std_return_monthly = period_returns.std()

            # Annualiser la moyenne et l'écart-type des rendements mensuels, avec la formule du taux de rendement annuel composé (CAGR)
            mean_return_annualized = (1 + mean_return_monthly/100) ** 12 - 1 if not is_decimal else (1 + mean_return_monthly) ** 12 - 1
            std_return_annualized = std_return_monthly * np.sqrt(12) if not is_decimal else std_return_monthly * np.sqrt(12) * 100
            
            # Convertir en pourcentage pour faciliter les manipulations par la suite
            mean_return_annualized = mean_return_annualized * 100 

            # Calculer le ratio de Sharpe annuel pour chaque stratégie et période
            sharpe_ratio = (mean_return_annualized - mean_rf_annual) / std_return_annualized

            # Appliquer le formatage adapté à la valeur du ratio de Sharpe
            formatted_mean_return = format_value(mean_return_annualized)
            formatted_std_deviation = format_value(std_return_annualized)
            formatted_sharpe_ratio = format_value(sharpe_ratio)

            # Ajouter les résultats à la liste
            results.append({
                'Strategy': strategy_name,
                'Start Date': start_date,
                'End Date': end_date,
                'Annualized Mean Return': formatted_mean_return,
                'Annualized Std Deviation': formatted_std_deviation,
                'Sharpe Ratio': formatted_sharpe_ratio
            })

    # Créer un DataFrame à partir de la liste des résultats
    results_df = pd.DataFrame(results)

    # Définir les noms des stratégies et les périodes comme index du DataFrame
    results_df.index = pd.MultiIndex.from_frame(results_df[['Strategy', 'Start Date', 'End Date']])
    results_df = results_df.drop(columns=['Strategy', 'Start Date', 'End Date'])

    return results_df


# Partie C 
def standardize(df):
    """
    Cette fonction standardise les trois caractéristiques provenant chacunes d'un DataFrame, en utilisant la moyenne et l'écart-type.

    Parameters:
    - 'df' (DataFrame): Le DataFrame de la caractéristique dont les valeurs doivent être standardisées.

    Return:
    - Un nouveau DataFrame avec les données standardisées. Les valeurs ont une moyenne de 0 et un écart-type de unitaire. 
    Les noms de colonnes et les indices sont les mêmes que le DataFrame original.
    """
    
    
    # StandardScaler() est une classe de la bibliothèque sklearn.preprocessing qui standardise les données
    # en utilisant la formule (x - moyenne) / écart-type. Cela transforme les données de sorte que leur moyenne soit 0 et leur écart-type soit unitaire.
    # La méthode fit_transform calcule la moyenne et l'écart-type, puis effectue la transformation.
    df_scaled = StandardScaler().fit_transform(df)

    # Crée ou nouveau DataFrame avec les données standardisées. On conserve les mêmes noms de colonnes et indices que le DataFrame original.
    df_standardized = pd.DataFrame(df_scaled, columns=df.columns, index=df.index)

    # Retourne le DataFrame standardisé. Ses valeurs ont une moyenne de 0 et un écart-type de 1.
    return df_standardized


def calculate_initial_regression_coefficients(N, M, df_48Ind, MC_standardized, BM_standardized, MOM_standardized):
    """
    Cette fonction ajuste un modèle de régression linéaire pour chaque caractéristique (MC, BM, MOM) en utilisant les rendements des actifs comme variable dépendante et les caractéristiques comme variables indépendantes,
    sur l'ensemble de la période de temps T.  Ensuite la fonction calcule la moyenne des coefficients de régression pour chaque caractéristique à travers tous les actifs, le résultat est utilisé comme estimation initiale 
    pour les coefficients thetas dans l'optimisation ultérieure.
    
    Paramètres :
    - 'N' (int) : Le nombre d'actifs (48).
    - 'M' (int) : Le nombre de caractéristiques (3).
    - 'df_48Ind' (DataFrame) : Le DataFrame contenant les rendements mensuels des 48 industries.
    - 'MC_standardized' (DataFrame) : Le DataFrame contenant les données standardisées de la caractéristique MC.
    - 'BM_standardized' (DataFrame) : Le DataFrame contenant les données standardisées de la caractéristique BM.
    - 'MOM_standardized' (DataFrame) : Le DataFrame contenant les données standardisées de la caractéristique MOM.

    Retour :
    - Un tableau numpy contenant les coefficients de régression moyens pour chaque caractéristique, ce qui permet d'obtenir un bon proxy pour les coefficients thetas initiaux à plugger dans le modèle à optimiser.
    """
    
    
    model = LinearRegression()
    
    # Initialiser la matrice des coefficients de régression
    theta_initial = np.zeros((N, M))
    
    # Ajuster le modèle de régression pour chaque actif et stocker les coefficients
    for i in range(N):
        y = df_48Ind.iloc[:, i]
        X_MC = MC_standardized.iloc[:, i].values.reshape(-1, 1)
        X_BM = BM_standardized.iloc[:, i].values.reshape(-1, 1)
        X_MOM = MOM_standardized.iloc[:, i].values.reshape(-1, 1)
        
        # Fitter le modèle de régression pour chaque caractéristique et stocker les coefficients
        model.fit(X_MC, y)
        theta_initial[i, 0] = model.coef_[0]
        
        model.fit(X_BM, y)
        theta_initial[i, 1] = model.coef_[0]
        
        model.fit(X_MOM, y)
        theta_initial[i, 2] = model.coef_[0]

    # Moyenne des coefficients à travers tous les actifs pour obtenir un bon proxy pour les 3 coefficients initiaux à inclure dans l'optimisation
    theta_initial = theta_initial.mean(axis=0)

    return theta_initial


def Expanding_window_optimization(data, mkt_weights, MC_standardized, BM_standardized, MOM_standardized, returns):
    """
    Cette fonction effectue une optimisation de portefeuille pour obtenir les coefficients du modèle (selon l'équation (6) de l'article) en utilisant une fenêtre d'expansion grandissante de 12 mois, 
    en utilisant les données de rendement des 48 industries, les poids du portefeuille marché (non tilté), et les caractéristiques de style standardisées.
    
    Paramètres:
    - 'data' (DataFrame) : DataFrame contenant les rendements mensuels des 48 industries.
    - 'mkt_weights' (DataFrame) : DataFrame contenant les poids du portefeuille de marché (non tilté) pour chaque industrie.
    - 'MC_standardized' (DataFrame) : DataFrame contenant les données standardisées de la caractéristique SMB (Small Minus Big).
    - 'BM_standardized' (DataFrame) : DataFrame contenant les données standardisées de la caractéristique HML (High Minus Low).
    - 'MOM_standardized' (DataFrame) : DataFrame contenant les données standardisées de la caractéristique UMD (Up Minus Down).
    - 'returns' (DataFrame) : DataFrame contenant les rendements mensuels des 48 industries shiftés d'une période pour être à t+1.
    
    Retourne:
    - Un dictionnaire contenant les coefficients optimaux pour chaque période de temps dans la fenêtre d'expansion grandissante.
    """
    
    # définie la fonction d'utilité CRRA à maximiser 
    def CRRA_utility(rp, gamma=5):
        return (1 + rp) ** (1 - gamma) / (1 - gamma)


    # Définie la fonction objectif à maximiser, équation (6) provenant de l'article 
    def objective(theta, mkt_weights, MC, BM, MOM, returns, T, N, M):
        """
        Cette fonction optimise les coefficients du modèle (selon l'équation (6) de l'article) en minimisant l'utilité négative CRRA, maximisant ainsi l'utilité espérée CRRA,
        en utilisant les données de rendement, les poids du portefeuille de marché, et les caractéristiques de style standardisées.
        
        Paramètres:
        - 'theta' (array) : Les coefficients du modèle à optimiser selon l'équation (6) de l'article.
        - 'mkt_weights' (DataFrame) : DataFrame contenant les poids du portefeuille de marché (non tilté) pour chaque industrie.
        - 'MC' (DataFrame) : DataFrame contenant les données standardisées de la caractéristique SMB (Small Minus Big).
        - 'BM' (DataFrame) : DataFrame contenant les données standardisées de la caractéristique HML (High Minus Low).
        - 'MOM' (DataFrame) : DataFrame contenant les données standardisées de la caractéristique UMD (Up Minus Down).
        - 'returns' (DataFrame) : DataFrame contenant les rendements mensuels des 48 industries shiftés d'une période pour être à t+1.
        - 'T' (int) : Le nombre de périodes de temps.
        - 'N' (int) : Le nombre d'actifs (48 industries).
        - 'M' (int) : Le nombre de caractéristiques de style (3).
        
        Retourne:
        - La valeur de l'utilité espérée CRRA négative moyenne sur la période de temps T.
        """
        
        
        rp = [sum((mkt_weights.iloc[t, i] + 
                   (1/N) * (theta[0] * MC.iloc[t, i] + 
                            theta[1] * BM.iloc[t, i] + 
                            theta[2] * MOM.iloc[t, i])) 
                # L'expression (returns.iloc[t+1, i] if t+1 < T else 0) vérifie si t+1 est inférieur à T. 
                # Si c'est le cas, elle utilise le rendement de l'actif i à la période t+1 (returns.iloc[t+1, i]).
                # Si t+1 n'est pas inférieur à T (c'est-à-dire, si t+1 est égal à T), elle utilise 0 comme rendement.
                  * (returns.iloc[t+1, i] if t+1 < T else 0) 
                for i in range(N)) for t in range(T)]
        
        utility_CRRA = sum(CRRA_utility(r) for r in rp)
        
        # Minimiser l'utilité négative CRRA (équivalent à maximiser l'utilité espérée CRRA)
        return -utility_CRRA / (T) 



    # Boucle de rolling optimization pour chaque période de temp depuis 1973 jusqu'à la dernière date dans les données de rendement
    optimal_thetas = {} 
    # Date de début fixée à la première date dans les données de rendement 
    start_year = data.index.min() 
    # Date de fin initiale fixée à décembre 1973
    initial_end_year = pd.to_datetime('1973-12-01') 
    # Date de fin courante, commence par initial_end_year et sera ajustée dans la boucle 
    current_end_year = initial_end_year 

        
    # Boucle jusqu'à la dernière date dans les données de rendement
    while current_end_year <= data.index.max(): 
        
        # Extraire les composantes nécessaires pour l'optimisation des coefficients pour la période sélectionnée dans la window
        mkt_weights_filtred = mkt_weights.loc[start_year:current_end_year]
        MC_filtred = MC_standardized.loc[start_year:current_end_year]
        BM_filtred = BM_standardized.loc[start_year:current_end_year]
        MOM_filtred = MOM_standardized.loc[start_year:current_end_year]
        returns_filtred = returns.loc[start_year:current_end_year]
        N = len(mkt_weights.columns)
        M = 3
        T = len(returns_filtred)

        # Guess initial pour theta
        optimal_initial_theta = calculate_initial_regression_coefficients(N, M, data, MC_standardized, BM_standardized, MOM_standardized)
    
        # Minimise la fonction objectif pour la période courante 
        result = minimize(objective, optimal_initial_theta, args=(mkt_weights_filtred, MC_filtred, BM_filtred, MOM_filtred, returns_filtred, T, N, M), method='SLSQP')

        optimal_thetas[current_end_year] = result.x if result.success else None

        # Élargir la fenêtre grandissante pour la prochaine itération en ajoutant 12 mois à la dernière date courante de fin (1 an)
        current_end_year += pd.DateOffset(months=12)

    return optimal_thetas


def Expanding_window_optimization_last_period_initial_theta(data, mkt_weights, MC_standardized, BM_standardized, MOM_standardized, returns):
    # Define the CRRA utility function 
    def CRRA_utility(rp, gamma=5):
        return (1 + rp) ** (1 - gamma) / (1 - gamma)

    # Define the objective function to be maximized, equation(6) from article
    def objective(theta, mkt_weights, MC, BM, MOM, returns, T, N, M):
        rp = [sum((mkt_weights.iloc[t, i] + 
                   (1/N) * (theta[0] * MC.iloc[t, i] + 
                            theta[1] * BM.iloc[t, i] + 
                            theta[2] * MOM.iloc[t, i])) 
                  * (returns.iloc[t+1, i] if t+1 < T else 0)
                for i in range(N)) for t in range(T)]
        utility = sum(CRRA_utility(r) for r in rp)
        return -utility / (T) # Minimize the negative utility (equivalent to maximizing the utility)

    # Boucle de rolling optimization
    optimal_thetas = {} # Dictionnaire pour stocker les coefficients optimaux pour chaque période
    start_year = data.index.min() # Date de début fixée à la première date dans les données de rendement
    initial_end_year = pd.to_datetime('1973-12-01') # Date de fin initiale fixée à décembre 1973
    current_end_year = initial_end_year # Date de fin courante, commence par initial_end_year et sera ajustée dans la boucle


    while current_end_year <= data.index.max(): # Boucle jusqu'à la dernière date dans les données de rendement 
        
        # Extraire les composantes nécessaires pour l'optimisation des coefficients pour la période sélectionnée dans la window
        mkt_weights_filtred = mkt_weights.loc[start_year:current_end_year]
        MC_filtred = MC_standardized.loc[start_year:current_end_year]
        BM_filtred = BM_standardized.loc[start_year:current_end_year]
        MOM_filtred = MOM_standardized.loc[start_year:current_end_year]
        returns_filtred = returns.loc[start_year:current_end_year]
        N = len(mkt_weights.columns)
        M = 3
        T = len(returns_filtred)

        # Initial guess for theta using the average regression coefficients for the entire period 
        optimal_initial_theta = calculate_initial_regression_coefficients(N, M, data, MC_standardized, BM_standardized, MOM_standardized) 
    
        # Run the optimization for the current period
        result = minimize(objective, optimal_initial_theta, args=(mkt_weights_filtred, MC_filtred, BM_filtred, MOM_filtred, returns_filtred, T, N, M), method='SLSQP')

        # Store the optimal theta for the current period
        optimal_thetas[current_end_year] = result.x if result.success else None

        # Updater le theta initial pour la prochaine itération de la fenêtre en utilisant le theta optimal de la dernière itération
        # pour affiner la convergence vers une solution optimale pour les thetas optimaux, plus stable dans le temps
        if result.success:
            optimal_initial_theta = result.x

        # Élargir la fenêtre pour la prochaine itération de window en ajoutant 12 mois à la dernière date courante de fin 
        current_end_year += pd.DateOffset(months=12)

    return optimal_thetas


def plot_theta_series(optimal_thetas):
    """
    Cette fonction convertit le dictionnaire des coefficients optimaux en DataFrame et trace une série temporelle pour chaque coefficient.

    Paramètres:
    - optimal_thetas : Dictionnaire contenant les coefficients optimaux pour chaque période de temps.

    Retourne:
    - Plot : Série temporelle pour chaque coefficient theta à travers le temps (T).
    """
    
    
    # Convertir le dictionnaire en DataFrame 
    df = pd.DataFrame.from_dict(optimal_thetas, orient='index')

    # Tracer chaque colonne du DataFrame (chaque coefficient theta optimisés)
    plt.figure(figsize=(12,6))
    for i in range(df.shape[1]):
        plt.plot(df.index, df[i], label=f'Theta {i+1}')

    plt.title('Séries temporelles des coefficients theta')
    plt.legend()
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gca().xaxis.set_major_locator(mdates.YearLocator())
    plt.gcf().autofmt_xdate()

    plt.show()
    
    
def portfolio_weight_function(mkt_weights, MC_standardized, BM_standardized, MOM_standardized, optimal_theta):
    """
    Calcule les poids optimisés pour chaque actif en multipliant les coefficients optimaux par les caractéristiques standardisées 
    et en ajustant les poids du marché pour ces ajustements, comme décrit dans l'équation (3) de l'article.
    
    Parameters:
    - mkt_weights (pd.Series): Les poids du marché pour chaque actif au temps t.
    - MC_standardized (pd.Series): La caractéristique de capitalisation boursière standardisée pour chaque actif au temps t.
    - BM_standardized (pd.Series): La caractéristique de Book-to-Market standardisée pour chaque actif au temps t.
    - MOM_standardized (pd.Series): La caractéristique de Momentum standardisée pour chaque actif au temps t.
    - optimal_theta (np.array): Le vecteur de coefficients theta optimaux pour l'année en cours pour chaque caractéristique.
    
    Returns:
    - Les poids optimisés pour chaque actif, normalisés pour que leur somme soit égale à 1 comme dans l'équation (16) de l'article.
    """
    
    N = len(mkt_weights)
    optimized_weights = np.zeros(N)
    
    # Calculer les pondérations optimisées pour chaque actif en utilisant les coefficients optimaux
    # multipliés par les caractéristiques standardisées au temps t
    for i in range(N):
        initial_weight = mkt_weights.iloc[i]
        adjustment_caracteristics = (optimal_theta[0] * MC_standardized.iloc[i] + 
                      optimal_theta[1] * BM_standardized.iloc[i] + 
                      optimal_theta[2] * MOM_standardized.iloc[i]) / N
        optimized_weights[i] = initial_weight + adjustment_caracteristics

    # Assurer la positivité des poids optimisés avant de les normaliser 
    positive_weights = np.maximum(0, optimized_weights)
    
    # Normaliser les pondérations de portefeuille pour que leur somme soit égale à 1, 
    # comme la méthode décrite par l'équation (16) de l'article
    normalized_weights = positive_weights / np.sum(positive_weights)
    
    return normalized_weights


def calculate_optimized_weights_monthly(mkt_weights, MC_standardized, BM_standardized, MOM_standardized, optimal_thetas):
    """
    Calcule les poids optimisés du portefeuille pour chaque mois, en utilisant la fonction calculate_optimized_weights définies selon l'équation (3).
    
    Parameters:
    - 'mkt_weights' (pd.DataFrame): Les poids du marché pour chaque actif pour chaque mois.
    - 'MC_standardized' (pd.DataFrame): La caractéristique de capitalisation boursière standardisée pour chaque actif pour chaque mois.
    - 'BM_standardized' (pd.DataFrame): La caractéristique de Book-to-Market standardisée pour chaque actif pour chaque mois.
    - 'MOM_standardized' (pd.DataFrame): La caractéristique de Momentum standardisée pour chaque actif pour chaque mois.
    - 'optimal_thetas' (dict): Un dictionnaire où chaque clé est une année et la valeur est le vecteur de coefficients theta optimaux pour cette année 
    (12 fois le même vecteur par an).
    
    Returns:
    - 'pd.DataFrame': Un DataFrame où chaque ligne correspond à un mois et chaque colonne correspond à un actif, 
    et les valeurs sont les poids optimisés selon les ajustements pour les coefficients pour chaque actif.
    """
    
    
    optimized_weights_monthly = {}

    start_month = pd.Timestamp('1974-01-01')
    filtered_months = mkt_weights[start_month:].index

    for current_month in filtered_months:
        current_year = pd.Timestamp(year=current_month.year, month=12, day=1)
        if current_year not in optimal_thetas:
            continue
        mkt_weights_for_month = mkt_weights.loc[current_month]
        MC_standardized_month = MC_standardized.loc[current_month]
        BM_standardized_month = BM_standardized.loc[current_month]
        MOM_standardized_month = MOM_standardized.loc[current_month]
        
        # Appliquer les coefficients theta constants à travers les actifs et dans le temps
        optimized_weights = portfolio_weight_function(mkt_weights_for_month, MC_standardized_month, BM_standardized_month, MOM_standardized_month, optimal_thetas[current_year])
        
        optimized_weights_monthly[current_month] = optimized_weights
        
    # Convertir le dictionnaire en DataFrame pour une meilleure visualisation
    optimized_weights_monthly_df = pd.DataFrame.from_dict(optimized_weights_monthly, orient='index', columns=mkt_weights.columns)

    return optimized_weights_monthly_df


def calculate_portfolio_returns(opt_weights_monthly, mkt_weights, returns):
    """
    Calcule les rendements du portefeuille optimisé et du portefeuille de marché (Benchmark portfolio) selon les poids mensuels optimisés avec l'équation (3)
    et les rendements des actifs.

    Parameters:
    - opt_weights_monthly (DataFrame) : DataFrame contenant les poids mensuels du portefeuille parametric optimisé.
    - mkt_weights (DataFrame) : DataFrame contenant les poids du portefeuille de marché pour chaque actif.
    - returns (DataFrame) : DataFrame contenant les rendements des actifs shiftés d'une période. 

    Return:
    - Tuple contenant deux DataFrames : les rendements mensuels du portefeuille optimisé et les rendements mensuels du portefeuille de marché.
    """
    
    
    # Calculer les rendements du portefeuille optimisé en utilisant les poids mensuels optimisés
    optimized_portfolio_returns = (opt_weights_monthly * returns.reindex(opt_weights_monthly.index)).sum(axis=1)/100

    # Aligner les dates des poids du portefeuille de marché avec celles des rendements des actifs
    mkt_weights_aligned = mkt_weights.reindex(returns.reindex(opt_weights_monthly.index).index)
    # Calculer les rendements du portefeuille de marché (Benchmark portfolio) en utilisant les dates alignées
    market_portfolio_returns = (mkt_weights_aligned * returns.reindex(opt_weights_monthly.index)).sum(axis=1)/100

    # Convertir les séries en DataFrame 
    optimized_port_returns = pd.DataFrame(optimized_portfolio_returns, columns=['Portfolio Monthly Return'])
    market_port_returns = pd.DataFrame(market_portfolio_returns, columns=['Portfolio Monthly Return'])

    return optimized_port_returns, market_port_returns


def plot_cumulative_returns_partC(df, strategies):
    """
    Cette fonction trace les rendements cumulatifs pour les stratégies de portefeuilles spécifiées, de marché et optimisées selon les coefficients theta.

    Parameters:
    - df: DataFrame contenant les données de rendement pour les deux stratégies de portefeuilles.
    - strategies: Liste des deux stratégies à utiliser dans le tracé des rendements cumulatifs.
    """
    
    # Définir la taille de la figure à tracer
    plt.figure(figsize=(11,6))

    # Boucle sur chaque stratégie de portefeuille pour tracer les rendements cumulatifs 
    for strategy in strategies:
        # Récupérer les rendements de la stratégie
        mensual_returns = df[strategy]
        
        # Convertir les pourcentages en décimales et ajouter 1
        adjusted_returns = mensual_returns + 1 

        # Calculer les retours cumulatifs avec la fonction cumprod() qui fait le produit cumulatif des returns mensuels
        cumulative_returns = adjusted_returns.cumprod()
        
        # Tracer les rendements cumulatifs pour chaque stratégie
        plt.plot(cumulative_returns.index, cumulative_returns, label=strategy)
        
    
    plt.title('Rendements cumulatifs du portefeuille de benchmark et du portefeuille optimisé')
    plt.xlabel('Date')
    plt.ylabel('Rendements cumulatifs (%)')  
    plt.legend()
    plt.show()