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
import matplotlib.dates as mdates
from datetime import datetime
import os


def importClean_10ind(csv_file_path, desired_returns):
    """
    Cette fonction importe des données à partir d'un fichier CSV spécifié, effectue un nettoyage de données,
    et retourne un DataFrame contenant des données filtrées. 

    Args:
        csv_file_path(str): Chemin vers le fichier CSV avec les données brutes.

    Returns:
        DataFrame: DataFrame contenant les données historiques filtrées pour les 10 industries.
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
    
    return df


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
    Calcul les poids optimisés des actifs dans le portefeuille qui maximisent le ratio de Sharpe avec et sans contrainte de vente à découvert,
    en utilisant le solveur de programmation linéaire GUROBI dans le cas avec contrainte sur vente à découverte et analytiquement pour le cas 
    sans contrainte de vente à découvert.

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
    Calcul les poids optimaux qui minimisent la variance du portefeuille avec une contrainte de budget dans les deux cas avec et sans contrainte de vente à découvert,
    en utilisant le solveur de programmation linéaire GUROBI.
    
    Parameters:
    - z_bar (Series): Série pandas contenant les rendements attendus pour chaque actif du portefeuille d'industrie.
    - Sigma (DataFrame): DataFrame pandas représentant la matrice de covariance des rendements des actifs du portefeuille. 
    - short_allowed (bool): Booléen indiquant si les positions de ventes à découvert sont autorisées (True) ou non autorisées (False).

    Returns:
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

    Returns:
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
    
    Returns:
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
    
    Returns : 
    - Plot des rendements cumulatifs pour les 7 stratégies de portefeuilles d'industries.
    """
    
    
    def cumulative_returns_calculation(mensual_returns):
        """
        Cette fonction calcule les retours cumulatifs à partir des séries temporelles de rendements mensuels pour les 7 stratégies de portefeuille.

        Parameters:
        - mensual_returns: Séries pandas contenant les retours mensuels du portefeuille sélectionné en décimal.
        
        Returns : 
        - cumulative_returns: Séries pandas contenant les retours cumulatifs du portefeuille sélectionné en décimal.
        """
        
        
        # Convertir les pourcentages en décimales et ajouter 1
        adjusted_returns = mensual_returns / 100 + 1

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


def annualized_statistics_and_sharpe_ratios(strategies_results, periods, df_rf):
    """
    Calcule les statistiques annualisées pour les rendements hors échantillon de différentes stratégies de portefeuille sur des périodes spécifiées,
    et détermine les ratios de Sharpe annuels pour ces stratégies.

    Paramètres :
    - strategies_results (dict) : Dictionnaire de DataFrames, où chaque clé représente le nom d'une stratégie et chaque valeur est
    un DataFrame contenant les résultats pour les poids optimaux de cette stratégie, y compris les 'Rendements Mensuels du Portefeuille'.
    - periods (liste de tuples) : Liste de tuples contenant la date de début et la date de fin d'une période à analyser,
    au format ('AAAA-MM-JJ', 'AAAA-MM-JJ').
    - df_rf (DataFrame) : DataFrame contenant les taux sans risque mensuels.

    Retour :
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
            mean_return_annualized = (1 + mean_return_monthly/100) ** 12 - 1
            std_return_annualized = std_return_monthly * np.sqrt(12)
            
            # Convertir en pourcentage pour faciliter les manipulations par la suite
            mean_return_annualized *= 100  

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