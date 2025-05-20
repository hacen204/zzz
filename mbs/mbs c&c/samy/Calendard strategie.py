import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from IPython.display import display

def is_business_day(date):
    """
    Vérifie si une date est un jour ouvré (non weekend)
    
    Args:
        date (datetime): Date à vérifier
        
    Returns:
        bool: True si c'est un jour ouvré, False sinon
    """
    return date.weekday() < 5  # 0-4 correspond à lundi-vendredi

def get_previous_business_day(date):
    """
    Retourne le jour ouvré précédent
    
    Args:
        date (datetime): Date de référence
        
    Returns:
        datetime: Jour ouvré précédent
    """
    current = date - timedelta(days=1)
    while not is_business_day(current):
        current = current - timedelta(days=1)
    return current

def calculate_current_forward(date, principal, spot_price, coupon, repo_rate, yearly_cpr, roll_date):
    """
    Calcule le prix forward pour le contrat current
    
    Args:
        date (datetime): Date du calcul
        principal (float): Principal (nominal) du MBS
        spot_price (float): Prix spot
        coupon (float): Taux du coupon en % (sera converti en décimal)
        repo_rate (float): Taux repo en % (sera converti en décimal)
        yearly_cpr (float): CPR annuel (taux de remboursement anticipé)
        roll_date (int): Date de règlement dans le mois
        
    Returns:
        float: Prix forward ajusté
    """
    coupon = coupon / 100
    repo_rate = repo_rate / 100
    delivery_month = date.month if date.day < roll_date else (date.month + 1 if date.month < 12 else 1)
    delivery_year = date.year if delivery_month > 1 else date.year + 1
    delivery_date = datetime(delivery_year, delivery_month, roll_date)
    payment_date_current_month = datetime(date.year, date.month, 25)
    
    time_to_settlement_date = (delivery_date - date).days / 360
    time_to_payment_date = max((payment_date_current_month - date).days / 360, 0)

    ## ehc
    time_to_accrued = (delivery_date - payment_date_current_month).days / 360
    accrued = principal*coupon*time_to_accrued
    spot_price = spot_price + accrued 
    #####
    if date > payment_date_current_month:
        forward_price = spot_price / np.exp(-repo_rate * time_to_settlement_date)
        adjusted_forward_price = forward_price
    else:
        discount_payment_date = np.exp(-repo_rate * time_to_payment_date)
        discount_settlement_date = np.exp(-repo_rate * time_to_settlement_date)
        next_notional = principal * (1 - (1 + coupon / 12) ** -359) / (1 - (1 + coupon / 12) ** -360)
        scheduled_redemption = principal - next_notional
        #print('################ current #################')
        #print( 'principal', principal)
        #print('next notional',next_notional)
        unscheduled_redemption = next_notional * (1 - (1 - yearly_cpr) ** (1/12))
        #print( 'Cpr',yearly_cpr)
        #print('unscheduled redemption',unscheduled_redemption)
        cashflow_redemption = scheduled_redemption + unscheduled_redemption
        ### ehc
        N1 = principal - cashflow_redemption
        ###
        #print('cash flow with redemption', cashflow_redemption)
        cashflow_coupon = principal * coupon / 12
        #print( 'coupon', coupon)
        cashflow_discounted_at_payment_date = (cashflow_coupon + cashflow_redemption) * discount_payment_date
        
        forward_price_payment_date = (spot_price - cashflow_discounted_at_payment_date) / discount_payment_date
        #print( 'forward price at payment date',forward_price_payment_date)

        cashflow_redemption_reinvested = (cashflow_redemption) * (1 - forward_price_payment_date/N1)
        #print('cash flow with redemption reinvested', cashflow_redemption_reinvested)
        cashflow_with_reinvested_redemption = (cashflow_coupon + cashflow_redemption_reinvested) * discount_payment_date

    
        forward_price_settlement_date_with_reinvested_redemption = (spot_price - cashflow_with_reinvested_redemption) / discount_settlement_date - accrued
        #print('cash flow reinvested ',cashflow_with_reinvested_redemption)

        #print( 'forward price current ',forward_price_settlement_date_with_reinvested_redemption)
        adjusted_forward_price = forward_price_settlement_date_with_reinvested_redemption

    return adjusted_forward_price

def calculate_next_forward(date, principal, spot_price, coupon, repo_rate, yearly_cpr, roll_date):
    """
    Calcule le prix forward pour le contrat next
    
    Args:
        date (datetime): Date du calcul
        principal (float): Principal (nominal) du MBS
        spot_price (float): Prix spot
        coupon (float): Taux du coupon en % (sera converti en décimal)
        repo_rate (float): Taux repo en % (sera converti en décimal)
        yearly_cpr (float): CPR annuel (taux de remboursement anticipé)
        roll_date (int): Date de règlement dans le mois
        
    Returns:
        float: Prix forward ajusté
    """
    import numpy as np
    from datetime import datetime
    
    # Conversion des pourcentages en décimal
    coupon_decimal = coupon / 100
    repo_decimal = repo_rate / 100
    
    # Déterminer le mois de livraison pour le contrat next
    if date.day < roll_date:
        current_month = date.month
    else:
        current_month = date.month + 1 if date.month < 12 else 1
    
    next_month = current_month + 1 if current_month < 12 else 1
    next_year = date.year if (current_month < 12 or (current_month == 12 and date.day < roll_date)) else date.year + 1
    
    # Créer la date de livraison
    delivery_date = datetime(next_year, next_month, roll_date)
    
    # Déterminer les dates de paiement (max 2 paiements possibles)
    payment_dates = []
    
    # Premier paiement potentiel (25 du mois courant)
    current_payment = datetime(date.year, date.month, 25)
    if current_payment > date and current_payment < delivery_date:
        payment_dates.append(current_payment)
    
    # Deuxième paiement potentiel (25 du mois suivant)
    next_payment_month = date.month + 1 if date.month < 12 else 1
    next_payment_year = date.year if date.month < 12 else date.year + 1
    next_payment = datetime(next_payment_year, next_payment_month, 25)
    
    if next_payment < delivery_date:
        payment_dates.append(next_payment)
    
    # Calculer le temps jusqu'à la date de règlement
    time_to_settlement = (delivery_date - date).days / 360
    discount_settlement = np.exp(-repo_decimal * time_to_settlement)
    
    # S'il n'y a pas de cashflows entre la date d'évaluation et la date de livraison
    if not payment_dates:
        forward_price = spot_price / discount_settlement
        return forward_price
    
    # Initialisation des variables
    remaining_principal = principal
    cumulative_cashflow_forward_settlement = 0
    cumulative_cashflow_forward_payment = 0
    
    # Pour chaque date de paiement
    for i, payment_date in enumerate(payment_dates):
        # Calculer le facteur d'actualisation pour cette date
        time_to_payment = (payment_date - date).days / 360
        discount_factor = np.exp(-repo_decimal * time_to_payment)
        
        # Calculer le nouveau principal après amortissement standard
        #print( '@@@@@@@@@@@@@@@@@@@@@@@@ next @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
        next_notional =principal * (1 - (1 + coupon_decimal / 12) ** -(360-i-1)) / (1 - (1 + coupon_decimal / 12) ** -(360))*(1 - yearly_cpr) ** ((i+1)/12)
        #print( 'coupon decimal ', coupon_decimal)
        #print('payment date ',payment_date)
        #print('date ', date)
        #print( 'time to payment ', time_to_payment,'discount factor ', discount_factor)
        #print('nex notional ', next_notional)
        
        # Total des remboursements
        total_redemption = remaining_principal - next_notional
        #print( 'total redemption ', total_redemption)
        
        # Paiement du coupon
        coupon_payment = remaining_principal * coupon_decimal / 12
        #print('coupon payment ', coupon_payment)
        cumulative_cashflow_forward_payment +=(coupon_payment +total_redemption)*discount_factor

        # Prix forward à cette date pour le réinvestissement
        forward_price_at_payment = (spot_price - cumulative_cashflow_forward_payment) / discount_factor
        #print( ' forwarad bond at payment date ', forward_price_at_payment)
        # CORRECTION: Le réinvestissement utilise le prix forward en pourcentage
        # Mais sans diviser par 100 (car spot_price est déjà en %)
        reinvestment_effect = total_redemption * (1 - forward_price_at_payment)
        #print('reinvestissement amount', reinvestment_effect)
        
        # Cashflow ajusté
        adjusted_cashflow = coupon_payment + reinvestment_effect
        
        # Ajouter le cashflow actualisé au cumul
        cumulative_cashflow_forward_settlement += adjusted_cashflow * discount_factor
        
        # Mettre à jour le principal restant
        remaining_principal = next_notional
    
    # Calculer les intérêts courus entre le dernier paiement et la date de livraison
    last_payment_date = payment_dates[-1]
    time_since_last_payment = (delivery_date - last_payment_date).days / 360
    
    # CORRECTION: Calcul correct des intérêts courus
    accrued_interest = remaining_principal * coupon_decimal * (np.exp(repo_decimal*time_since_last_payment)-1)

    #print('accrued interest ', accrued_interest)
    # Prix forward final
    forward_price = (spot_price - cumulative_cashflow_forward_settlement) / discount_settlement - accrued_interest
    #print( ' forward price ', forward_price)
    
    return forward_price

def calculate_forward_forward(date, principal, spot_price, coupon, repo_rate, yearly_cpr, roll_date):
    """
    Calcule le prix forward pour le contrat forward_forward
    
    Args:
        date (datetime): Date du calcul
        principal (float): Principal (nominal) du MBS
        spot_price (float): Prix spot
        coupon (float): Taux du coupon en % (sera converti en décimal)
        repo_rate (float): Taux repo en % (sera converti en décimal)
        yearly_cpr (float): CPR annuel (taux de remboursement anticipé)
        roll_date (int): Date de règlement dans le mois
        
    Returns:
        float: Prix forward ajusté
    """
    import numpy as np
    from datetime import datetime
    
    # Conversion des pourcentages en décimal
    coupon_decimal = coupon / 100
    repo_decimal = repo_rate / 100
    
    # Déterminer le mois de livraison pour le contrat forward_forward
    if date.day < roll_date:
        current_month = date.month
    else:
        current_month = date.month + 1 if date.month < 12 else 1
    
    next_month = current_month + 1 if current_month < 12 else 1
    forward_month = next_month + 1 if next_month < 12 else 1
    
    forward_year = date.year
    if (next_month == 12 and forward_month == 1) or (current_month == 12 and next_month == 1):
        forward_year += 1
    if current_month == 11 and next_month == 12 and forward_month == 1:
        forward_year += 1
    
    # Créer la date de livraison
    delivery_date = datetime(forward_year, forward_month, roll_date)
    
    # Déterminer les dates de paiement (max 3 paiements possibles)
    payment_dates = []
    
    # Premier paiement potentiel (25 du mois courant)
    current_payment = datetime(date.year, date.month, 25)
    if current_payment > date and current_payment < delivery_date:
        payment_dates.append(current_payment)
    
    # Deuxième paiement potentiel (25 du mois suivant)
    next_payment_month = date.month + 1 if date.month < 12 else 1
    next_payment_year = date.year if date.month < 12 else date.year + 1
    next_payment = datetime(next_payment_year, next_payment_month, 25)
    
    if next_payment < delivery_date:
        payment_dates.append(next_payment)
    
    # Troisième paiement potentiel (25 du mois d'après)
    third_payment_month = next_payment_month + 1 if next_payment_month < 12 else 1
    third_payment_year = next_payment_year if next_payment_month < 12 else next_payment_year + 1
    third_payment = datetime(third_payment_year, third_payment_month, 25)
    
    if third_payment < delivery_date:
        payment_dates.append(third_payment)
    
    # Calculer le temps jusqu'à la date de règlement
    time_to_settlement = (delivery_date - date).days / 360
    discount_settlement = np.exp(-repo_decimal * time_to_settlement)
    
    # S'il n'y a pas de cashflows entre la date d'évaluation et la date de livraison
    if not payment_dates:
        forward_price = spot_price / discount_settlement
        return forward_price
    
    # Initialisation des variables
    remaining_principal = principal
    cumulative_cashflow_forward_settlement = 0
    cumulative_cashflow_forward_payment = 0
    
     # Pour chaque date de paiement
    for i, payment_date in enumerate(payment_dates):
        # Calculer le facteur d'actualisation pour cette date
        time_to_payment = (payment_date - date).days / 360
        discount_factor = np.exp(-repo_decimal * time_to_payment)
        
        # Calculer le nouveau principal après amortissement standard
   
        next_notional =principal * (1 - (1 + coupon_decimal / 12) ** -(360-i-1)) / (1 - (1 + coupon_decimal / 12) ** -(360))*(1 - yearly_cpr) ** ((i+1)/12)
        
        
        # Total des remboursements
        total_redemption = remaining_principal - next_notional
        
        # Paiement du coupon
        coupon_payment = remaining_principal * coupon_decimal / 12

        cumulative_cashflow_forward_payment +=(coupon_payment +total_redemption)*discount_factor
        
        # Prix forward à cette date pour le réinvestissement
        forward_price_at_payment = (spot_price - cumulative_cashflow_forward_payment) / discount_factor
        
        # CORRECTION: Le réinvestissement utilise le prix forward en pourcentage
        # Mais sans diviser par 100 (car spot_price est déjà en %)
        reinvestment_effect = total_redemption * (1 - forward_price_at_payment)
        
        # Cashflow ajusté
        adjusted_cashflow = coupon_payment + reinvestment_effect
        
        # Ajouter le cashflow actualisé au cumul
        cumulative_cashflow_forward_settlement += adjusted_cashflow * discount_factor
        
        # Mettre à jour le principal restant
        remaining_principal = next_notional 
    
    # Calculer les intérêts courus entre le dernier paiement et la date de livraison
    last_payment_date = payment_dates[-1]
    time_since_last_payment = (delivery_date - last_payment_date).days / 360
    
    # CORRECTION: Calcul correct des intérêts courus
    accrued_interest = remaining_principal * coupon_decimal * (np.exp(repo_decimal*time_since_last_payment)-1)
    
    
    # Prix forward final
    forward_price = (spot_price - cumulative_cashflow_forward_settlement) / discount_settlement - accrued_interest
    
    return forward_price

def get_valid_days(df_month, max_days=2):
    """
    Retourne les deux premiers jours valides parmi 13, 12, 11, 10
    
    Args:
        df_month: DataFrame contenant les données d'un mois
        max_days: Nombre maximum de jours à retourner (par défaut: 2)
        
    Returns:
        list: Liste des jours valides (maximum max_days)
    """
    valid_days = []
    days_to_check = [13, 12, 11, 10]  # Jours à vérifier
    
    unique_days = df_month['Date'].dt.day.unique()
    
    for day in days_to_check:
        if day in unique_days:
            valid_days.append(day)
            if len(valid_days) >= max_days:
                break
                
    return valid_days

def calculate_cpr(coupon, mortgage_rate):
    """
    Calcule le CPR en fonction du coupon et du taux hypothécaire
    
    Args:
        coupon (float): Taux du coupon
        mortgage_rate (float): Taux hypothécaire
        
    Returns:
        float: CPR calculé
    """
    alpha = 2.5
    results = 0.05 + (0.45 - 0.05) * (1 / (1 + np.exp(-alpha * (coupon - mortgage_rate))))
    return results

def get_month_column_name(month_num):
    """Convertit un numéro de mois en nom de colonne ('Jan', 'Feb', etc.)"""
    months_map = {
        1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr',
        5: 'May', 6: 'Jun', 7: 'Jul', 8: 'Aug',
        9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'
    }
    return months_map[month_num]

def get_month_number(month_name):
    """Convertit un nom de mois ('Jan', 'Feb', etc.) en numéro de mois"""
    months_map = {
        'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4,
        'May': 5, 'Jun': 6, 'Jul': 7, 'Aug': 8,
        'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
    }
    return months_map[month_name]

def calculate_net_basis(df):
    """
    Calcule les net basis pour les trois types de contrats: current, next, et forward
    
    Args:
        df (pandas.DataFrame): DataFrame contenant les données
        
    Returns:
        pandas.DataFrame: DataFrame avec les résultats des calculs de net basis
    """
    # Remplacer les valeurs NA par NaN
    df.replace('#N/A N/A', np.nan, inplace=True)
    
    # Convertir la colonne Date au format datetime si nécessaire
    if not pd.api.types.is_datetime64_any_dtype(df['Date']):
        df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%y')
    
    # Calculer la CPR dynamique si nécessaire
    if 'CPR_Dynamic' not in df.columns:
        df['CPR_Dynamic'] = df.apply(lambda row: calculate_cpr(row['Coupon'], row['Mortgage rate']), axis=1)
    
    # Liste pour stocker les résultats
    results = []
    
    # Dictionnaire pour stocker les résultats temporaires (pour le lissage sur deux jours)
    temp_results = {}
    
    # Regrouper par mois pour le traitement des jours valides
    df_by_month = df.groupby(df['Date'].dt.to_period('M'))
    
    print("Traitement des données par mois...")
    
    # Traiter chaque mois
    for period, df_month in df_by_month:
        # Déterminer les jours valides pour ce mois
        valid_days = get_valid_days(df_month, max_days=2)
        
        if not valid_days:
            print(f"Pas de jours valides pour {period}")
            continue
        
        # Traiter chaque jour valide
        for roll_date in valid_days:
            # Filtrer les données pour ce jour
            df_roll_date = df_month[df_month['Date'].dt.day == roll_date]
            
            # Traiter chaque ligne
            for _, row in df_roll_date.iterrows():
                date = row['Date']
                repo_rate = row['Real repo']
                
                # Déterminer la CPR à utiliser
                coupon_value = row['Coupon']
                cpr_column = f"CPR Realized {coupon_value}"
                
                if cpr_column in df.columns and pd.notna(row[cpr_column]):
                    cpr_realized = row[cpr_column]
                else:
                    cpr_realized = row['CPR_Dynamic']
                
                cpr_model = row['CPR_Dynamic']
                
                # Traiter les mois disponibles
                months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                
                # 1. Traiter le net basis "current"
                for i in range(len(months) - 1):
                    current_month = months[i]
                    next_month = months[i + 1]
                    
                    if pd.notna(row[current_month]) and pd.notna(row[next_month]):
                        # Convertir les prix en décimal
                        current_price = float(row[current_month]) / 100
                        next_price = float(row[next_month]) / 100
                        
                        # Calculer les forwards avec CPR réalisé et modèle
                        current_forward_realized = calculate_current_forward(
                            date, 1, current_price, coupon_value, repo_rate, cpr_realized, roll_date
                        )
                        
                        current_forward_model = calculate_current_forward(
                            date, 1, current_price, coupon_value, repo_rate, cpr_model, roll_date
                        )
                        
                        # Calculer la sensibilité au CPR
                        cpr_shifted = cpr_model * 1.10
                        current_forward_shifted = calculate_current_forward(
                            date, 1, current_price, coupon_value, repo_rate, cpr_shifted, roll_date
                        )
                        
                        delta_cpr = current_forward_shifted - current_forward_model
                        
                        # Calculer les net basis et gains
                        if not np.isnan(current_forward_model):
                            net_basis_realized = 100 * (current_forward_realized - next_price) / current_price
                            gain_realized = -net_basis_realized
                            
                            net_basis_expected = 100 * (current_forward_model - next_price) / current_price
                            gain_expected = -net_basis_expected
                            
                            implied_repo_rate_spread = -(net_basis_expected / current_price) * 12
                            
                            # Créer une clé unique pour ce calcul
                            key = (period.strftime('%Y-%m'), coupon_value, 'current', f"{current_month}-{next_month}")
                            
                            if key not in temp_results:
                                temp_results[key] = []
                            
                            temp_results[key].append({
                                "Date": date,
                                "Year": date.year,
                                "Month": date.month,
                                "Day": date.day,
                                "Coupon": coupon_value,
                                "Forward Type": "current",
                                "Spot Month": current_month,
                                "TBA Month": next_month,
                                "Spot Price": current_price,
                                "TBA Price": next_price,
                                "Forward Price Realized": current_forward_realized,
                                "Forward Price Expected": current_forward_model,
                                "Net Basis Realized": net_basis_realized,
                                "Gain Realized": gain_realized,
                                "Net Basis Expected": net_basis_expected,
                                "Gain Expected": gain_expected,
                                "Implied Repo Rate Spread": implied_repo_rate_spread,
                                "Delta CPR 10%": 100 * delta_cpr,
                                "CPR Realized": cpr_realized,
                                "CPR Model": cpr_model,
                                "Roll Day": roll_date
                            })
                        
                        # On ne traite que la première paire de mois valide pour le current
                        break
                
                # 2. Traiter le net basis "next"
                for i in range(len(months) - 2):
                    next_month = months[i + 1]
                    forward_month = months[i + 2]
                    
                    if pd.notna(row[next_month]) and pd.notna(row[forward_month]):
                        # Convertir les prix en décimal
                        next_price = float(row[next_month]) / 100
                        forward_price = float(row[forward_month]) / 100
                        
                        # Calculer les forwards avec CPR réalisé et modèle
                        next_forward_realized = calculate_next_forward(
                            date, 1, next_price, coupon_value, repo_rate, cpr_realized, roll_date
                        )
                        
                        next_forward_model = calculate_next_forward(
                            date, 1, next_price, coupon_value, repo_rate, cpr_model, roll_date
                        )
                        
                        # Calculer la sensibilité au CPR
                        cpr_shifted = cpr_model * 1.10
                        next_forward_shifted = calculate_next_forward(
                            date, 1, next_price, coupon_value, repo_rate, cpr_shifted, roll_date
                        )
                        
                        delta_cpr = next_forward_shifted - next_forward_model
                        
                        # Calculer les net basis et gains
                        if not np.isnan(next_forward_model):
                            net_basis_realized = 100 * (next_forward_realized - forward_price) / next_price
                            gain_realized = -net_basis_realized
                            
                            net_basis_expected = 100 * (next_forward_model - forward_price) / next_price
                            gain_expected = -net_basis_expected
                            
                            implied_repo_rate_spread = -(net_basis_expected / next_price) * 12
                            
                            # Créer une clé unique pour ce calcul
                            key = (period.strftime('%Y-%m'), coupon_value, 'next', f"{next_month}-{forward_month}")
                            
                            if key not in temp_results:
                                temp_results[key] = []
                            
                            temp_results[key].append({
                                "Date": date,
                                "Year": date.year,
                                "Month": date.month,
                                "Day": date.day,
                                "Coupon": coupon_value,
                                "Forward Type": "next",
                                "Spot Month": next_month,
                                "TBA Month": forward_month,
                                "Spot Price": next_price,
                                "TBA Price": forward_price,
                                "Forward Price Realized": next_forward_realized,
                                "Forward Price Expected": next_forward_model,
                                "Net Basis Realized": net_basis_realized,
                                "Gain Realized": gain_realized,
                                "Net Basis Expected": net_basis_expected,
                                "Gain Expected": gain_expected,
                                "Implied Repo Rate Spread": implied_repo_rate_spread,
                                "Delta CPR 10%": 100 * delta_cpr,
                                "CPR Realized": cpr_realized,
                                "CPR Model": cpr_model,
                                "Roll Day": roll_date
                            })
                        
                        # On ne traite que la première paire de mois valide pour le next
                        break
                
                # 3. Traiter le net basis "forward"
                for i in range(len(months) - 3):
                    forward_month = months[i + 2]
                    far_forward_month = months[i + 3]
                    
                    if pd.notna(row[forward_month]) and pd.notna(row[far_forward_month]):
                        # Convertir les prix en décimal
                        forward_price = float(row[forward_month]) / 100
                        far_forward_price = float(row[far_forward_month]) / 100
                        
                        # Calculer les forwards avec CPR réalisé et modèle
                        forward_forward_realized = calculate_forward_forward(
                            date, 1, forward_price, coupon_value, repo_rate, cpr_realized, roll_date
                        )
                        
                        forward_forward_model = calculate_forward_forward(
                            date, 1, forward_price, coupon_value, repo_rate, cpr_model, roll_date
                        )
                        
                        # Calculer la sensibilité au CPR
                        cpr_shifted = cpr_model * 1.10
                        forward_forward_shifted = calculate_forward_forward(
                            date, 1, forward_price, coupon_value, repo_rate, cpr_shifted, roll_date
                        )
                        
                        delta_cpr = forward_forward_shifted - forward_forward_model
                        
                        # Calculer les net basis et gains
                        if not np.isnan(forward_forward_model):
                            net_basis_realized = 100 * (forward_forward_realized - far_forward_price) / forward_price
                            gain_realized = -net_basis_realized
                            
                            net_basis_expected = 100 * (forward_forward_model - far_forward_price) / forward_price
                            gain_expected = -net_basis_expected
                            
                            implied_repo_rate_spread = -(net_basis_expected / forward_price) * 12
                            
                            # Créer une clé unique pour ce calcul
                            key = (period.strftime('%Y-%m'), coupon_value, 'forward', f"{forward_month}-{far_forward_month}")
                            
                            if key not in temp_results:
                                temp_results[key] = []
                            
                            temp_results[key].append({
                                "Date": date,
                                "Year": date.year,
                                "Month": date.month,
                                "Day": date.day,
                                "Coupon": coupon_value,
                                "Forward Type": "forward",
                                "Spot Month": forward_month,
                                "TBA Month": far_forward_month,
                                "Spot Price": forward_price,
                                "TBA Price": far_forward_price,
                                "Forward Price Realized": forward_forward_realized,
                                "Forward Price Expected": forward_forward_model,
                                "Net Basis Realized": net_basis_realized,
                                "Gain Realized": gain_realized,
                                "Net Basis Expected": net_basis_expected,
                                "Gain Expected": gain_expected,
                                "Implied Repo Rate Spread": implied_repo_rate_spread,
                                "Delta CPR 10%": 100 * delta_cpr,
                                "CPR Realized": cpr_realized,
                                "CPR Model": cpr_model,
                                "Roll Day": roll_date
                            })
                        
                        # On ne traite que la première paire de mois valide pour le forward-forward
                        break
    
    # Calculer les moyennes pour chaque clé (lissage sur deux jours)
    for key, entries in temp_results.items():
        if not entries:
            continue
        
        # S'il n'y a qu'une seule entrée, l'utiliser directement
        if len(entries) == 1:
            results.append(entries[0])
            continue
        
        # Calculer la moyenne de chaque champ numérique
        avg_entry = entries[0].copy()  # Commencer avec le premier entrée pour les champs non-numériques
        
        numeric_fields = [
            "Spot Price", "TBA Price", "Forward Price Realized", "Forward Price Expected",
            "Net Basis Realized", "Gain Realized", "Net Basis Expected", "Gain Expected",
            "Implied Repo Rate Spread", "Delta CPR 10%", "CPR Realized", "CPR Model"
        ]
        
        for field in numeric_fields:
            values = [entry[field] for entry in entries]
            avg_entry[field] = sum(values) / len(values)
        
        # Ajouter des champs pour indiquer le lissage
        avg_entry["Roll Days Used"] = ", ".join(str(entry["Roll Day"]) for entry in entries)
        avg_entry["Num Days Averaged"] = len(entries)
        
        results.append(avg_entry)
    
    # Convertir les résultats en DataFrame
    net_basis_df = pd.DataFrame(results)
    
    # Ajouter le gain cumulé par type de forward
    if not net_basis_df.empty and 'Date' in net_basis_df.columns and 'Gain Realized' in net_basis_df.columns:
        for forward_type in net_basis_df['Forward Type'].unique():
            type_df = net_basis_df[net_basis_df['Forward Type'] == forward_type].copy()
            
            # Trier par date
            type_df = type_df.sort_values('Date')
            
            # Calculer le gain moyen par date
            gain_by_date = type_df.groupby('Date')['Gain Realized'].mean().reset_index()
            gain_by_date[f'Cumulative Gain {forward_type}'] = gain_by_date['Gain Realized'].cumsum()
            
            # Fusionner avec le DataFrame principal
            net_basis_df = pd.merge(
                net_basis_df, 
                gain_by_date[['Date', f'Cumulative Gain {forward_type}']], 
                on='Date', 
                how='left'
            )
    
    return net_basis_df

def create_visualizations(net_basis_df, output_prefix):
    """
    Crée des visualisations des net basis et gains pour les trois types de forward
    
    Args:
        net_basis_df (pandas.DataFrame): DataFrame avec les résultats
        output_prefix (str): Préfixe pour les fichiers de sortie
    """
    # 1. Évolution des net basis par type de forward
    plt.figure(figsize=(12, 8))
    forward_types = net_basis_df['Forward Type'].unique()
    
    for forward_type in forward_types:
        type_df = net_basis_df[net_basis_df['Forward Type'] == forward_type]
        type_df = type_df.sort_values('Date')
        plt.plot(type_df['Date'], type_df['Net Basis Realized'], label=f"{forward_type} Realized", marker='o', alpha=0.7)
        plt.plot(type_df['Date'], type_df['Net Basis Expected'], label=f"{forward_type} Expected", linestyle='--', alpha=0.5)
    
    plt.title('Évolution des Net Basis par Type de Forward')
    plt.xlabel('Date')
    plt.ylabel('Net Basis')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_net_basis.png")
    plt.show()  # Ajout de cette ligne pour afficher le graphique
    
    # 2. Évolution des gains cumulés par type de forward
    plt.figure(figsize=(12, 8))
    
    for forward_type in forward_types:
        col_name = f'Cumulative Gain {forward_type}'
        if col_name in net_basis_df.columns:
            type_df = net_basis_df[net_basis_df['Forward Type'] == forward_type].copy()
            type_df = type_df.sort_values('Date')
            plt.plot(type_df['Date'], type_df[col_name], label=f"{forward_type}", marker='o', alpha=0.7)
    
    plt.title('Évolution des Gains Cumulés par Type de Forward')
    plt.xlabel('Date')
    plt.ylabel('Gain Cumulé')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_cumulative_gains.png")
    plt.show()  # Ajout de cette ligne pour afficher le graphique
    
    # 3. Distribution des net basis par type de forward (boxplot)
    plt.figure(figsize=(15, 6))
    
    # Créer des sous-graphiques pour realized et expected
    plt.subplot(1, 2, 1)
    data_realized = [net_basis_df[net_basis_df['Forward Type'] == ftype]['Net Basis Realized'] for ftype in forward_types]
    plt.boxplot(data_realized, labels=forward_types)
    plt.title('Distribution des Net Basis Realized')
    plt.ylabel('Net Basis Realized')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.subplot(1, 2, 2)
    data_expected = [net_basis_df[net_basis_df['Forward Type'] == ftype]['Net Basis Expected'] for ftype in forward_types]
    plt.boxplot(data_expected, labels=forward_types)
    plt.title('Distribution des Net Basis Expected')
    plt.ylabel('Net Basis Expected')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_net_basis_distribution.png")
    plt.show()  # Ajout de cette ligne pour afficher le graphique
    
    # 4. Comparaison des forward prices vs TBA prices par type
    fig, axes = plt.subplots(len(forward_types), 1, figsize=(12, 4*len(forward_types)))
    
    for i, forward_type in enumerate(forward_types):
        ax = axes[i] if len(forward_types) > 1 else axes
        type_df = net_basis_df[net_basis_df['Forward Type'] == forward_type].copy()
        type_df = type_df.sort_values('Date')
        
        ax.plot(type_df['Date'], type_df['Forward Price Expected'], label='Forward Price', marker='o', color='blue')
        ax.plot(type_df['Date'], type_df['TBA Price'], label='TBA Price', marker='s', color='red')
        ax.set_title(f'Forward vs TBA Price - {forward_type}')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_forward_vs_tba.png")
    plt.show()  # Ajout de cette ligne pour afficher le graphique

def main():
    """
    Fonction principale pour calculer et visualiser les net basis
    """
    # Chemins des fichiers
    input_file = "TBA_database_reduced.csv"  # Remplacer par votre chemin d'entrée
    output_file = "net_basis_results.csv"    # Remplacer par votre chemin de sortie
    
    print(f"Chargement des données depuis {input_file}...")
    try:
        df = pd.read_csv(input_file)
        print(f"Données chargées: {len(df)} lignes")
    except Exception as e:
        print(f"Erreur lors du chargement des données: {e}")
        return
    
    print("Calcul des net basis pour les trois types de forward...")
    net_basis_df = calculate_net_basis(df)
    
    print(f"Enregistrement des résultats dans {output_file}...")
    net_basis_df.to_csv(output_file, index=False)
    
    # Afficher quelques statistiques
    print("\nStatistiques des résultats:")
    print(f"Nombre total de calculs: {len(net_basis_df)}")
    
    if not net_basis_df.empty:
        # Statistiques par type de forward
        forward_types = net_basis_df['Forward Type'].unique()
        for forward_type in forward_types:
            type_df = net_basis_df[net_basis_df['Forward Type'] == forward_type]
            
            print(f"\nType de forward: {forward_type}")
            print(f"  Nombre d'observations: {len(type_df)}")
            
            nb_realized_mean = type_df['Net Basis Realized'].mean()
            nb_realized_std = type_df['Net Basis Realized'].std()
            print(f"  Net Basis Realized: moyenne = {nb_realized_mean:.4f}, écart-type = {nb_realized_std:.4f}")
            
            nb_expected_mean = type_df['Net Basis Expected'].mean()
            nb_expected_std = type_df['Net Basis Expected'].std()
            print(f"  Net Basis Expected: moyenne = {nb_expected_mean:.4f}, écart-type = {nb_expected_std:.4f}")
            
            gain_realized_mean = type_df['Gain Realized'].mean()
            gain_realized_std = type_df['Gain Realized'].std()
            print(f"  Gain Realized: moyenne = {gain_realized_mean:.4f}, écart-type = {gain_realized_std:.4f}")
            
            # Taux de réussite (% de gains positifs)
            win_rate = len(type_df[type_df['Gain Realized'] > 0]) / len(type_df) * 100
            print(f"  Taux de réussite: {win_rate:.2f}%")
    
    # Créer les visualisations
    print("\nCréation des visualisations...")
    output_prefix = output_file.replace('.csv', '')
    create_visualizations(net_basis_df, output_prefix)
    
    print("Traitement terminé avec succès!")

if __name__ == "__main__":
    main()