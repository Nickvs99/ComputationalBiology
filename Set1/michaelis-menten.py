def michaelis_menten_equation(S, Vmax, Km):
    """
    Calculates the overall rate calculated with the Michaelis-Menten equation. 

    Arguments:
        S (float): substrate concentration
        Vmax (float): maximum rate
        Km (float): Michaelis constant

    Returns:
        v: overall rate
    """
    
    return Vmax * S / (Km + S)
