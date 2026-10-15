class HeatmapParams:
    # Parametri per la generazione delle heatmap degli ostia coronarici
    
    # Sigma in mm (verrà poi convertito in voxel basandosi sullo spacing)
    # Un sigma di 2.0 - 3.0 mm è solitamente un buon punto di partenza
    sigma = 2.0 
    
    # Valore massimo della gaussiana
    amplitude = 1.0
    
    # Se True, genera due canali separati (uno per ostio), 
    # se False, genera un unico canale con label diverse
    multi_channel = False 
    
    # Identificativi nel tensore label
    labels_map = {
        0: "background",
        1: "RCA_ostium",
        2: "LCA_ostium" 
    }

    # Background threshold: azzera i valori della gaussiana sotto questa soglia
    # Utile per mantenere la maschera "sparsa"
    min_threshold = 0.01