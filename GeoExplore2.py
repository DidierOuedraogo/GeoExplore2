import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import base64
from datetime import datetime
import colorsys
import scipy.stats as stats

# Configuration de la page
st.set_page_config(
    layout="wide", 
    page_title="GeoExplore Pro",
    page_icon="🔍"
)

# Thème personnalisé et styles CSS
st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: #2c3e50;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        font-size: 1.2rem;
        color: #7f8c8d;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #3498db;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        padding-bottom: 0.3rem;
        border-bottom: 2px solid #3498db;
    }
    .card {
        background-color: #f8f9fa;
        border-radius: 5px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        border-left: 4px solid #3498db;
    }
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 1.1rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 4px 4px 0px 0px;
        padding: 10px 16px;
        background-color: #f0f2f6;
    }
    .stTabs [aria-selected="true"] {
        background-color: #3498db !important;
        color: white !important;
    }
    .highlight {
        background-color: #e1f5fe;
        padding: 0.5rem;
        border-radius: 4px;
    }
    .sidebar-info {
        font-size: 0.85rem;
        color: #7f8c8d;
    }
    .sidebar-header {
        font-size: 1.1rem;
        font-weight: 600;
        margin-top: 1.2rem;
        margin-bottom: 0.8rem;
        color: #2c3e50;
    }
    .footer {
        text-align: center;
        margin-top: 2rem;
        padding-top: 1rem;
        border-top: 1px solid #eaeaea;
        font-size: 0.8rem;
        color: #95a5a6;
    }
    .stat-card {
        background-color: white;
        border-radius: 5px;
        padding: 1rem;
        margin-bottom: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24);
    }
    .stat-number {
        font-size: 1.8rem;
        font-weight: 700;
        color: #3498db;
    }
    .stat-label {
        font-size: 0.9rem;
        color: #7f8c8d;
    }
    .data-type-btn {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 5px;
        padding: 1rem;
        margin: 0.5rem;
        text-align: center;
        cursor: pointer;
        transition: all 0.2s ease;
    }
    .data-type-btn:hover {
        background-color: #e9ecef;
        transform: translateY(-2px);
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .data-type-btn.selected {
        background-color: #3498db;
        color: white;
        border-color: #2980b9;
    }
    .data-type-btn-icon {
        font-size: 2rem;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Logo et titre
col1, col2 = st.columns([1, 5])
with col1:
    st.markdown("# 🔍")
with col2:
    st.markdown('<div class="main-title">GeoExplore Pro</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Plateforme avancée d\'analyse de données d\'exploration géologique</div>', unsafe_allow_html=True)

# Initialisation des variables de session
if 'show_guide' not in st.session_state:
    st.session_state.show_guide = False
if 'data_type' not in st.session_state:
    st.session_state.data_type = None
if 'current_tab' not in st.session_state:
    st.session_state.current_tab = "Analyse"
if 'datasets' not in st.session_state:
    st.session_state.datasets = {}
if 'active_dataset' not in st.session_state:
    st.session_state.active_dataset = None

# Fonction pour afficher le guide d'utilisateur
def show_user_guide():
    st.markdown('<div class="section-header">Guide d\'Utilisation</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="card">
    <h3>Bienvenue dans GeoExplore Pro</h3>
    <p>Cette application est conçue pour aider les géologues à analyser rapidement leurs données d'exploration. Voici comment l'utiliser:</p>
    </div>
    """, unsafe_allow_html=True)
    
    with st.expander("📂 ÉTAPE 1: Choisir le type de données et les charger"):
        st.markdown("""
        - L'application accepte différents types de données géologiques:
          - **Données de Forage**: Collar, Survey, Lithologie, Analyses, Densité
          - **Composites 3D**: Données composites avec coordonnées X,Y,Z
          - **Données Statistiques**: Tout autre type de données (sans coordonnées nécessaires)
        - Chargez vos fichiers CSV ou Excel pour chaque catégorie
        """)
    
    with st.expander("🔍 ÉTAPE 2: Sélectionner les colonnes pertinentes"):
        st.markdown("""
        - Après le chargement, identifiez les colonnes importantes pour l'analyse:
          - ID de trou/échantillon
          - Coordonnées (X, Y, Z) si disponibles
          - Intervalles (From/To) pour les données de forage
          - Variables numériques pour l'analyse statistique
        """)
    
    with st.expander("📊 ÉTAPE 3: Analyse statistique avancée"):
        st.markdown("""
        - **Statistiques descriptives**: Calculez et visualisez des statistiques personnalisées
          - Percentiles personnalisables (0-100%)
          - Histogrammes avec contrôle précis des paramètres (nombre de bins, échelle, etc.)
          - Boîtes à moustaches et violons pour examiner la distribution
        - **Analyse bivariée**: Nuages de points, matrices de corrélation
        """)
    
    with st.expander("🧮 ÉTAPE 4: Filtrage des données"):
        st.markdown("""
        - Filtrez vos données selon divers critères:
          - Par valeurs numériques (plages, seuils)
          - Par catégories (lithologie, méthode d'analyse, etc.)
          - Par localisation spatiale
        """)
    
    st.markdown("""
    <div class="footer">
    Pour toute question ou assistance, contactez le développeur.
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("Fermer le guide"):
        st.session_state.show_guide = False
        st.rerun()  # Correction: remplacer experimental_rerun par rerun

# Fonctions utilitaires
def detect_column_type(df, column_name):
    """Détecte automatiquement le type de colonne"""
    col_lower = column_name.lower()
    
    # Détection d'ID
    if any(term in col_lower for term in ['id', 'hole', 'dhid', 'bhid']):
        return 'ID'
    
    # Détection de coordonnées
    if col_lower in ['x', 'east', 'easting', 'longitude', 'lon', 'x_coord']:
        return 'X'
    if col_lower in ['y', 'north', 'northing', 'latitude', 'lat', 'y_coord']:
        return 'Y'
    if col_lower in ['z', 'elev', 'elevation', 'z_coord', 'rl']:
        return 'Z'
    
    # Détection d'intervalles
    if col_lower in ['from', 'depth_from', 'start', 'top', 'début']:
        return 'From'
    if col_lower in ['to', 'depth_to', 'end', 'bottom', 'fin']:
        return 'To'
    
    # Détection de lithologie
    if any(term in col_lower for term in ['lith', 'rock', 'geol']):
        return 'Lithology'
    
    # Détection d'éléments chimiques courants
    elements = {
        'au': 'Au', 'ag': 'Ag', 'cu': 'Cu', 'pb': 'Pb', 'zn': 'Zn', 
        'ni': 'Ni', 'co': 'Co', 'fe': 'Fe', 'mn': 'Mn', 'cr': 'Cr'
    }
    
    for element, label in elements.items():
        if element in col_lower and any(unit in col_lower for unit in ['ppm', 'ppb', 'pct', '%', 'g/t']):
            return f'Assay_{label}'
    
    # Détection de densité
    if any(term in col_lower for term in ['dens', 'sg', 'specific_gravity']):
        return 'Density'
    
    # Détection d'azimuth, dip pour les données survey
    if any(term in col_lower for term in ['azim', 'azi', 'bear']):
        return 'Azimuth'
    if any(term in col_lower for term in ['dip', 'incl', 'inclinaison']):
        return 'Dip'
    
    # Par défaut, déterminer si c'est numérique ou catégoriel
    if pd.api.types.is_numeric_dtype(df[column_name]):
        return 'Numeric'
    else:
        return 'Categorical'

def load_data(file, file_type):
    """Charge les données depuis un fichier CSV ou Excel"""
    try:
        if file.name.endswith('.csv'):
            return pd.read_csv(file)
        elif file.name.endswith(('.xls', '.xlsx')):
            return pd.read_excel(file)
        else:
            st.error(f"Format de fichier non pris en charge: {file.name}")
            return None
    except Exception as e:
        st.error(f"Erreur lors du chargement du fichier: {e}")
        return None

def get_numeric_columns(df):
    """Retourne la liste des colonnes numériques d'un DataFrame"""
    return [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]

def get_categorical_columns(df):
    """Retourne la liste des colonnes catégorielles d'un DataFrame"""
    return [col for col in df.columns if pd.api.types.is_categorical_dtype(df[col]) or 
            (pd.api.types.is_object_dtype(df[col]) and df[col].nunique() / len(df) < 0.5)]

def calculate_custom_statistics(df, column, percentiles=None):
    """Calcule des statistiques personnalisées pour une colonne"""
    if percentiles is None:
        percentiles = [0, 5, 10, 25, 50, 75, 90, 95, 100]
    
    stats = df[column].describe(percentiles=[p/100 for p in percentiles])
    stats_dict = stats.to_dict()
    
    # Ajouter des statistiques supplémentaires
    stats_dict['cv'] = stats_dict['std'] / stats_dict['mean'] if stats_dict['mean'] != 0 else float('nan')
    stats_dict['iqr'] = stats_dict[f"{percentiles[6]}%"] - stats_dict[f"{percentiles[2]}%"]
    stats_dict['skewness'] = df[column].skew()
    stats_dict['kurtosis'] = df[column].kurtosis()
    
    return stats_dict

def create_histogram(df, column, bins=20, density=False, log_x=False, log_y=False, kde=True):
    """Crée un histogramme personnalisé"""
    fig = px.histogram(
        df, x=column, nbins=bins, histnorm='density' if density else None,
        log_x=log_x, log_y=log_y,
        title=f"Distribution de {column}",
        color_discrete_sequence=["#3498db"]
    )
    
    if kde and not log_x:
        # Ajouter une courbe KDE
        x_range = np.linspace(df[column].min(), df[column].max(), 1000)
        kde = stats.gaussian_kde(df[column].dropna())
        y_kde = kde(x_range)
        
        # Ajuster l'échelle si histnorm est density
        if density:
            scale = 1
        else:
            scale = len(df) * (df[column].max() - df[column].min()) / bins
            y_kde *= scale
        
        fig.add_trace(
            go.Scatter(
                x=x_range, y=y_kde,
                mode='lines', name='KDE',
                line=dict(color='red', width=2)
            )
        )
    
    # Ajouter des lignes pour les statistiques clés
    mean_val = df[column].mean()
    median_val = df[column].median()
    
    fig.add_vline(x=mean_val, line_dash="dash", line_color="red", 
                annotation_text="Moyenne", annotation_position="top right")
    fig.add_vline(x=median_val, line_dash="dash", line_color="green", 
                annotation_text="Médiane", annotation_position="top left")
    
    fig.update_layout(
        height=500,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(240,242,246,0.8)',
    )
    
    return fig

def create_box_violin_plot(df, column, plot_type="box", group_by=None, orientation="vertical"):
    """Crée une boîte à moustaches ou un violon plot"""
    if plot_type == "box":
        if group_by:
            if orientation == "vertical":
                fig = px.box(df, x=group_by, y=column, color=group_by,
                         title=f"Distribution de {column} par {group_by}")
            else:
                fig = px.box(df, y=group_by, x=column, color=group_by,
                         title=f"Distribution de {column} par {group_by}")
        else:
            if orientation == "vertical":
                fig = px.box(df, y=column, title=f"Distribution de {column}")
            else:
                fig = px.box(df, x=column, title=f"Distribution de {column}")
    else:  # violin
        if group_by:
            if orientation == "vertical":
                fig = px.violin(df, x=group_by, y=column, color=group_by, box=True,
                             title=f"Distribution de {column} par {group_by}")
            else:
                fig = px.violin(df, y=group_by, x=column, color=group_by, box=True,
                             title=f"Distribution de {column} par {group_by}")
        else:
            if orientation == "vertical":
                fig = px.violin(df, y=column, box=True, points="all",
                             title=f"Distribution de {column}")
            else:
                fig = px.violin(df, x=column, box=True, points="all",
                             title=f"Distribution de {column}")
    
    fig.update_layout(
        height=500,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(240,242,246,0.8)',
    )
    
    return fig

def create_scatter_plot(df, x_col, y_col, color_col=None, trendline=False, log_x=False, log_y=False):
    """Crée un nuage de points avec options avancées"""
    # Créer une copie pour les transformations
    plot_data = df.copy()
    
    # Appliquer les transformations logarithmiques
    if log_x and (plot_data[x_col] > 0).all():
        plot_data[x_col] = np.log10(plot_data[x_col])
        x_title = f"log10({x_col})"
    else:
        x_title = x_col
        log_x = False
        
    if log_y and (plot_data[y_col] > 0).all():
        plot_data[y_col] = np.log10(plot_data[y_col])
        y_title = f"log10({y_col})"
    else:
        y_title = y_col
        log_y = False
    
    # Créer le graphique
    if color_col:
        fig = px.scatter(
            plot_data, x=x_col, y=y_col, 
            color=color_col,
            title=f"Relation entre {x_col} et {y_col}",
            labels={x_col: x_title, y_col: y_title},
            trendline='ols' if trendline else None
        )
    else:
        fig = px.scatter(
            plot_data, x=x_col, y=y_col,
            title=f"Relation entre {x_col} et {y_col}",
            labels={x_col: x_title, y_col: y_title},
            trendline='ols' if trendline else None
        )
    
    fig.update_layout(
        height=600,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(240,242,246,0.8)',
    )
    
    return fig

def create_correlation_heatmap(df, columns=None, method='pearson'):
    """Crée une carte de chaleur de corrélation"""
    if columns is None:
        # Sélectionner seulement les colonnes numériques
        columns = get_numeric_columns(df)
    
    # Calculer la matrice de corrélation
    corr_matrix = df[columns].corr(method=method)
    
    # Créer la carte de chaleur
    fig = px.imshow(
        corr_matrix,
        text_auto=True,
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        title=f"Matrice de corrélation ({method})"
    )
    
    fig.update_layout(
        height=600,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
    )
    
    fig.update_traces(
        texttemplate='%{text:.2f}',
        textfont=dict(color="black")
    )
    
    return fig

# Interface principale
if st.sidebar.button("📘 Guide d'utilisation", key="guide_button"):
    st.session_state.show_guide = not st.session_state.show_guide

# Sidebar pour les contrôles
with st.sidebar:
    st.markdown('<div class="sidebar-header">Configuration</div>', unsafe_allow_html=True)
    
    # Menu de type de données
    st.markdown('<div class="sidebar-header">Type de données</div>', unsafe_allow_html=True)
    
    data_type_options = {
        "drill_hole": "Données de forage",
        "composites": "Composites 3D",
        "statistics": "Données statistiques"
    }
    
    selected_data_type = st.radio("Type d'analyse", list(data_type_options.values()), 
                                index=list(data_type_options.values()).index(data_type_options[st.session_state.data_type]) if st.session_state.data_type else 0)
    
    # Mettre à jour la variable de session
    st.session_state.data_type = next(k for k, v in data_type_options.items() if v == selected_data_type)
    
    if st.session_state.data_type == "drill_hole":
        # Options spécifiques pour les données de forage
        drill_data_options = {
            "collar": "Collar (Position des trous)",
            "survey": "Survey (Déviation)",
            "lithology": "Lithologie",
            "assay": "Analyses géochimiques",
            "density": "Densité"
        }
        
        drill_data_type = st.radio("Table de données", list(drill_data_options.values()))
        current_drill_type = next(k for k, v in drill_data_options.items() if v == drill_data_type)
        
        # Chargement du fichier approprié
        uploaded_file = st.file_uploader(f"Charger un fichier {drill_data_type}", type=["csv", "xlsx"])
        
        if uploaded_file:
            # Charger les données
            df = load_data(uploaded_file, current_drill_type)
            
            if df is not None:
                # Stocker dans la session
                key_name = f"drill_{current_drill_type}"
                st.session_state.datasets[key_name] = df
                st.session_state.active_dataset = key_name
                
                st.success(f"Données {drill_data_type} chargées avec succès: {df.shape[0]} lignes, {df.shape[1]} colonnes")
    
    elif st.session_state.data_type == "composites":
        # Chargement des données de composites
        uploaded_file = st.file_uploader("Charger un fichier de composites 3D", type=["csv", "xlsx"])
        
        if uploaded_file:
            # Charger les données
            df = load_data(uploaded_file, "composites")
            
            if df is not None:
                # Stocker dans la session
                st.session_state.datasets["composites"] = df
                st.session_state.active_dataset = "composites"
                
                st.success(f"Données de composites chargées avec succès: {df.shape[0]} lignes, {df.shape[1]} colonnes")
    
    elif st.session_state.data_type == "statistics":
        # Chargement des données statistiques générales
        uploaded_file = st.file_uploader("Charger un fichier de données", type=["csv", "xlsx"])
        
        if uploaded_file:
            # Charger les données
            df = load_data(uploaded_file, "statistics")
            
            if df is not None:
                # Stocker dans la session
                st.session_state.datasets["statistics"] = df
                st.session_state.active_dataset = "statistics"
                
                st.success(f"Données statistiques chargées avec succès: {df.shape[0]} lignes, {df.shape[1]} colonnes")
    
    # Afficher les jeux de données disponibles
    if st.session_state.datasets:
        st.markdown('<div class="sidebar-header">Datasets disponibles</div>', unsafe_allow_html=True)
        
        dataset_options = []
        for key, df in st.session_state.datasets.items():
            if key.startswith("drill_"):
                display_name = key.replace("drill_", "").capitalize() + " (forage)"
            else:
                display_name = key.capitalize()
            dataset_options.append((key, display_name, df.shape))
        
        for key, name, shape in dataset_options:
            if st.button(f"{name} - {shape[0]} lignes", key=f"select_{key}"):
                st.session_state.active_dataset = key
                st.rerun()  # Correction: remplacer experimental_rerun par rerun

# Afficher le guide si demandé
if st.session_state.show_guide:
    show_user_guide()
else:
    # Vérifier si des données sont chargées
    if not st.session_state.datasets or not st.session_state.active_dataset:
        # Afficher un écran d'accueil
        st.markdown('<div class="section-header">Bienvenue dans GeoExplore Pro</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            <div class="card">
            <h3>Plateforme d'analyse avancée pour géologues d'exploration</h3>
            <p>Cette application permet aux géologues de charger, visualiser et analyser rapidement leurs données d'exploration.
            Chargez vos données pour commencer à explorer vos résultats d'exploration.</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Sélection du type de données avec images et descriptions
            st.markdown('<div class="section-header">Choisissez votre type de données</div>', unsafe_allow_html=True)
            
            col_a, col_b, col_c = st.columns(3)
            
            with col_a:
                st.markdown("""
                <div class="data-type-btn">
                <div class="data-type-btn-icon">🕳️</div>
                <h3>Données de forage</h3>
                <p>Collar, Survey, Lithologie, Analyses</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col_b:
                st.markdown("""
                <div class="data-type-btn">
                <div class="data-type-btn-icon">🧊</div>
                <h3>Composites 3D</h3>
                <p>Échantillons avec coordonnées X,Y,Z</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col_c:
                st.markdown("""
                <div class="data-type-btn">
                <div class="data-type-btn-icon">📊</div>
                <h3>Données statistiques</h3>
                <p>Tout type de données pour analyse</p>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.write("**Commencer à utiliser l'application**")
            st.write("1. Sélectionnez le type de données dans le panneau latéral")
            st.write("2. Chargez votre fichier de données")
            st.write("3. Explorez vos données à travers les différentes analyses")
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.write("**Fonctionnalités statistiques avancées**")
            st.write("- Percentiles personnalisables")
            st.write("- Histogrammes avec contrôle précis")
            st.write("- Corrélations entre variables")
            st.write("- Visualisations interactives")
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.write("**Développé par:**")
            st.write("Didier Ouedraogo, P.Geo")
            st.write("© 2025")
            st.markdown('</div>', unsafe_allow_html=True)
    
    else:
        # Récupérer le jeu de données actif
        active_df = st.session_state.datasets[st.session_state.active_dataset]
        
        # Déterminer le type de données
        is_drill_data = st.session_state.active_dataset.startswith("drill_")
        drill_type = st.session_state.active_dataset.replace("drill_", "") if is_drill_data else None
        
        # Onglets pour les différentes analyses
        selected_tab = st.radio("", ["Données", "Statistiques", "Visualisation"], horizontal=True)
        
        if selected_tab == "Données":
            st.markdown('<div class="section-header">Aperçu des données</div>', unsafe_allow_html=True)
            
            # Afficher un aperçu des données
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.write(f"**Dataset actif:** {st.session_state.active_dataset}")
                st.write(f"**Dimensions:** {active_df.shape[0]} lignes × {active_df.shape[1]} colonnes")
                
                # Options d'affichage
                show_all_cols = st.checkbox("Afficher toutes les colonnes", value=False)
                if show_all_cols:
                    st.dataframe(active_df)
                else:
                    # Limiter à un nombre raisonnable de colonnes
                    max_cols = min(10, active_df.shape[1])
                    st.dataframe(active_df.iloc[:, :max_cols])
            
            with col2:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.write("**Informations sur les colonnes**")
                
                # Afficher les types de colonnes détectés
                for col in active_df.columns[:15]:  # Limiter pour la lisibilité
                    col_type = detect_column_type(active_df, col)
                    st.write(f"- **{col}**: {col_type}")
                
                if len(active_df.columns) > 15:
                    st.write(f"... et {len(active_df.columns) - 15} autres colonnes")
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Options d'export
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.write("**Exporter les données**")
                
                export_format = st.radio("Format", ["CSV", "Excel"])
                
                if export_format == "CSV":
                    csv = active_df.to_csv(index=False)
                    b64 = base64.b64encode(csv.encode()).decode()
                    export_filename = f"donnees_geoexplore_{datetime.now().strftime('%Y%m%d')}.csv"
                    href = f'<a href="data:file/csv;base64,{b64}" download="{export_filename}" class="guide-btn">Télécharger CSV</a>'
                    st.markdown(href, unsafe_allow_html=True)
                else:
                    # Pour Excel, on doit utiliser un buffer
                    buffer = io.BytesIO()
                    active_df.to_excel(buffer, index=False)
                    buffer.seek(0)
                    b64 = base64.b64encode(buffer.read()).decode()
                    export_filename = f"donnees_geoexplore_{datetime.now().strftime('%Y%m%d')}.xlsx"
                    href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{export_filename}" class="guide-btn">Télécharger Excel</a>'
                    st.markdown(href, unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
        
        elif selected_tab == "Statistiques":
            st.markdown('<div class="section-header">Statistiques avancées</div>', unsafe_allow_html=True)
            
            # Colonnes numériques disponibles
            numeric_cols = get_numeric_columns(active_df)
            
            if not numeric_cols:
                st.warning("Aucune variable numérique détectée dans ce jeu de données.")
            else:
                # Sélection de variables
                col1, col2 = st.columns([1, 3])
                
                with col1:
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.write("**Sélection des variables**")
                    
                    # Option pour filtrer les données
                    with st.expander("Filtrer les données", expanded=False):
                        # Filtres numériques
                        filter_numeric = st.checkbox("Appliquer un filtre numérique", value=False)
                        
                        if filter_numeric and numeric_cols:
                            filter_var = st.selectbox("Variable à filtrer", numeric_cols)
                            min_val = float(active_df[filter_var].min())
                            max_val = float(active_df[filter_var].max())
                            
                            filter_range = st.slider(
                                f"Plage de valeurs pour {filter_var}",
                                min_value=min_val,
                                max_value=max_val,
                                value=(min_val, max_val)
                            )
                            
                            filtered_df = active_df[
                                (active_df[filter_var] >= filter_range[0]) & 
                                (active_df[filter_var] <= filter_range[1])
                            ]
                        else:
                            filtered_df = active_df
                        
                        # Filtres catégoriels
                        cat_cols = get_categorical_columns(active_df)
                        filter_categorical = st.checkbox("Appliquer un filtre catégoriel", value=False)
                        
                        if filter_categorical and cat_cols:
                            cat_filter_var = st.selectbox("Variable catégorielle", cat_cols)
                            categories = sorted(active_df[cat_filter_var].dropna().unique())
                            selected_cats = st.multiselect("Catégories à inclure", categories, default=categories)
                            
                            if selected_cats:
                                filtered_df = filtered_df[filtered_df[cat_filter_var].isin(selected_cats)]
                        
                        # Afficher le nombre de lignes après filtrage
                        st.write(f"Données filtrées: {filtered_df.shape[0]} lignes")
                    
                    # Sélection des variables pour l'analyse
                    selected_vars = st.multiselect(
                        "Variables à analyser",
                        numeric_cols,
                        default=numeric_cols[:min(3, len(numeric_cols))]
                    )
                    
                    # Type d'analyse statistique
                    analysis_type = st.radio(
                        "Type d'analyse",
                        ["Statistiques descriptives", "Histogramme", "Boîte à moustaches/Violon"]
                    )
                    
                    # Options pour les percentiles personnalisés
                    if analysis_type == "Statistiques descriptives":
                        st.write("**Percentiles personnalisés**")
                        
                        use_custom_percentiles = st.checkbox("Personnaliser les percentiles", value=False)
                        
                        if use_custom_percentiles:
                            # Interface pour ajouter des percentiles
                            percentile_values = st.multiselect(
                                "Sélectionner des percentiles (%)",
                                list(range(1, 100)),
                                default=[5, 10, 25, 50, 75, 90, 95]
                            )
                            percentile_values = sorted([0] + percentile_values + [100])
                        else:
                            percentile_values = [0, 5, 10, 25, 50, 75, 90, 95, 100]
                    
                    # Options pour l'histogramme
                    elif analysis_type == "Histogramme":
                        st.write("**Options d'histogramme**")
                        
                        hist_bins = st.slider("Nombre d'intervalles", 5, 100, 20)
                        hist_density = st.checkbox("Densité", value=False)
                        hist_log_x = st.checkbox("Échelle log (X)", value=False)
                        hist_log_y = st.checkbox("Échelle log (Y)", value=False)
                        hist_kde = st.checkbox("Ajouter courbe KDE", value=True)
                    
                    # Options pour boîte à moustaches/violon
                    elif analysis_type in ["Boîte à moustaches/Violon"]:
                        st.write("**Options de visualisation**")
                        
                        plot_type = st.radio("Type de graphique", ["Boîte à moustaches", "Violon"])
                        
                        # Option pour grouper par une variable catégorielle
                        group_by_option = st.checkbox("Grouper par catégorie", value=False)
                        
                        if group_by_option:
                            cat_cols = get_categorical_columns(filtered_df)
                            if cat_cols:
                                group_by_var = st.selectbox("Variable de groupement", cat_cols)
                            else:
                                group_by_var = None
                                st.info("Aucune variable catégorielle disponible pour le groupement.")
                        else:
                            group_by_var = None
                        
                        # Orientation
                        orientation = st.radio("Orientation", ["Vertical", "Horizontal"])
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col2:
                    if not selected_vars:
                        st.info("Veuillez sélectionner au moins une variable à analyser.")
                    else:
                        if analysis_type == "Statistiques descriptives":
                            st.markdown('<div class="section-header">Statistiques descriptives détaillées</div>', unsafe_allow_html=True)
                            
                            for var in selected_vars:
                                st.subheader(f"Variable: {var}")
                                
                                # Calcul des statistiques avec percentiles personnalisés
                                stats_dict = calculate_custom_statistics(filtered_df, var, percentiles=percentile_values)
                                
                                # Afficher les statistiques dans un tableau formaté
                                col_a, col_b = st.columns(2)
                                
                                with col_a:
                                    st.write("**Statistiques de base**")
                                    
                                    basic_stats = {
                                        "Nombre": stats_dict['count'],
                                        "Moyenne": stats_dict['mean'],
                                        "Écart-type": stats_dict['std'],
                                        "Minimum": stats_dict['min'],
                                        "Maximum": stats_dict['max'],
                                        "Médiane": stats_dict['50%'] if '50%' in stats_dict else stats_dict['median'] if 'median' in stats_dict else None
                                    }
                                    
                                    # Statistiques additionnelles
                                    basic_stats["Coef. Variation"] = stats_dict['cv']
                                    basic_stats["Asymétrie"] = stats_dict['skewness']
                                    basic_stats["Aplatissement"] = stats_dict['kurtosis']
                                    basic_stats["IQR"] = stats_dict['iqr']
                                    
                                    # Afficher
                                    basic_df = pd.DataFrame.from_dict(basic_stats, orient='index', columns=["Valeur"])
                                    basic_df = basic_df.style.format("{:.3f}")
                                    st.dataframe(basic_df)
                                
                                with col_b:
                                    st.write("**Percentiles**")
                                    
                                    percentile_stats = {}
                                    for p in percentile_values:
                                        if p == 0:
                                            # Le percentile 0 est le minimum
                                            percentile_stats[f"{p}%"] = stats_dict['min']
                                        elif p == 100:
                                            # Le percentile 100 est le maximum
                                            percentile_stats[f"{p}%"] = stats_dict['max']
                                        else:
                                            # Rechercher le percentile dans stats_dict
                                            key = f"{p}%"
                                            if key in stats_dict:
                                                percentile_stats[key] = stats_dict[key]
                                            else:
                                                # Calculer si nécessaire
                                                percentile_stats[key] = np.nanpercentile(filtered_df[var], p)
                                    
                                    # Afficher
                                    percentile_df = pd.DataFrame.from_dict(percentile_stats, orient='index', columns=["Valeur"])
                                    percentile_df = percentile_df.style.format("{:.3f}")
                                    st.dataframe(percentile_df)
                                
                                # Créer un histogramme simple
                                hist_data = filtered_df[var].dropna()
                                
                                if len(hist_data) > 1:
                                    fig = px.histogram(
                                        filtered_df, x=var, nbins=20,
                                        title=f"Distribution de {var}"
                                    )
                                    
                                    fig.add_vline(x=hist_data.mean(), line_dash="dash", line_color="red", 
                                                annotation_text="Moyenne", annotation_position="top right")
                                    fig.add_vline(x=hist_data.median(), line_dash="dash", line_color="green", 
                                                annotation_text="Médiane", annotation_position="top left")
                                    
                                    st.plotly_chart(fig, use_container_width=True)
                                else:
                                    st.warning("Pas assez de données pour créer un histogramme.")
                                
                                st.markdown("---")
                        
                        elif analysis_type == "Histogramme":
                            st.markdown('<div class="section-header">Analyse par histogramme</div>', unsafe_allow_html=True)
                            
                            for var in selected_vars:
                                st.subheader(f"Variable: {var}")
                                
                                # Créer l'histogramme avec les options sélectionnées
                                hist_fig = create_histogram(
                                    filtered_df, var, bins=hist_bins, 
                                    density=hist_density, log_x=hist_log_x, 
                                    log_y=hist_log_y, kde=hist_kde
                                )
                                
                                st.plotly_chart(hist_fig, use_container_width=True)
                                
                                # Statistiques de base pour référence
                                with st.expander("Statistiques clés"):
                                    stats = filtered_df[var].describe()
                                    
                                    col_stats1, col_stats2 = st.columns(2)
                                    with col_stats1:
                                        st.write(f"Nombre: {stats['count']:.0f}")
                                        st.write(f"Moyenne: {stats['mean']:.3f}")
                                        st.write(f"Écart-type: {stats['std']:.3f}")
                                    
                                    with col_stats2:
                                        st.write(f"Minimum: {stats['min']:.3f}")
                                        st.write(f"Maximum: {stats['max']:.3f}")
                                        st.write(f"Médiane: {stats['50%']:.3f}")
                                
                                st.markdown("---")
                        
                        elif analysis_type == "Boîte à moustaches/Violon":
                            st.markdown('<div class="section-header">Analyse de distribution</div>', unsafe_allow_html=True)
                            
                            # Déterminer le type de plot
                            viz_type = "box" if plot_type == "Boîte à moustaches" else "violin"
                            orient = "vertical" if orientation == "Vertical" else "horizontal"
                            
                            for var in selected_vars:
                                st.subheader(f"Variable: {var}")
                                
                                # Créer le graphique
                                box_fig = create_box_violin_plot(
                                    filtered_df, var, plot_type=viz_type,
                                    group_by=group_by_var, orientation=orient
                                )
                                
                                st.plotly_chart(box_fig, use_container_width=True)
                                
                                # Si groupé, afficher des statistiques par groupe
                                if group_by_var:
                                    with st.expander("Statistiques par groupe"):
                                        group_stats = filtered_df.groupby(group_by_var)[var].agg([
                                            'count', 'mean', 'std', 'min', 
                                            lambda x: x.quantile(0.25),
                                            'median', 
                                            lambda x: x.quantile(0.75),
                                            'max'
                                        ]).reset_index()
                                        
                                        group_stats.columns = [group_by_var, 'N', 'Moyenne', 'Écart-type', 
                                                           'Min', 'Q1', 'Médiane', 'Q3', 'Max']
                                        
                                        # Formater les valeurs numériques
                                        for col in group_stats.columns[1:]:
                                            if group_stats[col].dtype in ['float64', 'float32']:
                                                group_stats[col] = group_stats[col].round(3)
                                        
                                        st.dataframe(group_stats)
                                
                                st.markdown("---")
        
        elif selected_tab == "Visualisation":
            st.markdown('<div class="section-header">Visualisations avancées</div>', unsafe_allow_html=True)
            
            # Colonnes numériques disponibles
            numeric_cols = get_numeric_columns(active_df)
            
            if len(numeric_cols) < 2:
                st.warning("Au moins deux variables numériques sont nécessaires pour les visualisations avancées.")
            else:
                # Sélection du type de visualisation
                viz_type = st.radio(
                    "Type de visualisation",
                    ["Nuage de points", "Matrice de corrélation", "Carte interactive"],
                    horizontal=True
                )
                
                if viz_type == "Nuage de points":
                    col1, col2 = st.columns([1, 3])
                    
                    with col1:
                        st.markdown('<div class="card">', unsafe_allow_html=True)
                        st.write("**Options du graphique**")
                        
                        # Sélection des variables
                        x_var = st.selectbox("Variable X", numeric_cols)
                        y_var = st.selectbox("Variable Y", [col for col in numeric_cols if col != x_var])
                        
                        # Filtrage des données
                        with st.expander("Filtrer les données"):
                            # Options de filtre
                            apply_filter = st.checkbox("Appliquer un filtre", value=False)
                            
                            if apply_filter:
                                filter_var = st.selectbox("Variable à filtrer", numeric_cols)
                                min_val = float(active_df[filter_var].min())
                                max_val = float(active_df[filter_var].max())
                                
                                filter_range = st.slider(
                                    f"Plage pour {filter_var}",
                                    min_value=min_val,
                                    max_value=max_val,
                                    value=(min_val, max_val)
                                )
                                
                                filtered_df = active_df[
                                    (active_df[filter_var] >= filter_range[0]) & 
                                    (active_df[filter_var] <= filter_range[1])
                                ]
                            else:
                                filtered_df = active_df
                        
                        # Options avancées
                        trendline = st.checkbox("Ajouter une ligne de tendance", value=True)
                        log_x = st.checkbox("Échelle logarithmique (X)", value=False)
                        log_y = st.checkbox("Échelle logarithmique (Y)", value=False)
                        
                        # Option de coloration
                        color_options = ["Aucune"] + numeric_cols + get_categorical_columns(filtered_df)
                        color_var = st.selectbox("Colorer par", color_options)
                        
                        if color_var == "Aucune":
                            color_var = None
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    with col2:
                        # Créer le scatter plot
                        scatter_fig = create_scatter_plot(
                            filtered_df, x_var, y_var,
                            color_col=color_var, trendline=trendline,
                            log_x=log_x, log_y=log_y
                        )
                        
                        st.plotly_chart(scatter_fig, use_container_width=True)
                        
                        # Statistiques de corrélation
                        corr = filtered_df[[x_var, y_var]].corr().iloc[0, 1]
                        
                        st.markdown('<div class="card">', unsafe_allow_html=True)
                        st.write(f"**Coefficient de corrélation: {corr:.3f}**")
                        
                        if abs(corr) < 0.3:
                            st.write("Corrélation faible")
                        elif abs(corr) < 0.7:
                            st.write("Corrélation modérée")
                        else:
                            st.write("Corrélation forte")
                        st.markdown('</div>', unsafe_allow_html=True)
                
                elif viz_type == "Matrice de corrélation":
                    col1, col2 = st.columns([1, 3])
                    
                    with col1:
                        st.markdown('<div class="card">', unsafe_allow_html=True)
                        st.write("**Options de corrélation**")
                        
                        # Sélection des variables
                        selected_vars = st.multiselect(
                            "Variables à inclure",
                            numeric_cols,
                            default=numeric_cols[:min(8, len(numeric_cols))]
                        )
                        
                        # Méthode de corrélation
                        corr_method = st.radio(
                            "Méthode de corrélation",
                            ["pearson", "spearman"],
                            format_func=lambda x: {
                                "pearson": "Pearson (linéaire)",
                                "spearman": "Spearman (rang)"
                            }.get(x, x)
                        )
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    with col2:
                        if not selected_vars or len(selected_vars) < 2:
                            st.info("Veuillez sélectionner au moins deux variables pour la matrice de corrélation.")
                        else:
                            # Créer la matrice de corrélation
                            corr_fig = create_correlation_heatmap(active_df, selected_vars, method=corr_method)
                            st.plotly_chart(corr_fig, use_container_width=True)
                
                elif viz_type == "Carte interactive":
                    # Détection des colonnes de coordonnées
                    x_options = [col for col in active_df.columns if detect_column_type(active_df, col) == 'X']
                    y_options = [col for col in active_df.columns if detect_column_type(active_df, col) == 'Y']
                    
                    if not x_options or not y_options:
                        st.warning("Colonnes de coordonnées (X/Y) non détectées. La carte ne peut pas être affichée.")
                    else:
                        col1, col2 = st.columns([1, 3])
                        
                        with col1:
                            st.markdown('<div class="card">', unsafe_allow_html=True)
                            st.write("**Options de la carte**")
                            
                            # Sélection des colonnes de coordonnées
                            x_col = st.selectbox("Colonne X", x_options)
                            y_col = st.selectbox("Colonne Y", y_options)
                            
                            # Coloration des points
                            color_options = ["Aucune"] + numeric_cols + get_categorical_columns(active_df)
                            color_var = st.selectbox("Colorer par", color_options)
                            
                            if color_var == "Aucune":
                                color_var = None
                            
                            # Taille des points
                            point_size = st.slider("Taille des points", 5, 20, 10)
                            
                            # Afficher les étiquettes
                            id_cols = [col for col in active_df.columns if detect_column_type(active_df, col) == 'ID']
                            
                            if id_cols:
                                label_col = st.selectbox("Colonne d'étiquette", ["Aucune"] + id_cols)
                                
                                if label_col == "Aucune":
                                    label_col = None
                                
                                show_labels = st.checkbox("Afficher les étiquettes", value=False)
                            else:
                                label_col = None
                                show_labels = False
                            
                            st.markdown('</div>', unsafe_allow_html=True)
                        
                        with col2:
                            # Créer la carte
                            if color_var:
                                fig = px.scatter(
                                    active_df, x=x_col, y=y_col, color=color_var, 
                                    hover_name=label_col,
                                    labels={x_col: "X / Longitude", y_col: "Y / Latitude"},
                                    title="Carte des données"
                                )
                            else:
                                fig = px.scatter(
                                    active_df, x=x_col, y=y_col,
                                    hover_name=label_col,
                                    labels={x_col: "X / Longitude", y_col: "Y / Latitude"},
                                    title="Carte des données"
                                )
                            
                            # Modifier la taille des points
                            fig.update_traces(marker=dict(size=point_size))
                            
                            # Ajouter les étiquettes si demandé
                            if show_labels and label_col:
                                fig.update_traces(
                                    textposition='top center',
                                    textfont=dict(color='black', size=10),
                                    text=active_df[label_col]
                                )
                            
                            # Mise en page
                            fig.update_layout(
                                height=600,
                                margin=dict(l=20, r=20, t=40, b=20),
                                yaxis=dict(
                                    scaleanchor="x",
                                    scaleratio=1
                                )
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)

# Pied de page
st.markdown("""
<div class="footer">
<p>GeoExplore Pro v1.0 | Développé par Didier Ouedraogo, P.Geo | © 2025</p>
</div>
""", unsafe_allow_html=True)