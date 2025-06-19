import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Configuration de la page
st.set_page_config(
    page_title="🇫🇷 Dashboard Électoral France",
    page_icon="🗳️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé pour un style élégant
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #1e3c72, #2a5298);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .kpi-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    
    .kpi-value {
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    
    .kpi-label {
        font-size: 1rem;
        opacity: 0.9;
    }
    
    .section-header {
        font-size: 1.8rem;
        font-weight: bold;
        color: #2c3e50;
        margin: 2rem 0 1rem 0;
        border-left: 5px solid #3498db;
        padding-left: 1rem;
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 4px 16px rgba(0,0,0,0.1);
        border: 1px solid #e1e8ed;
    }
</style>
""", unsafe_allow_html=True)

# Mapping des départements vers les régions
def get_region_mapping():
    """
    Retourne le mapping des départements vers les régions françaises
    """
    return {
        # Auvergne-Rhône-Alpes
        '01': 'Auvergne-Rhône-Alpes', '03': 'Auvergne-Rhône-Alpes', '07': 'Auvergne-Rhône-Alpes', 
        '15': 'Auvergne-Rhône-Alpes', '26': 'Auvergne-Rhône-Alpes', '38': 'Auvergne-Rhône-Alpes', 
        '42': 'Auvergne-Rhône-Alpes', '43': 'Auvergne-Rhône-Alpes', '63': 'Auvergne-Rhône-Alpes', 
        '69': 'Auvergne-Rhône-Alpes', '73': 'Auvergne-Rhône-Alpes', '74': 'Auvergne-Rhône-Alpes',
        
        # Bourgogne-Franche-Comté
        '21': 'Bourgogne-Franche-Comté', '25': 'Bourgogne-Franche-Comté', '39': 'Bourgogne-Franche-Comté', 
        '58': 'Bourgogne-Franche-Comté', '70': 'Bourgogne-Franche-Comté', '71': 'Bourgogne-Franche-Comté', 
        '89': 'Bourgogne-Franche-Comté', '90': 'Bourgogne-Franche-Comté',
        
        # Bretagne
        '22': 'Bretagne', '29': 'Bretagne', '35': 'Bretagne', '56': 'Bretagne',
        
        # Centre-Val de Loire
        '18': 'Centre-Val de Loire', '28': 'Centre-Val de Loire', '36': 'Centre-Val de Loire', 
        '37': 'Centre-Val de Loire', '41': 'Centre-Val de Loire', '45': 'Centre-Val de Loire',
        
        # Corse
        '2A': 'Corse', '2B': 'Corse',
        
        # Grand Est
        '08': 'Grand Est', '10': 'Grand Est', '51': 'Grand Est', '52': 'Grand Est', 
        '54': 'Grand Est', '55': 'Grand Est', '57': 'Grand Est', '67': 'Grand Est', 
        '68': 'Grand Est', '88': 'Grand Est',
        
        # Hauts-de-France
        '02': 'Hauts-de-France', '59': 'Hauts-de-France', '60': 'Hauts-de-France', 
        '62': 'Hauts-de-France', '80': 'Hauts-de-France',
        
        # Île-de-France
        '75': 'Île-de-France', '77': 'Île-de-France', '78': 'Île-de-France', 
        '91': 'Île-de-France', '92': 'Île-de-France', '93': 'Île-de-France', 
        '94': 'Île-de-France', '95': 'Île-de-France',
        
        # Normandie
        '14': 'Normandie', '27': 'Normandie', '50': 'Normandie', '61': 'Normandie', '76': 'Normandie',
        
        # Nouvelle-Aquitaine
        '16': 'Nouvelle-Aquitaine', '17': 'Nouvelle-Aquitaine', '19': 'Nouvelle-Aquitaine', 
        '23': 'Nouvelle-Aquitaine', '24': 'Nouvelle-Aquitaine', '33': 'Nouvelle-Aquitaine', 
        '40': 'Nouvelle-Aquitaine', '47': 'Nouvelle-Aquitaine', '64': 'Nouvelle-Aquitaine', 
        '79': 'Nouvelle-Aquitaine', '86': 'Nouvelle-Aquitaine', '87': 'Nouvelle-Aquitaine',
        
        # Occitanie
        '09': 'Occitanie', '11': 'Occitanie', '12': 'Occitanie', '30': 'Occitanie', 
        '31': 'Occitanie', '32': 'Occitanie', '34': 'Occitanie', '46': 'Occitanie', 
        '48': 'Occitanie', '65': 'Occitanie', '66': 'Occitanie', '81': 'Occitanie', '82': 'Occitanie',
        
        # Pays de la Loire
        '44': 'Pays de la Loire', '49': 'Pays de la Loire', '53': 'Pays de la Loire', 
        '72': 'Pays de la Loire', '85': 'Pays de la Loire',
        
        # Provence-Alpes-Côte d\'Azur
        '04': 'Provence-Alpes-Côte d\'Azur', '05': 'Provence-Alpes-Côte d\'Azur', 
        '06': 'Provence-Alpes-Côte d\'Azur', '13': 'Provence-Alpes-Côte d\'Azur', 
        '83': 'Provence-Alpes-Côte d\'Azur', '84': 'Provence-Alpes-Côte d\'Azur',
        
        # Départements d'outre-mer
        '971': 'Guadeloupe', '972': 'Martinique', '973': 'Guyane', '974': 'La Réunion', '976': 'Mayotte'
    }

@st.cache_data
def load_data():
    """
    Charge et prépare les données
    """
    # Remplacez par votre chemin de fichier
    file_path = "C:/Users/Massi/Desktop/MSPR BLOC 03/les_jointeurs/datasets/jointeur_elections_population/Dataset_Finale_MSPR_2025.csv"
    
    try:
        df = pd.read_csv(file_path, sep=';')
        
        # Mapping des orientations politiques
        orientation_map = {
            0: 'Extrême Gauche',
            1: 'Gauche', 
            2: 'Centre',
            3: 'Droite',
            4: 'Extrême Droite'
        }
        
        df['orientation_politique_label'] = df['orientation_politique'].map(orientation_map)
        
        # Calcul des métriques
        df['taux_participation'] = ((df['nb_inscrits'] - df['nb_abstentions']) / df['nb_inscrits'] * 100).round(2)
        df['taux_abstention'] = (df['nb_abstentions'] / df['nb_inscrits'] * 100).round(2)
        df['taux_chomage_pct'] = df['taux_chomage_pct'] * 100  # Conversion en pourcentage
        
        # Extraction du département depuis commune_id
        df['departement'] = df['commune_id'].astype(str).str[:2]
        
        # Ajout de la région
        region_mapping = get_region_mapping()
        df['region'] = df['departement'].map(region_mapping)
        
        # Gestion des départements non mappés
        df['region'] = df['region'].fillna('Autre')
        
        return df
    
    except FileNotFoundError:
        # Données de démonstration si le fichier n'est pas trouvé
        np.random.seed(42)
        n_communes = 1000
        
        data = {
            'commune_id': [f"{i:05d}" for i in range(1, n_communes+1)],
            'annee': np.random.choice([2017, 2022], n_communes),
            'nb_inscrits': np.random.randint(100, 10000, n_communes),
            'nb_abstentions': np.random.randint(10, 3000, n_communes),
            'orientation_politique': np.random.choice([0, 1, 2, 3, 4], n_communes, p=[0.08, 0.15, 0.35, 0.25, 0.17]),
            'Population_Totale': np.random.randint(150, 15000, n_communes),
            'pct_population_jeune': np.random.uniform(15, 35, n_communes),
            'pct_population_senior': np.random.uniform(12, 30, n_communes),
            'pct_population_sans_activite': np.random.uniform(20, 40, n_communes),
            'pct_population_etrangere': np.random.uniform(0, 15, n_communes),
            'nb_population_active': np.random.randint(50, 8000, n_communes),
            'taux_chomage_pct': np.random.uniform(0.02, 0.20, n_communes),
            'nb_crimes': np.random.randint(0, 1000, n_communes)
        }
        
        df = pd.DataFrame(data)
        
        # Mapping des orientations politiques
        orientation_map = {
            0: 'Extrême Gauche',
            1: 'Gauche', 
            2: 'Centre',
            3: 'Droite',
            4: 'Extrême Droite'
        }
        
        df['orientation_politique_label'] = df['orientation_politique'].map(orientation_map)
        
        # Calcul des métriques
        df['taux_participation'] = ((df['nb_inscrits'] - df['nb_abstentions']) / df['nb_inscrits'] * 100).round(2)
        df['taux_abstention'] = (df['nb_abstentions'] / df['nb_inscrits'] * 100).round(2)
        df['taux_chomage_pct'] = df['taux_chomage_pct'] * 100
        
        # Extraction du département
        df['departement'] = df['commune_id'].astype(str).str[:2]
        
        # Ajout de la région
        region_mapping = get_region_mapping()
        df['region'] = df['departement'].map(region_mapping)
        df['region'] = df['region'].fillna('Autre')
        
        return df

def create_kpi_card(value, label, color_start="#667eea", color_end="#764ba2"):
    """
    Crée une carte KPI stylisée
    """
    return f"""
    <div style="background: linear-gradient(135deg, {color_start} 0%, {color_end} 100%);
                padding: 1.5rem; border-radius: 15px; color: white; text-align: center;
                box-shadow: 0 8px 32px rgba(0,0,0,0.1); margin-bottom: 1rem;">
        <div style="font-size: 2.5rem; font-weight: bold; margin-bottom: 0.5rem;">{value}</div>
        <div style="font-size: 1rem; opacity: 0.9;">{label}</div>
    </div>
    """

def main():
    # Titre principal
    st.markdown('<h1 class="main-header">🇫🇷 Dashboard Électoral et Démographique Français</h1>', unsafe_allow_html=True)
    
    # Chargement des données
    with st.spinner('🔄 Chargement des données...'):
        df = load_data()
    
    # Sidebar pour les filtres
    st.sidebar.header("🎛️ Filtres")
    
    # Filtre par région
    regions_disponibles = ['Toute la France'] + sorted([r for r in df['region'].unique() if r != 'Autre'])
    if 'Autre' in df['region'].unique():
        regions_disponibles.append('Autre')
    
    region_selectionnee = st.sidebar.selectbox("🗺️ Région", regions_disponibles)
    
    # Filtre par année
    annees_disponibles = sorted(df['annee'].unique())
    annee_selectionnee = st.sidebar.selectbox("📅 Année", annees_disponibles, index=len(annees_disponibles)-1)
    
    # Filtre par orientation politique
    orientations_disponibles = ['Toutes'] + list(df['orientation_politique_label'].unique())
    orientation_selectionnee = st.sidebar.selectbox("🏛️ Orientation Politique", orientations_disponibles)
    
    # Filtrage des données
    df_filtered = df[df['annee'] == annee_selectionnee].copy()
    
    if region_selectionnee != 'Toute la France':
        df_filtered = df_filtered[df_filtered['region'] == region_selectionnee]
    
    if orientation_selectionnee != 'Toutes':
        df_filtered = df_filtered[df_filtered['orientation_politique_label'] == orientation_selectionnee]
    
    # Affichage de la région sélectionnée
    if region_selectionnee != 'Toute la France':
        st.info(f"📍 Analyse focalisée sur la région : **{region_selectionnee}**")
    
    # === SECTION KPI PRINCIPAUX ===
    st.markdown('<div class="section-header">📊 KPIs Principaux</div>', unsafe_allow_html=True)
    
    # Calcul des KPIs
    nb_communes = len(df_filtered)
    population_totale = df_filtered['Population_Totale'].sum()
    taux_participation_moyen = df_filtered['taux_participation'].mean()
    taux_chomage_moyen = df_filtered['taux_chomage_pct'].mean()
    total_crimes = df_filtered['nb_crimes'].sum()
    
    # Affichage des KPIs en colonnes
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown(create_kpi_card(f"{nb_communes:,}", "Communes", "#e74c3c", "#c0392b"), unsafe_allow_html=True)
    
    with col2:
        st.markdown(create_kpi_card(f"{population_totale:,.0f}", "Population Totale", "#3498db", "#2980b9"), unsafe_allow_html=True)
    
    with col3:
        st.markdown(create_kpi_card(f"{taux_participation_moyen:.1f}%", "Taux Participation Moyen", "#2ecc71", "#27ae60"), unsafe_allow_html=True)
    
    with col4:
        st.markdown(create_kpi_card(f"{taux_chomage_moyen:.1f}%", "Taux Chômage Moyen", "#f39c12", "#e67e22"), unsafe_allow_html=True)
    
    with col5:
        st.markdown(create_kpi_card(f"{total_crimes:,.0f}", "Total Crimes", "#9b59b6", "#8e44ad"), unsafe_allow_html=True)
    
    # === SECTION ANALYSE POLITIQUE ===
    st.markdown('<div class="section-header">🏛️ Analyse Politique</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Distribution des orientations politiques
        orientation_counts = df_filtered['orientation_politique_label'].value_counts()
        orientation_pct = (orientation_counts / orientation_counts.sum() * 100).round(2)
        
        colors_political = {
            'Extrême Gauche': '#8e44ad',
            'Gauche': '#e74c3c', 
            'Centre': '#f39c12',
            'Droite': '#3498db',
            'Extrême Droite': '#2c3e50'
        }
        
        fig_pie = px.pie(
            values=orientation_counts.values,
            names=orientation_counts.index,
            title="Distribution des Orientations Politiques",
            color=orientation_counts.index,
            color_discrete_map=colors_political,
            hole=0.4
        )
        
        fig_pie.update_traces(
            textposition='inside', 
            textinfo='percent+label',
            textfont_size=12,
            marker=dict(line=dict(color='white', width=2))
        )
        
        fig_pie.update_layout(
            font=dict(size=14),
            showlegend=True,
            legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.05),
            margin=dict(l=20, r=20, t=50, b=20),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        
        st.plotly_chart(fig_pie, use_container_width=True, key="pie_political")
    
    with col2:
        # Taux de participation par orientation
        participation_by_orientation = df_filtered.groupby('orientation_politique_label')['taux_participation'].mean().sort_values(ascending=True)
        
        fig_bar = px.bar(
            x=participation_by_orientation.values,
            y=participation_by_orientation.index,
            orientation='h',
            title="Taux de Participation Moyen par Orientation",
            color=participation_by_orientation.values,
            color_continuous_scale='viridis',
            text=[f"{val:.1f}%" for val in participation_by_orientation.values]
        )
        
        fig_bar.update_traces(
            textposition='inside',
            textfont=dict(color='white', size=12, family='Arial Black')
        )
        
        fig_bar.update_layout(
            xaxis_title="Taux de Participation (%)",
            yaxis_title="Orientation Politique",
            font=dict(size=12),
            margin=dict(l=20, r=20, t=50, b=20),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            showlegend=False
        )
        
        fig_bar.update_xaxes(showgrid=True, gridcolor='rgba(128,128,128,0.2)')
        fig_bar.update_yaxes(showgrid=False)
        
        st.plotly_chart(fig_bar, use_container_width=True, key="bar_participation")
    
    # === SECTION ANALYSE DÉMOGRAPHIQUE ===
    st.markdown('<div class="section-header">👥 Analyse Démographique</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Moyennes démographiques par orientation politique
        demo_means = df_filtered.groupby('orientation_politique_label')[
            ['pct_population_jeune', 'pct_population_senior', 'pct_population_etrangere']
        ].mean().round(1)
        
        fig_demo = px.bar(
            demo_means,
            title="Profil Démographique par Orientation Politique",
            color_discrete_sequence=['#3498db', '#e74c3c', '#f39c12'],
            barmode='group'
        )
        
        fig_demo.update_layout(
            xaxis_title="Orientation Politique",
            yaxis_title="Pourcentage Moyen (%)",
            font=dict(size=12),
            margin=dict(l=20, r=20, t=50, b=20),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            legend=dict(title="Indicateurs", orientation="v", yanchor="top", y=1, xanchor="left", x=1.05)
        )
        
        fig_demo.update_xaxes(tickangle=45, showgrid=False)
        fig_demo.update_yaxes(showgrid=True, gridcolor='rgba(128,128,128,0.2)')
        
        # Renommer les légendes pour plus de clarté
        fig_demo.data[0].name = "% Jeunes"
        fig_demo.data[1].name = "% Seniors" 
        fig_demo.data[2].name = "% Étrangers"
        
        st.plotly_chart(fig_demo, use_container_width=True, key="bar_demographic")
    
    with col2:
        # Évolution du chômage par orientation (2017 vs 2022)
        if len(df['annee'].unique()) > 1:
            # Appliquer le filtre de région pour l'évolution aussi
            df_evolution = df.copy()
            if region_selectionnee != 'Toute la France':
                df_evolution = df_evolution[df_evolution['region'] == region_selectionnee]
                
            chomage_evolution = df_evolution.groupby(['annee', 'orientation_politique_label'])['taux_chomage_pct'].mean().reset_index()
            
            fig_evolution = px.line(
                chomage_evolution,
                x='annee',
                y='taux_chomage_pct',
                color='orientation_politique_label',
                title="Évolution du Taux de Chômage (2017-2022)",
                markers=True,
                color_discrete_map=colors_political,
                line_shape='linear'
            )
            
            fig_evolution.update_layout(
                xaxis_title="Année",
                yaxis_title="Taux de Chômage Moyen (%)",
                font=dict(size=12),
                margin=dict(l=20, r=20, t=50, b=20),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                legend=dict(title="Orientation", orientation="v", yanchor="top", y=1, xanchor="left", x=1.05)
            )
            
            fig_evolution.update_traces(line=dict(width=3), marker=dict(size=8))
            fig_evolution.update_xaxes(showgrid=True, gridcolor='rgba(128,128,128,0.2)')
            fig_evolution.update_yaxes(showgrid=True, gridcolor='rgba(128,128,128,0.2)')
            
            st.plotly_chart(fig_evolution, use_container_width=True, key="line_chomage_evolution")
        else:
            # Si une seule année, graphique des crimes par orientation
            crimes_by_orientation = df_filtered.groupby('orientation_politique_label')['nb_crimes'].sum().sort_values(ascending=False)
            
            fig_crimes = px.bar(
                x=crimes_by_orientation.index,
                y=crimes_by_orientation.values,
                title="Nombre Total de Crimes par Orientation",
                color=crimes_by_orientation.values,
                color_continuous_scale='Reds',
                text=crimes_by_orientation.values
            )
            
            fig_crimes.update_traces(
                texttemplate='%{text:,.0f}',
                textposition='outside',
                textfont=dict(size=11, color='black')
            )
            
            fig_crimes.update_layout(
                xaxis_title="Orientation Politique",
                yaxis_title="Nombre Total de Crimes",
                font=dict(size=12),
                margin=dict(l=20, r=20, t=50, b=20),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                showlegend=False
            )
            
            fig_crimes.update_xaxes(tickangle=45, showgrid=False)
            fig_crimes.update_yaxes(showgrid=True, gridcolor='rgba(128,128,128,0.2)')
            
            st.plotly_chart(fig_crimes, use_container_width=True, key="bar_crimes")
    
    # === SECTION ANALYSE GÉOGRAPHIQUE ===
    st.markdown('<div class="section-header">🗺️ Analyse Géographique</div>', unsafe_allow_html=True)
    
    # Analyse par département
    dept_analysis = df_filtered.groupby('departement').agg({
        'Population_Totale': 'sum',
        'taux_participation': 'mean',
        'taux_chomage_pct': 'mean',
        'nb_crimes': 'sum',
        'commune_id': 'count'
    }).round(2)
    
    dept_analysis.columns = ['Population Totale', 'Taux Participation Moyen (%)', 'Taux Chômage Moyen (%)', 'Total Crimes', 'Nb Communes']
    dept_analysis = dept_analysis.sort_values('Population Totale', ascending=False).head(10)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Comparaison Participation vs Abstention
        participation_data = df_filtered.groupby('orientation_politique_label').agg({
            'taux_participation': 'mean',
            'taux_abstention': 'mean'
        }).round(1)
        
        fig_participation = px.bar(
            participation_data,
            title="Participation vs Abstention par Orientation",
            color_discrete_sequence=['#2ecc71', '#e74c3c'],
            barmode='group'
        )
        
        fig_participation.update_layout(
            xaxis_title="Orientation Politique",
            yaxis_title="Taux Moyen (%)",
            font=dict(size=12),
            margin=dict(l=20, r=20, t=50, b=20),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            legend=dict(title="Taux", orientation="v", yanchor="top", y=1, xanchor="left", x=1.05)
        )
        
        # Renommer les légendes
        fig_participation.data[0].name = "Participation"
        fig_participation.data[1].name = "Abstention"
        
        fig_participation.update_xaxes(tickangle=45, showgrid=False)
        fig_participation.update_yaxes(showgrid=True, gridcolor='rgba(128,128,128,0.2)')
        
        st.plotly_chart(fig_participation, use_container_width=True, key="bar_participation_comparison")
    
    with col2:
        # Top 10 des départements les plus sûrs (moins de crimes)
        dept_crimes = df_filtered.groupby('departement')['nb_crimes'].sum().sort_values(ascending=True).head(10)
        
        fig_safe_dept = px.bar(
            x=dept_crimes.index,
            y=dept_crimes.values,
            title="Top 10 Départements les Plus Sûrs",
            color=dept_crimes.values,
            color_continuous_scale='Greens',
            text=dept_crimes.values
        )
        
        fig_safe_dept.update_traces(
            texttemplate='%{text:.0f}',
            textposition='outside',
            textfont=dict(size=10, color='black')
        )
        
        fig_safe_dept.update_layout(
            xaxis_title="Département",
            yaxis_title="Nombre Total de Crimes",
            font=dict(size=12),
            margin=dict(l=20, r=20, t=50, b=20),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            showlegend=False
        )
        
        fig_safe_dept.update_xaxes(tickangle=0, showgrid=False)
        fig_safe_dept.update_yaxes(showgrid=True, gridcolor='rgba(128,128,128,0.2)')
        
        st.plotly_chart(fig_safe_dept, use_container_width=True, key="bar_safe_departments")
    
    # Tableau récapitulatif des départements
    st.subheader("📋 Tableau Récapitulatif - Top 10 Départements")
    
    # Formatage du tableau pour une meilleure lisibilité
    dept_display = dept_analysis.copy()
    dept_display['Population Totale'] = dept_display['Population Totale'].apply(lambda x: f"{x:,.0f}")
    dept_display['Total Crimes'] = dept_display['Total Crimes'].apply(lambda x: f"{x:.0f}")
    dept_display['Taux Participation Moyen (%)'] = dept_display['Taux Participation Moyen (%)'].apply(lambda x: f"{x:.1f}%")
    dept_display['Taux Chômage Moyen (%)'] = dept_display['Taux Chômage Moyen (%)'].apply(lambda x: f"{x:.1f}%")
    
    st.dataframe(
        dept_display,
        use_container_width=True,
        column_config={
            "Population Totale": st.column_config.TextColumn("👥 Population", width="medium"),
            "Taux Participation Moyen (%)": st.column_config.TextColumn("🗳️ Participation", width="medium"),
            "Taux Chômage Moyen (%)": st.column_config.TextColumn("💼 Chômage", width="medium"),
            "Total Crimes": st.column_config.TextColumn("🚨 Crimes", width="medium"),
            "Nb Communes": st.column_config.NumberColumn("🏘️ Communes", width="small")
        }
    )
    
    # === SECTION TABLEAU DE BORD INTERACTIF ===
    st.markdown('<div class="section-header">📋 Résumé Statistique</div>', unsafe_allow_html=True)
    
    # Statistiques descriptives
    stats_cols = ['taux_participation', 'taux_chomage_pct', 'pct_population_jeune', 'nb_crimes']
    stats_df = df_filtered[stats_cols].describe().round(2)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Statistiques Détaillées")
        st.dataframe(stats_df, use_container_width=True)
    
    with col2:
        st.subheader("🏆 Records")
        
        # Commune avec le plus haut taux de participation
        if len(df_filtered) > 0:
            max_participation = df_filtered.loc[df_filtered['taux_participation'].idxmax()]
            st.metric("🥇 Meilleur Taux de Participation", 
                     f"{max_participation['taux_participation']:.1f}%",
                     f"Commune {max_participation['commune_id']}")
            
            # Commune avec le plus faible taux de chômage
            min_chomage = df_filtered.loc[df_filtered['taux_chomage_pct'].idxmin()]
            st.metric("💼 Plus Faible Taux de Chômage", 
                     f"{min_chomage['taux_chomage_pct']:.1f}%",
                     f"Commune {min_chomage['commune_id']}")
            
            # Commune la plus sûre (moins de crimes)
            min_crimes = df_filtered.loc[df_filtered['nb_crimes'].idxmin()]
            st.metric("🛡️ Commune la Plus Sûre", 
                     f"{min_crimes['nb_crimes']:.0f} crimes",
                     f"Commune {min_crimes['commune_id']}")
        else:
            st.warning("Aucune donnée disponible pour les filtres sélectionnés.")
    
    # === ANALYSE SPÉCIFIQUE PAR RÉGION ===
    if region_selectionnee != 'Toute la France':
        st.markdown('<div class="section-header">🎯 Analyse Spécifique de la Région</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Comparaison avec la moyenne nationale
            df_national = df[df['annee'] == annee_selectionnee]
            
            comparaison_data = {
                'Indicateur': ['Taux Participation', 'Taux Chômage', '% Population Jeune', 'Crimes par 1000 hab'],
                'Région': [
                    df_filtered['taux_participation'].mean(),
                    df_filtered['taux_chomage_pct'].mean(),
                    df_filtered['pct_population_jeune'].mean(),
                    (df_filtered['nb_crimes'].sum() / df_filtered['Population_Totale'].sum() * 1000) if df_filtered['Population_Totale'].sum() > 0 else 0
                ],
                'France': [
                    df_national['taux_participation'].mean(),
                    df_national['taux_chomage_pct'].mean(),
                    df_national['pct_population_jeune'].mean(),
                    (df_national['nb_crimes'].sum() / df_national['Population_Totale'].sum() * 1000) if df_national['Population_Totale'].sum() > 0 else 0
                ]
            }
            
            df_comp = pd.DataFrame(comparaison_data)
            
            fig_comp = px.bar(
                df_comp,
                x='Indicateur',
                y=['Région', 'France'],
                title=f"Comparaison {region_selectionnee} vs France",
                barmode='group',
                color_discrete_sequence=['#e74c3c', '#3498db']
            )
            
            fig_comp.update_layout(
                font=dict(size=12),
                margin=dict(l=20, r=20, t=50, b=20),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                legend=dict(title="Zone")
            )
            
            fig_comp.update_xaxes(tickangle=45, showgrid=False)
            fig_comp.update_yaxes(showgrid=True, gridcolor='rgba(128,128,128,0.2)')
            
            st.plotly_chart(fig_comp, use_container_width=True, key="bar_region_comparison")
        
        with col2:
            # Distribution des départements dans la région
            dept_in_region = df_filtered['departement'].value_counts()
            
            fig_dept_region = px.pie(
                values=dept_in_region.values,
                names=dept_in_region.index,
                title=f"Répartition des communes par département - {region_selectionnee}",
                hole=0.3
            )
            
            fig_dept_region.update_traces(
                textposition='inside',
                textinfo='percent+label',
                textfont_size=11
            )
            
            fig_dept_region.update_layout(
                font=dict(size=12),
                margin=dict(l=20, r=20, t=50, b=20),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                showlegend=True,
                legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.05)
            )
            
            st.plotly_chart(fig_dept_region, use_container_width=True, key="pie_dept_region")
    
    # === FOOTER ===
    st.markdown("---")
    zone_affichage = region_selectionnee if region_selectionnee != 'Toute la France' else 'France entière'
    st.markdown(f"""
    <div style="text-align: center; color: #7f8c8d; font-size: 0.9rem;">
        📊 Dashboard créé avec Streamlit & Plotly | 
        🔄 Données mises à jour pour l'année {annee_selectionnee} | 
        📍 {nb_communes:,} communes analysées - Zone: {zone_affichage}
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()