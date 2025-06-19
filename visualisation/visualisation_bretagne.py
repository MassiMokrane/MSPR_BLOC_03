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
    page_title="🏴‍☠️ Dashboard Électoral Bretagne",
    page_icon="⚓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé pour un style élégant avec thème breton
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #1a237e, #3949ab);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .kpi-container {
        background: linear-gradient(135deg, #1565c0 0%, #1976d2 100%);
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
        color: #1565c0;
        margin: 2rem 0 1rem 0;
        border-left: 5px solid #1976d2;
        padding-left: 1rem;
    }
    
    .bretagne-info {
        background: linear-gradient(135deg, #eceff1 0%, #cfd8dc 100%);
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1976d2;
        margin: 1rem 0;
    }
    
    .dept-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 4px 16px rgba(0,0,0,0.1);
        border: 1px solid #e1e8ed;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """
    Charge et prépare les données spécifiquement pour la Bretagne
    """
    # Remplacez par votre chemin de fichier
    file_path = "C:/Users/Massi/Desktop/MSPR BLOC 03/les_jointeurs/datasets/jointeur_elections_population/Dataset_Finale_MSPR_2025.csv"
    
    # Départements bretons
    DEPARTEMENTS_BRETAGNE = ['22', '29', '35', '56']
    NOMS_DEPARTEMENTS = {
        '22': 'Côtes-d\'Armor',
        '29': 'Finistère',
        '35': 'Ille-et-Vilaine',
        '56': 'Morbihan'
    }

    try:
        df = pd.read_csv(file_path, sep=';')
        
        # Extraction du département depuis commune_id
        df['departement'] = df['commune_id'].astype(str).str[:2]
        
        # FILTRAGE SPÉCIFIQUE À LA BRETAGNE
        df = df[df['departement'].isin(DEPARTEMENTS_BRETAGNE)].copy()
        
        # Ajout du nom du département
        df['nom_departement'] = df['departement'].map(NOMS_DEPARTEMENTS)
        
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
        
        return df
    
    except FileNotFoundError:
        # Données de démonstration spécifiques à la Bretagne
        np.random.seed(42)
        n_communes = 1000
        
        # Génération de communes bretonnes réalistes
        departements_bretagne = ['22', '29', '35', '56']
        commune_ids = []
        for dept in departements_bretagne:
            for i in range(1, 251):  # ~250 communes par département
                commune_ids.append(f"{dept}{i:03d}")
        
        n_communes = len(commune_ids)
        
        data = {
            'commune_id': commune_ids,
            'annee': np.random.choice([2017, 2022], n_communes),
            'nb_inscrits': np.random.randint(100, 8000, n_communes),
            'nb_abstentions': np.random.randint(10, 2500, n_communes),
            'orientation_politique': np.random.choice([0, 1, 2, 3, 4], n_communes, p=[0.08, 0.18, 0.32, 0.28, 0.14]),
            'Population_Totale': np.random.randint(150, 12000, n_communes),
            'pct_population_jeune': np.random.uniform(18, 32, n_communes),
            'pct_population_senior': np.random.uniform(15, 28, n_communes),
            'pct_population_sans_activite': np.random.uniform(22, 38, n_communes),
            'pct_population_etrangere': np.random.uniform(1, 8, n_communes),
            'nb_population_active': np.random.randint(80, 6000, n_communes),
            'taux_chomage_pct': np.random.uniform(0.04, 0.15, n_communes),
            'nb_crimes': np.random.randint(0, 800, n_communes)  # Généralement plus faible en Bretagne
        }
        
        df = pd.DataFrame(data)
        
        # Extraction du département
        df['departement'] = df['commune_id'].astype(str).str[:2]
        
        # Ajout du nom du département
        NOMS_DEPARTEMENTS = {
            '22': 'Côtes-d\'Armor',
            '29': 'Finistère',
            '35': 'Ille-et-Vilaine',
            '56': 'Morbihan'
        }
        df['nom_departement'] = df['departement'].map(NOMS_DEPARTEMENTS)
        
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
        
        return df

def create_kpi_card(value, label, color_start="#1565c0", color_end="#1976d2"):
    """
    Crée une carte KPI stylisée avec les couleurs bretonnes
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
    # Titre principal avec thème breton
    st.markdown('<h1 class="main-header">🏴‍☠️ Dashboard Électoral et Démographique de Bretagne</h1>', unsafe_allow_html=True)
    
    # Information sur la région
    st.markdown("""
    <div class="bretagne-info">
        <h3>⚓ Région Bretagne</h3>
        <p><strong>Départements analysés :</strong> Côtes-d'Armor (22), Finistère (29), Ille-et-Vilaine (35), Morbihan (56)</p>
        <p><strong>Spécificités :</strong> Analyse électorale et démographique de la péninsule bretonne</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Chargement des données
    with st.spinner('🔄 Chargement des données bretonnes...'):
        df = load_data()
    
    # Vérification que nous avons bien des données bretonnes
    if df.empty:
        st.error("❌ Aucune donnée bretonne trouvée dans le dataset")
        return
    
    # Sidebar pour les filtres
    st.sidebar.header("🎛️ Filtres Bretagne")
    
    # Filtre par année
    annees_disponibles = sorted(df['annee'].unique())
    annee_selectionnee = st.sidebar.selectbox("📅 Année", annees_disponibles, index=len(annees_disponibles)-1)
    
    # Filtre par département breton
    departements_disponibles = ['Tous'] + list(df['nom_departement'].unique())
    departement_selectionne = st.sidebar.selectbox("🏛️ Département", departements_disponibles)
    
    # Filtre par orientation politique
    orientations_disponibles = ['Toutes'] + list(df['orientation_politique_label'].unique())
    orientation_selectionnee = st.sidebar.selectbox("🗳️ Orientation Politique", orientations_disponibles)
    
    # Filtrage des données
    df_filtered = df[df['annee'] == annee_selectionnee].copy()
    
    if departement_selectionne != 'Tous':
        df_filtered = df_filtered[df_filtered['nom_departement'] == departement_selectionne]
    
    if orientation_selectionnee != 'Toutes':
        df_filtered = df_filtered[df_filtered['orientation_politique_label'] == orientation_selectionnee]
    
    # === SECTION KPI PRINCIPAUX ===
    st.markdown('<div class="section-header">📊 KPIs Bretagne</div>', unsafe_allow_html=True)
    
    # Calcul des KPIs
    nb_communes = len(df_filtered)
    population_totale = df_filtered['Population_Totale'].sum()
    taux_participation_moyen = df_filtered['taux_participation'].mean()
    taux_chomage_moyen = df_filtered['taux_chomage_pct'].mean()
    total_crimes = df_filtered['nb_crimes'].sum()
    nb_departements = df_filtered['nom_departement'].nunique()
    
    # Affichage des KPIs en colonnes
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        st.markdown(create_kpi_card(f"{nb_departements}", "Départements", "#d32f2f", "#c62828"), unsafe_allow_html=True)
    
    with col2:
        st.markdown(create_kpi_card(f"{nb_communes:,}", "Communes", "#1565c0", "#1976d2"), unsafe_allow_html=True)
    
    with col3:
        st.markdown(create_kpi_card(f"{population_totale:,.0f}", "Population", "#388e3c", "#4caf50"), unsafe_allow_html=True)
    
    with col4:
        st.markdown(create_kpi_card(f"{taux_participation_moyen:.1f}%", "Participation", "#f57c00", "#ff9800"), unsafe_allow_html=True)
    
    with col5:
        st.markdown(create_kpi_card(f"{taux_chomage_moyen:.1f}%", "Chômage", "#7b1fa2", "#9c27b0"), unsafe_allow_html=True)
    
    with col6:
        st.markdown(create_kpi_card(f"{total_crimes:,.0f}", "Crimes", "#5d4037", "#795548"), unsafe_allow_html=True)
    
    # === SECTION ANALYSE PAR DÉPARTEMENT BRETON ===
    st.markdown('<div class="section-header">🗺️ Analyse par Département Breton</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Comparaison des départements bretons
        dept_stats = df_filtered.groupby('nom_departement').agg({
            'Population_Totale': 'sum',
            'taux_participation': 'mean',
            'taux_chomage_pct': 'mean',
            'nb_crimes': 'sum'
        }).round(2)
        
        fig_dept = px.bar(
            x=dept_stats.index,
            y=dept_stats['Population_Totale'],
            title="Population par Département Breton",
            color=dept_stats['Population_Totale'],
            color_continuous_scale='Blues',
            text=dept_stats['Population_Totale'].apply(lambda x: f"{x:,.0f}")
        )
        
        fig_dept.update_traces(
            textposition='outside',
            textfont=dict(size=11, color='black')
        )
        
        fig_dept.update_layout(
            xaxis_title="Département",
            yaxis_title="Population Totale",
            font=dict(size=12),
            margin=dict(l=20, r=20, t=50, b=20),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            showlegend=False
        )
        
        fig_dept.update_xaxes(tickangle=45, showgrid=False)
        fig_dept.update_yaxes(showgrid=True, gridcolor='rgba(128,128,128,0.2)')
        
        st.plotly_chart(fig_dept, use_container_width=True, key="bar_dept_population")
    
    with col2:
        # Taux de participation par département
        participation_dept = df_filtered.groupby('nom_departement')['taux_participation'].mean().sort_values(ascending=True)
        
        fig_participation_dept = px.bar(
            x=participation_dept.values,
            y=participation_dept.index,
            orientation='h',
            title="Taux de Participation par Département",
            color=participation_dept.values,
            color_continuous_scale='Greens',
            text=[f"{val:.1f}%" for val in participation_dept.values]
        )
        
        fig_participation_dept.update_traces(
            textposition='inside',
            textfont=dict(color='white', size=12, family='Arial Black')
        )
        
        fig_participation_dept.update_layout(
            xaxis_title="Taux de Participation (%)",
            yaxis_title="Département",
            font=dict(size=12),
            margin=dict(l=20, r=20, t=50, b=20),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            showlegend=False
        )
        
        fig_participation_dept.update_xaxes(showgrid=True, gridcolor='rgba(128,128,128,0.2)')
        fig_participation_dept.update_yaxes(showgrid=False)
        
        st.plotly_chart(fig_participation_dept, use_container_width=True, key="bar_participation_dept")
    
    # === SECTION ANALYSE POLITIQUE BRETONNE ===
    st.markdown('<div class="section-header">🏛️ Paysage Politique Breton</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Distribution des orientations politiques en Bretagne
        orientation_counts = df_filtered['orientation_politique_label'].value_counts()
        
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
            title="Orientations Politiques en Bretagne",
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
        
        st.plotly_chart(fig_pie, use_container_width=True, key="pie_political_bretagne")
    
    with col2:
        # Orientation politique par département
        political_by_dept = df_filtered.groupby(['nom_departement', 'orientation_politique_label']).size().unstack(fill_value=0)
        political_by_dept_pct = political_by_dept.div(political_by_dept.sum(axis=1), axis=0) * 100
        
        fig_political_dept = px.bar(
            political_by_dept_pct,
            title="Répartition Politique par Département (%)",
            color_discrete_map=colors_political,
            barmode='stack'
        )
        
        fig_political_dept.update_layout(
            xaxis_title="Département",
            yaxis_title="Pourcentage (%)",
            font=dict(size=12),
            margin=dict(l=20, r=20, t=50, b=20),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            legend=dict(title="Orientation", orientation="v", yanchor="top", y=1, xanchor="left", x=1.05)
        )
        
        fig_political_dept.update_xaxes(tickangle=45, showgrid=False)
        fig_political_dept.update_yaxes(showgrid=True, gridcolor='rgba(128,128,128,0.2)')
        
        st.plotly_chart(fig_political_dept, use_container_width=True, key="bar_political_dept")
    
    # === SECTION PROFIL DÉMOGRAPHIQUE BRETON ===
    st.markdown('<div class="section-header">👥 Profil Démographique Breton</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Pyramide démographique par département
        demo_by_dept = df_filtered.groupby('nom_departement')[
            ['pct_population_jeune', 'pct_population_senior', 'pct_population_etrangere']
        ].mean().round(1)
        
        fig_demo_dept = px.bar(
            demo_by_dept,
            title="Profil Démographique par Département",
            color_discrete_sequence=['#2196f3', '#ff5722', '#4caf50'],
            barmode='group'
        )
        
        fig_demo_dept.update_layout(
            xaxis_title="Département",
            yaxis_title="Pourcentage Moyen (%)",
            font=dict(size=12),
            margin=dict(l=20, r=20, t=50, b=20),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            legend=dict(title="Indicateurs", orientation="v", yanchor="top", y=1, xanchor="left", x=1.05)
        )
        
        fig_demo_dept.update_xaxes(tickangle=45, showgrid=False)
        fig_demo_dept.update_yaxes(showgrid=True, gridcolor='rgba(128,128,128,0.2)')
        
        # Renommer les légendes
        fig_demo_dept.data[0].name = "% Jeunes"
        fig_demo_dept.data[1].name = "% Seniors" 
        fig_demo_dept.data[2].name = "% Étrangers"
        
        st.plotly_chart(fig_demo_dept, use_container_width=True, key="bar_demo_dept")
    
    with col2:
        # Corrélation Chômage vs Crimes par département
        chomage_crimes = df_filtered.groupby('nom_departement').agg({
            'taux_chomage_pct': 'mean',
            'nb_crimes': 'sum'
        }).round(2)
        
        fig_correlation = px.scatter(
            chomage_crimes,
            x='taux_chomage_pct',
            y='nb_crimes',
            title="Relation Chômage-Criminalité par Département",
            text=chomage_crimes.index,
            size_max=20
        )
        
        fig_correlation.update_traces(
            textposition='top center',
            textfont=dict(size=12, color='black'),
            marker=dict(size=15, color='#1976d2', opacity=0.7)
        )
        
        fig_correlation.update_layout(
            xaxis_title="Taux de Chômage Moyen (%)",
            yaxis_title="Nombre Total de Crimes",
            font=dict(size=12),
            margin=dict(l=20, r=20, t=50, b=20),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        
        fig_correlation.update_xaxes(showgrid=True, gridcolor='rgba(128,128,128,0.2)')
        fig_correlation.update_yaxes(showgrid=True, gridcolor='rgba(128,128,128,0.2)')
        
        st.plotly_chart(fig_correlation, use_container_width=True, key="scatter_correlation")
    
    # === TABLEAU RÉCAPITULATIF DÉPARTEMENTS BRETONS ===
    st.markdown('<div class="section-header">📋 Tableau Récapitulatif Bretagne</div>', unsafe_allow_html=True)
    
    # Statistiques détaillées par département
    dept_summary = df_filtered.groupby('nom_departement').agg({
        'commune_id': 'count',
        'Population_Totale': 'sum',
        'taux_participation': 'mean',
        'taux_chomage_pct': 'mean',
        'nb_crimes': 'sum',
        'pct_population_jeune': 'mean',
        'pct_population_senior': 'mean'
    }).round(2)
    
    dept_summary.columns = [
        'Nb Communes', 'Population', 'Participation (%)', 
        'Chômage (%)', 'Total Crimes', '% Jeunes', '% Seniors'
    ]
    
    # Formatage pour l'affichage
    dept_display = dept_summary.copy()
    dept_display['Population'] = dept_display['Population'].apply(lambda x: f"{x:,.0f}")
    dept_display['Total Crimes'] = dept_display['Total Crimes'].apply(lambda x: f"{x:.0f}")
    
    for col in ['Participation (%)', 'Chômage (%)', '% Jeunes', '% Seniors']:
        dept_display[col] = dept_display[col].apply(lambda x: f"{x:.1f}%")
    
    st.dataframe(
        dept_display,
        use_container_width=True,
        column_config={
            "Nb Communes": st.column_config.NumberColumn("🏘️ Communes", width="small"),
            "Population": st.column_config.TextColumn("👥 Population", width="medium"),
            "Participation (%)": st.column_config.TextColumn("🗳️ Participation", width="medium"),
            "Chômage (%)": st.column_config.TextColumn("💼 Chômage", width="medium"),
            "Total Crimes": st.column_config.TextColumn("🚨 Crimes", width="medium"),
            "% Jeunes": st.column_config.TextColumn("👶 Jeunes", width="small"),
            "% Seniors": st.column_config.TextColumn("👴 Seniors", width="small")
        }
    )
    
    # === SECTION RECORDS BRETAGNE ===
    st.markdown('<div class="section-header">🏆 Records Bretagne</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("🥇 Meilleurs Indicateurs")
        
         # Département avec le plus haut taux de participation
        best_participation_dept = dept_summary['Participation (%)'].idxmax()
        best_participation_val = dept_summary.loc[best_participation_dept, 'Participation (%)']
        st.metric("🗳️ Meilleure Participation", 
                 f"{best_participation_val:.1f}%",
                 f"{best_participation_dept}")
        
        # Département le plus sûr
        safest_dept = dept_summary['Total Crimes'].idxmin()
        safest_crimes = dept_summary.loc[safest_dept, 'Total Crimes']
        st.metric("🛡️ Plus Sûr", 
                 f"{safest_crimes:.0f} crimes",
                 f"{safest_dept}")
    
    with col2:
        st.subheader("📊 Moyennes Bretagne")
        
        # Moyennes régionales
        avg_participation = df_filtered['taux_participation'].mean()
        avg_chomage = df_filtered['taux_chomage_pct'].mean()
        avg_jeunes = df_filtered['pct_population_jeune'].mean()
        
        st.metric("🗳️ Participation Moyenne", f"{avg_participation:.1f}%")
        st.metric("💼 Chômage Moyen", f"{avg_chomage:.1f}%")
        st.metric("👶 % Jeunes Moyen", f"{avg_jeunes:.1f}%")
    
    with col3:
        st.subheader("🔍 Points d'Attention")
        
        # Département avec le plus fort taux de chômage
        highest_chomage_dept = dept_summary['Chômage (%)'].idxmax()
        highest_chomage_val = dept_summary.loc[highest_chomage_dept, 'Chômage (%)']
        st.metric("⚠️ Chômage le Plus Élevé", 
                 f"{highest_chomage_val:.1f}%",
                 f"{highest_chomage_dept}")
        
        # Département avec le plus de crimes
        most_crimes_dept = dept_summary['Total Crimes'].idxmax()
        most_crimes_val = dept_summary.loc[most_crimes_dept, 'Total Crimes']
        st.metric("🚨 Plus de Criminalité", 
                 f"{most_crimes_val:.0f} crimes",
                 f"{most_crimes_dept}")
    
    # === FOOTER BRETON ===
    st.markdown("---")
    st.markdown(f"""
    <div style="text-align: center; color: #1565c0; font-size: 0.9rem;">
        ⚓ Dashboard Bretagne créé avec Streamlit & Plotly | 
        🔄 Données mises à jour pour l'année {annee_selectionnee} | 
        🗺️ {nb_communes:,} communes bretonnes analysées | 
        🏴‍☠️ {nb_departements} départements bretons
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()