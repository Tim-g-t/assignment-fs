"""
Steam Publisher M&A Analysis Dashboard
Dynamic scoring system with portfolio tier composition control
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Gaming Publisher M&A Analysis",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .info-box {
        background-color: #e8f4f8;
        padding: 10px;
        border-left: 4px solid #0066cc;
        margin: 10px 0;
        border-radius: 4px;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .tier-must-have { color: #28a745; font-weight: bold; }
    .tier-strategic { color: #17a2b8; font-weight: bold; }
    .tier-monitor { color: #ffc107; font-weight: bold; }
    .stButton>button {
        background-color: #0066cc;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Data loading with caching
@st.cache_data
def load_data():
    """Load the cleaned Steam dataset"""
    data_path = Path("/Users/timtoepper/Downloads/Banks_Assignment_Tim/output/steam_data_cleaned_complete.csv")
    
    if not data_path.exists():
        st.error(f"Data file not found at {data_path}")
        st.stop()
    
    df = pd.read_csv(data_path, low_memory=False)
    
    # Ensure boolean columns
    bool_cols = ['is_free', 'has_multiplayer', 'has_singleplayer', 'has_coop', 'has_vr', 
                 'is_indie', 'is_action', 'is_adventure', 'is_rpg', 'is_strategy',
                 'is_commercial_success', 'is_critical_success', 'is_hidden_gem', 
                 'is_zombie_game', 'is_sequel', 'is_pc_exclusive']
    
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].fillna(False).astype(bool)
    
    if 'release_date' in df.columns:
        df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
    
    return df

def calculate_publisher_metrics(df, weights):
    """Calculate comprehensive publisher metrics with custom weights"""
    
    # Filter valid publishers
    df_clean = df[df['publisher'].notna() & (df['publisher'] != 'Unknown')].copy()
    
    # Core aggregations
    agg_dict = {
        'app_id': 'count',
        'owners_avg': lambda x: x.fillna(0).sum(),
        'estimated_revenue_usd': lambda x: x.fillna(0).sum(),
        'positive_ratio': lambda x: x[x > 0].mean() if len(x[x > 0]) > 0 else 0,
        'total_reviews': lambda x: x.fillna(0).sum(),
        'concurrent_users_yesterday': lambda x: x.fillna(0).sum(),
        'playtime_average_forever': lambda x: x.fillna(0).mean(),
        'game_age_years': lambda x: x.fillna(0).mean(),
    }
    
    # Conditional aggregations
    for col in ['is_commercial_success', 'is_critical_success', 'is_hidden_gem', 
                'is_zombie_game', 'is_sequel', 'has_multiplayer', 'has_vr']:
        if col in df.columns:
            agg_dict[col] = 'sum'
    
    # Add genre diversity
    if 'primary_genre' in df_clean.columns:
        agg_dict['primary_genre'] = lambda x: x.nunique()
    
    publisher_df = df_clean.groupby('publisher').agg(agg_dict)
    
    # Rename columns
    rename_map = {
        'app_id': 'portfolio_size',
        'owners_avg': 'total_users',
        'estimated_revenue_usd': 'total_revenue',
        'positive_ratio': 'avg_rating',
        'concurrent_users_yesterday': 'active_users',
        'playtime_average_forever': 'avg_playtime',
        'game_age_years': 'avg_game_age',
        'is_commercial_success': 'commercial_successes',
        'is_critical_success': 'critical_successes',
        'is_hidden_gem': 'hidden_gems',
        'is_zombie_game': 'zombie_games',
        'is_sequel': 'sequels',
        'has_multiplayer': 'multiplayer_games',
        'has_vr': 'vr_games',
        'primary_genre': 'genre_diversity'
    }
    publisher_df = publisher_df.rename(columns=rename_map)
    
    # Calculate derived metrics
    publisher_df['success_rate'] = publisher_df['commercial_successes'] / publisher_df['portfolio_size']
    publisher_df['critical_rate'] = publisher_df['critical_successes'] / publisher_df['portfolio_size']
    publisher_df['zombie_rate'] = publisher_df['zombie_games'] / publisher_df['portfolio_size']
    publisher_df['sequel_ratio'] = publisher_df['sequels'] / publisher_df['portfolio_size']
    publisher_df['revenue_per_game'] = publisher_df['total_revenue'] / publisher_df['portfolio_size']
    publisher_df['users_per_game'] = publisher_df['total_users'] / publisher_df['portfolio_size']
    
    # Recent activity
    recent_mask = df_clean['game_age_years'] <= 3
    recent_releases = df_clean[recent_mask].groupby('publisher').size()
    publisher_df['recent_releases'] = recent_releases.reindex(publisher_df.index, fill_value=0)
    publisher_df['release_cadence'] = publisher_df['recent_releases'] / 3
    
    # Growth potential score (important for Tier 3)
    publisher_df['growth_potential'] = (
        publisher_df['release_cadence'] * 0.3 +
        publisher_df['success_rate'] * 0.3 +
        (publisher_df['recent_releases'] / (publisher_df['portfolio_size'] + 1)) * 0.2 +
        (1 - publisher_df['zombie_rate']) * 0.2
    ) * 100
    
    # Calculate component scores
    max_vals = {
        'portfolio': max(publisher_df['portfolio_size'].max(), 1),
        'revenue': max(publisher_df['total_revenue'].max(), 1),
        'users': max(publisher_df['total_users'].max(), 1),
        'active': max(publisher_df['active_users'].max(), 1),
        'rpg': max(publisher_df['revenue_per_game'].max(), 1),
        'upg': max(publisher_df['users_per_game'].max(), 1)
    }
    
    publisher_df['content_score'] = (
        publisher_df['release_cadence'] * weights['release_cadence'] +
        (publisher_df['portfolio_size'] / max_vals['portfolio']) * weights['portfolio_size'] +
        publisher_df['sequel_ratio'] * weights['franchise_strength']
    ) * 100
    
    publisher_df['quality_score'] = (
        publisher_df['critical_rate'] * weights['critical_success'] +
        publisher_df['success_rate'] * weights['commercial_success'] +
        publisher_df['avg_rating'] * weights['user_rating'] +
        (1 - publisher_df['zombie_rate']) * weights['avoid_failures']
    ) * 100
    
    publisher_df['market_score'] = (
        (publisher_df['total_revenue'] / max_vals['revenue']) * weights['revenue'] +
        (publisher_df['total_users'] / max_vals['users']) * weights['user_base'] +
        (publisher_df['active_users'] / max_vals['active']) * weights['active_users']
    ) * 100
    
    publisher_df['efficiency_score'] = (
        (publisher_df['revenue_per_game'] / max_vals['rpg']) * weights['revenue_efficiency'] +
        (publisher_df['users_per_game'] / max_vals['upg']) * weights['user_efficiency']
    ) * 100
    
    # Calculate final M&A score
    publisher_df['ma_score'] = (
        publisher_df['content_score'] * weights['content_weight'] +
        publisher_df['quality_score'] * weights['quality_weight'] +
        publisher_df['market_score'] * weights['market_weight'] +
        publisher_df['efficiency_score'] * weights['efficiency_weight']
    )
    
    # Assign tiers - FIXED naming
    # Must Have: Top performers (>70 score)
    # Strategic: Mid-tier filling gaps (50-70 score)  
    # Growth: High potential smaller publishers
    publisher_df['tier'] = pd.cut(
        publisher_df['ma_score'],
        bins=[0, 50, 70, 100],
        labels=['Growth', 'Strategic', 'Must Have']
    )
    
    # Override for high growth potential publishers
    high_growth = (publisher_df['growth_potential'] > 60) & (publisher_df['portfolio_size'] < 20)
    publisher_df.loc[high_growth & (publisher_df['tier'] != 'Must Have'), 'tier'] = 'Growth'
    
    return publisher_df.sort_values('ma_score', ascending=False)

def build_portfolio(publisher_df, composition, portfolio_size):
    """Build a portfolio based on user-defined tier composition"""
    portfolio = pd.DataFrame()
    
    # Get publishers by tier
    must_have_publishers = publisher_df[publisher_df['tier'] == 'Must Have']
    strategic_publishers = publisher_df[publisher_df['tier'] == 'Strategic']
    growth_publishers = publisher_df[publisher_df['tier'] == 'Growth']
    
    # For Growth tier, prioritize by growth_potential
    growth_publishers = growth_publishers.sort_values('growth_potential', ascending=False)
    
    # Add publishers from each tier
    if composition['must_have'] > 0:
        portfolio = pd.concat([portfolio, must_have_publishers.head(composition['must_have'])])
    
    if composition['strategic'] > 0:
        available_strategic = strategic_publishers[~strategic_publishers.index.isin(portfolio.index)]
        portfolio = pd.concat([portfolio, available_strategic.head(composition['strategic'])])
    
    if composition['growth'] > 0:
        available_growth = growth_publishers[~growth_publishers.index.isin(portfolio.index)]
        portfolio = pd.concat([portfolio, available_growth.head(composition['growth'])])
    
    # If we don't have enough publishers in specified tiers, fill from top scores
    if len(portfolio) < portfolio_size:
        remaining = portfolio_size - len(portfolio)
        available = publisher_df[~publisher_df.index.isin(portfolio.index)]
        portfolio = pd.concat([portfolio, available.head(remaining)])
    
    return portfolio.sort_values('ma_score', ascending=False)

def main():
    st.title("Gaming Publisher M&A Analysis Dashboard")
    st.markdown("#### Strategic Acquisition Portfolio Builder with Dynamic Tier Composition")
    
    # Load data
    with st.spinner("Loading Steam dataset..."):
        df = load_data()
    
    # Portfolio Composition Control
    st.markdown("---")
    st.markdown("### Portfolio Configuration")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        pcol1, pcol2, pcol3 = st.columns(3)
        
        with pcol1:
            portfolio_size = st.selectbox(
                "Total Portfolio Size", 
                [5, 10, 15, 20, 25, 30],
                index=2,  # Default to 15
                help="Total number of publishers to include in the acquisition portfolio"
            )
        
        with pcol2:
            with st.expander("Tier Definitions"):
                st.markdown("""
                **Must Have (Tier 1)**: 2-3 publishers with highest overall scores (>70)  
                **Strategic (Tier 2)**: 5-7 publishers filling genre/audience gaps (50-70)  
                **Growth (Tier 3)**: 5-10 smaller publishers with high potential
                """)
        
        with pcol3:
            with st.expander("Scoring Methodology"):
                st.markdown("""
                M&A Score = Weighted average of:
                - Content Production (portfolio, releases)
                - Quality Metrics (success rates, ratings)
                - Market Position (revenue, users)
                - Efficiency (per-game economics)
                """)
    
    with col2:
        st.metric("Total Publishers", f"{df['publisher'].nunique():,}")
        st.metric("Total Games", f"{len(df):,}")
    
    # Tier composition inputs
    st.markdown("#### Configure Portfolio Tier Mix")
    
    tier_col1, tier_col2, tier_col3, tier_col4 = st.columns(4)
    
    with tier_col1:
        must_have_count = st.number_input(
            "Must Have (Tier 1)",
            min_value=0,
            max_value=min(3, portfolio_size),
            value=min(2, portfolio_size),
            help="2-3 publishers with highest overall scores"
        )
    
    with tier_col2:
        max_strategic = min(7, portfolio_size - must_have_count)
        strategic_count = st.number_input(
            "Strategic (Tier 2)",
            min_value=0,
            max_value=max_strategic,
            value=min(5, max_strategic),
            help="5-7 publishers filling genre/audience gaps"
        )
    
    with tier_col3:
        max_growth = min(10, portfolio_size - must_have_count - strategic_count)
        growth_count = st.number_input(
            "Growth (Tier 3)",
            min_value=0,
            max_value=max_growth,
            value=max_growth,
            help="5-10 smaller publishers with high potential"
        )
    
    with tier_col4:
        total = must_have_count + strategic_count + growth_count
        if total == portfolio_size:
            st.success(f"Total: {total}/{portfolio_size}")
        else:
            st.error(f"Total: {total}/{portfolio_size}")
            st.caption("Adjust tier counts")
    
    composition = {
        'must_have': must_have_count,
        'strategic': strategic_count,
        'growth': growth_count
    }
    
    st.markdown("---")
    
    # Sidebar - Weight Configuration
    st.sidebar.header("Configure Scoring Weights")
    
    with st.sidebar.expander("Component Weights", expanded=True):
        st.info("These weights determine the relative importance of each scoring component in the final M&A score.")
        
        content_weight = st.slider(
            "Content Production", 0.0, 1.0, 0.25, 0.05,
            help="Portfolio size, release frequency, franchise strength"
        )
        quality_weight = st.slider(
            "Quality & Success", 0.0, 1.0, 0.25, 0.05,
            help="Critical/commercial success rates, user ratings"
        )
        market_weight = st.slider(
            "Market Position", 0.0, 1.0, 0.30, 0.05,
            help="Total revenue, user base, active players"
        )
        efficiency_weight = st.slider(
            "Efficiency", 0.0, 1.0, 0.20, 0.05,
            help="Revenue per game, user acquisition efficiency"
        )
        
        # Normalize
        total_weight = content_weight + quality_weight + market_weight + efficiency_weight
        if total_weight > 0:
            content_weight /= total_weight
            quality_weight /= total_weight
            market_weight /= total_weight
            efficiency_weight /= total_weight
        
        st.markdown("**Normalized Weights:**")
        st.text(f"Content: {content_weight:.1%}")
        st.text(f"Quality: {quality_weight:.1%}")
        st.text(f"Market: {market_weight:.1%}")
        st.text(f"Efficiency: {efficiency_weight:.1%}")
    
    with st.sidebar.expander("Detailed Metric Weights"):
        st.markdown("**Content Metrics**")
        release_cadence = st.slider(
            "Release Cadence", 0.0, 1.0, 0.4, 0.1,
            help="Average games released per year (last 3 years)"
        )
        portfolio_size_weight = st.slider(
            "Portfolio Size", 0.0, 1.0, 0.3, 0.1,
            help="Total number of games published"
        )
        franchise_strength = st.slider(
            "Franchise Strength", 0.0, 1.0, 0.3, 0.1,
            help="Ratio of sequels indicating IP development"
        )
        
        st.markdown("**Quality Metrics**")
        critical_success = st.slider(
            "Critical Success Rate", 0.0, 1.0, 0.3, 0.1,
            help="Games with >80% positive reviews & >50 total reviews"
        )
        commercial_success = st.slider(
            "Commercial Success Rate", 0.0, 1.0, 0.3, 0.1,
            help="Games exceeding $100K in revenue"
        )
        user_rating = st.slider(
            "User Ratings", 0.0, 1.0, 0.2, 0.1,
            help="Average positive review percentage"
        )
        avoid_failures = st.slider(
            "Avoid Failures", 0.0, 1.0, 0.2, 0.1,
            help="Penalty for zombie games (<1000 users, <10 reviews)"
        )
        
        st.markdown("**Market Metrics**")
        revenue = st.slider(
            "Total Revenue", 0.0, 1.0, 0.4, 0.1,
            help="Cumulative revenue across portfolio"
        )
        user_base = st.slider(
            "User Base Size", 0.0, 1.0, 0.3, 0.1,
            help="Total unique users"
        )
        active_users = st.slider(
            "Active Users", 0.0, 1.0, 0.3, 0.1,
            help="Concurrent players (yesterday)"
        )
        
        st.markdown("**Efficiency Metrics**")
        revenue_efficiency = st.slider(
            "Revenue per Game", 0.0, 1.0, 0.5, 0.1,
            help="Average revenue per title"
        )
        user_efficiency = st.slider(
            "Users per Game", 0.0, 1.0, 0.5, 0.1,
            help="Average users per title"
        )
    
    with st.sidebar.expander("Filters"):
        min_games = st.number_input(
            "Min Portfolio Size", 1, 100, 2,
            help="Exclude publishers with fewer games"
        )
        min_revenue = st.number_input(
            "Min Revenue ($M)", 0, 1000, 0
        ) * 1_000_000
        exclude_zombies = st.checkbox(
            "Exclude high zombie rate (>50%)", False
        )
    
    # Compile weights
    weights = {
        'content_weight': content_weight,
        'quality_weight': quality_weight,
        'market_weight': market_weight,
        'efficiency_weight': efficiency_weight,
        'release_cadence': release_cadence,
        'portfolio_size': portfolio_size_weight,
        'franchise_strength': franchise_strength,
        'critical_success': critical_success,
        'commercial_success': commercial_success,
        'user_rating': user_rating,
        'avoid_failures': avoid_failures,
        'revenue': revenue,
        'user_base': user_base,
        'active_users': active_users,
        'revenue_efficiency': revenue_efficiency,
        'user_efficiency': user_efficiency,
    }
    
    # Calculate metrics
    publisher_df = calculate_publisher_metrics(df, weights)
    
    # Apply filters
    publisher_df = publisher_df[publisher_df['portfolio_size'] >= min_games]
    if min_revenue > 0:
        publisher_df = publisher_df[publisher_df['total_revenue'] >= min_revenue]
    if exclude_zombies:
        publisher_df = publisher_df[publisher_df['zombie_rate'] <= 0.5]
    
    # Build portfolio
    portfolio_df = build_portfolio(publisher_df, composition, portfolio_size)
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Portfolio Overview", "Detailed Analysis", "Individual Assessment", 
        "Market Intelligence", "Export Data"
    ])
    
    with tab1:
        st.header("Selected Acquisition Portfolio")
        
        # Summary metrics
        if len(portfolio_df) > 0:
            met_col1, met_col2, met_col3, met_col4, met_col5 = st.columns(5)
            
            with met_col1:
                st.metric(
                    "Total Value",
                    f"${portfolio_df['total_revenue'].sum()/1e9:.2f}B",
                    help="Combined revenue of selected publishers"
                )
            
            with met_col2:
                st.metric(
                    "Total Users",
                    f"{portfolio_df['total_users'].sum()/1e6:.1f}M",
                    help="Combined user base"
                )
            
            with met_col3:
                st.metric(
                    "Avg M&A Score",
                    f"{portfolio_df['ma_score'].mean():.1f}",
                    help="Portfolio average score"
                )
            
            with met_col4:
                st.metric(
                    "Total Games",
                    f"{portfolio_df['portfolio_size'].sum():,}",
                    help="Combined portfolio"
                )
            
            with met_col5:
                tier_counts = portfolio_df['tier'].value_counts()
                st.metric(
                    "Tier Distribution",
                    f"{tier_counts.get('Must Have', 0)}/{tier_counts.get('Strategic', 0)}/{tier_counts.get('Growth', 0)}",
                    help="Must Have/Strategic/Growth"
                )
            
            # Portfolio table
            st.subheader("Portfolio Composition")
            
            display_df = pd.DataFrame({
                'Publisher': portfolio_df.index,
                'M&A Score': portfolio_df['ma_score'].round(1),
                'Tier': portfolio_df['tier'],
                'Growth Potential': portfolio_df['growth_potential'].round(1),
                'Games': portfolio_df['portfolio_size'].astype(int),
                'Revenue ($M)': (portfolio_df['total_revenue'] / 1e6).round(1),
                'Users (M)': (portfolio_df['total_users'] / 1e6).round(2),
                'Success Rate': (portfolio_df['success_rate'] * 100).round(1),
                'Content': portfolio_df['content_score'].round(1),
                'Quality': portfolio_df['quality_score'].round(1),
                'Market': portfolio_df['market_score'].round(1),
                'Efficiency': portfolio_df['efficiency_score'].round(1)
            })
            
            # Style tier column
            def style_tier(val):
                if val == 'Must Have':
                    return 'background-color: #d4edda'
                elif val == 'Strategic':
                    return 'background-color: #d1ecf1'
                else:  # Growth
                    return 'background-color: #e8daef'
            
            styled_df = display_df.style.applymap(style_tier, subset=['Tier'])
            st.dataframe(styled_df, use_container_width=True, height=500)
            
            # Tier breakdown chart
            st.subheader("Portfolio Tier Distribution")
            
            tier_data = portfolio_df['tier'].value_counts()
            fig_tier = px.pie(
                values=tier_data.values,
                names=tier_data.index,
                color_discrete_map={
                    'Must Have': '#28a745',
                    'Strategic': '#17a2b8',
                    'Growth': '#6610f2'
                }
            )
            st.plotly_chart(fig_tier, use_container_width=True)
    
    with tab2:
        st.header("Comparative Analysis")
        
        # Scatter plot
        fig_scatter = px.scatter(
            publisher_df.head(100),
            x='total_revenue',
            y='total_users',
            size='portfolio_size',
            color='tier',
            hover_data=['ma_score', 'success_rate', 'growth_potential'],
            labels={
                'total_revenue': 'Total Revenue ($)',
                'total_users': 'Total Users'
            },
            title="Publisher Landscape (Top 100)",
            color_discrete_map={
                'Must Have': '#28a745',
                'Strategic': '#17a2b8',
                'Growth': '#6610f2'
            }
        )
        
        # Highlight portfolio
        if len(portfolio_df) > 0:
            fig_scatter.add_scatter(
                x=portfolio_df['total_revenue'],
                y=portfolio_df['total_users'],
                mode='markers',
                marker=dict(size=20, color='red', symbol='star'),
                name='Selected Portfolio',
                showlegend=True
            )
        
        fig_scatter.update_layout(height=500)
        st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Component comparison
        if len(portfolio_df) > 0:
            st.subheader("Component Score Analysis")
            
            components = ['content_score', 'quality_score', 'market_score', 'efficiency_score']
            
            # Portfolio average vs market average
            portfolio_avg = portfolio_df[components].mean()
            market_avg = publisher_df.head(100)[components].mean()
            
            comparison_df = pd.DataFrame({
                'Portfolio': portfolio_avg,
                'Market Top 100': market_avg
            }).T
            
            fig_comp = px.bar(
                comparison_df,
                barmode='group',
                title="Portfolio vs Market Average Scores",
                labels={'value': 'Score', 'index': 'Metric'},
                color_discrete_sequence=['#0066cc', '#cccccc']
            )
            st.plotly_chart(fig_comp, use_container_width=True)
    
    with tab3:
        st.header("Publisher Deep Dive")
        
        if len(portfolio_df) > 0:
            selected_publisher = st.selectbox(
                "Select a publisher from portfolio:",
                portfolio_df.index.tolist()
            )
            
            if selected_publisher:
                pub_data = portfolio_df.loc[selected_publisher]
                pub_games = df[df['publisher'] == selected_publisher]
                
                # Key metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("M&A Score", f"{pub_data['ma_score']:.1f}")
                    st.metric("Tier", pub_data['tier'])
                
                with col2:
                    st.metric("Portfolio Size", f"{pub_data['portfolio_size']:.0f}")
                    st.metric("Growth Potential", f"{pub_data['growth_potential']:.1f}")
                
                with col3:
                    st.metric("Total Revenue", f"${pub_data['total_revenue']/1e6:.1f}M")
                    st.metric("Rev/Game", f"${pub_data['revenue_per_game']/1e3:.0f}K")
                
                with col4:
                    st.metric("Total Users", f"{pub_data['total_users']/1e6:.2f}M")
                    st.metric("Success Rate", f"{pub_data['success_rate']*100:.1f}%")
                
                # Score breakdown
                st.subheader("Score Components")
                
                score_df = pd.DataFrame({
                    'Component': ['Content', 'Quality', 'Market', 'Efficiency'],
                    'Score': [
                        pub_data['content_score'],
                        pub_data['quality_score'],
                        pub_data['market_score'],
                        pub_data['efficiency_score']
                    ],
                    'Weight': [
                        weights['content_weight'],
                        weights['quality_weight'],
                        weights['market_weight'],
                        weights['efficiency_weight']
                    ]
                })
                score_df['Contribution'] = score_df['Score'] * score_df['Weight']
                
                fig_score = px.bar(
                    score_df,
                    x='Component',
                    y=['Score', 'Contribution'],
                    title=f"Score Analysis - {selected_publisher}",
                    barmode='group',
                    color_discrete_sequence=['#0066cc', '#28a745']
                )
                st.plotly_chart(fig_score, use_container_width=True)
                
                # Top games
                st.subheader("Top 10 Games")
                top_games = pub_games.nlargest(10, 'estimated_revenue_usd')[
                    ['name', 'estimated_revenue_usd', 'owners_avg', 'positive_ratio', 'release_date']
                ].copy()
                
                top_games['Revenue'] = (top_games['estimated_revenue_usd'] / 1e6).round(2)
                top_games['Users'] = (top_games['owners_avg'] / 1e3).round(1)
                top_games['Rating'] = (top_games['positive_ratio'] * 100).round(1)
                
                st.dataframe(
                    top_games[['name', 'Revenue', 'Users', 'Rating', 'release_date']].rename(
                        columns={'name': 'Game', 'Revenue': 'Revenue ($M)', 
                                'Users': 'Users (K)', 'Rating': 'Rating (%)',
                                'release_date': 'Release Date'}
                    ),
                    use_container_width=True
                )
    
    with tab4:
        st.header("Market Intelligence")
        
        # Tier distribution analysis
        st.subheader("Market Tier Distribution")
        
        tier_analysis = publisher_df.groupby('tier').agg({
            'ma_score': 'mean',
            'total_revenue': 'sum',
            'total_users': 'sum',
            'portfolio_size': 'count',
            'growth_potential': 'mean'
        }).round(1)
        
        tier_analysis.columns = ['Avg Score', 'Total Revenue', 'Total Users', 'Publisher Count', 'Avg Growth Potential']
        tier_analysis['Avg Revenue'] = tier_analysis['Total Revenue'] / tier_analysis['Publisher Count']
        
        st.dataframe(tier_analysis, use_container_width=True)
        
        # Release patterns
        if len(portfolio_df) > 0:
            st.subheader("Release Activity - Portfolio Publishers")
            
            portfolio_pubs = portfolio_df.index.tolist()[:10]
            timeline_data = df[df['publisher'].isin(portfolio_pubs)].copy()
            timeline_data['release_year'] = timeline_data['release_date'].dt.year
            
            yearly = timeline_data.groupby(['release_year', 'publisher']).size().reset_index(name='releases')
            yearly = yearly[(yearly['release_year'] >= 2018) & (yearly['release_year'] <= 2024)]
            
            fig_timeline = px.line(
                yearly,
                x='release_year',
                y='releases',
                color='publisher',
                title="Release Patterns (2018-2024)",
                markers=True
            )
            st.plotly_chart(fig_timeline, use_container_width=True)
        
        # Efficiency matrix
        st.subheader("Efficiency vs Quality Matrix")
        
        fig_matrix = px.scatter(
            publisher_df.head(50),
            x='efficiency_score',
            y='quality_score',
            size='total_revenue',
            color='tier',
            hover_name=publisher_df.head(50).index,
            hover_data=['ma_score', 'growth_potential'],
            color_discrete_map={
                'Must Have': '#28a745',
                'Strategic': '#17a2b8',
                'Growth': '#6610f2'
            }
        )
        
        fig_matrix.add_hline(y=50, line_dash="dash", line_color="gray", opacity=0.3)
        fig_matrix.add_vline(x=50, line_dash="dash", line_color="gray", opacity=0.3)
        
        fig_matrix.update_layout(height=500)
        st.plotly_chart(fig_matrix, use_container_width=True)
    
    with tab5:
        st.header("Export Analysis")
        
        if len(portfolio_df) > 0:
            # Export data
            export_df = portfolio_df.copy()
            export_df['Revenue_M'] = export_df['total_revenue'] / 1e6
            export_df['Users_M'] = export_df['total_users'] / 1e6
            export_df['Success_%'] = export_df['success_rate'] * 100
            export_df['Growth_Potential'] = export_df['growth_potential']
            
            export_columns = st.multiselect(
                "Select columns to export:",
                export_df.columns.tolist(),
                default=['ma_score', 'tier', 'Growth_Potential', 'portfolio_size', 'Revenue_M', 'Users_M', 'Success_%']
            )
            
            if export_columns:
                export_data = export_df[export_columns].round(2)
                
                csv = export_data.to_csv()
                st.download_button(
                    label="Download Portfolio Analysis (CSV)",
                    data=csv,
                    file_name="ma_portfolio_analysis.csv",
                    mime="text/csv"
                )
                
                st.subheader("Export Preview")
                st.dataframe(export_data, use_container_width=True)
                
                # Configuration summary
                st.subheader("Configuration Summary")
                config_df = pd.DataFrame({
                    'Parameter': [
                        'Portfolio Size',
                        'Must Have Count',
                        'Strategic Count',
                        'Growth Count',
                        'Content Weight',
                        'Quality Weight',
                        'Market Weight',
                        'Efficiency Weight'
                    ],
                    'Value': [
                        portfolio_size,
                        must_have_count,
                        strategic_count,
                        growth_count,
                        f"{content_weight:.1%}",
                        f"{quality_weight:.1%}",
                        f"{market_weight:.1%}",
                        f"{efficiency_weight:.1%}"
                    ]
                })
                st.dataframe(config_df, use_container_width=True)

if __name__ == "__main__":
    main()