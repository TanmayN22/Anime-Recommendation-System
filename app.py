import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

@st.cache_data
def load_data():
    df = pd.read_csv("anime.csv")  
    df = df.dropna(subset=['Genres'])
    return df

def process_data(df):
    tfidf = TfidfVectorizer(stop_words='english')
    df['Genres'] = df['Genres'].fillna('')
    tfidf_matrix = tfidf.fit_transform(df['Genres'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    return cosine_sim

def recommend_anime(title, cosine_sim, df):
    idx = df[df['Name'] == title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    anime_indices = [i[0] for i in sim_scores]
    return df[['Name', 'Genres', 'Score']].iloc[anime_indices]

def main():
    st.set_page_config(page_title="Anime Recommendation System", page_icon="🎥", layout="wide")
    st.markdown(
        """<style>
        .header-image {
            width: 100%;
            height: auto;
            max-height: 200px; /* Change this value for smaller height */
            object-fit: cover; /* Adjusts the aspect ratio */
        }
        </style>
        """, unsafe_allow_html=True
    )
    st.image("image/anime2.jpeg", use_column_width=True, output_format='auto')
    st.title("Anime Recommendation System 🎥")
    st.markdown("""
        Welcome to the **Anime Recommendation System**!  
        Select your favorite anime from the dropdown and get **top 10** recommendations based on similarity!
        """)
    df = load_data()
    anime_list = df['Name'].values
    selected_anime = st.selectbox("Select an Anime 🎬", anime_list)
    cosine_sim = process_data(df)
    if st.button("Recommend"):
        st.markdown("### 🎉 Top 10 Recommendations for You 🎉")
        recommendations = recommend_anime(selected_anime, cosine_sim, df)
        for index, row in recommendations.iterrows():
            st.markdown(f"""
                - **Name:** {row['Name']}
                - **Genres:** {row['Genres']}
                - **Score:** {row['Score']}  
                ---
            """)
    st.markdown("""
        <br><br><hr>
        <div style='text-align: center;'>
        Developed by Tanmay Nayak, Kaustubh Gondakar, Tejas Gadekar, Piyush Das
        </div>
    """, unsafe_allow_html=True)

if __name__ == '__main__':
    main()
