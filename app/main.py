import streamlit as st
from src.database.init_db import init_db

# Ensure that this is the very first Streamlit command
st.set_page_config(
    page_title="Traffic Density Recognition System",
    page_icon="./src/assets/Favicon-2.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import other modules after setting page config
from src.state import provide_state, session_state_hash_func
from src.pages import *
from src.pages import PAGE_MAP
from streamlit_option_menu import option_menu

@provide_state()
def main(state=None):
    
    with st.sidebar:
        selected_page = option_menu(
            menu_title="TDRS",
            options=list(PAGE_MAP.keys()),
            icons=["columns-gap", "file-earmark-richtext", 'camera-video'],
            menu_icon="car-front",
            default_index=0,
            styles={
                "menu-title": {"font-size": "30px", "color": "#333", "font-weight": "500"},
                "container": {"padding": "5!important", "font-family": "Poppins"},
                "icon": {"font-size": "20px"},
                "nav-link": {
                    "font-size": "20px",
                    "text-align": "left",
                    "margin-top": "8px",
                    "padding": "14px 12px",
                    "box-shadow": "0 1px 1px rgba(0, 0, 0, 0.1)"
                },
                "nav-link-selected": {"background-color": "#4070f4", "color": "white", "font-weight": "500"},
            }
        )
    init_db()  
    PAGE_MAP[selected_page](state=state).write()

if __name__ == "__main__":
    main()
