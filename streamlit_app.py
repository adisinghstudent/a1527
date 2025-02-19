import os
import base64
import torch
import streamlit as st
from tic_tac_toe import TicTacToe
from alphazero_nn import AlphaZeroNN
from mcts import MCTS
from logger import logger
# ================================
# 🎨 Apply Custom Styling with Background Image
# ================================
st.markdown(
    """
    <style>
    /* Set the background to use sky.png */
    .stApp {
        background: url('sky.png') no-repeat center center fixed;
        background-size: cover;
    }

    /* Center title with black color */
    .title {
        text-align: center;
        font-size: 50px;
        font-weight: bold;
        margin-bottom: 30px;
        color: black; 
    }

    /* Space between board and title */
    .board-container {
        margin-top: 40px;
        text-align: center;
    }

    /* Game status */
    .game-status {
        margin-top: 20px;
        font-size: 22px;
        text-align: center;
    }

    /* Ensure button consistency */
    .stButton>button {
        font-size: 32px;
        width: 100px;
        height: 100px;
        border-radius: 10px;
        border: 3px solid white;
        background-color: rgba(255, 255, 255, 0.2);
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ================================
# 📷 Convert Image to Base64 for Background
# ================================
def get_base64_of_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()

# Ensure correct image path
IMAGE_PATH = os.path.join(os.path.dirname(__file__), "sky_fixed.png")

# Convert image to base64
if os.path.exists(IMAGE_PATH):
    bg_base64 = get_base64_of_image(IMAGE_PATH)
    background_css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{bg_base64}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }}
    </style>
    """
    st.markdown(background_css, unsafe_allow_html=True)
else:
    st.error("🚨 Background image 'sky_fixed.png' not found!")
# ================================
# 📥 Load Latest Model
# ================================
MODEL_DIR = "models/"

# Get the latest model file dynamically
def get_latest_model():
    model_files = sorted([f for f in os.listdir(MODEL_DIR) if f.endswith(".pth")])
    return os.path.join(MODEL_DIR, model_files[-1]) if model_files else None

latest_model_path = get_latest_model()

# Ensure we have a model
if latest_model_path is None:
    st.error("🚨 No trained model found! Please run `train.py` first.")
    st.stop()

# Load trained model
st.sidebar.success(f"✅ Loaded Model: {latest_model_path}")
model = AlphaZeroNN()
model.load_state_dict(torch.load(latest_model_path))
model.eval()

# ================================
# 🎮 Streamlit UI Setup
# ================================
st.markdown(
    "<h1 class='title' style='color: black;'>AlphaZero Tic-Tac-Toe</h1>",
    unsafe_allow_html=True
)

st.sidebar.header("⚙️ Settings")
play_as_x = st.sidebar.radio("Play as:", ["X (First)", "O (Second)"])
restart_game = st.sidebar.button("🔄 Restart Game")

# ================================
# 🎲 Initialize Game
# ================================
if "game" not in st.session_state or restart_game:
    st.session_state.game = TicTacToe()
    st.session_state.mcts = MCTS(model)
    st.session_state.human_turn = (play_as_x == "X")

# ================================
# 📌 Render Tic-Tac-Toe Board (Clickable X and O)
# ================================
def render_board():
    board = st.session_state.game.board
    symbols = {1: "❌", -1: "⭕", 0: "⬜"}  # Mapping for display
    
    st.markdown("<div class='board-container'>", unsafe_allow_html=True)
    cols = st.columns(3)  # Create a proper 3x3 grid

    for i in range(3):
        for j in range(3):
            with cols[j]:  # Ensure correct column placement
                if board[i, j] == 0:
                    # White button (⬜), click to make a move
                    if st.button("⬜", key=f"{i}-{j}"):
                        make_human_move(i, j)
                else:
                    # Instead of rendering text, make X and O buttons clickable
                    if board[i, j] == 1:
                        st.button("❌", key=f"x-{i}-{j}", disabled=False)
                    else:
                        st.button("⭕", key=f"o-{i}-{j}", disabled=False)
    
    st.markdown("</div>", unsafe_allow_html=True)

# ================================
# 🧑‍💻 Human Move Handler (Fixed)
# ================================
def make_human_move(i, j):
    if st.session_state.game.board[i, j] == 0 and st.session_state.human_turn:
        st.session_state.game.make_move((i, j))
        logger.info(f"👤 Human played: {(i, j)}")
        st.session_state.human_turn = False  # AI's turn next
        st.rerun()  # 🔄 Force Streamlit to refresh immediately

# ================================
# 🤖 AI Move Handler
# ================================
if not st.session_state.human_turn and st.session_state.game.check_winner() is None:
    ai_move = st.session_state.mcts.search(st.session_state.game)
    st.session_state.game.make_move(ai_move)
    logger.info(f"🤖 AI played: {ai_move}")
    st.session_state.human_turn = True  # Back to human

# ================================
# 🔄 Update Board & Check Winner
# ================================
render_board()
winner = st.session_state.game.check_winner()

# Add spacing and game status
if winner is None:
    st.markdown("<div class='game-status'>🤖 Machine is waiting for your turn...</div>", unsafe_allow_html=True)
elif winner == 1:
    st.success("🏆 X Wins!")
    logger.info("🏆 X Wins!")
elif winner == -1:
    st.success("🏆 O Wins!")
    logger.info("🏆 O Wins!")
else:
    st.warning("🤝 It's a Draw!")
    logger.info("🤝 It's a Draw!")