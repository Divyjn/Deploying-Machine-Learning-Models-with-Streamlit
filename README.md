# 🍷 Wine Class Predictor

🎉 **A modern, interactive Streamlit web app that predicts wine classes using a machine learning model trained on the Wine dataset. Input features, get instant predictions with confidence, and explore feature importances—all in a modern, emoji-rich UI!** 🍇🤖

---

## 🚀 Features
- 🖱️ **User-friendly UI** with sliders or number inputs
- 🍇 **Wine dataset** (from sklearn)
- 📊 **Prediction probabilities** and confidence
- 🌟 **Feature importances** visualization
- 🧭 **Sidebar** with dataset info and controls
- 💎 **Responsive layout** with emojis and modern Streamlit widgets

---

## 🛠️ Setup

1. 🌀 **Clone this repo**
   ```bash
   git clone <your-repo-url>
   cd Deploying-Machine-Learning-Models-with-Streamlit
   ```
2. 📦 **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
3. 🏗️ **Train the model** (creates `wine_model.pkl`)
   ```bash
   python model_train.py
   ```
4. 🚦 **Run the app**
   ```bash
   streamlit run app.py
   ```

---

## 🖥️ Screenshots

Add screenshots of your app in the `images/` folder and reference them below:

```markdown
![Wine Class Predictor UI](images/screenshot1.png)
![Prediction Example](images/screenshot2.png)
```

Example:

![Wine Class Predictor UI](images/screenshot1.png)

---

## 📦 Files
- `app.py` — Streamlit web app
- `model_train.py` — Model training script
- `wine_model.pkl` — Trained model (auto-generated)
- `requirements.txt` — Python dependencies

---

## 🤝 Contributing
Pull requests welcome! For major changes, open an issue first. 🙌

---

## 📄 License
MIT 📝

---

_Made with Streamlit, sklearn, and ❤️_ 🍷 