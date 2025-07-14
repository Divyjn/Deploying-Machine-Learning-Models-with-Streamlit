import streamlit as st
import pickle
import numpy as np
from sklearn.datasets import load_wine

# load model
def load_model():
    with open('wine_model.pkl', 'rb') as f:
        return pickle.load(f)
mdl = load_model()

# wine data for feature names
wine = load_wine()

# Sidebar for controls and info
st.sidebar.title('🍷 Wine Classifier Controls')
st.sidebar.markdown('---')
st.sidebar.header('📚 Dataset Info')
st.sidebar.write(f"**Classes:** {', '.join(wine.target_names)}")
st.sidebar.write(f"**Features:** {len(wine.feature_names)}")
st.sidebar.write(f"**Samples:** {wine.data.shape[0]}")
input_method = st.sidebar.radio('Input method:', ['Sliders', 'Number Input'])
show_importance = st.sidebar.checkbox('Show Feature Importances', value=True)

st.title('Wine Class Predictor 🍇')
st.caption('Predict wine class and explore model insights!')

# input section
vals = []
st.subheader('🔢 Input Features')
cols = st.columns(3)
for i, n in enumerate(wine.feature_names):
    col = cols[i % 3]
    if input_method == 'Sliders':
        v = col.slider(n, float(np.min(wine.data[:,i])), float(np.max(wine.data[:,i])), float(np.mean(wine.data[:,i])))
    else:
        v = col.number_input(n, float(np.min(wine.data[:,i])), float(np.max(wine.data[:,i])), float(np.mean(wine.data[:,i])))
    vals.append(v)

# show input summary
with st.expander('See your input summary 📝'):
    st.info(f'Your input: {dict(zip(wine.feature_names, vals))}')

# predict
if st.button('Predict 🚀'):
    pred = mdl.predict([vals])[0]
    probs = mdl.predict_proba([vals])[0]
    st.success(f'Prediction: {wine.target_names[pred]} 🍷')
    st.metric('Predicted Class', wine.target_names[pred])
    st.progress(float(np.max(probs)), text=f"Confidence: {np.max(probs):.2%}")
    st.write('Class probabilities:')
    st.table({name: [f'{p:.2%}'] for name, p in zip(wine.target_names, probs)})
    if show_importance:
        st.subheader('📊 Feature Importances')
        imp = mdl.feature_importances_
        imp_dict = {n: imp[i] for i, n in enumerate(wine.feature_names)}
        st.bar_chart(imp)
        st.caption('Feature importances above')
        st.write({k: f'{v:.2%}' for k, v in sorted(imp_dict.items(), key=lambda x: -x[1])[:5]})
    st.balloons()

st.markdown('---')
st.caption('Made with Streamlit, sklearn, and ❤️') 