import streamlit as st
import joblib 
import numpy as np

model=joblib.load('house_price_model.pkl')

st.title("🏠 House Price Prediction App Made by Pujara Ridham")
features=model.feature_names_in_

st.set_page_config(page_title="House Price Predictor")
st.write("Enter house details below:""Enter house details below:")
input_values=[]
feature_info = {
    "crime_rate": {
        "label": "Crime Rate",
        "desc": "વિસ્તારનો ગુનાનો દર (Index value, ઓછું હોય તો સારું)"
    },
    "resid_area": {
        "label": "Residential Area Index",
        "desc": "રહેણાંક વિસ્તારનું પ્રમાણ (Index)"
    },
    "air_qual": {
        "label": "Air Quality Index",
        "desc": "હવાની ગુણવત્તાનો સૂચક (Index, ઓછું = સારી હવા)"
    },
    "room_num": {
        "label": "Number of Rooms",
        "desc": "ઘરમાં કુલ રૂમોની સંખ્યા (Count)"
    },
    "age": {
        "label": "House Age",
        "desc": "ઘર કેટલા વર્ષ જૂનું છે (Years)"
    },
    "dist1": {
        "label": "Distance Index 1",
        "desc": "મુખ્ય સ્થળથી અંતર (Index, ઓછું = નજીક)"
    },
    "dist2": {
        "label": "Distance Index 2",
        "desc": "સુવિધાથી અંતર (Index)"
    },
    "dist3": {
        "label": "Distance Index 3",
        "desc": "કાર્યક્ષેત્રથી અંતર (Index)"
    },
    "dist4": {
        "label": "Distance Index 4",
        "desc": "ટ્રાન્સપોર્ટ હબથી અંતર (Index)"
    },
    "teachers": {
        "label": "Teachers Nearby",
        "desc": "શિક્ષણ સુવિધાનો સૂચક (Index / Count)"
    },
    "poor_prop": {
        "label": "Low Income Population %",
        "desc": "ઓછી આવકવાળા લોકોનું પ્રમાણ (%)"
    },
    "airport": {
        "label": "Airport Nearby",
        "desc": "એરપોર્ટ નજીક છે? (1 = હા, 0 = ના)"
    },
    "n_hos_beds": {
        "label": "Hospital Beds Nearby",
        "desc": "નજીકના હોસ્પિટલ બેડ્સ (Count)"
    },
    "n_hot_rooms": {
        "label": "Hotel Rooms Nearby",
        "desc": "નજીકના હોટેલ રૂમ્સ (Count)"
    },
    "rainfall": {
        "label": "Rainfall Index",
        "desc": "વર્ષાનો સૂચક (Index, cm/mm નથી)"
    },
    "bus_ter": {
        "label": "Bus Terminal Nearby",
        "desc": "બસ ટર્મિનલ નજીક છે? (1 = હા, 0 = ના)"
    },
    "parks": {
        "label": "Parks Nearby",
        "desc": "નજીકના પાર્ક્સ (Index)"
    },
    "waterbody_Lake and River": {
        "label": "Lake and River Nearby",
        "desc": "તળાવ અને નદી નજીક (1 = હા, 0 = ના)"
    },
    "waterbody_River": {
        "label": "River Nearby",
        "desc": "માત્ર નદી નજીક (1 = હા, 0 = ના)"
    },
    "waterbody_Unknown": {
        "label": "Waterbody Info Unknown",
        "desc": "પાણી સંબંધિત માહિતી ઉપલબ્ધ નથી"
    }
}
for feature in features:
    info = feature_info.get(feature, {"label": feature, "desc": ""})

    val = st.number_input(
        f"{info['label']} ({info['desc']})",
        value=0.0,
        key=f"input_{feature}"
    )
    input_values.append(val)
if st.button("Predict Price"):
    input_array=np.array(input_values).reshape(1, -1)
    prediction=model.predict(input_array)
    st.success(f"💰 Predicted House Price: {prediction[0]:.2f}lacs")