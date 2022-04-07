# źródło danych [https://www.kaggle.com/c/titanic/](https://www.kaggle.com/c/titanic)


'''
1.
#Create the sex_d variable and enter the appropriate labels for women and men (similarly as it was done in the case of pclass_d and embarked_d variables).

2.
 #Enter the title of application.

3.
 #Change the photo.

4.
 #Enter (in the left column) the new pclass_radio variable with three labels (first, second and third class).

5.
 In the right column there are variables with information about age, number of family members, etc. Check in the original dataset (what minimum and maximum values can be entered by the user) and change the min_value and max_value parameters.

6.
 Create an account on GitHub and Share Streamlit. Create a new repository in GitHub, where you will put the corrected app1.py, the requirements file (.txt) and the trained model. Use Share Streamlit to create an application.
'''

import streamlit as st
import pickle
from datetime import datetime
startTime = datetime.now()
# import znanych nam bibliotek

filename = "model.sv"
model = pickle.load(open(filename,'rb'))
# otwieramy wcześniej wytrenowany model

pclass_d = {0:"First",1:"Second", 2:"Third"}
embarked_d = {0:"Cherbourg", 1:"Queenstown", 2:"Southampton"}
sex_d = {0:"Female",1:"Male"}
# o ile wcześniej kodowaliśmy nasze zmienne, to teraz wprowadzamy etykiety z ich nazewnictwem

def main():
    st.set_page_config(page_title="Welcome to SinkDet!")
    overview = st.container()
    left, right = st.columns(2)
    prediction = st.container()

    st.image("https://lh5.googleusercontent.com/proxy/SnftXXyajiTbZ2c_N9YSkTKrgOAJ2yGFxnnwMcpe3wr92qFajkHsQIgXxfQVWhr5b4qSZKjKg-BznzpemRUeDik7zeo6kTu92xtTV2lTBbnIlFEmHBwf1sUgnofwOJAXrMooYorKlc6xHYNPldb_eIi3haNpZCLuEzEBho-bjfyKeUiTd_ZIEQ=s0-d")

    with overview:
        st.title("Would you sink? Find out today!")

    with left:
        pclass_radio = st.radio( "Class", list(pclass_d.keys()), format_func=lambda x : pclass_d[x])
        sex_radio = st.radio( "Sex", list(sex_d.keys()), format_func=lambda x : sex_d[x])
        embarked_radio = st.radio( "Harbour embarked", list(embarked_d.keys()), index=2, format_func= lambda x: embarked_d[x] )

    with right:
        age_slider = st.slider("Age", value=25.0, min_value=0.17, max_value=76.00)
        sibsp_slider = st.slider("Number of siblings  and/or partner", min_value=0, max_value=8)
        parch_slider = st.slider("Number of parents and/or kids", min_value=0, max_value=9)
        fare_slider = st.slider("Price of ticket", min_value=3.1708, max_value=512.3292, step=0.01)

    data = [[pclass_radio, sex_radio,  age_slider, sibsp_slider, parch_slider, fare_slider, embarked_radio]]
    survival = model.predict(data)
    s_confidence = model.predict_proba(data)

    with prediction:
        st.subheader("Would person like that survive?")
        st.subheader(("Yes" if survival[0] == 1 else "No"))
        st.write("Prediction confidence {0:.2f} %".format(s_confidence[0][survival][0] * 100))

if __name__ == "__main__":
    main()
