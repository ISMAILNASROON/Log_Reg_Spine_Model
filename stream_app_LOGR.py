import streamlit as st
import joblib

#Load Scalar object
model1 = joblib.load('scaler.pkl')

#Loas ML model
model2 = joblib.load('lr_model.pkl')

def mainf():       
    # front end elements of the web page 
    html_temp = """ 
    <div style ="background-color:blue;padding:10px"> 
    <h3 style ="color:black;text-align:center;">ML Logistic Regression - Spine Status App</h3> 
    </div> 
    """
    # display the front end aspect
    st.markdown(html_temp, unsafe_allow_html = True) 


    # following lines create boxes in which user can enter data required to make prediction 
    x1 = st.number_input("pelvic tilt")
    x2 = st.number_input("sacral_slope")
    x3 = st.text_input("pelvic_radius")
    x4 = st.number_input("degree_spondylolisthesis")
    x5 = st.number_input("pelvic_slope")
    x6 = st.number_input("Direct_tilt")
    x7 = st.text_input("thoracic_slope")
    x8 = st.number_input("cervical_tilt")
    x9 = st.number_input("sacrum_angle")
    x10 = st.number_input("scoliosis_slope")

    

    if st.button("Predict"):    
        scale_feature = model1.transform([[x1,x2,x3,x4,x5,x6,x7,x8,x9,x10]])

        result = model2.predict(scale_feature) 
        clean_result = prediction(result[0])
        html_temp = """ <h3 style ="color:black;text-align:left;">Output : </h3> """
        st.markdown(html_temp, unsafe_allow_html = True)
        
        st.success('Spine Status        :       {}'.format(clean_result))
        
        st.success('Spine Status        :       {}'.format( round(result[0],3)))
        
           
def prediction(result): 
    if result == 1:
        pred = 'ABNORMAL SPINE'
    else:
        pred = 'NORMAL SPINE'
    return pred

if __name__ == '__main__':
    mainf()