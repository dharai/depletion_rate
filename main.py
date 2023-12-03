import streamlit as st
import pandas as pd 
import datetime
import numpy as np
import plotly.graph_objs as go
import plotly.express as px
import database as db
import prediction_model as ml 

st.set_page_config(page_title='Laundris Depletion Rate', layout='wide') 


def color_depletion_table(val):
    if val == 1: 
        color = 'green' 
    elif val == 0: 
        color = 'white'
    else: 
        color = '#e95463'

    return f'background-color: {color}'  


def main(): 
    st.markdown('<h1 style="color:#4B7CA7;font-size:32px;">Laundris Depletion Rate Analysis</h1>', unsafe_allow_html=True)  

    inventory_name_list, inventory_id_list = db.fetch_inventory_list() # fetch customer list 

    col1, col2, col3, col4 = st.columns((4, 5, 5, 3))
    selected_customer_name = col1.selectbox('Select Customer', inventory_name_list) 
    selected_inventory_id = inventory_id_list[inventory_name_list.index(selected_customer_name)] 

    df, order_cycle_df = db.fetch_data(selected_inventory_id) # fetch main data 

    inactive_90_days_df = order_cycle_df[order_cycle_df.inactive_time > 90] 
    n_inactive_90_days = inactive_90_days_df.shape[0] 

    st.info("Number of items that are inactive for more than 90 days: **{:,}**".format(n_inactive_90_days), icon='🛑') 

    # Predict ragout [current state]
    inactive_90_days_df['customer_id'] = selected_inventory_id

    features = ['rfid_id', 'customer_id', 'item_type_id', 'total_washes', 'pickup_count', 'dropoff_count', 'creation_date', 'birthday', 'last_updated_date']
    labeled_data = ml.predict_ragout_group(inactive_90_days_df[features]) 
    inactive_90_days_df = pd.merge(inactive_90_days_df, labeled_data, on='rfid_id', how='inner') 
    
    inactive_90_days_df['label'] = 'ragout'

    # Label items as loss 
    # inactive_90_days_df.loc[inactive_90_days_df[''] condition, 'label'] =   


    # Prepare to show 
    inactive_90_days_df['creation_date'] = inactive_90_days_df['creation_date'].dt.date
    inactive_90_days_df['last_updated_date'] = inactive_90_days_df['last_updated_date'].dt.date 
    inactive_90_days_df['birthday'] = inactive_90_days_df['birthday'].dt.date  

    # Show main table 
    st.data_editor(inactive_90_days_df.style.applymap(color_depletion_table, subset=['prediction']), 
                   use_container_width=True, hide_index=True) 


if __name__ == '__main__':
    main() 