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
    if val == 'ragout': 
        color = '#ede579' 
    elif val == 'lost': 
        color = '#d66e69'
    else: 
        color = '#6bbf82'

    return f'background-color: {color}'  


def main(): 
    st.markdown('<h1 style="color:#4B7CA7;font-size:32px;">Laundris Depletion Rate Analysis</h1>', unsafe_allow_html=True)  

    inventory_name_list, inventory_id_list = db.fetch_inventory_list() # fetch customer list 

    col1, col2, col3, col4 = st.columns((4, 5, 5, 3))
    selected_customer_name = col1.selectbox('Select Customer', inventory_name_list) 
    selected_inventory_id = inventory_id_list[inventory_name_list.index(selected_customer_name)] 

    df, order_cycle_df = db.fetch_data(selected_inventory_id) # fetch main data 
    total_number = order_cycle_df.shape[0]

    inactive_90_days_df = order_cycle_df[order_cycle_df.inactive_time > 90] 
    n_inactive_90_days = inactive_90_days_df.shape[0] 
    p_inactive_90_days = (n_inactive_90_days/total_number)*100 

    st.info("Number of items that are inactive for more than 90 days: **{:,} ({:.2f}%)**".format(n_inactive_90_days, p_inactive_90_days), icon='🛑') 

    # Predict ragout [current state]
    inactive_90_days_df['customer_id'] = selected_inventory_id

    features = ['rfid_id', 'customer_id', 'item_type_id', 'total_washes', 'pickup_count', 'dropoff_count', 'creation_date', 'birthday', 'last_updated_date']
    labeled_data = ml.predict_ragout_group(inactive_90_days_df[features]) 
    inactive_90_days_df = pd.merge(inactive_90_days_df, labeled_data, on='rfid_id', how='inner') 
    
    inactive_90_days_df['Label'] = 'normal'
    inactive_90_days_df.loc[inactive_90_days_df['prediction'] == 1, 'Label'] =  'ragout'

    # Label items as loss 
    inactive_90_days_df.loc[(inactive_90_days_df['pickup_count'] <= 1) & (inactive_90_days_df['dropoff_count'] <= 1) 
                            & (inactive_90_days_df['prediction'] == 0), 'Label'] =  'lost'


    # Prepare to show 
    today = pd.Timestamp(datetime.date.today(), tz='UTC')  
    inactive_90_days_df['usage_period'] = today - inactive_90_days_df['creation_date']
    inactive_90_days_df['usage_period'] = inactive_90_days_df['usage_period'].dt.days 

    inactive_90_days_df['creation_date'] = inactive_90_days_df['creation_date'].dt.date
    inactive_90_days_df['last_updated_date'] = inactive_90_days_df['last_updated_date'].dt.date 
    inactive_90_days_df['birthday'] = inactive_90_days_df['birthday'].dt.date  


    # Show main table 
    show_columns = ['Label', 'rfid_id', 'creation_date', 'birthday', 'last_scan_date', 'item_type_name',
                    'total_washes', 'pickup_count', 'dropoff_count', 'usage_period', 'last_operation', 'inactive_time', 'predicted_ragout'] 
    
    st.dataframe(inactive_90_days_df[show_columns].style.applymap(color_depletion_table, subset=['Label']), 
                   use_container_width=True, hide_index=True) 
    

    n_ragout = inactive_90_days_df[inactive_90_days_df.Label == 'ragout'].shape[0] 
    n_normal = inactive_90_days_df[inactive_90_days_df.Label == 'normal'].shape[0] 
    n_lost   = inactive_90_days_df[inactive_90_days_df.Label == 'lost'].shape[0]

    st.info("Number of lost items: **{:,}**".format(n_lost), icon='🔎')  
    st.info("Number of ragout items: **{:,}**".format(n_ragout), icon='📦')  
    st.info("Number of normal items: **{:,}**".format(n_normal), icon='🧺')  


if __name__ == '__main__':
    main() 