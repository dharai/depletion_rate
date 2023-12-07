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

    ragout_df = inactive_90_days_df[inactive_90_days_df.Label == 'ragout']
    normal_df = inactive_90_days_df[inactive_90_days_df.Label == 'normal']
    lost_df   = inactive_90_days_df[inactive_90_days_df.Label == 'lost']

    n_ragout = ragout_df.shape[0] 
    n_normal = normal_df.shape[0] 
    n_lost   = lost_df.shape[0]

    p_depletion = (n_ragout + n_lost)/total_number

    st.info("Current depletion rate: **{:,} ({:.2f}%)**".format(n_ragout+n_lost, p_depletion*100), icon='🛑') 


    # Heatmap 
    depletion_grouped_data = inactive_90_days_df.groupby(['item_type_name', 'Label']).size().reset_index(name='count') 
    heatmap_pivot_table = depletion_grouped_data.pivot(index='item_type_name', columns='Label', values='count')
    heatmap_pivot_table.fillna(0, inplace=True)
    if n_ragout == 0: 
        heatmap_pivot_table['ragout'] = 0 
    elif n_lost == 0: 
        heatmap_pivot_table['lost'] = 0 
    elif n_normal == 0: 
        heatmap_pivot_table['normal'] = 0 
    else: 
        pass  

    heatmap_pivot_table.rename(columns={"lost":"Lost", "normal":"Normal","ragout":"Ragout"}, inplace=True)
    cols = ['Lost', 'Ragout', 'Normal'] 
    heatmap_pivot_table = heatmap_pivot_table[cols] 

    custom_color_scale = ['#FFFFFF', '#eb827f'] 
    delation_heatmap_fig = px.imshow(heatmap_pivot_table,  
                                labels=dict(x="Category", y="Item Type"), 
                                x=heatmap_pivot_table.columns, text_auto=True, color_continuous_scale=custom_color_scale, aspect="auto")  

    st.plotly_chart(delation_heatmap_fig, use_container_width=True) 

    # Show main table 
    show_columns = ['Label', 'rfid_id', 'creation_date', 'birthday', 'last_scan_date', 'item_type_name',
                    'total_washes', 'pickup_count', 'dropoff_count', 'usage_period', 'last_operation', 'inactive_time', 'predicted_ragout'] 
    
    st.dataframe(inactive_90_days_df[show_columns].style.applymap(color_depletion_table, subset=['Label']), 
                   use_container_width=True, hide_index=True) 

    ragout_group = ragout_df.item_type_name.value_counts()
    ragout_group = ragout_group.reset_index()  
    ragout_group.columns = ['Item Type', 'Items Count']
    ragout_last_operation_group = ragout_df.last_operation.value_counts()
    ragout_last_operation_group = ragout_last_operation_group.reset_index()  
    ragout_last_operation_group.columns = ['Last Operation', 'Items Count'] 


    normal_group = normal_df.item_type_name.value_counts()
    normal_group = normal_group.reset_index()  
    normal_group.columns = ['Item Type', 'Items Count']
    normal_last_operation_group = normal_df.last_operation.value_counts()
    normal_last_operation_group = normal_last_operation_group.reset_index()  
    normal_last_operation_group.columns = ['Last Operation', 'Items Count']

    lost_group = lost_df.item_type_name.value_counts()
    lost_group = lost_group.reset_index()  
    lost_group.columns = ['Item Type', 'Items Count']
    lost_last_operation_group = lost_df.last_operation.value_counts()
    lost_last_operation_group = lost_last_operation_group.reset_index()  
    lost_last_operation_group.columns = ['Last Operation', 'Items Count']

    lost_location_group = lost_df.location_type.value_counts().reset_index() 
    lost_location_group.columns = ['Location', 'Count'] 


    st.info("Number of lost items: **{:,}**".format(n_lost), icon='🔎')  
    expander = st.expander("📁 Detailed Analysis") 
    expander.dataframe(lost_df[show_columns].style.applymap(color_depletion_table, subset=['Label']), 
                   use_container_width=True, hide_index=True) 
    # plotly 
    location_pie_fig = px.pie(lost_location_group, values='Count', names='Location')
    lost_group_bar = px.bar(lost_group, y='Items Count', x='Item Type')  
    lost_group_bar.update_traces(marker_color='#3c8ff3') 
    # Display the chart in Streamlit
    col1, col2 = expander.columns((6, 4))  
    col1.markdown('<h4 style="color:#4B7CA7;font-size:16px;">Number of Items by Item Type</h4>', unsafe_allow_html=True) 
    col1.plotly_chart(lost_group_bar, use_container_width=True) 
    col2.markdown('<h4 style="color:#4B7CA7;font-size:16px;">Last Seen Location</h4>', unsafe_allow_html=True)
    col2.plotly_chart(location_pie_fig, use_container_width=True) 

    col1, col2, col3 = expander.columns((4,1,5)) 
    col1.markdown('<h4 style="color:#4B7CA7;font-size:16px;">Last Operation</h4>', unsafe_allow_html=True)
    lost_last_operation_fig = px.bar(lost_last_operation_group, x='Items Count', y='Last Operation')  
    lost_last_operation_fig.update_traces(marker_color='#3c8ff3')
    lost_last_operation_fig.update_traces(width=0.5)
    col1.plotly_chart(lost_last_operation_fig, use_container_width=True) 

    col3.markdown('<h4 style="color:#4B7CA7;font-size:16px;">Inactive Time Distribution</h4>', unsafe_allow_html=True)  
    lost_inactive_distribution_fig = px.histogram(lost_df, x="inactive_time")
    lost_inactive_distribution_fig.update_layout(bargap=0.2)
    lost_inactive_distribution_fig.update_traces(marker_color='#3c8ff3')
    col3.plotly_chart(lost_inactive_distribution_fig, use_container_width=True)


    
    st.info("Number of ragout items: **{:,}**".format(n_ragout), icon='📦') 
    expander = st.expander("📁 Detailed Analysis") 
    expander.dataframe(ragout_df[show_columns].style.applymap(color_depletion_table, subset=['Label']), 
                   use_container_width=True, hide_index=True) 
    # plotly 
    ragout_group_bar = px.bar(ragout_group, y='Items Count', x='Item Type')  
    ragout_group_bar.update_traces(marker_color='#3c8ff3') 
    # Display the chart in Streamlit
    expander.plotly_chart(ragout_group_bar, use_container_width=True) 

    col1, col2, col3 = expander.columns((4,1,5)) 
    col1.markdown('<h4 style="color:#4B7CA7;font-size:16px;">Last Operation</h4>', unsafe_allow_html=True)
    ragout_last_operation_fig = px.bar(ragout_last_operation_group, x='Items Count', y='Last Operation')  
    ragout_last_operation_fig.update_traces(marker_color='#3c8ff3')
    ragout_last_operation_fig.update_traces(width=0.5)
    col1.plotly_chart(ragout_last_operation_fig, use_container_width=True) 

    col3.markdown('<h4 style="color:#4B7CA7;font-size:16px;">Inactive Time Distribution</h4>', unsafe_allow_html=True)  
    ragout_inactive_distribution_fig = px.histogram(ragout_df, x="inactive_time")
    ragout_inactive_distribution_fig.update_layout(bargap=0.2)
    ragout_inactive_distribution_fig.update_traces(marker_color='#3c8ff3')
    col3.plotly_chart(ragout_inactive_distribution_fig, use_container_width=True)


    st.info("Number of normal items: **{:,}**".format(n_normal), icon='🧺')  

    # Predict future ragout time  
    features = ['rfid_id', 'customer_id', 'item_type_id', 'total_washes', 'pickup_count', 'dropoff_count', 'creation_date', 'birthday', 'last_updated_date']
    labeled_data = ml.predict_ragout_time_group(normal_df[features]) 
    normal_df = pd.merge(normal_df, labeled_data, on='rfid_id', how='inner')   


    # Show main table 
    show_columns = ['Label', 'rfid_id', 'creation_date', 'birthday', 'last_scan_date', 'item_type_name',
                    'total_washes', 'pickup_count', 'dropoff_count', 'usage_period', 'last_operation', 'inactive_time', 'predicted_ragout', 'predicted_ragout_time']  
    
    expander = st.expander("📁 Detailed Analysis") 
    expander.dataframe(normal_df[show_columns].style.applymap(color_depletion_table, subset=['Label']), 
                   use_container_width=True, hide_index=True) 
    # plotly 
    normal_group_bar = px.bar(normal_group, y='Items Count', x='Item Type')
    normal_group_bar.update_traces(marker_color='#3c8ff3')  
    # Display the chart in Streamlit
    expander.plotly_chart(normal_group_bar, use_container_width=True)

    col1, col2, col3 = expander.columns((4,1,5)) 
    col1.markdown('<h4 style="color:#4B7CA7;font-size:16px;">Last Operation</h4>', unsafe_allow_html=True)
    normal_last_operation_fig = px.bar(normal_last_operation_group, x='Items Count', y='Last Operation')  
    normal_last_operation_fig.update_traces(marker_color='#3c8ff3')
    normal_last_operation_fig.update_traces(width=0.5)
    col1.plotly_chart(normal_last_operation_fig, use_container_width=True) 

    col3.markdown('<h4 style="color:#4B7CA7;font-size:16px;">Inactive Time Distribution</h4>', unsafe_allow_html=True)  
    normal_inactive_distribution_fig = px.histogram(normal_df, x="inactive_time")
    normal_inactive_distribution_fig.update_layout(bargap=0.2)
    normal_inactive_distribution_fig.update_traces(marker_color='#3c8ff3')
    col3.plotly_chart(normal_inactive_distribution_fig, use_container_width=True)



if __name__ == '__main__':
    main() 