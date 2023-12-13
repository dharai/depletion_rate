import streamlit as st
import pandas as pd 
import datetime
import numpy as np
import plotly.graph_objs as go
import plotly.express as px
import database as db
import prediction_model as ml 
import math 

st.set_page_config(page_title='Laundris Depletion Rate', layout='wide') 


def color_depletion_table(val):
    if val == 'ragout': 
        color = '#ede579' 
    elif val == 'lost': 
        color = '#d66e69'
    else: 
        color = '#6bbf82'

    return f'background-color: {color}'  


def get_rounded_value(value): 
    return float('{:.2f}'.format(value)) 


def get_next_month(current_month):
    # Convert the month name to a datetime object
    current_date = datetime.datetime.strptime(current_month, '%B')

    # Calculate the first day of the next month
    first_day_of_next_month = current_date.replace(day=1) + datetime.timedelta(days=32)

    # Get the name of the next month
    next_month_name = first_day_of_next_month.strftime('%B')

    return next_month_name



def get_ragout_month(days): 
    current_date = datetime.datetime.now()  

    # Calculate the new date by adding the specified number of days
    new_date = current_date + datetime.timedelta(days=days)

    # Get the resulting month in words
    resulting_month = new_date.strftime("%B")

    return resulting_month

def main(): 
    st.markdown('<h1 style="color:#4B7CA7;font-size:32px;">Laundris Depletion Rate Analysis</h1>', unsafe_allow_html=True)  

    inventory_name_list, inventory_id_list = db.fetch_inventory_list() # fetch customer list 

    col1, col2, col3, col4 = st.columns((4, 5, 5, 3))
    selected_customer_name = col1.selectbox('Select Customer', inventory_name_list) 
    selected_inventory_id = inventory_id_list[inventory_name_list.index(selected_customer_name)] 

    df, order_cycle_df, inactive_status_items_df = db.fetch_data(selected_inventory_id) # fetch main data 
    total_number = order_cycle_df.shape[0]

    active_items_df = order_cycle_df[order_cycle_df.inactive_time <= 90]

    inactive_90_days_df = order_cycle_df[order_cycle_df.inactive_time > 90] 
    n_inactive_90_days = inactive_90_days_df.shape[0] 
    p_inactive_90_days = (n_inactive_90_days/total_number)*100 

    # Predict ragout [current state]
    active_items_df['customer_id'] = selected_inventory_id
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
    active_items_df['usage_period'] = today - active_items_df['creation_date']
    active_items_df['usage_period'] = active_items_df['usage_period'].dt.days 

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

    active_last_operation_group = active_items_df.last_operation.value_counts()
    active_last_operation_group = active_last_operation_group.reset_index()  
    active_last_operation_group.columns = ['Last Operation', 'Items Count']


    ### active items that are inactive for less than 90 days 
    features = ['rfid_id', 'customer_id', 'item_type_id', 'total_washes', 'pickup_count', 'dropoff_count', 'creation_date', 'birthday', 'last_updated_date']
    labeled_data = ml.predict_ragout_time_group(active_items_df[features], 30) 
    active_items_df = pd.merge(active_items_df, labeled_data, on='rfid_id', how='inner')  

    active_items_df.loc[(active_items_df['inactive_time'] < 30) & (active_items_df['last_operation'] == 'No order cycle'), 'predicted_ragout_time'] = 250 

    if n_normal > 0: 
        features = ['rfid_id', 'customer_id', 'item_type_id', 'total_washes', 'pickup_count', 'dropoff_count', 'creation_date', 'birthday', 'last_updated_date']
        labeled_data = ml.predict_ragout_time_group(normal_df[features], 0) 
        normal_df = pd.merge(normal_df, labeled_data, on='rfid_id', how='inner')  
        normal_df['ragout_month'] = normal_df.apply(lambda x: get_ragout_month (x['predicted_ragout_time']), axis=1)  


    active_items_df['ragout_month'] = active_items_df.apply(lambda x: get_ragout_month (x['predicted_ragout_time']), axis=1)  

    ## create par level heatmap s
    columns = ['rfid_id', 'item_type_name', 'side', 'last_operation', 'ragout_month', 'predicted_ragout_time']
     
    item_heatmap = order_cycle_df.item_type_name.value_counts()
    item_heatmap = item_heatmap.reset_index()  
    item_heatmap.columns = ['Item Type', 'Total Items Count']

    item_heatmap = item_heatmap.merge(ragout_group, on='Item Type', how='left') 
    item_heatmap.rename(columns={"Items Count": "Current Ragout Items"}, inplace=True)

    item_heatmap = item_heatmap.merge(lost_group, on='Item Type', how='left') 
    item_heatmap.rename(columns={"Items Count": "Current Lost Items"}, inplace=True)

    if n_normal > 0: 
        normal_items_df = pd.concat([normal_df[columns], active_items_df[columns]])
    else: 
        normal_items_df = active_items_df[columns]

    pickedup_items_df = normal_items_df[normal_items_df.side == 'facility']
    pickedup_items_group = pickedup_items_df.item_type_name.value_counts()
    pickedup_items_group = pickedup_items_group.reset_index()  
    pickedup_items_group.columns = ['Item Type', 'On Facility Items Count']

    item_heatmap = item_heatmap.merge(pickedup_items_group, on='Item Type', how='left') 
    
    # Get the current date and time
    current_date = datetime.datetime.now()
    # Get the current month in words
    current_month = current_date.strftime("%B")
    next_first_month = get_next_month(current_month)
    next_second_month =  get_next_month(next_first_month)

    current_month_df = normal_items_df[normal_items_df.ragout_month == current_month] 
    current_month_group = current_month_df.item_type_name.value_counts()
    current_month_group = current_month_group.reset_index()  
    current_month_group.columns = ['Item Type', f'Ragout on {current_month}']

    item_heatmap = item_heatmap.merge(current_month_group, on='Item Type', how='left')  

    item_heatmap.fillna(0, inplace=True) 
    item_heatmap['Current Available Items'] = item_heatmap['Total Items Count'] - item_heatmap['Current Lost Items'] - item_heatmap['Current Ragout Items'] - item_heatmap[f'Ragout on {current_month}']

    next_first_month_df = normal_items_df[normal_items_df.ragout_month == next_first_month] 
    next_second_month_df = normal_items_df[normal_items_df.ragout_month == next_second_month]

    next_first_month_group = next_first_month_df.item_type_name.value_counts()
    next_first_month_group = next_first_month_group.reset_index()  
    next_first_month_group.columns = ['Item Type', f'Ragout on {next_first_month}'] 

    next_second_month_group = next_second_month_df.item_type_name.value_counts()
    next_second_month_group = next_second_month_group.reset_index()  
    next_second_month_group.columns = ['Item Type', f'Ragout on {next_second_month}']   
    
    item_heatmap = item_heatmap.merge(next_first_month_group, on='Item Type', how='left')  
    item_heatmap.fillna(0, inplace=True)

    item_heatmap[f'Available on {next_first_month}'] = item_heatmap['Current Available Items'] - item_heatmap[f'Ragout on {next_first_month}']

    item_heatmap = item_heatmap.merge(next_second_month_group, on='Item Type', how='left')  
    item_heatmap.fillna(0, inplace=True)

    item_heatmap[f'Available on {next_second_month}'] = item_heatmap['Current Available Items'] - item_heatmap[f'Ragout on {next_first_month}'] - item_heatmap[f'Ragout on {next_second_month}']

    item_type_ids =  list(order_cycle_df.item_type_id.unique()) 

    room_profile_df = db.get_desired_quantity(item_type_ids, selected_inventory_id)
    item_heatmap = pd.merge(item_heatmap, room_profile_df, on='Item Type') 

    item_heatmap[f'{current_month}'] = item_heatmap['Current Available Items']/item_heatmap['Desired Quantity'] * 100
    item_heatmap[f'{next_first_month}'] = item_heatmap[f'Available on {next_first_month}']/item_heatmap['Desired Quantity'] * 100
    item_heatmap[f'{next_second_month}'] = item_heatmap[f'Available on {next_second_month}']/item_heatmap['Desired Quantity'] * 100 

    item_heatmap[f'{current_month}'] = item_heatmap[f'{current_month}'].apply(get_rounded_value)
    item_heatmap[f'{next_first_month}'] = item_heatmap[f'{next_first_month}'].apply(get_rounded_value)
    item_heatmap[f'{next_second_month}'] = item_heatmap[f'{next_second_month}'].apply(get_rounded_value)

    par_heatmap_data = item_heatmap[['Item Type', f'{current_month}', f'{next_first_month}', f'{next_second_month}']]

    # previous months data 
    current_date = pd.Timestamp(current_date, tz='UTC')
    last_first_month_start =  current_date - pd.Timedelta(days=60) 
    last_second_month_start =  current_date - pd.Timedelta(days=90)

    last_first_month_df = inactive_status_items_df[(inactive_status_items_df.ragout_date < current_date) & (inactive_status_items_df.ragout_date>last_first_month_start)]
    last_second_month_df = inactive_status_items_df[(inactive_status_items_df.ragout_date < last_first_month_start) & (inactive_status_items_df.ragout_date>last_second_month_start)] 

    st.dataframe(last_first_month_df)
    st.dataframe(last_second_month_df)

    par_heatmap_data.set_index("Item Type", inplace=True)
    custom_color_scale = ['#eb827f', '#FFFFFF','#FFFFFF','#FFFFFF','#FFFFFF','#FFFFFF','#FFFFFF'] 
    par_heatmap_fig = px.imshow(par_heatmap_data,  
                                labels=dict(x="Month", y="Item Type"), 
                                x=par_heatmap_data.columns, text_auto=True, color_continuous_scale=custom_color_scale, aspect="auto")   
    
    st.plotly_chart(par_heatmap_fig, use_container_width=True)  

    item_heatmap.rename(columns={"Current Available Items":f"Current Available Items ({current_month})", 
                                 f"{current_month}":f"Current Par level ({current_month})", 
                                 f"Ragout on {next_first_month}":f"{next_first_month} Ragout Items", 
                                 f"Ragout on {next_second_month}":f"{next_second_month} Ragout Items", 
                                 f"Available on {next_first_month}": f"Available Items ({next_first_month})", 
                                 f"Available on {next_second_month}": f"Available Items ({next_second_month})",  
                                 f"{next_first_month}": f"Par level ({next_first_month})",  
                                 f"{next_second_month}": f"Par level ({next_second_month})"}, inplace=True)

    expander = st.expander("📁 Detailed Table") 
    columns = ['Item Type', 'Current Ragout Items', 'Current Lost Items', f'Current Available Items ({current_month})', f'Current Par level ({current_month})', 
               f'{next_first_month} Ragout Items', f'Available Items ({next_first_month})', f'Par level ({next_first_month})',  
               f'{next_second_month} Ragout Items', f'Available Items ({next_second_month})', f'Par level ({next_second_month})', 'Desired Quantity']
    
    expander.dataframe(item_heatmap[columns], use_container_width=True, hide_index=True) 

    st.info("Number of items that are inactive for more than 90 days: **{:,} ({:.2f}%)**".format(n_inactive_90_days, p_inactive_90_days), icon='🛑') 
    st.info("Current depletion rate: **{:,} ({:.2f}%)**".format(n_ragout+n_lost, p_depletion*100), icon='🛑') 


    if n_normal > 0: 
        normal_df['predicted_ragout_time'] = normal_df['predicted_ragout_time'].astype(str) + ' days' 

    active_items_df['predicted_ragout_time'] = active_items_df['predicted_ragout_time'].astype(str) + ' days'  

    ### Heatmap -------------------------------------------------------------------
    st.plotly_chart(delation_heatmap_fig, use_container_width=True)  
    # Show main table 
    show_columns = ['Label', 'rfid_id', 'creation_date', 'birthday', 'last_scan_date', 'item_type_name',
                    'total_washes', 'pickup_count', 'dropoff_count', 'usage_period', 'last_operation', 'inactive_time', 'predicted_ragout'] 
    
    st.dataframe(inactive_90_days_df[show_columns].style.applymap(color_depletion_table, subset=['Label']), 
                   use_container_width=True, hide_index=True) 


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
    if n_normal > 0: 
        # Show main table 
        show_columns = ['Label', 'rfid_id', 'creation_date', 'birthday', 'last_scan_date', 'item_type_name',
                        'total_washes', 'pickup_count', 'dropoff_count', 'usage_period', 'last_operation', 'inactive_time', 'predicted_ragout', 'predicted_ragout_time', 'ragout_month']  
    
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


    # average lifetime of items 
    col1, col2 = expander.columns((4, 2))  

    col1.markdown('<h4 style="color:#4B7CA7;font-size:16px;">Average Lifetime of Items based on Ragout Items</h4>', unsafe_allow_html=True)
    lifetime_data = pd.read_csv("models/lifetime_items.csv")
    lifetime_data = lifetime_data[lifetime_data.customer_id == selected_inventory_id]
    
    df_item_type = db.fetch_item_type_names() 
    lifetime_data = lifetime_data.merge(df_item_type, on='item_type_id') 

    lifetime_group = lifetime_data[['item_type_name', 'lifetime']].groupby('item_type_name').mean()
    lifetime_group = lifetime_group.reset_index()  
    lifetime_group.columns = ['Item Type', 'Average Lifetime'] 
    lifetime_group['Average Lifetime'] = lifetime_group['Average Lifetime'].apply(math.ceil) 

    lifetime_group_fig = px.histogram(lifetime_group, y = 'Item Type', x="Average Lifetime")
    lifetime_group_fig.update_layout(bargap=0.2)
    lifetime_group_fig.update_traces(marker_color='#3c8ff3')
    col1.plotly_chart(lifetime_group_fig, use_container_width=True)



    ### active items that are inactive for less than 90 days
    st.markdown("***")
    st.markdown('<h3 style="color:#4B7CA7;font-size:20px;">Active Items Analysis</h3>', unsafe_allow_html=True)   

    st.info("Number of active items: **{:,}**".format(active_items_df.shape[0]), icon='🧺') 

    show_columns = ['rfid_id', 'creation_date', 'birthday', 'last_scan_date', 'item_type_name',
                        'total_washes', 'pickup_count', 'dropoff_count', 'usage_period', 'last_operation', 'inactive_time', 'predicted_ragout_time']   

    expander = st.expander("📁 Detailed Analysis") 
    expander.dataframe(active_items_df[show_columns], use_container_width=True, hide_index=True)  

    col1, col2, col3 = expander.columns((5,1,5)) 
    col1.markdown('<h4 style="color:#4B7CA7;font-size:16px;">Last Operation</h4>', unsafe_allow_html=True)
    active_last_operation_fig = px.bar(active_last_operation_group, x='Items Count', y='Last Operation')  
    active_last_operation_fig.update_traces(marker_color='#3c8ff3')
    active_last_operation_fig.update_traces(width=0.5)
    col1.plotly_chart(active_last_operation_fig, use_container_width=True) 

    col3.markdown('<h4 style="color:#4B7CA7;font-size:16px;">Inactive Time Distribution</h4>', unsafe_allow_html=True)  
    active_inactive_distribution_fig = px.histogram(active_items_df, x="inactive_time")
    active_inactive_distribution_fig.update_layout(bargap=0.2)
    active_inactive_distribution_fig.update_traces(marker_color='#3c8ff3')
    col3.plotly_chart(active_inactive_distribution_fig, use_container_width=True)


    col1.markdown('<h4 style="color:#4B7CA7;font-size:16px;">Usage Period Distribution</h4>', unsafe_allow_html=True)  
    active_usage_period_distribution_fig = px.histogram(active_items_df, x="usage_period")
    active_usage_period_distribution_fig.update_layout(bargap=0.2)
    active_usage_period_distribution_fig.update_traces(marker_color='#3c8ff3')
    col1.plotly_chart(active_usage_period_distribution_fig, use_container_width=True)


if __name__ == '__main__':
    main() 