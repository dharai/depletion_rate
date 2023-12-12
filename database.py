import streamlit as st 
import datetime
import pandas as pd 
import pandas.io.sql as psql 

import psycopg2 

def connect(): 
    conn = psycopg2.connect(
        host="laundris-db.postgres.database.azure.com",
        database="laundris",
        user="postgres",
        password="Laundris123")

    return conn 



@st.cache_data
def fetch_inventory_list(): 
    conn = connect() 
    command = "select customer_id, customer_name from customer where status='active' and customer_type='internal' and entity_type='hotel'"  
    
    df = pd.read_sql(command, conn) 
    inventory_list = df.customer_name.to_list()
    inventory_ids  = df.customer_id.to_list()

    return inventory_list, inventory_ids  


@st.cache_data 
def fetch_item_type_names(): 
    conn = connect() 
    command = "select id as item_type_id, customer_item_type_name as item_type_name from customer_customerinventoryitemtype"

    df = pd.read_sql(command, conn) 
    return df 


def get_desired_quantity(item_type_ids, customer_id):
    conn = connect()  
    # Convert the list to a string of comma-separated values
    item_type_ids_str = ', '.join(map(str, item_type_ids)) 

    if customer_id == 45: 
        room_profile_sql = """
        select r.customer_room_type_id, r.customer_item_type_id as item_type_id, r.item_quantity, h.quantity
            from customer_roomprofile r
            join customer_hotelprofile h on r.customer_room_type_id = h.customer_room_type_id
            where r.customer_item_type_id in ({})
            and h.customer_id in (37, 38)
        """.format(item_type_ids_str)
    else: 
        room_profile_sql = """
        select r.customer_room_type_id, r.customer_item_type_id as item_type_id, r.item_quantity, h.quantity
            from customer_roomprofile r
            join customer_hotelprofile h on r.customer_room_type_id = h.customer_room_type_id
            where r.customer_item_type_id in ({})
            and h.customer_id = {}
        """.format(item_type_ids_str, customer_id)



    par_level_sql = """
    select id as item_type_id, ideal_par_level, customer_item_type_name 
        from customer_customerinventoryitemtype
        where id in ({})
    """.format(item_type_ids_str)

    customer_par_level_sql = f"select par_level from customer where customer_id = {customer_id}"

    room_profile_df = pd.read_sql(room_profile_sql, conn) 
    par_level_df    = pd.read_sql(par_level_sql, conn) 
    customer_par_level = pd.read_sql(customer_par_level_sql, conn) 
    customer_par_level = customer_par_level['par_level'].iloc[0] 

    room_profile_df = pd.merge(room_profile_df, par_level_df, on='item_type_id') 

    room_profile_df['ideal_par_level'].fillna(customer_par_level, inplace=True) 

    room_profile_df['Desired Quantity'] = room_profile_df['item_quantity'] * room_profile_df['quantity'] * room_profile_df['ideal_par_level']

    par_level_group = room_profile_df[['customer_item_type_name', 'Desired Quantity']].groupby("customer_item_type_name").sum()
    par_level_group = par_level_group.reset_index()  
    par_level_group.columns = ['Item Type', 'Desired Quantity'] 
    
    return par_level_group




@st.cache_data
def fetch_data(customer_id):
    rfid_sql = f"""SELECT r.rfid_id, r.creation_date, r.last_updated_date, r.status, r.ragout_date, r.total_washes, r.last_scan_date, r.item_type_id, 
                  r.last_seen_location_id, r.location_id, r.birthday FROM rfid r WHERE r.customer_id = {customer_id};""" 

    order_binrfids_rfids_sql = "SELECT binrfids_id, rfid_id FROM order_binrfids_rfids;" 
    order_order_pickup_bins_sql = "SELECT binrfids_id, order_id FROM order_order_pickup_bins;" 
    order_order_dropoff_bins_sql = "SELECT binrfids_id, order_id FROM order_order_dropoff_bins;"

    item_type_sql = """SELECT id AS item_type_id, customer_item_type_name AS item_type_name 
                       FROM customer_customerinventoryitemtype;"""
    
    order_sql = """SELECT id AS order_id, actual_pickup_date, actual_dropoff_date, incoming_total_weight
                   FROM order_order"""
    
    location_sql = "SELECT id AS last_seen_location_id, name AS last_seen_location_name, location_type, customer_id as location_customer_id, side FROM inventory_location"
    
    today = pd.Timestamp(datetime.date.today(), tz='UTC')

    conn = connect()  

    rfid_df = pd.read_sql(rfid_sql, conn) 
    order_binrfids_rfids_df = pd.read_sql(order_binrfids_rfids_sql, conn) 
    order_order_pickup_bins_df = pd.read_sql(order_order_pickup_bins_sql, conn) 
    order_order_dropoff_bins_df = pd.read_sql(order_order_dropoff_bins_sql, conn)

    df = pd.merge(rfid_df, order_binrfids_rfids_df, on='rfid_id', how='left') 
    df = pd.merge(df, order_order_pickup_bins_df, on='binrfids_id', how='left')
    df.rename(columns={"order_id": "pickup_order_id"}, inplace=True)
    df = pd.merge(df, order_order_dropoff_bins_df, on='binrfids_id', how='left')
    df.rename(columns={"order_id": "dropoff_order_id"}, inplace=True)


    item_type_df = pd.read_sql(item_type_sql, conn) 
    location_df = pd.read_sql(location_sql, conn)
    pickup_order_df = pd.read_sql(order_sql, conn) 
    dropoff_order_df = pd.read_sql(order_sql, conn)
    pickup_order_df.rename(columns = {"order_id":"pickup_order_id", "actual_pickup_date":"last_pickup_date", "incoming_total_weight":"pickup_weight"}, inplace=True) 
    dropoff_order_df.rename(columns = {"order_id":"dropoff_order_id", "actual_dropoff_date":"last_dropoff_date", "incoming_total_weight":"dropoff_weight"}, inplace=True)

    df = pd.merge(df, pickup_order_df, on='pickup_order_id', how='left')
    df = pd.merge(df, dropoff_order_df, on='dropoff_order_id', how='left') 

    pickup_count_df = df.groupby('rfid_id')['pickup_order_id'].count().reset_index(name='pickup_count') 
    dropoff_count_df = df.groupby('rfid_id')['dropoff_order_id'].count().reset_index(name='dropoff_count') 

    pickup_dropoff_count_df = pd.merge(pickup_count_df, dropoff_count_df, on='rfid_id')
    pickup_dropoff_count_df = pd.merge(pickup_dropoff_count_df, rfid_df, on='rfid_id', how='left') 

    pickup_dropoff_count_df = pd.merge(pickup_dropoff_count_df, item_type_df, on='item_type_id', how='left')
    pickup_dropoff_count_df['last_updated_date'] = pd.to_datetime(pickup_dropoff_count_df['last_updated_date'], format='%Y-%m-%d %H:%M:%S')
    
    pickup_dropoff_count_df['inactive_time'] = today - pickup_dropoff_count_df['last_updated_date']
    pickup_dropoff_count_df['inactive_time'] = pickup_dropoff_count_df['inactive_time'].dt.days 

    pickup_dropoff_count_df['last_seen_location_id'].fillna(pickup_dropoff_count_df['location_id'], inplace=True) 
    pickup_dropoff_count_df.drop(['location_id'], axis=1, inplace=True)

    pickup_dropoff_count_df = pd.merge(pickup_dropoff_count_df, location_df, on='last_seen_location_id', how='left') 
    # Fill null locations with "not specified" 
    pickup_dropoff_count_df['last_seen_location_name'].fillna("Not specified", inplace=True) 
    pickup_dropoff_count_df['last_seen_location_name'].fillna("Not specified", inplace=True) 
    
    df['last_operation'] = 'pickup'
    df.loc[df['pickup_order_id'].isnull(), 'last_operation'] = 'dropoff'  
    df.loc[df['pickup_order_id'].isnull() & df['dropoff_order_id'].isnull(), 'last_operation'] = 'No order cycle'

    last_pickup_date = df[['rfid_id', 'last_pickup_date', 'pickup_order_id']].groupby('rfid_id').apply(lambda x: x.sort_values('last_pickup_date', ascending=False).iloc[0]).reset_index(drop=True)
    last_dropoff_date = df[['rfid_id', 'last_dropoff_date']].groupby('rfid_id').apply(lambda x: x.sort_values('last_dropoff_date', ascending=False).iloc[0]).reset_index(drop=True)

    last_operation_df = df.groupby('rfid_id')['last_operation'].last().reset_index()

    df.drop('last_operation', axis=1, inplace=True) 

    df = pd.merge(df, last_operation_df, on='rfid_id', how='left')
    pickup_dropoff_count_df = pd.merge(pickup_dropoff_count_df, last_operation_df, on='rfid_id', how='left')

    pickup_dropoff_count_df = pd.merge(pickup_dropoff_count_df, last_pickup_date, on='rfid_id', how='left')
    pickup_dropoff_count_df = pd.merge(pickup_dropoff_count_df, last_dropoff_date, on='rfid_id', how='left')

    pickup_dropoff_count_df['last_pickup_date'] = pickup_dropoff_count_df['last_pickup_date'].dt.date
    pickup_dropoff_count_df['last_dropoff_date'] = pickup_dropoff_count_df['last_dropoff_date'].dt.date
    # pickup_dropoff_count_df['creation_date'] = pickup_dropoff_count_df['creation_date'].dt.date
    # pickup_dropoff_count_df['last_updated_date'] = pickup_dropoff_count_df['last_updated_date'].dt.date
    pickup_dropoff_count_df['last_scan_date'] = pickup_dropoff_count_df['last_scan_date'].dt.date

    pickup_dropoff_count_df.drop(['ragout_date'], axis=1, inplace=True)
    pickup_dropoff_count_df.loc[pickup_dropoff_count_df.inactive_time < 0, 'inactive_time'] = 0 

    pickup_dropoff_count_df.loc[pickup_dropoff_count_df.location_type == 'other', 'location_type'] = None
    pickup_dropoff_count_df.loc[pickup_dropoff_count_df.location_type == 'laundry_chute', 'location_type'] = 'Laundry Chute'
    pickup_dropoff_count_df['location_type'].fillna(pickup_dropoff_count_df['last_seen_location_name'], inplace=True)
    pickup_dropoff_count_df.loc[(pickup_dropoff_count_df.location_customer_id != customer_id) & (pickup_dropoff_count_df.location_type != 'Not specified'), 'location_type'] = 'Other customer' 
    pickup_dropoff_count_df.loc[pickup_dropoff_count_df.side == 'facility', 'location_type'] = 'Facility' 

    pickup_dropoff_count_df = pickup_dropoff_count_df[pickup_dropoff_count_df.status == 'active'] 
    pickup_dropoff_count_df.drop(['status'], axis=1, inplace=True)
    
    return df, pickup_dropoff_count_df