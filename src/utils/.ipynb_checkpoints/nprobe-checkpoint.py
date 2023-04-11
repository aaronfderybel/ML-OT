import pandas as pd

from sklearn.preprocessing import MinMaxScaler


def preprocess_nprobe(df):
    #remove duplicates
    df.drop_duplicates(inplace=True)
    
    #preprocessing protocol layer 4
    ip_prot_map = {6:'TCP', 17:'UDP', 1:'ICMP', 2:'IGMP', 58:'IPv6-ICMP', 47:'GRE', 0:'HOPOPT'}
    df['PROTOCOL'] = df['PROTOCOL'].map(ip_prot_map).fillna('other')
    df = pd.get_dummies(df, columns=['PROTOCOL'])
    
    #dropping some columns
    df.drop(columns=['IPV4_SRC_ADDR','L4_SRC_PORT', 'IPV4_DST_ADDR', 'L4_DST_PORT'], inplace=True)
    
    #rename columns
    df.rename(columns={'Label':'label', 'Attack':'class'}, inplace=True)
    
    return df
    
def postprocess_nprobe(x_train, x_test):
    cols_scale = ['L7_PROTO','IN_BYTES','OUT_BYTES','IN_PKTS','OUT_PKTS','TCP_FLAGS',
             'FLOW_DURATION_MILLISECONDS']
    
    scaler = MinMaxScaler()
    scaler.fit(x_train[cols_scale])
    
    if x_test is not None:
        x_train[cols_scale], x_test[cols_scale] = scaler.transform(x_train[cols_scale]),\
        scaler.transform(x_test[cols_scale])
    else:
        x_train[cols_scale] = scaler.transform(x_train[cols_scale])
    
    return x_train, x_test

    
    