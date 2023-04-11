import numpy as np
import pandas as pd


def fill_empty(df):
    df = df.drop(columns=['tcp.srcport','tcp.dstport','udp.srcport','udp.dstport','tcp.analysis.lost_segment'], errors="ignore")
    if 'ip.proto' in df.columns:
        df = df[~df['ip.proto'].isnull()] #filter lege ip protocollen eruit
    
    df = df.fillna(0) #vul de lege tcp.flags op met waarde 0
    df = df.reset_index(drop=True)
    return df

def make_stats(df): 
    #zet frame.time om in minuten en seconden
    df['minutes']= df["frame.time"].apply(lambda x: float(x.split(':')[1])+ float(x.split(':')[-1][:12])/60)
    df['seconds']= df["frame.time"].apply(lambda x: float(x.split(':')[1])*60+ float(x.split(':')[-1][:12]))
    #aantal packets vorige minuut
    df['packets_per_min'] = np.nan
    for idx, a in enumerate(df['minutes']):
        df.loc[idx, 'packets_per_min'] = len(df[(df['minutes'] <= a) & (df['minutes'] >= (a-1))])
    #aantal packets vorige seconde    
    df['packets_per_sec'] = np.nan
    for idx, a in enumerate(df['seconds']):
        df.loc[idx, 'packets_per_sec'] = len(df[(df['seconds'] < a) & (df['seconds'] > (a-1))])
    #min, max en mean packets voorgaande seconde gezien.
    df['max_packets'] = np.nan
    df['min_packets'] = np.nan
    df['mean_packets'] = np.nan

    for idx, a in enumerate(df['seconds']):
        temp = df[(df['seconds'] <= a) & (df['seconds'] >= (a-1))]['packets_per_sec']
        if len(temp) != 0:
            df.loc[idx,'max_packets'] = max(temp)
            df.loc[idx,'min_packets'] = min(temp)
            df.loc[idx, 'mean_packets'] =  np.mean(temp)
            
    #number per packets per proto en ip-end adress.
    #we interpreteren dit als het aantal packetten van de laatste seconde gestuurd naar dit end-ip of proto
    #we zouden ook het mac-adress kunnen bekijken die slaat op de device zelf.
    #ip adress verandert als je op een andere locatie je kabel erin steekt/wifi connecteert. 
    #Of als je een tijdje niet die kabel hebt gebruikt.
    #stats.packets_per_proto
    
    if 'ip.proto' in df.columns:
        df['stats.packets_per_proto'] = np.nan
        for idx, row in df.iterrows():
            df.loc[idx,'stats.packets_per_proto'] = len(df[(df['seconds'] < row['seconds']) &
                                                  (df['seconds'] > (row['seconds']-1)) &
                                                  (df['ip.proto'] == row['ip.proto'])])
    df['packets_per_ip.dst'] = np.nan
    for idx, row in df.iterrows():
        df.loc[idx,'packets_per_ip.dst'] = len(df[(df['seconds'] < row['seconds']) &
                                                  (df['seconds'] > (row['seconds']-1)) &
                                                  (df['ip.dst'] == row['ip.dst'])])

    return df


def make_tcp_flag(df):
    #convert bytestrings to tcp flags
    #these are mapped like:
    # '0x00000002'->'SYN'
    # '0x00000012'->'SYN+ACK'
    # '0x00000010'->'ACK'
    # '0x00000018'->'PSH+ACK'
    # '0x00000011'->'FIN+ACK'
    
    df = df.reindex(columns = df.columns.tolist() + ["SYN","ACK","FIN","PSH"])
    for idx, row in df.iterrows():
        temp = row['tcp.flags']
        if temp == '0x00000002':
            df.loc[idx, 'SYN']=1
        elif temp == '0x00000012':
            df.loc[idx, 'SYN']=1
            df.loc[idx, 'ACK']=1
        elif temp == '0x00000010':
            df.loc[idx, 'ACK']=1
        elif temp == '0x00000018':
            df.loc[idx, 'ACK']=1
            df.loc[idx, 'PSH']=1
        elif temp == '0x00000011':
            df.loc[idx, 'ACK']=1
            df.loc[idx, 'FIN']=1
    
    df = df.fillna(0)
    return df

def map_ip_proto(df):
    #is tcp map to value 0, 17 is udp map to 1
    if 'ip.proto' in df.columns:
        ip_proto_map = {6.0:'tcp', 17.0:'udp'}
        df['ip.proto'] = df['ip.proto'].map(ip_proto_map)
        df = pd.get_dummies(df, columns=['ip.proto'])
        if 'ip.proto_udp' in list(df.columns.values):
            df = df.drop(columns=['ip.proto_udp'])
            
    return df

def standard_drop(df):
    to_drop = ['frame.number', 'frame.time', 'eth.src', 'eth.dst', 'ip.src', 'ip.dst',
           'tcp.flags', 'minutes', 'seconds']
    df = df.drop(columns=to_drop, errors="ignore")
    return df

def preprocess_tshark(df):
    df = fill_empty(df)
    df = make_stats(df)
    df = make_tcp_flag(df)
    df = map_ip_proto(df)
    df = standard_drop(df)
    return df