import pandas as pd

data=pd.read_csv(r"item_store_feature2.csv")
data['year']=data['date'].apply(lambda x: int(str(x)[:4]))
data['month']=data['date'].apply(lambda x: int(str(x)[4:6]))
data['day']=data['date'].apply(lambda x: int(str(x)[6:]))
for i in range(1,6):
    data[str(i)+'_demand']=0
    data.loc[data['store_code']==i,str(i)+'_demand']=data[data['store_code']==i]['qty_alipay']
data=data.drop(['cart_uv','num_gmv','qty_gmv','unum_gmv','amt_alipay','num_alipay','qty_alipay',
           'unum_alipay','num_alipay_njhs','amt_alipay_njhs','qty_alipay_njhs','unum_alipay_njhs','date'],axis=1)
data=data.groupby(by=['month','year','store_code','item_id']).agg({'day':"median",'month':'median','item_id':"median",'store_code':"median",'brand_id':"median",'supplier_id':"median",'pv_ipv':'sum','pv_uv':'sum','cart_ipv':'sum',
            'collect_uv':'sum','amt_gmv':'sum','ztc_pv_ipv':'sum','tbk_pv_ipv':'sum','ss_pv_ipv':'sum','jhs_pv_ipv':'sum','ztc_pv_uv':'sum','tbk_pv_uv':'sum','ss_pv_uv':'sum','jhs_pv_uv':'sum','1_demand':'sum','2_demand':'sum','3_demand':'sum','4_demand':'sum','5_demand':'sum'})
data=data.astype(int)
data.to_csv("data.csv",index=0)

