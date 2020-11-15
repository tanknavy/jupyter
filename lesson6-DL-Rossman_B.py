from fastai.structured import *
from fastai.column_data import *
np.set_printoptions(threshold=50, edgeitems=20)

PATH = 'C:/input/Fast_AI/rossmann-store-sales/'
filenames=glob(PATH + '*.csv')
print(filenames)
tables = [pd.read_csv(fname, low_memory=False) for fname in filenames]


def concat_csvs(dirname):  # 目录下全部csv文件合成一个
    path = '{}{}'.format(PATH, dirname)
    filenames = glob(PATH + '*.csv')

    wrote_header = False
    with open(path + '.csv', 'w') as outputfile:  # 输出文件
        for filename in filenames:
            name = filename.split(".")[0]
            with open(filename) as f:
                line = f.readline()
                if not wrote_header:
                    wrote_header = True
                    # outputfile.write('file,' + line)
                for line in f:
                    outputfile.write(name + ',' + line)
                outputfile.write("\n")

table_names = ['train', 'store', 'store_states', 'state_names', 'googletrend', 'weather', 'test']
googletrend, state_names, store, store_states, test, train, weather = tables

# turn to booleans
train.StateHoliday = train.StateHoliday != '0' # dtype: bool
test.StateHoliday = test.StateHoliday != '0'

def join_df(left, right, left_on, right_on=None, suffix='_y'):
    if right_on is None: right_on = left_on
    return left.merge(right, how='left', left_on=left_on, right_on=right_on,
                      suffixes=("", suffix))

weather = join_df(weather, state_names, "file","StateName") # file和StateName 相等
googletrend['Date'] = googletrend.week.str.split(' - ', expand=True)[0] # 分列成两列，
googletrend.columns
googletrend[:2]
googletrend['State'] = googletrend.file.str.split('_', expand=True)[2] # 取state编码
googletrend.columns
googletrend[:2]
googletrend.loc[googletrend.State=='NI',"State"] = 'HB,NI' # loc[rows, cols], 对应行和列
len(googletrend[googletrend.State=='HB,NI'])

add_datepart(weather,'Date', drop=False)
add_datepart(googletrend,'Date',drop=False)
add_datepart(train,'Date',drop=False)
add_datepart(test,'Date',drop=False)

trend_de = googletrend[googletrend.file=='Rossmann_DE']
trend_de[:2]
store = join_df(store, store_states,"Store")
len(store[store.State.isnull()]) # 在join之后，检查State栏位有多少null

joined = join_df(train,store,"Store")
joined_test = join_df(test,store,"Store")
len(joined[joined.StoreType.isnull()]), len(joined_test[joined_test.StoreType.isnull()])

joined = join_df(joined,googletrend,["State","Year", "Week"])
joined_test = join_df(joined_test, googletrend, ["State","Year", "Week"])
joined.columns
len(joined[joined.trend.isnull()]),len(joined_test[joined_test.trend.isnull()])

#joined.columns, trend_de.columns
joined = joined.merge(trend_de, 'left', ["Year","Week"], suffixes=('','_DE')) # left join,如果有相同栏位名，后者加_DE后缀
joined_test = joined_test.merge(trend_de, 'left', ["Year","Week"], suffixes=('','_DE'))
len(joined[joined.trend_DE.isnull()]), len(joined_test[joined_test.trend_DE.isnull()])

joined = join_df(joined, weather, ["State","Date"])
joined_test = join_df(joined_test, weather, ["State","Date"])
len(joined[joined.Mean_TemperatureC.isnull()]),len(joined_test[joined_test.Mean_TemperatureC.isnull()])

for df in (joined, joined_test):
    for c in df.columns:
        if c.endswith('_y'): # pd.merge在join两个df时，相同栏位则分别以_x和_y结尾
            if c in df.columns: df.drop(c, inplace=True, axis=1) # 包含相同的column，删除

# 填充NA (not available)
for df in (joined, joined_test):
    df['CompetitionOpenSinceYear'] = df.CompetitionOpenSinceYear.fillna(1900).astype(np.int32) #起始年度默认为1900
    df['CompetitionOpenSinceMonth'] = df.CompetitionOpenSinceMonth.fillna(1).astype(np.int32) # 起始月份为1
    df['Promo2SinceYear'] = df.Promo2SinceYear.fillna(1900).astype(np.int32)
    df['Promo2SinceWeek'] = df.Promo2SinceWeek.fillna(1).astype(np.int32)

for df in (joined, joined_test):
    df['CompetitionOpenSince'] = pd.to_datetime(dict(year=df.CompetitionOpenSinceYear,
                                                    month=df.CompetitionOpenSinceMonth, day=15)) # 转换为时间日期格式
    df["CompetitionDaysOpen"] = df.Date.subtract(df.CompetitionOpenSince).dt.days # pandas.Series.dt 日期时间类型的访问器对象

# df.loc定位行和列，行使用 df.column < 0 来选择
for df in (joined, joined_test):
    df.loc[df.CompetitionDaysOpen<0,"CompetitionDaysOpen"] = 0
    df.loc[df.CompetitionOpenSinceYear<1990, "CompetitionDaysOpen"] = 0

#添加 "CompetitionMonthsOpen" 栏位， 限制最大到2年
for df in (joined,joined_test):
    df["CompetitionMonthsOpen"] = df["CompetitionDaysOpen"]//30 # 产生天数
    df.loc[df.CompetitionMonthsOpen>24, "CompetitionMonthsOpen"] = 24 #大于24个月的设定为24
joined.CompetitionMonthsOpen.unique()  # 栏位值看看是否在24个月以内

# 对promo dates采取相同操作, 根据year和第几周计算相应周一的具体日期
for df in (joined,joined_test):
    df["Promo2Since"] = pd.to_datetime(df.apply(lambda x: Week(x.Promo2SinceYear, x.Promo2SinceWeek).monday(),axis=1).astype(pd.datetime))
    df["Promo2Days"] = df.Date.subtract(df["Promo2Since"]).dt.days

temp = joined[["Promo2SinceYear","Promo2SinceWeek"]][:10];joined[["Promo2SinceYear","Promo2SinceWeek","Promo2Since"]][:10],temp,type(temp),type(joined)
#temp2 = joined_test[["Promo2SinceYear","Promo2SinceWeek"]][:10]
#temp["Promo2Since_B"] = pd.to_datetime(temp.apply(lambda x: Week(x.Promo2SinceYear, x.Promo2SinceWeek).monday(),axis=1).astype(pd.datetime))

for df in (joined, joined_test):
    df.loc[df.Promo2Days<0,"Promo2Days"] = 0
    df.loc[df.Promo2SinceYear<1990, "Promo2Days"] = 0
    df["Promo2Weeks"] = df["Promo2Days"]//7
    df.loc[df.Promo2Weeks<0, "Promo2Weeks"] = 0
    df.loc[df.Promo2Weeks>25, "Promo2Weeks"] = 25
    df.Promo2Weeks.unique()

joined.to_feather('{}joined'.format(PATH))
joined_test.to_feather('{}joined_test'.format(PATH))

# 对每个Store，是否假日，日期三个栏位，分析相邻的两个的假日间隔了多少天
def get_elapsed(fld, pre): # 栏位，前缀
    # Datetimes and Timedeltas work together to provide ways for simple datetime calculations.
    day1 =      np.timedelta64(1,'D') # 相隔一天
    last_date = np.datetime64() # np.datetime64('2005-02-25')
    last_store = 0 #原始数据中store按照数字编码
    res = []
    for s,v,d in zip(df.Store.values,df[fld].values, df.Date.values): # store id, field, date
        if s != last_store: # 如果store不同于前一个
            last_date = np.datetime64() # 初始化时间
            last_store = s
        if v: last_date = d #如果栏位是真(是假日)，上次放假日期为当天
        res.append((d-last_date).astype('timedelta64[D]') / day1) # 距离上次日期多少天了
    df[pre+fld] = res # 增加新栏位

columns = ["Date", "Store", "Promo", "StateHoliday", "SchoolHoliday"]
df = train[columns].append(test[columns])

#分类Store, Date, 对每一行
fld = 'SchoolHoliday'
df = df.sort_values(['Store','Date']) # 默认都增量排序
get_elapsed(fld,'After') # 距离上次放假还有多少天了
#df[:20]
df = df.sort_values(['Store','Date'], ascending=[True, False]) # False按照日期递减
#df[:20]
get_elapsed(fld,'Before') # Date递减，距离下次放假还有多少天
df[:20]


# 州立节假日
fld = 'StateHoliday'
df = df.sort_values(['Store', 'Date'])
get_elapsed(fld, 'After')
df = df.sort_values(['Store', 'Date'], ascending=[True, False])
get_elapsed(fld, 'Before')

fld = 'Promo'
df = df.sort_values(['Store', 'Date'])
get_elapsed(fld, 'After')
df = df.sort_values(['Store', 'Date'], ascending=[True, False])
get_elapsed(fld, 'Before')

df = df.set_index("Date") # 将日期设置为索引
df[:3]

# 将elapsed field中空值设定为0
columns = ['SchoolHoliday', 'StateHoliday', 'Promo']
df.columns

for o in ['Before','After']: # 之前get_elasped分析了after和before时间序列两个栏位
    for c in columns:
        n = o+c
        df[n] = df[n].fillna(0).astype(int) # 空值填充为0，类型是int

# 按照index排序，分组，rolling
bwd = df[['Store']+columns].sort_index().groupby("Store").rolling(7, min_periods=1).sum() #窗口大小为7
fwd = df[['Store']+columns].sort_index(ascending=False).groupby("Store").rolling(7,min_periods=1).sum() # 倒序，

bwd.drop('Store',1,inplace=True) #删除Store索引，
#bwd[:10]
bwd.reset_index(inplace=True) # 重新产生新的index
fwd.drop('Store',1,inplace=True)
fwd.reset_index(inplace=True)
df.reset_index(inplace=True)
fwd[:10]

# 合并
df = df.merge(bwd,'left',['Date','Store'],suffixes=['','_bw'])
df = df.merge(fwd,'left',['Date','Store'],suffixes=['','_fw'])

df.columns.is_unique #True
len(df.columns) #17
df.drop(columns,1,inplace=True) #删除某些栏位，沿着列方向
len(df.columns) #14

# 大的中间结果最好保存起来
df.to_feather('{}df'.format(PATH))

#temp = df
#df = pd.read_feather('{}df'.format(PATH)) # read_feather错误,改成一下方式
import feather
df = feather.read_dataframe('{}df'.format(PATH))

type(df.Date) # 文件读取以后，是Series
df['Date'] = pd.to_datetime(df.Date)
type(df.Date)

joined = join_df(joined,df,['Store','Date'])
joined_test = join_df(joined_test,df,['Store','Date'])

# 在移除某些行以后，再次重置索引
joined.reset_index(inplace=True)
joined_test.reset_index(inplace=True)
joined[:10]

#再次保存
joined.to_feather('{}joined2'.format(PATH))
joined_test.to_feather('{}joined_test2'.format(PATH))

cols = set() # python 集合里面是唯一
for col in joined.columns:
    if col not in cols:
        cols.add(col)
    else:
        print(col) # StateName_y

# 现在有了最终特征工程后的features
# Create features
# 特征工程已经完成
# convert to input compatible with neural network
# this includes converting categorical variables into contiguous integers of one-hot encondings,
# normalizing continuous feathres to standard normal.etc...

cat_vars = ['Store', 'DayOfWeek', 'Year', 'Month', 'Day', 'StateHoliday', 'CompetitionMonthsOpen',
    'Promo2Weeks', 'StoreType', 'Assortment', 'PromoInterval', 'CompetitionOpenSinceYear', 'Promo2SinceYear',
    'State', 'Week', 'Events', 'Promo_fw', 'Promo_bw', 'StateHoliday_fw', 'StateHoliday_bw',
    'SchoolHoliday_fw', 'SchoolHoliday_bw']

contin_vars = ['CompetitionDistance', 'Max_TemperatureC', 'Mean_TemperatureC', 'Min_TemperatureC',
   'Max_Humidity', 'Mean_Humidity', 'Min_Humidity', 'Max_Wind_SpeedKm_h',
   'Mean_Wind_SpeedKm_h', 'CloudCover', 'trend', 'trend_DE',
   'AfterStateHoliday', 'BeforeStateHoliday', 'Promo', 'SchoolHoliday']

n = len(joined); n

dep = 'Sales' #因变量
joined = joined[cat_vars+contin_vars+[dep, 'Date']].copy()

joined_test[dep] = 0 # test数据集没有Sales栏位
joined_test = joined_test[cat_vars+contin_vars+[dep, 'Date', 'Id']].copy() # 观察数据
joined.head()

for v in cat_vars: joined[v] = joined[v].astype('category').cat.as_ordered()
joined.head()
apply_cats(joined_test,joined) # joined的栏位类别编码方式对joint_test进行相同编

for v in contin_vars:
    joined[v] = joined[v].fillna(0).astype('float32') # 连续变量的空值填充为0
    joined_test[v] = joined_test[v].fillna(0).astype('float32')

idxs = get_cv_idxs(n, val_pct=150000/n) # 抽样得到数据的索引
joined_samp = joined.iloc[idxs].set_index("Date") # 根据行号定位，再重设索引
samp_size = len(joined_samp); samp_size

df,y,nas,mapper = proc_df(joined_samp, 'Sales', do_scale=True)
yl = np.log(y)

joined_test = joined_test.set_index("Date")
df_test,_,nas,mapper = proc_df(joined_test, 'Sales', do_scale=True, skip_flds=['Id'],mapper=mapper, na_dict=nas)
mapper


train_ratio = 0.75
train_size = int(samp_size * train_ratio);train_size
val_idx = list(range(train_size,len(df))) # 取最后25%

# 取验证机方法二：取和测试集相同长度时间周期的数据
#val _idx = np.flatnonzero((df.index<=datetime.datetime(2014,9,17)) & (df.index>=datetime.datetime(2014,8,1))) #时间索引
#val_idx=[0]
# Deep Learning
# Root-mean-squared percent errors is the metirc kaggle used for this competition

def inv_y(a): return np.exp(a) #指数转换，之前的y值有Log转换吗？

def exp_rmspe(y_pred,targ): #预测和目标值
    targ = inv_y(targ)
    pct_var = (targ - inv_y(y_pred)) / targ
    return math.sqrt((pct_var ** 2).mean())

max_log_y = np.max(y)
y_range = (0, max_log_y * 1.2)
# 创建ModelData
#?ColumnarModelData @classmehod类方法，返回了该类
md = ColumnarModelData.from_data_frame(PATH, val_idx, df, yl.astype(np.float32), cat_flds=cat_vars, bs=128, test_df=df_test)
# 一些类别变量有多得多，比如Store, 有上千个
cat_sz = [(c, len(joined_samp[c].cat.categories)+1) for c in cat_vars] # 加以为了平滑？
emb_szs = [(c,min(50,(c+1)//2)) for _,c in cat_sz];emb_szs # 限制在最多50个，

# 在上述类方法中返回了一个类的实例，所以这里可以直接调用普通方法get_learner
m = md.get_learner(emb_szs, len(df.columns)-len(cat_vars),
                  0.04, 1, [1000,500], [0.001,0.01], y_range=y_range)
m.summary() # torch

lr = 1e-3
m.lr_find()
m.sched.plot(100)





