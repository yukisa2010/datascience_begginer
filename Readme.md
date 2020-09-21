# plt.plot(x,a*x+b,'r')

- plt.plot(x軸、y軸,'r')
    - r => 赤
    - y=ax + b：一次関数

- np.sqrt(x)
    - 平方根

# 重回帰分析
```python
import sklearn
from sklearn.linear_model import LinearRegression

# 重回帰分析モデルのインスタンスを作成
lreg = LinearRegression()

# xの値（多項式/説明変数）・yの値（目的変数・価格）
# y = b + a1x1 + a2x2 + ... + anxn / 多項式
X_multi = boston_df.drop('Price',1)
Y_target = boston_df.Price

# クラスへ変数を入れる
lreg.fit(X_multi,Y_target)

# 切片(b)値
print(lreg.intercept_)
# 係数(a1,a2,...an)[配列]
print(lreg.coef_)

# 学習用とテスト用にDataFrameを分ける
https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html

from sklearn.model_selection import train_test_split
# 毎回ランダムにデータ抽出
X_train, X_test, Y_train, Y_test = train_test_split(X_multi,Y_target)

lreg = LinearRegression()
lreg.fit(X_train,Y_train)

# Xの学習用・テスト用でそれぞれYの値を予測
pred_train = lreg.predict(X_train)
pred_test = lreg.predict(X_test)

# 実際値と予測値の誤差計算
print(np.mean((Y_train - pred_train)**2),'train')
print(np.mean((Y_test - pred_test)**2),'test')


```


## scatter 散布図

```python
# 散布図
plt.scatter(x軸(配列),y軸(配列))

# 水平線・垂直線
plt.hlines(y=0,xmin=1.0,xmax=50)

```
- alpha => 透明度
- c => 色(r,g,b)


# ロジスティック回帰分析

```python

# math.exp(-t) => e**-t
import math

# ロジスティック関数
def logistic(t):
    return 1.0/(1 + math.exp((-1.0)*t))

# -6から6までの点を500個（配列）
t = np.linspace(-6,6,500)
# 対象の式を作成
y = np.array([logistic(ele) for ele in t])

# 点を線で結ぶ
plt.plot(t,y)

```

## sns.countplot

```python
# 棒グラフ(x,y)
sns.countplot(x軸,data=y軸,hue=列名,)
# hue => 分類分けした複数の棒を表示


```

## ダミー変数
```python

# 列を展開
# 1.IT系、2.介護系 3.医療系、などのような、数値の大小と関連しないデータを1と0で表現する
acc_dummies = pd.get_dummies(DataFrame(列データ))

```
### 多重変遷性

=> ex. male = 0/female = 1
=> 他重回帰分析において同じ情報を持った列を複数入れるのは良くない。
=> X.drop(削除したい列,axis=1)





