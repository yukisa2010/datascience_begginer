# plt.plot(x,a*x+b,'r')

- plt.plot(x軸、y軸,'r')
    - r => 赤
    - y=ax + b：一次関数

- np.sqrt(x)
    - 平方根

- 重回帰分析
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

```