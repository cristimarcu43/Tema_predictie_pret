import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from gensim.models import Word2Vec
from sklearn.metrics import r2_score

train_data = pd.read_json('train.json')
test_data = pd.read_json('test.json')


train_data['marca_model'] = train_data['marca'].astype(str) + '_' + train_data['model'].astype(str)
test_data['marca_model'] = test_data['marca'].astype(str) + '_' + test_data['model'].astype(str)


mlb = MultiLabelBinarizer()
train_addons_encoded = pd.DataFrame(mlb.fit_transform(train_data['addons']), columns=mlb.classes_, index=train_data.index)
test_addons_encoded = pd.DataFrame(mlb.transform(test_data['addons']), columns=mlb.classes_, index=test_data.index)


train_data = train_data.drop(['addons', 'marca', 'model'], axis=1)
test_data = test_data.drop(['addons', 'marca', 'model'], axis=1)


train_data = pd.concat([train_data, train_addons_encoded], axis=1)
test_data = pd.concat([test_data, test_addons_encoded], axis=1)


label_encoder = LabelEncoder()
categorical_columns = ['cutie_de_viteze', 'combustibil', 'transmisie', 'caroserie', 'culoare', 'optiuni_culoare']
for column in categorical_columns:
    train_data[column] = label_encoder.fit_transform(train_data[column])
    test_data[column] = label_encoder.transform(test_data[column])

vector_size = 100
word2vec_model = Word2Vec(sentences=train_data['marca_model'].apply(lambda x: x.split('_')), vector_size=10, window=5, min_count=1, workers=4)
word2vec_columns = ['marca_model_' + str(i) for i in range(10)]
train_data[word2vec_columns] = train_data['marca_model'].apply(lambda x: pd.Series(word2vec_model.wv[x.split('_')[0]] if x.split('_')[0] in word2vec_model.wv else [0]*vector_size))
test_data[word2vec_columns] = test_data['marca_model'].apply(lambda x: pd.Series(word2vec_model.wv[x.split('_')[0]] if x.split('_')[0] in word2vec_model.wv else [0]*vector_size))

numeric_columns = ['an', 'km', 'putere']
scaler = StandardScaler()
train_data[numeric_columns] = scaler.fit_transform(train_data[numeric_columns])
test_data[numeric_columns] = scaler.transform(test_data[numeric_columns])

weights = {'marca_model': 1.5, 'an': 2.0, 'km': 2.5}

for column, weight in weights.items():
    train_data[column] = pd.to_numeric(train_data[column], errors='coerce')
    test_data[column] = pd.to_numeric(test_data[column], errors='coerce')
    train_data[column] = train_data[column] * weight
    test_data[column] = test_data[column] * weight


X = train_data.drop(['pret', 'id', 'marca_model'], axis=1)
y = train_data['pret']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train, y_train)

predictions_val = ridge_model.predict(X_val)

r2_val = r2_score(y_val, predictions_val)
print(f'R2 Score pe setul de validare (Ridge): {r2_val}')

X_test = test_data.drop(['pret', 'id', 'marca_model'], axis=1)
predictions_test = ridge_model.predict(X_test)

test_data['pret'] = predictions_test

test_data[['id', 'pret']].to_json('test.json', orient='records', lines=True)
