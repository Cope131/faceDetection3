from sklearn import preprocessing

# normalized the data - no duplicates
# label encoder object
le = preprocessing.LabelEncoder()

# data in array to be normalized - numerical or non-numerical
data = ['a', 'b', 'b', 'c', 'a']
le.fit(data)

# label encoder object
print(le)
# normalized data - numpy
print(le.classes_)
print(type(le.classes_))
# normalized data - list
print(list(le.classes_))
print(type(list(le.classes_)))

# maps the each value in data to the index of data that was normalized
data = ['c', 'a', 'b']
transformed_data = le.transform(data)
print(list(transformed_data))

le2 = preprocessing.LabelEncoder()
names = ['jimin', 'jimin', 'jimin', 'jimin', 'suga', 'suga', 'suga']
labels = le2.fit_transform(names)

# normalized names
print(list(le2.classes_))
# names mapped to index of normalized names
print(list(labels))


def valueType(value):
    return str(type(value))


print(valueType('hello'))
