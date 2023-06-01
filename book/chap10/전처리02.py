# 코퍼스에 카운터 벡터 적용
from sklearn.feature_extraction.text import CountVectorizer
corpus = [
    'This is last chance.',
    'and if you do not have this chance',
    'you will never get any chance.',
    'will you do get this one?',
    'please, get this chance',
]
vect = CountVectorizer()
vect.fit(corpus)
print(vect.vocabulary_)

# 배열 변환
print(vect.transform(['you will never get any chance.']).toarray())

# 불용어를 제거한 카운터 벡터
vect = CountVectorizer(stop_words=['and', 'is', 'please', 'this']).fit(corpus)
print(vect.vocabulary_)