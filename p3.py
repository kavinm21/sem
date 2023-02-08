import numpy as np
import random
import collections

def extract_elements(lst, i=1):
  return [items[i] for items in lst]

def static_ranking(n,m):
    df_train=np.random.randint(1,6, size=(n,m))
    ranks_sum=list(df_train.sum(axis=0))
    ranks_avg=[[(i/n).round(2)] for i in ranks_sum]
    for i in range(len(ranks_avg)):
      ranks_avg[i].append(ranks_avg.index(ranks_avg[i]))
    offline_ranks=sorted(ranks_avg,key=lambda l:l[0], reverse=True) 
    return extract_elements(offline_ranks,1)

def MAB(offline_ranking):
    item_pref=[]
    for j in range(10):
      item_pref.append(random.choice(offline_ranking))
    return collections.Counter(item_pref)

def final_ranking(presentation_ranking):
    sorted_dict={k: v for k, v in sorted(presentation_ranking.items(), key=lambda item: item[1],reverse=True)}
    print('Final ranking of items:', list(sorted_dict.keys()))

if __name__ == "__main__":
    n=int(input("Enter the number of users:"))
    m=int(input("Enter the number of items:"))
    offline_ranking=static_ranking(n,m)
    presentation_ranking={}
    k=int(input('Enter the value of k:'))
    for i in range(len(offline_ranking[0:k])):
      if(i==0):
        for val in offline_ranking[0:k]:
          presentation_ranking[val]=0
      dup=MAB(offline_ranking[0:k])
      for j in offline_ranking[0:k]:
        try:
          presentation_ranking[j]+=dup[j]
        except:
          presentation_ranking[j]=presentation_ranking[j]
    print('Offline ranking of items:',offline_ranking)
    final_ranking(presentation_ranking)
