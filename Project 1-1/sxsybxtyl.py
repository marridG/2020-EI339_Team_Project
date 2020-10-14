import constants
import environment
trans_dic={}
for i in range(1,22):
    dic_tmp={}
    for j in range(-10,0):
        dic_tmp[i+j]=1/30
    for j in range(1,11):
        dic_tmp[i+j]=1/15
    trans_dic[i]=dic_tmp

env = environment.Easy21Env()

stick_dic={}
for i in range(1,11):
    dic_tmp={}
    for j in range(1,22):
        dic_ttt={-1:0,0:0,1:0}
        for q in range(1000):
            env.set(i,j)
            state_next, reward, _ = env.step(action=1)
            dic_ttt[reward]+=1
        dic_ttt[0]/=1000
        dic_ttt[1]/=1000
        dic_ttt[-1]/=1000
        dic_tmp[j]=dic_ttt
    stick_dic[i]=dic_tmp

print(stick_dic)



