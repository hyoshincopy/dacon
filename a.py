
n=[1,2,3,4,5]
lost=[3,4]
if lost[0:len(lost)]==n[0:len(lost)] or lost[-len(lost):-1]==n[-len(lost):-1]:
    print(1)