def solution(A):
   a=frozenset(sorted(A))
   m=max(a)
   if m>0:
       for i in range(1,m):
           if i not in a:
              return i
       else:
          return m+1
   else:
       return 1