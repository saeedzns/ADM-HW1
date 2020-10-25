#!/usr/bin/env python
# coding: utf-8

# In[19]:


# Say "Hello, World!" With Python

Text="Hello, World!"
print(Text)


# In[25]:


# Python If-Else

n=int(input())
if (n%2)!=0:
    print("Weird")
elif n in range(2,6):
    print("Not Weird")
elif n in range(6,21):
    print("Weird")
elif n>20:
    print("Not Weird")  


# In[26]:


# Arithmetic Operators

a=int(input())
b=int(input())
print(a+b)
print(a-b)
print(a*b)


# In[28]:


# Python: Division

a=int(input())
b=int(input())
print(a//b)
print(a/b)


# In[21]:


# Loops
n = int(input())
for i in range(n):
    print(i**2)


# In[16]:


# Write a function

def is_leap(year):
    
    if year==1992 or year%400==0:
        leap=True
    else:
        leap=False
    
    return leap


# In[20]:


# Print Function

n = int(input())
s=''
for i in range(n):
    s=s+str(i+1)
print(s)


# In[145]:


# List Comprehensions

x=int(input())
y=int(input())
z=int(input())
n=int(input())

List=[[i,j,k] for i in range(x+1) for j in range(y+1) for k in range(z+1) if n!=i+j+k]

print(List)


# In[88]:


# Find the Runner-Up! Score

n=int(input())
if n in range(2,11):
    k=input()
    
A=[]
s=''
j=0

for i in range(len(k)):
    if k[i]!=' ':
        s=s+k[i]
    elif s!='':        
        j=j+1
        A.append(int(s))
        s=''
    if j==(n-1):
        A.append(k[i+1:len(k)])
        A[n-1]=int(A[n-1])


    if len(A)>n:
        A.remove(A[n])
        

Max=max(A)
Min=min(A)

for i in range(n):
    if A[i]==Max:
        A[i]=Min    
        
print(max(A))


# In[230]:


# Nested list

n=int(input())
L=[] 
if n in range(2,6):
    for i in range(n):
        name=input()
        score=float(input())
        A=[name,score]
        L.append(A)
        
l=sorted(set(score for name,score in L))

B=[]
for i in range(n):
    if L[i][1]==l[1]:
        B.append(L[i][0])   
        
second_lowest=sorted(B)

print(*second_lowest,sep='\n')


# # >> name print

# In[236]:


for i in range(4):
    name, *line = input().split()


# In[423]:


# Finding the percentage

n = int(input())
name_score = {}
for i in range(n):
    name,*scores= input().split()
    scores = list(map(float, scores))
    name_score[name]=scores
    
query=input()
mean=sum(name_score[query])/len(name_score[query])
    
print(format(mean,".2f"))   


# In[ ]:


# Lists

if __name__ == '__main__':
    N = int(input())
    l=[]
    for i in range(N):
        c=input().split()
        if c[0]=='insert':
            l.insert(int(c[1]),int(c[2]))
        elif c[0]=='print':
            print(l)
        elif c[0]=='remove':
            l.remove(int(c[1]))
        elif c[0]=='append':
            l.append(int(c[1]))
        elif c[0]=='sort':
            l.sort()    
        elif c[0]=='pop':
            l.pop()
        elif c[0]=='reverse':
            l.reverse()


# In[15]:


# Tuples

if __name__ == '__main__':
    n = int(input())
    integer_tuple = tuple(map(int, input().split()))
    print(hash(integer_tuple))


# In[2]:


# sWAP cASE

def swap_case(n):
    k=''
    if len(n) in range(1001):
        for i in n:
            if i.isupper():
                k=k+i.lower()
            else:
                k=k+i.upper()   
    return k


# In[38]:


# String Split and Join

def split_and_join(line):
    # write your code here
    line=line.split(' ')
    join='-'.join(line)
    return join


# In[41]:


# What's Your Name?
def print_full_name(a, b):
    
    print('Hello'+' '+a+' '+b+'!'+' '+'You just delved into python.')


# In[132]:


# Mutations
def mutate_string(string, position, character):
    string=list(string)
    string[position]=str(character)
    Finaly=''.join(string)
    return Finaly


# In[265]:


# Find a string
def count_substring(string, sub_string):
    count=0
    for i in range(len(string)):
        if string[i:i+len(sub_string)]==sub_string:
            count=1+count
    return count


# In[316]:


# String Validators

s=input()
if len(s) in range(1000):
    if any(i.isalnum() for i in s):
        print('True')
    else:
        print('False')
    if any(i.isalpha() for i in s):
        print('True')
    else:
        print('False')
    if any(i.isdigit() for i in s):
        print('True')
    else:
        print('False')
    if any(i.islower() for i in s):
        print('True')
    else:
        print('False')
    if any(i.isupper() for i in s):
        print('True')
    else:
        print('False')


# In[358]:


#Text Alignment

#Replace all ______ with rjust, ljust or center. 

thickness = int(input()) #This must be an odd number
c = 'H'

#Top Cone
for i in range(thickness):
    print((c*i).rjust(thickness-1)+c+(c*i).ljust(thickness-1))

#Top Pillars
for i in range(thickness+1):
    print((c*thickness).center(thickness*2)+(c*thickness).center(thickness*6))

#Middle Belt
for i in range((thickness+1)//2):
    print((c*thickness*5).center(thickness*6))    

#Bottom Pillars
for i in range(thickness+1):
    print((c*thickness).center(thickness*2)+(c*thickness).center(thickness*6))    

#Bottom Cone
for i in range(thickness):
    print(((c*(thickness-i-1)).rjust(thickness)+c+(c*(thickness-i-1)).ljust(thickness)).rjust(thickness*6))


# In[389]:


# Text Wrap
def wrap(string, max_width):
    string=list(string)
    for i in range(len(a)):
        result=''.join(string[i*max_width:(i*max_width)+max_width])
        print(result)
    return result


# In[403]:


# Designer Door Mat

n,m = map(int,input().split())
pattern = [ ((".|.")*(2*i+1)).center(m,'-') for i in range(n//2) ]
print('\n'.join( pattern + ["WELCOME".center(m,'-')] + pattern[::-1] ))


# In[417]:


# String Formatting

def print_formatted(number):
    w = len(str(bin(number)).replace('0b',''))
    for i in range(1,number+1):   
        decimal = str(i).rjust(w,' ')
        octal = oct(i)[2:].rjust(w, ' ')
        hexadecimal = hex(i)[2:].rjust(w, ' ').upper()
        binary = bin(i)[2:].rjust(w,' ')
        print(decimal, octal, hexadecimal, binary)
if __name__ == '__main__':
    n = int(input())
    print_formatted(n)


# # >>>>>

# In[418]:



N=int(input())
for i in range (-(N-1),N):
    for j in range (-2*(N-1),2*(N-1)+1):
        if j%2==0 and (abs(j//2)+abs(i))< N:
              print (chr(abs(j//2)+abs(i)+ord('a')),end='')
        else:
              print('-',end='')
    print()


# In[17]:


# Alphabet Rangoli

def print_rangoli(size):
    width  = size*4-3
    string = ''

    for i in range(1,size+1):
        for j in range(0,i):
            string += chr(96+size-j)
            if len(string) < width :
                string += '-'
        for k in range(i-1,0,-1):    
            string += chr(97+size-k)
            if len(string) < width :
                string += '-'
        print(string.center(width,'-'))
        string = ''

    for i in range(size-1,0,-1):
        string = ''
        for j in range(0,i):
            string += chr(96+size-j)
            if len(string) < width :
                string += '-'
        for k in range(i-1,0,-1):
            string += chr(97+size-k)
            if len(string) < width :
                string += '-'
        print(string.center(width,'-'))
if __name__ == '__main__':
    n = int(input())
    print_rangoli(n)


# In[53]:


# Capitalize!

def solve(s):
    fullname=s.spli`t(' ')
    FullName=[i.capitalize() for i in fullname]
    return ' '.join(FullName)


# In[172]:


# The Minion Game

import re
# The code on hackerrank is not mine,but this is my code which works here but no online
string=input()

Stuart=0
for i in range(len(string)):
    matches= re.finditer(r'(?=([^AEIOU][A-Z]{'+str(i)+'}))',string)
    for i in enumerate(matches):
        for group in range(len(match.groups())):
            Stuart+=1

Kevin=0
for i in range(len(string)):
    matches= re.finditer(r'(?=([AEIOU][A-Z]{'+str(i)+'}))',string)
    for i in enumerate(matches):
        for group in range(len(match.groups())):
            Kevin+=1 

if Stuart>Kevin:
    print("Stuart "+str(Stuart))
elif Kevin>Stuart:
    print("Kevin "+str(Kevin))
else:
    print("Draw")


# In[ ]:


# Merge the Tools!

# Help from thepoorcoder.com

def merge_the_tools(string, k):
    for i in range(0,len(string), k):
        line = string[i:i+k]
        seen = set()
        for i in line:
            if i not in seen:
                print(i,end='')
                seen.add(i)
        #prints a new line
        print()


# In[ ]:


# collections.Counter()

from collections import Counter
shoes_num=int(input())
shoes_size=Counter(map(int,input().split()))
users=int(input())
total=0

for i in range(users):
    size,price=(map(int,input().split()))
    
    if shoes_size[size]:
        total=price+total
        shoes_size[size]=shoes_size[size]-1    
                
print(total) 


# In[87]:


# Introduction to Sets

def average(array):
    array=set(map(int,array))
    average=sum(array)/len(array)
    return average

N=int(input())
array=input().split()
print(average(array))


# In[46]:


# DefaultDict Tutorial

from collections import defaultdict
n,m=map(int,input().split())
p=[]
d = defaultdict(list)
for i in range(n):
    d[input()].append(i+1)
    
for i in range(m):
     p.append(input())
        
for i in range(m):        
    if d[p[i]]!=[]:
        print(*d[p[i]])
    else:
        print(-1)
    


# In[188]:


# Calender Module

import calendar 
M,D,Y=map(int,input().split())
m=calendar.weekday(Y,M,D)
days=['MONDAY','TUESDAY','WEDNESDAY','THURSDAY','FRIDAY','SATURDAY','SUNDAY']
print(days[m])


# In[ ]:


# Exceptions

for i in range(int(input())):
    try:`
        a,b=map(int,input().split())
        print(a//b)
    except Exception as e:
        print("Error Code:",e)


# In[ ]:


# Collections.namedtuple()

from collections import namedtuple
n=int(input())
total=0
student=namedtuple('student',input())
for i in range(n):
    data=student(*input().split())
    total=total+int(data.MARKS)
    
print(format(total/n,'.2f'))    


# In[18]:


# No idea!

nm=list(map(int,input().split()))
n,m=nm[0],nm[1]

s=list(map(int,input().split()))

A=set(map(int,input().split()))
B=set(map(int,input().split()))

happiness=0
for i in list(s):
    if i in A:
        happiness+=1
    elif i in B:
        happiness-=1

print(happiness)


# In[ ]:


# Collections.OrderedDict()

from collections import OrderedDict
order=OrderedDict()
n=int(input())
for i in range(n):
    a=input().split()
    
    if a in order.keys():
        order[a[0]]=order[a[0]]+int(a[1])
    else:
        order[a[0]]=int(a[1])
for j in order.keys():
    print(j,order[j])


# In[27]:


# Symmetric difference

ne=int(input())
e=set(map(int,input().split()))
nf=int(input())
f=set(map(int,input().split()))
print(len(e^f))


# In[ ]:


# Incorrect regex

from re import compile
for i in range(int(input())):
    try:
        compile(input())
        print(True)
    except:
        print(False)  


# In[29]:


# set.add()

n=int(input())
s=set()
for i in range(n):
    c=input()
    s.add(c)
print(len(s))   


# In[16]:


# Word order

from collections import OrderedDict
order=OrderedDict()
n=int(input())
for i in range(n):
    a=input()
    if a in order.keys():
        order[a]=order[a]+1
    else:
        order[a]=1
print(len(order))
print(*order.values())       


# # >> Inter-set-union

# In[122]:


M=int(input())
m=set(list(map(int,input().split())))
N=int(input())
n=set(list(map(int,input().split())))
union=n.union(m)
inter=n.intersection(m)
diff=union.difference(inter)
diff=sorted(list(diff))

print(*diff,sep='\n')


# # >> pop-remove-discard

# In[ ]:


n=int(input())
s=set(map(int,input().split()))
N=int(input())
for i in range(N):
    c=input().split()
    if c[0]=='remove':
        s.remove(int(c[1]))
    elif c[0]=='discard':
        s.discard(int(c[1]))
    elif c[0]=='pop':
        s.pop()
        
print(sum(s)) 


# # >> at least one Subscribe

# In[25]:


ne=int(input())
e=set(map(int,input().split()))
nf=int(input())
f=set(map(int,input().split()))
print(len(e|f))


# # >> Both Subscribe

# In[25]:


ne=int(input())
e=set(map(int,input().split()))
nf=int(input())
f=set(map(int,input().split()))
print(len(e&f))


# # >> difference Subscribe

# In[27]:


ne=int(input())
e=set(map(int,input().split()))
nf=int(input())
f=set(map(int,input().split()))
print(len(e-f))


# # >> update-intersection_update-...

# In[2]:


na=int(input())
A=set(map(int,input().split()))
N=int(input())
B=set()
for i in range(N):
    k=list(input().split())
    s=set(map(int,input().split()))
    if k[0]=='update':
        A|=s
    elif k[0]=='intersection_update':
        A&=s
    elif k[0]=='difference_update':
        A-=s
    elif k[0]=='symmetric_difference_update':
        A^=s
        
print(sum(A)) 


# # >> Captain

# In[3]:


N=int(input())
Tourists=list(map(int,input().split()))
Captain=set()
Family=set()
for i in Tourists:
    if i not in Family:
        Family.add(i)
        Captain.add(i)
    else:
        Captain.discard(i)
print(Captain.pop())               


# # >> subset

# In[20]:


T=int(input())
p=list()

for i in range(T):
    nA=int(input())
    A=set(map(int,input().split()))
    nB=int(input())
    B=set(map(int,input().split()))
    if A<=B:
        p.append('True')
    else:
        p.append('False')
        
print(*p,sep='\n')


# # >> superset

# In[58]:


A=set(map(int,input().split()))
n=int(input())
p=list()

for i in range(n):
    B=set(map(int,input().split()))
    if A>B:
        p.append(True)
    else:
        p.append(False)
        
if sum(p)==n:        
    print('True')
else:
    print('False')   


# # >> Supermarket

# # >> deque()

# In[ ]:


from collections import deque
d=deque()
n=int(input())

for i in range(n):
    c=input().split()
    if c[0]=='append':
        d.append(int(c[1]))
    elif c[0]=='pop':
        d.pop()
    elif c[0]=='popleft':
        d.popleft()
    elif c[0]=='appendleft':
        d.appendleft(int(c[1]))
        
print(*d) 


# # >> company name

# In[102]:


from collections import Counter
s=Counter(sorted(input()))
m=s.most_common(3)
for a,b in m:
    print(a,b)


# # >> reverse numpy.array

# In[143]:


import numpy

def arrays(arr):
    arr.reverse()
    reverse_nparray=numpy.array(arr,float)
    return reverse_nparray

arr = input().strip().split(' ')
result = arrays(arr)
print(result)


# # >> shape reshape

# In[159]:


import numpy
l=numpy.array(input().split(),int)
l=l.reshape(3,3)
print(l)


# # >> transpose

# In[234]:


import numpy
nm=list(map(int,input().split()))
l=[]
for i in range(nm[0]):
    a=input().split()
    l=l+a
arr=numpy.array(l,int)
arr=arr.reshape(nm[0],nm[1])
print(numpy.transpose(arr))
print(numpy.array(l,int))


# # >> concatenate

# In[242]:


import numpy as np

n,m,p=map(int,input().split())
NP=np.array([input().split() for i in range(n)],int)
MP=np.array([input().split() for i in range(m)],int)
print(np.concatenate((NP,MP),axis=0))


# # >> Birthday Cake

# In[276]:


import os
def birthdayCakeCandles(candles):
    # Write your code here
    from collections import Counter
    s=Counter(candles)
    how_many_tallest=max(s.items())[1]
    
    return how_many_tallest
    
    
if __name__ == '__main__':

    candles_count = int(input().strip())

    candles = list(map(int, input().rstrip().split()))

    result = birthdayCakeCandles(candles)

    print(str(result) + '\n')


# # >> Zeros Ones

# In[129]:


import numpy as np
dim=tuple(map(int,input().split()))#tuple() can make (M,N,n) dimension
zero=np.zeros((dim),dtype=np.int)
one=np.ones((dim),dtype=np.int)
print(zero)
print(one)


# # >> Indentity

# In[168]:


import numpy as np
N,M=map(int,input().split())
np.set_printoptions(sign=' ')##copied from hackerank_extra spaces problem
print(np.eye(N,M))


# # >> Array Math

# In[263]:


import numpy as np
N,M=map(int,input().split())
A=np.array([list(map(int,input().split())) for i in range(N)])
B=np.array([list(map(int,input().split())) for i in range(N)])
print(np.add(A,B),np.subtract(A,B),np.multiply(A,B),np.floor_divide(A,B),np.mod(A,B),np.power(A,B),sep='\n')


# # >> Floor Ciel Rint

# In[268]:


import numpy as np
l=map(float,input().split())
arr=np.array(list(l))
np.set_printoptions(sign=' ')#solve extra spaces problem
print(np.floor(arr),np.ceil(arr),np.rint(arr),sep='\n')


# # >> Sum Prod

# In[288]:


import numpy as np
N,M=map(int,input().split())
arr=np.array([list(map(int,input().split())) for i in range(N)])
p=np.prod(np.sum(arr,axis=0))
print(p)


# # >> Min Max numpy

# In[ ]:


import numpy as np
N,M=map(int,input().split())
arr=np.array([list(map(int,input().split())) for i in range(N)])
m=np.max(np.min(arr,axis=1))
print(m)


# # >> mean var std numpy

# In[295]:


import numpy as np
N,M=map(int,input().split())
arr=np.array([list(map(int,input().split())) for i in range(N)])
mean=np.mean(arr,axis=1)
var=np.var(arr,axis=0)
std=np.std(arr)
np.set_printoptions(sign=' ')#copied from hackerank_extra spaces problem
np.set_printoptions(legacy='1.13')#copied from hackerank_1.13 version problem
print(mean,var,std,sep='\n')


# # >> matrix product

# In[311]:


import numpy as np
N=int(input())
A=np.array([list(map(int,input().split())) for i in range(N)])
B=np.array([list(map(int,input().split())) for i in range(N)])
prod=np.dot(A,B)
print(prod)


# # >> Inner Outer

# In[312]:


import numpy as np
A=np.array(list(map(int,input().split())))
B=np.array(list(map(int,input().split())))
inner=np.inner(A,B)
outer=np.outer(A,B)

print(inner,outer,sep='\n')


# In[317]:


import numpy as np
pol=list(map(float,input().split()))
result=np.polyval(pol,float(input()))
print(result)


# # >> determinate

# In[371]:


import numpy as np
N=int(input())
arr=np.array([list(map(float,input().split())) for i in range(N)])
det=np.linalg.det(arr)
print(round(det,2))


# # >> detect floating

# In[576]:


from re import match
#Use these regex >>> r:raw input -- A:Anchor "start" -- d:digit -- Z:$ "end" -- asterisk:zero or more
n=int(input())
result=[bool(match(r'\A[-+]?\d*\.\d+\Z',input())) for i in range(n)]
print(*result,sep='\n')


# # >> re.split()

# In[26]:


from re import split
string=input()
# r:raw input -- \:escape dot -- +:one or more  
result=split(r'[\.,]+',string)
print(*result,sep='\n')


# # >> Group() Groupdict()

# In[295]:


from re import search
# [A-Za-z0-9]\1+ >>> alphanumeric repetition
m=search(r'([A-Za-z0-9])\1+',input())
if m:
    print(m.group(1))
else:
    print(-1)  


# # >> Findall()

# In[366]:


from re import findall
# v:vowels -- c:consonants -- ?<=:lookbehind -- {2,}:at least two -- 
v='[AEIOUaeiou]'
c='[QWRTYPSDFGHJKLZXCVBNMqwrtypsdfghjklzxcvbnm]'
m=findall('('+'?<='+c+')'+'('+v+'{2,}'+')'+c,input())
if m:
    print(*m,sep='\n')
else:
    print(-1)


# # >> Start() End()

# In[ ]:


from re import finditer
string=input() 
k=input()
match=finditer(r'(?=(' + k + '))',string)
c=0
for i in match:
    c=c+1
    print((i.start(1), i.end(1)-1))

if c==0:
    print((-1, -1))


# # >> re.sub()

# In[620]:


from re import sub
n=int(input())
l=[]
for i in range(n):
    string=input()
    s=sub(r'(?<= )(&&)(?= )','and',string)
    ss=sub(r'(?<= )(\|\|)(?= )','or',s)
    l.append(ss)
print(*l,sep='\n')


# # >> Roman numerals

# In[623]:


from re import match
Roman=input()
Ms='M{,3}'
Cs='(C[MD]|D?C{,3})'
Xs='(X[CL]|L?X{,3})'
Nums='(I[VX]|V?I{,3})\Z'
if match(Ms+Cs+Xs+Nums,Roman):
    print(True)
else:
    print(False)


# # >> Email

# In[ ]:


from re import match
from email.utils import parseaddr

for i in range(int(input())):
    s=input()
    User=parseaddr(s)
    m=match(r'^[a-zA-Z][\w\._-]*@[a-zA-Z]+\.[a-zA-Z]{1,3}\Z',User[1])
    if m:
        print(s)


# # >> Hex color code

# In[55]:


import re
#Only css code is copied from:
#https://codeworld19.blogspot.com/2020/10/hex-color-code-in-python-hacker-rank.html
#Others are mine
in_css = False
for i in range(int(input())):
    s = input()
    if '{' in s:
        in_css = True
    elif '}' in s:
        in_css = False
    elif in_css:
        for color in re.findall(r'(#[0-9abcdefABCDEF]{6}|#[0-9abcdefABCDEF]{3})(?![0-9abcdefABCDEF])', s):
            print(color)


# # >> HTML parser part 1

# In[ ]:


import re
import html
from html.parser import HTMLParser
# create a subclass and override the handler methods
class MyHTMLParser(HTMLParser):
    def handle_starttag(self, tag, attrs):
        print("Start :", tag)
        for name,value in attrs:
            print("->",name+" >",value)
    def handle_endtag(self, tag):
        print("End   :", tag)
    def handle_startendtag(self, tag, attrs):
        print("Empty :", tag)
        for name,value in attrs:
            print("->",name+" >",value)
# instantiate the parser and fed it some HTML
parser = MyHTMLParser()
n = int(input())
for _ in range(n):
    parser.feed(input())


# # >> HTML parser part 2

# In[ ]:


from html.parser import HTMLParser
#With taking help from discussions
class MyHTMLParser(HTMLParser):
    def handle_comment(self, input):
        if "\n" in input:
            print(">>> Multi-line Comment  ",input,sep="\n")
        else:
            print(">>> Single-line Comment  ",input,sep="\n")
    def handle_data(self, input):
        if input != "\n":
            print(">>> Data",input,sep="\n")
html = ""       
for i in range(int(input())):
    html += input().rstrip()
    html += '\n'
    
parser = MyHTMLParser()
parser.feed(html)
parser.close()


# # >> Detection in HTML

# In[ ]:


from html.parser import HTMLParser
#with help from:
#https://codeworld19.blogspot.com/2020/10/detect-html-tags-attributes-and.html
class MyHTMLParser(HTMLParser):
    def handle_starttag(self,tag,attributes):
        print(tag)
        for i, value in attributes:
            print("->",i, ">",value)

    def handle_startendtag(self,tag,attributes):
        print(tag)
        for i, value in attributes:
            print("->",i,">",value)

html = ''
for i in range(int(input())):
    html += input().rstrip() + '\n'

parser = MyHTMLParser()
parser.feed(html)
parser.close()


# # >> Validating UID

# In[70]:


import re
l=[]
for i in range(int(input())):
    UID=input()
    m=re.search(r'^(?!.*(.).*\1)(?=(?:.*[A-Z]){2,})(?=(?:.*[0-9]){3,})[a-zA-Z0-9]{10}\Z', UID)
    if m:
        l.append('Valid')
    else:
        l.append('Invalid')
print(*l,sep='\n')  


# # >> Validing credit card

# In[1]:


import re
l=[]
for i in range(int(input())):
    Card=input()
    m=re.search(r'^[456]([0-9]{15}|[0-9]{3}(-[0-9]{4}){3})\Z',Card)
    if m:
        d=re.search(r'((\d)-?(?!(-?\2){3})){16}',Card)
        if d:
            l.append('Valid')
        else:
            l.append('Invalid')
    else:
         l.append('Invalid')
print(*l,sep='\n')


# # >> Email-Lexi order

# In[17]:


def fun(s):
    # return True if s is a valid email, else return False
    import re
    m=re.search(r'^[\w\-_]+@[a-zA-Z0-9]+\.[a-zA-Z]{1,3}\Z',s)
    res=bool(m)
    return res    

def filter_mail(emails):
    return list(filter(fun, emails))

if __name__ == '__main__':
    n = int(input())
    emails = []
    for _ in range(n):
        emails.append(input())

filtered_emails = filter_mail(emails)
filtered_emails.sort()
print(filtered_emails)


# # >> Reduce() function

# In[ ]:


from fractions import Fraction
from functools import reduce

def product(fracs):
    t=reduce(lambda x, y : x * y, fracs) # complete this line with a reduce statement
    return t.numerator, t.denominator

if __name__ == '__main__':
    fracs = []
    for _ in range(int(input())):
        fracs.append(Fraction(*map(int, input().split())))
    result = product(fracs)
    print(*result)


# # >> Quest triangle

# In[32]:


for i in range(1,int(input())): #More than 2 lines will result in 0 score. Do not leave a blank line also
    print(i*(10**i)//9)


# # >> Athlete sort

# In[96]:


if __name__ == '__main__':
    n,m=map(int,input().split())
    arr = []
    for _ in range(n):
        s=input()
        arr.append(list(map(int,s.rstrip().split())))
k = int(input())
arr.sort(key=lambda arr:arr[k])
for i in range(n):

    print(*arr[i])


# # >> Any() or All()

# In[180]:


n=int(input())
inp=input().split()
c=False
s=False
for i in range(n):
    a=int(inp[i])>=0
    c=c+a
if c==n:
    for i in inp:
        k=i[::-1]==i
        s=s+k
if s>=1:
    print(True)
else:
    print(False)


# # >> Eval()

# In[1]:


eval(input())


# # >> gnortS

# In[30]:


from re import findall
s=input()
reg='[a-z]','[A-Z]','[13579]','[02468]'
sort=[sorted(findall(i,s)) for i in reg]
k=[]
for i in sort:
    k=k+i
print(*k,sep='')


# # >> Default arguments

# In[ ]:


class EvenStream(object):
    def __init__(self):
        self.current = 0

    def get_next(self):
        to_return = self.current
        self.current += 2
        return to_return

class OddStream(object):
    def __init__(self):
        self.current = 1

    def get_next(self):
        to_return = self.current
        self.current += 2
        return to_return

def print_from_stream(n, stream=None):
    stream = stream or EvenStream()
    for i in range(n):
        print(stream.get_next())


queries = int(input())
for _ in range(queries):
    stream_name, n = input().split()
    n = int(n)
    if stream_name == "even":
        print_from_stream(n)
    else:
        print_from_stream(n, OddStream())


# # >> Zipped

# In[128]:


student,subject=map(int,input().split())
l=[]
for i in range(subject):
    l.append(list(map(float,input().split())))
l=list(zip(*l)) 
for i in l:
    print(sum(i)/len(i))


# # >> Fibonaci map and lambda

# In[133]:


cube = lambda x:x**3 # complete the lambda function 

def fibonacci(n):
    # return a list of fibonacci numbers
    if n==0:
        l=[]
    elif n==1:
        l=[0]
    else:        
        l=[0,1]
        for i in range(1,n-1):
            l.append(l[i]+l[i-1])
    return l

if __name__ == '__main__':
    n = int(input())
    print(list(map(cube, fibonacci(n))))


# # >> XML 1-find score

# In[ ]:


import sys
import xml.etree.ElementTree as etree

def get_attr_number(node):
    # your code goes here
    return sum(map(get_attr_number, node)) + len(node.attrib)

if __name__ == '__main__':
    sys.stdin.readline()
    xml = sys.stdin.read()
    tree = etree.ElementTree(etree.fromstring(xml))
    root = tree.getroot()
    print(get_attr_number(root))


# # >> phone numbers-closure decorator

# In[ ]:


def wrapper(f):
    def fun(l):
        # complete the function
        f('+91 {} {}'.format(i[-10:-5], i[-5:]) for i in l)
        return fun
    return fun

@wrapper
def sort_phone(l):
    print(*sorted(l), sep='\n')

if __name__ == '__main__':
    l = [input() for _ in range(int(input()))]
    sort_phone(l)


# # >> Decorators 2-name directory

# In[ ]:


import operator
def person_lister(f):
    def inner(people):
        # complete the function
        return map(f, sorted(people, key=lambda x: int(x[2])))         
    return inner

@person_lister
def name_format(person):
    return ("Mr. " if person[3] == "M" else "Ms. ") + person[0] + " " + person[1]

if __name__ == '__main__':
    people = [input().split() for i in range(int(input()))]
    print(*name_format(people), sep='\n')


# # >> Kangaroo

# In[137]:


# Complete the kangaroo function below.
def kangaroo(x1, v1, x2, v2):
    if v1>v2 and (x1-x2)%(v2-v1)==0:
        return 'YES'
    else:
        return 'NO'    

if __name__ == '__main__':

    x1,v1,x2,v2 =map(int,input().split())
    result = kangaroo(x1, v1, x2, v2)
print(result)


# # >> strange viral ad

# In[158]:


# Complete the viralAdvertising function below.
def viralAdvertising(n):
    c=2
    s=5
    if n>1:
        for i in range(n-1):
            s=(s//2)*3
            m=s 
            c=(m//2)+c
        return c 
    
    else:
        return 2


# # >> Superdigit()

# In[11]:


# Complete the superDigit function below.
# My first code works but this is a second faster one which i searched for
def superDigit(n, k):
       return 1 + (k * sum(int(x) for x in str(n)) - 1) % 9

n,k=map(int,input().split())
print(superDigit(n,k))


# In[6]:


# Complete the superDigit function below.
# My first code
def superDigit(n, k):
    p=str(n)*k
    while len(p)>1:
        c=0
        for i in p:
            c=c+int(i)
            p=str(c)
    return p

n,k=map(int,input().split())
print(superDigit(n,k))


# # >> Insertion sort 1

# In[12]:


# Help from discussions
def insertionSort1(n, arr):
    i=n-1
    val=arr[i]
    while(i>0 and val<arr[i-1]):
        arr[i]=arr[i-1]
        print(*arr)
        i-=1
    arr[i]=val
    print(*arr)


# # >> Insertion sort 2

# In[13]:


# Help from dicussions
def insertionSort2(n, arr):
    for i in range(1,n):
        key = arr[i]
        j = i-1
        while j>=0 and key < arr[j]:
            arr[j+1] = arr[j]
            j -= 1
        arr[j+1] = key
        print(*arr)


# # >> Matrix script

# In[ ]:


# With help from discussions
from re import sub
N,M=map(int, input().split())
matr=''.join([''.join(t) for t in zip(*[input() for i in range(N)])])
print(sub(r'\b[^a-zA-Z0-9]+\b',r' ',matr))


# # >> input() Built-ins

# In[18]:


# With help from discussions
x,k= map(int,input().split())
exp=input()
print(eval(exp)==k)

