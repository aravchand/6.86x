s = "(0,0)	(2,0)	(1,1)	(0,2)	(3,3)	(4,1)	(5,2)	(1,4)	(4,4)	(5,5)"
# remove all parentheses from s with one regex 
import re
s = re.sub(r'[()]', '', s)
# split s into a list of strings
s = s.split()
# convert s to a list of tuples
s = [tuple(map(int, x.split(','))) for x in s]
# print(s)
print("\n".join(map(str, s)))
# apply quadratic feature map to s
import math

def quadratic_feature_map(x):
    return (x[0] ** 2, math.sqrt(2) * x[0] * x[1] , x[1] ** 2)

s = [quadratic_feature_map(x) for x in s]
print(s)

features = s

s = "1	65	11	31	72	30	0	21	4	15".split()
s = [int(x) for x in s]
a = s

label = "-1	-1	-1	-1	-1	+1	+1	+1	+1	+1"
label = label.split()
label = [int(x) for x in label]

theta = [0, 0, 0]
for i in range(len(features)):
    theta[0] += features[i][0] * label[i] * a[i]
    theta[1] += features[i][1] * label[i] * a[i]
    theta[2] += features[i][2] * label[i] * a[i]
print(theta)