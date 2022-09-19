import numpy as np
import math
import cmath
import numpy as np
import pandas as pd
import random
from scipy.integrate import quad


## IMPORTANT: FOR FULL RANGE INPUT
## Assume we have A/D converter with 6 bits, at its input we have full range sinosoid signal. Calculate SNR of a sinosoidal signal in dB
# bits=float(input("Enter bits: "))
# snr_db=(6.02*bits)+1.76
# print(snr_db)




# Assume we have signal x(n)=[6,6.2,1.1,8.3]. Calculate the autocorrelation function of this signal rxx(m) according to the 
# lecture definition and provide  rxx(0) in the answer field. 
# x=[6.1,2.7,8.9,4.7]
# auto_corr=0
# for v in x:
#     auto_corr+=(v)**2
# print("!!!USE CORRECT x SEQUENCE!")
# print(auto_corr)



# If we upsample by the factor of N= 5 . 
# How many zeroes we first include in the upsampled sequence? 
# print("N-1, here would be 4")




# # ************** Quantization *****************
# amin = float(input('enter min value'))
# amax = float(input('enter max value'))
# vin = float(input('enter input value'))
# n = float(input("enter no. of bits"))
# stepSize = (amax - amin) / (2**n)
# print('step size = ',stepSize)
# qindex = round(vin / stepSize)
# print('quantiztion index = ', qindex)
#
# # # ***** Mid Tread *****
# amin = float(input('enter min value'))
# amax = float(input('enter max value'))
# vin = float(input('enter input value'))
# n = float(input("enter no. of bits"))
# stepSize = (amax - amin) / (2 ** n)
# index= np.round(vin/stepSize)
# print('index', index)
# reconstr=index*stepSize
# print('reconst', reconstr)
# # #
# # ***** Mid Rise *****
# amin = float(input('enter min value'))
# amax = float(input('enter max value'))
# vin = float(input('enter input value'))
# n = float(input("enter no. of bits"))
# stepSize = (amax - amin) / (2 ** n)
# index= np.floor(vin/stepSize)
# print('index', index)
# reconstr=index*stepSize
# print('reconst', reconstr)


# **** roots ***
# b = [7.2, 2.6]
# a = [1, -6]
# roots_b = np.roots(b)
# roots_a = np.roots(a)
# print('roots of b', roots_b)
# print('roots of a', roots_a)
#




######### past paper starts from here ############


#Q:1
# Assume the pole of the allpass filter's transfer function is equal to z=9.3+8.5j. What will be the magnitude of the zero? 
# z = np.absolute(9.3+8.5j)
# zeros = 1 / z
# print(zeros)

#Q:2
# Assume we have a signal x(n)=[8.2;3.2;6.1;9.8]. Calculate the autocorrelation function of this signal rxx(m) according to the lecture definition and provide rxx(0) in the answer field.
# a = float(input('enter a value'))
# b = float(input('enter b value'))
# c = float(input('enter c value'))
# d = float(input('enter d value'))
# x = (a**2) + (b**2) + (c**2) + (d**2)
# print(x)

#Q:3
# Imagine you have the following imaginary number. z=7.9+5.8j, Calculate the magnitude of z
# z = np.absolute(7.9+5.8j)
# print(z)


#Q:4
# *** IMPORTANT: downsampling with removal of zeros
# Take the following signal: 3.7;4.5;6.9;3.1 and downsample it by a factor of 2 (phase 0), including the removal of zeros,
# What is the resulting z-transform, for z=2.3?
# a = float(input('enter a value'))
# b = float(input('enter b value'))
# c = float(input('enter c value'))
# d = float(input('enter d value'))
# z = float(input('enter z-transform value'))
# zTransform = a + (c/z)
# print(zTransform)

#Q:5
# You are given the following input sequence: 
# xv = [4,6,1,6,4,6]
# Use the running average filter h = [1/2, 1/2] for aliasing reduction and downsample it by the factor N = 2. 
# Start the downsampling from the first sample. 
# Calculate the output signal. 
# What will be the second sample of the output sequence? 
# x = [6,5,8,3,5,3]  
# print(x)
# print("!!!!!!!!!CHANGE X ACCORDING TO THE QUESTION AND THIS IS FOR THE SECOND SAMPLE OUTPUT OF THE SEQUENCE!!!!!")
# sample_1=0.5*x[0]
# sample_2=sample_1+(x[1]*0.5)
# sample_3=(sample_2 - sample_1)+(0.5*x[2])
# print(sample_3)

#Q:6
# *** IMPORTANT: downsampling with zeros
# We have the following signal sequence 3.9;3.2;9.4;2.1. Downsample it by a factor of 2 (phase 0), STILL KEEPING THE ZEROS IN IT, 
# and then apply the z-transform. What is the resulting z-transform for z=3.1?
# a = float(input('enter a value'))
# b = float(input('enter b value'))
# c = float(input('enter c value'))
# d = float(input('enter d value'))
# z = float(input('enter z value'))
# zTransform = a + (c/(z**2))
# print(zTransform)


#Q:7
# Assume you are given the following discrete signal, which has an amplitude x(0)=8 and x(1)=3. Apply the DTFT transform to calculate the
# magnitude of the spectrum, if w=3.1 radians
# a = float(input('enter x(0) value'))
# b = float(input('enter x(1) value'))
# w = float(input('enter w value'))
# magnitude = math.sqrt(((a+b*(math.cos(w)))**2)+((b*math.sin(w))**2))
# print(magnitude)

#Q:8
# Compute the weighted square error for the following obtained vectors:
# HDes=[5,2.2,9,4]
# H=[1.2,3.3,7.2,4.3]
# W=[1,1,100,100]
# !!!!!!!!
# print("Change the obtained vectors accordingly")
# HDes=[5,2.2,9,4]
# H=[1.2,3.3,7.2,4.3]
# W=[1,1,100,100]
# ans=0
# for i in range(4):
#     xxxx=((HDes[i] - H[i])**2) * W[i] 
#     ans+=xxxx
# print(ans)


#Q:9
#### IMPORTANT: SECOND CODEBOOK VECTOR
# Assume given training set vector x = [7.7; 9.9; 9.3; 3.3] and initial codebook vectors are y; = [6.7; 8.6] and yz = [9.6; 2.5]. Compute Euclidean
# distance between the first training set vector and the second codebook vector. (Here semicolon is used to distinguish between two consecutive
# numbers)

# x0 = float(input('enter x0 value'))
# x1 = float(input('enter x1 value'))
# x2 = float(input('enter x2 value'))
# x3 = float(input('enter x3 value'))
# y11 = float(input('enter y11 value'))
# y12 = float(input('enter y12 value'))
# y21 = float(input('enter y21 value'))
# y22 = float(input('enter y22 value'))
# #
# dif = math.sqrt( ((x0-y21)**2) + ((x1 - y22)**2))
# print(dif)



#Q:10
#### IMPORTANT: FIRST CODEBOOK VECTOR
# Assume given training set vector x = [1.2; 9.6; 9.3; 3.3] and initial codebook vectors are y1 = [8; 5.4] and y2 = [9.6; 2.5]. Compute Euclidean
# distance between the first training set vector and the first codebook vector. (Here semicolon is used to distinguish between two consecutive
# numbers)
#
# x0 = float(input('enter x0 value'))
# x1 = float(input('enter x1 value'))
# x2 = float(input('enter x2 value'))
# x3 = float(input('enter x3 value'))
# y11 = float(input('enter y11 value'))
# y12 = float(input('enter y12 value'))
# y21 = float(input('enter y21 value'))
# y22 = float(input('enter y22 alue'))
# #
# dif = math.sqrt( ((x0-y11)**2) + ((x1 - y12)**2))
# print(dif)


#Q:11  (Almost sure but also recheck if any solved has similar values if you have time)
# Suppose you have the following transfer function: H(z)=1/(8.7z**2+1.8z+9.2)
# Calculate the poles of this function. Decide if the filter who has this transfer function is stable or not.
# In the case it is stable enter -1 and otherwise +1.
# #
# a = float(input('Enter a: '))
# b = float(input('Enter b: '))
# c = float(input('Enter c: '))
# # calculate the discriminant
# d = (b ** 2) - (4 * a * c)
# # find two solutions
# sol1 = (-b - cmath.sqrt(d)) / (2 * a)
# sol2 = (-b + cmath.sqrt(d)) / (2 * a)
# print('The solution are {0} and {1}'.format(sol1, sol2))
# print("if real part is less than 1 then it is stable, carefully read the -1, +1 part of the question")


#Q:12
# Calculate the Hilbert transform of sin(w * n) for n=7 and w = 1.82. Provide the answer in radians.

# n = float(input('Enter n: '))
# w = float(input('Enter w: '))
# hilbert = - math.cos(n * w)
# print(hilbert)

#Q:13
# Assume that we have an input sequence x=[3.3; 7.8]. What will be the output of Hilbert transformer at n=1. Assume that pi=3.14
# in your calculation.
#
# x0 = float(input('Enter x0: '))
# x1 = float(input('Enter x1: '))
# n = float(input('Enter n: '))
# if n==1:
#     hilbert = x0 * (2/(np.pi*n))
# if n==2:
#     hilbert = x1 * (2 / (np.pi * n))
# print(hilbert)


#Q:14
## IMPORTANT: UNDERSTAND THE CODE (Done: y0 must always be one if same question)
# You are given the IIR filter with the following difference equation: y(n) = 7*x(n) +4.4*x(n—1)+7.3*y(n-1).
# Calculate the transfer function for z = 11.5.
# x0 = float(input('enter x0 value'))
# x1 = float(input('enter x1 value'))
# y0 = float(1)
# y1 = float(input('enter y1 value'))
# z = float(input('enter z value'))

# zTranform = (x0 + (x1/z)) / (y0 - (y1/z))
# print(zTranform)

#Q:15
# We want to design a Matched filter for the following input signal: 
# x(n)=[ 9.0; 5.7; 2.0; 6.3] 
# What is the first sample of the Matched filter's impulse response? 
# Remark: Numbers are separated by semicolon(;) 
# x = [5.0,9.3,8.8,6.3]
# print("Inverse the sequence of x, AND MAKE SURE THAT x has correct values as in the question of the exam.")
# print("Answer: ",x[-1]) 


#Q:16
# We have matrix A of size 2x2. The entries of the matrix are the following:
# Calculate the inverse of matrix A and provide a11 entry in the answer field
# a11=input("Enter a11: ")
# a12=input("Enter a12: ")
# a21=input("Enter a21: ")
# a22=input("Enter a22: ")
# b = np.matrix([[a11, a12],[a21, a22]], dtype=float)
# y = np.linalg.inv(b)
# print(y)


#Q:17
# Given signal sequence is x=[8.3;6.5;7.2;1.4;2.6], the prediction coefficients h(2)=[0;0] and u = 1. Compute h(3) (at time n=2). In the
# answer field provide the last coefficient of the impulse response.

# x0 = float(input('enter x0 value'))
# x1 = float(input('enter x1 value'))
# x2 = float(input('enter x2 value'))
# x3 = float(input('enter x3 value'))
# x4 = float(input('enter x4 value'))
# prediction = x0 * x2
# print(prediction)


#Q:18
# We want to design a Matched filter for the following input signal: 
# x(n)=[ 9.0; 5.7; 2.0; 6.3] 
# What is the first sample of the Matched filter's impulse response? 
# Remark: Numbers are separated by semicolon(;) 
# x = [3.3,8.6,5.8,1.7]
# print("Inverse the sequence of x, AND MAKE SURE THAT x has correct values as in the question of the exam.")
# print("Answer: ",x[-1]) 


#Q:19
# Given is the input signal:
# x(n)=[9.7;4.5;1.9;5.4;7.6;1.8]
# Find the output signal after filtering it with Matched Filter. In the answer field provide the first sample of the output signal.
# IMPORTANT: MATCHED FILTER
# x0 = float(input('enter x0 value'))
# x1 = float(input('enter x1 value'))
# x2 = float(input('enter x2 value'))
# x3 = float(input('enter x3 value'))
# x4 = float(input('enter x4 value'))
# x5 = float(input('enter x5 value'))
# matchfilter = x0 * x5
# print(matchfilter)


#Q:20 (Cross-check at the end with the solved solutions that are available)
# Determine the decision boundaries bk and reconstruction values yk after one iterations for a 2-bit Max-Lloyd quantizer of range 0 < x < 4
# with a signal with uniform distribution between 0 and 4
# y0= 1.7, y1=4.1, y2=5.5, y3=7.2
# b-1=
# b0=
# print("Waleed said that its always y0. My understanding is that whatever value comes in between 0 and 4 is the answer, so if y0 is between 0 and 4 then its the answer")
# print("In this case, the answer would be 1.70")


#Q:21
# You have a given signal: x(n)=[1.5,2.8,7.2,9.8]. 
# After you filter the shown signal with Matched filter what will be the maximum value of the output signal?
# x=[1.5,2.8,7.2,9.8]
# ans=0
# for val in x:
#     ans+=val**2
# print("Answer: ", np.round(ans, decimals=2))


#Q:22
# Given the position of zero z=6.6 outside the unit circle, to which position we need to move it so it will become the minimum phase
# system?
# x0 = float(input('enter z value'))
# print(1/x0)

#Q.23
# Assume you are given the following expression: (Integral cos(t) * § (t-3.8)dt)
# Calculate the numerical result of the expression. Result should be calculated in radians!
# print("answer: ",math.cos(3.8))

#Q:24
# Imagine we have a signal and we want to quantize it with May-Lloyd quantizer. Assume our current signal value is 
# x=4.9 and Euclidean distance to the first reconstruction value (y1) is 5.8 and to the second one (y2) is 19.4
# Reconstruction with which distance will be chosen?

# print("Select the nearest Euclidean distance from the current signal value")
# print("So in this case as signal value is 4.9, nearest value is 5.8 so answer is 5.80")


#Q:25
# Given is a sampling rate of 14.6 KHz. The audio signal should only contain frequencies smaller than what frequency (in Khz)
# to avoid overlapping of aliasing. 
# f=float(input("Enter main frequency: "))
# print(float(f/2))


#Q:26
### IMPORTANT: PROVIDES THE SECOND SAMPLE OF THE OUTPUT.
# Assume you are give the following difference equation of FIR filter:
# y(n)=0.351*x(n)+0.435*x(n-1)+0.576*x(n-2)+0.94*x(n-3)
# Calculate output of the filter if input signal is x=[1.5; 9.1; 7.1; 8.4]. In the answer field provide the second sample of the output
# signal.
# print("!!!!!!!MAKE SURE YOU ENTER THE CORRECT X(n) AND Y(n) VALUES!!!!!")
# x0 = float(input('enter x0 value'))
# x1 = float(input('enter x1 value'))
# x2 = float(input('enter x2 value'))
# x3 = float(input('enter x3 value'))
# y0 = float(input('enter y0 value'))
# y1 = float(input('enter y1 value'))
# y2 = float(input('enter y2 value'))
# y3 = float(input('enter y3 alue'))
# print((y0*x1)+ (y1*x0))



#Q:27
# Suppose you have the following filter impulse response h=[1.3;3.4;7.6;10.0]. Decompose the filter into polyphase components if N =2, 
# what will be the upsampled polyphase component 1 for z=4.3 (H1 (z**2))?

# x0 = float(input('enter h0 value'))
# x1 = float(input('enter h1 value'))
# x2 = float(input('enter h2 value'))
# x3 = float(input('enter h3 value'))
# z = float(input('enter z value'))
# print(x1 + (x3/(z**2)))


#Q:28
# The current autocorrelation coefficients are: rxx(0)=7.7, rxx(1)=7.1, rxx(2)=4.8. 
# Compute the prediction coefficients and enter the first coefficient of the resulting vector

# a11=float(input("Enter rxx(0): "))
# a12=float(input("Enter rxx(1): "))
# rx2=float(input("Enter rxx(2): "))
# b = np.matrix([[a11, a12],[a12, a11]], dtype=float)
# y = np.linalg.inv(b)
# ans=y*[[a12],[rx2]]
# print(ans)
# print(np.round(ans[0,0], decimals=4))


#Q:29  -- If you have time, cross check with solved solutions (past papers, quizzes etc)
# Imagine we have an A/D converter with an input range of -4V to 4V, 6 bit accuracy and we have 7.1 V at its input. What will be the reconstructed value?
# Use the mid-tread quantizer in your calculations
# amin = float(input('enter min value'))
# amax = float(input('enter max value'))
# vin = float(input('enter input value'))
# n = float(input("enter no. of bits"))
# stepSize = (amax - amin) / (2 ** n)
# index= np.round(vin/stepSize)
# print('index', index)
# reconstr=index*stepSize
# print('reconst', reconstr)

# MISSING QUESTION 30

#Q:31
# Assume we have a signal, 0.3*sin(t). Calculate the expected power of the given signal.
# a = float(input('enter amplitude value'))
# print((a**2)/2)




#Q:32
# What is the SNR (db) for a uniformly distributed signal with a full range of -2V to 2V using a uniform 5-bit quantizer?
# print("ROUND OFF UPTO 2 DECIMALS")
# a = float(input('enter no. of bit'))
# print("IF THE SAME QUESTION VALUES, ENTER: 30.10")
# print(6.02*a)




#Q:33
# Assume we have the first order allpass system with a real pole (r=5.2) and we are interested what will be the phase of this system if
# the normalized frequency equals to 0.1 radians.
# Provide the answer in radians.

# r = float(input('enter real pole value'))
# w = float(input('enter angle value'))
# s = r * math.sin(w)
# c = 1 - (r * math.cos(w))
# t = math.atan(s/c)
# print(-w-(2*t))


#Q:34
# You are given the following autocorrelation and cross-correlation coefficients of the output signal y(n) and input signal x(n):
# Calculate coefficients of the Wiener filter and provide the first coefficient in the answer field
# print("!!!!!!THIS IS FOR THE FIRST COEFFIFIENT; CHANGE ACCORDINGLY!!!!")
# a11=float(input("Enter ryy(0): "))
# a12=float(input("Enter ryy(1): "))
# rx1=float(input("Enter rxy(0): "))
# rx2=float(input("Enter rxy(1): "))
# b = np.matrix([[a11, a12],[a12, a11]], dtype=float)
# y = np.linalg.inv(b)
# ans=y*[[rx1],[rx2]]
# print(ans)
# print("first coefficient: ",np.round(ans[0,0], decimals=4))


#Q:35
# Suppose you have the following transfer function: X(z)= 1/(0.3z**2+9.8z,0.5)
# Calculate the poles of this function.In the answer provide the biggest pole.

# a = float(input('Enter a: '))
# b = float(input('Enter b: '))
# c = float(input('Enter c: '))
# # calculate the discriminant
# d = (b ** 2) - (4 * a * c)
# # find two solutions
# sol1 = (-b - cmath.sqrt(d)) / (2 * a)
# sol2 = (-b + cmath.sqrt(d)) / (2 * a)
# print('The solution are {0} and {1}'.format(sol1, sol2))
# print("smallest magnitude of the real number is biggest pole. answer must include the sign as well e.g if -32 and -0.05, -0.05 would be the answer")


#Q:36
# You are given the following input sequence:  x = [9;8;2;5] 
# Upsample it by the factor N=2 and use the linear interpolating filter: h = [1/2, 1, 1/2]
# Calculate the output signal.
# What will be the third sample of the output sequence?
# x = [2,5,6,3] 
# print(x)
# print("!!!!!!!!!CHANGE X ACCORDING TO THE QUESTION AND PROVIDES 3rd Sample!!!")
# sample_1=0.5*x[0]
# sample_2=sample_1+(1*0)
# sample_3=sample_2+(0.5*x[1])
# print(sample_3)


#Q:37
#You are given the following sequence: x = [8.5;5.8; 3.6; 3.3]. Compute the Z-transform for z=1.8. 
# x = [1.4,3.5,6.2,3.1]
# z=float(input("Enter z variable: "))
# print("!!!!!CHANGE VALUES ACCORDING TO THE QUESTION!")
# z_transform=(x[0] + (x[1]/z) + (x[2]/(z**2)) + (x[3]/(z**3)))
# print(z_transform)
    

#Q:38
# If we sample a signal by a factor of 8 with keeping the zeros between the sample points, then how many aliasing components will we have?
# factor=int(input("Enter the factor number: "))
# print("ans(N-1)=", factor-1)


#Q:39
# Imagine we have the following signal to quantize x=[10,15,7,8,9,5] and inital codebook vectors are y1=[4,5], y2=[10,11] with M=N=2,
# Compute euclidean distances between each training set vector and codebook vector.
# print("Formula for the calculation: whole sqrt((x1-y1)²+(x2-y2)²")

# x1=[9,5]
# y1=[10,11]
# print("Keep on changing the x1 and y1 vectors for different values")
# print("Ans",np.round(np.sqrt((x1[0]-y1[0])**2 + (x1[1]-y1[1])**2), decimals=2))


#Q:40
#Determine the decision boundaries bk and reconstruction values yk after one iterations for a 2-bit Max-Lloyd quantizer
# of range 0 < x < 6 with a signal with uniform distribution between 0 and 6.

# print("Half soln is done on the register and half below, follow the steps on the register")
# # "The following code is for the calculations of y"
# lower_limit=0.5
# upper_limit=6  #get from b
# Num,Nerr=quad(lambda x: x,lower_limit,upper_limit)
# den,_Nerr=quad(lambda x: 1,lower_limit,upper_limit)
# print(Num/den)

#Q:41
#Theory
# This picture shows a 2-dimensional Vector Quantizer.
# The red stars are the 'codevectors'
# The green dots are the 'signal samples'
# The blue lines are the 'boundaries of the Voronoi regions'


#Q:42
# Assume we want to use vector quantization and quantize the following signal x=[7, 9, 0, 40, 15.5, 10] with N=3. How the 
# resulting vectors of the training set will look like? 
# x=[7, 9, 0, 40, 15.5, 10]
# x1=x[:3]
# x2=x[3:]
# print("!!!!!!!!!Change x according to the question and note that N=3")
# print("x1:", x1)
# print("x2: ", x2)

#Q:43
#Theory
# The magnitude of the 'allpass' filter's transfer function is equal to '1'.

#Q:44
#Theory
# If you transform a signal from the time domain to the frequency domain with the DTFT, the spectrum of the signal will be 'periodical' and 'continuous'.

#Q:45
#Theory
# Max-Lloyd algorithm is the 'iterative'  type of algorithm, which is based on the 'nearest' neighbor principle

#Q:46
#Theory
# The 'mid-rise' quantizer can be seen as more accurate, because it also reacts to very 'small' input values,
# and the 'mid-tread' quantizer can be seen as saving bit-rate because it always quantizes very small values to '0'





# #
# b = np.array([[6.3, 4.2],[4.2, 6.3]])
# y = np.linalg.inv(b)
#
# x = np.matrix('6.3; 4.4')
#
# Result = y * x
# print(Result)