#                                                                                                                       Assignment 4
#                                                                                                                   Munir Ahamed M, EE20b086

from cProfile import label
from pylab import *
from scipy import integrate

def exponent(x):                                                            #defining function to calculate exp(x)
    return exp(x)

def cosofcos(x):                                                            #defining function to calculate cos(cos(x))
    return (cos(cos(x)))

def u_exponent(x,k):                                                        #defining function to calculate f(x)*cos(kx)
    return (exp(x)*cos(k*x))                                                #and f(x)*sin(kx)

def v_exponent(x,k):
    return (exp(x)*sin(k*x))

def u_cosofcos(x,k):
    return (cos(cos(x))*cos(k*x))

def v_cosofcos(x,k):
    return (cos(cos(x))*sin(k*x))

a = linspace(-2*pi,4*pi,600)                                                #calculating function values and different
a_p = tile(linspace(0,2*pi,200), 3)                                         #intervals between -2pi to 4pi.
y1 = exponent(a)                                                                                                     
y1_p = exponent(a_p)                                                        #as exp(x) is not periodic, y1_p denotes it's
y2 = cosofcos(a)                                                            #periodic extension.

a_0 = integrate.quad(exponent, 0, 2*pi)[0]/(2*pi)
fn1 = [a_0]
a_0 = integrate.quad(cosofcos, 0, 2*pi)[0]/(2*pi)
fn2 = [a_0]

for k in range(1,26):                                                       #loading fn1, fn2 vectors with coefficients 
    u = integrate.quad(u_exponent, 0, 2*pi, args=(k))[0]/pi                 #of corresponding equations
    v = integrate.quad(v_exponent, 0, 2*pi, args=(k))[0]/pi
    fn1.extend([u,v])
    u = integrate.quad(u_cosofcos, 0, 2*pi, args=(k))[0]/pi
    v = integrate.quad(v_cosofcos, 0, 2*pi, args=(k))[0]/pi
    fn2.extend([u,v])

n = range(26)                                                               #seperating cos and sin coefficients and getting                   
u_fn1, v_fn1 = [abs(fn1[0])], [abs(fn1[0])]                                 #their absolute value
u_fn2, v_fn2 = [abs(fn2[0])], [abs(fn2[0])]

for i in range(1,26):                                                       #u_fn1 has cos coeff. v_fn1 has sin coeff.
    u_fn1.append(abs(fn1[(2*i)-1]))
    v_fn1.append(abs(fn1[2*i]))
    u_fn2.append(abs(fn2[(2*i)-1]))
    v_fn2.append(abs(fn2[2*i]))

A = empty((400,51))                                                         #creating empty A matrix to solve for c
x = linspace(0, 2*pi, 401)
x = x[:-1]

for i in range(400):
    A[i][0] = 1
    for k in range(1,26):
        A[i][2*k-1] = cos(k*x[i])
        A[i][2*k] = sin(k*x[i])

b1 = empty((400,1))
b2 = empty((400,1))

for k in range(400):                                                        #b matrix which contains function values
    b1[k][0] = exponent(x[k])
    b2[k][0] = cosofcos(x[k])

c1 = lstsq(A,b1,rcond=None)[0]                                              #obtaining coeff. by least squares
c2 = lstsq(A,b2,rcond=None)[0]
u_c1, v_c1 = [abs(c1[0])], [abs(c1[0])]                         
u_c2, v_c2 = [abs(c2[0])], [abs(c2[0])]

for i in range(1,26):                                                       #seperating coeff. as cos ans sin and 
    u_c1.append(abs(c1[2*i-1]))                                             #getting their absolite value
    v_c1.append(abs(c1[2*i]))
    u_c2.append(abs(c2[2*i-1]))
    v_c2.append(abs(c2[2*i]))

dev_exp = transpose(abs(fn1-transpose(c1)))                                 #finding deviations of coeff. obtained
devmax_exp = max(dev_exp)                                                   #by both methods and then finding max
dev_coc = transpose(abs(fn2-transpose(c2)))
devmax_coc = max(dev_coc)
f_pred1 = dot(A, c1)                                                        #predicted plot obtained using least squares coeff.
f_pred2 = dot(A, c2)                                                        

print("The maximum deviation of predicted coefficeints for exp(x) is: %.12f" % (devmax_exp) )
print("The maximum deviation of predicted coefficeints for cos(cos(x)) is: %.12f" % (devmax_coc) )

figure(1)                                                                   #plotting exp(x) function's true and predicted value
semilogy(a,y1, label='True value')
semilogy(a,y1_p, label='Periodic extension')
# semilogy(x,f_pred1, 'og', label='f(x) based on least squares approx.')
title("exp(x) semilog", size=18)
xlabel(r'$x\rightarrow$', size=18)
ylabel(r'$e^x\rightarrow$', size=18)
grid(True)
legend()

figure(2)                                                                   #plotting cos(cosx) function's true and predicted value
plot(a,y2, label='True value')
# plot(x,f_pred2, 'og', label='f(x) based on least squares approx.')
title("cos(cos(x))", size=18)
xlabel(r'$x\rightarrow$', size=18)
ylabel(r'$cos(cos(x))\rightarrow$', size=18)
grid(True)
legend()

figure(3)                                                                   #plotting fourier coefficients obtained by integration
semilogy(n, u_fn1, 'or', label='Coeff. based on integration')               #and least squares
semilogy(n, v_fn1, 'or')
semilogy(n, u_c1, 'og', label='Coeff. based on least squares')
semilogy(n, v_c1, 'og')
title("Coefficients of exp(x) semilog", size=18)
xlabel(r'$n\rightarrow$', size=18)
ylabel(r'$Coefficients\rightarrow$', size=18)
grid(True)
legend()

figure(4)
loglog(n, u_fn1, 'or', label='Coeff. based on integration')
loglog(n, v_fn1, 'or') 
loglog(n, u_c1, 'og', label='Coeff. based on least squares')
loglog(n, v_c1, 'og') 
title("Coefficients of exp(x) log-log", size=18)
xlabel(r'$n\rightarrow$', size=18)
ylabel(r'$Coefficients\rightarrow$', size=18)
grid(True)
legend()

figure(5)
semilogy(n, u_fn2, 'or', label='Coeff. based on integration')
semilogy(n, v_fn2, 'or')
semilogy(n, u_c2, 'og', label='Coeff. based on least squares')
semilogy(n, v_c2, 'og')
title("Coefficients of cos(cos(x)) semilog", size=18)
xlabel(r'$n\rightarrow$', size=18)
ylabel(r'$Coefficients\rightarrow$', size=18)
grid(True)
legend()

figure(6)
loglog(n, u_fn2, 'or', label='Coeff. based on integration')
loglog(n, v_fn2, 'or')
loglog(n, u_c2, 'og', label='Coeff. based on least squares')
loglog(n, v_c2, 'og')
title("Coefficients of cos(cos(x)) log-log", size=18)
xlabel(r'$n\rightarrow$', size=18)
ylabel(r'$Coefficients\rightarrow$', size=18)
grid(True)
legend()
show()

