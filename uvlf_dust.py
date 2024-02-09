#%%
import matplotlib.pyplot as plt
import scipy.integrate as spy
import numpy as np
from hmf import MassFunction
import sympy as sy
from scipy.optimize import fsolve
from scipy import optimize
import sympy as sym

#%%
# Constants from PLANCK collaboration
omega_m = 0.3075
omega_l = 1-omega_m
omega_0 = omega_m + omega_l
omega_b = 0.0486
n = 0.9667
h = 0.6774
M_od = 1.989 * 1e30 #kg
H_0 = 6.93e-11 #/yr
rho0 = 1.46*10**11*h # units of h

def find_nearest(array, value):                                   # Returns index of the array element with value closest to value
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

#%%
# Input redshift here
z = 7 #int(input("input redshift:\n"))

#%%
# Calling the HMF as Sheth and Tormen ellipsoidal pertrubation model - Library by Steven Murray
mf = MassFunction(z=z,transfer_model="BBKS", hmf_model="ST",Mmin = 10, Mmax = 14,dlog10m= 0.01)


# Defining mass so that its in units of M_sun only
m = mf.m*h


# HMF getting updated at every redshift as a function of mass and redshift
def hmf1(m, z):
	mf.update(z=z)
	return np.interp(m, mf.m, mf.dndm)  


# Constants required for the model as described by Ferrara et al 2022 https://arxiv.org/pdf/2208.00720.pdf
zeta = 0.06
e_0 = 0.02
f_w = 0.1
v_s = 975 # km/s


# Halo circular velocity from Pratika et al Review https://arxiv.org/pdf/1809.09136.pdf (in km/s)
def v_c(m,z):
    	return (23.4 * (m/(1e8))**(1.0/3.0) * ((1+z)/10)**(1.0/2.0))
    	
    
# Star Formation Efficiency including the feedback given by Pratika et al https://arxiv.org/pdf/1405.4862.pdf
def e_s(m,z):
    	return e_0 * (v_c(m,z)**2.0) / (v_c(m,z)**2.0 + (f_w * v_s**2.0))
    
    
# Star Formation Rate: Essentially Stellar Mass divided by total cosmic time but simplified in M_sun/yr
def sfr(m,z):
    	return 22.7 * (e_s(m,z)/0.01) * ((1+z)/8.0)**(3.0/2.0) * m/1e12
   
   
# Model: model for effective optical depth from Ferrara et al 2022  
if z >= 10:   
	def tau_eff(m,z):
    		return 0.7 + 0.0164 * (sfr(m,z)/10)**(1.45)
def tau_eff(m,z):
	if z <=10:
		tef = 0.7 + 0.0164 * (sfr(m,z)/10)**(1.45)
	else:
		tef = 0
	return tef
    
    
Kuv = 0.587e10 
def luv1(m,z):
    	return sfr(m,z) * Kuv
    

# Effect of dust on Apparent Magnitude depending on the effective optical depth: Ferrara et al 2022
def muv(m,z):
    	return -2.5 * np.log10(luv1(m,z)) + 5.89 + 1.087 * tau_eff(m,z)

def dmuvdM(m,z):
	a = 0.000000001*m
	return np.abs((-muv(m-3*a,z)+9*muv(m-2*a,z) - 45*muv(m-a,h) + 45*muv(m+a,h) - 9*muv(m+2*a,z) +muv(m+3*a,z))/(60*a))
    	
    	
# No dust M_UV
def muv1(m,z):
    	return -2.5 * np.log10(luv1(m,z)) + 5.89
    
def dmuvdM1(m,z):
	a = 0.000000001*m
	return np.abs((-muv1(m-3*a,z)+9*muv1(m-2*a,z) - 45*muv1(m-a,z) + 45*muv1(m+a,z) - 9*muv1(m+2*a,z) +muv1(m+3*a,z))/(60*a))
    
# Phi_uv using Simple Chain rule as done by Ferrara et al 2022
def phi_uv(m,z):
	phiuv = hmf1(m,z)/(dmuvdM(m,z))/h**4
	return phiuv
	
def phi_uvnd(m,z):
    	return hmf1(m,z)/(dmuvdM1(m,z))/h**4

plt.plot(muv(m,z),dmuvdM(m,z))
plt.show()
    	
#%%
# Chain rule does not works in a simple manner when transformation is non monotonic in nature, we use a different chain rule, for calculating that
# Plotting phi_luv as a function of Luv

def dluvdM(m,z):
	a = 0.0000001*m
	return np.abs((-luv1(m-3*a,z)+9*luv1(m-2*a,z) - 45*luv1(m-a,z) + 45*luv1(m+a,z) - 9*luv1(m+2*a,z) +luv1(m+3*a,z))/(60*a))

def uvlf_l(m,z):
    	return hmf1(m,z)/np.abs(dluvdM(m,z))/h**4
    	
def fl(Luvs,z):
	return np.interp(Luvs, luv1(m,z), uvlf_l(m,z))
	
# Writing phi_Muv as function of phi_Luv    	
# Calculating the roots
A = 5.89 + 1.087*0.7
B = 1.087 * 0.0164/(10*Kuv)**(1.45) 

Luv = luv1(m,z)
def fMuv(Luv):
    	return -2.5 * np.log10(Luv) + B * Luv ** 1.45 + A  
    	
def f(log10_Luv, Muv):
	return -2.5 * log10_Luv + B * (10**log10_Luv) ** 1.45 + A - Muv
	

Muvs = muv(m,z)
Luv_sol = [10 ** fsolve(f, [3, 13], args = (Muv,)) for Muv in Muvs]
#print(Luv_sol)
	
#plt.plot(luv1(m,z), fMuv(Luv))
sol1 = np.array([i[0] for i in Luv_sol])
sol2 = np.array([i[1] for i in Luv_sol])
#plt.plot(sol1, Muvs, color = "red")
#plt.plot(sol2, Muvs, color = "orange")
#plt.grid()
#plt.xscale("log")
#plt.minorticks_on()
#plt.show()


#plt.plot(luv1(m,z), fMuv(Luv))
#plt.grid()
#plt.xscale("log")
#plt.xlabel("$L_{UV}$ $[L_{\odot}]$",fontsize=15)
#plt.ylabel("$M_{UV}$",fontsize=15)
#plt.minorticks_on()
#plt.show()

def dfMuv(Luv):
	a = 1e-8*Luv
	return np.abs((-fMuv(Luv-3*a)+9*fMuv(Luv-2*a) - 45*fMuv(Luv-a) + 45*fMuv(Luv+a) - 9*fMuv(Luv+2*a) +fMuv(Luv+3*a))/(60*a))

# correct phi_Muv!!
phi_Muv = (fl(sol1,z)/dfMuv(sol1)) + (fl(sol2,z)/dfMuv(sol2))

#%%
# Bouwens 20 at z=7 data    	
data = np.loadtxt("Bouwens_20_UVLF_z7.txt",delimiter = ",")
a = data[:,0]
b = data[:,1]
bu = data[:,2:3].T

# Bouwens 20 at z=6 data
dataa = np.loadtxt("Bouwens_20_UVLF_z6.txt",delimiter = ",")
aa = dataa[:,0]

bb = dataa[:,1]
bbu = dataa[:,2:3].T

#z = 13.5
data99 = np.loadtxt("z13.5.txt",delimiter = ",")
a99 = data99[:,0]
b99 = data99[:,1]
bu99 = data99[:,2:3].T
au99 = data99[:,3:4].T


# Plotting UV Luminosity     
f = plt.figure(figsize=(7,7))
ax = f.add_subplot(111)

plt.plot(muv(m,z),np.log10(phi_uv(m,z)),color="purple",linewidth = 2.0)
plt.plot(muv1(m,z),np.log10(phi_uvnd(m,z)),color="purple", linewidth = 2.0)
#plt.plot(fMuv(Luv),np.log10(phi_Muv),color="orange",label="New Chain Rule")

if z == 14:
        # Data points from Maisie's galaxy Finkelstein22
        MUV_obs =np.array([-20.3])
        lg_Phi_obs = np.array([-5.12])
        errp_lg_Phi_obs = np.array([0.35])
        errm_lg_Phi_obs = np.array([0.57])
        err= np.array(list(zip(errm_lg_Phi_obs, errp_lg_Phi_obs))).T
        plt.errorbar(MUV_obs, lg_Phi_obs, yerr=err, capsize=3, ls='none',ecolor='black', elinewidth=0.7)
        plt.scatter(MUV_obs, lg_Phi_obs,color='navy', marker='o', edgecolors='k', s=180, alpha=1.0)

        #plt.text(-19,-4.3, 'Bowler+20',  color='gold',fontsize=16, fontstyle='normal', rotation=0)
        #plt.text(-18.5,-5., 'Bouwens+21',  color='brown',fontsize=16, fontstyle='normal', rotation=0)
        #plt.text(MUV_obs-1, lg_Phi_obs+.5,'Maisie\'s Galaxy', color='navy', fontsize=12, fontstyle='normal')

if z == 13.25:
		MUV_obs = np.array([-19.1])
		lg_Phi_obs = np.array([-4.77])
		errp_lg_Phi_obs = np.array([0.34])
		errm_lg_Phi_obs = np.array([0.53])
		err= np.array(list(zip(errm_lg_Phi_obs, errp_lg_Phi_obs))).T
		plt.errorbar(MUV_obs, lg_Phi_obs, yerr=err, capsize=3, ls='none',ecolor='black', elinewidth=0.7)
		plt.scatter(MUV_obs, lg_Phi_obs,color='navy', marker='o', edgecolors='k', s=180, alpha=1.0)

		MUV_obs = np.array([-20.35])
		lg_Phi_obs = np.array([-5.8])
		errp_lg_Phi_obs = np.array([0.32])
		errm_lg_Phi_obs = np.array([0.54])
		err= np.array(list(zip(errm_lg_Phi_obs, errp_lg_Phi_obs))).T
		plt.errorbar(MUV_obs, lg_Phi_obs, yerr=err, capsize=3, ls='none',ecolor='black', elinewidth=0.7)
		plt.scatter(MUV_obs, lg_Phi_obs,color='navy', marker='o', edgecolors='k', s=180, alpha=1.0)

if z == 11.5:
        # Data points from Naidu+22 at z=10-13: purple point in Fig 5 of https://arxiv.org/pdf/2207.09434.pdf
        MUV_obs =np.array([-21.0])
        lg_Phi_obs = np.array([-5.05])
        errp_lg_Phi_obs = np.array([0.37])
        errm_lg_Phi_obs = np.array([0.45])
        err= np.array(list(zip(errm_lg_Phi_obs, errp_lg_Phi_obs))).T
        plt.errorbar(MUV_obs, lg_Phi_obs, yerr=err, capsize=3, ls='none',ecolor='black', elinewidth=0.7)
        plt.scatter(MUV_obs, lg_Phi_obs,color='navy', marker='o', edgecolors='k', s=180, alpha=1.0)

        # Data points for Gz11 from N aidu+22: gold point in Fig 5 of https://arxiv.org/pdf/2207.09434.pdf
        MUV_obs =np.array([-22.1])
        lg_Phi_obs = np.array([-6.1])
        errp_lg_Phi_obs = np.array([0.5])
        errm_lg_Phi_obs = np.array([0.8])
        err= np.array(list(zip(errm_lg_Phi_obs, errp_lg_Phi_obs))).T
        errm_lg_MUVobs = np.array([0.2])
        plt.errorbar(MUV_obs, lg_Phi_obs, yerr=err, capsize=3, ls='none',ecolor='black', elinewidth=0.7)
        plt.scatter(MUV_obs, lg_Phi_obs,color='purple', marker='o', edgecolors='k', s=180, alpha=1.0)

        #plt.text(-19,-3.4, 'Bowler+20',  color='gold',fontsize=16, fontstyle='normal', rotation=37)
        #plt.text(-19,-4.1, 'Bouwens+21',  color='brown',fontsize=16, fontstyle='normal', rotation=37)
	
# if z == 7:
#         # Data points from Bouwens+21 of the LF at z=7
#         MUV_obs =np.array([-22.19, -21.69, -21.19, -20.69, -20.19, -19.69, -19.19, -18.69, -17.94, -16.94])
#         LUV_obs = 10.**((89.85 - MUV_obs)/2.5)
#         Phi_obs = 1.0e-6*np.array([1., 41., 47., 198., 283., 589., 1172., 1433., 5760., 8320.])
#         plt.scatter(MUV_obs, np.log10(Phi_obs),color='brown', marker='s', edgecolors='k', s=150, alpha=0.7)
#         plt.text(-23.3,-7.4, 'Bouwens+21',  color='brown',fontsize=16, fontstyle='normal', rotation=78)
#         plt.text(-24.2,-7.2, 'Bowler+20',  color='gold',fontsize=16, fontstyle='normal', rotation=60)



#         # REB 25 UV Magnitude, positioned manually on the model LF density value
#         REB25_MUV_obs =np.array([-21.7])
#         REB25_err_MUV_obs = np.array([0.2])
#         idx=find_nearest(muv(m,z), REB25_MUV_obs)        # Finding idx does not work as LF is bi-valued!! By hand -7.7   
#         plt.errorbar(REB25_MUV_obs, -7.7, xerr=REB25_err_MUV_obs, capsize=3, ls='none',ecolor='black', elinewidth=0.7)
#         plt.scatter(REB25_MUV_obs, -7.7, color='pink', marker='*', edgecolors='k', s=400, alpha=1.0)
#         plt.text(-21.4, -7.7, 'REBELS-25',  color='black',fontsize=16, fontstyle='normal')

#         # SFR signposts (computed by hand, bi-valued function is a problem to find index) 
#         post= np.array([[1,-17.5,-2.3],[5,-19.5,-3.3],[25, -20.9, -4.4], [50, -21.5, -5.1], [100, -22.1, -6.0], [200,-22.3, -7.1], [400, -20.5, -8.6]])
#         for i in range(0,6):
#             plt.scatter(post[i,1], post[i,2] , color='blue', marker='o', edgecolors='k', s=50, alpha=1.0)
#             plt.text(post[i,1]+0.2, post[i,2]-0.02, str(int(post[i,0])),  color='blue',fontsize=12, fontstyle='normal')
        # REB 29-2 & 12-2 (Fudamoto): calculation on brown notebook
        # plt.arrow(-22.18, np.log10(2e-5), 0.5, 0, width = 0.01, head_width=0.08)                       # REB 29-2
        # plt.arrow(-22.18, np.log10(2e-5), 0, -0.5, width = 0.01, head_width=0.08)                      
        # plt.scatter(-22.18, np.log10(2e-5), color='k', marker='*', edgecolors='k', s=100, alpha=1.0)
        # #plt.text(-22.18-0.3, np.log10(2e-5)+0.1, 'REBELS 29-2',  color='k',fontsize=10, fontstyle='normal', rotation=90)
        # plt.arrow(-23.73, np.log10(2e-5), 0.5, 0, width = 0.01, head_width=0.08)                       # REB 12-2
        # plt.scatter(-23.73, np.log10(2e-5), color='k', marker='*', edgecolors='k', s=100, alpha=1.0)
        # #plt.text(-23.73-0.3, np.log10(2e-5)+0.1, 'REBELS 12-2',  color='k',fontsize=10, fontstyle='normal', rotation=90)
        
# # Observed Luminosity Function at different z: Bowler+20 MNRAS 493, 2059 (2020)
# # Double power-law fit to the data
# MUVstar = -21.03+0.49*(z- 6),
# Phistar =  10**-3.52                       # in Mpc^-3
# alpha   = -1.99-0.09*(z - 6)
# beta    = -4.92+0.45*(z - 6)
# # https://www.astro.umd.edu/~richard/ASTRO620/LumFunction-pp.pdf
# MUV_var = np.arange(-30., -10, 0.1)
# pwr = (MUV_var-MUVstar)/2.5
# Phi = np.log(10.)/2.5*Phistar/(10**((alpha+1.)*pwr) + 10**((beta+1.)*pwr))   
# lg_Phi = np.log10(Phi)
# #plt.plot(MUV_var, lg_Phi, '--', dashes=[30,5], color='gold', linewidth=1.0, label='Bowler+20')

# Points for z=7
if z==7:
	plt.errorbar(a,b,yerr = bu, fmt="-o", capsize=5,ls="none",linewidth = 2.0,color = "brown")

# Points for z=6
if z==6:
	plt.errorbar(aa,bb,yerr = bbu, fmt="-o", capsize=5,ls="none",color="brown",linewidth = 2.0)
	
# axis labelling
plt.xlabel("$\mathrm{M_{uv}}$",fontsize="18")
ax.yaxis.set_ticks_position('both')
ax.xaxis.set_ticks_position('both')
plt.ylabel("$\log \Phi$ [\mag \Mp$c^3$]",fontsize="18")

# Legend
# if z==7:
# 	plt.legend(["Model z=7","No dust z = 7","Bouwens+21 z=7","SFR"],fontsize="18")
plt.legend(["UV LF at z = 13.25","Donnan + 22"],fontsize="12",loc="upper left")

if z==6:
	plt.legend(["UV LF at z = 6","no dust","Bouwens + 21"],fontsize="12",loc="upper left")
	

#plt.minorticks_on()
#plt.grid("minor")
plt.axis([-23,-18,-9,-1])
ax.yaxis.set_ticks(np.arange(-9,-1,1))
ax.tick_params(which='both', direction='in')
plt.tick_params(which='both', length=10, width=1, colors='k', labelsize='18', pad=10)
plt.show()



#%%
# Obscured fraction
def fobs(m,z):
    	return (1-np.exp(-tau_eff(m,z)))
    	
def sfr_ir(m,z):
	return (fobs(m,z))*sfr(m,z)
    
# IR Luminosity- Calculated straight from IR Luminosity Function
def lir(m,z):
    	return sfr(m,z)*1.2e10#*(fobs(m,z))

def lir2(m,z):
		return sfr(m,z)*1.2e10
    	
def lir1(m,z):
	return fobs(m,z)*luv1(m,z)
	
# plt.plot(np.log10(lir(m,z)),np.log10(lir1(m,z)))
# plt.grid()
# plt.minorticks_on()
# # plt.show()

	
def dlirdM(m,z):
	a = 0.000001*m
	return np.abs((lir(m+a,z)-lir(m-a,z))/(2*a))
	
def dlir1dM(m,z):
	a = 0.000001*m
	return np.abs((lir1(m+a,z)-lir1(m-a,z))/(2*a))
	
def phi_ir(m,z):
	return lir(m,z)*hmf1(m,z)/(dlirdM(m,z))/h**4       #phi_uv(m,z)* lir1(m,z) * dmuvdM(m,z)/dluvdM(m,z) * dluvdM(m,z)/dlir1dM(m,z)
	

def f_ir(Lir,z):
	return np.interp(Lir, lir(m,z), phi_ir(m,z))
    

def phi_ir1(m,z):
	return lir1(m,z)*hmf1(m,z)/(dlir1dM(m,z))/h**4 
	
def phi_ir2(m,z):
	return phi_Muv* lir(m,z) * dmuvdM(m,z)/dluvdM(m,z) * dluvdM(m,z)/dlirdM(m,z)


#for i,j in zip(phi_ir1(m,z),phi_ir(m,z)):
	#print(i,j)
    
data1 = np.loadtxt("Pratika22IR.txt",delimiter = ",")
a1 = data1[:,0]
b1 = data1[:,1]

data2 = np.loadtxt("SchecterIR.txt",delimiter = ",")
a2 = data2[:,0]
b2 = data2[:,1]

data3 = np.loadtxt("REBELSirlf.txt", delimiter = ",")
a3 = data3[:,0]
b3 = data3[:,1]
b3u = data3[:,2:3].T
    
data4 = np.loadtxt("SHARK.txt", delimiter =",")
a4 = data4[:,0]
b4 = data4[:,1]

data5 = np.loadtxt("REBELdetection.txt",delimiter = ",")
a5 = data5[:,0]
b5 = data5[:,1]
b5u = data5[:,2:3].T

# z = 6 data
data6 = np.loadtxt("Gruppioni_20_IR_Z6.txt",delimiter = ",")
a6 = data6[:,0]
b6 = data6[:,1]
a6u = data6[:,2:3].T
b6u = data6[:,4:5].T

data7 = np.loadtxt("TNG100600_model.txt",delimiter = ",")
a7 = data7[:,0]
b7 = data7[:,1]

data11 = np.loadtxt("Yan_20_IR_z6.txt",delimiter = ",")
a11 = data11[:,0]
b11 = data11[:,1]
b11u = data11[:,2:3].T


# z = 4 data
data8 = np.loadtxt("Gruppioni20_IR_z4.txt", delimiter = ",")
a8 = data8[:,0]
b8 = data8[:,1]
a8u = data8[:,2:3].T
b8u = data8[:,4:5].T

data9 = np.loadtxt("TNG50.txt",delimiter = ",")
a9 = data9[:,0]
b9 = data9[:,1]

data10 = np.loadtxt("Yan_20_IR_z4.txt",delimiter = ",")
a10 = data10[:,0]
b10 = data10[:,1]
b10u = data10[:,2:3].T

data15 = np.loadtxt("schechter_ir_5.5.txt",delimiter = ",")
b15 = data15[:,0]
a15 = data15[:,1]


data16 = np.loadtxt("schechter_ir_4.txt",delimiter = ",")
b16 = data16[:,0]
a16 = data16[:,1]

#%%
#read csv files
datan = np.loadtxt("Prajna_data.txt",delimiter = ",")
# Generating some random data
xs = datan[:,3]
ys = datan[:,0]
ys1 = datan[:,1:2].T
colors = datan[:,4]

ys2 = np.polyfit(xs,ys,1)
print(ys2)

xs2 = np.linspace(-24,-18)
ys3 =  ys2[1] + ys2[0]*xs2 
#plt.plot(xs2,ys3,color="red",linewidth =2.0)
plt.figure(figsize=(10,8))
# Creating a scatter plot with different colors with error bars on y-axis of ys1
#plt.scatter(xs, ys, c=colors)


# plot points with colorbar and errorbar
sct = plt.scatter(xs, ys, c=colors, cmap='viridis',s=50, zorder = 1)
_, __ , errorlinecollection = plt.errorbar(xs, ys,yerr = ys1, ls = '', zorder = 0,ecolor='k',capsize=2,elinewidth=0.7)
# Adding a color bar
plt.colorbar(sct)

plt.plot(muv1(m,z),np.log10(lir(m,z)),color="purple",linewidth =2.0)
plt.plot(muv1(m,z),np.log10(lir2(m,z)),color="green",linewidth =2.0,ls = '--')
plt.plot(muv1(m,z),np.log10(lir1(m,z)),color="blue",linewidth =2.0,ls = '--')
plt.xlabel(r'$M_{UV}$', fontsize=18)
plt.ylabel(r'$L_{IR}$', fontsize=18)
# plot a vertical line at x = -22.3
plt.axvline(x=-22.3, color='grey', linestyle='-.')
# plot a horizontal line at y = 11.7
plt.axhline(y=11.66, color='grey', linestyle='-.')
plt.axis([-23.5,-20.5,11,12.5])
plt.tick_params(which='both', length=10, width=1, colors='k', labelsize='18', pad=10)
plt.legend(["REBELS Det","Relation-1","Relation-2","Relation-3"],loc='upper right',fontsize=18)
plt.show()

#%%
f1 = plt.figure(figsize=(7,7))
ax1 = f1.add_subplot(111)
ax1.yaxis.set_ticks_position('both')
ax1.xaxis.set_ticks_position('both')
#plotting IR-LF
plt.plot(np.log10(lir(m,z)),(np.log10(phi_ir(m,z))),color="purple",linewidth =2.0)
#plt.plot(np.log10(lir(m,z)),(np.log10(f_ir(Lir,z))),color="blue")
#plt.plot(np.log10(lir1(m,z)),(np.log10(phi_ir1(m,z))),color="orange")
#plt.plot(np.log10(lir(m,z)),(np.log10(phi_ir2(m,z))),color="green")
# z = 7 data points
if z == 7:
	plt.plot(a1,b1,ls="--",color = "violet",linewidth = 2.0)
	plt.plot(a2,b2,linewidth = 2.0,color = "green")
	plt.errorbar(a3,b3,yerr = b3u, fmt="-o", capsize=5,ls="none",color = "green",linewidth = 2.0)
	plt.plot(a4,b4,ls="-.",color = "tomato",linewidth = 2.0)
	plt.errorbar(a5,b5,yerr = b5u, fmt="-o", capsize=5,ls="none",color="brown", linewidth = 2.0)
	plt.legend(["This Work z=7","Dayal +22 z = 7","Schecter Function", "SHARK Lagos z = 7.1", "REBELS Total Sample","REBELS detections"],fontsize="12",loc="upper right")
	# plot vertical line at x = 11.7
	plt.axvline(x=11.7, color='k', linestyle='--')
	plt.axis([10.5,12.5,-6.5,-2])
	plt.grid()
	#plt.arrow(11.5,-7.5,0.5,0.5,head_width=0.1, head_length=0.1, fc='k', ec='k',text = "This work")
#z = 6 data points

if z == 5.5:
	plt.errorbar(a6,b6,yerr = a6u, fmt="-o", capsize=5,ls="none",linewidth = 1.5,color = "green")
	plt.plot(a7,b7,color="orange",linewidth = 2.0,ls="--")
	plt.errorbar(a11,b11,yerr = b11u, fmt="-o", capsize=5,ls="none",linewidth = 2.0,color ="brown")
	plt.plot(a15,b15,color="green",linewidth = 2.0)
	plt.legend(["This Work z = 5.5","TNG100-TNG600 (z=6)","Schechter 5.5","Gruppioni+20 z=4.5-6.0", "Yan+20 z=5.5"],fontsize="12")
	plt.axis([10,12.75,-7,-1.5])
	
# z = 4 data points
if z == 4:
	plt.errorbar(a8,b8,yerr = a8u, fmt="-o", capsize=5,ls="none",linewidth = 2.0,color="green")
	plt.errorbar(a10,b10,yerr = b10u, fmt="-o", capsize=5,ls="none",linewidth = 2.0,color = "brown")
	plt.plot(a9,b9,color="orange",linewidth = 2.,ls = "--")
	plt.plot(a16,b16,color = "green",linewidth = 2.0)
	plt.legend(["This Work z = 4","TNG+50+100+300 (z=4)","Schechter 4","Gruppioni+20 z=3.5-4.5","Yan+20 z =4.5"],fontsize="12",loc = "upper right")
	plt.axis([10,13,-7,-1.25])


plt.xlabel("$\mathrm{L_{ir}}$ [$\mathrm{L_{\odot}}$]",fontsize="20")
plt.ylabel("$\log\Phi\: \mathrm{[Mpc^{-3} dex^{-1}]}$",fontsize="20")
plt.tick_params(which='both', length=10, width=1, colors='k', labelsize='18', pad=10)
# plt.xlim(10.5,12.5)
# plt.ylim(-8,-2.5)
#plt.minorticks_on()
#plt.axis([12,13,-60,0])

plt.show()

#%%
# at z=7: Defining phi_star , number of missing galaxies from our theory
from scipy.interpolate import make_interp_spline, BSpline
Lir = lir(m,z)
def f_ir(Lir,z):
	return np.interp(Lir, lir(m,z), phi_ir(m,z))
	
fir = f_ir(Lir,z)
	
f_ir_range = fir[(np.log10(Lir) >= a2[8]) & (np.log10(Lir) <= a2[21])]
lir_range = Lir[(np.log10(Lir) >= a2[8]) & (np.log10(Lir) <= a2[21])]
#print(np.log10(f_ir_range))
#print(np.log10(lir_range))

def f_ir_schecter(lir_range):
	return np.interp(np.log10(lir_range),a2,b2)
	
#print(f_ir_schecter(lir_range,z))

phi_star = f_ir_schecter(lir_range)-np.log10(f_ir_range)
#print(phi_star)

def phi_star_func(lir_range):
	return np.interp((lir_range),lir_range,phi_star)

B_spline_coeff = make_interp_spline(lir_range, phi_star_func(lir_range))
X_Final = np.linspace(lir_range.min(), lir_range.max(), 1000)
Y_Final = B_spline_coeff(X_Final)


	
if z==7:
	f3 = plt.figure(figsize=(7,7))
	ax3 = f3.add_subplot(111)
	ax3.yaxis.set_ticks_position('both')
	ax3.xaxis.set_ticks_position('both')
	ax3.tick_params(which='both', direction='in')
	plt.plot(np.log10(X_Final),Y_Final,color="purple",linewidth=2.0)
	#plt.plot(np.log10(lir_range),phi_star_func_final,color="purple",linewidth=2.0)
	plt.xlabel("$\mathrm{L_{IR} [L_{\odot}]}$", fontsize = 18)
	plt.ylabel("$\log{\phi_{*}}$", fontsize = 18)
	#plt.title("Missing galaxies in z = 7")
	plt.tick_params(which='both', length=10, width=1, colors='k', labelsize='18', pad=10)
	#plt.grid()
	#plt.minorticks_on()
	plt.show()



phi_star_percentage = phi_star_func(lir_range)/np.abs(np.log10(f_ir_range)) * 1e2

def phi_star_per_func(lir_range):
	return np.interp((lir_range),lir_range,phi_star_percentage)

B_spline_coeff1 = make_interp_spline(lir_range, phi_star_per_func(lir_range))
X_Final1 = np.linspace(lir_range.min(), lir_range.max(), 1000)
Y_Final1 = B_spline_coeff1(X_Final)

if z==7:
	f3 = plt.figure(figsize=(7,7))
	ax3 = f3.add_subplot(111)
	ax3.yaxis.set_ticks_position('both')
	ax3.xaxis.set_ticks_position('both')
	ax3.tick_params(which='both', direction='in')
	ax.yaxis.set_ticks(np.arange(4,20,5))
	plt.plot(np.log10(X_Final1),Y_Final1,color="purple",linewidth=2.0)
	#plt.plot(np.log10(lir_range),phi_star_per_func(lir_range),color="green",linewidth=2.0)
	plt.xlabel("$\mathrm{L_{IR} [L_{\odot}]}$", fontsize = 18)
	plt.ylabel("${\phi_{*}}$ %", fontsize = 18)
	#plt.title("Missing galaxies in z = 7")
	#plt.grid()
	plt.tick_params(which='both', length=10, width=1, colors='k', labelsize='18', pad=10)
	#plt.minorticks_on()
	
	plt.show()




# z = 5.5 missing galaxies:
f_ir_range1 = fir[(np.log10(Lir) >= a15[26]) & (np.log10(Lir) <= a15[69])]
lir_range1 = Lir[(np.log10(Lir) >= a15[26]) & (np.log10(Lir) <= a15[69])]
# print(np.log10(f_ir_range1))
# print(np.log10(lir_range1))

def f_ir_schecter1(lir_range1):
	return np.interp(np.log10(lir_range1),a15,b15)
	
#print(f_ir_schecter(lir_range))

phi_star1 = f_ir_schecter1(lir_range1)-np.log10(f_ir_range1)
#print(phi_star1)

def phi_star_func1(lir_range1):
	return np.interp((lir_range1),lir_range1,phi_star1)
	
if z==5.5:	
	f3 = plt.figure(figsize=(7,7))
	ax3 = f3.add_subplot(111)
	ax3.yaxis.set_ticks_position('both')
	ax3.xaxis.set_ticks_position('both')
	ax3.tick_params(which='both', direction='in')
	plt.plot(np.log10(lir_range1),phi_star_func1(lir_range1),color="purple",linewidth=2.0)
	plt.xlabel("$\mathrm{L_{IR} [L_{\odot}}]$", fontsize = 18)
	plt.ylabel("$\log{\phi_{*}}$", fontsize = 18)
	#plt.title("Missing galaxies in z = 5.5")
	#plt.grid()
	plt.tick_params(which='both', length=10, width=1, colors='k', labelsize='18', pad=10)
	#plt.minorticks_on()
	plt.show()


phi_star_percentage1 = phi_star_func1(lir_range1)/np.abs(np.log10(f_ir_range1)) * 1e2

def phi_star_per_func1(lir_range1):
	return np.interp((lir_range1),lir_range1,phi_star_percentage1)

if z==5.5:
	f3 = plt.figure(figsize=(7,7))
	ax3 = f3.add_subplot(111)
	ax3.yaxis.set_ticks_position('both')
	ax3.xaxis.set_ticks_position('both')
	ax3.tick_params(which='both', direction='in')
	plt.plot(np.log10(lir_range1),phi_star_per_func1(lir_range1),color="purple",linewidth=2.0)
	plt.xlabel("$\mathrm{L_{IR} [L_{\odot}]}$", fontsize = 18)
	plt.ylabel("${\phi_{*}}$ %", fontsize = 18)
	#plt.title("Missing galaxies in z = 5.5")
	#plt.grid()
	plt.tick_params(which='both', length=10, width=1, colors='k', labelsize='18', pad=10)
	#plt.minorticks_on()
	plt.show()


# z = 4
f_ir_range2 = fir[(np.log10(Lir) >= a16[22]) & (np.log10(Lir) <= a16[47])]
lir_range2 = Lir[(np.log10(Lir) >= a16[22]) & (np.log10(Lir) <= a16[47])]
#print(np.log10(f_ir_range))
#print(np.log10(lir_range))

def f_ir_schecter2(lir_range2):
	return np.interp(np.log10(lir_range2),a16,b16)
	
#print(f_ir_schecter(lir_range,z))

phi_star2 = f_ir_schecter2(lir_range2)-np.log10(f_ir_range2)
#print(phi_star2)

def phi_star_func2(lir_range2):
	return np.interp((lir_range2),lir_range2,phi_star2)

if z==4:
	f3 = plt.figure(figsize=(7,7))
	ax3 = f3.add_subplot(111)
	ax3.yaxis.set_ticks_position('both')
	ax3.xaxis.set_ticks_position('both')
	ax3.tick_params(which='both', direction='in')
	plt.plot(np.log10(lir_range2),phi_star_func2(lir_range2),color="purple",linewidth=2.0)
	plt.xlabel("$\mathrm{L_{IR} [L_{\odot}]}$", fontsize = 18)
	plt.ylabel("$\log{\Phi_{*}}$", fontsize = 18)
	#plt.title("Missing galaxies in z = 4")
	#plt.grid()
	#plt.minorticks_on()
	plt.tick_params(which='both', length=10, width=1, colors='k', labelsize='18', pad=10)
	plt.show()

phi_star_percentage2 = phi_star_func2(lir_range2)/np.abs(np.log10(f_ir_range2)) * 1e2

def phi_star_per_func2(lir_range2):
	return np.interp((lir_range2),lir_range2,phi_star_percentage2)
if z==4:
	f3 = plt.figure(figsize=(7,7))
	ax3 = f3.add_subplot(111)
	ax3.yaxis.set_ticks_position('both')
	ax3.xaxis.set_ticks_position('both')
	ax3.tick_params(which='both', direction='in')
	plt.plot(np.log10(lir_range2),phi_star_per_func2(lir_range2),color="purple",linewidth=2.0)
	plt.xlabel("$\mathrm{L_{IR} [L_{\odot}]}$", fontsize = 18)
	plt.ylabel("${\Phi_{*}}$ %", fontsize = 18)
	#plt.title("Missing galaxies in z = 4")
	#plt.grid()
	#plt.minorticks_on()
	plt.tick_params(which='both', length=10, width=1, colors='k', labelsize='18', pad=10)
	plt.show()


#for i,j,k,l in zip(muv(m,z),np.log10(luv1(m,z)),np.log10(lir(m,z)),tau_eff(m,z)):
	#print(i,j,k,l)

#plt.plot(np.log10(m),np.log10(phi_ir(m,z)),color="green")
#plt.plot(np.log10(m),np.log10(phi_uv(m,z)),color="blue",linestyle="-.")
#plt.plot(np.log10(m),np.log10(phi_uv(m,z)+phi_uvnd(m,z)),color="orange")
#plt.xlabel("M $(M_{\odot})$")
#plt.ylabel("$\log(\phi)$")
#plt.legend(["UV-dust","IR","UV-int","UV-sum"])
#plt.xlim(11,14)
#plt.ylim(-20,-6)
#plt.show()
#print(np.log(phi_uv(m=1e13,z=z)/phi_uvnd(m=1e13,z=z)))


# plt.plot(np.log10(sfr(m,z)),np.log10(fobs(m,z)),color="green",linewidth = 2.0)
# #plt.xlim(1.6,2.35)
# #plt.ylim(-0.3,0)
# plt.xlabel("$\log \mathrm{SFR}$ [$M_{\odot}$/yr]", fontsize = 18)
# plt.grid()
# plt.ylabel("$\log$ $\mathrm{f_{obs}}$",fontsize = 18)
# #plt.minorticks_on()
# plt.tick_params(which='both', length=10, width=1, colors='k', labelsize='18', pad=10)
# plt.axis([1.6,2.4,-0.4,0.1])

# # Required obscuration to fit the z=7 LF
# x=np.arange(-1, 2.5, 0.01)
# tau_eff_needed = 0.7+0.0164*(10**x/10.)**(1.45)                      # tau NOT in log     
# fobs_needed = 1-np.exp(-tau_eff_needed)
# plt.plot(x, np.log10(fobs_needed), '-', color='green', linewidth=2, alpha=0.9)
# #plt.text(x[320]+0.06, np.log10(fobs_needed[320]), 'Best fit to LF',  fontsize=16, color='brown',fontstyle='normal')


# # Whitaker+2017 fit https://iopscience.iop.org/article/10.3847/1538-4357/aa94ce/pdf, eq. 2 (use z=2.5)
# a = 10.701
# b = -2.516
# fobs_fit = 1./(1. + a*np.exp(b*x))
# plt.plot(x, np.log10(fobs_fit), '-', color='blue', dashes=[30,5], linewidth=0.8)
# #plt.text(x[300], np.log10(fobs_fit[300]+0.05), 'Whitaker+17, z=2.5',  fontsize=16, color='grey',fontstyle='normal')

# # REBELS Data Bouwens+22 (in Model Quick Data Analysis_v2, sheet fobs) Flagship paper (Tab 2)
# SFR_Bouwens    = np.array([82.,54.,64.,80.,92.,76.,72.,66.,200.,55.,60.,51.,102.,116.,89.,52.])
# fobs_Bouwens   = np.array([0.720,0.741,0.766,0.800,0.674,0.539,0.569,0.788,0.925,0.636,0.583,0.725,0.725,0.845,0.584,0.673])
# #plt.text(2.0, -0.33, 'REBELS, z=7',  fontsize=22, color='pink',fontstyle='normal')

# # ... and the associated errors
# SFR_Bouwens_errp = np.array([20.,23.,28.,48.,39.,27.,23.,31.,101.,19.,20.,23.,18.,54.,31.,20.])
# SFR_Bouwens_errm = np.array([36.,16.,19.,25.,29.,21.,16.,23.,64.,14.,14.,17.,41.,35.,21.,14.])
# fobs_Bouwens_errp   = np.array([0.30,0.53,0.55,0.77,0.50,0.37,0.37,0.60,0.69,0.41,0.39,0.56,0.22,0.61,0.39,0.46])
# fobs_Bouwens_errm   = np.array([0.54,0.37,0.38,0.39,0.37,0.27,0.26,0.45,0.44,0.28,0.27,0.41,0.50,0.40,0.26,0.32])
# # ... transform them to be plotted in log10 in the Figure
# SFR_Bouwens_errp = np.log10((SFR_Bouwens+SFR_Bouwens_errp)/SFR_Bouwens)
# SFR_Bouwens_errm = -np.log10((SFR_Bouwens-SFR_Bouwens_errm)/SFR_Bouwens)
# fobs_Bouwens_errp = np.log10((fobs_Bouwens+fobs_Bouwens_errp)/fobs_Bouwens)
# fobs_Bouwens_errm = -np.log10((fobs_Bouwens-fobs_Bouwens_errm)/fobs_Bouwens)
# # ... produce asymmetric errors matrix
# err_SFR = np.array(list(zip(SFR_Bouwens_errm, SFR_Bouwens_errp))).T
# err_fobs= np.array(list(zip(fobs_Bouwens_errm, fobs_Bouwens_errp))).T

# plt.errorbar(np.log10(SFR_Bouwens) , np.log10(fobs_Bouwens), xerr=err_SFR, yerr=err_fobs, capsize=3, ls='none',ecolor='black', elinewidth=0.7)
# plt.scatter(np.log10(SFR_Bouwens) , np.log10(fobs_Bouwens), color='red', edgecolors='k')#, s=400, alpha=1.0)

# # Beta slope of the 10234 galaxy in Tab. 2 of Adams+22.
# # Uses F200, 277, 356 fluxes
# wl = np.array([1.989, 2.762, 3.568])
# lg_wl = np.log10(wl)
# m  = -np.array([27.23, 27.73, 28.08])
# err = 0.5
# # https://stackoverflow.com/questions/26052365/scipy-polyfit-x-y-weights-error-bars
# f = lambda x, a, b: a*x + b  # function to fit
# # fit with initial guess for parameters [1, 1]
# pars, corr = optimize.curve_fit(f, lg_wl, m, [-3, -26], err)
# a2, b2 = pars
# #beta_fit = np.polyfit(lg_wl, m,1)            # Fit coefficients  (0 = highest order)
# #print 'Adams22 beta Fit coefficients =',a2, b2






# plt.show()




# # %%

# %%
